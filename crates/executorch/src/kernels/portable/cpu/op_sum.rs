//! Literal port of kernels/portable/cpu/op_sum.cpp.

#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_reduction_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    MapReduceOverDimListPlan, parallel_for_each_reduce_over_dim_list_output_index,
    resize_reduction_out,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_contiguous, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `optional<ArrayRef<int64_t>> dim_list` maps to
// `Option<ArrayRef<i64>>`; `optional<ScalarType> dtype` to `Option<ScalarType>`.
// The ported reduce_util helpers take `&Option<ArrayRef<i64>>`.
//
// PORT-NOTE: the C++ `ACC = std::conditional_t<Half||BFloat16, float, CTYPE>`
// accumulation-type selection is reproduced with an `AccType` trait carrying the
// associated `Acc` type plus the CTYPE<->ACC conversions and ACC arithmetic.
// Two ACC families are needed because the C++ uses two different CTYPE sets:
// the fast/complex paths accumulate over the same dtype as CTYPE (REALHBBF16),
// while the general path's ACC is chosen from CTYPE_OUT and the map is over
// CTYPE_IN. `ToAcc`/`FromAcc` reproduce the general-path `static_cast<ACC>` /
// `static_cast<CTYPE_OUT>`.

// Fast path: ACC == float for Half/BFloat16, else CTYPE. `acc = 0; acc += row[j];
// out = static_cast<CTYPE>(acc)`.
trait FastAcc: Copy {
    type Acc: Copy;
    fn acc_zero() -> Self::Acc;
    fn acc_add_self(acc: Self::Acc, v: Self) -> Self::Acc;
    fn from_acc(acc: Self::Acc) -> Self;
}
macro_rules! impl_fast_acc_native {
    ($t:ty) => {
        impl FastAcc for $t {
            type Acc = $t;
            fn acc_zero() -> $t {
                0 as $t
            }
            fn acc_add_self(acc: $t, v: $t) -> $t {
                acc + v
            }
            fn from_acc(acc: $t) -> Self {
                acc
            }
        }
    };
}
impl_fast_acc_native!(u8);
impl_fast_acc_native!(i8);
impl_fast_acc_native!(i16);
impl_fast_acc_native!(i32);
impl_fast_acc_native!(i64);
impl_fast_acc_native!(f32);
impl_fast_acc_native!(f64);
// Bool: ACC == bool, `acc += row[j]` promotes to int, back to bool.
impl FastAcc for bool {
    type Acc = bool;
    fn acc_zero() -> bool {
        false
    }
    fn acc_add_self(acc: bool, v: bool) -> bool {
        ((acc as i32) + (v as i32)) != 0
    }
    fn from_acc(acc: bool) -> Self {
        acc
    }
}
macro_rules! impl_fast_acc_lowp {
    ($t:ty) => {
        impl FastAcc for $t {
            type Acc = f32;
            fn acc_zero() -> f32 {
                0.0
            }
            fn acc_add_self(acc: f32, v: $t) -> f32 {
                acc + v.to_f32()
            }
            fn from_acc(acc: f32) -> Self {
                <$t>::from_f32(acc)
            }
        }
    };
}
impl_fast_acc_lowp!(Half);
impl_fast_acc_lowp!(BFloat16);

// General path: ACC chosen from CTYPE_OUT (float for Half/BFloat16, else
// CTYPE_OUT). Carries `acc_zero` and `acc_add` plus `static_cast<CTYPE_OUT>`.
trait AccType: Copy {
    type Acc: Copy;
    fn acc_zero() -> Self::Acc;
    fn acc_add(a: Self::Acc, b: Self::Acc) -> Self::Acc;
    fn from_acc(acc: Self::Acc) -> Self;
}
macro_rules! impl_acc_type_native_int {
    ($t:ty) => {
        impl AccType for $t {
            type Acc = $t;
            fn acc_zero() -> $t {
                0 as $t
            }
            fn acc_add(a: $t, b: $t) -> $t {
                // PORT-NOTE: C++ accumulates `acc += row[j]` in the (possibly
                // narrow) ACC integer type, relying on silent wraparound
                // (well-defined for unsigned, 2's-complement wrap in practice for
                // signed). Rust `+` panics on overflow in debug; `wrapping_add`
                // reproduces the C++ behavior. See TypeConversionTest (byte out,
                // 64*4 = 256 -> 0).
                a.wrapping_add(b)
            }
            fn from_acc(acc: $t) -> Self {
                acc
            }
        }
    };
}
macro_rules! impl_acc_type_native_float {
    ($t:ty) => {
        impl AccType for $t {
            type Acc = $t;
            fn acc_zero() -> $t {
                0 as $t
            }
            fn acc_add(a: $t, b: $t) -> $t {
                a + b
            }
            fn from_acc(acc: $t) -> Self {
                acc
            }
        }
    };
}
impl_acc_type_native_int!(u8);
impl_acc_type_native_int!(i8);
impl_acc_type_native_int!(i16);
impl_acc_type_native_int!(i32);
impl_acc_type_native_int!(i64);
impl_acc_type_native_float!(f32);
impl_acc_type_native_float!(f64);
impl AccType for bool {
    type Acc = bool;
    fn acc_zero() -> bool {
        false
    }
    fn acc_add(a: bool, b: bool) -> bool {
        ((a as i32) + (b as i32)) != 0
    }
    fn from_acc(acc: bool) -> Self {
        acc
    }
}
macro_rules! impl_acc_type_lowp {
    ($t:ty) => {
        impl AccType for $t {
            type Acc = f32;
            fn acc_zero() -> f32 {
                0.0
            }
            fn acc_add(a: f32, b: f32) -> f32 {
                a + b
            }
            fn from_acc(acc: f32) -> Self {
                <$t>::from_f32(acc)
            }
        }
    };
}
impl_acc_type_lowp!(Half);
impl_acc_type_lowp!(BFloat16);

// The general path's map converts CTYPE_IN -> ACC with `static_cast<ACC>`.
trait ToAcc<Acc> {
    fn to_acc_conv(self) -> Acc;
}
macro_rules! impl_to_acc {
    ($src:ty, $acc:ty) => {
        impl ToAcc<$acc> for $src {
            fn to_acc_conv(self) -> $acc {
                self as $acc
            }
        }
    };
}
// Numeric sources into each of the possible ACC types (u8..f64 and f32).
macro_rules! impl_to_acc_row {
    ($acc:ty) => {
        impl_to_acc!(u8, $acc);
        impl_to_acc!(i8, $acc);
        impl_to_acc!(i16, $acc);
        impl_to_acc!(i32, $acc);
        impl_to_acc!(i64, $acc);
        impl_to_acc!(f32, $acc);
        impl_to_acc!(f64, $acc);
    };
}
impl_to_acc_row!(u8);
impl_to_acc_row!(i8);
impl_to_acc_row!(i16);
impl_to_acc_row!(i32);
impl_to_acc_row!(i64);
impl_to_acc_row!(f32);
impl_to_acc_row!(f64);
// Bool source: `static_cast<ACC>(bool)` promotes false->0, true->1.
macro_rules! impl_to_acc_bool {
    ($acc:ty) => {
        impl ToAcc<$acc> for bool {
            fn to_acc_conv(self) -> $acc {
                (self as i32) as $acc
            }
        }
    };
}
impl_to_acc_bool!(u8);
impl_to_acc_bool!(i8);
impl_to_acc_bool!(i16);
impl_to_acc_bool!(i32);
impl_to_acc_bool!(i64);
impl_to_acc_bool!(f32);
impl_to_acc_bool!(f64);
impl ToAcc<bool> for bool {
    fn to_acc_conv(self) -> bool {
        self
    }
}
// Sources into a bool ACC (when CTYPE_OUT is Bool): `static_cast<bool>` is
// nonzero-test.
macro_rules! impl_to_acc_bool_dest {
    ($src:ty) => {
        impl ToAcc<bool> for $src {
            fn to_acc_conv(self) -> bool {
                self != (0 as $src)
            }
        }
    };
}
impl_to_acc_bool_dest!(u8);
impl_to_acc_bool_dest!(i8);
impl_to_acc_bool_dest!(i16);
impl_to_acc_bool_dest!(i32);
impl_to_acc_bool_dest!(i64);
impl_to_acc_bool_dest!(f32);
impl_to_acc_bool_dest!(f64);
impl ToAcc<bool> for Half {
    fn to_acc_conv(self) -> bool {
        self.to_f32() != 0.0
    }
}
impl ToAcc<bool> for BFloat16 {
    fn to_acc_conv(self) -> bool {
        self.to_f32() != 0.0
    }
}
// Half/BFloat16 sources: `static_cast<ACC>` via f32 widening.
macro_rules! impl_to_acc_lowp {
    ($src:ty, $acc:ty) => {
        impl ToAcc<$acc> for $src {
            fn to_acc_conv(self) -> $acc {
                self.to_f32() as $acc
            }
        }
    };
}
impl_to_acc_lowp!(Half, u8);
impl_to_acc_lowp!(Half, i8);
impl_to_acc_lowp!(Half, i16);
impl_to_acc_lowp!(Half, i32);
impl_to_acc_lowp!(Half, i64);
impl_to_acc_lowp!(Half, f32);
impl_to_acc_lowp!(Half, f64);
impl_to_acc_lowp!(BFloat16, u8);
impl_to_acc_lowp!(BFloat16, i8);
impl_to_acc_lowp!(BFloat16, i16);
impl_to_acc_lowp!(BFloat16, i32);
impl_to_acc_lowp!(BFloat16, i64);
impl_to_acc_lowp!(BFloat16, f32);
impl_to_acc_lowp!(BFloat16, f64);

// Complex accumulation: `CTYPE sum(0, 0); sum = acc + outv` on c10::complex.
trait ComplexSum: Copy {
    fn c_zero() -> Self;
    fn c_add(self, other: Self) -> Self;
}
macro_rules! impl_complex_sum {
    ($comp:ty, $to:expr, $from:expr) => {
        impl ComplexSum for Complex<$comp> {
            fn c_zero() -> Self {
                Complex {
                    real: $from(0.0),
                    imag: $from(0.0),
                }
            }
            fn c_add(self, other: Self) -> Self {
                Complex {
                    real: $from($to(self.real) + $to(other.real)),
                    imag: $from($to(self.imag) + $to(other.imag)),
                }
            }
        }
    };
}
impl_complex_sum!(Half, |x: Half| x.to_f64(), |x: f64| Half::from_f64(x));
impl_complex_sum!(f32, |x: f32| x as f64, |x: f64| x as f32);
impl_complex_sum!(f64, |x: f64| x, |x: f64| x);

// [spec:et:def:op-sum.torch.executor.native.sum-dim-out-fn]
// [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn]
#[executorch_macros::et_kernel("aten::sum.IntList_out")]
pub fn sum_dim_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    keepdim: bool,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // PORT-NOTE: C++ calls `check_reduction_args` unconditionally, but the ported
    // reduce_util gates the portable arg-checkers behind `#[cfg(not(aten))]`
    // (absent in the ATen build); gated here to match the ported util. Unresolved
    // cross-module reference for the fixer: the C++ call was not `#ifndef`-guarded.
    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_reduction_args(in_, &dim_list, keepdim, dtype, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    // Fast path: contiguous tensor, single innermost dim reduction, same dtype.
    // Bypasses generic MapReduceOverDimListPlan to use a tight vectorizable loop.
    if in_.numel() > 0
        && dim_list.is_some()
        && dim_list.as_ref().unwrap().size() == 1
        && !is_complex_type(in_.scalar_type())
        && in_.scalar_type() == out.scalar_type()
    {
        let d: i64 = if *dim_list.as_ref().unwrap().at(0) < 0 {
            *dim_list.as_ref().unwrap().at(0) + in_.dim() as i64
        } else {
            *dim_list.as_ref().unwrap().at(0)
        };
        if d >= 0 && d < in_.dim() as i64 && d == in_.dim() as i64 - 1 && tensor_is_contiguous(in_)
        {
            let reduce_size: i64 = in_.size(d as _) as i64;
            let outer_size: i64 = in_.numel() as i64 / reduce_size;

            let op_name = "sum.IntList_out";
            // For half-precision inputs, accumulate in float to avoid saturation.
            // Matches ATen's acc_type behavior. See also op_grid_sampler_2d.cpp.
            crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
                let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
                let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
                for i in 0..outer_size {
                    let row: *const CTYPE = unsafe { in_data.offset((i * reduce_size) as isize) };
                    let mut acc: <CTYPE as FastAcc>::Acc = <CTYPE as FastAcc>::acc_zero();
                    for j in 0..reduce_size {
                        acc = <CTYPE as FastAcc>::acc_add_self(acc, unsafe {
                            *row.offset(j as isize)
                        });
                    }
                    unsafe {
                        *out_data.offset(i as isize) = <CTYPE as FastAcc>::from_acc(acc);
                    }
                }
            });
            return out;
        }
    }

    let mut plan: Option<MapReduceOverDimListPlan> = None;
    if in_.numel() > 0 {
        plan = Some(MapReduceOverDimListPlan::new(in_, &dim_list));
    }
    let op_name = "sum.IntList_out";

    if is_complex_type(in_.scalar_type()) {
        crate::et_kernel_check!(
            ctx,
            in_.scalar_type() == out.scalar_type(),
            InvalidArgument,
            out
        );

        crate::et_switch_complexh_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
            let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            let success = parallel_for_each_reduce_over_dim_list_output_index(
                in_,
                dim_list,
                out,
                &|begin: i64, end: i64| {
                    for out_ix in begin..end {
                        let out_ix = out_ix as usize;
                        let mut sum: CTYPE = <CTYPE as ComplexSum>::c_zero();
                        if let Some(ref plan) = plan {
                            sum = plan.execute::<CTYPE, CTYPE, _, _>(
                                |v: CTYPE| -> CTYPE { v },
                                |outv: CTYPE, acc: CTYPE| -> CTYPE { acc.c_add(outv) },
                                out_ix,
                            );
                        }
                        unsafe {
                            *out_data.add(out_ix) = sum;
                        }
                    }
                },
            );
            crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
        });
    } else {
        crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE_IN, {
            crate::et_switch_realhbbf16_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
                let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
                let success = parallel_for_each_reduce_over_dim_list_output_index(
                    in_,
                    dim_list,
                    out,
                    &|begin: i64, end: i64| {
                        for out_ix in begin..end {
                            let out_ix = out_ix as usize;
                            let mut sum: <CTYPE_OUT as AccType>::Acc =
                                <CTYPE_OUT as AccType>::acc_zero();
                            if let Some(ref plan) = plan {
                                sum = plan.execute::<CTYPE_IN, <CTYPE_OUT as AccType>::Acc, _, _>(
                                    |v: CTYPE_IN| -> <CTYPE_OUT as AccType>::Acc {
                                        <CTYPE_IN as ToAcc<<CTYPE_OUT as AccType>::Acc>>::to_acc_conv(
                                            v,
                                        )
                                    },
                                    |outv: <CTYPE_OUT as AccType>::Acc,
                                     acc: <CTYPE_OUT as AccType>::Acc|
                                     -> <CTYPE_OUT as AccType>::Acc {
                                        <CTYPE_OUT as AccType>::acc_add(acc, outv)
                                    },
                                    out_ix,
                                );
                            }
                            unsafe {
                                *out_data.add(out_ix) = <CTYPE_OUT as AccType>::from_acc(sum);
                            }
                        }
                    },
                );
                crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
            });
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{
        BFloat16, Complex, ComplexDouble, ComplexFloat, Half,
    };
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_sum_intlist_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim: Option<ArrayRef<i64>>,
        keepdim: bool,
        dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        sum_dim_out(ctx, self_, dim, keepdim, dtype, out)
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REALHBF16-and-Bool factory element types used by the templated helpers.
    trait FromI32Data: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_data_num {
        ($($t:ty),*) => {$(impl FromI32Data for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_data_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32Data for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }
    impl FromI32Data for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32Data for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn d<T: FromI32Data>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    // Wraps a `&[i64]` in the non-owning `ArrayRef<i64>` dim list.
    fn dim_ref(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    fn test_sum_dim_out_invalid_dimensions<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let self_ = tf_in.make_default(
            vec![2, 3, 4],
            d::<IN>(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, //
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            ]),
        );
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let dtype = Some(OUT::VALUE);

        // out-of-bound dim in dim list
        let dims_1 = [3i64];
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // the same dim appears multiple times in list of dims
        let dims_2 = [2i64, 2];
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_2)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_sum_dim_out_invalid_shape<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let self_ = tf_in.make_default(
            vec![2, 3, 4],
            d::<IN>(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, //
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            ]),
        );

        // dimension size mismatch when keepdim is true
        let out = tf_out.zeros_default(vec![2, 4]);
        let dtype = Some(OUT::VALUE);
        let dims_1 = [1i64];
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // dimension size mismatch when keepdim is false
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), false, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // Builds a `Complex<T>` from real/imag `int` values (mirrors `CTYPE(re, im)`).
    fn c<T: FromI32Data>(re: i32, im: i32) -> Complex<T> {
        Complex {
            real: T::from_i32(re),
            imag: T::from_i32(im),
        }
    }

    fn test_complex_dtype<T>()
    where
        T: FromI32Data,
        Complex<T>: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<Complex<T>>::new();

        let self_ = tf.make_default(
            vec![2, 3, 2],
            vec![
                c::<T>(1, 1),
                c::<T>(2, 2),
                c::<T>(3, 3),
                c::<T>(4, 4),
                c::<T>(5, 5),
                c::<T>(6, 6),
                c::<T>(7, 7),
                c::<T>(8, 8),
                c::<T>(9, 9),
                c::<T>(10, 10),
                c::<T>(11, 11),
                c::<T>(12, 12),
            ],
        );

        let out1 = tf.make_default(vec![2, 3, 1], vec![c::<T>(0, 0); 6]);
        let dims_1 = [2i64];
        let dtype = Some(<Complex<T> as CppTypeToScalarType>::VALUE);
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out1);
        let expected1 = tf.make_default(
            vec![2, 3, 1],
            vec![
                c::<T>(3, 3),
                c::<T>(7, 7),
                c::<T>(11, 11),
                c::<T>(15, 15),
                c::<T>(19, 19),
                c::<T>(23, 23),
            ],
        );
        assert_tensor_close!(out1, expected1);

        let out2 = tf.make_default(vec![2, 1, 2], vec![c::<T>(0, 0); 4]);
        let dims_2 = [1i64];
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_2)), true, dtype, &out2);
        let expected2 = tf.make_default(
            vec![2, 1, 2],
            vec![c::<T>(9, 9), c::<T>(12, 12), c::<T>(27, 27), c::<T>(30, 30)],
        );
        assert_tensor_close!(out2, expected2);

        let out3 = tf.make_default(vec![1, 1, 1], vec![c::<T>(0, 0)]);
        op_sum_intlist_out(&mut ctx, &self_, None, true, dtype, &out3);
        let expected3 = tf.make_default(vec![1, 1, 1], vec![c::<T>(78, 78)]);
        assert_tensor_close!(out3, expected3);
    }

    fn test_sum_dim_out_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        OUT: CppTypeToScalarType + FactoryValue + FromI32Data,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let mut self_ = tf_in.make_default(
            vec![2, 3, 4],
            d::<IN>(&[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, //
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            ]),
        );
        let dtype = Some(OUT::VALUE);

        // keepdim=true should work
        let dims_1 = [2i64];
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 3, 1], d::<OUT>(&[6, 22, 38, 54, 70, 86]))
        );

        // keepdim=false should work
        let out = tf_out.zeros_default(vec![2, 3]);
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), false, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 3], d::<OUT>(&[6, 22, 38, 54, 70, 86]))
        );

        // dim list with multiple dimensions should work
        let dims_01 = [0i64, 1];
        let out = tf_out.zeros_default(vec![1, 1, 4]);
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_01)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![1, 1, 4], d::<OUT>(&[60, 66, 72, 78]))
        );

        let out = tf_out.zeros_default(vec![4]);
        op_sum_intlist_out(
            &mut ctx,
            &self_,
            Some(dim_ref(&dims_01)),
            false,
            dtype,
            &out,
        );
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![4], d::<OUT>(&[60, 66, 72, 78]))
        );

        let dims_02 = [0i64, 2];
        let out = tf_out.zeros_default(vec![1, 3, 1]);
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_02)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![1, 3, 1], d::<OUT>(&[60, 92, 124]))
        );

        let out = tf_out.zeros_default(vec![3]);
        op_sum_intlist_out(
            &mut ctx,
            &self_,
            Some(dim_ref(&dims_02)),
            false,
            dtype,
            &out,
        );
        assert_tensor_close!(out, tf_out.make_default(vec![3], d::<OUT>(&[60, 92, 124])));

        // dim list with negative dimensions should work
        let dims_3 = [-2i64];
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_3)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 1, 4], d::<OUT>(&[12, 15, 18, 21, 48, 51, 54, 57]))
        );

        // empty/null dim list should work
        self_ = tf_in.make_default(
            vec![2, 2, 4],
            d::<IN>(&[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]),
        );
        let out = tf_out.zeros_default(vec![1, 1, 1]);
        op_sum_intlist_out(&mut ctx, &self_, None, true, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![1, 1, 1], d::<OUT>(&[56])));

        let empty_dims: [i64; 0] = [];
        op_sum_intlist_out(
            &mut ctx,
            &self_,
            Some(dim_ref(&empty_dims)),
            true,
            dtype,
            &out,
        );
        assert_tensor_close!(out, tf_out.make_default(vec![1, 1, 1], d::<OUT>(&[56])));

        let out = tf_out.zeros_default(vec![]);
        op_sum_intlist_out(&mut ctx, &self_, None, false, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![], d::<OUT>(&[56])));

        op_sum_intlist_out(
            &mut ctx,
            &self_,
            Some(dim_ref(&empty_dims)),
            false,
            dtype,
            &out,
        );
        assert_tensor_close!(out, tf_out.make_default(vec![], d::<OUT>(&[56])));
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_bfloat16_generic_path_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        const N: i32 = 512;
        let x = tf.ones_default(vec![N, 1]);
        let out = tf.zeros_default(vec![1]);
        let dim = [0i64];
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&dim)), false, None, &out);
        let expected = tf.full(
            vec![1],
            BFloat16::from_f32(N as f32),
            crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_bfloat16_large_dim_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        const N: i32 = 512;
        let x = tf.ones_default(vec![1, N]);
        let out = tf.zeros_default(vec![1]);
        let dim = [1i64];
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&dim)), false, None, &out);
        let expected = tf.full(
            vec![1],
            BFloat16::from_f32(N as f32),
            crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC,
        );
        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_invalid_dimension_list_dies() {
        // ET_FORALL_REAL_TYPES_AND(Bool) x ET_FORALL_REAL_TYPES
        fn enumerate_out<IN>()
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        {
            test_sum_dim_out_invalid_dimensions::<IN, u8>();
            test_sum_dim_out_invalid_dimensions::<IN, i8>();
            test_sum_dim_out_invalid_dimensions::<IN, i16>();
            test_sum_dim_out_invalid_dimensions::<IN, i32>();
            test_sum_dim_out_invalid_dimensions::<IN, i64>();
            test_sum_dim_out_invalid_dimensions::<IN, f32>();
            test_sum_dim_out_invalid_dimensions::<IN, f64>();
        }
        enumerate_out::<u8>();
        enumerate_out::<i8>();
        enumerate_out::<i16>();
        enumerate_out::<i32>();
        enumerate_out::<i64>();
        enumerate_out::<f32>();
        enumerate_out::<f64>();
        enumerate_out::<bool>();
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_invalid_shape_dies() {
        // ET_FORALL_REAL_TYPES x ET_FORALL_REAL_TYPES
        fn enumerate_out<IN>()
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        {
            test_sum_dim_out_invalid_shape::<IN, u8>();
            test_sum_dim_out_invalid_shape::<IN, i8>();
            test_sum_dim_out_invalid_shape::<IN, i16>();
            test_sum_dim_out_invalid_shape::<IN, i32>();
            test_sum_dim_out_invalid_shape::<IN, i64>();
            test_sum_dim_out_invalid_shape::<IN, f32>();
            test_sum_dim_out_invalid_shape::<IN, f64>();
        }
        enumerate_out::<u8>();
        enumerate_out::<i8>();
        enumerate_out::<i16>();
        enumerate_out::<i32>();
        enumerate_out::<i64>();
        enumerate_out::<f32>();
        enumerate_out::<f64>();
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_mismatched_d_types_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = tf_int.make_default(
            vec![2, 3, 4],
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, //
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            ],
        );

        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims_1 = [2i64];
        let dtype = Some(ScalarType::Double);

        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_all_real_input_real_output_passes() {
        // ET_FORALL_REALHBF16_TYPES x ET_FORALL_REALHBF16_TYPES
        fn enumerate_out<IN>()
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32Data,
        {
            test_sum_dim_out_dtype::<IN, u8>();
            test_sum_dim_out_dtype::<IN, i8>();
            test_sum_dim_out_dtype::<IN, i16>();
            test_sum_dim_out_dtype::<IN, i32>();
            test_sum_dim_out_dtype::<IN, i64>();
            test_sum_dim_out_dtype::<IN, f32>();
            test_sum_dim_out_dtype::<IN, f64>();
            test_sum_dim_out_dtype::<IN, Half>();
            test_sum_dim_out_dtype::<IN, BFloat16>();
        }
        enumerate_out::<u8>();
        enumerate_out::<i8>();
        enumerate_out::<i16>();
        enumerate_out::<i32>();
        enumerate_out::<i64>();
        enumerate_out::<f32>();
        enumerate_out::<f64>();
        enumerate_out::<Half>();
        enumerate_out::<BFloat16>();
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_type_conversion_test() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = tf_int.make_default(
            vec![2, 3, 4],
            vec![
                0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, //
                8, 8, 8, 8, 16, 16, 16, 16, 64, 64, 64, 64,
            ],
        );

        let dims_1 = [2i64];
        let dtype: Option<ScalarType> = None;

        // int -> bool conversion should work
        let out = tf_bool.zeros_default(vec![2, 3, 1]);
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_bool.make_default(vec![2, 3, 1], vec![false, true, true, true, true, true])
        );

        // int -> byte conversion should work
        let out = tf_byte.zeros_default(vec![2, 3, 1]);
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims_1)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_byte.make_default(vec![2, 3, 1], vec![0, 8, 16, 32, 64, 0])
        );
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_all_complex_dtypes_supported() {
        // ET_FORALL_COMPLEXH_TYPES in the non-ATen build: ComplexFloat, ComplexDouble.
        test_complex_dtype::<f32>();
        test_complex_dtype::<f64>();
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_infinity_and_nan_test() {
        let tf_float = TensorFactory::<f32>::new();
        let inf = f32::INFINITY;
        let nan = f32::NAN;

        let self_ = tf_float.make_default(
            vec![2, 3, 4],
            vec![
                0.0, 1.0, 2.0, inf, //
                inf, -inf, 1.0, 0.0, //
                nan, inf, -inf, 2.0, //
                nan, nan, 1.0, 0.0, //
                0.0, inf, nan, 4.0, //
                1.0, nan, 3.14, 2.0,
            ],
        );

        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims = [-1i64];
        let dtype: Option<ScalarType> = None;
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &self_, Some(dim_ref(&dims)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_float.make_default(vec![2, 3, 1], vec![inf, nan, nan, nan, nan, nan])
        );
    }

    // [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn/test]
    #[test]
    fn op_sum_out_test_empty_input() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2, 0, 3], vec![]);
        let dtype = Some(ScalarType::Float);
        let empty_dims: [i64; 0] = [];

        let out = tf.ones_default(vec![1, 1, 1]);
        let mut ctx = context();
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&empty_dims)), true, dtype, &out);
        assert_tensor_close!(out, tf.zeros_default(vec![1, 1, 1]));

        let out = tf.ones_default(vec![]);
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&empty_dims)), false, dtype, &out);
        assert_tensor_close!(out, tf.zeros_default(vec![]));

        let dims1 = [1i64];
        let out = tf.ones_default(vec![2, 3]);
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&dims1)), false, dtype, &out);
        assert_tensor_close!(out, tf.zeros_default(vec![2, 3]));

        let dims2 = [2i64];
        let out = tf.make_default(vec![2, 0, 1], vec![]);
        op_sum_intlist_out(&mut ctx, &x, Some(dim_ref(&dims2)), true, dtype, &out);
        assert_tensor_close!(out, tf.make_default(vec![2, 0, 1], vec![]));
    }
}
