//! Literal port of kernels/portable/cpu/op_mean.cpp.

#[cfg(not(feature = "aten"))]
use crate::kernels::portable::cpu::util::reduce_util::check_mean_dim_args;
use crate::kernels::portable::cpu::util::reduce_util::{
    MapReduceOverDimListPlan, get_reduced_dim_product,
    parallel_for_each_reduce_over_dim_list_output_index, resize_reduction_out,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_contiguous, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `optional<ArrayRef<int64_t>> dim_list`
// maps to `Option<ArrayRef<i64>>`; `optional<ScalarType> dtype` to
// `Option<ScalarType>`. The ported reduce_util helpers take `&Option<ArrayRef<i64>>`.
//
// PORT-NOTE: the C++ `ACC = std::conditional_t<Half||BFloat16, float, CTYPE>`
// accumulation-type selection is reproduced with an `AccType` trait carrying the
// associated `Acc` type plus the CTYPE<->ACC conversions and ACC arithmetic used
// by both the fast and general paths; the closures then operate through those
// conversions, matching the C++ implicit conversions.

trait AccType: Copy {
    type Acc: Copy;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(acc: Self::Acc) -> Self;
    fn acc_zero() -> Self::Acc;
    fn acc_add(a: Self::Acc, b: Self::Acc) -> Self::Acc;
    // fast path: acc / denom, denom of type ACC (== static_cast<ACC>(reduce_size))
    fn acc_div(a: Self::Acc, denom: Self::Acc) -> Self::Acc;
    fn acc_from_i64(v: i64) -> Self::Acc;
    // general path: sum / static_cast<float>(num)
    fn acc_div_f32(a: Self::Acc, num: f32) -> Self::Acc;
}

macro_rules! impl_acc_type_float_native {
    ($t:ty) => {
        impl AccType for $t {
            type Acc = $t;
            fn to_acc(self) -> $t {
                self
            }
            fn from_acc(acc: $t) -> Self {
                acc
            }
            fn acc_zero() -> $t {
                0 as $t
            }
            fn acc_add(a: $t, b: $t) -> $t {
                a + b
            }
            fn acc_div(a: $t, denom: $t) -> $t {
                a / denom
            }
            fn acc_from_i64(v: i64) -> $t {
                v as $t
            }
            fn acc_div_f32(a: $t, num: f32) -> $t {
                (a as $t) / (num as $t)
            }
        }
    };
}
impl_acc_type_float_native!(f32);
impl_acc_type_float_native!(f64);

macro_rules! impl_acc_type_lowp {
    ($t:ty) => {
        impl AccType for $t {
            type Acc = f32;
            fn to_acc(self) -> f32 {
                self.to_f32()
            }
            fn from_acc(acc: f32) -> Self {
                <$t>::from_f32(acc)
            }
            fn acc_zero() -> f32 {
                0.0
            }
            fn acc_add(a: f32, b: f32) -> f32 {
                a + b
            }
            fn acc_div(a: f32, denom: f32) -> f32 {
                a / denom
            }
            fn acc_from_i64(v: i64) -> f32 {
                v as f32
            }
            fn acc_div_f32(a: f32, num: f32) -> f32 {
                a / num
            }
        }
    };
}
impl_acc_type_lowp!(Half);
impl_acc_type_lowp!(BFloat16);

// PORT-NOTE: the general path's map converts CTYPE_IN -> ACC with `static_cast`;
// mirrored by this conversion trait parameterized on the ACC type (f32 or f64) of
// CTYPE_OUT.
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
impl_to_acc!(u8, f32);
impl_to_acc!(i8, f32);
impl_to_acc!(i16, f32);
impl_to_acc!(i32, f32);
impl_to_acc!(i64, f32);
impl_to_acc!(f32, f32);
impl_to_acc!(f64, f32);
impl_to_acc!(u8, f64);
impl_to_acc!(i8, f64);
impl_to_acc!(i16, f64);
impl_to_acc!(i32, f64);
impl_to_acc!(i64, f64);
impl_to_acc!(f32, f64);
impl_to_acc!(f64, f64);
impl ToAcc<f32> for bool {
    fn to_acc_conv(self) -> f32 {
        (self as i32) as f32
    }
}
impl ToAcc<f64> for bool {
    fn to_acc_conv(self) -> f64 {
        (self as i32) as f64
    }
}
impl ToAcc<f32> for Half {
    fn to_acc_conv(self) -> f32 {
        self.to_f32()
    }
}
impl ToAcc<f64> for Half {
    fn to_acc_conv(self) -> f64 {
        self.to_f32() as f64
    }
}
impl ToAcc<f32> for BFloat16 {
    fn to_acc_conv(self) -> f32 {
        self.to_f32()
    }
}
impl ToAcc<f64> for BFloat16 {
    fn to_acc_conv(self) -> f64 {
        self.to_f32() as f64
    }
}

// Cast an ACC value to CTYPE_OUT (mirrors the C++ `static_cast<CTYPE_OUT>`).
trait FromAcc<Acc> {
    fn from_acc_conv(acc: Acc) -> Self;
}
impl FromAcc<f32> for f32 {
    fn from_acc_conv(acc: f32) -> Self {
        acc
    }
}
impl FromAcc<f64> for f64 {
    fn from_acc_conv(acc: f64) -> Self {
        acc
    }
}
impl FromAcc<f32> for Half {
    fn from_acc_conv(acc: f32) -> Self {
        Half::from_f32(acc)
    }
}
impl FromAcc<f32> for BFloat16 {
    fn from_acc_conv(acc: f32) -> Self {
        BFloat16::from_f32(acc)
    }
}

// [spec:et:def:op-mean.torch.executor.native.mean-dim-out-fn]
// [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn]
pub fn mean_dim_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    keepdim: bool,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    #[cfg(not(feature = "aten"))]
    crate::et_kernel_check!(
        ctx,
        check_mean_dim_args(in_, dim_list, keepdim, dtype, out),
        InvalidArgument,
        out
    );
    #[cfg(feature = "aten")]
    let _ = dtype;

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        resize_reduction_out(in_, &dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Fast path: contiguous tensor, single innermost dim reduction, same dtype.
    // Bypasses generic MapReduceOverDimListPlan to use a tight vectorizable loop.
    if in_.numel() > 0
        && dim_list.is_some()
        && dim_list.as_ref().unwrap().size() == 1
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

            let op_name = "mean.out";
            // For half-precision inputs, accumulate in float to avoid saturation.
            // Matches ATen's acc_type behavior.
            crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
                let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
                let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
                let denom: <CTYPE as AccType>::Acc = <CTYPE as AccType>::acc_from_i64(reduce_size);
                for i in 0..outer_size {
                    let row: *const CTYPE = unsafe { in_data.offset((i * reduce_size) as isize) };
                    let mut acc: <CTYPE as AccType>::Acc = <CTYPE as AccType>::acc_zero();
                    for j in 0..reduce_size {
                        acc = <CTYPE as AccType>::acc_add(
                            acc,
                            unsafe { *row.offset(j as isize) }.to_acc(),
                        );
                    }
                    unsafe {
                        *out_data.offset(i as isize) =
                            <CTYPE as AccType>::from_acc(<CTYPE as AccType>::acc_div(acc, denom));
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
    let op_name = "mean.out";
    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE_IN, {
        crate::et_switch_floathbf16_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
            let out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
            let num: usize = get_reduced_dim_product(in_, &dim_list);
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
                                    <CTYPE_IN as ToAcc<<CTYPE_OUT as AccType>::Acc>>::to_acc_conv(v)
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
                            *out_data.add(out_ix) =
                                <CTYPE_OUT as FromAcc<<CTYPE_OUT as AccType>::Acc>>::from_acc_conv(
                                    <CTYPE_OUT as AccType>::acc_div_f32(sum, num as f32),
                                );
                        }
                    }
                },
            );
            crate::et_kernel_check_msg!(ctx, success, Internal, out, "parallel_for failed");
        });
    });

    out
}

// [spec:et:def:op-mean.torch.executor.native.mean-dtype-out-fn]
// [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn]
pub fn mean_dtype_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    mean_dim_out(ctx, in_, Some(ArrayRef::<i64>::new()), false, dtype, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn op_mean_out<'a, 'b>(
        self_: &Tensor,
        dim: Option<ArrayRef<i64>>,
        keepdim: bool,
        dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        mean_dim_out(&mut ctx, self_, dim, keepdim, dtype, out)
    }

    fn op_mean_dtype_out<'a, 'b>(
        self_: &Tensor,
        dtype: Option<ScalarType>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        mean_dtype_out(&mut ctx, self_, dtype, out)
    }

    fn iarr(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }
    // Bool input is dispatched to the bool-specialized path, so its generic body
    // is never instantiated; this impl only satisfies the trait bound.
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    fn in_data<IN: FromF64Elem>() -> Vec<IN> {
        (0..24).map(|i| IN::from_f64(i as f64)).collect()
    }

    fn test_mean_dim_out_invalid_dimensions<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = tf_in.make_default(vec![2, 3, 4], in_data::<IN>());
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let dtype = Some(OUT::VALUE);

        // out-of-bound dim in dim list
        let dims_1 = [3i64];
        let mut ctx = context();
        mean_dim_out(&mut ctx, &self_, Some(iarr(&dims_1)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // the same dim appears multiple times in list of dims
        let dims_2 = [2i64, 2];
        let mut ctx = context();
        mean_dim_out(&mut ctx, &self_, Some(iarr(&dims_2)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_mean_dim_out_invalid_shape<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let self_ = tf_in.make_default(vec![2, 3, 4], in_data::<IN>());
        let dtype = Some(OUT::VALUE);
        let dims_1 = [1i64];

        // dimension size mismatch when keepdim is true
        let out = tf_out.zeros_default(vec![2, 4]);
        let mut ctx = context();
        mean_dim_out(&mut ctx, &self_, Some(iarr(&dims_1)), true, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // dimension size mismatch when keepdim is false
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        let mut ctx = context();
        mean_dim_out(&mut ctx, &self_, Some(iarr(&dims_1)), false, dtype, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_mean_dim_out_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        // The C++ template specializes Bool input to test_mean_dim_out_bool; the
        // caller dispatches, so this generic body is only instantiated for real
        // (non-bool) inputs.
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let dout = |v: &[f64]| -> Vec<OUT> { v.iter().map(|&x| OUT::from_f64(x)).collect() };
        let self_ = tf_in.make_default(vec![2, 3, 4], in_data::<IN>());
        let dtype = Some(OUT::VALUE);

        // keepdim=true should work
        let out = tf_out.zeros_default(vec![2, 3, 1]);
        let dims_1 = [2i64];
        op_mean_out(&self_, Some(iarr(&dims_1)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 3, 1], dout(&[1.5, 5.5, 9.5, 13.5, 17.5, 21.5]))
        );

        // keepdim=false should work
        let out = tf_out.zeros_default(vec![2, 3]);
        op_mean_out(&self_, Some(iarr(&dims_1)), false, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 3], dout(&[1.5, 5.5, 9.5, 13.5, 17.5, 21.5]))
        );

        // dim list with multiple dimensions should work
        let out = tf_out.zeros_default(vec![1, 1, 4]);
        let dims_2 = [0i64, 1];
        op_mean_out(&self_, Some(iarr(&dims_2)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![1, 1, 4], dout(&[10., 11., 12., 13.]))
        );

        let out = tf_out.zeros_default(vec![4]);
        op_mean_out(&self_, Some(iarr(&dims_2)), false, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![4], dout(&[10., 11., 12., 13.]))
        );

        // dim list with negative dimensions should work
        let out = tf_out.zeros_default(vec![2, 1, 4]);
        let dims_3 = [-2i64];
        op_mean_out(&self_, Some(iarr(&dims_3)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 1, 4], dout(&[4., 5., 6., 7., 16., 17., 18., 19.]))
        );

        // empty/null dim list should work
        let out = tf_out.zeros_default(vec![1, 1, 1]);
        op_mean_out(&self_, None, true, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![1, 1, 1], dout(&[11.5])));

        let empty: [i64; 0] = [];
        op_mean_out(&self_, Some(iarr(&empty)), true, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![1, 1, 1], dout(&[11.5])));

        let out = tf_out.zeros_default(vec![]);
        op_mean_out(&self_, None, false, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![], dout(&[11.5])));

        op_mean_out(&self_, Some(iarr(&empty)), false, dtype, &out);
        assert_tensor_close!(out, tf_out.make_default(vec![], dout(&[11.5])));
    }

    fn test_mean_dim_out_bool<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_bool = TensorFactory::<bool>::new();
        let tf_float = TensorFactory::<OUT>::new();
        let dout = |v: &[f64]| -> Vec<OUT> { v.iter().map(|&x| OUT::from_f64(x)).collect() };
        let self_ = tf_bool.make_default(
            vec![2, 3, 4],
            vec![
                true, false, true, false, false, false, false, false, false, true, true, false,
                false, false, true, false, false, false, false, true, true, true, true, true,
            ],
        );
        let out = tf_float.zeros_default(vec![1, 1, 4]);
        let dims = [0i64, 1];
        let dtype = Some(OUT::VALUE);
        op_mean_out(&self_, Some(iarr(&dims)), true, dtype, &out);
        assert_tensor_close!(
            out,
            tf_float.make_default(
                vec![1, 1, 4],
                dout(&[0.333333, 0.333333, 0.666667, 0.333333])
            )
        );
    }

    // Dispatch mirroring the C++ template specializations for Bool input.
    fn test_mean_dim_out_dtype_dispatch<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        if IN::VALUE == ScalarType::Bool {
            test_mean_dim_out_bool::<OUT>();
        } else {
            test_mean_dim_out_dtype::<IN, OUT>();
        }
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_bfloat16_generic_path_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        const N: usize = 512;
        let x = tf.ones_default(vec![N as i32, 1]);
        let out = tf.zeros_default(vec![1]);
        let dim = [0i64];
        op_mean_out(&x, Some(iarr(&dim)), false, None, &out);
        let expected = tf.full(
            vec![1],
            BFloat16::from_f32(1.0),
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_bfloat16_large_dim_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        const N: usize = 512;
        let x = tf.ones_default(vec![1, N as i32]);
        let out = tf.zeros_default(vec![1]);
        let dim = [1i64];
        op_mean_out(&x, Some(iarr(&dim)), false, None, &out);
        let expected = tf.full(
            vec![1],
            BFloat16::from_f32(1.0),
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_invalid_dimension_list_dies() {
        // ET_FORALL_REAL_TYPES(in) x ET_FORALL_FLOAT_TYPES(out)
        macro_rules! for_in {
            ($in:ty) => {
                test_mean_dim_out_invalid_dimensions::<$in, f32>();
                test_mean_dim_out_invalid_dimensions::<$in, f64>();
            };
        }
        for_in!(u8);
        for_in!(i8);
        for_in!(i16);
        for_in!(i32);
        for_in!(i64);
        for_in!(f32);
        for_in!(f64);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_invalid_shape_dies() {
        macro_rules! for_in {
            ($in:ty) => {
                test_mean_dim_out_invalid_shape::<$in, f32>();
                test_mean_dim_out_invalid_shape::<$in, f64>();
            };
        }
        for_in!(u8);
        for_in!(i8);
        for_in!(i16);
        for_in!(i32);
        for_in!(i64);
        for_in!(f32);
        for_in!(f64);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_mismatched_dtypes_dies() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_int = TensorFactory::<i32>::new();
        let self_ = tf_int.make_default(vec![2, 3, 4], (0..24).collect());
        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims_1 = [2i64];

        // self must have a floating dtype when dtype is not specified
        let mut ctx = context();
        mean_dim_out(&mut ctx, &self_, Some(iarr(&dims_1)), true, None, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // out tensor should be same dtype as dtype when dtype is specified
        let mut ctx = context();
        mean_dim_out(
            &mut ctx,
            &self_,
            Some(iarr(&dims_1)),
            true,
            Some(ScalarType::Double),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_all_real_input_float_output_passes() {
        // ET_FORALL_REALHBBF16_TYPES(in) x ET_FORALL_FLOATHBF16_TYPES(out)
        macro_rules! for_in_out {
            ($in:ty) => {
                test_mean_dim_out_dtype_dispatch::<$in, f32>();
                test_mean_dim_out_dtype_dispatch::<$in, f64>();
                test_mean_dim_out_dtype_dispatch::<$in, Half>();
                test_mean_dim_out_dtype_dispatch::<$in, BFloat16>();
            };
        }
        for_in_out!(u8);
        for_in_out!(i8);
        for_in_out!(i16);
        for_in_out!(i32);
        for_in_out!(i64);
        for_in_out!(f32);
        for_in_out!(f64);
        for_in_out!(Half);
        for_in_out!(BFloat16);
        for_in_out!(bool);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_half_support() {
        // ET_FORALL_REALH_TYPES(in) with out=Half
        test_mean_dim_out_dtype_dispatch::<u8, Half>();
        test_mean_dim_out_dtype_dispatch::<i8, Half>();
        test_mean_dim_out_dtype_dispatch::<i16, Half>();
        test_mean_dim_out_dtype_dispatch::<i32, Half>();
        test_mean_dim_out_dtype_dispatch::<i64, Half>();
        test_mean_dim_out_dtype_dispatch::<f32, Half>();
        test_mean_dim_out_dtype_dispatch::<f64, Half>();
        test_mean_dim_out_dtype_dispatch::<Half, Half>();

        // ET_FORALL_FLOATH_TYPES(out) with in=Half
        test_mean_dim_out_dtype::<Half, f32>();
        test_mean_dim_out_dtype::<Half, f64>();
        test_mean_dim_out_dtype::<Half, Half>();
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_infinity_and_nan_test() {
        let tf_float = TensorFactory::<f32>::new();
        let inf = f32::INFINITY;
        let nan = f32::NAN;
        let self_ = tf_float.make_default(
            vec![2, 3, 4],
            vec![
                0., 1., 2., inf, inf, -inf, 1., 0., nan, inf, -inf, 2., nan, nan, 1., 0., 0., inf,
                nan, 4., 1., nan, 3.14, 2.,
            ],
        );
        let out = tf_float.zeros_default(vec![2, 3, 1]);
        let dims = [-1i64];
        op_mean_out(&self_, Some(iarr(&dims)), true, None, &out);
        assert_tensor_close!(
            out,
            tf_float.make_default(vec![2, 3, 1], vec![inf, nan, nan, nan, nan, nan])
        );
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.ones_default(vec![10, 10]);
        let expected_result = tf.ones_default(vec![10]);
        let out = tf.zeros_default(vec![10]);
        let dim = [1i64];
        op_mean_out(&x, Some(iarr(&dim)), false, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    fn dyn_shape_data<'a>(tf: &'a TensorFactory<f32>) -> (Tensor<'a>, Tensor<'a>) {
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.49627798795700073,
                0.40115922689437866,
                0.5627331733703613,
                0.3858276605606079,
                0.4964867830276489,
                0.5637965202331543,
            ],
        );
        let expected = tf.make_default(
            vec![3],
            vec![0.4487186074256897, 0.4742804169654846, 0.5301416516304016],
        );
        (x, expected)
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![3], TensorShapeDynamism::DYNAMIC_BOUND);
        let dim = [1i64];
        op_mean_out(&x, Some(iarr(&dim)), false, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let dim = [1i64];
        op_mean_out(&x, Some(iarr(&dim)), false, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: DISABLED_DynamicShapeUnbound in C++; ported and #[ignore]d.
    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    #[ignore]
    fn op_mean_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let dim = [1i64];
        op_mean_out(&x, Some(iarr(&dim)), false, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn/test]
    #[test]
    fn op_mean_out_test_dtype_out_float_valid() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.ones_default(vec![10, 10]);
        let expected_result = tf.make_default(vec![], vec![1.0]);
        let out = tf.zeros_default(vec![]);
        op_mean_dtype_out(&x, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn/test]
    #[test]
    fn op_mean_out_test_dtype_out_float_to_bool_invalid() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.ones_default(vec![10, 10]);
        let out = tf.zeros_default(vec![]);
        let mut ctx = context();
        mean_dtype_out(&mut ctx, &x, Some(ScalarType::Bool), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn/test]
    #[test]
    fn op_mean_out_test_dtype_out_float_infinity() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![2, 1], vec![f32::INFINITY, f32::INFINITY]);
        let expected_result = tf.make_default(vec![], vec![f32::INFINITY]);
        let out = tf.zeros_default(vec![]);
        op_mean_dtype_out(&x, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn/test]
    #[test]
    fn op_mean_out_test_dtype_out_float_nan() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![2, 1], vec![f32::NAN, f32::INFINITY]);
        let expected_result = tf.make_default(vec![], vec![f32::NAN]);
        let out = tf.zeros_default(vec![]);
        op_mean_dtype_out(&x, Some(ScalarType::Float), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn/test]
    #[test]
    fn op_mean_out_test_empty_input() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![2, 0, 3], vec![]);
        let dtype = Some(ScalarType::Float);

        let empty: [i64; 0] = [];
        let out = tf.zeros_default(vec![1, 1, 1]);
        op_mean_out(&x, Some(iarr(&empty)), true, dtype, &out);
        assert_tensor_close!(out, tf.make_default(vec![1, 1, 1], vec![f32::NAN]));

        let out = tf.zeros_default(vec![]);
        op_mean_out(&x, Some(iarr(&empty)), false, dtype, &out);
        assert_tensor_close!(out, tf.make_default(vec![], vec![f32::NAN]));

        let dims1 = [1i64];
        let out = tf.zeros_default(vec![2, 3]);
        op_mean_out(&x, Some(iarr(&dims1)), false, dtype, &out);
        assert_tensor_close!(
            out,
            tf.make_default(
                vec![2, 3],
                vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN]
            )
        );

        let dims2 = [2i64];
        let out = tf.make_default(vec![2, 0, 1], vec![]);
        op_mean_out(&x, Some(iarr(&dims2)), true, dtype, &out);
        assert_tensor_close!(out, tf.make_default(vec![2, 0, 1], vec![]));
    }
}
