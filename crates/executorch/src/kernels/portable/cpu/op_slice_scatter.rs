//! Literal port of kernels/portable/cpu/op_slice_scatter.cpp.

use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::kernels::portable::cpu::util::slice_util::{
    adjust_slice_indices, check_slice_scatter_args,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, getTrailingDims, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The void-typed
// `memcpy(out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes())` is a
// byte-wise `copy_nonoverlapping`.

// PORT-NOTE: local port of the scalar_type_util `convert<To, From>` template
// (`convert<CTYPE, CTYPE_SRC>`). Two C++ overloads: floating `From` -> integral
// `To` goes through `static_cast<int64_t>` first; every other pair is a plain
// `static_cast<To>`. Mirrors the `Convert` trait established in op_unbind_copy.
trait Convert<From> {
    fn convert(val: From) -> Self;
}
macro_rules! impl_convert_plain {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<$from>>::static_cast(val)
            }
        }
    };
}
macro_rules! impl_convert_float_to_int {
    ($to:ty, $from:ty) => {
        impl Convert<$from> for $to {
            #[inline]
            fn convert(val: $from) -> Self {
                <$to as StaticCast<i64>>::static_cast(<i64 as StaticCast<$from>>::static_cast(val))
            }
        }
    };
}

use crate::runtime::core::portable_type::{BFloat16, Half};

macro_rules! impl_convert_row_plain {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_plain!($to, f32);
        impl_convert_plain!($to, f64);
        impl_convert_plain!($to, Half);
        impl_convert_plain!($to, BFloat16);
    };
}
impl_convert_row_plain!(f32);
impl_convert_row_plain!(f64);
impl_convert_row_plain!(Half);
impl_convert_row_plain!(BFloat16);

macro_rules! impl_convert_row_int {
    ($to:ty) => {
        impl_convert_plain!($to, u8);
        impl_convert_plain!($to, i8);
        impl_convert_plain!($to, i16);
        impl_convert_plain!($to, i32);
        impl_convert_plain!($to, i64);
        impl_convert_plain!($to, bool);
        impl_convert_float_to_int!($to, f32);
        impl_convert_float_to_int!($to, f64);
        impl_convert_float_to_int!($to, Half);
        impl_convert_float_to_int!($to, BFloat16);
    };
}
impl_convert_row_int!(u8);
impl_convert_row_int!(i8);
impl_convert_row_int!(i16);
impl_convert_row_int!(i32);
impl_convert_row_int!(i64);
impl_convert_row_int!(bool);

// [spec:et:def:op-slice-scatter.torch.executor.native.slice-scatter-out-fn]
// [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn]
pub fn slice_scatter_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    src: &Tensor,
    mut dim: i64,
    start_val: Option<i64>,
    end_val: Option<i64>,
    step: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    if dim < 0 {
        dim += input.dim() as i64;
    }

    // resize out tensor for dynamic shapes
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(input, out),
        InvalidArgument,
        out
    );

    if input.numel() == 0 {
        return out;
    }

    crate::et_kernel_check!(
        ctx,
        dim >= 0 && dim < input.dim() as i64,
        InvalidArgument,
        out
    );

    // If user do not set value to end_val, set end to input.size(dim) (largest
    // value available)
    let mut end: i64 = match end_val {
        Some(v) => v,
        None => input.size(dim as isize) as i64,
    };
    // If user do not set value to start_val, set start to 0 (smallest value
    // available)
    let mut start: i64 = match start_val {
        Some(v) => v,
        None => 0,
    };

    crate::et_kernel_check!(ctx, step > 0, InvalidArgument, out);

    let num_values: i64 =
        adjust_slice_indices(input.size(dim as isize) as i64, &mut start, &mut end, step);

    crate::et_kernel_check!(
        ctx,
        check_slice_scatter_args(input, src, dim, num_values, step, out),
        InvalidArgument,
        out
    );

    let dim_length: usize = input.size(dim as isize) as usize;
    let leading_dims: usize = getLeadingDims(input, dim);
    let trailing_dims: usize = getTrailingDims(input, dim);

    // To start, copy the input into the output
    unsafe {
        core::ptr::copy_nonoverlapping(
            input.const_data_ptr_typed() as *const u8,
            out.mutable_data_ptr_typed() as *mut u8,
            input.nbytes(),
        );
    }

    let in_type: ScalarType = input.scalar_type();
    let src_type: ScalarType = src.scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "slice_scatter.out", CTYPE, {
        crate::et_switch_realhbbf16_types!(src_type, ctx, "slice_scatter.out", CTYPE_SRC, {
            let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
            let src_data: *const CTYPE_SRC = src.const_data_ptr::<CTYPE_SRC>();

            let mut src_offset: usize = 0;

            for i in 0..leading_dims {
                let mut out_offset: usize = (i * dim_length + start as usize) * trailing_dims;
                for _j in 0..num_values {
                    for k in 0..trailing_dims {
                        unsafe {
                            *out_data.add(out_offset + k) = <CTYPE as Convert<CTYPE_SRC>>::convert(
                                *src_data.add(src_offset + k),
                            );
                        }
                    }
                    src_offset += trailing_dims;
                    out_offset += step as usize * trailing_dims;
                }
            }
        });
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        #[rustfmt::skip]
        let input = tf.make_default(
            vec![3, 4],
            vec![
                T::from_i32(1),  T::from_i32(2),  T::from_i32(3),  T::from_i32(4),
                T::from_i32(5),  T::from_i32(6),  T::from_i32(7),  T::from_i32(8),
                T::from_i32(9),  T::from_i32(10), T::from_i32(11), T::from_i32(12),
            ],
        );

        #[rustfmt::skip]
        let src = tf.make_default(
            vec![2, 4],
            vec![
                T::from_i32(5), T::from_i32(6), T::from_i32(7), T::from_i32(8),
                T::from_i32(1), T::from_i32(2), T::from_i32(3), T::from_i32(4),
            ],
        );
        #[rustfmt::skip]
        let expect_ret = tf.make_default(
            vec![3, 4],
            vec![
                T::from_i32(5), T::from_i32(6),  T::from_i32(7),  T::from_i32(8),
                T::from_i32(1), T::from_i32(2),  T::from_i32(3),  T::from_i32(4),
                T::from_i32(9), T::from_i32(10), T::from_i32(11), T::from_i32(12),
            ],
        );

        let out = tf.zeros_default(vec![3, 4]);
        let mut ctx = context();
        let ret = slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(2), 1, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, expect_ret);
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_legal_dim_supported() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        #[rustfmt::skip]
        let src_dim_0 = tf.make_default(
            vec![1, 3, 4],
            vec![
                8.,   7.,   6.,   5.,
                4.,   3.,   2.,   1.,
                1.,  14.,  18.,  19.,
            ],
        );
        #[rustfmt::skip]
        let expected_dim_0 = tf.make_default(
            vec![2, 3, 4],
            vec![
                 8.,   7.,   6.,   5.,
                 4.,   3.,   2.,   1.,
                 1.,  14.,  18.,  19.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let src_dim_1 = tf.make_default(
            vec![2, 1, 4],
            vec![
                 4.,   3.,   2.,   1.,
                -4.,  -3.,  -2.,  -1.,
            ],
        );
        #[rustfmt::skip]
        let expected_dim_1 = tf.make_default(
            vec![2, 3, 4],
            vec![
                 4.,   3.,   2.,   1.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -4.,  -3.,  -2.,  -1.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let src_dim_2 = tf.make_default(
            vec![2, 3, 1],
            vec![
                 7.,   1.,   6.,
                -5.,  -9.,  -2.,
            ],
        );
        #[rustfmt::skip]
        let expected_dim_2 = tf.make_default(
            vec![2, 3, 4],
            vec![
                 7.,   2.,   3.,   4.,
                 1.,   6.,   7.,   8.,
                 6.,  10.,  11.,  12.,
                -5.,  -2.,  -3.,  -4.,
                -9.,  -6.,  -7.,  -8.,
                -2., -10., -11., -12.,
            ],
        );

        for dim in -3i64..3 {
            let testcase_idx = dim + 3;
            let (src, expected_ret) = match testcase_idx {
                0 | 3 => (&src_dim_0, &expected_dim_0),
                1 | 4 => (&src_dim_1, &expected_dim_1),
                _ => (&src_dim_2, &expected_dim_2),
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = slice_scatter_out(&mut ctx, &input, src, dim, Some(0), Some(1), 1, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_all_start_vals_supported() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        #[rustfmt::skip]
        let src_start_0_or_below = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        let expected_start_0_or_below = src_start_0_or_below_clone(&tf);
        #[rustfmt::skip]
        let src_start_1 = tf.make_default(
            vec![2, 2, 4],
            vec![
                -9., -10., -11., -12.,
                -5.,  -6.,  -7.,  -8.,
                 9.,  10.,  11.,  12.,
                 5.,   6.,   7.,   8.,
            ],
        );
        #[rustfmt::skip]
        let expected_start_1 = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                -9., -10., -11., -12.,
                -5.,  -6.,  -7.,  -8.,
                -1.,  -2.,  -3.,  -4.,
                 9.,  10.,  11.,  12.,
                 5.,   6.,   7.,   8.,
            ],
        );
        #[rustfmt::skip]
        let src_start_2 = tf.make_default(
            vec![2, 1, 4],
            vec![
                 1.,  19.,  18.,  17.,
                -1., -19., -18., -17.,
            ],
        );
        #[rustfmt::skip]
        let expected_start_2 = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 1.,  19.,  18.,  17.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -1., -19., -18., -17.,
            ],
        );
        let src_start_3_or_above = tf.make_default(vec![2, 0, 4], vec![]);
        #[rustfmt::skip]
        let expected_start_3_or_above = tf.make_default(
            vec![2, 3, 4],
            vec![
                1.,   2.,   3.,   4.,
                5.,   6.,   7.,   8.,
                9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let dim: i64 = 1;
        let end: i64 = 10;
        let step: i64 = 1;
        for start in -3i64..4 {
            let testcase_idx = start + 3;
            let (src, expected_ret) = match testcase_idx {
                0 | 3 => (&src_start_0_or_below, &expected_start_0_or_below),
                1 | 4 => (&src_start_1, &expected_start_1),
                2 | 5 => (&src_start_2, &expected_start_2),
                _ => (&src_start_3_or_above, &expected_start_3_or_above),
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = slice_scatter_out(
                &mut ctx,
                &input,
                src,
                dim,
                Some(start),
                Some(end),
                step,
                &out,
            );
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_all_end_vals_supported() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let src_end_0_or_below = tf.make_default(vec![2, 0, 4], vec![]);
        #[rustfmt::skip]
        let expected_end_0_or_below = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let src_end_1 = tf.make_default(
            vec![2, 1, 4],
            vec![
                -4.,  -3.,  -2.,  -1.,
                 4.,   3.,   2.,   1.,
            ],
        );
        #[rustfmt::skip]
        let expected_end_1 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -4.,  -3.,  -2.,  -1.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                 4.,   3.,   2.,   1.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let src_end_2 = tf.make_default(
            vec![2, 2, 4],
            vec![
                -8.,  -7.,  -6.,  -5.,
                -4.,  -3.,  -2.,  -1.,
                 8.,   7.,   6.,   5.,
                 4.,   3.,   2.,   1.,
            ],
        );
        #[rustfmt::skip]
        let expected_end_2 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -8.,  -7.,  -6.,  -5.,
                -4.,  -3.,  -2.,  -1.,
                 9.,  10.,  11.,  12.,
                 8.,   7.,   6.,   5.,
                 4.,   3.,   2.,   1.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let src_end_3_or_above = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let expected_end_3_or_above = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );

        let dim: i64 = 1;
        let start: i64 = 0;
        let step: i64 = 1;
        for end in -3i64..4 {
            let testcase_idx = end + 3;
            let (src, expected_ret) = match testcase_idx {
                0 | 3 => (&src_end_0_or_below, &expected_end_0_or_below),
                1 | 4 => (&src_end_1, &expected_end_1),
                2 | 5 => (&src_end_2, &expected_end_2),
                _ => (&src_end_3_or_above, &expected_end_3_or_above),
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = slice_scatter_out(
                &mut ctx,
                &input,
                src,
                dim,
                Some(start),
                Some(end),
                step,
                &out,
            );
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_legal_steps_supported() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        #[rustfmt::skip]
        let src_0 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let expected_0 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let src_1 = tf.make_default(
            vec![2, 2, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let expected_1 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                 5.,   6.,   7.,   8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                -5.,  -6.,  -7.,  -8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let src_2 = tf.make_default(
            vec![2, 1, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                 1.,   2.,   3.,   4.,
            ],
        );
        #[rustfmt::skip]
        let expected_2 = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
                 1.,   2.,   3.,   4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let start: i64 = 0;
        let dim: i64 = 1;
        let end: i64 = 10;
        for step in 1i64..4 {
            let testcase_idx = step - 1;
            let (src, expected_ret) = match testcase_idx {
                0 => (&src_0, &expected_0),
                1 => (&src_1, &expected_1),
                _ => (&src_2, &expected_2),
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = slice_scatter_out(
                &mut ctx,
                &input,
                src,
                dim,
                Some(start),
                Some(end),
                step,
                &out,
            );
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_all_real_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_empty_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 0, 1]);
        let src = tf.zeros_default(vec![1, 0, 1]);
        let out = tf.zeros_default(vec![1, 0, 1]);

        let expect = tf.ones_default(vec![1, 0, 1]);

        let mut dim: i64 = 0;
        while dim > input.dim() as i64 {
            let mut ctx = context();
            let ret = slice_scatter_out(&mut ctx, &input, &src, dim, Some(0), Some(1), 1, &out);
            assert_tensor_eq!(*ret, out);
            assert_tensor_eq!(*ret, expect);
            dim += 1;
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_empty_size_input_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![]);
        let src = tf.ones_default(vec![]);
        let out = tf.ones_default(vec![]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(0), 1, &out)
        );
        et_expect_kernel_failure!(
            ctx,
            slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(1), 1, &out)
        );
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_non_positive_steps_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let src = tf.zeros_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_steps: [i64; 3] = [-2, -1, 0];
        for step in invalid_steps {
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(1), step, &out)
            );
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let src = tf.zeros_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_dims: [i64; 6] = [3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                slice_scatter_out(&mut ctx, &input, &src, dim, Some(0), Some(1), 1, &out)
            );
        }
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_mismatched_out_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let input = tf_int.zeros_default(vec![1, 2, 2]);
        let src = tf_int.zeros_default(vec![1, 2, 2]);

        let out = tf_float.ones_default(vec![1, 2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(1), 1, &out)
        );
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_out_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);
        let src = tf.zeros_default(vec![2, 4, 7, 5]);

        let out = tf.zeros_default(vec![2, 4, 7]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(2), 1, &out)
        );
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    // also verifies check_slice_scatter_args: src rank/size differing from input
    // (here src is 3-dim vs 4-dim input) is rejected.
    // [spec:et:sem:slice-util.torch.executor.check-slice-scatter-args-fn/test]
    #[test]
    fn op_slice_scatter_out_test_src_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);
        let src = tf.zeros_default(vec![2, 4, 7]);

        let out = tf.zeros_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), Some(2), 1, &out)
        );
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_default_start_val_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);
        let src = tf.ones_default(vec![2, 4, 7, 5]);

        let out = tf.zeros_default(vec![2, 4, 7, 5]);
        let expected = tf.ones_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        let ret_default_start =
            slice_scatter_out(&mut ctx, &input, &src, 0, None, Some(2), 1, &out);
        assert_tensor_eq!(*ret_default_start, out);
        assert_tensor_eq!(*ret_default_start, expected);
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_default_end_val_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);
        let src = tf.ones_default(vec![2, 4, 7, 5]);

        let out = tf.zeros_default(vec![2, 4, 7, 5]);
        let expected = tf.ones_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        let ret_default_end = slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), None, 1, &out);
        assert_tensor_eq!(*ret_default_end, out);
        assert_tensor_eq!(*ret_default_end, expected);
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_dynamic_shape_test() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![1, 4, 4]);
        let src = tf.ones_default(vec![1, 4, 4]);

        let out = tf.zeros(vec![1, 2, 8], TensorShapeDynamism::DYNAMIC_BOUND);
        let expected = tf.ones_default(vec![1, 4, 4]);

        let mut ctx = context();
        let ret_default_end = slice_scatter_out(&mut ctx, &input, &src, 0, Some(0), None, 1, &out);
        assert_tensor_eq!(*ret_default_end, out);
        assert_tensor_eq!(*ret_default_end, expected);
    }

    // [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn/test]
    #[test]
    fn op_slice_scatter_out_test_large_end_value() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![1, 1, 2, 5, 3, 3]);
        let src = tf.ones_default(vec![1, 1, 2, 5, 3, 3]);

        let out = tf.zeros_default(vec![1, 1, 2, 5, 3, 3]);
        let expected = tf.ones_default(vec![1, 1, 2, 5, 3, 3]);

        let mut ctx = context();
        let ret = slice_scatter_out(
            &mut ctx,
            &input,
            &src,
            1,
            Some(0),
            Some(9223372036854775807),
            1,
            &out,
        );
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expected);
    }

    // Helper: the C++ test builds `expected_start_0_or_below` with the same data
    // as `src_start_0_or_below`; rebuilt here to obtain an independent Tensor.
    fn src_start_0_or_below_clone(tf: &TensorFactory<f64>) -> Tensor<'_> {
        #[rustfmt::skip]
        let t = tf.make_default(
            vec![2, 3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
                 1.,   2.,   3.,   4.,
                 5.,   6.,   7.,   8.,
                 9.,  10.,  11.,  12.,
            ],
        );
        t
    }
}
