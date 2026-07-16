//! Literal port of kernels/portable/cpu/op_slice_copy.cpp.

use crate::kernels::portable::cpu::util::slice_util::{
    adjust_slice_indices, check_slice_copy_args, compute_slice, get_slice_copy_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The C++
// `Tensor::SizesType target_sizes[kTensorDimensionLimit]` stack buffer is a Rust
// fixed-size array; `resize_tensor(out, {target_sizes, target_ndim})` becomes an
// `ArrayRef` built from the filled prefix.

// [spec:et:def:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn]
// [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn]
#[executorch_macros::et_kernel("aten::slice_copy.Tensor_out")]
pub fn slice_copy_Tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    start_val: Option<i64>,
    end_val: Option<i64>,
    step: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_slice_copy_args(in_, dim, step, out),
        InvalidArgument,
        out
    );

    if dim < 0 {
        dim += in_.dim() as i64;
    }

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    // If user do not set value to end_val, set end to in.size(dim) (largest
    // value available)
    let mut end: i64 = match end_val {
        Some(v) => v,
        None => in_.size(dim as isize) as i64,
    };
    // If user do not set value to start_val, set start to 0 (smallest value
    // available)
    let mut start: i64 = match start_val {
        Some(v) => v,
        None => 0,
    };

    let length: i64 =
        adjust_slice_indices(in_.size(dim as isize) as i64, &mut start, &mut end, step);

    let mut target_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_slice_copy_out_target_size(
            in_,
            dim,
            length,
            target_sizes.as_mut_ptr(),
            &mut target_ndim,
        );
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    compute_slice(ctx, in_, dim, start, length, step, out);

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
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
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
        let expect_ret = tf.make_default(
            vec![2, 4],
            vec![
                T::from_i32(1), T::from_i32(2), T::from_i32(3), T::from_i32(4),
                T::from_i32(5), T::from_i32(6), T::from_i32(7), T::from_i32(8),
            ],
        );

        let out = tf.zeros_default(vec![2, 4]);
        let mut ctx = context();
        let ret = slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(2), 1, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, expect_ret);
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    // also verifies compute_slice: copies the correct element strides for each dim
    // [spec:et:sem:slice-util.torch.executor.compute-slice-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_legal_dim_supported() {
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
        let expected_dim_0 = tf.make_default(
            vec![1, 3, 4],
            vec![
                1.,   2.,   3.,   4.,
                5.,   6.,   7.,   8.,
                9.,  10.,  11.,  12.,
            ],
        );
        #[rustfmt::skip]
        let expected_dim_1 = tf.make_default(
            vec![2, 1, 4],
            vec![
                1.,   2.,   3.,   4.,
                -1.,  -2.,  -3.,  -4.,
            ],
        );
        #[rustfmt::skip]
        let expected_dim_2 = tf.make_default(
            vec![2, 3, 1],
            vec![
                1.,   5.,   9.,
                -1.,  -5.,  -9.,
            ],
        );

        for dim in -3i64..3 {
            let testcase_idx = dim + 3;
            let expected_ret = match testcase_idx {
                0 | 3 => &expected_dim_0,
                1 | 4 => &expected_dim_1,
                _ => &expected_dim_2,
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = slice_copy_Tensor_out(&mut ctx, &input, dim, Some(0), Some(1), 1, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    // also verifies adjust_slice_indices: negative/overflow start values clamp to
    // [0, dim_length] and yield the right num_values per case.
    // [spec:et:sem:slice-util.torch.executor.adjust-slice-indices-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_all_start_vals_supported() {
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
        let expected_start_0_or_below = tf.make_default(
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
        let expected_start_1 = tf.make_default(
            vec![2, 2, 4],
            vec![
                5.,   6.,   7.,   8.,
                9.,  10.,  11.,  12.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let expected_start_2 = tf.make_default(
            vec![2, 1, 4],
            vec![
                9.,  10.,  11.,  12.,
                -9., -10., -11., -12.,
            ],
        );
        let expected_start_3_or_above = tf.make_default(vec![2, 0, 4], vec![]);

        let dim: i64 = 1;
        let end: i64 = 10;
        let step: i64 = 1;
        for start in -3i64..4 {
            let testcase_idx = start + 3;
            let expected_ret = match testcase_idx {
                0 | 3 => &expected_start_0_or_below,
                1 | 4 => &expected_start_1,
                2 | 5 => &expected_start_2,
                _ => &expected_start_3_or_above,
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret =
                slice_copy_Tensor_out(&mut ctx, &input, dim, Some(start), Some(end), step, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_all_end_vals_supported() {
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

        let expected_end_0_or_below = tf.make_default(vec![2, 0, 4], vec![]);
        #[rustfmt::skip]
        let expected_end_1 = tf.make_default(
            vec![2, 1, 4],
            vec![
                1.,   2.,   3.,   4.,
                -1.,  -2.,  -3.,  -4.,
            ],
        );
        #[rustfmt::skip]
        let expected_end_2 = tf.make_default(
            vec![2, 2, 4],
            vec![
                1.,   2.,   3.,   4.,
                5.,   6.,   7.,   8.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
            ],
        );
        #[rustfmt::skip]
        let expected_end_3_or_above = tf.make_default(
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
        let start: i64 = 0;
        let step: i64 = 1;
        for end in -3i64..4 {
            let testcase_idx = end + 3;
            let expected_ret = match testcase_idx {
                0 | 3 => &expected_end_0_or_below,
                1 | 4 => &expected_end_1,
                2 | 5 => &expected_end_2,
                _ => &expected_end_3_or_above,
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret =
                slice_copy_Tensor_out(&mut ctx, &input, dim, Some(start), Some(end), step, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_legal_steps_supported() {
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
        let expected_0 = tf.make_default(
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
        let expected_1 = tf.make_default(
            vec![2, 2, 4],
            vec![
                1.,   2.,   3.,   4.,
                9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -9., -10., -11., -12.,
            ],
        );
        #[rustfmt::skip]
        let expected_2 = tf.make_default(
            vec![2, 1, 4],
            vec![
                1.,   2.,   3.,   4.,
                -1.,  -2.,  -3.,  -4.,
            ],
        );

        let start: i64 = 0;
        let dim: i64 = 1;
        let end: i64 = 10;
        for step in 1i64..4 {
            let testcase_idx = step - 1;
            let expected_ret = match testcase_idx {
                0 => &expected_0,
                1 => &expected_1,
                _ => &expected_2,
            };
            let out = tf.zeros_like(expected_ret, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret =
                slice_copy_Tensor_out(&mut ctx, &input, dim, Some(start), Some(end), step, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(*ret, *expected_ret);
        }
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here (this is the
    // non-ATen kernel), so the body always runs.
    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_empty_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 0, 1]);
        let out = tf.zeros_default(vec![1, 0, 1]);

        let expect = tf.ones_default(vec![1, 0, 1]);

        // Some invalid dim values. (loop body never executes, mirroring the C++.)
        let mut dim: i64 = 0;
        while dim > input.dim() as i64 {
            let mut ctx = context();
            let ret = slice_copy_Tensor_out(&mut ctx, &input, dim, Some(0), Some(1), 1, &out);
            assert_tensor_eq!(*ret, out);
            assert_tensor_eq!(*ret, expect);
            dim += 1;
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_empty_size_input_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![]);
        let out = tf.ones_default(vec![]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(0), 1, &out)
        );
        et_expect_kernel_failure!(
            ctx,
            slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(1), 1, &out)
        );
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_zero_length_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 3]);
        let out = tf.ones_default(vec![2, 0]);

        let expect = tf.ones_default(vec![2, 0]);

        let mut ctx = context();
        let ret = slice_copy_Tensor_out(&mut ctx, &input, 1, Some(1), Some(1), 1, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);

        let ret = slice_copy_Tensor_out(&mut ctx, &input, 1, Some(-1), Some(-1), 1, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    // also verifies check_slice_copy_args rejects step <= 0.
    // [spec:et:sem:slice-util.torch.executor.check-slice-copy-args-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_non_positive_steps_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_steps: [i64; 3] = [-2, -1, 0];
        for step in invalid_steps {
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(1), step, &out)
            );
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_dims: [i64; 6] = [3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(
                ctx,
                slice_copy_Tensor_out(&mut ctx, &input, dim, Some(0), Some(1), 1, &out)
            );
        }
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let input = tf_int.zeros_default(vec![1, 2, 2]);

        let out = tf_float.ones_default(vec![1, 2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(1), 1, &out)
        );
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    // also verifies get_slice_copy_out_target_size: target ndim = in.dim() (4),
    // which mismatches the 3-dim out and fails resize.
    // [spec:et:sem:slice-util.torch.executor.get-slice-copy-out-target-size-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_out_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);

        let out = tf.zeros_default(vec![2, 4, 7]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), Some(2), 1, &out)
        );
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_default_start_val_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);

        let out = tf.ones_default(vec![2, 4, 7, 5]);
        let expected = tf.zeros_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        let ret_default_start = slice_copy_Tensor_out(&mut ctx, &input, 0, None, Some(2), 1, &out);
        assert_tensor_eq!(*ret_default_start, out);
        assert_tensor_eq!(*ret_default_start, expected);
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_default_end_val_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.zeros_default(vec![2, 4, 7, 5]);

        let out = tf.ones_default(vec![2, 4, 7, 5]);
        let expected = tf.zeros_default(vec![2, 4, 7, 5]);

        let mut ctx = context();
        let ret_default_end = slice_copy_Tensor_out(&mut ctx, &input, 0, Some(0), None, 1, &out);
        assert_tensor_eq!(*ret_default_end, out);
        assert_tensor_eq!(*ret_default_end, expected);
    }

    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 6, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.41940832138061523,
                0.5529070496559143,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
                0.3051000237464905,
                0.9320003986358643,
                0.17591017484664917,
                0.2698335647583008,
                0.15067976713180542,
                0.03171950578689575,
            ],
        );
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
            ],
        );

        let out = tf.zeros(vec![2, 2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        slice_copy_Tensor_out(&mut ctx, &x, 1, Some(1), Some(5), 2, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel, whose `SupportedFeatures::output_resize` default is `false`
    // (kernels/test/supported_features.yaml). Mirrored as an early skip.
    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
        return;
        #[allow(unreachable_code)]
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 6, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.41940832138061523,
                0.5529070496559143,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
                0.3051000237464905,
                0.9320003986358643,
                0.17591017484664917,
                0.2698335647583008,
                0.15067976713180542,
                0.03171950578689575,
            ],
        );
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
            ],
        );

        let out = tf.zeros(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        slice_copy_Tensor_out(&mut ctx, &x, 1, Some(1), Some(5), 2, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (output_resize default false). Mirrored as an early skip.
    // [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn/test]
    #[test]
    fn op_slice_copy_tensor_out_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
        return;
        #[allow(unreachable_code)]
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 6, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.41940832138061523,
                0.5529070496559143,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
                0.3051000237464905,
                0.9320003986358643,
                0.17591017484664917,
                0.2698335647583008,
                0.15067976713180542,
                0.03171950578689575,
            ],
        );
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
            ],
        );

        let out = tf.zeros(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        slice_copy_Tensor_out(&mut ctx, &x, 1, Some(1), Some(5), 2, &out);
        assert_tensor_eq!(out, expected);
    }
}
