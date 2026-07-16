//! Literal port of kernels/portable/cpu/op_index_select.cpp.

use crate::kernels::portable::cpu::util::index_util::{
    check_index_select_args, get_index_select_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, nonzero_dim,
    resize_tensor_same_type, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::core::portable_type::tensor_impl::ssize_t;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the tensor's `*mut TensorImpl`).

// [spec:et:def:op-index-select.torch.executor.native.index-select-out-fn]
// [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn]
pub fn index_select_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_index_select_args(in_, dim, index, out),
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

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    let mut expected_ndim: usize = 0;
    let mut expected_size: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_index_select_out_target_size(
            in_,
            dim,
            index,
            expected_size.as_mut_ptr(),
            &mut expected_ndim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_size.as_ptr(), expected_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if in_.dim() == 0 {
        unsafe {
            core::ptr::copy_nonoverlapping(
                in_.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                in_.nbytes(),
            );
        }
        return out;
    }

    let leading_dims: usize = getLeadingDims(in_, dim);
    let trailing_dims: usize = getTrailingDims(in_, dim);

    if leading_dims == 0 || trailing_dims == 0 {
        return out;
    }

    let out_dim_length: usize = out.size(dim as ssize_t) as usize;
    let in_dim_length: usize = in_.size(dim as ssize_t) as usize;

    let length_per_step: usize = trailing_dims * in_.element_size() as usize;

    let input_data: *const u8 = in_.const_data_ptr::<u8>();
    let out_data: *mut u8 = out.mutable_data_ptr::<u8>();

    let ix_type: ScalarType = index.scalar_type();

    crate::et_switch_two_types!(Long, Int, ix_type, ctx, "index_select.out", CTYPE, {
        let index_arr: *const CTYPE = index.mutable_data_ptr::<CTYPE>();
        for i in 0..leading_dims {
            let src: *const u8 = unsafe { input_data.add(i * in_dim_length * length_per_step) };
            let mut dest: *mut u8 = unsafe { out_data.add(i * out_dim_length * length_per_step) };
            for j in 0..out_dim_length {
                let copy_src: *const u8 =
                    unsafe { src.add(*index_arr.add(j) as usize * length_per_step) };
                unsafe {
                    core::ptr::copy_nonoverlapping(copy_src, dest, length_per_step);
                    dest = dest.add(length_per_step);
                }
            }
        }
    });

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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

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

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let tfl = TensorFactory::<i64>::new();

        // test index_select on dimension 0. all ones are from x, zeros from y.
        let x = tf.make_default(
            vec![3, 2, 4],
            vec![
                T::one(),
                T::one(),
                T::one(),
                T::one(), // [0, 0, :]
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), // [0, 1, :]
                T::one(),
                T::one(),
                T::one(),
                T::one(), // [1, 0, :]
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), // [1, 1, :]
                T::one(),
                T::one(),
                T::one(),
                T::one(), // [2, 0, :]
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(), // [2, 1, :]
            ],
        );

        let out_0 = tf.zeros_default(vec![3, 1, 4]);
        let out_1 = tf.ones_default(vec![3, 1, 4]);
        let index_0 = tfl.make_default(vec![1], vec![0]);
        let index_1 = tfl.make_default(vec![1], vec![1]);

        let mut ctx = context();
        let ret_0 = index_select_out(&mut ctx, &x, 1, &index_0, &out_0);
        assert_tensor_eq!(*ret_0, out_0);
        let ret_1 = index_select_out(&mut ctx, &x, 1, &index_1, &out_1);
        assert_tensor_eq!(*ret_1, out_1);

        assert_tensor_eq!(out_0, tf.ones_default(vec![3, 1, 4]));
        assert_tensor_eq!(out_1, tf.zeros_default(vec![3, 1, 4]));
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();
        let tf_index = TensorFactory::<i64>::new();

        let input = tf.make_default(
            vec![2, 3, 4],
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
            ],
        );
        let index = tf_index.make_default(vec![2], vec![0, 2]);
        let expected = tf.make_default(
            vec![2, 3, 2],
            vec![
                0.49625658988952637,
                0.08847743272781372,
                0.30742281675338745,
                0.4900934100151062,
                0.455627977848053,
                0.3488934636116028,
                0.022325754165649414,
                0.2938884496688843,
                0.6976675987243652,
                0.16102945804595947,
                0.6816085577011108,
                0.39709991216659546,
            ],
        );
        let out = tf.zeros(out_shape, dynamism);

        let mut ctx = context();
        index_select_out(&mut ctx, &input, 2, &index, &out);
        assert_tensor_close!(out, expected);
    }

    // Run the test by selecting Tensor x on given dim and all available indexes
    // on that dimension.
    fn run_test_cases(x: &Tensor, dim: i64, index: &Tensor, expected: &Tensor) {
        // Generated out tensor sharing same size and dtype with expected tensor.
        let tf = TensorFactory::<f64>::new();

        let expected_sizes = expected.sizes();
        let out_size: Vec<i32> = (0..expected_sizes.size())
            .map(|i| *expected_sizes.at(i))
            .collect();
        let out = tf.ones_default(out_size);

        let mut ctx = context();
        let ret = index_select_out(&mut ctx, x, dim, index, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, *expected);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    // also verifies check_index_select_args (valid path) and
    // get_index_select_out_target_size (output shape [index.numel, ...] pins the
    // per-dim size computation).
    // [spec:et:sem:index-util.torch.executor.check-index-select-args-fn/test]
    // [spec:et:sem:index-util.torch.executor.get-index-select-out-target-size-fn/test]
    #[test]
    fn op_index_select_out_test_select_front_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let out_size = vec![1, 3, 4];
        let index = tfl.make_default(vec![1], vec![0]);
        let expected = tf.make_default(
            out_size,
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );

        run_test_cases(&x, 0, &index, &expected);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_select_middle_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let out_size = vec![2, 2, 4];
        let index = tfl.make_default(vec![2], vec![0, 2]);
        let expected = tf.make_default(
            out_size,
            vec![
                1., 2., 3., 4., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        run_test_cases(&x, 1, &index, &expected);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_select_end_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let out_size = vec![2, 3, 2];
        let index = tfl.make_default(vec![2], vec![0, 2]);
        let expected = tf.make_default(
            out_size,
            vec![
                1., 3., 5., 7., 9., 11., // [0, :, :]
                -1., -3., -5., -7., -9., -11., // [1, :, :]
            ],
        );
        run_test_cases(&x, 2, &index, &expected);
    }

    // A generic smoke test that works for any dtype that supports ones() and zeros().
    // ET_FORALL_REAL_TYPES_AND(Bool).
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_non_empty_input_empty_output_with_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(vec![10], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let index = tfl.make_default(vec![1], vec![5]);

        // Make an empty-size out tensor and demonstrate that it has data.
        let out = tf.make_default(vec![], vec![0]);
        assert_eq!(out.numel(), 1);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 0, &index, &out));
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_empty_input_empty_output_with_matching_dim_supported() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let index = tfl.make_default(vec![1], vec![3]);

        // Using empty tensors as input.
        let x = tf.make_default(vec![3, 0, 10, 3], vec![]);
        assert_eq!(x.numel(), 0);

        // Output whose shape is appropriate for selecting along dim(2).
        let out = tf.make_default(vec![3, 0, 1, 3], vec![]);
        assert_eq!(out.numel(), 0);

        let mut ctx = context();
        let ret = index_select_out(&mut ctx, &x, 2, &index, &out);
        assert_eq!(ret.numel(), 0);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);
        let index = tfl.make_default(vec![1], vec![0]);

        let invalid_dims: [i64; 6] = [3, 4, 5, -4, -5, -6];
        for &dim in invalid_dims.iter() {
            let mut ctx = context();
            et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, dim, &index, &out));
        }
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();

        let x = tf_int.zeros_default(vec![1, 2, 2]);

        // Size is compatible to the output, but a mismatched dtype.
        let out = tf_float.ones_default(vec![1, 2, 2]);
        let index = tf_long.make_default(vec![1], vec![0]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 0, &index, &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_out_match_numel_lack_dim_at_end_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![1, 2, 2, 1]);
        let index = tfl.make_default(vec![1], vec![0]);

        // Out shares the same dtype and numel as the expected output, but a
        // mismatched size (out.dim() should always equal to x.dim()).
        let out = tf.ones_default(vec![1, 2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 0, &index, &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_out_match_numel_extra_dim_at_front_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![2, 2]);
        let index = tfl.make_default(vec![1], vec![0]);

        // Out shares the same dtype as the expected output, but a mismatched size.
        let out = tf.ones_default(vec![1, 1, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 0, &index, &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_out_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let index = tfl.make_default(vec![1], vec![3]);

        // Should be {2, 4, 1, 5} to match the x when calling index_select() with dim 2.
        let out = tf.zeros_default(vec![2, 4, 7]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 2, &index, &out));
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_index_with_invalid_dtype_dies() {
        let tf = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        let index = tff.make_default(vec![1], vec![3.0]);

        let out = tf.zeros_default(vec![2, 1, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 1, &index, &out));
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_index_with_invalid_dim_dies() {
        let tf = TensorFactory::<i32>::new();
        let tfl = TensorFactory::<i64>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);
        // 2-D Tensor, will error out.
        let index = tfl.make_default(vec![1, 1], vec![3]);

        let out = tf.zeros_default(vec![2, 1, 7, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, index_select_out(&mut ctx, &x, 1, &index, &out));
    }

    // PORT-NOTE: C++ gates this on `#if !defined(USE_ATEN_LIB)`; the port is the
    // non-ATen build, so it runs unconditionally.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_upper_bound_out_tensor() {
        let tf = TensorFactory::<f64>::new();
        let tfl = TensorFactory::<i64>::new();
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., // [0, :, :]
                -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12., // [1, :, :]
            ],
        );

        let out_size = vec![1, 3, 4];
        let out = tf.zeros(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let index = tfl.make_default(vec![1], vec![0]);
        let expected = tf.make_default(
            out_size,
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );

        let mut ctx = context();
        let ret = index_select_out(&mut ctx, &x, 0, &index, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(*ret, expected);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    fn op_index_select_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `DynamicShapeUnbound` uses DYNAMIC_UNBOUND; the non-ATen
    // portable kernel does not support unbound resize, so this is `#[ignore]`d to
    // match the effective skip on the portable build.
    // [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn/test]
    #[test]
    #[ignore = "Dynamic shape unbound not supported"]
    fn op_index_select_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
