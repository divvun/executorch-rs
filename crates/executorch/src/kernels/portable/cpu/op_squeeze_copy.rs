//! Literal port of kernels/portable/cpu/op_squeeze_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_squeeze_copy_dim_args, check_squeeze_copy_dims_args,
    get_squeeze_copy_dim_out_target_size, get_squeeze_copy_dims_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, nonzero_dim, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ untyped `memcpy(out.mutable_data_ptr(),
// in.const_data_ptr(), in.nbytes())` becomes `core::ptr::copy_nonoverlapping`
// over `in.nbytes()` bytes (mirroring op_alias_copy).

// [spec:et:def:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn]
// [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn]
pub fn squeeze_copy_dim_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    crate::et_kernel_check!(
        ctx,
        check_squeeze_copy_dim_args(in_, dim, out),
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

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_squeeze_copy_dim_out_target_size(
            in_,
            dim,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if in_.nbytes() > 0 {
        // Note that this check is important. It's valid for a tensor with numel 0
        // to have a null data pointer, but in some environments it's invalid to
        // pass a null pointer to memcpy() even when the size is zero.
        unsafe {
            core::ptr::copy_nonoverlapping(
                in_.const_data_ptr_typed() as *const u8,
                out.mutable_data_ptr_typed() as *mut u8,
                in_.nbytes(),
            );
        }
    }
    out
}

// [spec:et:def:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn]
// [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn]
#[executorch_macros::et_kernel("aten::squeeze_copy.dims_out")]
pub fn squeeze_copy_dims_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dims: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    crate::et_kernel_check!(
        ctx,
        check_squeeze_copy_dims_args(in_, dims, out),
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

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_squeeze_copy_dims_out_target_size(
            in_,
            dims,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if in_.nbytes() > 0 {
        // Note that this check is important. It's valid for a tensor with numel 0
        // to have a null data pointer, but in some environments it's invalid to
        // pass a null pointer to memcpy() even when the size is zero.
        unsafe {
            core::ptr::copy_nonoverlapping(
                in_.const_data_ptr_typed() as *const u8,
                out.mutable_data_ptr_typed() as *mut u8,
                in_.nbytes(),
            );
        }
    }
    out
}

#[cfg(test)]
mod squeeze_dim_tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_data_eq, assert_tensor_eq};

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

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_d_types_mismatch_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_d = TensorFactory::<f64>::new();
        let t_in = tf_int.ones_default(vec![2]);
        let t_out = tf_d.ones_default(vec![2]);
        let dim: i64 = 0;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_0d_tensor_squeeze() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![]);
        let t_out = tf.zeros_default(vec![]);
        let t_expected = tf.ones_default(vec![]);
        let dim: i64 = 0;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_0d_tensor_squeeze_invalid_dim1_dies() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![]);
        let t_out = tf.ones_default(vec![]);
        let dim: i64 = 1;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_1d_tensor_squeeze_to0d() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![1]);
        let t_out = tf.make_default(vec![], vec![99]);
        let t_expected = tf.make_default(vec![], vec![1]);
        let dim: i64 = 0;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_unchange() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.make_default(vec![2, 1], vec![4, 3]);
        let t_expected = tf.ones_default(vec![2, 1]);
        let dim: i64 = 0;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    // also verifies check_squeeze_copy_dim_args (dtype/dim gate) and
    // get_squeeze_copy_dim_out_target_size (size==1 at dim removes it, out {2})
    // [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dim-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dim-out-target-size-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_to1d() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.make_default(vec![2], vec![4, 3]);
        let t_expected = tf.ones_default(vec![2]);
        let dim: i64 = 1;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // PORT-NOTE: C++ `#ifndef USE_ATEN_LIB` — active in the portable (non-ATen)
    // build; ported here.
    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_downward_dim_resize_out() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.zeros(vec![4, 1], TensorShapeDynamism::DYNAMIC_BOUND);
        let t_expected = tf.ones_default(vec![2, 1]);
        let dim: i64 = 0;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_upward_dim_resize_out_die() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_BOUND);
        let dim: i64 = 0;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_remove_a_dim_resize_out_die() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.zeros(vec![2, 1, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let dim: i64 = 0;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_2d_tensor_squeeze_add_dims_resize_out_die() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.ones_default(vec![2, 1]);
        let t_out = tf.zeros(vec![2], TensorShapeDynamism::DYNAMIC_BOUND);
        let dim: i64 = 0;

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_tensor_squeeze() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.make_default(vec![3, 1, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.zeros_default(vec![3, 2, 1]);
        let t_expected = tf.make_default(vec![3, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        let dim: i64 = 1;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_tensor_squeeze_negative_dim() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.make_default(vec![3, 1, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.zeros_default(vec![3, 2, 1]);
        let t_expected = tf.make_default(vec![3, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        let dim: i64 = -3;

        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out);
        assert_tensor_eq!(t_expected, t_out);
        assert_tensor_data_eq!(t_expected, t_out);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_tensor_squeeze_invaid_dim() {
        let tf = TensorFactory::<i32>::new();
        let t_in = tf.make_default(vec![3, 1, 2, 1], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.zeros_default(vec![3, 2, 1]);
        let invalid_dims: [i64; 2] = [t_in.dim() as i64, -(t_in.dim() as i64) - 1];

        for dim in invalid_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(ctx, squeeze_copy_dim_out(&mut ctx, &t_in, dim, &t_out));
        }
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 1, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
            ],
        );
        let expected = tf.make_default(
            vec![2, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
            ],
        );

        let out = tf.zeros(vec![2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        squeeze_copy_dim_out(&mut ctx, &x, 1, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)` skips this for the portable
    // kernel (output_resize default false). Mirrored as an early skip.
    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)` skips this for the portable
    // kernel. Mirrored as an early skip.
    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn/test]
    #[test]
    fn op_squeeze_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }
}

#[cfg(test)]
mod squeeze_dims_tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn dims(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn/test]
    // also verifies check_squeeze_copy_dims_args (dtype/dim/no-duplicate gate) and
    // get_squeeze_copy_dims_out_target_size (dims [0,2] both size-1 removed,
    // out {2,5})
    // [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dims-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dims-out-target-size-fn/test]
    #[test]
    fn op_squeeze_copy_dims_out_test_sanity_test4d() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![1, 2, 1, 5],
            vec![
                -26.5, 5.75, 95.75, 92.625, -97.25, 65.5, -92.25, -67.625, 54.75, 27.125,
            ],
        );
        let dim_vec: [i64; 2] = [0, 2];
        let out = tf_float.zeros_default(vec![2, 5]);
        let out_expected = tf_float.make_default(
            vec![2, 5],
            vec![
                -26.5, 5.75, 95.75, 92.625, -97.25, 65.5, -92.25, -67.625, 54.75, 27.125,
            ],
        );

        let mut ctx = context();
        squeeze_copy_dims_out(&mut ctx, &self_, dims(&dim_vec), &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn/test]
    #[test]
    fn op_squeeze_copy_dims_out_test_sanity_check5d() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![1, 2, 1, 5, 4],
            vec![
                -73.5, -67.625, -54.375, 51.625, -11.125, -28.625, -40.75, 45.625, 84.375, 65.625,
                95.125, -47.125, -21.25, 32.25, -86.125, 55.875, -62.25, 47.125, -71.875, 43.0,
                47.875, -73.375, 97.75, 69.25, 64.125, -59.875, 59.75, -52.25, 59.5, 44.875,
                -51.25, 20.875, -67.0, 32.5, -26.625, 83.75, 45.5, 85.5, -92.875, 60.0,
            ],
        );
        let dim_vec: [i64; 4] = [0, 3, 2, 1];
        let out = tf_float.zeros_default(vec![2, 5, 4]);
        let out_expected = tf_float.make_default(
            vec![2, 5, 4],
            vec![
                -73.5, -67.625, -54.375, 51.625, -11.125, -28.625, -40.75, 45.625, 84.375, 65.625,
                95.125, -47.125, -21.25, 32.25, -86.125, 55.875, -62.25, 47.125, -71.875, 43.0,
                47.875, -73.375, 97.75, 69.25, 64.125, -59.875, 59.75, -52.25, 59.5, 44.875,
                -51.25, 20.875, -67.0, 32.5, -26.625, 83.75, 45.5, 85.5, -92.875, 60.0,
            ],
        );

        let mut ctx = context();
        squeeze_copy_dims_out(&mut ctx, &self_, dims(&dim_vec), &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn/test]
    #[test]
    fn op_squeeze_copy_dims_out_test_sanity_check5d_unchanged() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![1, 2, 1, 5, 4],
            vec![
                -0.375, -40.125, 5.75, 21.25, -34.875, -19.375, 15.75, -60.75, -41.75, 53.125,
                -76.0, -64.25, -84.5, -37.25, -39.125, 22.875, -69.0, 30.25, -21.25, 85.5, 8.875,
                41.625, 12.375, -1.125, -14.875, 78.5, 43.0, -78.625, -58.625, -58.375, 47.5,
                -67.375, -82.375, 35.0, 83.25, 49.625, -9.875, -46.75, 17.875, -68.375,
            ],
        );
        let dim_vec: [i64; 3] = [1, 4, 3];
        let out = tf_float.zeros_default(vec![1, 2, 1, 5, 4]);
        let out_expected = tf_float.make_default(
            vec![1, 2, 1, 5, 4],
            vec![
                -0.375, -40.125, 5.75, 21.25, -34.875, -19.375, 15.75, -60.75, -41.75, 53.125,
                -76.0, -64.25, -84.5, -37.25, -39.125, 22.875, -69.0, 30.25, -21.25, 85.5, 8.875,
                41.625, 12.375, -1.125, -14.875, 78.5, 43.0, -78.625, -58.625, -58.375, 47.5,
                -67.375, -82.375, 35.0, 83.25, 49.625, -9.875, -46.75, 17.875, -68.375,
            ],
        );

        let mut ctx = context();
        squeeze_copy_dims_out(&mut ctx, &self_, dims(&dim_vec), &out);
        assert_tensor_close!(out, out_expected);
    }
}
