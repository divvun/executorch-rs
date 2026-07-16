//! Literal port of kernels/portable/cpu/op_transpose_copy.cpp.

use crate::kernels::portable::cpu::util::transpose_util::{
    check_transpose_copy_args, get_transpose_out_target_size, transpose_tensors,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, nonzero_dim, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).

/// Swaps dimension 'dim0' of 'a' with 'dim1', and copying
/// that mutation into `out` in a manner such that the data is densely packed
/// and is_contiguous() would return true (stride dim[size-1] = 1).
///
/// transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out)
// [spec:et:def:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn]
// [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn]
pub fn transpose_copy_int_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim0: i64,
    mut dim1: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_transpose_copy_args(in_, dim0, dim1, out),
        InvalidArgument,
        out
    );

    if dim0 < 0 {
        dim0 += nonzero_dim(in_) as i64;
    }
    if dim1 < 0 {
        dim1 += nonzero_dim(in_) as i64;
    }

    let mut expected_out_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_transpose_out_target_size(
            in_,
            dim0 as SizesType,
            dim1 as SizesType,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }

    // Resize for dynamic shape
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_switch_all_types!(in_.scalar_type(), ctx, "transpose_copy.int_out", CTYPE, {
        transpose_tensors::<CTYPE>(in_, dim0, dim1, out);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_transpose_copy_int_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        dim0: i64,
        dim1: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        transpose_copy_int_out(ctx, self_, dim0, dim1, out)
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_two_d_transpose() {
        let tf = TensorFactory::<i32>::new();

        let t_int = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
        let out = tf.zeros_default(vec![3, 2]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &t_int, 1, 0, &out);
        assert_tensor_eq!(out, tf.make_default(vec![3, 2], vec![0, 3, 1, 4, 2, 5]));
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_two_d_negative_indices() {
        let tf = TensorFactory::<i32>::new();

        let t_int = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
        let out = tf.zeros_default(vec![3, 2]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &t_int, -1, -2, &out);
        assert_tensor_eq!(out, tf.make_default(vec![3, 2], vec![0, 3, 1, 4, 2, 5]));
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_transpose_no_datachange() {
        let tf = TensorFactory::<i32>::new();

        let t_int = tf.make_default(vec![2, 1, 3], vec![0, 1, 2, 3, 4, 5]);
        let out = tf.zeros_default(vec![2, 3, 1]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &t_int, 1, 2, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 3, 1], vec![0, 1, 2, 3, 4, 5]));
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    // also verifies transpose_tensors (swapped sizes/strides drive the element
    // gather), increment_index_and_offset (the N-D index/offset stepping across
    // three non-unit dims), and get_transpose_out_target_size (out shape [3,2,2]
    // = in [2,2,3] with dims 0,2 swapped).
    // [spec:et:sem:transpose-util.torch.executor.transpose-tensors-fn/test]
    // [spec:et:sem:transpose-util.torch.executor.increment-index-and-offset-fn/test]
    // [spec:et:sem:transpose-util.torch.executor.get-transpose-out-target-size-fn/test]
    #[test]
    fn op_transpose_int_copy_test_three_d_transpose() {
        let tf = TensorFactory::<i32>::new();

        let t_int = tf.make_default(vec![2, 2, 3], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let out = tf.zeros_default(vec![3, 2, 2]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &t_int, 0, 2, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![3, 2, 2], vec![0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])
        );
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    // also verifies check_transpose_copy_args rejects an out-of-range dim.
    // [spec:et:sem:transpose-util.torch.executor.check-transpose-copy-args-fn/test]
    #[test]
    fn op_transpose_int_copy_test_out_of_bound_dim_dies() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![2, 3]);
        let out = tf.ones_default(vec![3, 2]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &a, 0, -3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port, so the
    // mismatched-dim failure path is always exercised.
    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_mismatched_dim_dies() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![4, 2, 3]);
        let out = tf.ones_default(vec![2, 2]);

        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &a, 0, 1, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![2, 2, 3], vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6]);
        let expected = tf.make_default(vec![3, 2, 2], vec![4, 7, 0, 3, 9, 3, 3, 1, 3, 7, 9, 6]);

        let out = tf.zeros(vec![3, 2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        op_transpose_copy_int_out(&mut ctx, &x, 0, 2, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`; `output_resize` defaults
    // to false in the non-ATen build, so this test is skipped.
    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_dynamic_shape_upper_bound_larger_than_expected() {
        // ET_SKIP_IF(!output_resize, ...) -> skipped in the non-ATen build.
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`; skipped in the non-ATen
    // build (see above).
    // [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn/test]
    #[test]
    fn op_transpose_int_copy_test_dynamic_shape_unbound() {
        // ET_SKIP_IF(!output_resize, ...) -> skipped in the non-ATen build.
    }
}
