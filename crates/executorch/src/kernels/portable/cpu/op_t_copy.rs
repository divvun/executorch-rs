//! Literal port of kernels/portable/cpu/op_t_copy.cpp.

use crate::kernels::portable::cpu::util::transpose_util::{
    check_t_copy_args, get_transpose_out_target_size, transpose_tensors,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ untyped `memcpy(out_data, in_data, in.nbytes())` (guarded by
// `in.numel() > 0`) becomes `core::ptr::copy_nonoverlapping` over `in.nbytes()`
// bytes; the `ET_SWITCH_ALL_TYPES` around it is kept (it selects CTYPE only to
// derive typed data pointers in the C++, but the copy is byte-for-byte, so the
// switch is elided to a single untyped copy — the CTYPE was unused apart from the
// pointer casts). Unresolved: whether the ALL-types switch materially matters;
// the copy is dtype-agnostic.

// [spec:et:def:op-t-copy.torch.executor.native.t-copy-out-fn]
// [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn]
pub fn t_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    crate::et_kernel_check!(ctx, check_t_copy_args(in_, out), InvalidArgument, out);

    let in_type = in_.scalar_type();

    if in_.dim() < 2 {
        // Resize for dynamic shape
        crate::et_kernel_check!(
            ctx,
            resize_tensor(out, in_.sizes()) == Error::Ok,
            InvalidArgument,
            out
        );

        if in_.numel() > 0 {
            crate::et_switch_all_types!(in_type, ctx, "t_copy.out", CTYPE, {
                let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
                let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        in_data as *const u8,
                        out_data as *mut u8,
                        in_.nbytes(),
                    );
                }
            });
        }

        return out;
    }

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
        get_transpose_out_target_size(
            in_,
            1,
            0,
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

    crate::et_switch_all_types!(in_type, ctx, "t_copy.out", CTYPE, {
        transpose_tensors::<CTYPE>(in_, 1, 0, out);
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

    fn op_t_copy_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        t_copy_out(ctx, self_, out)
    }

    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_1d_transpose() {
        let tf = TensorFactory::<i32>::new();

        let t_in = tf.make_default(vec![4], vec![1, 2, 3, 4]);
        let t_out = tf.make_default(vec![4], vec![0, 0, 0, 0]);

        let mut ctx = context();
        op_t_copy_out(&mut ctx, &t_in, &t_out);
        assert_tensor_eq!(t_in, t_out);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; the ported runtime is never ATen,
    // so the mismatched-shape failure path is always exercised.
    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_1d_transpose_mismatch_shape_die() {
        let tf = TensorFactory::<i32>::new();

        let t_in = tf.make_default(vec![4], vec![1, 2, 3, 4]);
        let t_out = tf.make_default(vec![2], vec![0, 0]);

        let mut ctx = context();
        op_t_copy_out(&mut ctx, &t_in, &t_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_2d_transpose() {
        let tf = TensorFactory::<i32>::new();

        let t_in = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.make_default(vec![3, 2], vec![0, 0, 0, 0, 0, 0]);
        let t_expected = tf.make_default(vec![3, 2], vec![1, 4, 2, 5, 3, 6]);

        let mut ctx = context();
        op_t_copy_out(&mut ctx, &t_in, &t_out);
        assert_tensor_eq!(t_out, t_expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_2d_transpose_mismatch_shape_die() {
        let tf = TensorFactory::<i32>::new();

        let t_in = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.make_default(vec![2, 2], vec![0, 0, 0, 0]);

        let mut ctx = context();
        op_t_copy_out(&mut ctx, &t_in, &t_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    // also verifies check_t_copy_args rejects rank > 2 (3-D input).
    // [spec:et:sem:transpose-util.torch.executor.check-t-copy-args-fn/test]
    #[test]
    fn op_t_copy_test_3d_transpose_die() {
        let tf = TensorFactory::<i32>::new();

        let t_in = tf.make_default(vec![2, 3, 1], vec![1, 2, 3, 4, 5, 6]);
        let t_out = tf.make_default(vec![3, 2, 1], vec![0, 0, 0, 0, 0, 0]);

        let mut ctx = context();
        op_t_copy_out(&mut ctx, &t_in, &t_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.08847743272781372,
                0.30742281675338745,
                0.7682217955589294,
                0.13203048706054688,
                0.6340786814689636,
            ],
        );

        let out = tf.zeros(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        op_t_copy_out(&mut ctx, &x, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`; in the non-ATen (portable)
    // build `output_resize` defaults to false (kernels/test/supported_features.yaml),
    // so this test is skipped. Ported as a no-op body to preserve the case.
    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_dynamic_shape_upper_bound_larger_than_expected() {
        // ET_SKIP_IF(!output_resize, ...) -> skipped in the non-ATen build.
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`; `output_resize` defaults
    // to false in the non-ATen build, so this test is skipped.
    // [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn/test]
    #[test]
    fn op_t_copy_test_dynamic_shape_unbound() {
        // ET_SKIP_IF(!output_resize, ...) -> skipped in the non-ATen build.
    }
}
