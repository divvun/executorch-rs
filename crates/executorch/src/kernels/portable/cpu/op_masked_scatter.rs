//! Literal port of kernels/portable/cpu/op_masked_scatter.cpp.

use core::cell::Cell;

use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, resize_to_broadcast_target_size,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_realhbbf16_type, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE: the C++ compute lambda captures `idx` / `src_numel_check` by
// mutable reference (`&idx`, `&src_numel_check`) and post-increments `idx`. The
// ported `apply_binary_elementwise_fn` takes an `Fn` closure, so the mutable
// captured state is wrapped in `Cell` to preserve the interior-mutation
// semantics literally. `src_data[idx++]` becomes read-then-increment on the Cell.

// [spec:et:def:op-masked-scatter.torch.executor.native.masked-scatter-out-fn]
// [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn]
pub fn masked_scatter_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mask: &Tensor,
    src: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let in_type: ScalarType = in_.scalar_type();

    crate::et_kernel_check!(ctx, tensor_is_realhbbf16_type(in_), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        mask.scalar_type() == ScalarType::Bool,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, src.scalar_type() == in_type, InvalidArgument, out);
    crate::et_kernel_check!(ctx, out.scalar_type() == in_type, InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, mask, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(in_, mask, out) == Error::Ok,
        InvalidArgument,
        out
    );

    let idx: Cell<i64> = Cell::new(0);
    let src_numel: i64 = src.numel() as i64;
    let src_numel_check: Cell<bool> = Cell::new(true);

    let name = "masked_scatter.out";

    crate::et_switch_realhbbf16_types!(in_type, ctx, name, CTYPE, {
        let src_data: *const CTYPE = src.const_data_ptr::<CTYPE>();
        apply_binary_elementwise_fn::<CTYPE, bool, CTYPE, _>(
            |val_in: CTYPE, val_mask: bool| -> CTYPE {
                if val_mask && idx.get() >= src_numel {
                    src_numel_check.set(false);
                    return val_in;
                }
                if val_mask {
                    let i = idx.get();
                    idx.set(i + 1);
                    unsafe { *src_data.offset(i as isize) }
                } else {
                    val_in
                }
            },
            in_,
            mask,
            out,
        );
    });

    crate::et_kernel_check_msg!(
        ctx,
        src_numel_check.get(),
        InvalidArgument,
        out,
        "masked_scatter: src doesn't have enough elements"
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_smoke_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let src = tf.make_default(vec![3], vec![10, 20, 30]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 3], vec![10, 2, 3, 20, 5, 30]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_broadcast_input() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![3], vec![1, 2, 3]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let src = tf.make_default(vec![3], vec![10, 20, 30]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 3], vec![10, 2, 3, 20, 2, 30]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_broadcast_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![3], vec![false, true, false]);
        let src = tf.make_default(vec![2], vec![10, 20]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 3], vec![1, 10, 3, 4, 20, 6]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_src_with_more_elements() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let src = tf.make_default(vec![4], vec![10, 20, 30, 40]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 3], vec![10, 2, 3, 20, 5, 30]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_src_with_less_elements_fails() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let src = tf.make_default(vec![2], vec![10, 20]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_empty_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 1], vec![100, 200]);
        let mask = tf_bool.make_default(vec![2, 0], vec![]);
        let src = tf.make_default(vec![4], vec![10, 20, 30, 40]);
        let out = tf.zeros_default(vec![2, 0]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 0], vec![]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_empty_src() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 1], vec![100, 200]);
        let mask = tf_bool.make_default(vec![2, 1], vec![false, false]);
        let src = tf.make_default(vec![0], vec![]);
        let out = tf.zeros_default(vec![2, 1]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 1], vec![100, 200]));
    }

    // [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn/test]
    #[test]
    fn op_masked_scatter_out_test_empty_mask_and_src() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![2, 1], vec![100, 200]);
        let mask = tf_bool.make_default(vec![0], vec![]);
        let src = tf.make_default(vec![0], vec![]);
        let out = tf.zeros_default(vec![2, 0]);

        let mut ctx = context();
        masked_scatter_out(&mut ctx, &in_, &mask, &src, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 0], vec![]));
    }
}
