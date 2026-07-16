//! Literal port of kernels/portable/cpu/op_alias_copy.cpp.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-alias-copy.torch.executor.native.alias-copy-out-fn]
// [spec:et:sem:op-alias-copy.torch.executor.native.alias-copy-out-fn]
pub fn alias_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
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
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // [spec:et:sem:op-alias-copy.torch.executor.native.alias-copy-out-fn/test]
    #[test]
    fn op_alias_copy_test_smoke_test() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![2, 2], vec![2.0, 3.0, 2.0, 5.0]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        alias_copy_out(&mut ctx, &a, &out);
        assert_tensor_eq!(a, out);
    }
}
