//! Literal port of kernels/portable/cpu/op_bitwise_left_shift.cpp.

use crate::kernels::portable::cpu::pattern::bitwise_op::{
    BitLshift, bitwise_scalar_out, bitwise_tensor_out,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `internal::bit_lshift` is the `BitOp` template argument; ported
// as the `BitLshift` marker.

// [spec:et:def:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn]
// [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn]
pub fn bitwise_left_shift_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    bitwise_tensor_out::<BitLshift>(ctx, a, b, out)
}

// [spec:et:def:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn]
// [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn]
pub fn bitwise_left_shift_tensor_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    bitwise_scalar_out::<BitLshift>(ctx, a, b, out)
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

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_tensor_out_test_smoke_test_int() {
        let tf = TensorFactory::<i32>::new();

        // Test basic left shift: [1, 2, 4, 8] << [1, 2, 1, 2] = [2, 8, 8, 32]
        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let b = tf.make_default(vec![2, 2], vec![1, 2, 1, 2]);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![2, 8, 8, 32]));
    }

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_tensor_out_test_smoke_test_byte() {
        let tf = TensorFactory::<u8>::new();

        // Test with byte values: [1, 5, 10, 15] << [0, 1, 2, 3] = [1, 10, 40, 120]
        let a = tf.make_default(vec![2, 2], vec![1, 5, 10, 15]);
        let b = tf.make_default(vec![2, 2], vec![0, 1, 2, 3]);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![1, 10, 40, 120]));
    }

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_tensor_out_test_zero_shift() {
        let tf = TensorFactory::<i32>::new();

        // Shifting by 0 should return the original value
        let a = tf.make_default(vec![2, 2], vec![5, 10, 15, 20]);
        let b = tf.zeros_default(vec![2, 2]);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![5, 10, 15, 20]));
    }

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_scalar_out_test_smoke_test_int() {
        let tf = TensorFactory::<i32>::new();

        // Test shifting by scalar: [1, 2, 3, 4] << 2 = [4, 8, 12, 16]
        let a = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let b = Scalar::from_i64(2);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![4, 8, 12, 16]));
    }

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_scalar_out_test_shift_by_one() {
        let tf = TensorFactory::<i32>::new();

        // Shifting by 1 should double the value
        let a = tf.make_default(vec![2, 2], vec![1, 5, 10, 100]);
        let b = Scalar::from_i64(1);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![2, 10, 20, 200]));
    }

    // [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn/test]
    #[test]
    fn op_bitwise_left_shift_scalar_out_test_shift_by_zero() {
        let tf = TensorFactory::<i32>::new();

        // Shifting by 0 should return the original value
        let a = tf.make_default(vec![2, 2], vec![7, 14, 21, 28]);
        let b = Scalar::from_i64(0);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_left_shift_tensor_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![7, 14, 21, 28]));
    }
}
