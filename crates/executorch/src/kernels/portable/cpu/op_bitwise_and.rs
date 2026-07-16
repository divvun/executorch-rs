//! Literal port of kernels/portable/cpu/op_bitwise_and.cpp.

use crate::kernels::portable::cpu::pattern::bitwise_op::{
    BitwiseAnd, bitwise_scalar_out, bitwise_tensor_out,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `std::bit_and` is the `BitOp` template argument; the ported
// `bitwise_tensor_out`/`bitwise_scalar_out` are generic over the `BitwiseOp`
// marker `BitwiseAnd`. The `op_name` template parameter is dropped by the port
// (see bitwise_op.rs).

// [spec:et:def:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn]
// [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn]
#[executorch_macros::et_kernel("aten::bitwise_and.Tensor_out")]
pub fn bitwise_and_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    bitwise_tensor_out::<BitwiseAnd>(ctx, a, b, out)
}

// [spec:et:def:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn]
// [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn]
pub fn bitwise_and_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    bitwise_scalar_out::<BitwiseAnd>(ctx, a, b, out)
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

    // [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn/test]
    #[test]
    fn op_bitwise_and_tensor_out_test_smoke_test_int() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 5]);
        let b = tf.make_default(vec![2, 2], vec![1, 6, 2, 3]);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_and_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![0, 2, 2, 1]));
    }

    // [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn/test]
    #[test]
    fn op_bitwise_and_tensor_out_test_smoke_test_bool() {
        let tf = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![2, 2], vec![true, false, true, false]);
        let b = tf.make_default(vec![2, 2], vec![true, true, false, false]);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_and_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 2], vec![true, false, false, false])
        );
    }

    // [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn/test]
    #[test]
    fn op_bitwise_and_tensor_out_test_smoke_test_mixed() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.make_default(vec![2, 2], vec![2, 3, 2, 5]);
        let b = tf_bool.make_default(vec![2, 2], vec![true, true, false, false]);

        let out = tf_int.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_and_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf_int.make_default(vec![2, 2], vec![0, 1, 0, 0]));
    }

    // [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn/test]
    #[test]
    fn op_bitwise_and_scalar_out_test_smoke_test_int() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 5]);
        let b = Scalar::from_i64(6);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_and_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![2, 2, 2, 4]));
    }

    // [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn/test]
    #[test]
    fn op_bitwise_and_scalar_out_test_smoke_test_bool() {
        let tf = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![2, 2], vec![true, false, true, false]);
        let b = Scalar::from_bool(true);

        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_and_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 2], vec![true, false, true, false])
        );
    }
}
