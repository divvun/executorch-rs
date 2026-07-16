//! Literal port of kernels/portable/cpu/op_lift_fresh_copy.cpp.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the tensor's `*mut TensorImpl`).

// [spec:et:def:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn]
// [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn]
pub fn lift_fresh_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
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
                in_.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
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
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.ones_default(vec![2, 4]);
        let out = tf.zeros_default(vec![2, 4]);

        let mut ctx = context();
        lift_fresh_copy_out(&mut ctx, &self_, &out);
        assert_tensor_eq!(self_, out);

        let self_empty = tf.make_default(vec![], vec![T::one()]);
        let out_empty = tf.make_default(vec![], vec![T::zero()]);

        lift_fresh_copy_out(&mut ctx, &self_empty, &out_empty);
        assert_tensor_eq!(self_empty, out_empty);
    }

    fn test_empty_input<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 0, 1, 2], vec![]);
        let out = tf.zeros_default(vec![3, 0, 1, 2]);
        let mut ctx = context();
        lift_fresh_copy_out(&mut ctx, &self_, &out);
        assert_tensor_eq!(self_, out);
    }

    // ET_FORALL_REAL_TYPES_AND(Bool): Byte, Char, Short, Int, Long, Float, Double,
    // Bool.
    // regular test for lift_fresh_copy.out
    // [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn/test]
    #[test]
    fn op_lift_fresh_copy_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn/test]
    #[test]
    fn op_lift_fresh_copy_test_empty_input_supported() {
        test_empty_input::<u8>();
        test_empty_input::<i8>();
        test_empty_input::<i16>();
        test_empty_input::<i32>();
        test_empty_input::<i64>();
        test_empty_input::<f32>();
        test_empty_input::<f64>();
        test_empty_input::<bool>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn/test]
    #[test]
    fn op_lift_fresh_copy_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 2, 1, 1]);
        let mut ctx = context();
        lift_fresh_copy_out(&mut ctx, &self_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn/test]
    #[test]
    fn op_lift_fresh_copy_test_mismatched_d_type_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let self_ = tf_in.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);
        let mut ctx = context();
        lift_fresh_copy_out(&mut ctx, &self_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
