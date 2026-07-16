//! Literal port of kernels/portable/cpu/op_detach_copy.cpp.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

/// Copy the tener `self` to `out`, assume `self` and `out` have same type and
/// shape
// [spec:et:def:op-detach-copy.torch.executor.native.detach-copy-out-fn]
// [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn]
pub fn detach_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_shape_and_dtype2(self_, out),
        InvalidArgument,
        out
    );

    if self_.nbytes() > 0 {
        // Note that this check is important. It's valid for a tensor with numel 0
        // to have a null data pointer, but in some environments it's invalid to
        // pass a null pointer to memcpy() even when the size is zero.
        unsafe {
            core::ptr::copy_nonoverlapping(
                self_.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                self_.nbytes(),
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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_detach_copy_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        detach_copy_out(&mut ctx, self_, out)
    }

    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_num {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_num!(u8, i8, i16, i32, i64, f32, f64);

    // Common testing for detach_copy operator over the integral/real ctypes.
    fn test_detach_copy_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![2, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![
                T::from_i32(1),
                T::from_i32(2),
                T::from_i32(3),
                T::from_i32(4),
            ],
        );
        let out = tf.zeros_default(sizes.clone());

        // Valid input should give the expected output
        op_detach_copy_out(&in_, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![
                    T::from_i32(1),
                    T::from_i32(2),
                    T::from_i32(3),
                    T::from_i32(4)
                ]
            )
        );
    }

    // template <> ScalarType::Bool
    fn test_detach_copy_out_bool() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_detach_copy_out(
            &tf.make_default(sizes.clone(), vec![true, false, true, false]),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, vec![true, false, true, false]));
    }

    // template <> ScalarType::Float
    fn test_detach_copy_out_float() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_detach_copy_out(
            &tf.make_default(
                sizes.clone(),
                vec![3.14, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            ),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![3.14, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]
            )
        );
    }

    fn test_detach_copy_out_invalid_shape<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let in_sizes = vec![2, 2];
        let out_sizes = vec![4];

        let in_ = tf.ones_default(in_sizes);
        let out = tf.zeros_default(out_sizes);

        let mut ctx = context();
        detach_copy_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_all_scalar_input_output_support() {
        // ET_FORALL_REAL_TYPES_AND(Bool)
        test_detach_copy_out::<u8>();
        test_detach_copy_out::<i8>();
        test_detach_copy_out::<i16>();
        test_detach_copy_out::<i32>();
        test_detach_copy_out::<i64>();
        test_detach_copy_out_float();
        test_detach_copy_out::<f64>();
        test_detach_copy_out_bool();
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure runs.
    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_mismatched_shapes_dies() {
        test_detach_copy_out_invalid_shape::<u8>();
        test_detach_copy_out_invalid_shape::<i8>();
        test_detach_copy_out_invalid_shape::<i16>();
        test_detach_copy_out_invalid_shape::<i32>();
        test_detach_copy_out_invalid_shape::<i64>();
        test_detach_copy_out_invalid_shape::<f32>();
        test_detach_copy_out_invalid_shape::<f64>();
        // Bool
        let tf = TensorFactory::<bool>::new();
        let in_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![4]);
        let mut ctx = context();
        detach_copy_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_mismatched_input_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();
        let sizes = vec![2, 2];
        let in_ = tf_byte.ones_default(sizes.clone());
        let out = tf_char.ones_default(sizes);
        let mut ctx = context();
        detach_copy_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let out = tf.zeros_default(vec![10, 10]);
        op_detach_copy_out(&x, &out);
        assert!(
            crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &out,
                &expected_result,
                crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            )
        );
    }

    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_detach_copy_out(&x, &out);
        assert!(
            crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &out,
                &expected_result,
                crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            )
        );
    }

    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    fn op_detach_copy_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_detach_copy_out(&x, &out);
        assert!(
            crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &out,
                &expected_result,
                crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            )
        );
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_detach_copy_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.18719732761383057,
                0.03402292728424072,
                0.944246232509613,
                0.8801798820495605,
                0.0012360215187072754,
                0.5935860276222229,
            ],
        );
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_detach_copy_out(&x, &out);
        assert!(
            crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close(
                &out,
                &expected_result,
                crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                None
            )
        );
    }
}
