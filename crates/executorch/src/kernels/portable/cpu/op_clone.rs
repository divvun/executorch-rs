//! Literal port of kernels/portable/cpu/op_clone.cpp.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out)
// -> Tensor(a!)
// [spec:et:def:op-clone.torch.executor.native.clone-out-fn]
// [spec:et:sem:op-clone.torch.executor.native.clone-out-fn]
#[executorch_macros::et_kernel("aten::clone.out")]
pub fn clone_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    self_: &Tensor,
    memory_format: Option<MemoryFormat>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        context,
        resize_tensor_same_type(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // The input and out shall share same dtype and size
    crate::et_kernel_check!(
        context,
        tensors_have_same_shape_and_dtype2(self_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        context,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    // Right now we only focus on contiguous memory, memory_format shall always
    // either a nullopt or exec::aten::MemoryFormat::Contiguous
    crate::et_kernel_check!(
        context,
        memory_format.is_none() || memory_format.unwrap() == MemoryFormat::Contiguous,
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

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(
            vec![2, 4],
            [2, 3, 2, 4, 1, 5, 1, 6]
                .iter()
                .map(|&x| T::from_i32(x))
                .collect(),
        );
        let out_nullopt = tf.zeros_default(vec![2, 4]);
        let out_contiguous = tf.zeros_default(vec![2, 4]);

        let mut ctx = context();
        let out_nullopt_ret = clone_out(&mut ctx, &input, None, &out_nullopt);
        // PORT-NOTE: `out_nullopt_ret` aliases `out_nullopt`; compared before the
        // second call to mirror the C++ (which binds a Tensor value first).
        assert_tensor_eq!(input, out_nullopt);
        assert_tensor_eq!(input, *out_nullopt_ret);

        let out_contiguous_ret = clone_out(
            &mut ctx,
            &input,
            Some(MemoryFormat::Contiguous),
            &out_contiguous,
        );

        assert_tensor_eq!(input, out_contiguous);
        assert_tensor_eq!(input, *out_contiguous_ret);
    }

    fn test_empty_input<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![3, 0, 1, 2], vec![]);
        let out = tf.zeros_default(vec![3, 0, 1, 2]);
        let mut ctx = context();
        clone_out(&mut ctx, &input, None, &out);
        assert_tensor_eq!(input, out);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_empty_input_supported() {
        test_empty_input::<u8>();
        test_empty_input::<i8>();
        test_empty_input::<i16>();
        test_empty_input::<i32>();
        test_empty_input::<i64>();
        test_empty_input::<f32>();
        test_empty_input::<f64>();
        test_empty_input::<bool>();
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 2, 1, 1]);
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, clone_out(&mut ctx, &input, None, &out));
    }

    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_mismatched_types_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let input = tf_in.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, clone_out(&mut ctx, &input, None, &out));
    }

    // PORT-NOTE: C++ casts `static_cast<MemoryFormat>(55)` — an out-of-range
    // enum value — to exercise the "not Contiguous" branch. Rust cannot hold an
    // invalid enum discriminant safely; `MemoryFormat::Preserve` is the only other
    // valid variant and drives the same `== Contiguous` check failure. C++ also
    // guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_mismatched_memory_format_die() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let input = tf_in.make_default(vec![3, 1, 1, 2], vec![1., 2., 3., 4., 5., 6.]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            clone_out(&mut ctx, &input, Some(MemoryFormat::Preserve), &out)
        );
    }

    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![1.0f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        let _ret = clone_out(&mut ctx, &x, Some(MemoryFormat::Contiguous), &out);
        assert_tensor_close!(out, expected_result);
    }

    fn dyn_inputs(tf: &TensorFactory<f32>) -> (Tensor<'_>, Tensor<'_>) {
        let data = vec![
            0.04876953363418579,
            0.816348671913147,
            0.44230276346206665,
            0.2767965793609619,
            0.8998266458511353,
            0.09595239162445068,
        ];
        let x = tf.make_default(vec![3, 2], data.clone());
        let expected = tf.make_default(vec![3, 2], data);
        (x, expected)
    }

    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected_result) = dyn_inputs(&tf);
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = clone_out(&mut ctx, &x, Some(MemoryFormat::Contiguous), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    fn op_clone_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected_result) = dyn_inputs(&tf);
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = clone_out(&mut ctx, &x, Some(MemoryFormat::Contiguous), &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED in C++ (dynamic shape unbound not supported). Ported +
    // ignored.
    // [spec:et:sem:op-clone.torch.executor.native.clone-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape unbound not supported"]
    fn op_clone_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected_result) = dyn_inputs(&tf);
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        let _ret = clone_out(&mut ctx, &x, Some(MemoryFormat::Contiguous), &out);
        assert_tensor_close!(out, expected_result);
    }
}
