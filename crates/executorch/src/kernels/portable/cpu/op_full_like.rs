//! Literal port of kernels/portable/cpu/op_full_like.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped.

// [spec:et:def:op-full-like.torch.executor.native.full-like-out-fn]
// [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn]
pub fn full_like_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    fill_value: &Scalar,
    memory_format: Option<MemoryFormat>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    if memory_format.is_some() {
        crate::et_kernel_check_msg!(
            ctx,
            memory_format.unwrap() == MemoryFormat::Contiguous
                || memory_format.unwrap() == MemoryFormat::Preserve,
            InvalidArgument,
            out,
            "memory_format must be contiguous"
        );
    }

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let out_type: ScalarType = out.scalar_type();

    let op_name = "full_like.out";

    crate::et_switch_realhbbf16_types!(out_type, ctx, op_name, CTYPE_OUT, {
        let opt_val_casted = check_overflow_scalar_cast::<CTYPE_OUT>(fill_value);
        crate::et_kernel_check!(ctx, opt_val_casted.is_some(), InvalidArgument, out);
        let val_casted = opt_val_casted.unwrap();
        let data_out = out.mutable_data_ptr::<CTYPE_OUT>();
        for i in 0..out.numel() {
            unsafe {
                *data_out.add(i as usize) = val_casted;
            }
        }
    });

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
    use crate::runtime::core::portable_type::{BFloat16, Half};
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

    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_full_like_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![2, 2];
        let in_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());
        let value = Scalar::from_i64(42);
        let memory_format = MemoryFormat::Contiguous;

        let mut ctx = context();
        full_like_out(&mut ctx, &in_, &value, Some(memory_format), &out);
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![T::from_i32(42); 4])
        );

        let value = Scalar::from_i64(1);
        full_like_out(&mut ctx, &in_, &value, Some(memory_format), &out);
        assert_tensor_eq!(out, tf.ones_default(sizes));
    }

    // template<> specialization for Bool.
    fn test_full_like_out_bool() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let in_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());
        let value = Scalar::from_bool(true);
        let memory_format = MemoryFormat::Contiguous;

        let mut ctx = context();
        full_like_out(&mut ctx, &in_, &value, Some(memory_format), &out);
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![true, true, true, true])
        );

        let value = Scalar::from_bool(false);
        full_like_out(&mut ctx, &in_, &value, Some(memory_format), &out);
        assert_tensor_eq!(out, tf.zeros_default(sizes));
    }

    fn test_full_like_out_mismatched_shape<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.zeros_default(vec![2, 2]);
        let out = tf.zeros_default(vec![4, 2]);
        let value = Scalar::from_i64(42);
        // MemoryFormat default-constructs to Contiguous (enum value 0).
        let memory_format = MemoryFormat::Contiguous;

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            full_like_out(&mut ctx, &in_, &value, Some(memory_format), &out)
        );
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![2, 2];
        let in_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, full_like_out(&mut ctx, &in_, &bad_value, None, &out));
    }

    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_all_dtype_output_passes() {
        // ET_FORALL_REALHBBF16_TYPES
        test_full_like_out::<u8>();
        test_full_like_out::<i8>();
        test_full_like_out::<i16>();
        test_full_like_out::<i32>();
        test_full_like_out::<i64>();
        test_full_like_out::<Half>();
        test_full_like_out::<f32>();
        test_full_like_out::<f64>();
        test_full_like_out_bool();
        test_full_like_out::<BFloat16>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_mismatched_shape_dies() {
        // ET_FORALL_REAL_TYPES_AND(Bool)
        test_full_like_out_mismatched_shape::<u8>();
        test_full_like_out_mismatched_shape::<i8>();
        test_full_like_out_mismatched_shape::<i16>();
        test_full_like_out_mismatched_shape::<i32>();
        test_full_like_out_mismatched_shape::<i64>();
        test_full_like_out_mismatched_shape::<f32>();
        test_full_like_out_mismatched_shape::<f64>();
        test_full_like_out_mismatched_shape::<bool>();
    }

    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![3.0f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        let ret = full_like_out(
            &mut ctx,
            &x,
            &Scalar::from_double(3.0),
            Some(MemoryFormat::Contiguous),
            &out,
        );
        assert_tensor_close!(out, expected_result);
        let _ = ret;
    }

    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = full_like_out(
            &mut ctx,
            &x,
            &Scalar::from_double(3.0),
            Some(MemoryFormat::Contiguous),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = full_like_out(
            &mut ctx,
            &x,
            &Scalar::from_double(3.0),
            Some(MemoryFormat::Contiguous),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED: Dynamic shape unbound not supported
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    #[ignore]
    fn op_full_like_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.04876953363418579,
                0.816348671913147,
                0.44230276346206665,
                0.2767965793609619,
                0.8998266458511353,
                0.09595239162445068,
            ],
        );
        let expected_result = tf.make_default(vec![3, 2], vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        let _ret = full_like_out(
            &mut ctx,
            &x,
            &Scalar::from_double(3.0),
            Some(MemoryFormat::Contiguous),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_half_support() {
        let tf = TensorFactory::<Half>::new();
        let in_ = tf.ones_default(vec![2, 3]);
        let out = tf.zeros_default(vec![2, 3]);
        let mut ctx = context();

        full_like_out(&mut ctx, &in_, &Scalar::from_bool(false), None, &out);
        assert_tensor_close!(
            out,
            tf.full(vec![2, 3], Half::from_f64(0.0), TensorShapeDynamism::STATIC)
        );

        full_like_out(&mut ctx, &in_, &Scalar::from_bool(true), None, &out);
        assert_tensor_close!(
            out,
            tf.full(vec![2, 3], Half::from_f64(1.0), TensorShapeDynamism::STATIC)
        );

        full_like_out(&mut ctx, &in_, &Scalar::from_i64(7), None, &out);
        assert_tensor_close!(
            out,
            tf.full(vec![2, 3], Half::from_f64(7.0), TensorShapeDynamism::STATIC)
        );

        full_like_out(&mut ctx, &in_, &Scalar::from_double(2.5), None, &out);
        assert_tensor_close!(
            out,
            tf.full(vec![2, 3], Half::from_f64(2.5), TensorShapeDynamism::STATIC)
        );

        full_like_out(
            &mut ctx,
            &in_,
            &Scalar::from_double(f64::INFINITY),
            None,
            &out,
        );
        assert_tensor_close!(
            out,
            tf.full(
                vec![2, 3],
                Half::from_f64(f64::INFINITY),
                TensorShapeDynamism::STATIC
            )
        );
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpFullLikeTest)
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_char_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_short_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_float_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn/test]
    #[test]
    fn op_full_like_out_test_float_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
