//! Literal port of kernels/portable/cpu/op_ne.cpp.

use crate::kernels::portable::cpu::pattern::comparison_op::{
    ComparisonOp, comparison_scalar_out, comparison_tensor_out,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `std::not_equal_to` functor becomes a zero-sized `NotEqualTo`
// type implementing `ComparisonOp`, reproducing `Comparison<CTYPE_COMPUTE>()` via
// `NotEqualTo::apply::<CTYPE_COMPUTE>`. `PartialOrd` (the `ComparisonOp` bound)
// subsumes the `!=` used here. Mirror of op_eq.
struct NotEqualTo;
impl ComparisonOp for NotEqualTo {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a != b
    }
}

// PORT-NOTE (cross-module): the compile-time `op_name` template parameter
// ("ne.Tensor_out" / "ne.Scalar_out") is dropped — the ported comparison
// patterns take no op-name argument.

// [spec:et:def:op-ne.torch.executor.native.ne-tensor-out-fn]
// [spec:et:sem:op-ne.torch.executor.native.ne-tensor-out-fn]
pub fn ne_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    comparison_tensor_out::<NotEqualTo>(ctx, a, b, out)
}

// [spec:et:def:op-ne.torch.executor.native.ne-scalar-out-fn]
// [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn]
#[executorch_macros::et_kernel("aten::ne.Scalar_out")]
pub fn ne_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    comparison_scalar_out::<NotEqualTo>(ctx, a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REALHBF16 factory element types used by these tests.
    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI64 for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    // OpNeTest::test_dtype (tensor variant)
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf_input = TensorFactory::<T>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_i64(x)).collect() };

        let a = tf_input.make_default(vec![2, 2], d(&[2, 3, 2, 4]));
        let b = tf_input.make_default(vec![2, 2], d(&[2, 2, 2, 2]));
        let out = tf_bool.zeros_default(vec![2, 2]);

        let mut ctx = context();
        ne_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(vec![2, 2], vec![false, true, false, true])
        );
    }

    // OpNeScalarOutTest::test_ne_scalar_out
    fn test_ne_scalar_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let tf_out = TensorFactory::<bool>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_i64(x)).collect() };

        let sizes = vec![2, 2];
        let out = tf_out.ones_default(sizes.clone());
        let other = Scalar::from_i64(2);

        let mut ctx = context();
        ne_scalar_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), d(&[2, 3, 2, 3])),
            &other,
            &out,
        );
        assert_tensor_eq!(
            out,
            tf_out.make_default(sizes, vec![false, true, false, true])
        );
    }

    // OpNeScalarOutTest::test_ne_all_output_dtypes
    fn test_ne_all_output_dtypes<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 5];

        let in_ = tf_float.ones_default(sizes.clone());
        let out = tf_out.zeros_default(sizes.clone());
        let other = Scalar::from_i64(3);

        let mut ctx = context();
        ne_scalar_out(&mut ctx, &in_, &other, &out);
        assert_tensor_eq!(out, tf_out.ones_default(sizes));
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_all_real_input_bool_output_support() {
        // ET_FORALL_REALHBF16_TYPES
        test_ne_scalar_out::<u8>();
        test_ne_scalar_out::<i8>();
        test_ne_scalar_out::<i16>();
        test_ne_scalar_out::<i32>();
        test_ne_scalar_out::<i64>();
        test_ne_scalar_out::<f32>();
        test_ne_scalar_out::<f64>();
        test_ne_scalar_out::<Half>();
        test_ne_scalar_out::<BFloat16>();
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_bool_input_dtype() {
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let a = tf_bool.make_default(sizes.clone(), vec![false, true, false, true]);
        let out = tf_bool.zeros_default(sizes.clone());
        let other = Scalar::from_i64(1);

        let mut ctx = context();
        ne_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(sizes, vec![true, false, true, false])
        );
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)` — the Rust port is the non-aten branch,
    // so the skip never triggers and the body runs.
    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_mismatched_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);
        let other = Scalar::from_i64(3);

        let mut ctx = context();
        ne_scalar_out(&mut ctx, &a, &other, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_all_real_output_d_types_supported() {
        // ET_FORALL_REALHBF16_TYPES
        test_ne_all_output_dtypes::<u8>();
        test_ne_all_output_dtypes::<i8>();
        test_ne_all_output_dtypes::<i16>();
        test_ne_all_output_dtypes::<i32>();
        test_ne_all_output_dtypes::<i64>();
        test_ne_all_output_dtypes::<f32>();
        test_ne_all_output_dtypes::<f64>();
        test_ne_all_output_dtypes::<Half>();
        test_ne_all_output_dtypes::<BFloat16>();
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-tensor-out-fn/test]
    #[test]
    fn op_ne_test_all_dtypes_supported() {
        // ET_FORALL_REALHBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<bool>::new();

        let x = tf.make_default(vec![3, 2], vec![2, 0, 2, 0, 1, 0]);
        let expected = tf_out.make_default(vec![3, 2], vec![false, true, false, true, true, true]);

        let other = Scalar::from_i64(2);

        let out = tf_out.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        ne_scalar_out(&mut ctx, &x, &other, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<bool>::new();

        let x = tf.make_default(vec![3, 2], vec![2, 0, 2, 0, 1, 0]);
        let expected = tf_out.make_default(vec![3, 2], vec![false, true, false, true, true, true]);

        let other = Scalar::from_i64(2);

        let out = tf_out.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        ne_scalar_out(&mut ctx, &x, &other, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: `ET_SKIP_IF(!output_resize, ...)` — the portable (non-aten) kernel
    // reports `output_resize = false`, so this test is SKIPPED. `DYNAMIC_UNBOUND`
    // output resize is genuinely unsupported by the portable kernel. Body preserved
    // for correspondence; guarded by the skip.
    // [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn/test]
    #[test]
    fn op_ne_scalar_out_test_dynamic_shape_unbound() {
        const OUTPUT_RESIZE: bool = false;
        if !OUTPUT_RESIZE {
            return;
        }
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<bool>::new();

        let x = tf.make_default(vec![3, 2], vec![2, 0, 2, 0, 1, 0]);
        let expected = tf_out.make_default(vec![3, 2], vec![false, true, false, true, true, true]);

        let other = Scalar::from_i64(2);

        let out = tf_out.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        ne_scalar_out(&mut ctx, &x, &other, &out);
        assert_tensor_eq!(out, expected);
    }
}
