//! Literal port of kernels/portable/cpu/op_gt.cpp.

use crate::kernels::portable::cpu::pattern::comparison_op::{
    ComparisonOp, comparison_scalar_out, comparison_tensor_out,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `std::greater` functor becomes a zero-sized `Greater` type
// implementing `ComparisonOp`, reproducing `Comparison<CTYPE_COMPUTE>()` via
// `Greater::apply::<CTYPE_COMPUTE>`.
struct Greater;
impl ComparisonOp for Greater {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a > b
    }
}

// PORT-NOTE (cross-module): the compile-time `op_name` template parameter
// ("gt.Tensor_out" / "gt.Scalar_out") is dropped — the ported comparison
// patterns take no op-name argument.

// [spec:et:def:op-gt.torch.executor.native.gt-tensor-out-fn]
// [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn]
#[executorch_macros::et_kernel("aten::gt.Tensor_out")]
pub fn gt_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    comparison_tensor_out::<Greater>(ctx, a, b, out)
}

// [spec:et:def:op-gt.torch.executor.native.gt-scalar-out-fn]
// [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn]
pub fn gt_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    comparison_scalar_out::<Greater>(ctx, a, b, out)
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

    trait FromCmp: Copy {
        fn from_i32(v: i32) -> Self;
        fn from_bool(v: bool) -> Self;
    }
    macro_rules! impl_from_cmp_num {
        ($($t:ty),*) => {$(impl FromCmp for $t {
            fn from_i32(v: i32) -> Self { v as $t }
            fn from_bool(v: bool) -> Self { v as i32 as $t }
        })*};
    }
    impl_from_cmp_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromCmp for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            Half::from_f32(v as i32 as f32)
        }
    }
    impl FromCmp for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            BFloat16::from_f32(v as i32 as f32)
        }
    }
    impl FromCmp for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
        fn from_bool(v: bool) -> Self {
            v
        }
    }

    fn b<T: FromCmp>(v: &[bool]) -> Vec<T> {
        v.iter().map(|&x| T::from_bool(x)).collect()
    }
    fn i<T: FromCmp>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
        OUT: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        let tf_input = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let a = tf_input.make_default(vec![2, 2], i::<IN>(&[2, 3, 2, 4]));
        let bb = tf_input.make_default(vec![2, 2], i::<IN>(&[1, 4, 2, 3]));
        let out = tf_out.zeros_default(vec![2, 2]);

        let mut ctx = context();
        gt_tensor_out(&mut ctx, &a, &bb, &out);
        assert_tensor_eq!(
            out,
            tf_out.make_default(vec![2, 2], b::<OUT>(&[true, false, false, true]))
        );
    }

    fn test_gt_scalar_out<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
        OUT: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        let tf = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let out = tf_out.ones_default(sizes.clone());
        let other = Scalar::from_i64(2);

        let a = tf.make_default(sizes.clone(), i::<IN>(&[3, 1, 2, 4]));
        let mut ctx = context();
        gt_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_out.make_default(sizes, b::<OUT>(&[true, false, false, true]))
        );
    }

    fn forall_realhbf16_out<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        test_gt_scalar_out::<IN, u8>();
        test_gt_scalar_out::<IN, i8>();
        test_gt_scalar_out::<IN, i16>();
        test_gt_scalar_out::<IN, i32>();
        test_gt_scalar_out::<IN, i64>();
        test_gt_scalar_out::<IN, Half>();
        test_gt_scalar_out::<IN, BFloat16>();
        test_gt_scalar_out::<IN, f32>();
        test_gt_scalar_out::<IN, f64>();
        test_gt_scalar_out::<IN, bool>();
    }

    fn forall_realhbf16_out_tensor<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        test_dtype::<IN, u8>();
        test_dtype::<IN, i8>();
        test_dtype::<IN, i16>();
        test_dtype::<IN, i32>();
        test_dtype::<IN, i64>();
        test_dtype::<IN, Half>();
        test_dtype::<IN, BFloat16>();
        test_dtype::<IN, f32>();
        test_dtype::<IN, f64>();
        test_dtype::<IN, bool>();
    }

    // [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn/test]
    // [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn/test]
    #[test]
    fn op_gt_scalar_out_test_all_real_input_bool_output_support() {
        forall_realhbf16_out::<u8>();
        forall_realhbf16_out::<i8>();
        forall_realhbf16_out::<i16>();
        forall_realhbf16_out::<i32>();
        forall_realhbf16_out::<i64>();
        forall_realhbf16_out::<Half>();
        forall_realhbf16_out::<BFloat16>();
        forall_realhbf16_out::<f32>();
        forall_realhbf16_out::<f64>();
    }

    // [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn/test]
    #[test]
    fn op_gt_scalar_out_test_bool_input_dtype() {
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let a = tf_bool.make_default(sizes.clone(), vec![false, true, false, true]);
        let out = tf_bool.zeros_default(sizes.clone());
        let other = Scalar::from_double(0.5);

        let mut ctx = context();
        gt_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(sizes, vec![false, true, false, true])
        );
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn/test]
    #[test]
    fn op_gt_scalar_out_test_mismatched_in_out_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);
        let other = Scalar::from_i64(3);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gt_scalar_out(&mut ctx, &a, &other, &out));
    }

    // [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn/test]
    #[test]
    fn op_gt_scalar_out_test_dynamic_out_shape_test() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];
        let out_sizes = vec![4, 1];

        let out = tf.zeros(out_sizes, TensorShapeDynamism::DYNAMIC_BOUND);
        let other = Scalar::from_i64(2);

        let a = tf.make_default(sizes.clone(), vec![3, 1, 2, 4]);
        let mut ctx = context();
        gt_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(out, tf.make_default(sizes, vec![1, 0, 0, 1]));
    }

    // [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn/test]
    // [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn/test]
    #[test]
    fn op_gt_tensor_out_test_all_dtypes_supported() {
        forall_realhbf16_out_tensor::<u8>();
        forall_realhbf16_out_tensor::<i8>();
        forall_realhbf16_out_tensor::<i16>();
        forall_realhbf16_out_tensor::<i32>();
        forall_realhbf16_out_tensor::<i64>();
        forall_realhbf16_out_tensor::<Half>();
        forall_realhbf16_out_tensor::<BFloat16>();
        forall_realhbf16_out_tensor::<f32>();
        forall_realhbf16_out_tensor::<f64>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn/test]
    #[test]
    fn op_gt_tensor_out_test_mismatched_in_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let bb = tf_int.ones_default(vec![2, 2]);
        let out = tf_bool.ones_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gt_tensor_out(&mut ctx, &a, &bb, &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn/test]
    #[test]
    fn op_gt_tensor_out_test_mismatched_in_out_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let bb = tf_int.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, gt_tensor_out(&mut ctx, &a, &bb, &out));
    }

    // [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn/test]
    #[test]
    fn op_gt_tensor_out_test_dynamic_out_shape_test() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 4]);
        let bb = tf.make_default(vec![2, 2], vec![1, 4, 2, 3]);

        let out = tf.zeros(vec![1, 4], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        gt_tensor_out(&mut ctx, &a, &bb, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![1, 0, 0, 1]));
    }
}
