//! Literal port of kernels/portable/cpu/op_arange.cpp.

use crate::kernels::portable::cpu::scalar_utils::extract_scalar;
use crate::kernels::portable::cpu::util::arange_util::{
    arange_out_impl, arange_out_impl_end, compute_arange_out_size,
};
use crate::kernels::portable::cpu::util::kernel_ops_util::check_arange_args;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `const Scalar&` maps to
// `&Scalar`; the ported `extract_scalar` takes `Scalar` by value, so `*end` etc.
// are passed. `Tensor::SizesType` == `TensorSizesType`. `resize_tensor(out,
// {&out_length, 1})` is `ArrayRef::from_raw_parts(&out_length, 1)`. The two-arg
// `arange_out_impl(ctx, end, out)` overload maps to `arange_out_impl_end`.

// [spec:et:def:op-arange.torch.executor.native.arange-out-fn]
// [spec:et:sem:op-arange.torch.executor.native.arange-out-fn]
pub fn arange_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    end: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let mut end_val: f64 = 0.0;
    crate::et_kernel_check!(
        ctx,
        extract_scalar(*end, &mut end_val),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        check_arange_args(0.0, end_val, 1.0, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

    let out_length: TensorSizesType = compute_arange_out_size(0.0, end_val, 1.0);

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::from_raw_parts(&out_length, 1)) == Error::Ok,
        InvalidArgument,
        out
    );

    arange_out_impl_end(ctx, end_val, out);

    out
}

// [spec:et:def:op-arange.torch.executor.native.arange-start-out-fn]
// [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn]
#[executorch_macros::et_kernel("aten::arange.start_out")]
pub fn arange_start_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    start: &Scalar,
    end: &Scalar,
    step: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    let mut d_start: f64 = 0.0;
    crate::et_kernel_check!(
        ctx,
        extract_scalar(*start, &mut d_start),
        InvalidArgument,
        out
    );

    let mut d_end: f64 = 0.0;
    crate::et_kernel_check!(ctx, extract_scalar(*end, &mut d_end), InvalidArgument, out);

    let mut d_step: f64 = 0.0;
    crate::et_kernel_check!(
        ctx,
        extract_scalar(*step, &mut d_step),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        check_arange_args(d_start, d_end, d_step, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

    let out_length: TensorSizesType = compute_arange_out_size(d_start, d_end, d_step);

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::from_raw_parts(&out_length, 1)) == Error::Ok,
        InvalidArgument,
        out
    );

    arange_out_impl(ctx, d_start, d_end, d_step, out);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `Scalar(static_cast<CTYPE>(v))` builds a Scalar from the tensor's
    // element ctype; reproduced per-type via `to_scalar` and integer-literal make()
    // values via `from_i64`.
    trait ScalarCtype: CppTypeToScalarType + FactoryValue {
        fn from_i64(v: i64) -> Self;
        fn to_scalar(v: i64) -> Scalar;
    }
    macro_rules! impl_scalar_ctype_int {
        ($($t:ty),*) => {$(impl ScalarCtype for $t {
            fn from_i64(v: i64) -> Self { v as $t }
            fn to_scalar(v: i64) -> Scalar { Scalar::from_i64(v as $t as i64) }
        })*};
    }
    impl_scalar_ctype_int!(u8, i8, i16, i32, i64);
    macro_rules! impl_scalar_ctype_float {
        ($($t:ty),*) => {$(impl ScalarCtype for $t {
            fn from_i64(v: i64) -> Self { v as $t }
            fn to_scalar(v: i64) -> Scalar { Scalar::from_double(v as $t as f64) }
        })*};
    }
    impl_scalar_ctype_float!(f32, f64);
    impl ScalarCtype for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
        fn to_scalar(v: i64) -> Scalar {
            Scalar::from_half(Half::from_f32(v as f32))
        }
    }
    impl ScalarCtype for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn to_scalar(v: i64) -> Scalar {
            Scalar::from_bfloat16(BFloat16::from_f32(v as f32))
        }
    }

    fn test_arange_dtype<T: ScalarCtype>() {
        let tf = TensorFactory::<T>::new();

        let end = T::to_scalar(10);
        let out = tf.zeros_default(vec![10]);

        let mut ctx = context();
        let ret = arange_out(&mut ctx, &end, &out);

        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![10], (0..10).map(T::from_i64).collect());
        assert_tensor_eq!(out, expected);
    }

    fn test_arange_start_dtype<T: ScalarCtype>() {
        let tf = TensorFactory::<T>::new();

        let start = T::to_scalar(0);
        let end = T::to_scalar(10);
        let step = T::to_scalar(1);

        let out = tf.zeros_default(vec![10]);

        let mut ctx = context();
        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);

        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![10], (0..10).map(T::from_i64).collect());
        assert_tensor_eq!(out, expected);
    }

    // also exercises arange_out_impl (value fill) and compute_arange_out_size
    // across every REALHBF16 dtype.
    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    // [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn/test]
    // [spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn/test]
    // Also exercises check_arange_args (1-d out, step/sign consistency) in-path.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-arange-args-fn/test]
    #[test]
    fn op_arange_out_test_all_real_hbf16_dtypes_supported() {
        test_arange_dtype::<u8>();
        test_arange_dtype::<i8>();
        test_arange_dtype::<i16>();
        test_arange_dtype::<i32>();
        test_arange_dtype::<i64>();
        test_arange_dtype::<f32>();
        test_arange_dtype::<f64>();
        test_arange_dtype::<Half>();
        test_arange_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    #[test]
    fn op_arange_out_test_float_number_not_equal_int_support() {
        let tf = TensorFactory::<f32>::new();

        let end = Scalar::from_double(5.5);
        let out = tf.zeros_default(vec![6]);

        let mut ctx = context();
        let ret = arange_out(&mut ctx, &end, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    #[test]
    fn op_arange_out_test_out_dim_unsupported_die() {
        let tf = TensorFactory::<f32>::new();

        let end = Scalar::from_i64(5);
        let out = tf.zeros_default(vec![5, 1]);

        let mut ctx = context();
        arange_out(&mut ctx, &end, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    #[test]
    fn op_arange_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let expected_result = tf.make_default(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros(vec![5], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        arange_out(&mut ctx, &Scalar::from_i64(5), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    #[test]
    fn op_arange_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let expected_result = tf.make_default(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        arange_out(&mut ctx, &Scalar::from_i64(5), &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: `DynamicShapeUnbound` is `ET_SKIP_IF(!is_aten, ...)`; the portable
    // (non-ATen) build always skips it. Ported as a no-op skip.
    // [spec:et:sem:op-arange.torch.executor.native.arange-out-fn/test]
    #[test]
    fn op_arange_out_test_dynamic_shape_unbound() {
        // Dynamic Unbound not supported in the portable build; skipped.
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_all_real_hbf16_dtypes_supported() {
        test_arange_start_dtype::<u8>();
        test_arange_start_dtype::<i8>();
        test_arange_start_dtype::<i16>();
        test_arange_start_dtype::<i32>();
        test_arange_start_dtype::<i64>();
        test_arange_start_dtype::<f32>();
        test_arange_start_dtype::<f64>();
        test_arange_start_dtype::<Half>();
        test_arange_start_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_float_number_not_equal_int_support() {
        let tf = TensorFactory::<f32>::new();

        let start = Scalar::from_i64(0);
        let end = Scalar::from_double(5.5);
        let step = Scalar::from_i64(1);

        let out = tf.zeros_default(vec![6]);

        let mut ctx = context();
        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_out_dim_unsupported_die() {
        let tf = TensorFactory::<f32>::new();

        let start = Scalar::from_i64(0);
        let end = Scalar::from_i64(5);
        let step = Scalar::from_i64(1);

        let out = tf.zeros_default(vec![5, 1]);

        let mut ctx = context();
        arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let expected_result = tf.make_default(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros(vec![5], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        arange_start_out(
            &mut ctx,
            &Scalar::from_i64(0),
            &Scalar::from_i64(5),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let expected_result = tf.make_default(vec![5], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros(vec![10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        arange_start_out(
            &mut ctx,
            &Scalar::from_i64(0),
            &Scalar::from_i64(5),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: `DynamicShapeUnbound` is `ET_SKIP_IF(!is_aten, ...)`; the portable
    // (non-ATen) build always skips it. Ported as a no-op skip.
    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_dynamic_shape_unbound() {
        // Dynamic Unbound not supported in the portable build; skipped.
    }

    // Fractional start/step (1.1, end 5.51) exercises compute_arange_out_size's
    // ceil-based size math (5 elements) and arange_out_impl's value fill.
    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    // [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn/test]
    // [spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn/test]
    #[test]
    fn op_arange_start_out_test_start_out() {
        let tf = TensorFactory::<f32>::new();

        let start = Scalar::from_double(1.1);
        let end = Scalar::from_double(5.5);
        let step = Scalar::from_double(1.1);

        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![4], vec![1.1, 2.2, 3.3, 4.4]);
        assert_tensor_eq!(out, expected);

        let end = Scalar::from_double(5.51);
        let out = tf.zeros_default(vec![5]);

        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![5], vec![1.1, 2.2, 3.3, 4.4, 5.5]);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn/test]
    #[test]
    fn op_arange_start_out_test_start_out_negative_step() {
        let tf = TensorFactory::<f32>::new();

        let start = Scalar::from_double(5.5);
        let end = Scalar::from_double(1.1);
        let step = Scalar::from_double(-1.1);

        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![4], vec![5.5, 4.4, 3.3, 2.2]);
        assert_tensor_eq!(out, expected);

        let end = Scalar::from_double(1.09);
        let out = tf.zeros_default(vec![5]);

        let ret = arange_start_out(&mut ctx, &start, &end, &step, &out);
        assert_tensor_eq!(*ret, out);

        let expected = tf.make_default(vec![5], vec![5.5, 4.4, 3.3, 2.2, 1.1]);
        assert_tensor_eq!(out, expected);
    }
}
