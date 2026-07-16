//! Literal port of kernels/portable/cpu/op_log_softmax.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::check_log_softmax_args;
use crate::kernels::portable::cpu::util::functional_util::{
    apply_unary_map_fn, apply_unary_map_reduce_fn, apply_unary_reduce_fn,
};
use crate::kernels::portable::cpu::util::reduce_util::apply_over_dim_whole;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    nonzero_dim, resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ dispatches over FLOATHBF16 and derives an accumulation type
// `ACC = std::conditional_t<CTYPE is Half or BFloat16, float, CTYPE>`. Rust has
// no `std::conditional_t`, so — following math_util.rs's per-type-trait strategy
// — a `LogSoftmaxCtype` trait carries the associated `Acc` type plus the
// per-CTYPE `std::max`, `static_cast<ACC>`, and `static_cast<CTYPE>` operations,
// and `LogSoftmaxAcc` supplies `std::exp`/`std::log` and add on `ACC` (always
// f32 or f64).
trait LogSoftmaxAcc: Copy {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
}

impl LogSoftmaxAcc for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
}

impl LogSoftmaxAcc for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
}

trait LogSoftmaxCtype: Copy {
    type Acc: LogSoftmaxAcc;
    fn max(a: Self, b: Self) -> Self;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(val: Self::Acc) -> Self;
}

impl LogSoftmaxCtype for f32 {
    type Acc = f32;
    fn max(a: Self, b: Self) -> Self {
        f32::max(a, b)
    }
    fn to_acc(self) -> Self::Acc {
        self
    }
    fn from_acc(val: Self::Acc) -> Self {
        val
    }
}

impl LogSoftmaxCtype for f64 {
    type Acc = f64;
    fn max(a: Self, b: Self) -> Self {
        f64::max(a, b)
    }
    fn to_acc(self) -> Self::Acc {
        self
    }
    fn from_acc(val: Self::Acc) -> Self {
        val
    }
}

impl LogSoftmaxCtype for Half {
    type Acc = f32;
    fn max(a: Self, b: Self) -> Self {
        if a > b { a } else { b }
    }
    fn to_acc(self) -> Self::Acc {
        self.to_f32()
    }
    fn from_acc(val: Self::Acc) -> Self {
        Half::from_f32_const(val)
    }
}

impl LogSoftmaxCtype for BFloat16 {
    type Acc = f32;
    fn max(a: Self, b: Self) -> Self {
        if a > b { a } else { b }
    }
    fn to_acc(self) -> Self::Acc {
        self.to_f32()
    }
    fn from_acc(val: Self::Acc) -> Self {
        BFloat16::from_f32_const(val)
    }
}

// [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-out-fn]
// [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn]
pub fn log_softmax_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    half_to_float: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_log_softmax_args(in_, dim, half_to_float, out),
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

    // Adjust for negative dim
    let dim: i64 = if dim < 0 {
        dim + nonzero_dim(in_) as i64
    } else {
        dim
    };

    // For half-precision inputs, the exp-sum is accumulated in float to avoid
    // saturation (BFloat16 saturates near 256, Half near 2048). Matches ATen's
    // acc_type behavior. See also op_grid_sampler_2d.cpp.
    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, "_log_softmax.out", CTYPE, {
        type ACC = <CTYPE as LogSoftmaxCtype>::Acc;
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        apply_over_dim_whole(
            |size: usize, stride: usize, base: usize| {
                // calculate max in log_softmax dim. During log_softmax
                // computation each value is subtracted by the maximum in
                // value before calling exp to preserve numerical stability.
                let max_in: CTYPE = apply_unary_reduce_fn(
                    |val_in: CTYPE, val_accum: CTYPE| -> CTYPE {
                        <CTYPE as LogSoftmaxCtype>::max(val_in, val_accum)
                    },
                    unsafe { in_data.add(base) },
                    size as i64,
                    stride as i64,
                );

                let exp_sum: ACC = apply_unary_map_reduce_fn::<CTYPE, ACC, _, _>(
                    |val_in: CTYPE| -> ACC {
                        LogSoftmaxAcc::exp(
                            LogSoftmaxCtype::to_acc(val_in).sub(LogSoftmaxCtype::to_acc(max_in)),
                        )
                    },
                    |mapped_in: ACC, val_accum: ACC| -> ACC { val_accum.add(mapped_in) },
                    unsafe { in_data.add(base) },
                    size as i64,
                    stride as i64,
                );
                let log_sum: ACC = LogSoftmaxAcc::ln(exp_sum);

                apply_unary_map_fn(
                    |val_in: CTYPE| -> CTYPE {
                        <CTYPE as LogSoftmaxCtype>::from_acc(
                            LogSoftmaxCtype::to_acc(val_in)
                                .sub(LogSoftmaxCtype::to_acc(max_in))
                                .sub(log_sum),
                        )
                    },
                    unsafe { in_data.add(base) },
                    unsafe { out_data.add(base) },
                    size as i64,
                    stride as i64,
                );
            },
            in_,
            &Some(dim),
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::{assert_tensor_close, assert_tensor_close_with_tol, assert_tensor_eq};
    // PORT-NOTE (out of assignment): op_log_softmax's ported test module used
    // `TensorShapeDynamism::*` without importing it, breaking the whole test
    // build. Added the missing import to unblock; flagged for the owner.
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn f<T: FromF64>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(vec![2, 3], f::<T>(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);

        let expected = tf.make_default(
            vec![2, 3],
            f::<T>(&[-2.40761, -1.40761, -0.407606, -2.40761, -1.40761, -0.407606]),
        );

        if T::VALUE == ScalarType::BFloat16 {
            assert!(tensors_are_close(
                &out,
                &expected,
                1e-2,
                Some(internal::K_DEFAULT_ATOL)
            ));
        } else {
            assert_tensor_close!(out, expected);
        }
    }

    fn test_dtype_noncontiguous_dim<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(
            vec![9, 3],
            f::<T>(&[
                0.0, 9.0, 18.0, 1.0, 10.0, 19.0, 2.0, 11.0, 20.0, 3.0, 12.0, 21.0, 4.0, 13.0, 22.0,
                5.0, 14.0, 23.0, 6.0, 15.0, 24.0, 7.0, 16.0, 25.0, 8.0, 17.0, 26.0,
            ]),
        );
        let out = tf.zeros_default(vec![9, 3]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 0, false, &out);

        let expected = tf.make_default(
            vec![9, 3],
            f::<T>(&[
                -8.45855, -8.45855, -8.45855, -7.45855, -7.45855, -7.45855, -6.45855, -6.45855,
                -6.45855, -5.45855, -5.45855, -5.45855, -4.45855, -4.45855, -4.45855, -3.45855,
                -3.45855, -3.45855, -2.45855, -2.45855, -2.45855, -1.45855, -1.45855, -1.45855,
                -0.458552, -0.458552, -0.458552,
            ]),
        );

        if T::VALUE == ScalarType::BFloat16 {
            assert!(tensors_are_close(
                &out,
                &expected,
                1e-2,
                Some(internal::K_DEFAULT_ATOL)
            ));
        } else {
            assert_tensor_close!(out, expected);
        }
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_smoke() {
        let tff = TensorFactory::<f32>::new();
        let sizes = vec![1, 3];
        let in_ = tff.make_default(sizes.clone(), vec![0.0, 1.0, 2.0]);
        let out = tff.zeros_default(sizes);

        let mut ctx = context();
        let ret = log_softmax_out(&mut ctx, &in_, 1, false, &out);

        // Should always return the provided out Tensor.
        assert_tensor_eq!(ret, out);

        let expected = tff.make_default(vec![1, 3], vec![-2.40761, -1.40761, -0.407606]);
        assert_tensor_close!(out, expected);
    }

    // ET_FORALL_FLOATHBF16_TYPES: Float, Double, Half, BFloat16.
    // PORT-NOTE: C++ guards on `op_log_softmax_dtype_double`, which is true for the
    // portable kernel; the skip never triggers.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_all_dtypes_supported() {
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_non_contiguous() {
        test_dtype_noncontiguous_dim::<f32>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    // also verifies check_log_softmax_args: dim=3 out of bounds fails the
    // tensor_has_dim check.
    // [spec:et:sem:activation-ops-util.torch.executor.check-log-softmax-args-fn/test]
    #[test]
    fn op_log_softmax_out_test_mismatched_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();

        let x = tff.make_default(vec![1, 3], vec![0.0, 1.0, 2.0]);
        let out = tff.zeros_default(vec![1, 3]);

        // Dim out of bounds
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 3, false, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.ones_default(vec![3, 4]);
        let wrong_out = tf.zeros_default(vec![2, 10, 4]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &wrong_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ guards on `op_log_softmax_dtype_double` (true) and `is_aten`
    // (false) for the portable kernel; both skips are inactive.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_test_with_large_number() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(vec![1, 2], vec![-1e5, 1e5]);
        let out = tf.zeros_default(vec![1, 2]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);

        let expected = tf.make_default(vec![1, 2], vec![-200000.0, 0.0]);
        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: C++ guards on `op_log_softmax_dtype_double` (true) and `is_aten`
    // (false); both skips are inactive.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_negative_dim() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = tf.zeros_default(vec![2, 3]);
        let out_negative_dim = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, -1, false, &out_negative_dim);

        let expected = tf.make_default(
            vec![2, 3],
            vec![-2.40761, -1.40761, -0.407606, -2.40761, -1.40761, -0.407606],
        );
        assert_tensor_close!(out, expected);
        assert_tensor_close!(out_negative_dim, expected);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 0, false, &out);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, -2, false, &out_negative_dim);

        let expected = tf.make_default(
            vec![2, 3],
            vec![
                -3.04859, -3.04859, -3.04859, -0.0485874, -0.0485874, -0.0485874,
            ],
        );
        assert_tensor_close!(out, expected);
        assert_tensor_close!(out_negative_dim, expected);
    }

    // Gated in C++ by `#if !defined(USE_ATEN_LIB)`; the ported runtime is never ATen.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_upper_bound_out_tensor() {
        let tff = TensorFactory::<f32>::new();

        let x = tff.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = tff.zeros(vec![5, 9], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);

        let expected = tff.make_default(
            vec![2, 3],
            vec![-2.40761, -1.40761, -0.407606, -2.40761, -1.40761, -0.407606],
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![-2.3025851249694824f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_bfloat16_large_dim_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        const N: i32 = 512;
        let x = tf.zeros_default(vec![1, N]);
        let out = tf.zeros_default(vec![1, N]);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        let expected = tf.full(
            vec![1, N],
            BFloat16::from_f32(-(N as f32).ln()),
            TensorShapeDynamism::STATIC,
        );
        assert_tensor_close_with_tol!(out, expected, 1e-5, 1e-1);
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.754019558429718,
                0.8973914980888367,
                0.34469079971313477,
                0.40464818477630615,
                0.36159539222717285,
                0.1138353943824768,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.7674003839492798,
                -0.6240284442901611,
                -0.7235751748085022,
                -0.6636177897453308,
                -0.576920747756958,
                -0.824680745601654,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.754019558429718,
                0.8973914980888367,
                0.34469079971313477,
                0.40464818477630615,
                0.36159539222717285,
                0.1138353943824768,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.7674003839492798,
                -0.6240284442901611,
                -0.7235751748085022,
                -0.6636177897453308,
                -0.576920747756958,
                -0.824680745601654,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED in C++: Dynamic shape not supported.
    // PORT-NOTE: ported as `#[ignore]` mirroring the `DISABLED_` gtest prefix.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    #[ignore]
    fn op_log_softmax_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.754019558429718,
                0.8973914980888367,
                0.34469079971313477,
                0.40464818477630615,
                0.36159539222717285,
                0.1138353943824768,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.7674003839492798,
                -0.6240284442901611,
                -0.7235751748085022,
                -0.6636177897453308,
                -0.576920747756958,
                -0.824680745601654,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        log_softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: C++ guards on `op_log_softmax_dtype_double` (true) for the portable
    // kernel, so the `expect_failure()` branch and its early skip do not trigger.
    // [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn/test]
    #[test]
    fn op_log_softmax_out_test_double_case() {
        let tf = TensorFactory::<f64>::new();

        let input = tf.zeros_default(vec![8, 5, 7]);
        let in_data = input.mutable_data_ptr::<f64>();
        for i in 0..(8 * 5 * 7) {
            unsafe {
                *in_data.add(i) = (i as f64) * 0.01;
            }
        }

        let out = tf.zeros_default(vec![8, 5, 7]);

        let mut ctx = context();
        log_softmax_out(&mut ctx, &input, 2, false, &out);

        assert_eq!(*out.sizes().at(0), 8);
        assert_eq!(*out.sizes().at(1), 5);
        assert_eq!(*out.sizes().at(2), 7);

        let out_data = out.const_data_ptr::<f64>();
        for i in 0..(8 * 5 * 7) {
            let v = unsafe { *out_data.add(i) };
            assert!(!v.is_nan(), "Output should not contain NaN at index {}", i);
            assert!(
                !v.is_infinite(),
                "Output should not contain Inf at index {}",
                i
            );
        }
        for i in 0..(8 * 5 * 7) {
            let v = unsafe { *out_data.add(i) };
            assert!(v <= 0.0, "Log softmax values should be <= 0 at index {}", i);
        }
        for batch in 0..8 {
            for channel in 0..5 {
                let mut sum_exp = 0.0f64;
                for dim2 in 0..7 {
                    let idx = batch * 5 * 7 + channel * 7 + dim2;
                    sum_exp += unsafe { *out_data.add(idx) }.exp();
                }
                assert!(
                    (sum_exp - 1.0).abs() <= 1e-6,
                    "Sum of exp(log_softmax) should be 1.0 for batch={}, channel={}",
                    batch,
                    channel
                );
            }
        }
    }
}
