//! Literal port of kernels/portable/cpu/op_softmax.cpp.

use crate::kernels::portable::cpu::util::activation_ops_util::check_softmax_args;
use crate::kernels::portable::cpu::util::functional_util::{
    apply_unary_map_fn, apply_unary_map_reduce_fn, apply_unary_reduce_fn,
};
use crate::kernels::portable::cpu::util::reduce_util::apply_over_dim_whole;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    nonzero_dim, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ dispatches over FLOATHBF16 and derives an accumulation type
// `ACC = std::conditional_t<CTYPE is Half or BFloat16, float, CTYPE>`. Rust has
// no `std::conditional_t`, so — mirroring op_log_softmax.rs's per-type-trait
// strategy — a `SoftmaxCtype` trait carries the associated `Acc` type plus the
// per-CTYPE `std::max`, `static_cast<ACC>`, and `static_cast<CTYPE>` operations,
// and `SoftmaxAcc` supplies `std::exp`, add, sub, and div on `ACC` (always f32
// or f64).
trait SoftmaxAcc: Copy {
    fn exp(self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
}

impl SoftmaxAcc for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
}

impl SoftmaxAcc for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
}

trait SoftmaxCtype: Copy {
    type Acc: SoftmaxAcc;
    fn max(a: Self, b: Self) -> Self;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(val: Self::Acc) -> Self;
}

impl SoftmaxCtype for f32 {
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

impl SoftmaxCtype for f64 {
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

impl SoftmaxCtype for Half {
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

impl SoftmaxCtype for BFloat16 {
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

// [spec:et:def:op-softmax.torch.executor.native.softmax-out-fn]
// [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn]
#[executorch_macros::et_kernel("aten::_softmax.out")]
pub fn softmax_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    half_to_float: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    crate::et_kernel_check!(
        ctx,
        check_softmax_args(in_, dim, half_to_float, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
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
    crate::et_switch_floathbf16_types!(in_.scalar_type(), ctx, "_softmax.out", CTYPE, {
        type ACC = <CTYPE as SoftmaxCtype>::Acc;
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        apply_over_dim_whole(
            |size: usize, stride: usize, base: usize| {
                // calculate max in softmax dim. During softmax computation each
                // value is subtracted by the maximum in value before calling exp
                // to preserve numerical stability.
                let max_in: CTYPE = apply_unary_reduce_fn(
                    |val_in: CTYPE, val_accum: CTYPE| -> CTYPE {
                        <CTYPE as SoftmaxCtype>::max(val_in, val_accum)
                    },
                    unsafe { in_data.add(base) },
                    size as i64,
                    stride as i64,
                );

                let temp_sum: ACC = apply_unary_map_reduce_fn::<CTYPE, ACC, _, _>(
                    |val_in: CTYPE| -> ACC {
                        SoftmaxAcc::exp(
                            SoftmaxCtype::to_acc(val_in).sub(SoftmaxCtype::to_acc(max_in)),
                        )
                    },
                    |mapped_in: ACC, val_accum: ACC| -> ACC { val_accum.add(mapped_in) },
                    unsafe { in_data.add(base) },
                    size as i64,
                    stride as i64,
                );

                apply_unary_map_fn(
                    |val_in: CTYPE| -> CTYPE {
                        <CTYPE as SoftmaxCtype>::from_acc(
                            SoftmaxAcc::exp(
                                SoftmaxCtype::to_acc(val_in).sub(SoftmaxCtype::to_acc(max_in)),
                            )
                            .div(temp_sum),
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
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{
        internal::K_DEFAULT_ATOL, tensors_are_close,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
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

    trait FromF64 {
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

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 3],
            vec![
                T::from_f64(0.), T::from_f64(1.), T::from_f64(2.),
                T::from_f64(3.), T::from_f64(4.), T::from_f64(5.),
            ],
        );

        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                T::from_f64(0.0900306), T::from_f64(0.244728), T::from_f64(0.665241),
                T::from_f64(0.0900306), T::from_f64(0.244728), T::from_f64(0.665241),
            ],
        );

        if T::VALUE == ScalarType::BFloat16 {
            assert!(
                tensors_are_close(&out, &expected, 1e-2, Some(K_DEFAULT_ATOL)),
                "tensors are not close within tolerance"
            );
        } else {
            assert_tensor_close!(out, expected);
        }
    }

    // softmax([0,1,2]) drives all three functional_util reducers: the max via
    // apply_unary_reduce_fn, the sum-of-exp via apply_unary_map_reduce_fn, and the
    // final divide via apply_unary_map_fn; the exact [0.09,0.24,0.665] output pins
    // each reducer's accumulation.
    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    // [spec:et:sem:functional-util.torch.executor.apply-unary-reduce-fn-fn/test]
    // [spec:et:sem:functional-util.torch.executor.apply-unary-map-reduce-fn-fn/test]
    #[test]
    fn op_softmax_out_test_smoke() {
        let tff = TensorFactory::<f32>::new();
        let sizes = vec![1, 3];
        let in_ = tff.make_default(sizes.clone(), vec![0., 1., 2.]);
        let out = tff.zeros_default(sizes);

        let mut ctx = context();
        let ret = softmax_out(&mut ctx, &in_, 1, false, &out);

        assert_tensor_eq!(*ret, out);

        let expected = tff.make_default(vec![1, 3], vec![0.0900306, 0.244728, 0.665241]);

        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_half_support() {
        let tfh = TensorFactory::<Half>::new();
        let sizes = vec![1, 4];
        let in_ = tfh.ones_default(sizes.clone());
        let out = tfh.zeros_default(sizes);

        let mut ctx = context();
        let ret = softmax_out(&mut ctx, &in_, 1, false, &out);

        assert_tensor_eq!(*ret, out);

        let expected = tfh.make_default(
            vec![1, 4],
            vec![
                Half::from_f32(0.25),
                Half::from_f32(0.25),
                Half::from_f32(0.25),
                Half::from_f32(0.25),
            ],
        );

        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_all_dtypes_supported() {
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    // also verifies check_softmax_args: dim=3 on a rank-2 tensor fails the
    // tensor_has_dim check it forwards to.
    // [spec:et:sem:activation-ops-util.torch.executor.check-softmax-args-fn/test]
    #[test]
    fn op_softmax_out_test_mismatched_dimensions_dies() {
        let tff = TensorFactory::<f32>::new();

        let x = tff.make_default(vec![1, 3], vec![0., 1., 2.]);

        let out = tff.zeros_default(vec![1, 3]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, softmax_out(&mut ctx, &x, 3, false, &out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.ones_default(vec![3, 4]);

        let wrong_out = tf.zeros_default(vec![2, 10, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, softmax_out(&mut ctx, &x, 1, false, &wrong_out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_negative_dim() {
        let tf = TensorFactory::<f32>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![2, 3],
            vec![
                0., 1., 2.,
                3., 4., 5.,
            ],
        );

        let out = tf.zeros_default(vec![2, 3]);
        let out_negative_dim = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        softmax_out(&mut ctx, &x, -1, false, &out_negative_dim);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.0900306, 0.244728, 0.665241,
                0.0900306, 0.244728, 0.665241,
            ],
        );

        assert_tensor_close!(out, expected);
        assert_tensor_close!(out_negative_dim, expected);

        softmax_out(&mut ctx, &x, 0, false, &out);
        softmax_out(&mut ctx, &x, -2, false, &out_negative_dim);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.0474259, 0.0474259, 0.0474259,
                0.952574, 0.952574, 0.952574,
            ],
        );

        assert_tensor_close!(out, expected);
        assert_tensor_close!(out_negative_dim, expected);
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![0.10000000149011612f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_bfloat16_large_dim_accumulates_in_float() {
        let tf = TensorFactory::<BFloat16>::new();
        // N=512: without fp32 accumulation the exp-sum saturates at BFloat16's
        // precision limit (~256), so the output is ~1/256 instead of 1/512.
        const N: usize = 512;
        let x = tf.zeros_default(vec![1, N as i32]);
        let out = tf.zeros_default(vec![1, N as i32]);
        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        let expected = tf.full(
            vec![1, N as i32],
            BFloat16::from_f32(1.0f32 / N as f32),
            TensorShapeDynamism::STATIC,
        );
        assert!(
            tensors_are_close(&out, &expected, 1e-5, Some(1e-3)),
            "tensors are not close within tolerance"
        );
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.3893158435821533,
                0.4583776593208313,
                0.14476794004440308,
                0.44050133228302,
                0.2491583228111267,
                0.8098345994949341,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.4827413856983185,
                0.5172585844993591,
                0.426600843667984,
                0.5733991861343384,
                0.3633909821510315,
                0.6366089582443237,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    fn op_softmax_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.3893158435821533,
                0.4583776593208313,
                0.14476794004440308,
                0.44050133228302,
                0.2491583228111267,
                0.8098345994949341,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.4827413856983185,
                0.5172585844993591,
                0.426600843667984,
                0.5733991861343384,
                0.3633909821510315,
                0.6366089582443237,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUnbound` (dynamic shape unbound not
    // supported). Ported and ignored.
    // [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn/test]
    #[test]
    #[ignore]
    fn op_softmax_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.3893158435821533,
                0.4583776593208313,
                0.14476794004440308,
                0.44050133228302,
                0.2491583228111267,
                0.8098345994949341,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.4827413856983185,
                0.5172585844993591,
                0.426600843667984,
                0.5733991861343384,
                0.3633909821510315,
                0.6366089582443237,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        softmax_out(&mut ctx, &x, 1, false, &out);
        assert_tensor_close!(out, expected_result);
    }
}
