//! Literal port of kernels/portable/cpu/op_round.cpp.

use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_integral_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_realhbf16_type, tensors_have_same_dim_order2,
    tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// Rounds a floating point value to the closest integer. Values with a
// fractional part of exactly 0.5 are rounded to the closest even integer. Uses
// the implementation from torch/src/jit/runtime/register_ops_utils.h.
//
// PORT-NOTE: the C++ `round_to_even<CTYPE>` performs `std::floor`/`std::round`
// with the literals `0.5`/`2.0` in `double` regardless of CTYPE, then returns
// `CTYPE`. Modeled as a trait method computing in `f64` and narrowing back,
// mirroring the C++ double-arithmetic bit-for-bit. Integer instantiations are
// dead at runtime (guarded by the `isIntegralType` check) but must still exist
// for monomorphization, matching the C++ template.
// [spec:et:def:op-round.torch.executor.native.round-to-even-fn]
// [spec:et:sem:op-round.torch.executor.native.round-to-even-fn]
trait RoundToEven: Copy {
    fn round_to_even(self) -> Self;
}
macro_rules! impl_round_to_even_via_f64 {
    ($t:ty, $from:expr, $to:expr) => {
        impl RoundToEven for $t {
            fn round_to_even(self) -> Self {
                let a: f64 = $from(self);
                let r: f64 = if a - a.floor() == 0.5 {
                    (a * 0.5).round() * 2.0
                } else {
                    a.round()
                };
                $to(r)
            }
        }
    };
}
impl_round_to_even_via_f64!(u8, |x: u8| x as f64, |r: f64| r as u8);
impl_round_to_even_via_f64!(i8, |x: i8| x as f64, |r: f64| r as i8);
impl_round_to_even_via_f64!(i16, |x: i16| x as f64, |r: f64| r as i16);
impl_round_to_even_via_f64!(i32, |x: i32| x as f64, |r: f64| r as i32);
impl_round_to_even_via_f64!(i64, |x: i64| x as f64, |r: f64| r as i64);
impl_round_to_even_via_f64!(f32, |x: f32| x as f64, |r: f64| r as f32);
impl_round_to_even_via_f64!(f64, |x: f64| x, |r: f64| r);
impl_round_to_even_via_f64!(Half, |x: Half| x.to_f64(), |r: f64| Half::from_f64_const(r));
impl_round_to_even_via_f64!(BFloat16, |x: BFloat16| x.to_f64(), |r: f64| {
    BFloat16::from_f64_const(r)
});

// [spec:et:def:op-round.torch.executor.native.round-out-fn]
// [spec:et:sem:op-round.torch.executor.native.round-out-fn]
pub fn round_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_shape_and_dtype2(in_, out),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, tensor_is_realhbf16_type(out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_scalar_type = in_.scalar_type();

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "round.out", CTYPE, {
        apply_unary_map_fn(
            |val_in: CTYPE| -> CTYPE {
                if is_integral_type(in_scalar_type, /*includeBool=*/ false) {
                    val_in
                } else {
                    val_in.round_to_even()
                }
            },
            in_.const_data_ptr::<CTYPE>(),
            out.mutable_data_ptr::<CTYPE>(),
            in_.numel() as i64,
            1,
        );
    });

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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_round_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        round_out(&mut ctx, self_, out)
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

    // PORT-NOTE: local `from_f64` bridge (mirrors the op_add.rs test helper).
    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_round_execution_floats<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![11];

        let in_ = tf.make_default(
            sizes.clone(),
            [1.5, -1.5, 0.0, 1.5, 2.5, 3.5, 4.5, 1.4, -1.4, 1.7, -1.7]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );

        let out = tf.zeros_default(sizes.clone());

        op_round_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                [2.0, -2.0, 0.0, 2.0, 2.0, 4.0, 4.0, 1.0, -1.0, 2.0, -2.0]
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect()
            )
        );
    }

    fn test_round_execution_ints<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![6];

        let in_ = tf.make_default(
            sizes.clone(),
            [-1.0, 2.0, 0.0, 3.0, 0.0, -5.0]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );

        let out = tf.zeros_default(sizes.clone());

        op_round_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                [-1.0, 2.0, 0.0, 3.0, 0.0, -5.0]
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect()
            )
        );
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn float_tensors() {
        test_round_execution_floats::<f32>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn double_tensors() {
        test_round_execution_floats::<f64>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn half_tensors() {
        test_round_execution_floats::<Half>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn bfloat16_tensors() {
        test_round_execution_floats::<BFloat16>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn byte_tensors() {
        let tf = TensorFactory::<u8>::new();

        let sizes = vec![6];

        let in_ = tf.make_default(sizes.clone(), vec![1, 2, 0, 3, 0, 5]);

        let out = tf.zeros_default(sizes.clone());

        op_round_out(&in_, &out);

        assert_tensor_eq!(out, tf.make_default(sizes, vec![1, 2, 0, 3, 0, 5]));
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn char_tensors() {
        test_round_execution_ints::<i8>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn short_tensors() {
        test_round_execution_ints::<i16>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn int_tensors() {
        test_round_execution_ints::<i32>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn long_tensors() {
        test_round_execution_ints::<i64>();
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn inf_and_nan_preserved() {
        let tf = TensorFactory::<f32>::new();

        let sizes = vec![7];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![1.7, 1.4, f32::NAN, f32::INFINITY, 1.5, -1.5, 0.0],
        );

        let out = tf.zeros_default(sizes.clone());

        op_round_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![2.0, 1.0, f32::NAN, f32::INFINITY, 2.0, -2.0, 0.0],
            )
        );
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    fn unhandled_dtype_dies() {
        // round() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);

        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, round_out(&mut ctx, &a, &out));
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                -0.03743410110473633,
                2.682218074798584,
                -4.115225791931152,
                -3.6796951293945312,
                -1.925771713256836,
                1.3407869338989258,
            ],
        );
        let expected = tf.make_default(vec![3, 2], vec![-0.0, 3.0, -4.0, -4.0, -2.0, 1.0]);

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_round_out(&x, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    // [spec:et:sem:op-round.torch.executor.native.round-to-even-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                -0.03743410110473633,
                2.682218074798584,
                -4.115225791931152,
                -3.6796951293945312,
                -1.925771713256836,
                1.3407869338989258,
            ],
        );
        let expected = tf.make_default(vec![3, 2], vec![-0.0, 3.0, -4.0, -4.0, -2.0, 1.0]);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_round_out(&x, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: DISABLED in C++ (DynamicShapeUnbound: dynamic shape unbound not
    // supported); ported and `#[ignore]`d to preserve the suite.
    // [spec:et:sem:op-round.torch.executor.native.round-out-fn/test]
    #[test]
    #[ignore]
    fn dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                -0.03743410110473633,
                2.682218074798584,
                -4.115225791931152,
                -3.6796951293945312,
                -1.925771713256836,
                1.3407869338989258,
            ],
        );
        let expected = tf.make_default(vec![3, 2], vec![-0.0, 3.0, -4.0, -4.0, -2.0, 1.0]);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_round_out(&x, &out);
        assert_tensor_eq!(out, expected);
    }
}
