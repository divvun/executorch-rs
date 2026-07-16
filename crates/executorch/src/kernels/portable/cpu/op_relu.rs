//! Literal port of kernels/portable/cpu/op_relu.cpp.

use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::kernels::portable::cpu::util::math_util::isnan_override;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_realhbf16_type, tensors_have_same_dim_order2,
    tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `CTYPE(0)` is a value-construction of the zero of the ctype; Rust
// has no single builtin covering u8..=f64, Half and BFloat16, so a local
// `ReluZero` trait supplies `ZERO` (and `PartialOrd` gives `>=`), reproducing
// `val_in >= CTYPE(0) ? val_in : CTYPE(0)`.
trait ReluZero: Copy + PartialOrd {
    const ZERO: Self;
}
macro_rules! impl_relu_zero {
    ($t:ty, $z:expr) => {
        impl ReluZero for $t {
            const ZERO: Self = $z;
        }
    };
}
impl_relu_zero!(u8, 0);
impl_relu_zero!(i8, 0);
impl_relu_zero!(i16, 0);
impl_relu_zero!(i32, 0);
impl_relu_zero!(i64, 0);
impl_relu_zero!(f32, 0.0);
impl_relu_zero!(f64, 0.0);
impl_relu_zero!(Half, Half::from_f32_const(0.0));
impl_relu_zero!(BFloat16, BFloat16::from_f32_const(0.0));

// [spec:et:def:op-relu.torch.executor.native.relu-out-fn]
// [spec:et:sem:op-relu.torch.executor.native.relu-out-fn]
#[executorch_macros::et_kernel("aten::relu.out")]
pub fn relu_out<'a, 'b>(
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

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "relu.out", CTYPE, {
        apply_unary_map_fn(
            |val_in: CTYPE| -> CTYPE {
                if isnan_override(val_in) || val_in >= CTYPE::ZERO {
                    val_in
                } else {
                    CTYPE::ZERO
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
    use crate::assert_tensor_close;
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

    fn op_relu_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        relu_out(&mut ctx, self_, out)
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

    // PORT-NOTE: local `from_f64` bridge for the element types used across the
    // relu suites (mirrors the op_add.rs test helper).
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

    fn test_relu_execution_floats<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            [-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, 0.0]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );

        let out = tf.zeros_default(sizes.clone());

        op_relu_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                [0.0, 0.2948, 0.0, 1.8690, 0.0, 0.0]
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect()
            )
        );
    }

    fn test_relu_execution_ints<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            [-1.0, 2.0, 0.0, 3.0, 0.0, -5.0]
                .iter()
                .map(|&x| T::from_f64(x))
                .collect(),
        );

        let out = tf.zeros_default(sizes.clone());

        op_relu_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                [0.0, 2.0, 0.0, 3.0, 0.0, 0.0]
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect()
            )
        );
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn float_tensors() {
        test_relu_execution_floats::<f32>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn double_tensors() {
        test_relu_execution_floats::<f64>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn half_tensors() {
        test_relu_execution_floats::<Half>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn bfloat16_tensors() {
        test_relu_execution_floats::<BFloat16>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn byte_tensors() {
        let tf = TensorFactory::<u8>::new();

        let sizes = vec![3, 2];

        let in_ = tf.make_default(sizes.clone(), vec![1, 2, 0, 3, 0, 5]);

        let out = tf.zeros_default(sizes.clone());

        op_relu_out(&in_, &out);

        assert_tensor_eq!(out, tf.make_default(sizes, vec![1, 2, 0, 3, 0, 5]));
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn char_tensors() {
        test_relu_execution_ints::<i8>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn short_tensors() {
        test_relu_execution_ints::<i16>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn int_tensors() {
        test_relu_execution_ints::<i32>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn long_tensors() {
        test_relu_execution_ints::<i64>();
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn inf_and_nan_preserved() {
        let tf = TensorFactory::<f32>::new();

        let sizes = vec![4, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![
                -0.4775,
                0.2948,
                -0.3984,
                f32::NAN,
                f32::INFINITY,
                -1.0 * f32::INFINITY,
                0.3,
                -0.4848,
            ],
        );

        let out = tf.zeros_default(sizes.clone());

        op_relu_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![0.0, 0.2948, 0.0, f32::NAN, f32::INFINITY, 0.0, 0.3, 0.0],
            )
        );
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn unhandled_dtype_dies() {
        // relu() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);

        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, relu_out(&mut ctx, &a, &out));
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn upper_bound_out_tensor() {
        let tf = TensorFactory::<f32>::new();

        let sizes = vec![3, 2];

        let in_ = tf.make_default(
            sizes.clone(),
            vec![-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, 0.0],
        );

        let out = tf.zeros(vec![5, 7], TensorShapeDynamism::DYNAMIC_BOUND);

        op_relu_out(&in_, &out);

        assert_tensor_eq!(
            out,
            tf.make_default(sizes, vec![0.0, 0.2948, 0.0, 1.8690, 0.0, 0.0])
        );
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![1.0; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        op_relu_out(&x, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_relu_out(&x, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    fn dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_relu_out(&x, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED in C++ (DynamicShapeUnbound: unbound dynamic shape not
    // supported); ported and `#[ignore]`d to preserve the suite.
    // [spec:et:sem:op-relu.torch.executor.native.relu-out-fn/test]
    #[test]
    #[ignore]
    fn dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.676039457321167,
                0.06196027994155884,
                0.36154472827911377,
                0.7953161001205444,
                0.7633233070373535,
                0.5809110999107361,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_relu_out(&x, &out);
        assert_tensor_close!(out, expected_result);
    }
}
