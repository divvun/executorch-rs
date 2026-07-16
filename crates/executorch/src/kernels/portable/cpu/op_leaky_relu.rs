//! Literal port of kernels/portable/cpu/op_leaky_relu.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ closure body `if (val_in >= 0) return val_in; else return
// val_in * negative_slope_casted;` is monomorphized per CTYPE over
// FLOATHBF16 {Half, Float, Double, BFloat16}. Rust cannot compare `Half`/
// `BFloat16` against the integer literal `0`, so — mirroring math_util.rs's
// per-type-trait strategy — the element op is a `LeakyReluElem` trait with a
// per-type impl reproducing the same branch.
trait LeakyReluElem: Copy {
    fn leaky_relu(self, negative_slope: Self) -> Self;
}

macro_rules! impl_leaky_relu_elem_float {
    ($t:ty) => {
        impl LeakyReluElem for $t {
            fn leaky_relu(self, negative_slope: Self) -> Self {
                if self >= 0 as $t {
                    self
                } else {
                    self * negative_slope
                }
            }
        }
    };
}
impl_leaky_relu_elem_float!(f32);
impl_leaky_relu_elem_float!(f64);

macro_rules! impl_leaky_relu_elem_half {
    ($t:ty) => {
        impl LeakyReluElem for $t {
            fn leaky_relu(self, negative_slope: Self) -> Self {
                if self >= <$t>::from_f32_const(0.0) {
                    self
                } else {
                    self * negative_slope
                }
            }
        }
    };
}
impl_leaky_relu_elem_half!(Half);
impl_leaky_relu_elem_half!(BFloat16);

// [spec:et:def:op-leaky-relu.torch.executor.native.leaky-relu-out-fn]
// [spec:et:sem:op-leaky-relu.torch.executor.native.leaky-relu-out-fn]
pub fn leaky_relu_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    negative_slope: &Scalar,
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
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();
    let out_type = out.scalar_type();

    crate::et_kernel_check!(ctx, in_type == out_type, InvalidArgument, out);

    crate::et_switch_floathbf16_types!(in_type, ctx, "leaky_relu.out", CTYPE, {
        let opt_negative_slope_casted = check_overflow_scalar_cast::<CTYPE>(negative_slope);
        crate::et_kernel_check!(
            ctx,
            opt_negative_slope_casted.is_some(),
            InvalidArgument,
            out
        );
        let negative_slope_casted = opt_negative_slope_casted.unwrap();

        apply_unary_map_fn(
            |val_in: CTYPE| -> CTYPE { val_in.leaky_relu(negative_slope_casted) },
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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn test_leaky_relu_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let negative_slope = Scalar::from_double(-0.01);
        let mut ctx = context();
        let ret = leaky_relu_out(&mut ctx, &in_, &negative_slope, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, tf.ones_default(vec![2, 2]));
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        leaky_relu_out(&mut ctx, &in_, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-leaky-relu.torch.executor.native.leaky-relu-out-fn/test]
    #[test]
    fn op_leaky_relu_test_sanity_check() {
        test_leaky_relu_dtype::<Half>();
        test_leaky_relu_dtype::<f32>();
        test_leaky_relu_dtype::<f64>();
        test_leaky_relu_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-leaky-relu.torch.executor.native.leaky-relu-out-fn/test]
    #[test]
    fn op_leaky_relu_test_float_tensor_too_small_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }

    // [spec:et:sem:op-leaky-relu.torch.executor.native.leaky-relu-out-fn/test]
    #[test]
    fn op_leaky_relu_test_float_tensor_too_large_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
