//! Literal port of kernels/portable/cpu/op_neg.cpp.

use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_unitensor_elementwise_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `-val_in` over REALHBF16 includes unsigned `Byte` (u8), where
// negation is two's-complement wraparound modulo 256. Rust's unary `-` is not
// defined for `u8`, so this module-local trait reproduces the C++ negation: the
// signed integer arms use `wrapping_neg` (matching C++ integer overflow wrap),
// `u8` wraps modulo 256, and the float arms flip the sign bit via `-`.
trait NegOverride: Copy {
    fn neg(self) -> Self;
}
macro_rules! impl_neg_wrapping {
    ($($t:ty),*) => {$(
        impl NegOverride for $t {
            fn neg(self) -> Self { self.wrapping_neg() }
        }
    )*};
}
impl_neg_wrapping!(u8, i8, i16, i32, i64);
macro_rules! impl_neg_float {
    ($($t:ty),*) => {$(
        impl NegOverride for $t {
            fn neg(self) -> Self { -self }
        }
    )*};
}
impl_neg_float!(f32, f64);
impl NegOverride for Half {
    fn neg(self) -> Self {
        -self
    }
}
impl NegOverride for BFloat16 {
    fn neg(self) -> Self {
        -self
    }
}

// [spec:et:def:op-neg.torch.executor.native.neg-out-fn]
// [spec:et:sem:op-neg.torch.executor.native.neg-out-fn]
pub fn neg_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
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

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let op_name = "neg.out";
    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
        apply_unitensor_elementwise_fn::<CTYPE, _>(
            |vals: &[CTYPE]| -> CTYPE { vals[0].neg() },
            ctx,
            in_,
            SupportedTensorDtypes::REALHBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            false,
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

    fn run_smoke_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };

        let in_ = tf.make_default(vec![1, 7], vv(&[-3.0, -2.5, -1.01, 0.0, 1.01, 2.5, 3.0]));
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(vec![1, 7], vv(&[3.0, 2.5, 1.01, 0.0, -1.01, -2.5, -3.0]));

        let mut ctx = context();
        let ret = neg_out(&mut ctx, &in_, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-neg.torch.executor.native.neg-out-fn/test]
    #[test]
    fn op_neg_test_smoke_test() {
        // TODO: cover all REALHBF16 types with generalized unary function test
        // harness. ET_FORALL_FLOATHBF16_TYPES.
        run_smoke_test::<f32>();
        run_smoke_test::<f64>();
        run_smoke_test::<Half>();
        run_smoke_test::<BFloat16>();
    }
}
