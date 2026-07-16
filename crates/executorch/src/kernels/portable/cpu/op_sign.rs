//! Literal port of kernels/portable/cpu/op_sign.cpp.

use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::kernels::portable::cpu::util::math_util::isnan_override;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The void-typed
// `memcpy` of the Bool branch is a byte-wise `copy_nonoverlapping` over
// `in.nbytes()`. The C++ elementwise expression
// `static_cast<CTYPE>((val_in > 0) - (val_in < 0))` relies on integer promotion
// of the two bool comparisons; reproduced here as `(gt as i32 - lt as i32)`
// cast to CTYPE via `SignFromInt`.

// PORT-NOTE: `static_cast<CTYPE>((val_in > 0) - (val_in < 0))` — the difference of
// two C++ bools is an `int` in {-1,0,1}, then `static_cast<CTYPE>`. Rust `bool`
// has no subtraction, so this trait reproduces the per-ctype `static_cast<CTYPE>`
// of that `int` value for the REALHBF16 set. For unsigned Byte, `-1` wraps to
// `255` exactly as C++ `static_cast<uint8_t>(-1)`, but the `-1` case cannot occur
// for Byte (values never `< 0`).
trait SignFromInt: Copy {
    fn sign_from_int(v: i32) -> Self;
    fn gt_zero(self) -> bool;
    fn lt_zero(self) -> bool;
}
macro_rules! impl_sign_prim {
    ($($t:ty),*) => {$(
        impl SignFromInt for $t {
            fn sign_from_int(v: i32) -> Self { v as $t }
            fn gt_zero(self) -> bool { self > (0 as $t) }
            fn lt_zero(self) -> bool { self < (0 as $t) }
        }
    )*};
}
impl_sign_prim!(u8, i8, i16, i32, i64, f32, f64);

use crate::runtime::core::portable_type::{BFloat16, Half};
macro_rules! impl_sign_half {
    ($t:ty) => {
        impl SignFromInt for $t {
            fn sign_from_int(v: i32) -> Self {
                <$t>::from_f32(v as f32)
            }
            fn gt_zero(self) -> bool {
                self.to_f32() > 0.0
            }
            fn lt_zero(self) -> bool {
                self.to_f32() < 0.0
            }
        }
    };
}
impl_sign_half!(Half);
impl_sign_half!(BFloat16);

// [spec:et:def:op-sign.torch.executor.native.sign-out-fn]
// [spec:et:sem:op-sign.torch.executor.native.sign-out-fn]
pub fn sign_out<'a, 'b>(
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
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_shape_and_dtype2(in_, out),
        InvalidArgument,
        out
    );

    if in_.scalar_type() == ScalarType::Bool {
        unsafe {
            core::ptr::copy_nonoverlapping(
                in_.const_data_ptr_typed() as *const u8,
                out.mutable_data_ptr_typed() as *mut u8,
                in_.nbytes(),
            );
        }
    } else {
        crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "sign.out", CTYPE, {
            apply_unary_map_fn(
                |val_in: CTYPE| -> CTYPE {
                    if isnan_override(val_in) {
                        val_in
                    } else {
                        <CTYPE as SignFromInt>::sign_from_int(
                            (val_in.gt_zero() as i32) - (val_in.lt_zero() as i32),
                        )
                    }
                },
                in_.const_data_ptr::<CTYPE>(),
                out.mutable_data_ptr::<CTYPE>(),
                in_.numel() as i64,
                1,
            );
        });
    }

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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_sign_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        sign_out(&mut ctx, self_, out)
    }

    // Local `from_f64` bridge (infinity/nan preserved) for the FLOATHBF16 set.
    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
        fn infinity() -> Self;
        fn neg_infinity() -> Self;
        fn nan() -> Self;
    }
    macro_rules! impl_from_f64_float {
        ($($t:ty),*) => {$(impl FromF64Elem for $t {
            fn from_f64(v: f64) -> Self { v as $t }
            fn infinity() -> Self { <$t>::INFINITY }
            fn neg_infinity() -> Self { <$t>::NEG_INFINITY }
            fn nan() -> Self { <$t>::NAN }
        })*};
    }
    impl_from_f64_float!(f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
        fn infinity() -> Self {
            Half::from_f32(f32::INFINITY)
        }
        fn neg_infinity() -> Self {
            Half::from_f32(f32::NEG_INFINITY)
        }
        fn nan() -> Self {
            Half::from_f32(f32::NAN)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn infinity() -> Self {
            BFloat16::from_f32(f32::INFINITY)
        }
        fn neg_infinity() -> Self {
            BFloat16::from_f32(f32::NEG_INFINITY)
        }
        fn nan() -> Self {
            BFloat16::from_f32(f32::NAN)
        }
    }

    // test_et_dtype<CTYPE, DTYPE>
    fn test_et_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let infinity = T::infinity();
        let nan = T::nan();
        let in_ = tf.make_default(
            vec![1, 7],
            vec![
                T::neg_infinity(),
                T::from_f64(-3.0),
                T::from_f64(-1.5),
                T::from_f64(0.0),
                T::from_f64(1.5),
                nan,
                infinity,
            ],
        );
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(
            vec![1, 7],
            vec![
                T::from_f64(-1.0),
                T::from_f64(-1.0),
                T::from_f64(-1.0),
                T::from_f64(0.0),
                T::from_f64(1.0),
                nan,
                T::from_f64(1.0),
            ],
        );

        let ret = op_sign_out(&in_, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-sign.torch.executor.native.sign-out-fn/test]
    #[test]
    fn op_sign_test_et_sanity_check_float() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_et_dtype::<f64>();
        test_et_dtype::<f32>();
        test_et_dtype::<Half>();
        test_et_dtype::<BFloat16>();
    }

    // PORT-NOTE: OpSignTest.ATenSanityCheckFloat runs only in ATen mode
    // (ET_SKIP_IF(!is_aten)); the Rust port has no ATen backend, so this test is
    // inapplicable. Ported for completeness and ignored.
    // [spec:et:sem:op-sign.torch.executor.native.sign-out-fn/test]
    #[test]
    #[ignore]
    fn op_sign_test_a_ten_sanity_check_float() {
        let tf = TensorFactory::<f32>::new();

        let in_ = tf.make_default(
            vec![1, 7],
            vec![
                f32::NEG_INFINITY,
                -3.,
                -1.5,
                0.,
                1.5,
                f32::NAN,
                f32::INFINITY,
            ],
        );
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(vec![1, 7], vec![-1., -1., -1., 0., 1., 0., 1.]);

        let ret = op_sign_out(&in_, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-sign.torch.executor.native.sign-out-fn/test]
    #[test]
    fn op_sign_test_sanity_check_bool() {
        let tf = TensorFactory::<bool>::new();

        let in_ = tf.make_default(vec![1, 6], vec![false, true, false, false, true, true]);
        let out = tf.zeros_default(vec![1, 6]);
        let expected = tf.make_default(vec![1, 6], vec![false, true, false, false, true, true]);

        let ret = op_sign_out(&in_, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_close!(out, expected);
    }
}
