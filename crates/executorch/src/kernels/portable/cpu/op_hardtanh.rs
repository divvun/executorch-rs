//! Literal port of kernels/portable/cpu/op_hardtanh.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::kernels::portable::cpu::util::math_util::{max_override, min_override};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `(void)ctx;` dropped. `Tensor& out` / returned `Tensor&` become
// `&'a Tensor` (interior mutation through `*mut TensorImpl`).

// [spec:et:def:op-hardtanh.torch.executor.native.hardtanh-out-fn]
// [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn]
pub fn hardtanh_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    min: &Scalar,
    max: &Scalar,
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

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    crate::et_kernel_check!(ctx, in_type == out_type, InvalidArgument, out);

    crate::et_switch_realhbf16_types!(in_type, ctx, "hardtanh.out", CTYPE, {
        let opt_min_casted = check_overflow_scalar_cast::<CTYPE>(min);
        crate::et_kernel_check!(ctx, opt_min_casted.is_some(), InvalidArgument, out);
        let min_casted = opt_min_casted.unwrap();

        let opt_max_casted = check_overflow_scalar_cast::<CTYPE>(max);
        crate::et_kernel_check!(ctx, opt_max_casted.is_some(), InvalidArgument, out);
        let max_casted = opt_max_casted.unwrap();

        apply_unary_map_fn(
            |val_in: CTYPE| -> CTYPE { min_override(max_override(val_in, min_casted), max_casted) },
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
    use crate::runtime::core::portable_type::{BFloat16, Half};

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

    // Mirrors the C++ `std::numeric_limits<CTYPE>::is_signed` branch plus the
    // per-CTYPE integer→element construction the factory needs.
    trait HardTanhElem: Copy {
        const IS_SIGNED: bool;
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_hardtanh_elem_num {
        ($($t:ty),*) => {$(impl HardTanhElem for $t {
            const IS_SIGNED: bool = <$t>::MIN != 0 as $t;
            fn from_i32(v: i32) -> Self { v as $t }
        })*};
    }
    impl_hardtanh_elem_num!(u8, i8, i16, i32, i64, f32, f64);
    impl HardTanhElem for Half {
        const IS_SIGNED: bool = true;
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl HardTanhElem for BFloat16 {
        const IS_SIGNED: bool = true;
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + HardTanhElem,
    {
        let tf = TensorFactory::<T>::new();
        let lowest_test_element: T;
        let lower_bound: T;
        if T::IS_SIGNED {
            lowest_test_element = T::from_i32(-3);
            lower_bound = T::from_i32(-2);
        } else {
            lowest_test_element = T::from_i32(0);
            lower_bound = T::from_i32(0);
        }
        let in_ = tf.make_default(
            vec![2, 2],
            vec![
                lowest_test_element,
                T::from_i32(0),
                T::from_i32(1),
                T::from_i32(100),
            ],
        );
        let out = tf.zeros_default(vec![2, 2]);

        // `lower_bound` is a CTYPE; Scalar built via integer/double per the C++
        // implicit conversion at the call site. `2` is an int Scalar.
        let min = if T::IS_SIGNED {
            Scalar::from_i64(-2)
        } else {
            Scalar::from_i64(0)
        };
        let max = Scalar::from_i64(2);

        let mut ctx = context();
        let ret = hardtanh_out(&mut ctx, &in_, &min, &max, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(
            out,
            tf.make_default(
                vec![2, 2],
                vec![lower_bound, T::from_i32(0), T::from_i32(1), T::from_i32(2)]
            )
        );
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        // Test overflow for min parameter (using valid max)
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            hardtanh_out(&mut ctx, &in_, &bad_value, &Scalar::from_double(1.0), &out)
        );

        // Test overflow for max parameter (using valid min)
        et_expect_kernel_failure!(
            ctx,
            hardtanh_out(&mut ctx, &in_, &Scalar::from_double(-1.0), &bad_value, &out)
        );
    }

    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_sanity_check() {
        // ET_FORALL_REALHBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<Half>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<BFloat16>();
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpHardTanhTest)
    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_char_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_short_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_float_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn/test]
    #[test]
    fn op_hardtanh_out_test_float_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
