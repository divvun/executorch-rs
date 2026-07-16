//! Literal port of kernels/portable/cpu/op_isnan.cpp.

// [spec:et:def:op-isnan.torch.executor.native.isnan-float-fn]
// [spec:et:sem:op-isnan.torch.executor.native.isnan-float-fn]
pub fn isnan_float(x: f32) -> bool {
    x.is_nan()
}

// [spec:et:def:op-isnan.torch.executor.native.isnan-double-fn]
// [spec:et:sem:op-isnan.torch.executor.native.isnan-double-fn]
pub fn isnan_double(x: f64) -> bool {
    x.is_nan()
}

// [spec:et:def:op-isnan.torch.executor.native.isnan-out-fn]
// [spec:et:sem:op-isnan.torch.executor.native.isnan-out-fn]
crate::define_unary_ufunc_realhbbf16_to_bool!(isnan_out, isnan_float, isnan_double);

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
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // Bridges the C++ `NAN` / `std::numeric_limits<CTYPE>::infinity()` and integer
    // literals `-1, 0, 1` (implicitly converted to CTYPE) used by the test data.
    trait TestFloat: CppTypeToScalarType + FactoryValue {
        fn from_i(v: i32) -> Self;
        fn nan() -> Self;
        fn inf() -> Self;
    }
    impl TestFloat for f32 {
        fn from_i(v: i32) -> Self {
            v as f32
        }
        fn nan() -> Self {
            f32::NAN
        }
        fn inf() -> Self {
            f32::INFINITY
        }
    }
    impl TestFloat for f64 {
        fn from_i(v: i32) -> Self {
            v as f64
        }
        fn nan() -> Self {
            f64::NAN
        }
        fn inf() -> Self {
            f64::INFINITY
        }
    }
    impl TestFloat for Half {
        fn from_i(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn nan() -> Self {
            Half::NAN
        }
        fn inf() -> Self {
            Half::INFINITY
        }
    }
    impl TestFloat for BFloat16 {
        fn from_i(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn nan() -> Self {
            BFloat16::NAN
        }
        fn inf() -> Self {
            BFloat16::INFINITY
        }
    }

    fn test_sanity_check<T: TestFloat>() {
        let tf = TensorFactory::<T>::new();
        let tfb = TensorFactory::<bool>::new();

        let in_ = tf.make_default(
            vec![1, 5],
            vec![
                T::from_i(-1),
                T::from_i(0),
                T::from_i(1),
                T::nan(),
                T::inf(),
            ],
        );
        let out = tfb.zeros_default(vec![1, 5]);
        let expected = tfb.make_default(vec![1, 5], vec![false, false, false, true, false]);

        let mut ctx = context();
        let ret = isnan_out(&mut ctx, &in_, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-isnan.torch.executor.native.isnan-out-fn/test]
    // isnan_out forwards to internal::unary_ufunc_realhbbf16_to_bool; this compute
    // test exercises its resize, dim-order check, REALHBBF16 dtype switch, and the
    // per-element float/double predicate path (and its pattern.h re-export).
    // [spec:et:sem:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn/test]
    // [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn/test]
    // also verifies isnan_float (f32/Half/BFloat16 widen to f32) and isnan_double
    // (f64 input takes the double path): only the nan element must map to true.
    // [spec:et:sem:op-isnan.torch.executor.native.isnan-float-fn/test]
    // [spec:et:sem:op-isnan.torch.executor.native.isnan-double-fn/test]
    #[test]
    fn op_is_nan_test_sanity_check() {
        test_sanity_check::<f32>();
        test_sanity_check::<f64>();
        test_sanity_check::<Half>();
        test_sanity_check::<BFloat16>();
    }

    // [spec:et:sem:op-isnan.torch.executor.native.isnan-out-fn/test]
    #[test]
    fn op_is_nan_test_sanity_check_out_dtype() {
        let tf = TensorFactory::<i32>::new();

        let in_ = tf.make_default(vec![1, 5], vec![1, 2, 3, 4, 5]);
        let out = tf.zeros_default(vec![1, 5]);

        let mut ctx = context();
        isnan_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
