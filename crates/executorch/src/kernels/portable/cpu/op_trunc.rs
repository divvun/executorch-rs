//! Literal port of kernels/portable/cpu/op_trunc.cpp.

use crate::define_unary_ufunc_realhbf16;

// [spec:et:def:op-trunc.torch.executor.native.trunc-out-fn]
// [spec:et:sem:op-trunc.torch.executor.native.trunc-out-fn]
// C++: `DEFINE_UNARY_UFUNC_REALHBF16(trunc_out, std::trunc)`. The macro resolves
// `std::trunc` separately for the `float` and `double` call sites; the two
// overloads are supplied explicitly per pattern.rs's `DEFINE_UNARY_UFUNC_*`
// convention.
define_unary_ufunc_realhbf16!(trunc_out, |x: f32| x.trunc(), |x: f64| x.trunc());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `test_trunc_float_dtype<DTYPE>` is a helper templated over the
    // element type. The C++ `AllFloatDtypeSupport` test dispatches over
    // `ET_FORALL_FLOATHBF16_TYPES` in the non-ATen build (f32, f64, Half,
    // BFloat16). Each dtype instantiation is expanded here as a separate call.
    fn test_trunc_float_dtype<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let in_ = tf.make_default(
            vec![1, 6],
            vec![
                T::from_f64(60.5),
                T::from_f64(16.25),
                T::from_f64(-95.0),
                T::from_f64(-36.125),
                T::from_f64(19.0),
                T::from_f64(-47.75),
            ],
        );
        let out = tf.zeros_default(vec![1, 6]);
        let expected = tf.make_default(
            vec![1, 6],
            vec![
                T::from_f64(60.0),
                T::from_f64(16.0),
                T::from_f64(-95.0),
                T::from_f64(-36.0),
                T::from_f64(19.0),
                T::from_f64(-47.0),
            ],
        );

        let mut ctx = context();
        let ret = trunc_out(&mut ctx, &in_, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
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

    // [spec:et:sem:op-trunc.torch.executor.native.trunc-out-fn/test]
    #[test]
    fn op_trunc_test_all_float_dtype_support() {
        test_trunc_float_dtype::<f32>();
        test_trunc_float_dtype::<f64>();
        test_trunc_float_dtype::<Half>();
        test_trunc_float_dtype::<BFloat16>();
    }
}
