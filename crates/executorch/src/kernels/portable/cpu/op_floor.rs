//! Literal port of kernels/portable/cpu/op_floor.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBF16(floor_out, std::floor)` expands to a
// kernel fn delegating to `internal::unary_ufunc_realhbf16` (in/out same dtype).
// `std::floor` resolves to the `float` and `double` overloads; `f32::floor` /
// `f64::floor` are the concrete Rust equivalents.

// [spec:et:def:op-floor.torch.executor.native.floor-out-fn]
// [spec:et:sem:op-floor.torch.executor.native.floor-out-fn]
crate::define_unary_ufunc_realhbf16!(floor_out, |x: f32| x.floor(), |x: f64| x.floor());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_floor_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        floor_out(&mut ctx, self_, out)
    }

    // PORT-NOTE: local `from_f64` bridge for the FLOATHBF16 element types.
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

    fn test_floor_float_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };

        let in_ = tf.make_default(vec![1, 7], d(&[-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0]));
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(vec![1, 7], d(&[-3.0, -3.0, -2.0, 0.0, 1.0, 2.0, 3.0]));

        let ret = op_floor_out(&in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // PORT-NOTE: the C++ selects ET_FORALL_FLOAT_TYPES (aten) vs
    // ET_FORALL_FLOATHBF16_TYPES (non-aten); the Rust port is the non-aten
    // (`torch::executor`) branch, so the FLOATHBF16 set is used.
    // [spec:et:sem:op-floor.torch.executor.native.floor-out-fn/test]
    #[test]
    fn op_floor_test_all_float_dtype_support() {
        test_floor_float_dtype::<f32>();
        test_floor_float_dtype::<f64>();
        test_floor_float_dtype::<Half>();
        test_floor_float_dtype::<BFloat16>();
    }
}
