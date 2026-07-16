//! Literal port of kernels/portable/cpu/op_ceil.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBF16(ceil_out, std::ceil)` expands to a
// kernel fn forwarding to `internal::unary_ufunc_realhbf16(std::ceil, std::ceil,
// ...)`. The `define_unary_ufunc_*!` macro form is sibling in-flight work in
// pattern.rs; this is the literal expansion of that macro. `std::ceil` is passed
// as both the float and double function pointers (`f32::ceil` / `f64::ceil`).

// [spec:et:def:op-ceil.torch.executor.native.ceil-out-fn]
// [spec:et:sem:op-ceil.torch.executor.native.ceil-out-fn]
pub fn ceil_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbf16(f32::ceil, f64::ceil, ctx, in_, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `test_ceil_float_dtype<DTYPE>` is a helper templated over the
    // element type. The C++ `AllFloatDtypeSupport` test dispatches over
    // `ET_FORALL_FLOATHBF16_TYPES` in the non-ATen build (f32, f64, Half,
    // BFloat16). Each dtype instantiation is expanded here as a separate call.
    fn test_ceil_float_dtype<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let in_ = tf.make_default(
            vec![1, 7],
            vec![
                T::from_f64(-3.0),
                T::from_f64(-2.99),
                T::from_f64(-1.01),
                T::from_f64(0.0),
                T::from_f64(1.01),
                T::from_f64(2.99),
                T::from_f64(3.0),
            ],
        );
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(
            vec![1, 7],
            vec![
                T::from_f64(-3.0),
                T::from_f64(-2.0),
                T::from_f64(-1.0),
                T::from_f64(0.0),
                T::from_f64(2.0),
                T::from_f64(3.0),
                T::from_f64(3.0),
            ],
        );

        let mut ctx = context();
        let ret = ceil_out(&mut ctx, &in_, &out);

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

    // [spec:et:sem:op-ceil.torch.executor.native.ceil-out-fn/test]
    // ceil_out is a thin wrapper over internal::unary_ufunc_realhbf16 (resize,
    // shape/dtype/dim-order checks, the REALHBF16 dtype switch, and the float/double
    // compute path all live in the pattern fn); this multi-dtype compute test
    // genuinely exercises it (and its pattern.h re-export).
    // [spec:et:sem:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn/test]
    // [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbf16-fn/test]
    #[test]
    fn op_ceil_test_all_float_dtype_support() {
        test_ceil_float_dtype::<f32>();
        test_ceil_float_dtype::<f64>();
        test_ceil_float_dtype::<Half>();
        test_ceil_float_dtype::<BFloat16>();
    }
}
