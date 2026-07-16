//! Literal port of kernels/portable/cpu/op_ones.cpp.

use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` — the
// non-owning handle mutates its impl through the raw pointer, matching the
// established interior-mutation pattern (see the unary ufunc patterns).

// [spec:et:def:op-ones.torch.executor.native.ones-out-fn]
// [spec:et:sem:op-ones.torch.executor.native.ones-out-fn]
pub fn ones_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    size: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    // Resize for dynamic shape
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, size) == Error::Ok,
        InvalidArgument,
        out
    );

    let out_type: ScalarType = out.scalar_type();
    crate::et_switch_realhbbf16_types!(out_type, ctx, "ones.out", CTYPE, {
        let out_data = out.mutable_data_ptr::<CTYPE>();
        for i in 0..out.numel() {
            unsafe {
                *out_data.add(i as usize) = <CTYPE as FromI32>::from_i32(1);
            }
        }
    });

    out
}

// PORT-NOTE: C++ `static_cast<CTYPE>(1)` over the REALHBBF16 ctype set (integers,
// Bool, Half, BFloat16, Float, Double). Rust has no single `as`-style trait over
// all these types, so a local `FromI32` reproduces the `static_cast<CTYPE>(1)`
// conversion per ctype (bool: 1 -> true; Half/BFloat16 via their from-f32).
trait FromI32 {
    fn from_i32(v: i32) -> Self;
}
macro_rules! impl_from_i32 {
    ($($t:ty),*) => {$(
        impl FromI32 for $t {
            fn from_i32(v: i32) -> Self { v as $t }
        }
    )*};
}
impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
impl FromI32 for bool {
    fn from_i32(v: i32) -> Self {
        v != 0
    }
}
impl FromI32 for crate::runtime::core::portable_type::Half {
    fn from_i32(v: i32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32_const(v as f32)
    }
}
impl FromI32 for crate::runtime::core::portable_type::BFloat16 {
    fn from_i32(v: i32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32_const(v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::array_ref::ArrayRef;
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

    fn test_ones_out<T>(size_int32_t: Vec<i32>)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let size_int64_t: Vec<i64> = size_int32_t.iter().map(|&v| v as i64).collect();
        let aref = ArrayRef::from_raw_parts(size_int64_t.as_ptr(), size_int64_t.len());

        // Before: `out` consists of 0s.
        let out = tf.zeros_default(size_int32_t.clone());

        // After: `out` consists of 1s.
        let mut ctx = context();
        ones_out(&mut ctx, aref, &out);

        assert_tensor_eq!(out, tf.ones_default(size_int32_t));
    }

    // GENERATE_TEST over ET_FORALL_REALHBBF16_TYPES
    fn generate_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        test_ones_out::<T>(vec![]);
        test_ones_out::<T>(vec![1]);
        test_ones_out::<T>(vec![1, 1, 1]);
        test_ones_out::<T>(vec![2, 0, 4]);
        test_ones_out::<T>(vec![2, 3, 4]);
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_byte_tensors() {
        generate_test::<u8>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_char_tensors() {
        generate_test::<i8>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_short_tensors() {
        generate_test::<i16>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_int_tensors() {
        generate_test::<i32>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_long_tensors() {
        generate_test::<i64>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_float_tensors() {
        generate_test::<f32>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_double_tensors() {
        generate_test::<f64>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_bool_tensors() {
        generate_test::<bool>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_half_tensors() {
        generate_test::<crate::runtime::core::portable_type::Half>();
    }

    // [spec:et:sem:op-ones.torch.executor.native.ones-out-fn/test]
    #[test]
    fn op_ones_out_test_b_float16_tensors() {
        generate_test::<crate::runtime::core::portable_type::BFloat16>();
    }
}
