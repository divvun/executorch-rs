//! Literal port of kernels/portable/cpu/op_full.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped.

// [spec:et:def:op-full.torch.executor.native.full-out-fn]
// [spec:et:sem:op-full.torch.executor.native.full-out-fn]
#[executorch_macros::et_kernel("aten::full.out")]
pub fn full_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    sizes: IntArrayRef,
    fill_value: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    let out_type: ScalarType = out.scalar_type();

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, sizes) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let op_name = "full.out";

    crate::et_switch_realhbbf16_types!(out_type, ctx, op_name, CTYPE_OUT, {
        let opt_val_casted = check_overflow_scalar_cast::<CTYPE_OUT>(fill_value);
        crate::et_kernel_check!(ctx, opt_val_casted.is_some(), InvalidArgument, out);
        let val_casted = opt_val_casted.unwrap();
        let data_out = out.mutable_data_ptr::<CTYPE_OUT>();
        for i in 0..out.numel() {
            unsafe {
                *data_out.add(i as usize) = val_casted;
            }
        }
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
    use crate::runtime::core::portable_type::tensor_impl::SizesType;
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

    fn ir(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    fn test_ones_out<T>(size_int32_t: Vec<SizesType>)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let size_int64_t: Vec<i64> = size_int32_t.iter().map(|&x| x as i64).collect();
        let aref = ir(&size_int64_t);

        // Boolean Scalar
        let out = tf.zeros_default(size_int32_t.clone());
        let mut ctx = context();
        full_out(&mut ctx, aref, &Scalar::from_bool(true), &out);
        assert_tensor_eq!(out, tf.ones_default(size_int32_t.clone()));

        // Integral Scalar
        let out = tf.zeros_default(size_int32_t.clone());
        full_out(&mut ctx, aref, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, tf.ones_default(size_int32_t.clone()));

        // Floating Point Scalar
        let out = tf.zeros_default(size_int32_t.clone());
        full_out(&mut ctx, aref, &Scalar::from_double(1.0), &out);
        assert_tensor_eq!(out, tf.ones_default(size_int32_t));
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let sizes: Vec<SizesType> = vec![2, 2];
        let sizes_int64_t: Vec<i64> = sizes.iter().map(|&x| x as i64).collect();
        let aref = ir(&sizes_int64_t);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, full_out(&mut ctx, aref, &bad_value, &out));
    }

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

    // ET_FORALL_REALHBF16_TYPES(GENERATE_TEST)
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_byte_tensors() {
        generate_test::<u8>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_char_tensors() {
        generate_test::<i8>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_short_tensors() {
        generate_test::<i16>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_int_tensors() {
        generate_test::<i32>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_long_tensors() {
        generate_test::<i64>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_float_tensors() {
        generate_test::<f32>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_double_tensors() {
        generate_test::<f64>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_half_tensors() {
        generate_test::<Half>();
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_bfloat16_tensors() {
        generate_test::<BFloat16>();
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpFullOutTest)
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_char_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_short_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_float_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_float_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }

    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_half_support() {
        let tf = TensorFactory::<Half>::new();

        let sizes_int64_t_vec: Vec<i64> = vec![2, 3];
        let sizes_in32_t_vec: Vec<SizesType> = vec![2, 3];
        let sizes = ir(&sizes_int64_t_vec);

        // Boolean Scalar
        let out = tf.zeros_default(sizes_in32_t_vec.clone());
        let mut ctx = context();
        full_out(&mut ctx, sizes, &Scalar::from_bool(true), &out);
        assert_tensor_eq!(out, tf.ones_default(sizes_in32_t_vec.clone()));

        // Integral Scalar
        let out = tf.zeros_default(sizes_in32_t_vec.clone());
        full_out(&mut ctx, sizes, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, tf.ones_default(sizes_in32_t_vec.clone()));

        // Floating Point Scalar
        let out = tf.zeros_default(sizes_in32_t_vec.clone());
        full_out(&mut ctx, sizes, &Scalar::from_double(3.1415926535), &out);
        assert_tensor_eq!(
            out,
            tf.full(
                sizes_in32_t_vec,
                Half::from_f64(3.1415926535),
                crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC
            )
        );
    }

    // [spec:et:sem:op-full.torch.executor.native.full-out-fn/test]
    #[test]
    fn op_full_out_test_zero_dim() {
        let tf = TensorFactory::<Half>::new();

        let sizes_int64_t_vec: Vec<i64> = vec![];
        let sizes_in32_t_vec: Vec<SizesType> = vec![];
        let sizes = ir(&sizes_int64_t_vec);

        // Boolean Scalar
        let out = tf.zeros_default(sizes_in32_t_vec.clone());
        let mut ctx = context();
        full_out(&mut ctx, sizes, &Scalar::from_bool(true), &out);
        assert_tensor_eq!(out, tf.ones_default(sizes_in32_t_vec));
    }
}
