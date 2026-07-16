//! Literal port of kernels/portable/cpu/op_zeros.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

type IntArrayRef = ArrayRef<i64>;

// [spec:et:def:op-zeros.torch.executor.native.check-sizes-fn]
// [spec:et:sem:op-zeros.torch.executor.native.check-sizes-fn]
fn check_sizes(size_int64_t: ArrayRef<i64>, size_int32_t: ArrayRef<SizesType>) -> bool {
    // PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to
    // `ET_CHECK_OR_RETURN_FALSE(cond, "")`; the crate's `et_log_and_return_if_false!`
    // is a module-private macro in tensor_util, so use the exported
    // `et_check_or_return_false!` with an empty message, the literal expansion.
    crate::et_check_or_return_false!(size_int64_t.size() == size_int32_t.size(), "");
    for i in 0..size_int64_t.size() {
        crate::et_check_or_return_false!((*size_int32_t.at(i) as i64) == *size_int64_t.at(i), "");
    }

    true
}

/*
 * Zero the out tensor
 *
 * zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
 */
// [spec:et:def:op-zeros.torch.executor.native.zeros-out-fn]
// [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn]
pub fn zeros_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    size: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, size) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(ctx, check_sizes(size, out.sizes()), InvalidArgument, out);

    let out_data = out.mutable_data_ptr_typed();
    if !out_data.is_null() {
        /*
         * Assuming storage is contiguous and zero tensor is indeed a string of
         * zeros
         */
        unsafe {
            core::ptr::write_bytes(out_data as *mut u8, 0, out.nbytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn test_zeros_out<T>(size_int32_t: Vec<i32>)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let sizes: Vec<i64> = size_int32_t.iter().map(|&v| v as i64).collect();
        let aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let out = tf.ones_default(size_int32_t.clone());

        let mut ctx = context();
        zeros_out(&mut ctx, aref, &out);

        crate::assert_tensor_eq!(out, tf.zeros_default(size_int32_t));
    }

    fn generate_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        test_zeros_out::<T>(vec![2, 3, 4]);
        test_zeros_out::<T>(vec![2, 0, 4]);
        test_zeros_out::<T>(vec![]);
    }

    // ET_FORALL_REAL_TYPES_AND(Bool): Byte,Char,Short,Int,Long,Float,Double,Bool

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    // also verifies check_sizes: the passed i64 size list is compared elementwise
    // against the (resized) out tensor's i32 sizes; generate_test's [2,3,4]/[2,0,4]/[]
    // cases pass the equality gate before the zero-fill runs.
    // [spec:et:sem:op-zeros.torch.executor.native.check-sizes-fn/test]
    #[test]
    fn op_zeros_out_test_byte_tensors() {
        generate_test::<u8>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_char_tensors() {
        generate_test::<i8>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_short_tensors() {
        generate_test::<i16>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_int_tensors() {
        generate_test::<i32>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_long_tensors() {
        generate_test::<i64>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_float_tensors() {
        generate_test::<f32>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_double_tensors() {
        generate_test::<f64>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_bool_tensors() {
        generate_test::<bool>();
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let expected = tf.zeros_default(vec![3, 2]);

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let out = tf.ones(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        zeros_out(&mut ctx, sizes_aref, &out);
        crate::assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    fn op_zeros_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let expected = tf.zeros_default(vec![3, 2]);

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let out = tf.ones(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        zeros_out(&mut ctx, sizes_aref, &out);
        crate::assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: the portable kernel's
    // `output_resize` SupportedFeature is false, so this test is skipped in the
    // portable build. Ported as `#[ignore]`.
    // [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_zeros_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let expected = tf.zeros_default(vec![3, 2]);

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let out = tf.ones(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        zeros_out(&mut ctx, sizes_aref, &out);
        crate::assert_tensor_eq!(out, expected);
    }
}
