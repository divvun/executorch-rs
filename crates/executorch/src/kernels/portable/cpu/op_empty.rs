//! Literal port of kernels/portable/cpu/op_empty.cpp.

use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)context;` is dropped; the
// `memory_format` argument is unused (mirrors the C++ ignore).

/*
 * Empty out tensor
 *
 * empty.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
 */
// [spec:et:def:op-empty.torch.executor.native.empty-out-fn]
// [spec:et:sem:op-empty.torch.executor.native.empty-out-fn]
pub fn empty_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    size: IntArrayRef,
    _memory_format: Option<MemoryFormat>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        context,
        resize_tensor(out, size) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

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

    fn test_empty_out<T>(size_int32_t: Vec<i32>)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let sizes: Vec<i64> = size_int32_t.iter().map(|&v| v as i64).collect();
        let aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let memory_format: Option<MemoryFormat> = None;
        let out = tf.ones_default(size_int32_t);

        let mut ctx = context();
        empty_out(&mut ctx, aref, memory_format, &out);
    }

    // GENERATE_TEST over ET_FORALL_REAL_TYPES_AND(Bool)
    fn generate_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        test_empty_out::<T>(vec![2, 3, 4]);
        test_empty_out::<T>(vec![2, 0, 4]);
        test_empty_out::<T>(vec![]);
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_byte_tensors() {
        generate_test::<u8>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_char_tensors() {
        generate_test::<i8>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_short_tensors() {
        generate_test::<i16>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_int_tensors() {
        generate_test::<i32>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_long_tensors() {
        generate_test::<i64>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_float_tensors() {
        generate_test::<f32>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_double_tensors() {
        generate_test::<f64>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_bool_tensors() {
        generate_test::<bool>();
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let memory_format: Option<MemoryFormat> = None;
        let out = tf.ones(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        empty_out(&mut ctx, sizes_aref, memory_format, &out);
    }

    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    fn op_empty_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let memory_format: Option<MemoryFormat> = None;
        let out = tf.ones(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        empty_out(&mut ctx, sizes_aref, memory_format, &out);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: portable's `output_resize`
    // SupportedFeature is false, so this test is skipped in the portable build.
    // [spec:et:sem:op-empty.torch.executor.native.empty-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_empty_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let memory_format: Option<MemoryFormat> = None;
        let out = tf.ones(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        empty_out(&mut ctx, sizes_aref, memory_format, &out);
    }
}
