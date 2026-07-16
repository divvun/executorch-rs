//! Literal port of kernels/portable/cpu/op__empty_dim_order.cpp.

use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::exec_aten::DimOrderType;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order,
};
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

type OptionalIntArrayRef = Option<ArrayRef<i64>>;
type DimOrderArrayRef = ArrayRef<DimOrderType>;

// PORT-NOTE: the C++ `ET_LOG_AND_RETURN_IF_FALSE(cond)` logs and `return
// false`s on failure. `et_log_and_return_if_false!` is a module-private macro in
// tensor_util.rs (not exported), so each check is written as an explicit early
// `return false`, preserving the same short-circuit control flow.

// [spec:et:def:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn]
// [spec:et:sem:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn]
fn _check__empty_out_dim_order(dim_order: OptionalIntArrayRef, out: &Tensor) -> bool {
    let out_dim_order: DimOrderArrayRef = out.dim_order();

    if dim_order.is_some() {
        // out tensor's dim order shall equal to input dim order
        let dim_order_ref: IntArrayRef = dim_order.unwrap();

        // PORT-NOTE: C++ `is_*_dim_order(dim_order.value().data(), size)` binds
        // the function template to the int64 `IntArrayRef` element type. The
        // ported `is_*_dim_order` is fixed to `DimOrderType` (u8), so the int64
        // dim order is first narrowed into a `[DimOrderType; K]` buffer, mirroring
        // the established conversion in copy_ops_util::check__to_dim_order_copy_args.
        // Unresolved cross-module reference: the util should be generic over the
        // dim-order element type to match the C++ template.
        let mut dim_order_bytes: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
            [0; K_TENSOR_DIMENSION_LIMIT];
        for i in 0..dim_order_ref.size() {
            dim_order_bytes[i] = *dim_order_ref.at(i) as DimOrderType;
        }

        if !(unsafe { is_channels_last_dim_order(dim_order_bytes.as_ptr(), dim_order_ref.size()) }
            || unsafe { is_contiguous_dim_order(dim_order_bytes.as_ptr(), dim_order_ref.size()) })
        {
            return false;
        }

        // Out tensor shall have same dim order as dim_order
        if !(out_dim_order.size() == dim_order_ref.size()) {
            return false;
        }
        for i in 0..dim_order_ref.size() {
            if !(*out_dim_order.at(i) as i64 == *dim_order_ref.at(i)) {
                return false;
            }
        }
    } else {
        // dim_order is not set, out tensor should be contiguous memory format
        if !unsafe { is_contiguous_dim_order(out_dim_order.data(), out_dim_order.size()) } {
            return false;
        }
    }
    true
}

/*
 * Empty out tensor with specified dim order
 *
 * _empty_dim_order.out(SymInt[] size, *, int[]? dim_order=None, Tensor(a!) out)
 * -> Tensor(a!)
 */
// [spec:et:def:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn]
// [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn]
pub fn _empty_dim_order_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    size: IntArrayRef,
    dim_order: OptionalIntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &context;

    // Check if dim_order is valid.
    // PORT-NOTE: the boolean result is intentionally discarded (not wrapped in
    // ET_KERNEL_CHECK); a failing dim-order check only logs, it does not abort.
    let _ = _check__empty_out_dim_order(dim_order, out);

    // Resize for dynamic shape.
    // PORT-NOTE: C++ `resize_tensor(out, size)` takes the int64 `IntArrayRef`
    // `size` and narrows to SizesType internally; the ported `resize_tensor<T>`
    // is generic over the passed element type, so passing `size` directly
    // instantiates `resize_tensor::<i64>` with the same per-element narrowing.
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

    fn test_op_empty_dim_order_out<T>(size_int32_t: Vec<i32>)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let sizes: Vec<i64> = size_int32_t.iter().map(|&v| v as i64).collect();
        let aref = ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len());
        let dim_order: OptionalIntArrayRef = None;
        let out = tf.ones_default(size_int32_t);

        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, aref, dim_order, &out);
    }

    fn generate_test<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        test_op_empty_dim_order_out::<T>(vec![2, 3, 4]);
        test_op_empty_dim_order_out::<T>(vec![2, 0, 4]);
        test_op_empty_dim_order_out::<T>(vec![]);
    }

    // ET_FORALL_REAL_TYPES_AND(Bool): Byte,Char,Short,Int,Long,Float,Double,Bool

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_byte_tensors() {
        generate_test::<u8>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_char_tensors() {
        generate_test::<i8>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_short_tensors() {
        generate_test::<i16>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_int_tensors() {
        generate_test::<i32>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_long_tensors() {
        generate_test::<i64>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_float_tensors() {
        generate_test::<f32>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_double_tensors() {
        generate_test::<f64>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_bool_tensors() {
        generate_test::<bool>();
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let dim_order: OptionalIntArrayRef = None;
        let out = tf.ones(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_contiguous_dim_order_succees() {
        let tf = TensorFactory::<f32>::new();

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);

        let raw_dim_order: [i64; 2] = [0, 1];
        let dim_order: OptionalIntArrayRef =
            Some(ArrayRef::from_raw_parts(raw_dim_order.as_ptr(), 2));
        let out = tf.ones(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_channels_lasts_dim_order_succees() {
        let tf = TensorFactory::<f32>::new();

        let sizes: [i64; 4] = [3, 2, 4, 5];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 4);

        let raw_dim_order: [i64; 4] = [0, 2, 3, 1];
        let dim_order: OptionalIntArrayRef =
            Some(ArrayRef::from_raw_parts(raw_dim_order.as_ptr(), 4));
        let out = tf.full_channels_last(vec![3, 2, 4, 5], 1.0, TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    fn op_empty_dim_order_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let dim_order: OptionalIntArrayRef = None;
        let out = tf.ones(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: the portable kernel's
    // `output_resize` SupportedFeature is false, so this test is skipped in the
    // portable build. Ported as `#[ignore]`.
    // [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_empty_dim_order_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let dim_order: OptionalIntArrayRef = None;
        let out = tf.ones(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    // Directly pins the pure predicate `_check__empty_out_dim_order`, whose
    // boolean result the op intentionally discards (so no op-level assertion
    // exercises it). Semantics per op__empty_dim_order.cpp.
    // [spec:et:sem:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn/test]
    #[test]
    fn check_empty_out_dim_order_predicate() {
        let tf = TensorFactory::<f32>::new();

        // None + contiguous out → true.
        let out_contig = tf.ones(vec![3, 2], TensorShapeDynamism::STATIC);
        assert!(_check__empty_out_dim_order(None, &out_contig));

        // Some(contiguous) matching a contiguous out → true.
        let d_contig: [i64; 2] = [0, 1];
        let dim_order_contig = Some(ArrayRef::from_raw_parts(d_contig.as_ptr(), 2));
        assert!(_check__empty_out_dim_order(dim_order_contig, &out_contig));

        // Some(channels_last) matching a channels_last out → true.
        let out_cl = tf.full_channels_last(vec![3, 2, 4, 5], 1.0, TensorShapeDynamism::STATIC);
        let d_cl: [i64; 4] = [0, 2, 3, 1];
        let dim_order_cl = Some(ArrayRef::from_raw_parts(d_cl.as_ptr(), 4));
        assert!(_check__empty_out_dim_order(dim_order_cl, &out_cl));

        // Some(dim_order) whose size differs from out's dim order → false.
        let d_short: [i64; 2] = [0, 1];
        let dim_order_short = Some(ArrayRef::from_raw_parts(d_short.as_ptr(), 2));
        assert!(!_check__empty_out_dim_order(dim_order_short, &out_cl));

        // Some(dim_order) that is neither contiguous nor channels_last → false.
        let d_bad: [i64; 2] = [1, 2];
        let dim_order_bad = Some(ArrayRef::from_raw_parts(d_bad.as_ptr(), 2));
        assert!(!_check__empty_out_dim_order(dim_order_bad, &out_contig));

        // None + a channels_last (non-contiguous) out → false.
        assert!(!_check__empty_out_dim_order(None, &out_cl));
    }

    // PORT-NOTE: `too_short_dim_order_die`, `illegal_dim_order_die`, and
    // `wrong_dim_order_die` are defined in the C++ fixture but never wired into a
    // TEST_F case (no macro or test references them). They are ported here as
    // non-test helpers to preserve the source, and — matching the C++
    // implementation which discards the `_check__empty_out_dim_order` result — the
    // op does not actually record a failure for a bad dim order.
    #[allow(dead_code)]
    fn too_short_dim_order_die() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 3] = [3, 2, 4];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 3);
        let raw_dim_order: [i64; 2] = [0, 1];
        let dim_order: OptionalIntArrayRef =
            Some(ArrayRef::from_raw_parts(raw_dim_order.as_ptr(), 2));
        let out = tf.ones(vec![3, 2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    #[allow(dead_code)]
    fn illegal_dim_order_die() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 2] = [3, 2];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 2);
        let raw_dim_order: [i64; 2] = [1, 2];
        let dim_order: OptionalIntArrayRef =
            Some(ArrayRef::from_raw_parts(raw_dim_order.as_ptr(), 2));
        let out = tf.ones(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }

    #[allow(dead_code)]
    fn wrong_dim_order_die() {
        let tf = TensorFactory::<f32>::new();
        let sizes: [i64; 4] = [3, 2, 4, 5];
        let sizes_aref = ArrayRef::from_raw_parts(sizes.as_ptr(), 4);
        // should be {0, 2, 3, 1}
        let raw_dim_order: [i64; 4] = [0, 1, 2, 3];
        let dim_order: OptionalIntArrayRef =
            Some(ArrayRef::from_raw_parts(raw_dim_order.as_ptr(), 4));
        let out = tf.full_channels_last(vec![3, 2, 4, 5], 1.0, TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        _empty_dim_order_out(&mut ctx, sizes_aref, dim_order, &out);
    }
}
