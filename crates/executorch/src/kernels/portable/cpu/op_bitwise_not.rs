//! Literal port of kernels/portable/cpu/op_bitwise_not.cpp.

use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_integral_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`), matching the unary pattern modules.
//
// PORT-NOTE: the C++ `ET_KERNEL_CHECK_MSG(..., "Unsupported input dtype %" PRId8,
// static_cast<int8_t>(in.scalar_type()))` carries a formatted dtype arg; the
// ported `et_kernel_check_msg!` drops trailing format args (see the analogous
// note in distance_util.rs), so the dtype value is not interpolated. Unresolved
// cross-module reference (logging macro fidelity).

/// Computes the bitwise NOT of the given input tensor. The input tensor must be
/// of Integral or Boolean types. For bool tensors, it computes the logical NOT.
// [spec:et:def:op-bitwise-not.torch.executor.native.bitwise-not-out-fn]
// [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn]
#[executorch_macros::et_kernel("aten::bitwise_not.out")]
pub fn bitwise_not_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    if in_.scalar_type() == ScalarType::Bool {
        apply_unary_map_fn(
            |val_in: bool| !val_in,
            in_.const_data_ptr::<bool>(),
            out.mutable_data_ptr::<bool>(),
            in_.numel() as i64,
            1,
        );
    } else if is_integral_type(in_.scalar_type(), /*include_bool=*/ false) {
        crate::et_switch_int_types!(in_.scalar_type(), ctx, "bitwise_not.out", CTYPE, {
            apply_unary_map_fn(
                |val_in: CTYPE| !val_in,
                in_.const_data_ptr::<CTYPE>(),
                out.mutable_data_ptr::<CTYPE>(),
                in_.numel() as i64,
                1,
            );
        });
    } else {
        crate::et_kernel_check_msg!(ctx, false, InvalidArgument, out, "Unsupported input dtype");
    }

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

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: the C++ `test_bitwise_not_out<DTYPE>` general template uses data
    // {0, -1, -2, 3} -> {-1, 0, 1, -4}, valid for the signed integer types
    // (Char/Short/Int/Long). Byte and Bool have explicit template
    // specializations, ported as separate helpers below.
    fn test_bitwise_not_out_signed<T>()
    where
        T: CppTypeToScalarType
            + FactoryValue
            + core::ops::Not<Output = T>
            + core::convert::TryFrom<i32>,
        <T as core::convert::TryFrom<i32>>::Error: core::fmt::Debug,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 2];

        // Destination for the bitwise_not operator.
        let out = tf.zeros_default(sizes.clone());

        let cvt = |v: i32| -> T { T::try_from(v).unwrap() };

        // Check that it matches the expected output.
        let mut ctx = context();
        bitwise_not_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), vec![cvt(0), cvt(-1), cvt(-2), cvt(3)]),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![cvt(-1), cvt(0), cvt(1), cvt(-4)])
        );
    }

    fn test_bitwise_not_out_byte() {
        let tf = TensorFactory::<u8>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        bitwise_not_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), vec![0, 1, 2, 3]),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![255, 254, 253, 252])
        );
    }

    fn test_bitwise_not_out_bool() {
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        bitwise_not_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), vec![true, false, true, false]),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![false, true, false, true])
        );
    }

    // Unhandled output dtypes.
    fn test_bitwise_not_invalid_dtype_dies<T>()
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 5];

        let in_ = tf.ones_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        bitwise_not_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_all_int_input_output_support() {
        test_bitwise_not_out_byte();
        test_bitwise_not_out_signed::<i8>();
        test_bitwise_not_out_signed::<i16>();
        test_bitwise_not_out_signed::<i32>();
        test_bitwise_not_out_signed::<i64>();
    }

    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_bool_input_output_support() {
        test_bitwise_not_out_bool();
    }

    // Mismatched shape tests. Non-ATen kernel records a failure.
    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_mismatched_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![4]);
        let out = tf.ones_default(vec![2, 2]);

        let mut ctx = context();
        bitwise_not_out(&mut ctx, &a, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_all_float_input_dtype_dies() {
        test_bitwise_not_invalid_dtype_dies::<f32>();
        test_bitwise_not_invalid_dtype_dies::<f64>();
    }

    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_dynamic_shape_upper_bound_same_as_expected() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 2], vec![4, 9, 3, 0, 3, 9]);
        let expected = tf.make_default(vec![3, 2], vec![-5, -10, -4, -1, -4, -10]);

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        bitwise_not_out(&mut ctx, &x, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    fn op_bitwise_not_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 2], vec![4, 9, 3, 0, 3, 9]);
        let expected = tf.make_default(vec![3, 2], vec![-5, -10, -4, -1, -4, -10]);

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        bitwise_not_out(&mut ctx, &x, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `DISABLED_DynamicShapeUnbound` is disabled ("Dynamic shape
    // unbound not supported"); ported and `#[ignore]`d to match.
    // [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn/test]
    #[test]
    #[ignore = "Dynamic shape unbound not supported"]
    fn op_bitwise_not_out_test_dynamic_shape_unbound() {
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 2], vec![4, 9, 3, 0, 3, 9]);
        let expected = tf.make_default(vec![3, 2], vec![-5, -10, -4, -1, -4, -10]);

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        bitwise_not_out(&mut ctx, &x, &out);
        assert_tensor_eq!(out, expected);
    }
}
