//! Literal port of kernels/portable/cpu/op_remainder.cpp.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::math_util::remainder_override;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    CppTypeToScalarType, can_cast, is_integral_type, promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-remainder.torch.executor.native.remainder-tensor-out-fn]
// [spec:et:sem:op-remainder.torch.executor.native.remainder-tensor-out-fn]
pub fn remainder_Tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && common_type != ScalarType::Bool,
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "remainder.Tensor_out";

    // PORT-NOTE: the elementwise util requires `Op: Fn`, but the C++ lambda
    // captures `div_by_zero_error` by reference and mutates it. A `Cell` gives
    // the same interior-mutation-through-a-shared-reference the C++ has.
    let div_by_zero_error = core::cell::Cell::new(false);

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a = vals[0];
                let val_b = vals[1];
                // TODO: rewrite this to be vectorization-capable.
                let mut value: CTYPE_COMPUTE = 0 as CTYPE_COMPUTE;
                if is_integral_type(
                    <CTYPE_COMPUTE as CppTypeToScalarType>::VALUE,
                    /*includeBool=*/ true,
                ) {
                    if val_b == 0 as CTYPE_COMPUTE {
                        div_by_zero_error.set(true);
                        return value;
                    }
                }
                value = remainder_override(val_a, val_b);
                value
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            false,
        );
    });

    crate::et_kernel_check_msg!(
        ctx,
        !div_by_zero_error.get(),
        InvalidArgument,
        out,
        "Remainder operation encountered integer division by zero"
    );

    out
}

// [spec:et:def:op-remainder.torch.executor.native.remainder-scalar-out-fn]
// [spec:et:sem:op-remainder.torch.executor.native.remainder-scalar-out-fn]
pub fn remainder_Scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && common_type != ScalarType::Bool,
        InvalidArgument,
        out
    );

    // Check for intergral division by zero
    crate::et_kernel_check_msg!(
        ctx,
        !(is_integral_type(common_type, true) && scalar_to::<f64>(b) == 0.0),
        InvalidArgument,
        out,
        "Remainder operation encountered integer division by zero"
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "remainder.Scalar_out";

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a = vals[0];
                // TODO: rewrite this to be vectorization-capable.
                remainder_override(val_a, val_b)
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            false,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_remainder_tensor_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        remainder_Tensor_out(&mut ctx, self_, other, out)
    }

    fn op_remainder_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        remainder_Scalar_out(&mut ctx, self_, other, out)
    }

    // The C++ OpRemainderOutTest declares op_remainder_scalar_out but has no TEST_F
    // exercising it. This focused unit test pins the Scalar path against the C++
    // semantics: integral remainder_override(46, 4) == 2, mirroring the Tensor SmokeTest.
    // [spec:et:sem:op-remainder.torch.executor.native.remainder-scalar-out-fn/test]
    // [spec:et:sem:math-util.torch.executor.native.utils.remainder-override-fn/test]
    #[test]
    fn scalar_smoke_test() {
        let tf_long = TensorFactory::<i64>::new();

        let self_ = tf_long.full(vec![2, 2], 46, TensorShapeDynamism::STATIC);
        let out = tf_long.zeros_default(vec![2, 2]);
        let out_expected = tf_long.full(vec![2, 2], 2, TensorShapeDynamism::STATIC);
        op_remainder_scalar_out(&self_, &Scalar::from_i64(4), &out);
        assert_tensor_close!(out, out_expected);
    }

    // PORT-NOTE: the C++ `SmokeTest` names its Long factory `tfDouble`; the
    // dtype is Long throughout (self/out/expected), Int for `other`.
    // [spec:et:sem:op-remainder.torch.executor.native.remainder-tensor-out-fn/test]
    // also verifies remainder_override integral overload (46 % 4 == 2)
    // [spec:et:sem:math-util.torch.executor.native.utils.remainder-override-fn/test]
    #[test]
    fn smoke_test() {
        let tf_double = TensorFactory::<i64>::new();
        let tf_long = TensorFactory::<i64>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = tf_long.full(vec![2, 2], 46, TensorShapeDynamism::STATIC);
        let other = tf_int.full(vec![2, 2], 4, TensorShapeDynamism::STATIC);
        let out = tf_double.zeros_default(vec![2, 2]);
        let out_expected = tf_double.full(vec![2, 2], 2, TensorShapeDynamism::STATIC);
        op_remainder_tensor_out(&self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }
}
