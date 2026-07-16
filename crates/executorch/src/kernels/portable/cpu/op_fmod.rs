//! Literal port of kernels/portable/cpu/op_fmod.cpp.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::vectorized_math::fmod as math_fmod;
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

// [spec:et:def:op-fmod.torch.executor.native.fmod-tensor-out-fn]
// [spec:et:sem:op-fmod.torch.executor.native.fmod-tensor-out-fn]
pub fn fmod_Tensor_out<'a, 'b>(
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
    let mut compute_type: ScalarType = get_compute_type(&mut common_type_mut);
    if compute_type != ScalarType::Float {
        compute_type = ScalarType::Double;
    }

    let op_name = "fmod.Tensor_out";

    // PORT-NOTE: the elementwise util requires `Op: Fn`, but the C++ lambda
    // captures `div_by_zero_error` by reference and mutates it. A `Cell` gives
    // the same interior-mutation-through-a-shared-reference the C++ has.
    let div_by_zero_error = core::cell::Cell::new(false);

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a = vals[0];
                let val_b = vals[1];
                // TODO: rewrite this to be vectorization-capable?
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
                // PORT-NOTE: C++ tensor path uses `std::fmod`; the ported
                // `vectorized_math::fmod` (`self % other`) is the same C fmod.
                value = math_fmod::<CTYPE_COMPUTE>(val_a, val_b);
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
        "Fmod operation encountered integer division by zero"
    );

    out
}

// [spec:et:def:op-fmod.torch.executor.native.fmod-scalar-out-fn]
// [spec:et:sem:op-fmod.torch.executor.native.fmod-scalar-out-fn]
pub fn fmod_Scalar_out<'a, 'b>(
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
        "Fmod operation encountered integer division by zero"
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
    let mut compute_type: ScalarType = get_compute_type(&mut common_type_mut);
    if compute_type != ScalarType::Float {
        compute_type = ScalarType::Double;
    }

    let op_name = "fmod.Scalar_out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a = vals[0];
                math_fmod::<CTYPE_COMPUTE>(val_a, val_b)
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
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
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

    fn op_fmod_tensor_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        fmod_Tensor_out(&mut ctx, self_, other, out)
    }

    fn op_fmod_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        fmod_Scalar_out(&mut ctx, self_, other, out)
    }

    // [spec:et:sem:op-fmod.torch.executor.native.fmod-tensor-out-fn/test]
    #[test]
    fn op_fmod_test_smoke_test() {
        // PORT-NOTE: the C++ names the output/expected factory `tfDouble` but
        // constructs it as `TensorFactory<ScalarType::Long>`; mirrored as i64.
        let tf_double = TensorFactory::<i64>::new();
        let tf_long = TensorFactory::<i64>::new();
        let tf_int = TensorFactory::<i32>::new();

        let self_ = tf_long.full(vec![2, 2], 46, TensorShapeDynamism::STATIC);
        let other = tf_int.full(vec![2, 2], 4, TensorShapeDynamism::STATIC);
        let out = tf_double.zeros_default(vec![2, 2]);
        let out_expected = tf_double.full(vec![2, 2], 2, TensorShapeDynamism::STATIC);
        op_fmod_tensor_out(&self_, &other, &out);
        assert!(tensors_are_close(
            &out,
            &out_expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-fmod.torch.executor.native.fmod-scalar-out-fn/test]
    #[test]
    fn op_fmod_test_scalar_smoke_test() {
        let tf_float = TensorFactory::<f32>::new();
        // std::iota(a.begin(), a.end(), -8): a = {-8, -7, ..., 9}.
        let a: Vec<f32> = (0..18).map(|i| (i as f32) - 8.0).collect();
        let self_ = tf_float.make_default(vec![18], a);
        let other = Scalar::from_i64(3);
        let out = tf_float.zeros_default(vec![18]);
        let out_expected = tf_float.make_default(
            vec![18],
            vec![
                -2., -1., -0., -2., -1., -0., -2., -1., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0.,
            ],
        );
        op_fmod_scalar_out(&self_, &other, &out);
        assert!(tensors_are_close(
            &out,
            &out_expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }
}
