//! Literal port of kernels/portable/cpu/op_pow.cpp.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::vectorized_math::pow as math_pow;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-pow.torch.executor.native.pow-tensor-tensor-out-fn]
// [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn]
pub fn pow_tensor_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType = promote_types(
        a.scalar_type(),
        b.scalar_type(),
        /*half_to_float=*/ false,
    );

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
    let mut compute_type: ScalarType = get_compute_type(&mut common_type.clone());
    if compute_type != ScalarType::Float {
        compute_type = ScalarType::Double;
    }

    let op_name = "pow.Tensor_Tensor_out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { math_pow(vals[0], vals[1]) },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            /*support_noncontiguous=*/ false,
        );
    });

    out
}

// [spec:et:def:op-pow.torch.executor.native.pow-tensor-scalar-out-fn]
// [spec:et:sem:op-pow.torch.executor.native.pow-tensor-scalar-out-fn]
pub fn pow_tensor_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType =
        promote_type_with_scalar(a.scalar_type(), *b, /*half_to_float=*/ false);

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
    let mut compute_type: ScalarType = get_compute_type(&mut common_type.clone());
    if compute_type != ScalarType::Float {
        compute_type = ScalarType::Double;
    }

    let op_name = "pow.Tensor_Scalar_out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            // Casting val_b here supports vectorization; it does nothing if we
            // are not vectorizing (casts to CTYPE_COMPUTE) and casts to a
            // vectorized type otherwise.
            move |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { math_pow(vals[0], val_b) },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            /*support_noncontiguous=*/ false,
        );
    });

    out
}

// [spec:et:def:op-pow.torch.executor.native.pow-scalar-out-fn]
// [spec:et:sem:op-pow.torch.executor.native.pow-scalar-out-fn]
pub fn pow_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Scalar,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType =
        promote_type_with_scalar(b.scalar_type(), *a, /*half_to_float=*/ false);

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
        tensors_have_same_dim_order2(b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, b.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut compute_type: ScalarType = get_compute_type(&mut common_type.clone());
    if compute_type != ScalarType::Float {
        compute_type = ScalarType::Double;
    }

    let op_name = "pow.Scalar_out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_a: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(a);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            // Casting val_a here supports vectorization; it does nothing if we
            // are not vectorizing (casts to CTYPE_COMPUTE) and casts to a
            // vectorized type otherwise.
            move |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { math_pow(val_a, vals[0]) },
            ctx,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            /*support_noncontiguous=*/ false,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::Half;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_pow_scalar_out<'a, 'b>(
        self_: &Scalar,
        exponent: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        pow_scalar_out(&mut ctx, self_, exponent, out)
    }

    fn op_pow_tensor_scalar_out<'a, 'b>(
        self_: &Tensor,
        exponent: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        pow_tensor_scalar_out(&mut ctx, self_, exponent, out)
    }

    fn op_pow_tensor_tensor_out<'a, 'b>(
        self_: &Tensor,
        exponent: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        pow_tensor_tensor_out(&mut ctx, self_, exponent, out)
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn/test]
    #[test]
    fn op_pow_test_tensor_tensor_sanity_check() {
        let tf = TensorFactory::<u8>::new();
        let self_ = tf.make_default(vec![2, 2], vec![2, 2, 2, 2]);
        let exp = tf.make_default(vec![2, 1], vec![4, 4]);
        let out = tf.make_default(vec![2, 2], vec![16, 16, 16, 16]);

        let ret = op_pow_tensor_tensor_out(&self_, &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![16, 16, 16, 16]));
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn/test]
    #[test]
    fn op_pow_test_tensor_tensor_sanity_check_larger_no_broadcasting() {
        let tf = TensorFactory::<f32>::new();
        let self_ = tf.full(vec![18], 2.0, TensorShapeDynamism::STATIC);
        let exp = tf.full(vec![18], 4.0, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![18]);
        let out_expected = tf.full(vec![18], 16.0, TensorShapeDynamism::STATIC);

        let ret = op_pow_tensor_tensor_out(&self_, &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(out_expected, out);
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn/test]
    #[test]
    fn op_pow_test_tensor_tensor_sanity_check2() {
        let tf1 = TensorFactory::<f32>::new();
        let tf2 = TensorFactory::<i32>::new();
        let tf3 = TensorFactory::<f64>::new();

        let self_ = tf1.make_default(vec![2, 2], vec![2.0, 3.0, 4.0, 5.0]);
        let exp = tf2.make_default(vec![2, 1], vec![2, 2]);
        let out = tf3.zeros_default(vec![2, 2]);

        let ret = op_pow_tensor_tensor_out(&self_, &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(
            out,
            tf3.make_default(vec![2, 2], vec![4.0, 9.0, 16.0, 25.0])
        );
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn/test]
    #[test]
    fn op_pow_test_tensor_tensor_half_support() {
        let tf = TensorFactory::<Half>::new();

        let self_ = tf.make_default(
            vec![2, 2],
            vec![
                Half::from_f32(2.0),
                Half::from_f32(3.0),
                Half::from_f32(4.0),
                Half::from_f32(5.0),
            ],
        );
        let exp = tf.make_default(vec![2, 1], vec![Half::from_f32(3.0), Half::from_f32(2.0)]);
        let out = tf.zeros_default(vec![2, 2]);

        let ret = op_pow_tensor_tensor_out(&self_, &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(
            out,
            tf.make_default(
                vec![2, 2],
                vec![
                    Half::from_f32(8.0),
                    Half::from_f32(27.0),
                    Half::from_f32(16.0),
                    Half::from_f32(25.0),
                ]
            )
        );
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-scalar-out-fn/test]
    #[test]
    fn op_pow_test_tensor_scalar_sanity_check() {
        let tf = TensorFactory::<u8>::new();
        let self_ = tf.make_default(vec![2, 2], vec![2, 2, 2, 2]);
        let out = tf.make_default(vec![2, 2], vec![16, 16, 16, 16]);

        let ret = op_pow_tensor_scalar_out(&self_, &Scalar::from_i64(4), &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![16, 16, 16, 16]));
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-tensor-scalar-out-fn/test]
    #[test]
    fn op_pow_test_tensor_scalar_half_support() {
        let tf = TensorFactory::<Half>::new();
        let self_ = tf.make_default(
            vec![2, 2],
            vec![
                Half::from_f32(2.0),
                Half::from_f32(2.0),
                Half::from_f32(2.0),
                Half::from_f32(2.0),
            ],
        );
        let out = tf.zeros_default(vec![2, 2]);

        let ret = op_pow_tensor_scalar_out(&self_, &Scalar::from_i64(4), &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(
            out,
            tf.make_default(
                vec![2, 2],
                vec![
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                ]
            )
        );
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-scalar-out-fn/test]
    #[test]
    fn op_pow_test_scalar_sanity_check() {
        let tf = TensorFactory::<u8>::new();
        let exp = tf.make_default(vec![2, 2], vec![2, 2, 2, 2]);
        let out = tf.make_default(vec![2, 2], vec![16, 16, 16, 16]);

        let ret = op_pow_scalar_out(&Scalar::from_i64(4), &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![16, 16, 16, 16]));
    }

    // [spec:et:sem:op-pow.torch.executor.native.pow-scalar-out-fn/test]
    #[test]
    fn op_pow_test_scalar_half_support() {
        let tf = TensorFactory::<Half>::new();
        let exp = tf.make_default(
            vec![2, 2],
            vec![
                Half::from_f32(2.0),
                Half::from_f32(2.0),
                Half::from_f32(2.0),
                Half::from_f32(2.0),
            ],
        );
        let out = tf.zeros_default(vec![2, 2]);

        let ret = op_pow_scalar_out(&Scalar::from_i64(4), &exp, &out);

        assert_tensor_eq!(out, ret);
        assert_tensor_eq!(
            out,
            tf.make_default(
                vec![2, 2],
                vec![
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                    Half::from_f32(16.0),
                ]
            )
        );
    }
}
