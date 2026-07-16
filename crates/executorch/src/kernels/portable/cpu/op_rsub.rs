//! Literal port of kernels/portable/cpu/op_rsub.cpp.

use crate::kernels::portable::cpu::scalar_utils::{
    get_scalar_dtype, promote_type_with_scalar, scalar_to,
};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::can_cast;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). The C++ elementwise closure
// `[val_b, val_alpha](const auto& val_a) { return val_b - val_alpha * val_a; }`
// receives the compute-typed value; here it is `vals[0]` from the `&[CTYPE]`
// slice the ported `apply_unitensor_elementwise_fn` passes.

// [spec:et:def:op-rsub.torch.executor.native.rsub-scalar-out-fn]
// [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn]
pub fn rsub_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let alpha_type: ScalarType = get_scalar_dtype(*alpha);

    // Check alpha type
    crate::et_kernel_check!(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    // Common Dtype
    let common_type: ScalarType = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        common_type == out.scalar_type() && can_cast(alpha_type, common_type),
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
        resize_tensor(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "rsub.Scalar_out";

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        let val_alpha: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(alpha);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| val_b - val_alpha * vals[0],
            ctx,
            a,
            SupportedTensorDtypes::REALHBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            false,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_rsub_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        alpha: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        rsub_scalar_out(&mut ctx, self_, other, alpha, out)
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    // test_integer_rsub_scalar_out<DTYPE>
    fn test_integer_rsub_scalar_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        op_rsub_scalar_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    T::from_f64(1.0),
                    T::from_f64(2.0),
                    T::from_f64(4.0),
                    T::from_f64(5.0),
                ],
            ),
            &Scalar::from_i64(10),
            &Scalar::from_i64(2),
            &out,
        );

        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![
                    T::from_f64(8.0),
                    T::from_f64(6.0),
                    T::from_f64(2.0),
                    T::from_f64(0.0),
                ],
            )
        );
    }

    // test_floating_point_rsub_scalar_out<DTYPE>
    fn test_floating_point_rsub_scalar_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        op_rsub_scalar_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    T::from_f64(1.25),
                    T::from_f64(2.25),
                    T::from_f64(4.5),
                    T::from_f64(8.875),
                ],
            ),
            &Scalar::from_double(1.0),
            &Scalar::from_i64(1),
            &out,
        );

        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![
                    T::from_f64(-0.25),
                    T::from_f64(-1.25),
                    T::from_f64(-3.5),
                    T::from_f64(-7.875),
                ],
            )
        );
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                9.007486343383789,
                8.463556289672852,
                9.823044776916504,
                9.735939025878906,
                9.385154724121094,
                8.731842994689941,
            ],
        );

        let other = Scalar::from_i64(10);
        let alpha = Scalar::from_i64(2);

        let out = tf.zeros(out_shape, dynamism);
        op_rsub_scalar_out(&x, &other, &alpha, &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_byte_tensors() {
        test_integer_rsub_scalar_out::<u8>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_char_tensors() {
        test_integer_rsub_scalar_out::<i8>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_short_tensors() {
        test_integer_rsub_scalar_out::<i16>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_int_tensors() {
        test_integer_rsub_scalar_out::<i32>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_long_tensors() {
        test_integer_rsub_scalar_out::<i64>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_int_tensor_float_alpha_dies() {
        // op_rsub_scalar_out() doesn't handle floating alpha for intergal inputs
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        rsub_scalar_out(
            &mut ctx,
            &tf.ones_default(sizes),
            &Scalar::from_i64(0),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_float_tensors() {
        test_floating_point_rsub_scalar_out::<f32>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_double_tensors() {
        test_floating_point_rsub_scalar_out::<f64>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_half_tensors() {
        test_floating_point_rsub_scalar_out::<Half>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_bfloat16_tensors() {
        test_floating_point_rsub_scalar_out::<BFloat16>();
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_unhandled_dtype_dies() {
        // op_rsub_scalar_out() doesn't handle Bool.
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);

        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        rsub_scalar_out(
            &mut ctx,
            &a,
            &Scalar::from_bool(false),
            &Scalar::from_i64(0),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // The output tensor may not have a dtype different from the input even if it
    // has the same shape.
    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_mismatched_output_dtype_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();

        let sizes = vec![2, 2];

        let a = tf_byte.ones_default(sizes.clone());

        let out = tf_char.zeros_default(sizes);

        let mut ctx = context();
        rsub_scalar_out(
            &mut ctx,
            &a,
            &Scalar::from_i64(1),
            &Scalar::from_i64(0),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let a = tf.ones_default(sizes);

        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        rsub_scalar_out(
            &mut ctx,
            &a,
            &Scalar::from_i64(1),
            &Scalar::from_i64(0),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    fn op_rsub_scalar_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards this with ET_SKIP_IF(!output_resize); the portable
    // build does not support DYNAMIC_UNBOUND resize, so the test is #[ignore]d.
    // [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn/test]
    #[test]
    #[ignore = "DynamicShapeUnbound: dynamic shape not supported"]
    fn op_rsub_scalar_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
