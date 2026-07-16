//! Literal port of kernels/portable/cpu/op_sigmoid.cpp.

use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::vectorized_math::Float;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_floating_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_floating_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The C++ elementwise
// closure computes `one / (one + executorch::math::exp(-val_in))` at the compute
// type (`f32`/`f64` under `ET_SWITCH_FLOAT_TYPES`); `Float::exp` is the ported
// `executorch::math::exp`. The loaded value arrives as `vals[0]` from the
// `&[CTYPE]` slice the ported `apply_unitensor_elementwise_fn` passes.

// [spec:et:def:op-sigmoid.torch.executor.native.sigmoid-out-fn]
// [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn]
pub fn sigmoid_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(ctx, tensor_is_floating_type(out), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let mut compute_type: ScalarType = if is_floating_type(in_.scalar_type()) {
        in_.scalar_type()
    } else {
        ScalarType::Float
    };
    compute_type = get_compute_type(&mut compute_type);

    let op_name = "sigmoid.out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| {
                let val_in = vals[0];
                let one: CTYPE_COMPUTE = <CTYPE_COMPUTE as Float>::one();
                let out_val = one / (one + <CTYPE_COMPUTE as Float>::exp(-val_in));
                out_val
            },
            ctx,
            in_,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::FLOATHBF16,
            false,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_sigmoid_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        sigmoid_out(&mut ctx, self_, out)
    }

    // Local `from_f64` bridge for the element types used across the sigmoid suites.
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
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    // test_integer_sigmoid_out<DTYPE, OUTPUT_DTYPE>
    fn test_integer_sigmoid_out<DTYPE, OUT>()
    where
        DTYPE: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<DTYPE>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        op_sigmoid_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    DTYPE::from_f64(1.0),
                    DTYPE::from_f64(2.0),
                    DTYPE::from_f64(4.0),
                    DTYPE::from_f64(8.0),
                ],
            ),
            &out,
        );

        assert_tensor_close!(
            out,
            tf_out.make_default(
                sizes,
                vec![
                    OUT::from_f64(0.731059),
                    OUT::from_f64(0.880797),
                    OUT::from_f64(0.982014),
                    OUT::from_f64(0.999665),
                ],
            )
        );

        let out = tf_out.zeros_default(vec![18]);
        op_sigmoid_out(
            &tf.full(
                vec![18],
                DTYPE::from_f64(2.0),
                crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC,
            ),
            &out,
        );
        assert_tensor_close!(
            out,
            tf_out.full(
                vec![18],
                OUT::from_f64(0.880797),
                crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC
            )
        );
    }

    // test_boolean_sigmoid_out<OUTPUT_DTYPE>
    fn test_boolean_sigmoid_out<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<bool>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        op_sigmoid_out(
            &tf.make_default(sizes.clone(), vec![true, false, true, false]),
            &out,
        );

        assert_tensor_close!(
            out,
            tf_out.make_default(
                sizes,
                vec![
                    OUT::from_f64(0.731059),
                    OUT::from_f64(0.5),
                    OUT::from_f64(0.731059),
                    OUT::from_f64(0.5),
                ],
            )
        );

        let out = tf_out.zeros_default(vec![3]);
        op_sigmoid_out(&tf.make_default(vec![3], vec![true, true, true]), &out);
        assert_tensor_close!(
            out,
            tf_out.full(
                vec![3],
                OUT::from_f64(0.731059),
                crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC
            )
        );

        let out = tf_out.zeros_default(vec![3]);
        op_sigmoid_out(&tf.make_default(vec![3], vec![false, false, false]), &out);
        assert_tensor_close!(
            out,
            tf_out.full(
                vec![3],
                OUT::from_f64(0.5),
                crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC
            )
        );
    }

    // test_sigmoid_invalid_output_dtype_dies<OUTPUT_DTYPE>
    fn test_sigmoid_invalid_output_dtype_dies<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 5];

        let in_ = tf.ones_default(sizes.clone());
        let out = tf_out.zeros_default(sizes);

        let mut ctx = context();
        sigmoid_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_all_real_input_half_output_support() {
        // ET_FORALL_REALH_TYPES
        test_integer_sigmoid_out::<u8, Half>();
        test_integer_sigmoid_out::<i8, Half>();
        test_integer_sigmoid_out::<i16, Half>();
        test_integer_sigmoid_out::<i32, Half>();
        test_integer_sigmoid_out::<i64, Half>();
        test_integer_sigmoid_out::<f32, Half>();
        test_integer_sigmoid_out::<f64, Half>();
        test_integer_sigmoid_out::<Half, Half>();
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_all_real_input_float_output_support() {
        // ET_FORALL_REAL_TYPES
        test_integer_sigmoid_out::<u8, f32>();
        test_integer_sigmoid_out::<i8, f32>();
        test_integer_sigmoid_out::<i16, f32>();
        test_integer_sigmoid_out::<i32, f32>();
        test_integer_sigmoid_out::<i64, f32>();
        test_integer_sigmoid_out::<f32, f32>();
        test_integer_sigmoid_out::<f64, f32>();
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_all_real_input_double_output_support() {
        // ET_FORALL_REAL_TYPES
        test_integer_sigmoid_out::<u8, f64>();
        test_integer_sigmoid_out::<i8, f64>();
        test_integer_sigmoid_out::<i16, f64>();
        test_integer_sigmoid_out::<i32, f64>();
        test_integer_sigmoid_out::<i64, f64>();
        test_integer_sigmoid_out::<f32, f64>();
        test_integer_sigmoid_out::<f64, f64>();
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_boolean_input_float_output_support() {
        test_boolean_sigmoid_out::<f32>();
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_boolean_input_double_output_support() {
        test_boolean_sigmoid_out::<f64>();
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_mismatched_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![4]);
        let out = tf_out.ones_default(vec![2, 2]);

        let mut ctx = context();
        sigmoid_out(&mut ctx, &a, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn/test]
    #[test]
    fn op_sigmoid_out_test_all_non_float_output_d_type_dies() {
        // ET_FORALL_INT_TYPES
        test_sigmoid_invalid_output_dtype_dies::<u8>();
        test_sigmoid_invalid_output_dtype_dies::<i8>();
        test_sigmoid_invalid_output_dtype_dies::<i16>();
        test_sigmoid_invalid_output_dtype_dies::<i32>();
        test_sigmoid_invalid_output_dtype_dies::<i64>();
    }
}
