//! Literal port of kernels/portable/cpu/op_minimum.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::math_util::min_override;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). Mirror of op_maximum.
//
// PORT-NOTE (cross-module): the compile-time `op_name` template parameter
// ("minimum.out") of the C++ `apply_bitensor_elementwise_fn` is dropped — the
// ported wrapper takes no op-name argument. The C++ call omits the
// `SupportNoncontiguousInputTensors` tag, so `support_noncontiguous_tensors` is
// `false`. The compute closure takes the loaded
// inputs as a `&[CTYPE_COMPUTE]` slice (`vals[0]`, `vals[1]`), mirroring the C++
// generic-lambda `(val_a, val_b)`. `utils::get_compute_type` maps to
// `elementwise_util::get_compute_type`; `utils::min_override` to
// `math_util::min_override`.

// [spec:et:def:op-minimum.torch.executor.native.minimum-out-fn]
// [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn]
pub fn minimum_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type: ScalarType = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()),
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
    let compute_type: ScalarType = get_compute_type(&mut common_type);

    crate::et_switch_realb_types!(compute_type, ctx, "minimum.out", CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { min_override(vals[0], vals[1]) },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBBF16,
            false,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::Half;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_minimum_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        minimum_out(&mut ctx, self_, other, out)
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

    fn test_minimum_out_same_size<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        op_minimum_out(
            &tf.make_default(sizes.clone(), d(&[1., 2., 4., 8.])),
            &tf.make_default(sizes.clone(), d(&[3., 0., 4., 9.])),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, d(&[1., 0., 4., 8.])));
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_byte_tensors() {
        test_minimum_out_same_size::<u8>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_char_tensors() {
        test_minimum_out_same_size::<i8>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_short_tensors() {
        test_minimum_out_same_size::<i16>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_int_tensors() {
        test_minimum_out_same_size::<i32>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_long_tensors() {
        test_minimum_out_same_size::<i64>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_half_tensors() {
        test_minimum_out_same_size::<Half>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    // also verifies min_override (elementwise min of two f32 tensors)
    // [spec:et:sem:math-util.torch.executor.native.utils.min-override-fn/test]
    #[test]
    fn op_minimum_out_test_float_tensors() {
        test_minimum_out_same_size::<f32>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_double_tensors() {
        test_minimum_out_same_size::<f64>();
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_both_scalar_tensors() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![1, 1];
        let out = tf.zeros_default(sizes.clone());
        op_minimum_out(
            &tf.make_default(sizes.clone(), vec![1.2]),
            &tf.make_default(sizes.clone(), vec![3.5]),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, vec![1.2]));
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_left_scalar_tensor() {
        let tf = TensorFactory::<f32>::new();
        let sizes_1 = vec![1, 1];
        let sizes_2 = vec![2, 2];
        let out1 = tf.zeros_default(sizes_2.clone());
        let out2 = tf.zeros_default(sizes_2.clone());
        let a = tf.make_default(sizes_1, vec![1.0]);
        let b = tf.make_default(sizes_2.clone(), vec![3.5, -1.0, 0.0, 5.5]);
        op_minimum_out(&a, &b, &out1);
        assert_tensor_eq!(
            out1,
            tf.make_default(sizes_2.clone(), vec![1.0, -1.0, 0.0, 1.0])
        );
        op_minimum_out(&b, &a, &out2);
        assert_tensor_eq!(out2, tf.make_default(sizes_2, vec![1.0, -1.0, 0.0, 1.0]));
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_mismatched_input_shapes_dies() {
        let tf = TensorFactory::<f32>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let mut ctx = context();
        minimum_out(
            &mut ctx,
            &tf.ones_default(vec![2, 2]),
            &tf.ones_default(vec![3, 3]),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<f32>::new();
        let out = tf.zeros_default(vec![3, 3]);
        let mut ctx = context();
        minimum_out(
            &mut ctx,
            &tf.ones_default(vec![2, 2]),
            &tf.ones_default(vec![3, 3]),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_mismatched_output_shape_with_singleton_dies() {
        let tf = TensorFactory::<f32>::new();
        let out = tf.zeros_default(vec![4, 4]);
        let mut ctx = context();
        minimum_out(
            &mut ctx,
            &tf.ones_default(vec![1, 1]),
            &tf.ones_default(vec![3, 3]),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn dyn_shape_data<'a>(tf: &'a TensorFactory<f32>) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
            ],
        );
        let expected = tf.make_default(
            vec![3, 2],
            vec![
                0.4900934100151062,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.40171730518341064,
            ],
        );
        (x, y, expected)
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_minimum_out(&x, &y, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_minimum_out(&x, &y, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ DynamicShapeUnbound is ET_SKIP_IF-guarded on
    // output_resize support (unsupported in portable). Ported and #[ignore]d.
    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    #[ignore]
    fn op_minimum_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_minimum_out(&x, &y, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn/test]
    #[test]
    fn op_minimum_out_test_smoke_test_larger() {
        let tf_float = TensorFactory::<f32>::new();
        let a: Vec<f32> = (0..18).map(|i| (i as f32) - 8.0).collect();
        let self_ = tf_float.make_default(vec![18], a);
        let other = tf_float.full(vec![18], 4.0f32, TensorShapeDynamism::STATIC);
        let out = tf_float.zeros_default(vec![18]);
        let out_expected = tf_float.make_default(
            vec![18],
            vec![
                -8., -7., -6., -5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 4., 4., 4., 4., 4.,
            ],
        );
        op_minimum_out(&self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }
}
