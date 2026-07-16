//! Literal port of kernels/portable/cpu/op_masked_fill.cpp.

use crate::kernels::portable::cpu::scalar_utils::{extract_scalar, get_scalar_dtype};
use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, resize_to_broadcast_target_size,
};
use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::kernel_ops_util::check_masked_fill_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). The `utils::` namespaced helpers
// (`get_scalar_dtype`, `extract_scalar`) are free functions in `scalar_utils`.

// [spec:et:def:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn]
// [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn]
pub fn masked_fill_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mask: &Tensor,
    value: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_masked_fill_args(in_, mask, value, out),
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let val_type: ScalarType = get_scalar_dtype(*value);

    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(in_, mask, out) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, mask, out),
        InvalidArgument,
        out
    );

    crate::et_switch_realhbbf16_types!(in_type, ctx, "masked_fill.Scalar_out", CTYPE, {
        crate::et_switch_real_types_and!(
            Bool,
            val_type,
            ctx,
            "masked_fill.Scalar_out",
            CTYPE_VAL,
            {
                let mut value_v: CTYPE_VAL = Default::default();
                extract_scalar(*value, &mut value_v);
                let val: CTYPE = <CTYPE as StaticCast<CTYPE_VAL>>::static_cast(value_v);

                apply_binary_elementwise_fn::<CTYPE, bool, CTYPE, _>(
                    |val_in: CTYPE, val_mask: bool| -> CTYPE {
                        if val_mask { val } else { val_in }
                    },
                    in_,
                    mask,
                    out,
                );
            }
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
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromNum: Copy {
        fn from_i32(v: i32) -> Self;
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_num {
        ($($t:ty),*) => {$(impl FromNum for $t {
            fn from_i32(v: i32) -> Self { v as $t }
            fn from_f64(v: f64) -> Self { v as $t }
        })*};
    }
    impl_from_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromNum for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromNum for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn i<T: FromNum>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }
    fn f<T: FromNum>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    fn test_integer_masked_fill_scalar_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromNum,
    {
        let tf = TensorFactory::<T>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        masked_fill_scalar_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), i::<T>(&[23, 29, 31, 37])),
            &tf_bool.make_default(sizes.clone(), vec![false, true, true, false]),
            &Scalar::from_i64(71),
            &out,
        );

        assert_tensor_eq!(out, tf.make_default(sizes, i::<T>(&[23, 71, 71, 37])));
    }

    fn test_floating_point_masked_fill_scalar_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromNum,
    {
        let tf = TensorFactory::<T>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        masked_fill_scalar_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), f::<T>(&[1.1, 2.2, 4.4, 8.8])),
            &tf_bool.make_default(sizes.clone(), vec![true, false, false, true]),
            &Scalar::from_double(3.3),
            &out,
        );

        assert_tensor_close!(out, tf.make_default(sizes, f::<T>(&[3.3, 2.2, 4.4, 3.3])));
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_byte_tensors() {
        test_integer_masked_fill_scalar_out::<u8>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_char_tensors() {
        test_integer_masked_fill_scalar_out::<i8>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_short_tensors() {
        test_integer_masked_fill_scalar_out::<i16>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_int_tensors() {
        test_integer_masked_fill_scalar_out::<i32>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    // Exercises check_masked_fill_args (in/out same dtype, bool mask) in-path.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-masked-fill-args-fn/test]
    #[test]
    fn op_masked_fill_test_long_tensors() {
        test_integer_masked_fill_scalar_out::<i64>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_int_tensor_float_alpha_dies() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        // Elementwise op on an integral tensor with a floating value should fail.
        let mut ctx = context();
        masked_fill_scalar_out(
            &mut ctx,
            &tf.ones_default(sizes.clone()),
            &tf.ones_default(sizes),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ET_FORALL_FLOATHBF16_TYPES: Float, Double, Half, BFloat16.
    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_floating_point_tensors() {
        test_floating_point_masked_fill_scalar_out::<f32>();
        test_floating_point_masked_fill_scalar_out::<f64>();
        test_floating_point_masked_fill_scalar_out::<Half>();
        test_floating_point_masked_fill_scalar_out::<BFloat16>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_double_tensors() {
        test_floating_point_masked_fill_scalar_out::<f64>();
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_bool_tensors() {
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let self_ = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let mask = tf.make_default(sizes.clone(), vec![true, false, true, false]);
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_bool(true), &out);
        assert_tensor_close!(out, tf.ones_default(sizes));
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_mismatched_input_and_value_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();

        let sizes = vec![2, 2];
        let self_ = tf_byte.ones_default(sizes.clone());
        let mask = tf_char.ones_default(sizes.clone());
        let out = tf_byte.zeros_default(sizes);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_double(1.3), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_mismatched_output_dtype_dies() {
        let tf_bool = TensorFactory::<bool>::new();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();

        let sizes = vec![2, 2];
        let self_ = tf_byte.ones_default(sizes.clone());
        let mask = tf_bool.ones_default(sizes.clone());
        let out = tf_char.zeros_default(sizes);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_mismatched_mask_dtype_dies() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];
        let self_ = tf.ones_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());
        let mask = tf.ones_default(sizes);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_mismatched_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let self_ = tf.ones_default(vec![4]);
        let mask = tf_bool.ones_default(vec![2]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_broadcast_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let self_ = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let mask = tf_bool.make_default(vec![2], vec![true, false]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &self_, &mask, &Scalar::from_i64(3), &out);
        assert_tensor_close!(out, tf.make_default(vec![2, 2], vec![3, 2, 3, 8]));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let a = tf.ones_default(sizes.clone());
        let b = tf_bool.ones_default(sizes);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &a, &b, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_broadcast_dim_size_is_one_ab() {
        let tf = TensorFactory::<f32>::new();
        let bool_tf = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9701170325279236,
                0.4185227155685425,
                0.39851099252700806,
                0.8725584745407104,
                0.714692234992981,
                0.3167606592178345,
            ],
        );
        let y = bool_tf.make_default(vec![1, 2], vec![false, false]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.9701170325279236,
                0.4185227155685425,
                0.39851099252700806,
                0.8725584745407104,
                0.714692234992981,
                0.3167606592178345,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &Scalar::from_double(3.0), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_broadcast_dim_size_missing_ab() {
        let tf = TensorFactory::<f32>::new();
        let bool_tf = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9701170325279236,
                0.4185227155685425,
                0.39851099252700806,
                0.8725584745407104,
                0.714692234992981,
                0.3167606592178345,
            ],
        );
        let y = bool_tf.make_default(vec![2], vec![false, false]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.9701170325279236,
                0.4185227155685425,
                0.39851099252700806,
                0.8725584745407104,
                0.714692234992981,
                0.3167606592178345,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &Scalar::from_double(3.0), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let bool_tf = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                0.49675875902175903,
            ],
        );
        let y = bool_tf.make_default(vec![3, 2], vec![false, false, false, false, false, true]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                3.0,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &Scalar::from_double(3.0), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let bool_tf = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                0.49675875902175903,
            ],
        );
        let y = bool_tf.make_default(vec![3, 2], vec![false, false, false, false, false, true]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                3.0,
            ],
        );

        let out = tf.zeros(vec![6, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &Scalar::from_double(3.0), &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED in C++: Dynamic shape unbound not supported.
    // PORT-NOTE: ported as `#[ignore]` mirroring the `DISABLED_` gtest prefix.
    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    #[ignore]
    fn op_masked_fill_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let bool_tf = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                0.49675875902175903,
            ],
        );
        let y = bool_tf.make_default(vec![3, 2], vec![false, false, false, false, false, true]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.974706768989563,
                0.46383917331695557,
                0.050839245319366455,
                0.26296138763427734,
                0.8404526114463806,
                3.0,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &Scalar::from_double(3.0), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_broadcast_dim_size_is_one_ba() {
        let tf = TensorFactory::<f32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.38566190004348755,
                0.47776442766189575,
                0.1954779028892517,
                0.6691004633903503,
                0.6580829620361328,
                0.48968571424484253,
            ],
        );
        let y = tf_bool.make_default(vec![2], vec![false, false]);
        let z = Scalar::from_double(3.0);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.38566190004348755,
                0.47776442766189575,
                0.1954779028892517,
                0.6691004633903503,
                0.6580829620361328,
                0.48968571424484253,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &z, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn/test]
    #[test]
    fn op_masked_fill_test_broadcast_dim_size_missing_ba() {
        let tf = TensorFactory::<f32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.38566190004348755,
                0.47776442766189575,
                0.1954779028892517,
                0.6691004633903503,
                0.6580829620361328,
                0.48968571424484253,
            ],
        );
        let y = tf_bool.make_default(vec![2], vec![false, false]);
        let z = Scalar::from_double(3.0);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.38566190004348755,
                0.47776442766189575,
                0.1954779028892517,
                0.6691004633903503,
                0.6580829620361328,
                0.48968571424484253,
            ],
        );

        let out = tf.zeros_default(vec![3, 2]);
        let mut ctx = context();
        masked_fill_scalar_out(&mut ctx, &x, &y, &z, &out);
        assert_tensor_close!(out, expected_result);
    }
}
