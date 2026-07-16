//! Literal port of kernels/portable/cpu/op_logit.cpp.

use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::kernels::portable::cpu::util::vectorized_math::Float;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_floating_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE: `std::optional<double> eps` -> `Option<f64>`. The per-element
// closure captures `eps` by copy, mirroring the C++ `[eps]` capture. Inside each
// `ET_SWITCH_FLOAT_TYPES` arm `CTYPE_OUT` is a concrete `f32`/`f64`; the C++
// `static_cast`s and `log` overload are reproduced via `StaticCast` and the
// `Float` trait's `log`/`one`. The clamp comparisons/assignments mix `CTYPE_OUT`
// with the `double` `eps`, matching the C++ implicit float<->double promotions
// (comparisons done in `f64`, the clamp assignment narrowed back to `CTYPE_OUT`).

// [spec:et:def:op-logit.torch.executor.native.logit-out-fn]
// [spec:et:sem:op-logit.torch.executor.native.logit-out-fn]
pub fn logit_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    eps: Option<f64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_floating_type(out), InvalidArgument, out);

    let in_type: ScalarType = in_.scalar_type();
    let out_type: ScalarType = out.scalar_type();
    crate::et_switch_realhbbf16_types!(in_type, ctx, "logit.out", CTYPE_IN, {
        crate::et_switch_float_types!(out_type, ctx, "logit.out", CTYPE_OUT, {
            apply_unary_map_fn(
                |val_in: CTYPE_IN| -> CTYPE_OUT {
                    let mut xi: CTYPE_OUT =
                        <CTYPE_OUT as StaticCast<CTYPE_IN>>::static_cast(val_in);
                    if let Some(eps_v) = eps {
                        if <f64 as StaticCast<CTYPE_OUT>>::static_cast(xi) < eps_v {
                            xi = <CTYPE_OUT as StaticCast<f64>>::static_cast(eps_v);
                        } else if <f64 as StaticCast<CTYPE_OUT>>::static_cast(xi) > 1.0 - eps_v {
                            xi = <CTYPE_OUT as StaticCast<f64>>::static_cast(1.0 - eps_v);
                        }
                    }
                    <CTYPE_OUT as StaticCast<CTYPE_OUT>>::static_cast(Float::log(
                        xi / (<CTYPE_OUT as Float>::one() - xi),
                    ))
                },
                in_.const_data_ptr::<CTYPE_IN>(),
                out.mutable_data_ptr::<CTYPE_OUT>(),
                in_.numel() as i64,
                1,
            );
        });
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
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // `static_cast<CTYPE>(...)` element builders for the various factory types.
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

    // PORT-NOTE: mirrors the general `test_integer_logit_out<DTYPE, OUTPUT_DTYPE>`
    // template. The `<Float, Float>` instantiation is overridden by a C++ template
    // specialization; that specialized body is dispatched below via a runtime
    // check on the concrete type pair.
    fn test_integer_logit_out<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromNum,
        OUT: CppTypeToScalarType + FactoryValue + FromNum,
    {
        let tf = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        logit_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), i::<IN>(&[1, 2, 4, 8])),
            Some(0.0),
            &out,
        );
        assert_tensor_close!(
            out,
            tf_out.make_default(
                sizes,
                f::<OUT>(&[f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY])
            )
        );
    }

    // C++ `template <> test_integer_logit_out<Float, Float>()` specialization.
    fn test_integer_logit_out_float_float() {
        let tf = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        logit_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), vec![0.1, 0.2, 0.4, 0.8]),
            Some(0.0),
            &out,
        );
        assert_tensor_close!(
            out,
            tf_out.make_default(sizes, vec![-2.197224, -1.386294, -0.405465, 1.3862943])
        );
    }

    fn test_integer_logit_out_eps_set<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromNum,
        OUT: CppTypeToScalarType + FactoryValue + FromNum,
    {
        let tf = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        logit_out(
            &mut ctx,
            &tf.make_default(sizes.clone(), i::<IN>(&[1, 2, 4, 8])),
            Some(0.1),
            &out,
        );

        let expected =
            tf_out.make_default(sizes, f::<OUT>(&[2.197224, 2.197224, 2.197224, 2.197224]));
        if IN::VALUE == ScalarType::Half || IN::VALUE == ScalarType::BFloat16 {
            assert!(tensors_are_close(
                &out,
                &expected,
                1e-2,
                Some(internal::K_DEFAULT_ATOL)
            ));
        } else {
            assert_tensor_close!(out, expected);
        }
    }

    fn test_logit_invalid_output_dtype_dies<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 5];
        let in_ = tf.ones_default(sizes.clone());
        let out = tf_out.zeros_default(sizes);

        let mut ctx = context();
        logit_out(&mut ctx, &in_, Some(0.0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ET_FORALL_REALHBF16_TYPES with the <Float,Float> specialization override.
    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_all_real_input_float_output_support() {
        test_integer_logit_out::<u8, f32>();
        test_integer_logit_out::<i8, f32>();
        test_integer_logit_out::<i16, f32>();
        test_integer_logit_out::<i32, f32>();
        test_integer_logit_out::<i64, f32>();
        test_integer_logit_out_float_float();
        test_integer_logit_out::<f64, f32>();
        test_integer_logit_out::<Half, f32>();
        test_integer_logit_out::<BFloat16, f32>();
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_all_real_input_double_output_support() {
        test_integer_logit_out::<u8, f64>();
        test_integer_logit_out::<i8, f64>();
        test_integer_logit_out::<i16, f64>();
        test_integer_logit_out::<i32, f64>();
        test_integer_logit_out::<i64, f64>();
        test_integer_logit_out::<f32, f64>();
        test_integer_logit_out::<f64, f64>();
        test_integer_logit_out::<Half, f64>();
        test_integer_logit_out::<BFloat16, f64>();
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_all_real_input_float_output_support_eps_set() {
        test_integer_logit_out_eps_set::<u8, f32>();
        test_integer_logit_out_eps_set::<i8, f32>();
        test_integer_logit_out_eps_set::<i16, f32>();
        test_integer_logit_out_eps_set::<i32, f32>();
        test_integer_logit_out_eps_set::<i64, f32>();
        test_integer_logit_out_eps_set::<f32, f32>();
        test_integer_logit_out_eps_set::<f64, f32>();
        test_integer_logit_out_eps_set::<Half, f32>();
        test_integer_logit_out_eps_set::<BFloat16, f32>();
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_all_real_input_double_output_support_eps_set() {
        test_integer_logit_out_eps_set::<u8, f64>();
        test_integer_logit_out_eps_set::<i8, f64>();
        test_integer_logit_out_eps_set::<i16, f64>();
        test_integer_logit_out_eps_set::<i32, f64>();
        test_integer_logit_out_eps_set::<i64, f64>();
        test_integer_logit_out_eps_set::<f32, f64>();
        test_integer_logit_out_eps_set::<f64, f64>();
        test_integer_logit_out_eps_set::<Half, f64>();
        test_integer_logit_out_eps_set::<BFloat16, f64>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_mismatched_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![4]);
        let out = tf_out.ones_default(vec![2, 2]);

        let mut ctx = context();
        logit_out(&mut ctx, &a, Some(0.0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ET_FORALL_INT_TYPES: Byte, Char, Short, Int, Long.
    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_all_non_float_output_d_type_dies() {
        test_logit_invalid_output_dtype_dies::<u8>();
        test_logit_invalid_output_dtype_dies::<i8>();
        test_logit_invalid_output_dtype_dies::<i16>();
        test_logit_invalid_output_dtype_dies::<i32>();
        test_logit_invalid_output_dtype_dies::<i64>();
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![2.1972243785858154f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        logit_out(&mut ctx, &x, Some(0.1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9622091054916382,
                0.511866569519043,
                0.15690308809280396,
                0.7423648834228516,
                0.627659797668457,
                0.4892460107803345,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                2.1972243785858154,
                0.04747522622346878,
                -1.6814535856246948,
                1.05829656124115,
                0.5221903324127197,
                -0.043022606521844864,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        logit_out(&mut ctx, &x, Some(0.1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    fn op_logit_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9622091054916382,
                0.511866569519043,
                0.15690308809280396,
                0.7423648834228516,
                0.627659797668457,
                0.4892460107803345,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                2.1972243785858154,
                0.04747522622346878,
                -1.6814535856246948,
                1.05829656124115,
                0.5221903324127197,
                -0.043022606521844864,
            ],
        );

        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        logit_out(&mut ctx, &x, Some(0.1), &out);
        assert_tensor_close!(out, expected_result);
    }

    // DISABLED in C++: Dynamic shape unbound not supported.
    // PORT-NOTE: ported as `#[ignore]` mirroring the `DISABLED_` gtest prefix.
    // [spec:et:sem:op-logit.torch.executor.native.logit-out-fn/test]
    #[test]
    #[ignore]
    fn op_logit_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9622091054916382,
                0.511866569519043,
                0.15690308809280396,
                0.7423648834228516,
                0.627659797668457,
                0.4892460107803345,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                2.1972243785858154,
                0.04747522622346878,
                -1.6814535856246948,
                1.05829656124115,
                0.5221903324127197,
                -0.043022606521844864,
            ],
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        logit_out(&mut ctx, &x, Some(0.1), &out);
        assert_tensor_close!(out, expected_result);
    }
}
