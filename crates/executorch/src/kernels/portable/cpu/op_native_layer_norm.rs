//! Literal port of kernels/portable/cpu/op_native_layer_norm.cpp.

use crate::kernels::portable::cpu::util::normalization_ops_util::{
    LayerNormCtype, check_layer_norm_args, get_layer_norm_out_target_size, layer_norm_scalar,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, resize_tensor,
    tensor_is_default_dim_order, tensors_have_same_dim_order2, tensors_have_same_dim_order4,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out/mean/rstd` and the returned `std::tuple<Tensor&,
// Tensor&, Tensor&>` become `&'a Tensor` handles and a Rust 3-tuple.
// `const optional<Tensor>&` maps to `Option<&Tensor>`. `IntArrayRef` maps to the
// `IntArrayRef = ArrayRef<i64>` alias. The templated anonymous-namespace worker
// `layer_norm<CTYPE>` becomes a generic `fn layer_norm<CTYPE: LayerNormCtype>`.
//
// PORT-NOTE (cross-module): the ported `layer_norm_scalar` takes `eps: f32`
// (the C++ signature takes `CTYPE eps` and does `variance + eps` in float). The
// C++ narrows `double eps` -> CTYPE at the `layer_norm<CTYPE>` call and again at
// the `layer_norm_scalar` call; here the outer call narrows `eps: f64` ->
// `CTYPE::from_f64`, then `CTYPE::to_f32` feeds the util. Not redesigning the
// util (noted for the fixer if the double->CTYPE->f32 rounding matters).

// [spec:et:def:op-native-layer-norm.torch.executor.native.layer-norm-fn]
// [spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn]
fn layer_norm<CTYPE: LayerNormCtype>(
    input: &Tensor,
    normalized_shape: IntArrayRef,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: CTYPE,
    out: &Tensor,
    mean: &Tensor,
    rstd: &Tensor,
) {
    let dim: usize = (input.dim() as usize) - normalized_shape.size();
    let dim_size: usize = input.size(dim as _) as usize;

    let leading: usize = getLeadingDims(input, dim as i64);
    let normalized: usize = getTrailingDims(input, dim as i64) * dim_size;

    if leading == 0 {
        return;
    }

    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
    let mean_data: *mut CTYPE = mean.mutable_data_ptr::<CTYPE>();
    let rstd_data: *mut CTYPE = rstd.mutable_data_ptr::<CTYPE>();

    if normalized == 0 {
        for i in 0..leading {
            unsafe {
                *mean_data.add(i) = CTYPE::from_i32(0);
                *rstd_data.add(i) = CTYPE::from_f32(f32::NAN);
            }
        }
        return;
    }

    let input_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let weight_data: *const CTYPE = match weight {
        Some(w) => w.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };
    let bias_data: *const CTYPE = match bias {
        Some(b) => b.const_data_ptr::<CTYPE>(),
        None => core::ptr::null(),
    };

    unsafe {
        layer_norm_scalar::<CTYPE>(
            input_data,
            weight_data,
            bias_data,
            out_data,
            mean_data,
            rstd_data,
            leading,
            normalized,
            CTYPE::to_f32(eps),
        );
    }
}

// [spec:et:def:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn]
// [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn native_layer_norm_out<'a, 'b, 'c, 'd>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    normalized_shape: IntArrayRef,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
    out: &'a Tensor<'b>,
    mean_out: &'a Tensor<'c>,
    rstd_out: &'a Tensor<'d>,
) -> (&'a Tensor<'b>, &'a Tensor<'c>, &'a Tensor<'d>) {
    // (void)ctx;

    let ret_val = (out, mean_out, rstd_out);

    crate::et_kernel_check!(
        ctx,
        check_layer_norm_args(
            input,
            normalized_shape,
            weight,
            bias,
            out,
            mean_out,
            rstd_out
        ),
        InvalidArgument,
        ret_val
    );

    // Only support default dim order for now.
    // TODO: Support other dim orders.
    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(input),
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order4(input, out, mean_out, rstd_out),
        InvalidArgument,
        ret_val
    );

    if let Some(weight) = weight {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, weight),
            InvalidArgument,
            ret_val
        );
    }

    if let Some(bias) = bias {
        crate::et_kernel_check!(
            ctx,
            tensors_have_same_dim_order2(input, bias),
            InvalidArgument,
            ret_val
        );
    }

    let mut mean_rstd_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut mean_rstd_ndim: usize = 0;
    unsafe {
        get_layer_norm_out_target_size(
            input,
            normalized_shape,
            mean_rstd_sizes.as_mut_ptr(),
            &mut mean_rstd_ndim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            mean_out,
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), mean_rstd_ndim)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            rstd_out,
            ArrayRef::from_raw_parts(mean_rstd_sizes.as_ptr(), mean_rstd_ndim)
        ) == Error::Ok,
        InvalidArgument,
        ret_val
    );

    crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, "native_layer_norm.out", CTYPE, {
        layer_norm::<CTYPE>(
            input,
            normalized_shape,
            weight,
            bias,
            <CTYPE as LayerNormCtype>::from_f64(eps),
            out,
            mean_out,
            rstd_out,
        );
    });

    ret_val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::internal::{
        K_DEFAULT_BFLOAT16_ATOL, K_DEFAULT_HALF_ATOL,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_close_with_tol};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    // For the Int death-test case (integer input dtype, which the kernel rejects).
    impl FromF64 for i32 {
        fn from_f64(v: f64) -> Self {
            v as i32
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    struct TestCase {
        sizes: Vec<i32>,
        input_data: Vec<f64>,
        normalized_shape: Vec<i32>,
        weight_data: Vec<f64>,
        bias_data: Vec<f64>,
        eps: f64,
        expected_data: Vec<f64>,
    }

    // run_test_cases<DTYPE>
    fn run_test_cases<T>(test_cases: Vec<TestCase>)
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };
        for tc in &test_cases {
            let in_ = tf.make_default(tc.sizes.clone(), vv(&tc.input_data));
            let weight = tf.make_default(tc.normalized_shape.clone(), vv(&tc.weight_data));
            let bias = tf.make_default(tc.normalized_shape.clone(), vv(&tc.bias_data));
            let out0 = tf.zeros_default(tc.sizes.clone());
            let out1 = tf.zeros(tc.sizes.clone(), TensorShapeDynamism::DYNAMIC_BOUND);
            let out2 = tf.zeros(tc.sizes.clone(), TensorShapeDynamism::DYNAMIC_BOUND);
            let normalized_shape_vec: Vec<i64> =
                tc.normalized_shape.iter().map(|&x| x as i64).collect();
            let normalized_shape = IntArrayRef::from_raw_parts(
                normalized_shape_vec.as_ptr(),
                normalized_shape_vec.len(),
            );
            let mut ctx = context();
            let result = native_layer_norm_out(
                &mut ctx,
                &in_,
                normalized_shape,
                Some(&weight),
                Some(&bias),
                tc.eps,
                &out0,
                &out1,
                &out2,
            );
            assert_tensor_close!(out0, *result.0);

            let expected = tf.make_default(tc.sizes.clone(), vv(&tc.expected_data));
            if T::VALUE == ScalarType::BFloat16 {
                assert_tensor_close_with_tol!(out0, expected, 1e-2, K_DEFAULT_BFLOAT16_ATOL);
            } else if T::VALUE == ScalarType::Half {
                assert_tensor_close_with_tol!(out0, expected, 1e-3, K_DEFAULT_HALF_ATOL);
            } else {
                assert_tensor_close!(out0, expected);
            }
        }
    }

    fn run_floating_point_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        let test_cases = vec![
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![1.0, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0],
                eps: 1.0e-5,
                expected_data: vec![1.22474, 0.0, -1.22474, -0.925819, 1.38873, -0.46291],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![1.0, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0],
                eps: 1.0e-3,
                expected_data: vec![1.22383, 0.0, -1.22383, -0.925721, 1.38858, -0.46286],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![2.0, 2.0, 2.0],
                bias_data: vec![0.0, 0.0, 0.0],
                eps: 1.0e-5,
                expected_data: vec![2.44947, 0.0, -2.44947, -1.85164, 2.77746, -0.925819],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![1.0, 1.0, 1.0],
                bias_data: vec![1.0, 1.0, 1.0],
                eps: 1.0e-5,
                expected_data: vec![2.22474, 1.0, -0.224736, 0.0741809, 2.38873, 0.53709],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![inf, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![1.0, 1.0, 1.0],
                bias_data: vec![1.0, 1.0, 1.0],
                eps: 1.0e-5,
                expected_data: vec![-nan, -nan, -nan, 0.0741809, 2.38873, 0.53709],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![nan, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![1.0, 1.0, 1.0],
                bias_data: vec![1.0, 1.0, 1.0],
                eps: 1.0e-5,
                expected_data: vec![-nan, -nan, -nan, 0.0741809, 2.38873, 0.53709],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![nan, 1.0, 1.0],
                bias_data: vec![1.0, 1.0, 1.0],
                eps: 1.0e-5,
                expected_data: vec![nan, 1.0, -0.224736, nan, 2.38873, 0.53709],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![inf, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0],
                eps: 1.0e-5,
                expected_data: vec![inf, 0.0, -1.22474, -inf, 1.38873, -0.46291],
            },
            TestCase {
                sizes: vec![2, 3],
                input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
                normalized_shape: vec![3],
                weight_data: vec![inf, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0],
                eps: 1.0e-5,
                expected_data: vec![inf, 0.0, -1.22474, -inf, 1.38873, -0.46291],
            },
            TestCase {
                sizes: vec![1, 2, 3],
                input_data: vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
                normalized_shape: vec![1, 2, 3],
                weight_data: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                bias_data: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                eps: 1.0e-5,
                expected_data: vec![-1.46385, -0.87831, -0.29277, 0.29277, 0.87831, 1.46385],
            },
        ];
        run_test_cases::<T>(test_cases);
    }

    // run_death_test_cases: builds weight/bias only when data is non-empty.
    fn run_death_test_cases<T>(test_cases: Vec<TestCase>)
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();
        let vv = |vals: &[f64]| -> Vec<T> { vals.iter().map(|&x| T::from_f64(x)).collect() };
        for tc in &test_cases {
            let in_ = tf.make_default(tc.sizes.clone(), vv(&tc.input_data));
            let weight = if !tc.weight_data.is_empty() {
                Some(tf.make_default(tc.normalized_shape.clone(), vv(&tc.weight_data)))
            } else {
                None
            };
            let bias = if !tc.bias_data.is_empty() {
                Some(tf.make_default(tc.normalized_shape.clone(), vv(&tc.bias_data)))
            } else {
                None
            };
            let out0 = tf.zeros_default(tc.sizes.clone());
            let out1 = tf.zeros_default(tc.sizes.clone());
            let out2 = tf.zeros_default(tc.sizes.clone());
            let normalized_shape_vec: Vec<i64> =
                tc.normalized_shape.iter().map(|&x| x as i64).collect();
            let normalized_shape = IntArrayRef::from_raw_parts(
                normalized_shape_vec.as_ptr(),
                normalized_shape_vec.len(),
            );
            let mut ctx = context();
            native_layer_norm_out(
                &mut ctx,
                &in_,
                normalized_shape,
                weight.as_ref(),
                bias.as_ref(),
                tc.eps,
                &out0,
                &out1,
                &out2,
            );
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // run_int_test_cases<Int>
    fn run_int_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let test_cases = vec![TestCase {
            // Cannot be represented by a type other than float.
            sizes: vec![2, 3],
            input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
            normalized_shape: vec![3],
            weight_data: vec![1.0, 1.0, 1.0],
            bias_data: vec![0.0, 0.0, 0.0],
            eps: 1.0,
            expected_data: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }];
        run_death_test_cases::<T>(test_cases);
    }

    // run_wrong_shape_test_cases<Float>
    fn run_wrong_shape_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let test_cases = vec![TestCase {
            sizes: vec![2, 3],
            input_data: vec![1.0, 0.0, -1.0, -1.0, 4.0, 0.0],
            normalized_shape: vec![1], // wrong normalized shape
            weight_data: vec![1.0],
            bias_data: vec![0.0],
            eps: 1.0e-5,
            expected_data: vec![1.22474, 0.0, -1.22474, -0.925819, 1.38873, -0.46291],
        }];
        run_death_test_cases::<T>(test_cases);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();

        let input = tf.make_default(
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
        let weight = tf.make_default(vec![3], vec![1.0, 1.0, 1.0]);
        let bias = tf.make_default(vec![3], vec![0.0, 0.0, 0.0]);
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.16205203533172607,
                1.1355723142623901,
                -1.2976245880126953,
                -1.0853172540664673,
                -0.24233698844909668,
                1.3276543617248535,
            ],
        );
        let out0 = tf.zeros(out_shape.clone(), dynamism);
        let out1 = tf.zeros(out_shape.clone(), dynamism);
        let out2 = tf.zeros(out_shape, dynamism);

        let normalized_shape_vec: Vec<i64> = vec![3];
        let normalized_shape =
            IntArrayRef::from_raw_parts(normalized_shape_vec.as_ptr(), normalized_shape_vec.len());

        let mut ctx = context();
        native_layer_norm_out(
            &mut ctx,
            &input,
            normalized_shape,
            Some(&weight),
            Some(&bias),
            1e-05,
            &out0,
            &out1,
            &out2,
        );
        assert_tensor_close!(out0, expected);
    }

    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn/test]
    // also verifies layer_norm_scalar (normalized outputs asserted against C++ semantics
    // across float/double/half/bfloat16, incl. weight/bias/eps effects and nan/inf handling)
    // [spec:et:sem:normalization-ops-util.torch.executor.layer-norm-scalar-fn/test]
    #[test]
    fn op_native_layer_norm_test_float_tensors() {
        run_floating_point_test_cases::<f32>();
        run_floating_point_test_cases::<f64>();
        run_floating_point_test_cases::<Half>();
        run_floating_point_test_cases::<BFloat16>();
    }

    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    #[test]
    fn op_native_layer_norm_test_int_tensors_dies() {
        // Cannot be represented by a type other than float.
        run_int_test_cases::<i32>();
    }

    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    // also verifies check_layer_norm_args rejects normalized_shape=[1] mismatching input's last dim (3)
    // [spec:et:sem:normalization-ops-util.torch.executor.check-layer-norm-args-fn/test]
    #[test]
    fn op_native_layer_norm_test_wrong_nomalized_shape() {
        // Normalized shape does not match last dim of input.
        run_wrong_shape_test_cases::<f32>();
    }

    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    #[test]
    fn op_native_layer_norm_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    #[test]
    fn op_native_layer_norm_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: `ET_SKIP_IF(!output_resize, ...)` — the portable (non-aten) kernel
    // reports `output_resize = false`, so this test is SKIPPED. Body preserved for
    // correspondence; guarded by the skip.
    // [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn/test]
    #[test]
    fn op_native_layer_norm_test_dynamic_shape_unbound() {
        const OUTPUT_RESIZE: bool = false;
        if !OUTPUT_RESIZE {
            return;
        }
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
