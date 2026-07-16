//! Literal port of kernels/quantized/cpu/op_choose_qparams.cpp.
//!
//! For an input tensor, use the scale and zero_point arguments to quantize it.

use crate::kernels::portable::cpu::vec_ops::{vec_maxf, vec_minf};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface::parallel_for;

// PORT-NOTE: `ET_CHECK_MSG` is the C++ fatal check; mirrored with a local abort
// on failure (message dropped since a fatal abort follows), matching the
// established pattern in tensor_util.rs / op_embedding.rs.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

const SMALL_SCALE_THRESHOLD: f32 = 6.1e-5f32;

/// Asserts that the parameters are valid.
// [spec:et:def:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]
// [spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]
fn check_quantize_per_tensor_args(
    input: &Tensor,
    qmin: i64,
    qmax: i64,
    dtype: ScalarType,
    scale_out: &Tensor,
    zero_point_out: &Tensor,
    is_per_token: bool,
) {
    let _ = dtype;
    et_check_msg!(qmin < qmax, "qmin should be less than qmax");
    et_check_msg!(
        input.scalar_type() == ScalarType::Float,
        "Expected input to be Float tensor"
    );
    et_check_msg!(
        scale_out.scalar_type() == ScalarType::Double,
        "Expected scale to be Double tensor"
    );
    et_check_msg!(
        zero_point_out.scalar_type() == ScalarType::Long,
        "Expected scale to be Long tensor"
    );

    if is_per_token {
        let mut i: i64 = 0;
        while i < input.dim() as i64 - 1 {
            et_check_msg!(
                scale_out.size(i as isize) == input.size(i as isize),
                "Exepcted scale to have the same number of elements at dimentions"
            );
            et_check_msg!(
                zero_point_out.size(i as isize) == input.size(i as isize),
                "Exepcted zero pont to have the same number of elements at dimentions"
            );
            i += 1;
        }
        et_check_msg!(
            scale_out.size(input.dim() - 1) == 1,
            "Exepcted scale to have only one element at dimentions"
        );
        et_check_msg!(
            zero_point_out.size(input.dim() - 1) == 1,
            "Exepcted zero point to have only one element at dimentions"
        );
    } else {
        et_check_msg!(
            scale_out.numel() == 1,
            "Exepcted scale to only have one element received"
        );
        et_check_msg!(
            zero_point_out.numel() == 1,
            "Exepcted zero_point to only have one element received"
        );
    }
}

fn calculate_scale_and_zero_point(
    mut min: f32,
    mut max: f32,
    qmin: i32,
    qmax: i32,
    scale: &mut f64,
    zero_point: &mut i32,
) {
    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min = f32::min(min, 0.0f32);
    max = f32::max(max, 0.0f32);

    // Use double precision for intermediate computation but use single precision
    // in final number to reflect the actual number used during quantization.
    *scale = (max as f64 - min as f64) / (qmax - qmin) as f64;
    // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
    // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
    // infinity because some of fbgemm code pre-computes scale's reciprocal to do
    // multiplication instead of division in the time critical part of code.
    if (*scale as f32) == 0.0f32 || (1.0f32 / (*scale as f32)).is_infinite() {
        *scale = 0.1;
    }
    et_check_msg!(*scale > 0.0, "quantization scale should be > 0");

    // Cut off small scale
    if *scale < SMALL_SCALE_THRESHOLD as f64 {
        let org_scale: f32 = *scale as f32;
        *scale = SMALL_SCALE_THRESHOLD as f64;
        // Adjust the min and max based on the new scale
        if min == 0.0f32 {
            max = SMALL_SCALE_THRESHOLD * (qmax - qmin) as f32;
        } else if max == 0.0f32 {
            min = -SMALL_SCALE_THRESHOLD * (qmax - qmin) as f32;
        } else {
            let amplifier: f32 = SMALL_SCALE_THRESHOLD / org_scale;
            min *= amplifier;
            max *= amplifier;
        }
    }

    // Zero-point computation.
    // First the initial floating-point computation. The zero-point can be
    // determined from solving an affine equation for any known pair
    // (real value, corresponding quantized value).
    // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    // The arithmetic error on the zero point computed from either pair
    // will be roughly machine_epsilon * (sum of absolute values of terms)
    // so we want to use the variant that adds the smaller terms.
    let zero_point_from_min: f64 = qmin as f64 - min as f64 / *scale;
    let zero_point_from_max: f64 = qmax as f64 - max as f64 / *scale;
    let zero_point_from_min_error: f64 = (qmin as f64).abs() - (min as f64 / *scale).abs();
    let zero_point_from_max_error: f64 = (qmax as f64).abs() - (max as f64 / *scale).abs();
    let initial_zero_point: f64 = if zero_point_from_min_error < zero_point_from_max_error {
        zero_point_from_min
    } else {
        zero_point_from_max
    };

    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the requirement
    // to be able to represent the real value "0" exactly as a quantized value,
    // which is required in multiple places, for example in Im2col with zero
    // padding).
    let nudged_zero_point: i32;
    if initial_zero_point < qmin as f64 {
        nudged_zero_point = qmin;
    } else if initial_zero_point > qmax as f64 {
        nudged_zero_point = qmax;
    } else {
        // nearbyint(static_cast<float>(initial_zero_point)) then implicit
        // narrowing to int32.
        nudged_zero_point = (initial_zero_point as f32).round_ties_even() as i32;
    }
    *zero_point = nudged_zero_point;
}

// [spec:et:def:op-choose-qparams.torch.choose-qparams-fn]
// [spec:et:sem:op-choose-qparams.torch.choose-qparams-fn]
fn choose_qparams(
    input: &Tensor,
    qmin: i32,
    qmax: i32,
    scale_out: &Tensor,
    zero_point_out: &Tensor,
) {
    let x_fp32: *const f32 = input.const_data_ptr::<f32>();
    // Compute x_min, x_max and q_params (scale, zero_point)
    let min: f32 = unsafe { vec_minf(x_fp32, input.numel() as usize) };
    let max: f32 = unsafe { vec_maxf(x_fp32, input.numel() as usize) };

    let mut scale: f64 = 0.0;
    let mut zero_point: i32 = 0;
    calculate_scale_and_zero_point(min, max, qmin, qmax, &mut scale, &mut zero_point);

    unsafe {
        *scale_out.mutable_data_ptr::<f64>().add(0) = scale;
        *zero_point_out.mutable_data_ptr::<i64>().add(0) = zero_point as i64;
    }
}

// [spec:et:def:op-choose-qparams.torch.choose-qparams-per-token-fn]
// [spec:et:sem:op-choose-qparams.torch.choose-qparams-per-token-fn]
fn choose_qparams_per_token(
    input: &Tensor,
    qmin: i32,
    qmax: i32,
    scale_out: &Tensor,
    zero_point_out: &Tensor,
) {
    let mut x_fp32: *const f32 = input.const_data_ptr::<f32>();
    // Compute x_min, x_max and q_params (scale, zero_point)
    let mut num_tokens: i64 = 1;
    let mut i: i64 = 0;
    while i < input.dim() as i64 - 1 {
        num_tokens *= input.size(i as isize) as i64;
        i += 1;
    }
    let token_dim_size: i64 = input.size(input.dim() - 1) as i64;

    let total_elements: i64 = num_tokens * token_dim_size;
    const MIN_ELEMENTS_FOR_PARALLEL: i64 = 512;
    let use_parallel: bool = total_elements >= MIN_ELEMENTS_FOR_PARALLEL;

    if use_parallel {
        let scale_data: *mut f64 = scale_out.mutable_data_ptr::<f64>();
        let zero_point_data: *mut i64 = zero_point_out.mutable_data_ptr::<i64>();

        // PORT-NOTE: raw pointers captured by the parallel closure are not `Send`;
        // the ported no-threadpool `parallel_for` runs the closure inline on the
        // calling thread, so this is sound. `x_fp32`, `scale_data`, and
        // `zero_point_data` are wrapped in `usize` to move them across the closure
        // boundary and reconstituted inside, mirroring the C++ capture-by-value of
        // the base pointers.
        let x_base = x_fp32 as usize;
        let scale_base = scale_data as usize;
        let zp_base = zero_point_data as usize;
        let _ = parallel_for(0, num_tokens, 1, &|begin: i64, end: i64| {
            let x_fp32 = x_base as *const f32;
            let scale_data = scale_base as *mut f64;
            let zero_point_data = zp_base as *mut i64;
            let mut i: i64 = begin;
            while i < end {
                let token_data: *const f32 =
                    unsafe { x_fp32.offset((i * token_dim_size) as isize) };
                let min: f32 = unsafe { vec_minf(token_data, token_dim_size as usize) };
                let max: f32 = unsafe { vec_maxf(token_data, token_dim_size as usize) };
                let mut scale: f64 = 0.0;
                let mut zero_point: i32 = 0;
                calculate_scale_and_zero_point(min, max, qmin, qmax, &mut scale, &mut zero_point);
                unsafe {
                    *scale_data.offset(i as isize) = scale;
                    *zero_point_data.offset(i as isize) = zero_point as i64;
                }
                i += 1;
            }
        });
    } else {
        let mut i: i64 = 0;
        while i < num_tokens {
            // vec_minf uses std::min_element. Check if it actually
            // gets vectorized.
            let min: f32 = unsafe { vec_minf(x_fp32, token_dim_size as usize) };
            let max: f32 = unsafe { vec_maxf(x_fp32, token_dim_size as usize) };
            let mut scale: f64 = 0.0;
            let mut zero_point: i32 = 0;
            calculate_scale_and_zero_point(min, max, qmin, qmax, &mut scale, &mut zero_point);
            unsafe {
                *scale_out.mutable_data_ptr::<f64>().offset(i as isize) = scale;
                *zero_point_out.mutable_data_ptr::<i64>().offset(i as isize) = zero_point as i64;
                x_fp32 = x_fp32.offset(token_dim_size as isize);
            }
            i += 1;
        }
    }
}

pub fn choose_qparams_tensor_out<'s, 'z, 'sb, 'zb>(
    input: &Tensor,
    quant_min: i64,
    quant_max: i64,
    _eps: f64,
    dtype: ScalarType,
    scale_out: &'s Tensor<'sb>,
    zero_point_out: &'z Tensor<'zb>,
) -> (&'s Tensor<'sb>, &'z Tensor<'zb>) {
    check_quantize_per_tensor_args(
        input,
        quant_min,
        quant_max,
        dtype,
        scale_out,
        zero_point_out,
        false,
    );

    choose_qparams(
        input,
        quant_min as i32,
        quant_max as i32,
        scale_out,
        zero_point_out,
    );
    (scale_out, zero_point_out)
}

// [spec:et:def:op-choose-qparams.choose-qparams-tensor-out-fn]
// [spec:et:sem:op-choose-qparams.choose-qparams-tensor-out-fn]
pub fn choose_qparams_tensor_out_context<'s, 'z, 'sb, 'zb>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    quant_min: i64,
    quant_max: i64,
    eps: f64,
    dtype: ScalarType,
    scale_out: &'s Tensor<'sb>,
    zero_point_out: &'z Tensor<'zb>,
) -> (&'s Tensor<'sb>, &'z Tensor<'zb>) {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let _ = context;
    choose_qparams_tensor_out(
        input,
        quant_min,
        quant_max,
        eps,
        dtype,
        scale_out,
        zero_point_out,
    )
}

// [spec:et:def:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn]
// [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn]
pub fn choose_qparams_per_token_asymmetric_out<'s, 'z, 'sb, 'zb>(
    input: &Tensor,
    dtype: ScalarType,
    scale_out: &'s Tensor<'sb>,
    zero_point_out: &'z Tensor<'zb>,
) -> (&'s Tensor<'sb>, &'z Tensor<'zb>) {
    let quant_min: i64 = -128;
    let quant_max: i64 = 127;
    let mut output_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut i: isize = 0;
    while i < input.dim() - 1 {
        output_sizes[i as usize] = input.size(i) as TensorSizesType;
        i += 1;
    }
    output_sizes[(input.dim() - 1) as usize] = 1;
    let output_dim: usize = input.dim() as usize;
    let mut err: Error = resize_tensor(
        scale_out,
        ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_dim),
    );
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize scale_out Tensor in choose_qparams"
    );
    err = resize_tensor(
        zero_point_out,
        ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_dim),
    );
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize zero_point_out Tensor in choose_qparams"
    );

    check_quantize_per_tensor_args(
        input,
        quant_min,
        quant_max,
        dtype,
        scale_out,
        zero_point_out,
        true, /* is_per_token*/
    );

    choose_qparams_per_token(
        input,
        quant_min as i32,
        quant_max as i32,
        scale_out,
        zero_point_out,
    );
    (scale_out, zero_point_out)
}

pub fn choose_qparams_per_token_asymmetric_out_context<'s, 'z, 'sb, 'zb>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    dtype: ScalarType,
    scale_out: &'s Tensor<'sb>,
    zero_point_out: &'z Tensor<'zb>,
) -> (&'s Tensor<'sb>, &'z Tensor<'zb>) {
    let _ = context;
    choose_qparams_per_token_asymmetric_out(input, dtype, scale_out, zero_point_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::{DYNAMIC_BOUND, STATIC};

    // `EXPECT_TENSOR_CLOSE` -> default rtol, dtype-derived atol.
    macro_rules! assert_tensor_close {
        ($t1:expr, $t2:expr) => {
            assert!(
                tensors_are_close(
                    &$t1,
                    &$t2,
                    crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                    None
                ),
                "tensors are not close"
            )
        };
    }

    // `EXPECT_TENSOR_CLOSE_WITH_TOL`.
    macro_rules! assert_tensor_close_with_tol {
        ($t1:expr, $t2:expr, $rtol:expr, $atol:expr) => {
            assert!(
                tensors_are_close(&$t1, &$t2, $rtol, Some($atol)),
                "tensors are not close within tolerance"
            )
        };
    }

    // `EXPECT_TENSOR_EQ`.
    macro_rules! assert_tensor_eq {
        ($t1:expr, $t2:expr) => {
            assert!(
                tensors_are_close(&$t1, &$t2, 0.0, Some(0.0)),
                "tensors are not equal"
            )
        };
    }

    /// A generic smoke test that works for any dtype that supports ones() and
    /// zeros(). The C++ `test_dtype<DTYPE>` template is not instantiated by any
    /// C++ TEST; it is driven here by `op_choose_qparams_tensor_out_test_float`.
    fn test_dtype(dtype: ScalarType) {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.make_default(vec![2, 2], vec![1.0, 2.5, 3.2, 15.4]);
        let scale_out = tf_double.zeros(vec![1], STATIC);
        let zero_point_out = tf_long.zeros(vec![1], STATIC);
        let expected_scale = tf_double.make_default(vec![1], vec![0.0603922]);
        let expected_zero_point = tf_long.make_default(vec![1], vec![0]);

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        choose_qparams_tensor_out(
            &input,
            quant_min,
            quant_max,
            0.0,
            dtype,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close!(scale_out, expected_scale);
        assert_tensor_eq!(zero_point_out, expected_zero_point);
    }

    // PORT-NOTE: the C++ `test_dtype<DTYPE>` template above is never instantiated
    // by a TEST, so the non-per-token path (choose_qparams_tensor_out ->
    // choose_qparams, and check_quantize_per_tensor_args's is_per_token=false
    // branch) has no running coverage in the ported suite. This focused test
    // drives that path with the same literal reference values the C++ helper uses.
    // [spec:et:sem:op-choose-qparams.choose-qparams-tensor-out-fn/test]
    // [spec:et:sem:op-choose-qparams.torch.choose-qparams-fn/test]
    // [spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn/test]
    #[test]
    fn op_choose_qparams_tensor_out_test_float() {
        test_dtype(ScalarType::Float);
    }

    // choose_qparams_per_token_asymmetric_out runs check_quantize_per_tensor_args
    // (is_per_token=true branch) and choose_qparams_per_token; the exact expected
    // scale/zero_point pin both.
    // [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn/test]
    // [spec:et:sem:op-choose-qparams.torch.choose-qparams-per-token-fn/test]
    // [spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn/test]
    #[test]
    fn op_choose_qparams_per_token_asymmetric_tensor_out_test_float() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.make_default(vec![2, 3], vec![-0.5, 0.3, 1.2, 0.1, -0.8, 2.1]);
        let scale_out = tf_double.zeros(vec![2, 1], STATIC);
        let zero_point_out = tf_long.zeros(vec![2, 1], STATIC);
        let expected_scale = tf_double.make_default(vec![2, 1], vec![0.00666667, 0.0113725485]);
        let expected_zero_point = tf_long.make_default(vec![2, 1], vec![-53, -58]);

        choose_qparams_per_token_asymmetric_out(
            &input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close_with_tol!(scale_out, expected_scale, 1e-4, 1e-4);
        assert_tensor_eq!(zero_point_out, expected_zero_point);
    }

    // [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn/test]
    #[test]
    fn op_choose_qparams_per_token_asymmetric_tensor_out_test_extra_dim_float() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.make_default(vec![1, 2, 3], vec![-0.5, 0.3, 1.2, 0.1, -0.8, 2.1]);
        let scale_out = tf_double.zeros(vec![1, 2, 1], STATIC);
        let zero_point_out = tf_long.zeros(vec![1, 2, 1], STATIC);
        let expected_scale = tf_double.make_default(vec![1, 2, 1], vec![0.00666667, 0.0113725485]);
        let expected_zero_point = tf_long.make_default(vec![1, 2, 1], vec![-53, -58]);

        choose_qparams_per_token_asymmetric_out(
            &input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close_with_tol!(scale_out, expected_scale, 1e-4, 1e-4);
        assert_tensor_eq!(zero_point_out, expected_zero_point);
    }

    // [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn/test]
    #[test]
    fn op_choose_qparams_per_token_asymmetric_tensor_out_test_large_array() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.make_default(
            vec![5, 17],
            vec![
                0.41654, 0.26599, 0.4141, 0.83809, 0.02938, 0.12199, 0.53667, 0.799, 0.6606,
                0.46657, 0.66142, 0.71787, 0.56098, 0.30202, 0.059377, 0.85473, 0.8017, 0.2703,
                0.44299, 0.49045, 0.75581, 0.24429, 0.43906, 0.78652, 0.83885, 0.31034, 0.76534,
                0.74422, 0.62549, 0.80006, 0.38144, 0.70652, 0.33553, 0.89136, 0.49126, 0.072916,
                0.75654, 0.82057, 0.083848, 0.29753, 0.62718, 0.95579, 0.83097, 0.47293, 0.15666,
                0.6248, 0.21672, 0.14626, 0.71834, 0.93664, 0.23382, 0.68931, 0.70866, 0.60545,
                0.98648, 0.30335, 0.62439, 0.19195, 0.1923, 0.75638, 0.81114, 0.34778, 0.0070671,
                0.50918, 0.19698, 0.19969, 0.57687, 0.062786, 0.18447, 0.22961, 0.29656, 0.25486,
                0.75965, 0.11328, 0.86468, 0.21264, 0.99591, 0.75231, 0.97834, 0.042441, 0.39978,
                0.9633, 0.9297, 0.12188, 0.73564,
            ],
        );
        let scale_out = tf_double.zeros(vec![5, 1], STATIC);
        let zero_point_out = tf_long.zeros(vec![5, 1], STATIC);
        let expected_scale = tf_double.make_default(
            vec![5, 1],
            vec![0.0033519, 0.0034955, 0.0037482, 0.0038685, 0.0039055],
        );
        let expected_zero_point =
            tf_long.make_default(vec![5, 1], vec![-128, -128, -128, -128, -128]);

        choose_qparams_per_token_asymmetric_out(
            &input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close_with_tol!(scale_out, expected_scale, 1e-5, 1e-5);
        assert_tensor_eq!(zero_point_out, expected_zero_point);
    }

    // [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn/test]
    #[test]
    fn op_choose_qparams_per_token_asymmetric_tensor_out_test_dynamic_shape_float() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.make_default(vec![1, 2, 3], vec![-0.5, 0.3, 1.2, 0.1, -0.8, 2.1]);
        let scale_out = tf_double.zeros(vec![1, 5, 1], DYNAMIC_BOUND);
        let zero_point_out = tf_long.zeros(vec![1, 5, 1], DYNAMIC_BOUND);
        let expected_scale = tf_double.make_default(vec![1, 2, 1], vec![0.00666667, 0.0113725485]);
        let expected_zero_point = tf_long.make_default(vec![1, 2, 1], vec![-53, -58]);

        choose_qparams_per_token_asymmetric_out(
            &input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close_with_tol!(scale_out, expected_scale, 1e-4, 1e-4);
        assert_tensor_eq!(zero_point_out, expected_zero_point);

        let new_input = tf_float.make_default(
            vec![1, 5, 8],
            vec![
                5.2254, 5.6041, 5.7653, -1.0126, -0.86126, -0.1606, -0.99196, -1.067, 5.5913,
                5.7713, 5.4901, -0.43128, -1.1759, -0.60466, -0.82913, -0.73623, 5.4588, 5.4066,
                5.2644, -0.89692, -0.16866, -0.63169, -0.42352, -0.48866, 5.594, 5.5223, 5.5277,
                -0.17658, -0.30669, -1.1777, -0.65389, -0.36422, 5.6375, 5.1857, 5.0743, -0.46654,
                -0.43817, -0.41506, -0.94515, -0.60247,
            ],
        );
        let new_expected_scale = tf_double.make_default(
            vec![1, 5, 1],
            vec![0.026793, 0.027244, 0.024924, 0.026556, 0.025814],
        );
        let new_expected_zero_point =
            tf_long.make_default(vec![1, 5, 1], vec![-88, -85, -92, -84, -91]);

        choose_qparams_per_token_asymmetric_out(
            &new_input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        assert_tensor_close_with_tol!(scale_out, new_expected_scale, 1e-4, 1e-4);
        assert_tensor_eq!(zero_point_out, new_expected_zero_point);
    }

    // [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn/test]
    #[test]
    fn op_choose_qparams_per_token_asymmetric_tensor_out_test_large_input_parallelization() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Create input with 8 tokens x 128 elements per token = 1024 total elements
        // This exceeds the MIN_ELEMENTS_FOR_PARALLEL threshold of 512
        let num_tokens = 8usize;
        let token_size = 128usize;
        let mut input_data = vec![0.0f32; num_tokens * token_size];

        // Generate test data with known min/max per token for easier verification
        let mut expected_min = vec![0.0f32; num_tokens];
        let mut expected_max = vec![0.0f32; num_tokens];

        for i in 0..num_tokens {
            let token_min = -1.0f32 * (i as f32 + 1.0);
            let token_max = 2.0f32 * (i as f32 + 1.0);
            expected_min[i] = token_min;
            expected_max[i] = token_max;

            for j in 0..token_size {
                // Linearly interpolate between min and max
                let t = j as f32 / (token_size as f32 - 1.0);
                input_data[i * token_size + j] = token_min + t * (token_max - token_min);
            }
        }

        let input = tf_float.make_default(vec![num_tokens as i32, token_size as i32], input_data);
        let scale_out = tf_double.zeros(vec![num_tokens as i32, 1], STATIC);
        let zero_point_out = tf_long.zeros(vec![num_tokens as i32, 1], STATIC);

        choose_qparams_per_token_asymmetric_out(
            &input,
            ScalarType::Float,
            &scale_out,
            &zero_point_out,
        );

        // Manually calculate expected scale and zero_point using the same algorithm
        // as calculate_scale_and_zero_point function
        let qmin: i32 = -128;
        let qmax: i32 = 127;
        const SMALL_SCALE_THRESHOLD: f32 = 6.1e-5f32;

        let scale_ptr = scale_out.const_data_ptr::<f64>();
        let zp_ptr = zero_point_out.const_data_ptr::<i64>();
        for i in 0..num_tokens {
            let mut min = expected_min[i].min(0.0f32);
            let mut max = expected_max[i].max(0.0f32);

            // Calculate scale
            let mut scale = (max as f64 - min as f64) / (qmax - qmin) as f64;
            if scale as f32 == 0.0f32 || (1.0f32 / scale as f32).is_infinite() {
                scale = 0.1;
            }

            // Cut off small scale
            if scale < SMALL_SCALE_THRESHOLD as f64 {
                scale = SMALL_SCALE_THRESHOLD as f64;
                if min == 0.0f32 {
                    max = SMALL_SCALE_THRESHOLD * (qmax - qmin) as f32;
                } else if max == 0.0f32 {
                    min = -SMALL_SCALE_THRESHOLD * (qmax - qmin) as f32;
                } else {
                    let amplifier = SMALL_SCALE_THRESHOLD / scale as f32;
                    min *= amplifier;
                    max *= amplifier;
                }
            }

            // Calculate zero_point
            let zero_point_from_min = qmin as f64 - min as f64 / scale;
            let zero_point_from_max = qmax as f64 - max as f64 / scale;
            let zero_point_from_min_error = (qmin as f64).abs() - (min as f64 / scale).abs();
            let zero_point_from_max_error = (qmax as f64).abs() - (max as f64 / scale).abs();
            let initial_zero_point = if zero_point_from_min_error < zero_point_from_max_error {
                zero_point_from_min
            } else {
                zero_point_from_max
            };

            let nudged_zero_point: i32 = if initial_zero_point < qmin as f64 {
                qmin
            } else if initial_zero_point > qmax as f64 {
                qmax
            } else {
                (initial_zero_point as f32).round_ties_even() as i32
            };

            // Verify computed values match expected (EXPECT_NEAR / EXPECT_EQ)
            assert!((unsafe { *scale_ptr.add(i) } - scale).abs() < 1e-6);
            assert_eq!(unsafe { *zp_ptr.add(i) }, nudged_zero_point as i64);
        }
    }
}
