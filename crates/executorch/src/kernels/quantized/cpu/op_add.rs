//! Literal port of kernels/quantized/cpu/op_add.cpp.

use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_shape_and_dtype;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `ET_CHECK_SAME_SHAPE_AND_DTYPE3` / `ET_CHECK_MSG` are C++ fatal
// checks; mirrored with a local abort on failure (message dropped since a fatal
// abort follows), matching the established pattern in tensor_util.rs /
// op_embedding.rs. `ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out)` expands to a
// fatal check that all three tensors share shape and dtype
// (`tensors_have_same_shape_and_dtype`).
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `INPUT_T`/`OUTPUT_T` are template params; the ported free functions
// take them as generic type parameters. The integer/float casts are expressed
// with explicit `as` conversions and a small `Cast` trait so the mixed-type
// `static_cast` chain stays literal and bit-exact.

// PORT-NOTE: reproduces the C++ `static_cast<OUTPUT_T>` from an `int64_t`
// quantized value at the end of `quantize_val`, and `static_cast<OUTPUT_T>`
// (float) at the end of `dequantize_val`. `FromI64` narrows int64 to the storage
// type (wrapping/truncating exactly as the C++ integral cast does). `FromF64`
// narrows the `(value - zero_point) * scale` double product to OUTPUT_T (float
// at the call sites here).
trait FromI64 {
    fn from_i64(v: i64) -> Self;
}
macro_rules! impl_from_i64 {
    ($t:ty) => {
        impl FromI64 for $t {
            fn from_i64(v: i64) -> Self {
                v as $t
            }
        }
    };
}
impl_from_i64!(u8);
impl_from_i64!(i8);
impl_from_i64!(i16);
impl_from_i64!(i32);
impl_from_i64!(i64);

// PORT-NOTE: `value - zero_point` in `dequantize_val`: `value` (INPUT_T integer)
// and `zero_point` (int64) promote to int64 for the subtraction, then multiply
// by `scale` (double). `AsI64` reproduces the integer promotion of the quantized
// storage type to int64.
trait AsI64 {
    fn as_i64(self) -> i64;
}
macro_rules! impl_as_i64 {
    ($t:ty) => {
        impl AsI64 for $t {
            fn as_i64(self) -> i64 {
                self as i64
            }
        }
    };
}
impl_as_i64!(u8);
impl_as_i64!(i8);
impl_as_i64!(i16);
impl_as_i64!(i32);
impl_as_i64!(i64);

// [spec:et:def:op-add.torch.executor.native.quantize-val-fn]
// [spec:et:sem:op-add.torch.executor.native.quantize-val-fn]
#[allow(non_camel_case_types)]
fn quantize_val<INPUT_T, OUTPUT_T>(
    scale: f64,
    zero_point: i64,
    value: INPUT_T,
    quant_min: i64,
    quant_max: i64,
) -> OUTPUT_T
where
    INPUT_T: Into<f32>,
    OUTPUT_T: FromI64,
{
    let qvalue: i64;
    let inv_scale: f32 = 1.0f32 / (scale as f32);
    // qvalue = static_cast<int64_t>(zero_point + std::nearbyint(inv_scale *
    // value)). `std::nearbyint` returns a float; `zero_point` (int64) promotes
    // to float for the `+`, and the float sum is truncated to int64. Done in
    // float here (not double) to match the C++ promotion. std::nearbyint rounds
    // to nearest using the current rounding mode (default round-half-to-even),
    // matching f32::round_ties_even.
    qvalue = (zero_point as f32 + (inv_scale * value.into()).round_ties_even()) as i64;
    let qvalue = core::cmp::max::<i64>(qvalue, quant_min);
    let qvalue = core::cmp::min::<i64>(qvalue, quant_max);
    OUTPUT_T::from_i64(qvalue)
}

// [spec:et:def:op-add.torch.executor.native.dequantize-val-fn]
// [spec:et:sem:op-add.torch.executor.native.dequantize-val-fn]
#[allow(non_camel_case_types)]
fn dequantize_val<INPUT_T, OUTPUT_T>(scale: f64, zero_point: i64, value: INPUT_T) -> OUTPUT_T
where
    INPUT_T: AsI64,
    OUTPUT_T: FromF64,
{
    OUTPUT_T::from_f64((value.as_i64() - zero_point) as f64 * scale)
}

// PORT-NOTE: final narrowing of the double product to OUTPUT_T. At the call
// sites in this file OUTPUT_T is `float`, so the double result is narrowed to
// f32 (`static_cast<float>`).
trait FromF64 {
    fn from_f64(v: f64) -> Self;
}
impl FromF64 for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// Perform element wise addition of the input tensors into out.
/// Should be numerically equivalent to Dq -> fp add -> Q
// [spec:et:def:op-add.torch.executor.native.add-tensors-fn]
// [spec:et:sem:op-add.torch.executor.native.add-tensors-fn]
fn add_tensors<CTYPE>(
    a: &Tensor,
    a_scale: f32,
    a_zero_point: i32,
    b: &Tensor,
    b_scale: f32,
    b_zero_point: i32,
    out: &Tensor,
    out_scale: f32,
    out_zero_point: i32,
    out_quant_min: i64,
    out_quant_max: i64,
) where
    CTYPE: Copy + AsI64 + FromI64,
{
    let n: usize = a.numel() as usize;

    let data_a = a.const_data_ptr::<CTYPE>();
    let data_b = b.const_data_ptr::<CTYPE>();
    let data_out = out.mutable_data_ptr::<CTYPE>();

    for i in 0..n {
        // Dq -> fp add -> Q. Can be optimized further
        let dqa: f32 = dequantize_val::<CTYPE, f32>(a_scale as f64, a_zero_point as i64, unsafe {
            *data_a.add(i)
        });
        let dqb: f32 = dequantize_val::<CTYPE, f32>(b_scale as f64, b_zero_point as i64, unsafe {
            *data_b.add(i)
        });
        let accumulate: f32 = dqa + dqb;

        unsafe {
            *data_out.add(i) = quantize_val::<f32, CTYPE>(
                out_scale as f64,
                out_zero_point as i64,
                accumulate,
                out_quant_min,
                out_quant_max,
            );
        }
    }
}

/// Perform element wise addition of the input tensors into out. Should be
/// numerically equivalent to Dq -> fp add -> Q
///
/// PREREQ: a and b should be the same shape, quant_min and max should be in
/// range [0,255]. a and b and out should be the same dtype.
// [spec:et:def:op-add.torch.executor.native.quantized-add-out-fn]
// [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_add_out<'a, 'b>(
    a: &Tensor,
    a_scale_d: f64,
    a_zero_point_l: i64,
    a_quant_min: i64,
    a_quant_max: i64,
    b: &Tensor,
    b_scale_d: f64,
    b_zero_point_l: i64,
    b_quant_min: i64,
    b_quant_max: i64,
    out_scale_d: f64,
    out_zero_point_l: i64,
    out_quant_min: i64,
    out_quant_max: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    et_check_msg!(
        tensors_have_same_shape_and_dtype(a, b, out),
        "ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out)"
    );
    et_check_msg!(
        a_quant_min >= 0 && a_quant_max <= 255 && a_quant_min <= a_quant_max,
        "invalid quant_min or quant_max for input tensor a"
    );
    et_check_msg!(
        b_quant_min >= 0 && b_quant_max <= 255 && b_quant_min <= b_quant_max,
        "invalid quant_min or quant_max for input tensor b"
    );
    et_check_msg!(
        out_quant_min >= 0 && out_quant_max <= 255 && out_quant_min <= out_quant_max,
        "invalid quant_min or quant_max for output tensor"
    );

    // downsize to maintain numerical consistency with fbgemm
    let a_scale: f32 = a_scale_d as f32;
    let b_scale: f32 = b_scale_d as f32;
    let out_scale: f32 = out_scale_d as f32;

    let a_zero_point: i32 = a_zero_point_l as i32;
    let b_zero_point: i32 = b_zero_point_l as i32;
    let out_zero_point: i32 = out_zero_point_l as i32;

    // PORT-NOTE: the C++ `ADD_TENSORS` macro over `ET_FORALL_INT_TYPES`
    // (Byte/Char/Short/Int/Long) is expanded here as an explicit match; the
    // `default` branch fires the fatal `ET_CHECK_MSG(false, ...)`.
    macro_rules! add_tensors_case {
        ($ctype:ty) => {
            add_tensors::<$ctype>(
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                out,
                out_scale,
                out_zero_point,
                out_quant_min,
                out_quant_max,
            )
        };
    }

    match a.scalar_type() {
        ScalarType::Byte => add_tensors_case!(u8),
        ScalarType::Char => add_tensors_case!(i8),
        ScalarType::Short => add_tensors_case!(i16),
        ScalarType::Int => add_tensors_case!(i32),
        ScalarType::Long => add_tensors_case!(i64),
        _ => {
            et_check_msg!(false, "Unhandled dtype");
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_add_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    a: &Tensor,
    a_scale: f64,
    a_zero_point: i64,
    a_quant_min: i64,
    a_quant_max: i64,
    b: &Tensor,
    b_scale: f64,
    b_zero_point: i64,
    b_quant_min: i64,
    b_quant_max: i64,
    out_scale: f64,
    out_zero_point: i64,
    out_quant_min: i64,
    out_quant_max: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let _ = context;
    quantized_add_out(
        a,
        a_scale,
        a_zero_point,
        a_quant_min,
        a_quant_max,
        b,
        b_scale,
        b_zero_point,
        b_quant_min,
        b_quant_max,
        out_scale,
        out_zero_point,
        out_quant_min,
        out_quant_max,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::kernels::portable::cpu::op_add::add_out;
    use crate::kernels::quantized::cpu::op_dequantize::dequantize_per_tensor_out;
    use crate::kernels::quantized::cpu::op_quantize::quantize_per_tensor_out;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar::Scalar;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // A generic smoke test that works for any dtype that supports ones() and
    // zeros(). Only Byte is exercised by AllDtypesSupported.
    //
    // The Dq -> fp add -> Q pipeline runs through add_tensors, which calls
    // dequantize_val on each input element and quantize_val on each output; the
    // exact expected byte values below fail if any of those three are wrong.
    // [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn/test]
    // [spec:et:sem:op-add.torch.executor.native.add-tensors-fn/test]
    // [spec:et:sem:op-add.torch.executor.native.dequantize-val-fn/test]
    // [spec:et:sem:op-add.torch.executor.native.quantize-val-fn/test]
    fn test_dtype() {
        let tf = TensorFactory::<f32>::new();

        let input1 = tf.full(vec![3, 5], 3.5, STATIC);
        let input2 = tf.full(vec![3, 5], 3.5, STATIC);
        let scale: f64 = 0.5;

        let zero_point: i64 = 1;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let qinput1 = tfo.zeros(vec![3, 5], STATIC);
        let qinput2 = tfo.zeros(vec![3, 5], STATIC);
        let qoutput = tfo.zeros(vec![3, 5], STATIC);
        // 3.5 / 0.5 + 1 = 8
        quantize_per_tensor_out(
            &input1,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput1,
        );

        quantize_per_tensor_out(
            &input2,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput2,
        );

        quantized_add_out(
            &qinput1, scale, zero_point, quant_min, quant_max, &qinput2, scale, zero_point,
            quant_min, quant_max, scale, zero_point, quant_min, quant_max, &qoutput,
        );

        // can lossessly dq here so retrive the full 3.5 in operation
        // (3.5 + 3.5) / 0.5 + 1 = 15
        let expected = tfo.full(vec![3, 5], 15, STATIC);

        assert_tensor_eq!(qoutput, expected);
    }

    #[test]
    fn op_quantize_add_test_all_dtypes_supported() {
        test_dtype();
    }

    // [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn/test]
    #[test]
    fn op_quantize_add_test_different_q_params() {
        let tf = TensorFactory::<f32>::new();

        let input1 = tf.full(vec![3, 5], 3.5, STATIC);
        let input2 = tf.full(vec![3, 5], 3.5, STATIC);
        let a_scale: f64 = 0.5;
        let a_zero_point: i64 = 1;

        let b_scale: f64 = 0.25;
        let b_zero_point: i64 = 2;

        let out_scale: f64 = 0.1;
        let out_zero_point: i64 = 5;

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let qinput1 = tfo.zeros(vec![3, 5], STATIC);
        let qinput2 = tfo.zeros(vec![3, 5], STATIC);
        let qoutput = tfo.zeros(vec![3, 5], STATIC);
        // 3.5 / 0.5 + 1 = 8
        quantize_per_tensor_out(
            &input1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput1,
        );

        // 3.5 / 0.25 + 2 = 16
        quantize_per_tensor_out(
            &input2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput2,
        );

        quantized_add_out(
            &qinput1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            &qinput2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            &qoutput,
        );

        // can lossessly dq here so retrive the full 3.5 in operation
        // (3.5 + 3.5) / 0.1 + 5 = 75
        let expected = tfo.full(vec![3, 5], 75, STATIC);

        assert_tensor_eq!(qoutput, expected);
    }

    // Q -> DQ -> FP ADD -> Q -> DQ should be == to Q -> QADD -> DQ
    // [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn/test]
    #[test]
    fn op_quantize_add_test_consitency_with_reference_pattern() {
        let tf = TensorFactory::<f32>::new();

        let input1 = tf.full(vec![3, 5], 3.5, STATIC);
        let input2 = tf.full(vec![3, 5], 3.5, STATIC);
        let dq_input1 = tf.zeros(vec![3, 5], STATIC);
        let dq_input2 = tf.zeros(vec![3, 5], STATIC);
        let reference_op_output = tf.zeros(vec![3, 5], STATIC);
        let reference_pattern_output = tf.zeros(vec![3, 5], STATIC);
        let fp_output = tf.zeros(vec![3, 5], STATIC);

        let a_scale: f64 = 0.5;
        let a_zero_point: i64 = 1;

        let b_scale: f64 = 0.25;
        let b_zero_point: i64 = 2;

        let out_scale: f64 = 0.1;
        let out_zero_point: i64 = 5;

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let qinput1 = tfo.zeros(vec![3, 5], STATIC);
        let qinput2 = tfo.zeros(vec![3, 5], STATIC);
        let qoutput = tfo.zeros(vec![3, 5], STATIC);

        let out_dtype: Option<ScalarType> = None;

        let mut ctx = context();
        // q -> qadd -> dq
        // 3.5 / 0.5 + 1 = 8
        quantize_per_tensor_out(
            &input1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput1,
        );

        // 3.5 / 0.25 + 2 = 16
        quantize_per_tensor_out(
            &input2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput2,
        );

        quantized_add_out(
            &qinput1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            &qinput2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            &qoutput,
        );
        dequantize_per_tensor_out(
            &qoutput,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            out_dtype,
            &reference_op_output,
        );

        // now get results for q -> dq -> fp add -> q -> dq
        dequantize_per_tensor_out(
            &qinput1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            out_dtype,
            &dq_input1,
        );

        dequantize_per_tensor_out(
            &qinput2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            out_dtype,
            &dq_input2,
        );

        add_out(
            &mut ctx,
            &dq_input1,
            &dq_input2,
            &Scalar::from_double(1.0),
            &fp_output,
        );
        // reuse 'qoutput' tensor as an intermediate
        quantize_per_tensor_out(
            &fp_output,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qoutput,
        );

        dequantize_per_tensor_out(
            &qoutput,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            out_dtype,
            &reference_pattern_output,
        );

        let expected = tf.full(vec![3, 5], 7.0, STATIC);

        // Pattern and op results should both be equal to expected and each other,
        // check all cases explicitly instead of relying on transitivity
        assert_tensor_eq!(reference_op_output, expected);
        assert_tensor_eq!(reference_pattern_output, expected);
        assert_tensor_eq!(reference_op_output, reference_pattern_output);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` on the invalid out_quant_min/out_quant_max
    // check goes through `et_check_msg!` -> `runtime_abort()` -> `libc::abort()`,
    // which terminates the process instead of unwinding, so `#[should_panic]`
    // cannot catch it and running it would kill the whole test binary. Ported and
    // `#[ignore]`d; the abort semantics are asserted by inspection of
    // `quantized_add_out`'s out_quant_min/out_quant_max check.
    // [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantize_add_test_invalid_min_max_dies() {
        let tf = TensorFactory::<f32>::new();

        let input1 = tf.full(vec![3, 5], 3.5, STATIC);
        let input2 = tf.full(vec![3, 5], 3.5, STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 1;

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;
        let out_quant_min: i64 = -1;
        let out_quant_max: i64 = 256;

        let tfo = TensorFactory::<u8>::new();
        let qinput1 = tfo.zeros(vec![3, 5], STATIC);
        let qinput2 = tfo.zeros(vec![3, 5], STATIC);
        let qoutput = tfo.zeros(vec![3, 5], STATIC);
        // 3.5 / 0.5 + 1 = 8
        quantize_per_tensor_out(
            &input1,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput1,
        );

        // 3.5 / 0.25 + 2 = 16
        quantize_per_tensor_out(
            &input2,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput2,
        );

        quantized_add_out(
            &qinput1,
            scale,
            zero_point,
            quant_min,
            quant_max,
            &qinput2,
            scale,
            zero_point,
            quant_min,
            quant_max,
            scale,
            zero_point,
            out_quant_min,
            out_quant_max,
            &qoutput,
        );
    }

    // [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn/test]
    #[test]
    fn op_quantize_add_test_top_of_range_test() {
        let tf = TensorFactory::<f32>::new();

        let input1 = tf.full(vec![3, 5], 255.0, STATIC);
        let input2 = tf.full(vec![3, 5], 255.0, STATIC);
        let a_scale: f64 = 1.0;
        let a_zero_point: i64 = 0;

        let b_scale: f64 = 1.0;
        let b_zero_point: i64 = 0;

        let out_scale: f64 = 1.0;
        let out_zero_point: i64 = 0;

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let qinput1 = tfo.zeros(vec![3, 5], STATIC);
        let qinput2 = tfo.zeros(vec![3, 5], STATIC);
        let qoutput = tfo.zeros(vec![3, 5], STATIC);

        quantize_per_tensor_out(
            &input1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput1,
        );

        // 3.5 / 0.25 + 2 = 16
        quantize_per_tensor_out(
            &input2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qinput2,
        );

        quantized_add_out(
            &qinput1,
            a_scale,
            a_zero_point,
            quant_min,
            quant_max,
            &qinput2,
            b_scale,
            b_zero_point,
            quant_min,
            quant_max,
            out_scale,
            out_zero_point,
            quant_min,
            quant_max,
            &qoutput,
        );

        let expected = tfo.full(vec![3, 5], 255, STATIC);

        assert_tensor_eq!(qoutput, expected);
    }
}
