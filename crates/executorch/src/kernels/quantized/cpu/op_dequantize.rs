//! Literal port of kernels/quantized/cpu/op_dequantize.cpp.
//!
//! For an input tensor, use the scale and zero_point arguments to quantize it.

use crate::kernels::portable::cpu::util::reduce_util::{apply_over_dim_list, apply_over_dim_whole};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    dim_order_to_stride_nocheck, is_contiguous_dim_order,
};
#[cfg(feature = "aten")]
use crate::runtime::core::exec_aten::util::tensor_util::tensor_is_contiguous;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, nonzero_dim, resize_tensor,
    tensor_has_dim,
};
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl,
};
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

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

// PORT-NOTE: `ET_CHECK_VALID_DIM(dim, upper_bound)` fatally checks that `dim` is
// in `[-upper_bound, upper_bound)`. `ET_NORMALIZE_IX(IX, UPPER_BOUND)` yields
// `IX < 0 ? IX + UPPER_BOUND : IX`. Mirrors reduce_util.rs's local definitions.
macro_rules! et_check_valid_dim {
    ($dim:expr, $upper_bound:expr) => {
        et_check_msg!(
            $dim >= -($upper_bound) && $dim < ($upper_bound),
            "invalid dim"
        )
    };
}
macro_rules! et_normalize_ix {
    ($ix:expr, $upper_bound:expr) => {
        if $ix < 0 { $ix + $upper_bound } else { $ix }
    };
}

// PORT-NOTE: input element types Byte/Char/Short/Int (ET_FORALL_INT_TYPES) plus
// Bits16/UInt16 read as uint16_t. The dequant subtraction `input[i] -
// static_cast<int32_t>(zero_point)` is done in the input's promoted integer
// type. `SubI32` reproduces `T - i32` promoting to the wider integer, then the
// result is cast to CTYPE_OUT and multiplied by `float(scale)`. The C++ integer
// promotion of `T - int32_t` widens narrow types to int and keeps int64 at
// int64; here each type subtracts in its own arithmetic and the difference is
// carried as f32 into the output cast (all in-range values are exact through
// f32 for the widths used, matching `static_cast<CTYPE_OUT>(...)` which for the
// float/half out types goes through float anyway).
trait DequantIn: Copy {
    // Compute `static_cast<CTYPE_OUT>(self - int32(zero_point)) * float(scale)`
    // where the subtraction happens in the input's promoted integer type and
    // then is fed through the CTYPE_OUT cast. Returned as the pre-scale float
    // difference so the caller can apply the CTYPE_OUT cast + scale multiply.
    fn sub_i32_as_f32(self, zero_point: i32) -> f32;
}

macro_rules! impl_dequant_in_signed {
    ($t:ty) => {
        impl DequantIn for $t {
            fn sub_i32_as_f32(self, zero_point: i32) -> f32 {
                // `self` promotes to int (or int64 for i64) for the subtraction,
                // exactly as the C++ integer promotion does.
                (self as i64 - zero_point as i64) as f32
            }
        }
    };
}
impl_dequant_in_signed!(u8);
impl_dequant_in_signed!(i8);
impl_dequant_in_signed!(i16);
impl_dequant_in_signed!(i32);
impl_dequant_in_signed!(i64);
impl_dequant_in_signed!(u16);

// PORT-NOTE: output element types Float/Double/Half (ET_FORALL_FLOATH_TYPES_WITH).
// The C++ has TWO distinct cast structures that differ for Half/Double out:
//   * `product_then_cast`: `static_cast<CTYPE_OUT>(diff * float(scale))` — used
//     by dequantize_per_tensor_out and the multi-dim per-channel path. `diff`
//     (integer) times `float(scale)` promotes to float; the float product is
//     then cast to CTYPE_OUT.
//   * `cast_then_multiply`: `static_cast<CTYPE_OUT>(diff) * float(scale)` — used
//     ONLY by the `input.dim() == 1` per-channel path. `diff` is cast to
//     CTYPE_OUT first, then multiplied by `float(scale)` (with usual arithmetic
//     conversions), then assigned back to a CTYPE_OUT slot. For Double this
//     multiplies in double; for Float/Half it multiplies in float. These two
//     forms are bit-identical for Float but differ for Double/Half, so both are
//     preserved.
// `diff` is carried as the pre-scale f32 difference (all in-range integer widths
// used here are exact through f32).
trait DequantOut: Copy {
    fn product_then_cast(diff: f32, scale: f32) -> Self;
    fn cast_then_multiply(diff: f32, scale: f32) -> Self;
}
impl DequantOut for f32 {
    fn product_then_cast(diff: f32, scale: f32) -> Self {
        diff * scale
    }
    fn cast_then_multiply(diff: f32, scale: f32) -> Self {
        diff * scale
    }
}
impl DequantOut for f64 {
    fn product_then_cast(diff: f32, scale: f32) -> Self {
        // static_cast<double>(diff * float(scale)): multiply in float, then
        // widen the float product to double.
        (diff * scale) as f64
    }
    fn cast_then_multiply(diff: f32, scale: f32) -> Self {
        // static_cast<double>(diff) * float(scale): the double times float
        // promotes the float scale to double.
        diff as f64 * scale as f64
    }
}
impl DequantOut for crate::runtime::core::portable_type::Half {
    fn product_then_cast(diff: f32, scale: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(diff * scale)
    }
    fn cast_then_multiply(diff: f32, scale: f32) -> Self {
        // static_cast<Half>(diff) * float(scale): Half promotes to float for the
        // multiply, then the float result narrows back to Half on assignment.
        let h = crate::runtime::core::portable_type::Half::from_f32(diff);
        crate::runtime::core::portable_type::Half::from_f32(h.to_f32() * scale)
    }
}

/// Asserts that the parameters are valid.
// [spec:et:def:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]
fn check_dequantize_per_tensor_args(
    input: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: &Option<ScalarType>,
    out: &Tensor,
) {
    et_check_msg!(
        input.scalar_type() == ScalarType::Byte
            || input.scalar_type() == ScalarType::Char
            || input.scalar_type() == ScalarType::Bits16
            || input.scalar_type() == ScalarType::UInt16
            || input.scalar_type() == ScalarType::Short
            || input.scalar_type() == ScalarType::Int,
        "input.scalar_type() is not supported:"
    );

    et_check_msg!(
        input.scalar_type() == dtype,
        "input.scalar_type() is not matching dtype argumenta:"
    );

    if let Some(od) = out_dtype {
        et_check_msg!(
            out.scalar_type() == *od,
            "output_dtype must match the dtype of the out tensor"
        );
    }

    et_check_msg!(
        quant_min <= quant_max,
        "quant min is greater than quant max"
    );
}

/// Useful to reduce a tensor `in` over a given dimension `dim` using the reduce
/// function `fn`, which should have the signature
/// `void fn(const size_t size, const size_t stride, const size_t base_ix)`.
// [spec:et:def:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn]
fn apply_over_unpacked_dim<Fn: FnMut(usize, usize, usize)>(mut fn_: Fn, in_: &Tensor, dim: i64) {
    if in_.numel() == 0 {
        return;
    }

    et_check_msg!(
        in_.dim() > 0,
        "Input tensor must have at least one dimension"
    );
    et_check_valid_dim!(dim, in_.dim() as i64);

    let d: usize = et_normalize_ix!(dim, in_.dim() as i64) as usize;
    let dim_size: usize = in_.size(d as isize) as usize;
    let outer_size: usize = getLeadingDims(in_, d as i64);
    let inner_size: usize = getTrailingDims(in_, d as i64);
    // Loop through all outer dimensions
    for outer_idx in 0..outer_size {
        // Loop through dim
        for unpacked_dim_idx in 0..dim_size {
            fn_(inner_size, outer_idx, unpacked_dim_idx);
        }
    }
}

// [spec:et:def:op-dequantize.torch.executor.native.dequantize-optimized-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.dequantize-optimized-fn]
///
/// # Safety
/// `in` and `out` must point to at least `numel` valid elements of their types.
//
// PORT-NOTE: the `#if defined(__aarch64__) || defined(__ARM_NEON)` NEON vector
// loop is a hardware-specific fast path whose result is (per the C++ comment and
// the spec) numerically equivalent to the scalar remainder loop. The scalar loop
// is ported here; the NEON intrinsics loop is omitted (its only effect is
// performance). `i` starts at 0 so the scalar loop covers all elements, matching
// the non-NEON build.
unsafe fn dequantize_optimized(
    in_: *const i8,
    scale: f64,
    zero_point: i64,
    out: *mut f32,
    quant_min: i64,
    quant_max: i64,
    numel: usize,
) {
    et_check_msg!(zero_point >= quant_min, "zero_point must be <= quant_min");
    et_check_msg!(zero_point <= quant_max, "zero_point must be >= quant_max");
    // PORT-NOTE: the scalar C++ line is `out[i] = (in[i] - zero_point) * scale;`
    // where `in[i]` (int8) and `zero_point` (int64) subtract in int64, the
    // product is `int64 * double`, and the double result narrows to float on
    // store. Reproduced as `(in - zp) as f64 * scale` narrowed to f32.
    let mut i: usize = 0;
    while i < numel {
        unsafe {
            *out.add(i) = ((*in_.add(i) as i64 - zero_point) as f64 * scale) as f32;
        }
        i += 1;
    }
}

// [spec:et:def:op-dequantize.torch.executor.native.get-scale-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.get-scale-fn]
fn get_scale(scale: &Tensor, channel_ix: usize) -> f32 {
    et_check_msg!(
        (scale.scalar_type() == ScalarType::Double) || (scale.scalar_type() == ScalarType::Float),
        "scale.scalar_type() is not double or float type"
    );
    if scale.scalar_type() == ScalarType::Double {
        unsafe { *scale.const_data_ptr::<f64>().add(channel_ix) as f32 }
    } else {
        unsafe { *scale.const_data_ptr::<f32>().add(channel_ix) }
    }
}

// [spec:et:def:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]
fn can_use_optimized_dequantize_per_channel(
    in_: &Tensor,
    in_dtype: ScalarType,
    out_dtype: &Option<ScalarType>,
) -> bool {
    let is_contiguous: bool;
    #[cfg(feature = "aten")]
    {
        // PORT-NOTE: C++ (USE_ATEN_LIB) calls `at::Tensor::is_contiguous()`. The
        // port has no separate `at::Tensor`; mirror the ATen strided-contiguity
        // check with the ET-equivalent free function `tensor_is_contiguous`,
        // matching how other aten-gated paths substitute ET helpers for
        // at::Tensor methods.
        is_contiguous = tensor_is_contiguous(in_);
    }
    #[cfg(not(feature = "aten"))]
    {
        is_contiguous =
            unsafe { is_contiguous_dim_order(in_.dim_order().data(), in_.dim() as usize) };
    }
    if !is_contiguous
        || (in_dtype != ScalarType::Char)
        || (out_dtype.is_some() && out_dtype.unwrap() != ScalarType::Float)
    {
        return false;
    }
    true
}

// [spec:et:def:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn]
#[allow(clippy::too_many_arguments)]
fn dequantize_per_channel_optimized(
    in_: &Tensor,
    scales: &Tensor,
    opt_zero_points: &Option<&Tensor>,
    out: &Tensor,
    axis: i64,
    quant_min: i64,
    quant_max: i64,
    in_dtype: ScalarType,
    out_dtype: &Option<ScalarType>,
) {
    check_dequantize_per_tensor_args(in_, quant_min, quant_max, in_dtype, out_dtype, out);
    et_check_msg!(
        in_dtype == ScalarType::Char,
        "in.scalar_type() is not supported:"
    );
    if let Some(od) = out_dtype {
        et_check_msg!(*od == ScalarType::Float, "Only float output is supported");
    }
    let in_data: *const i8 = in_.const_data_ptr::<i8>();
    let out_data: *mut f32 = out.mutable_data_ptr::<f32>();
    let zero_points_data: *const i64 = if let Some(zp) = opt_zero_points {
        zp.const_data_ptr::<i64>()
    } else {
        core::ptr::null()
    };
    let axis_stride: StridesType = *in_.strides().at(axis as usize);
    let outer_stride: StridesType = in_.size(axis as isize) as StridesType * axis_stride;
    apply_over_unpacked_dim(
        |numel: usize, outer_idx: usize, unpacked_dim_idx: usize| {
            let in_data_local: *const i8 = unsafe {
                in_data.offset(
                    outer_idx as isize * outer_stride as isize
                        + unpacked_dim_idx as isize * axis_stride as isize,
                )
            };
            let scale: f64 = get_scale(scales, unpacked_dim_idx) as f64;
            let zero_point: i64 = if !zero_points_data.is_null() {
                unsafe { *zero_points_data.add(unpacked_dim_idx) }
            } else {
                0
            };
            let out_data_local: *mut f32 = unsafe {
                out_data.offset(
                    outer_idx as isize * outer_stride as isize
                        + unpacked_dim_idx as isize * axis_stride as isize,
                )
            };
            unsafe {
                dequantize_optimized(
                    in_data_local,
                    scale,
                    zero_point,
                    out_data_local,
                    quant_min,
                    quant_max,
                    numel,
                );
            }
        },
        in_,
        axis,
    );
}

/// Dequantizes the input tensor according to the formula (input - zero_point) *
/// scale
///
/// NOTE: quant_min and quant_max are not used in computation, but rather
/// metadata that is passed around which can be useful for pattern matching.
// [spec:et:def:op-dequantize.torch.executor.native.dequantize-per-tensor-out-fn]
// [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-tensor-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_tensor_out<'a, 'b>(
    input: &Tensor,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in dequantize_per_tensor_out"
    );

    check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype, &out_dtype, out);

    // calculate the dequantized output, cast scale to float to match fbgemm
    // behavior
    // PORT-NOTE: the C++ `DEQUANTIZE_IMPL` / `CALCULATE_INT_TYPE` macro pair over
    // ET_FORALL_INT_TYPES + Bits16/UInt16 (input) and ET_FORALL_FLOATH_TYPES_WITH
    // (output) is expanded here as explicit nested matches. The `default` arms
    // fire the fatal `ET_CHECK_MSG(false, ...)`.
    fn dequantize_impl<IN: DequantIn, OUT: DequantOut>(
        input: &Tensor,
        out: &Tensor,
        scale: f64,
        zero_point: i64,
    ) {
        let out_data_ptr: *mut OUT = out.mutable_data_ptr::<OUT>();
        let input_data_ptr: *const IN = input.const_data_ptr::<IN>();
        let input_numel = input.numel();
        for i in 0..(input_numel as usize) {
            let diff: f32 = unsafe { *input_data_ptr.add(i) }.sub_i32_as_f32(zero_point as i32);
            unsafe {
                *out_data_ptr.add(i) = OUT::product_then_cast(diff, scale as f32);
            }
        }
    }

    macro_rules! calculate_int_type {
        ($in:ty) => {
            match out.scalar_type() {
                ScalarType::Float => dequantize_impl::<$in, f32>(input, out, scale, zero_point),
                ScalarType::Double => dequantize_impl::<$in, f64>(input, out, scale, zero_point),
                ScalarType::Half => {
                    dequantize_impl::<$in, crate::runtime::core::portable_type::Half>(
                        input, out, scale, zero_point,
                    )
                }
                _ => {
                    et_check_msg!(false, "Unhandled output dtype");
                }
            }
        };
    }

    match input.scalar_type() {
        ScalarType::Byte => calculate_int_type!(u8),
        ScalarType::Char => calculate_int_type!(i8),
        ScalarType::Short => calculate_int_type!(i16),
        ScalarType::Int => calculate_int_type!(i32),
        ScalarType::Bits16 => calculate_int_type!(u16),
        ScalarType::UInt16 => calculate_int_type!(u16),
        _ => {
            et_check_msg!(false, "Unhandled input dtype");
        }
    }

    out
}

// [spec:et:def:op-dequantize.torch.executor.dequantize-per-tensor-tensor-args-out-fn]
// [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-tensor-args-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_tensor_tensor_args_out<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    et_check_msg!(
        scale.scalar_type() == ScalarType::Double,
        "Expected scale to be Double tensor received:"
    );
    et_check_msg!(
        zero_point.scalar_type() == ScalarType::Long,
        "Expected scale to be Long tensor received:"
    );
    et_check_msg!(
        scale.numel() == 1,
        "Exepcted scale to only have one element received:"
    );
    et_check_msg!(
        zero_point.numel() == 1,
        "Exepcted zero_point to only have one element received:"
    );

    dequantize_per_tensor_out(
        input,
        unsafe { *scale.const_data_ptr::<f64>().add(0) },
        unsafe { *zero_point.const_data_ptr::<i64>().add(0) },
        quant_min,
        quant_max,
        dtype,
        out_dtype,
        out,
    );
    out
}

// [spec:et:def:op-dequantize.torch.executor.dequantize-per-channel-out-fn]
// [spec:et:sem:op-dequantize.torch.executor.dequantize-per-channel-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_channel_out<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    opt_zero_points: &Option<&Tensor>,
    mut axis: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // normalize axis
    et_check_msg!(
        tensor_has_dim(input, axis),
        "axis is not legal it should be -input.dim() <= axis < input.dim()"
    );

    if axis < 0 {
        axis += nonzero_dim(input) as i64;
    }

    et_check_msg!(
        scale.numel() == input.size(axis as isize),
        "scale.numel() != input.size(axis)"
    );

    if let Some(zero_point) = opt_zero_points {
        et_check_msg!(
            zero_point.scalar_type() == ScalarType::Int
                || zero_point.scalar_type() == ScalarType::Long,
            "zero_point.scalar_type() is not integer type"
        );

        et_check_msg!(
            zero_point.numel() == input.size(axis as isize),
            "zero_point.numel() != input.size(axis)"
        );
    }

    check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype, &out_dtype, out);

    if can_use_optimized_dequantize_per_channel(input, dtype, &out_dtype) {
        dequantize_per_channel_optimized(
            input,
            scale,
            opt_zero_points,
            out,
            axis,
            quant_min,
            quant_max,
            dtype,
            &out_dtype,
        );
        return out;
    }

    // a list contains all dimensions except axis
    let mut dims: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    {
        let mut i: i64 = 0;
        while i < input.dim() as i64 - 1 {
            if i < axis {
                dims[i as usize] = i;
            } else {
                dims[i as usize] = i + 1;
            }
            i += 1;
        }
    }
    let zero_point_data: *const i64 = if let Some(zp) = opt_zero_points {
        zp.const_data_ptr::<i64>()
    } else {
        core::ptr::null()
    };

    let optional_dim_list: Option<ArrayRef<i64>> = Some(ArrayRef::from_raw_parts(
        dims.as_ptr(),
        (input.dim() - 1) as usize,
    ));

    // Actual dequantization logic
    // PORT-NOTE: the C++ `DEQUANTIZE_IMPL` / `CALCULATE_FLOAT_TYPE` macro pair
    // over ET_FORALL_INT_TYPES + Bits16/UInt16 (input) and
    // ET_FORALL_FLOATH_TYPES_WITH (output) is expanded here as explicit nested
    // matches. The `input.dim() == 1` branch iterates via apply_over_dim (whole
    // tensor); the multi-dim branch loops over channels via apply_over_dim_list.
    fn dequantize_impl<IN: DequantIn, OUT: DequantOut>(
        input: &Tensor,
        out: &Tensor,
        scale: &Tensor,
        axis: i64,
        zero_point_data: *const i64,
        optional_dim_list: &Option<ArrayRef<i64>>,
    ) {
        if input.dim() == 1 {
            let out_data_ptr: *mut OUT = out.mutable_data_ptr::<OUT>();
            let input_data_ptr: *const IN = input.const_data_ptr::<IN>();
            et_check_msg!(axis == 0, "Axis must be 0 for a single dimensional tensors");
            let dim: Option<i64> = None;
            apply_over_dim_whole(
                |numel: usize, stride: usize, base_ix: usize| {
                    for i in 0..numel {
                        let current_ix: usize = base_ix * stride + i;
                        let scale_v: f32 = get_scale(scale, current_ix);
                        let mut zero_point: i64 = 0;
                        if !zero_point_data.is_null() {
                            zero_point = unsafe { *zero_point_data.add(current_ix) };
                        }
                        let diff: f32 = unsafe { *input_data_ptr.add(current_ix) }
                            .sub_i32_as_f32(zero_point as i32);
                        unsafe {
                            *out_data_ptr.add(current_ix) = OUT::cast_then_multiply(diff, scale_v);
                        }
                    }
                },
                input,
                &dim,
            );
            return;
        }
        let mut channel_ix: usize = 0;
        while channel_ix < input.size(axis as isize) as usize {
            let scale_v: f32 = get_scale(scale, channel_ix);
            let mut zero_point: i64 = 0;
            if !zero_point_data.is_null() {
                zero_point = unsafe { *zero_point_data.add(channel_ix) };
            }
            let out_data_ptr: *mut OUT = out.mutable_data_ptr::<OUT>();
            let input_data_ptr: *const IN = input.const_data_ptr::<IN>();
            apply_over_dim_list(
                |in_ix: usize| {
                    let diff: f32 =
                        unsafe { *input_data_ptr.add(in_ix) }.sub_i32_as_f32(zero_point as i32);
                    unsafe {
                        *out_data_ptr.add(in_ix) = OUT::product_then_cast(diff, scale_v);
                    }
                },
                input,
                optional_dim_list,
                channel_ix,
                0,
                -1,
            );
            channel_ix += 1;
        }
    }

    macro_rules! calculate_float_type {
        ($in:ty) => {
            match out.scalar_type() {
                ScalarType::Float => dequantize_impl::<$in, f32>(
                    input,
                    out,
                    scale,
                    axis,
                    zero_point_data,
                    &optional_dim_list,
                ),
                ScalarType::Double => dequantize_impl::<$in, f64>(
                    input,
                    out,
                    scale,
                    axis,
                    zero_point_data,
                    &optional_dim_list,
                ),
                ScalarType::Half => {
                    dequantize_impl::<$in, crate::runtime::core::portable_type::Half>(
                        input,
                        out,
                        scale,
                        axis,
                        zero_point_data,
                        &optional_dim_list,
                    )
                }
                _ => {
                    et_check_msg!(false, "Unhandled output dtype");
                }
            }
        };
    }

    match input.scalar_type() {
        ScalarType::Byte => calculate_float_type!(u8),
        ScalarType::Char => calculate_float_type!(i8),
        ScalarType::Short => calculate_float_type!(i16),
        ScalarType::Int => calculate_float_type!(i32),
        ScalarType::Bits16 => calculate_float_type!(u16),
        ScalarType::UInt16 => calculate_float_type!(u16),
        _ => {
            et_check_msg!(false, "Unhandled input dtype");
        }
    }

    out
}

pub fn dequantize_per_channel_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    opt_zero_points: &Option<&Tensor>,
    axis: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = context;
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in dequantize_per_channel_out"
    );

    dequantize_per_channel_out(
        input,
        scale,
        opt_zero_points,
        axis,
        quant_min,
        quant_max,
        dtype,
        out_dtype,
        out,
    )
}

// [spec:et:def:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]
// [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_tensor_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let _ = context;
    dequantize_per_tensor_out(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_tensor_tensor_args_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let _ = context;
    dequantize_per_tensor_tensor_args_out(
        input, scale, zero_point, quant_min, quant_max, dtype, out_dtype, out,
    )
}

// [spec:et:def:op-dequantize.torch.executor.dequantize-per-token-out-fn]
// [spec:et:sem:op-dequantize.torch.executor.dequantize-per-token-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn dequantize_per_token_out<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    zero_points: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Refactor this into a util
    let mut num_channels: usize = 1;
    {
        let mut i: usize = 0;
        while (i as isize) < input.dim() - 1 {
            num_channels *= input.size(i as isize) as usize;
            i += 1;
        }
    }
    // This unfortunate change is needed because we compile op_quantize for aten
    // mode as well
    let mut input_sizes: [SizesType; 2] = [0; 2];
    input_sizes[0] = num_channels as SizesType;
    input_sizes[1] = input.size(input.dim() - 1) as SizesType;

    // PORT-NOTE: the ATen `at::from_blob` reshape branch is behind `USE_ATEN_LIB`;
    // the portable branch builds a rank-2 view TensorImpl over `input`'s data
    // and (portable-only) resizes `out`. Ported unconditionally as the portable
    // branch (this crate does not build the ATen tensor path here).
    let mut input_dim_order: [DimOrderType; 2] = [0, 1];
    let mut input_strides: [StridesType; 2] = [0; 2];
    unsafe {
        dim_order_to_stride_nocheck(
            input_sizes.as_ptr(),
            input_dim_order.as_ptr(),
            2,
            input_strides.as_mut_ptr(),
        );
    }
    let input_data: *mut core::ffi::c_void = input.mutable_data_ptr_typed();
    let mut reshaped_input_impl: TensorImpl = TensorImpl::new(
        input.scalar_type(),
        2,
        input_sizes.as_mut_ptr(),
        input_data,
        input_dim_order.as_mut_ptr(),
        input_strides.as_mut_ptr(),
        TensorShapeDynamism::STATIC,
        DeviceType::CPU,
        0 as DeviceIndex,
    );
    let reshaped_input: Tensor = Tensor::new(&mut reshaped_input_impl);
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in dequantize_per_channel_out"
    );

    dequantize_per_channel_out(
        &reshaped_input,
        scale,
        &Some(zero_points),
        0, /* axis */
        quant_min,
        quant_max,
        dtype,
        Some(out_dtype),
        out,
    )
}

pub fn dequantize_per_token_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    zero_points: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out_dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = context;
    dequantize_per_token_out(
        input,
        scale,
        zero_points,
        quant_min,
        quant_max,
        dtype,
        out_dtype,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::portable_type::bits_types::bits16;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    // Element-type constructor from an integer literal, mirroring the C++
    // `TensorFactory<DTYPE>::full`/`make` implicit int conversion for the
    // integer input dtypes.
    trait FromI64Ctor: Copy {
        fn ctor(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_ctor {
        ($($t:ty),*) => {$(impl FromI64Ctor for $t { fn ctor(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_ctor!(u8, i8, i16, i32, u16);
    impl FromI64Ctor for bits16 {
        fn ctor(v: i64) -> Self {
            bits16::new(v as u16)
        }
    }

    // Output element-type constructor from a double, for the Float/Double/Half
    // output dtypes.
    trait FromF64Ctor: Copy {
        fn ctor(v: f64) -> Self;
    }
    impl FromF64Ctor for f32 {
        fn ctor(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64Ctor for f64 {
        fn ctor(v: f64) -> Self {
            v
        }
    }
    impl FromF64Ctor for Half {
        fn ctor(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Ctor for BFloat16 {
        fn ctor(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    /// A generic smoke test that works for any dtype that supports ones() and
    /// zeros().
    //
    // The context wrapper's sem id (`native.dequantize-per-tensor-out-fn`) shares
    // this function body, which is exercised directly here, and
    // check_dequantize_per_tensor_args runs on every call.
    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-tensor-out-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn/test]
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64Ctor,
    {
        let tf = TensorFactory::<T>::new();

        let input = tf.full(vec![3, 5], T::ctor(100), STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 30;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f32>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // (100 - 30) * 0.5
        let expected = tfo.full(vec![3, 5], 35.0, STATIC);
        dequantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            T::VALUE,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    #[test]
    fn op_dequantize_out_test_all_dtypes_supported() {
        crate::runtime::platform::runtime::runtime_init();
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<bits16>();
        test_dtype::<u16>();
        test_dtype::<i32>();
    }

    /// Test all supported output dtypes for dequantization
    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn/test]
    fn test_output_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Ctor,
    {
        let tf = TensorFactory::<u8>::new();

        let input = tf.full(vec![3, 5], 100, STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 30;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<T>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // (100 - 30) * 0.5 = 35
        let expected = tfo.full(vec![3, 5], T::ctor(35.0), STATIC);
        dequantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            Some(T::VALUE),
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    #[test]
    fn op_dequantize_out_test_all_output_dtypes_supported() {
        crate::runtime::platform::runtime::runtime_init();
        test_output_dtype::<f32>();
        test_output_dtype::<f64>();
        test_output_dtype::<Half>();
    }

    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn/test]
    #[test]
    fn op_dequantize_out_test_half_output() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<u8>::new();

        let input = tf.full(vec![3, 5], 10, STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 100000;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<Half>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // (10 - 100000) * 0.5 = -49995
        dequantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            Some(ScalarType::Half),
            &out,
        );

        // The expected result should be (10 - 100000) * 0.5 = -49995
        let expected = tfo.full(vec![3, 5], Half::from_f32(-49995.0), STATIC);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn/test]
    #[test]
    fn op_dequantize_out_test_double_output() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<u8>::new();

        let input = tf.full(vec![3, 5], 10, STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 100000;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f64>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        dequantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            Some(ScalarType::Double),
            &out,
        );

        // The expected result should be (10 - 100000) * 0.5 = -49995
        let expected = tfo.full(vec![3, 5], -49995.0, STATIC);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn/test]
    #[test]
    fn op_dequantize_out_test_non_whole_numbers() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<u8>::new();

        let input = tf.full(vec![3, 5], 100, STATIC);
        let scale: f64 = 0.45;
        let zero_point: i64 = 30;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f32>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // (100 - 30) * 0.5
        let expected = tfo.full(vec![3, 5], 31.5, STATIC);
        dequantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-tensor-args-out-fn/test]
    #[test]
    fn op_dequantize_out_test_tensor_arg_overload() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_byte.full(vec![3, 5], 100, STATIC);
        let scale = tf_double.make_default(vec![1], vec![0.45]);
        let zero_point = tf_long.make_default(vec![1], vec![30]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f32>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // (100 - 30) * 0.5
        let expected = tfo.full(vec![3, 5], 31.5, STATIC);
        dequantize_per_tensor_tensor_args_out(
            &input,
            &scale,
            &zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // The i8/Char + Float + contiguous instantiation takes the optimized
    // per-channel path (can_use_optimized_dequantize_per_channel ->
    // dequantize_per_channel_optimized -> apply_over_unpacked_dim ->
    // dequantize_optimized), while both instantiations exercise get_scale and
    // check_dequantize_per_tensor_args.
    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-channel-out-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.get-scale-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn/test]
    // [spec:et:sem:op-dequantize.torch.executor.native.dequantize-optimized-fn/test]
    fn test_per_channel_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64Ctor,
    {
        let tf = TensorFactory::<T>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let mut input = tf.full(vec![3, 2], T::ctor(100), STATIC);
        let mut scale = tf_double.make_default(vec![2], vec![0.5, 1.0]);
        let mut zero_point = tf_long.make_default(vec![2], vec![30, 60]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f32>::new();
        let mut out = tfo.zeros(vec![3, 2], STATIC);
        // (100 - 30) * 0.5
        // (100 - 60) * 1
        let mut expected = tfo.make_default(vec![3, 2], vec![35.0, 40.0, 35.0, 40.0, 35.0, 40.0]);
        dequantize_per_channel_out(
            &input,
            &scale,
            &Some(&zero_point),
            /*axis=*/ 1,
            quant_min,
            quant_max,
            T::VALUE,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);

        // Test with a different axis
        out = tfo.zeros(vec![3, 2], STATIC);
        scale = tf_double.make_default(vec![3], vec![0.5, 0.75, 1.0]);
        zero_point = tf_long.make_default(vec![3], vec![30, 50, 60]);
        // (100 - 30) * 0.5
        // (100 - 50) * 0.75
        // (100 - 60) * 1
        expected = tfo.make_default(vec![3, 2], vec![35.0, 35.0, 37.5, 37.5, 40.0, 40.0]);
        dequantize_per_channel_out(
            &input,
            &scale,
            &Some(&zero_point),
            /*axis=*/ 0,
            quant_min,
            quant_max,
            T::VALUE,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);

        // Test with a different axis
        out = tfo.zeros(vec![3], STATIC);
        input = tf.make_default(vec![3], vec![T::ctor(100), T::ctor(100), T::ctor(100)]);
        scale = tf_double.make_default(vec![3], vec![0.5, 0.75, 1.0]);
        zero_point = tf_long.make_default(vec![3], vec![30, 50, 60]);
        // (100 - 30) * 0.5
        // (100 - 50) * 0.75
        // (100 - 60) * 1
        expected = tfo.make_default(vec![3], vec![35.0, 37.5, 40.0]);
        dequantize_per_channel_out(
            &input,
            &scale,
            &Some(&zero_point),
            /*axis=*/ 0,
            quant_min,
            quant_max,
            T::VALUE,
            None,
            &out,
        );
        assert_tensor_eq!(out, expected);

        // Test with a different axis
        input = tf.full(vec![3, 19], T::ctor(100), STATIC);
        out = tfo.zeros(vec![3, 19], STATIC);
        scale = tf_double.make_default(vec![3], vec![0.5, 0.75, 1.0]);
        zero_point = tf_long.make_default(vec![3], vec![30, 50, 60]);
        // (100 - 30) * 0.5
        // (100 - 50) * 0.75
        // (100 - 60) * 1
        expected = tfo.make_default(
            vec![3, 19],
            vec![
                35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0,
                35.0, 35.0, 35.0, 35.0, 35.0, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5,
                37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 37.5, 40.0, 40.0, 40.0, 40.0,
                40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
                40.0,
            ],
        );
        dequantize_per_channel_out(
            &input,
            &scale,
            &Some(&zero_point),
            /*axis=*/ 0,
            quant_min,
            quant_max,
            T::VALUE,
            None,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    #[test]
    fn op_dequantize_out_test_dequantize_per_channel() {
        crate::runtime::platform::runtime::runtime_init();
        test_per_channel_dtype::<u8>();
        test_per_channel_dtype::<i8>();
    }

    // PORT-NOTE: the C++ op_dequantize_test.cpp suite has no per-token case, so
    // this focused test pins the ported `dequantize_per_token_out` against the
    // sem rule: each token (last-dim vector) uses `scale[row]`/`zero_point[row]`,
    // computing `(input - zp) * scale` per token. Input [2, 3] flattens to 2
    // tokens; row 0 uses scale 0.5 / zp 10, row 1 uses scale 1.0 / zp 20.
    // [spec:et:sem:op-dequantize.torch.executor.dequantize-per-token-out-fn/test]
    #[test]
    fn op_dequantize_out_test_dequantize_per_token() {
        crate::runtime::platform::runtime::runtime_init();
        let tf_byte = TensorFactory::<u8>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_byte.full(vec![2, 3], 100, STATIC);
        let scale = tf_double.make_default(vec![2, 1], vec![0.5, 1.0]);
        let zero_point = tf_long.make_default(vec![2, 1], vec![10, 20]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<f32>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // row 0: (100 - 10) * 0.5 = 45
        // row 1: (100 - 20) * 1.0 = 80
        let expected = tfo.make_default(vec![2, 3], vec![45.0, 45.0, 45.0, 80.0, 80.0, 80.0]);
        dequantize_per_token_out(
            &input,
            &scale,
            &zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            ScalarType::Float,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }
}
