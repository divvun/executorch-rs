//! Literal port of backends/xnnpack/runtime/utils/utils.cpp +
//! backends/xnnpack/runtime/utils/utils.h.
//!
//! PORT-NOTE: This module is pure ExecuTorch/math and does not itself depend on
//! the XNNPACK C API, so it is not feature-gated behind `xnnpack`. The
//! `__aarch64__` NEON path is compiled behind `#[cfg(target_arch =
//! "aarch64")]` using `core::arch::aarch64` intrinsics that mirror the C++
//! `<arm_neon.h>` calls one-for-one.
#![allow(non_snake_case)]

use crate::runtime::core::error::Error;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;

// PORT-NOTE: `ET_CHECK_MSG` (used by `GetMinMax`) aborts the process on failure.
// It is not exported by the platform-core group as a name-shared macro; mirror
// it here as a file-local macro over `runtime_abort`, matching the established
// pattern in the portable kernels.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {{
        if !($cond) {
            crate::et_log!(Error, $($arg)*);
            crate::runtime::platform::abort::runtime_abort();
        }
    }};
}

// constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;
const SMALL_SCALE_THRESHOLD: f32 = 6.1e-5f32;

// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantization-params]
pub struct QuantizationParams {
    pub scale: f64,
    pub zero_point: i32,
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn]
//
// PORT-NOTE: The C++ has default arguments `preserve_sparsity = false,
// force_scale_power_of_two = false, reduce_range = false`. Rust has no default
// arguments; the header's out-of-line declaration drops the defaults, so all
// callers pass them explicitly.
#[must_use]
pub fn ChooseQuantizationParams(
    mut min: f32,
    mut max: f32,
    mut qmin: i32,
    mut qmax: i32,
    result: &mut QuantizationParams,
    preserve_sparsity: bool,
    force_scale_power_of_two: bool,
    reduce_range: bool,
) -> Error {
    crate::et_check_or_return_error!(
        min <= max,
        Internal,
        "In ChooseQuantizationParams, min should be less than or equal to max. min: {}, max: {}",
        min,
        max
    );

    if reduce_range {
        qmin = qmin / 2;
        qmax = qmax / 2;
    }
    if min < 0.0 && max > 0.0 && preserve_sparsity {
        let symmetric_qmin: i32 = -((qmax - qmin) / 2 + 1);
        let symmetric_qmax: i32 = (qmax - qmin) / 2;
        let max_scale: f64 = f64::max(
            (min as f64 / symmetric_qmin as f64).abs(),
            (max as f64 / symmetric_qmax as f64).abs(),
        );
        min = (max_scale * symmetric_qmin as f64) as f32;
        max = (max_scale * symmetric_qmax as f64) as f32;
    }

    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min = f32::min(min, 0.0f32);
    max = f32::max(max, 0.0f32);

    crate::et_check_or_return_error!(
        qmin < qmax,
        Internal,
        "In ChooseQuantizationParams, qmin should be less than qmax"
    );

    // Use double precision for intermediate computation but use single precision
    // in final number to reflect the actual number used during quantization.
    let mut scale: f64 = (max as f64 - min as f64) / (qmax - qmin) as f64;
    // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
    // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
    // infinity because some of fbgemm code pre-computes scale's reciprocal to do
    // multiplication instead of division in the time critical part of code.
    if (scale as f32) == 0.0f32 || (1.0f32 / (scale as f32)).is_infinite() {
        scale = 0.1;
    }
    crate::et_check_or_return_error!(scale > 0.0, Internal, "quantization scale should be > 0");

    if force_scale_power_of_two {
        if scale < 1.0 {
            scale = 1.0 / ((1i32 << ((1.0 / scale).ln() / 2.0f64.ln()).floor() as i32) as f64);
        } else {
            scale = (1i32 << (scale.ln() / 2.0f64.ln()).ceil() as i32) as f64;
        }
    }

    // Cut off small scale
    if scale < SMALL_SCALE_THRESHOLD as f64 {
        let org_scale: f32 = scale as f32;
        scale = SMALL_SCALE_THRESHOLD as f64;
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
    let zero_point_from_min: f64 = qmin as f64 - min as f64 / scale;
    let zero_point_from_max: f64 = qmax as f64 - max as f64 / scale;
    let zero_point_from_min_error: f64 = (qmin as f64).abs() - (min as f64 / scale).abs();
    let zero_point_from_max_error: f64 = (qmax as f64).abs() - (max as f64 / scale).abs();
    let mut initial_zero_point: f64 = if zero_point_from_min_error < zero_point_from_max_error {
        zero_point_from_min
    } else {
        zero_point_from_max
    };

    // for symmetric quantization (preserve_sparsity == true), we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if min < 0.0 && max > 0.0 && preserve_sparsity {
        initial_zero_point = (qmin + qmax) as f64 / 2.0;
    }

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
        nudged_zero_point = libm::rint(initial_zero_point) as i32;
    }

    result.scale = scale;
    result.zero_point = nudged_zero_point;
    Error::Ok
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.generate-requantization-scale-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.generate-requantization-scale-fn]
#[must_use]
pub fn GenerateRequantizationScale(
    weight_scales: &Tensor,
    input_scale: f32,
    output_scale: f32,
    requant_scales: &mut Vec<f32>,
) -> Error {
    // Since weight scale is allocated with padding
    // weight_scales.numel() gives us padded num elements.
    let num_output_channels_padded = weight_scales.numel();
    let weight_scales_data = weight_scales.const_data_ptr::<f32>();
    if (requant_scales.len() as i64) < num_output_channels_padded as i64 {
        requant_scales.resize(num_output_channels_padded as usize, 0.0f32);
    }
    for i in 0..num_output_channels_padded {
        let inverse_output_scale = 1.0f32 / output_scale;
        requant_scales[i as usize] =
            (unsafe { *weight_scales_data.offset(i) } * input_scale) * inverse_output_scale;
        crate::et_check_or_return_error!(
            requant_scales[i as usize] > 0.0f32 && requant_scales[i as usize].is_normal(),
            Internal,
            "failed to create op with requantization scale"
        );
    }
    Error::Ok
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.get-min-max-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.get-min-max-fn]
pub fn GetMinMax(ft: &Tensor) -> (f32, f32) {
    let mut min: f32 = f32::MAX;
    let mut max: f32 = -f32::MAX;
    et_check_msg!(
        ft.scalar_type() == ScalarType::Float,
        "Expected float tensor but got {}",
        ft.scalar_type() as i8
    );
    let d = ft.const_data_ptr::<f32>();
    for i in 0..ft.numel() {
        let di = unsafe { *d.offset(i) };
        min = if di < min { di } else { min };
        max = if di > max { di } else { max };
    }
    (min, max)
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.round-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.round-fn]
//
// PORT-NOTE: The C++ has two build variants (old-Android non-NDK vs. generic);
// both reduce to `nearbyint`. Every call site uses `T = f32` (`Round(value *
// inv_scale)`), so this port provides the single `f32` version calling
// `libm::nearbyintf`, i.e. `std::nearbyint(f32)`.
#[inline]
pub fn Round(x: f32) -> f32 {
    // PORT-NOTE: `std::nearbyint` maps to `libm::rintf`; both are
    // round-to-nearest under the current FP mode (ties-to-even by default). The
    // only difference (`nearbyint` suppresses FE_INEXACT) is unobservable here.
    libm::rintf(x)
}

// PORT-NOTE: `quantize_val<T>` and the aarch64 helpers are templates over `T ∈
// {uint8_t, int8_t}`. The type-parameter set is captured by the `Quantizable`
// trait, which also carries the NEON store/narrow specializations
// (`vqmov`/`vst1`) on aarch64 so a single generic body stands in for the C++
// explicit specializations. Only `u8`/`i8` implement it, mirroring the C++
// link-time restriction to those two instantiations.
// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-val-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn]
pub trait Quantizable: Copy {
    const QMIN: i64;
    const QMAX: i64;
    fn from_i64(v: i64) -> Self;

    /// Lane vector type produced by the NEON narrow (`underlying_x8_t`).
    #[cfg(target_arch = "aarch64")]
    type Lane: Copy;

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-fn]
    //
    /// # Safety
    /// NEON intrinsic; caller runs on aarch64.
    #[cfg(target_arch = "aarch64")]
    unsafe fn vqmov(vraw: core::arch::aarch64::int16x8_t) -> Self::Lane;

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-fn]
    //
    /// # Safety
    /// Stores 8 contiguous elements at `out`; caller ensures capacity.
    #[cfg(target_arch = "aarch64")]
    unsafe fn vst1(out: *mut Self, vout: Self::Lane);
}

impl Quantizable for u8 {
    const QMIN: i64 = u8::MIN as i64;
    const QMAX: i64 = u8::MAX as i64;
    fn from_i64(v: i64) -> Self {
        v as u8
    }

    #[cfg(target_arch = "aarch64")]
    type Lane = core::arch::aarch64::uint8x8_t;

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn]
    #[cfg(target_arch = "aarch64")]
    unsafe fn vqmov(vraw: core::arch::aarch64::int16x8_t) -> core::arch::aarch64::uint8x8_t {
        unsafe { core::arch::aarch64::vqmovun_s16(vraw) }
    }

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn]
    #[cfg(target_arch = "aarch64")]
    unsafe fn vst1(out: *mut u8, vout: core::arch::aarch64::uint8x8_t) {
        unsafe { core::arch::aarch64::vst1_u8(out, vout) }
    }
}

impl Quantizable for i8 {
    const QMIN: i64 = i8::MIN as i64;
    const QMAX: i64 = i8::MAX as i64;
    fn from_i64(v: i64) -> Self {
        v as i8
    }

    #[cfg(target_arch = "aarch64")]
    type Lane = core::arch::aarch64::int8x8_t;

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn]
    #[cfg(target_arch = "aarch64")]
    unsafe fn vqmov(vraw: core::arch::aarch64::int16x8_t) -> core::arch::aarch64::int8x8_t {
        unsafe { core::arch::aarch64::vqmovn_s16(vraw) }
    }

    // [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn]
    #[cfg(target_arch = "aarch64")]
    unsafe fn vst1(out: *mut i8, vout: core::arch::aarch64::int8x8_t) {
        unsafe { core::arch::aarch64::vst1_s8(out, vout) }
    }
}

pub fn quantize_val<T: Quantizable>(scale: f64, zero_point: i64, value: f32) -> T {
    // std::nearbyint results in nearest integer value according to the current
    // rounding mode and the default rounding mode is rounds to even in half-way
    // cases in most popular processor architectures like x86 and ARM.
    let mut qvalue: i64;
    let qmin: i64 = T::QMIN;
    let qmax: i64 = T::QMAX;
    let inv_scale: f32 = 1.0f32 / (scale as f32);
    qvalue = (zero_point as f64 + Round(value * inv_scale) as f64) as i64;
    qvalue = i64::max(qvalue, qmin);
    qvalue = i64::min(qvalue, qmax);
    T::from_i64(qvalue)
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]
//
// PORT-NOTE: Template `quantize_tensor_arm64_q8<underlying_t, underlying_x8_t>`.
// The two type params collapse into the single `Quantizable`-bound scalar `T`;
// `T::Lane` supplies `underlying_x8_t`.
//
// # Safety
// NEON path; `in`/`out` must be valid for `N` elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn quantize_tensor_arm64_q8<T: Quantizable>(
    mut in_: *const f32,
    out: *mut T,
    n: i64,
    scale: f32,
    zero_point: i32,
) {
    use core::arch::aarch64::*;
    let inv_scale: f32 = 1.0f32 / scale;
    let mut i: u32 = 0;
    let mut out_underlying: *mut T = out;
    let vinv_scale: float32x4_t = unsafe { vdupq_n_f32(inv_scale) };

    let vzero_point: int16x8_t = unsafe { vdupq_n_s16((zero_point as u16) as i16) };
    while (i as i64) + 8 <= n {
        let vin0123: float32x4_t = unsafe { vld1q_f32(in_) };
        in_ = unsafe { in_.add(4) };
        let vin4567: float32x4_t = unsafe { vld1q_f32(in_) };
        in_ = unsafe { in_.add(4) };
        let v0123_rounded: int32x4_t = unsafe { vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale)) };
        let v4567_rounded: int32x4_t = unsafe { vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale)) };
        let v01234567_packed: int16x8_t = unsafe {
            vqaddq_s16(
                vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
                vzero_point,
            )
        };
        let vout01234567: T::Lane = unsafe { T::vqmov(v01234567_packed) };
        unsafe { T::vst1(out_underlying, vout01234567) };
        out_underlying = unsafe { out_underlying.add(8) };
        i += 8;
    }
    while (i as i64) < n {
        let v = quantize_val::<T>(scale as f64, zero_point as i64, unsafe { *in_ });
        unsafe {
            *out_underlying = v;
            out_underlying = out_underlying.add(1);
            in_ = in_.add(1);
        }
        i += 1;
    }
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn]
// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn]
// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn]
//
// PORT-NOTE: The C++ wrapper has a bodyless primary template plus two explicit
// specializations, each forwarding to `quantize_tensor_arm64_q8<T,
// corresponding-x8-type>`. Because `Quantizable::Lane` already binds `T` to its
// lane vector, this single generic body IS the wrapper: it is only instantiable
// for the two `Quantizable` impls (i8/u8), a compile-time restriction
// equivalent to "only the explicit specializations are defined".
//
// # Safety
// NEON path; `in`/`out` valid for `N` elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn quantize_tensor_arm64_q8_wrapper<T: Quantizable>(
    in_: *const f32,
    out: *mut T,
    n: i64,
    scale: f32,
    zero_point: i32,
) {
    unsafe { quantize_tensor_arm64_q8::<T>(in_, out, n, scale, zero_point) };
}

// [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn]
// [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn]
//
// PORT-NOTE: Template `QuantizePerTensor<T = uint8_t>`. The C++ default type
// argument has no Rust equivalent; callers pass `T` explicitly. The
// `std::is_same` compile-time check that `T` is `uint8_t`/`int8_t` is surfaced
// at runtime in the C++ via `ET_CHECK_OR_RETURN_ERROR`; the `T: Quantizable`
// bound already restricts `T` to those two types, but the runtime check is kept
// bug-for-bug. On aarch64 the NEON wrapper is used, else the scalar loop.
#[must_use]
pub fn QuantizePerTensor<T: Quantizable + 'static>(
    rtensor: &Tensor,
    qtensor: &Tensor,
    scale: f64,
    zero_point: i32,
) -> Error {
    let rdata: *const f32 = rtensor.const_data_ptr::<f32>();
    let numel: i32 = rtensor.numel() as i32;
    crate::et_check_or_return_error!(
        core::any::TypeId::of::<T>() == core::any::TypeId::of::<u8>()
            || core::any::TypeId::of::<T>() == core::any::TypeId::of::<i8>(),
        Internal,
        "Expecting quantized output tensor of dtype uint8_t or int8_t"
    );
    crate::et_check_or_return_error!(
        rtensor.numel() <= qtensor.numel(),
        Internal,
        "Expecting quantized output tensor of same or smaller size as input, {} vs. {}",
        qtensor.numel(),
        rtensor.numel()
    );
    let qdata: *mut T = qtensor.mutable_data_ptr::<T>();

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            quantize_tensor_arm64_q8_wrapper::<T>(
                rdata,
                qdata,
                numel as i64,
                scale as f32,
                zero_point,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..numel {
            unsafe {
                *qdata.offset(i as isize) =
                    quantize_val::<T>(scale, zero_point as i64, *rdata.offset(i as isize));
            }
        }
    }
    Error::Ok
}

// Literal port of backends/xnnpack/test/runtime/test_runtime_utils.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::portable_type::qint_types::{qint8, quint8};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn/test]
    #[test]
    fn test_utils_choose_quantization_params() {
        let mut qparams = QuantizationParams {
            scale: 0.0,
            zero_point: 0,
        };
        let min: f32 = -128.0 * 10.0;
        let max: f32 = 127.0 * 10.0;
        let e = ChooseQuantizationParams(min, max, 0, 255, &mut qparams, false, false, false);
        assert_eq!(e, Error::Ok);
        assert_eq!(qparams.zero_point, 128);
        assert_eq!(qparams.scale, 10.0);
    }

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn/test]
    #[test]
    fn test_utils_choose_quantization_params_fails() {
        crate::runtime::platform::runtime::runtime_init();
        let mut qparams = QuantizationParams {
            scale: 0.0,
            zero_point: 0,
        };
        let min: f32 = -128.0 * 10.0;
        let max: f32 = 127.0 * 10.0;
        // min and max swapped: min > max should fail.
        let e = ChooseQuantizationParams(max, min, 0, 255, &mut qparams, false, false, false);
        assert_eq!(e, Error::Internal);
    }

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn/test]
    // On aarch64 (the CI/dev host) QuantizePerTensor routes through the NEON path,
    // so this numel=15, T=u8 case genuinely exercises the vectorized loop (first 8
    // elements) plus the scalar tail (remaining 7): the wrapper, the core arm64_q8
    // kernel, and the u8 narrow/store lane ops. A bug in any would change the 15
    // output bytes (all 135) and fail the tensor-equality assertion below.
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-fn/test]
    // T=u8, so on aarch64 the monomorphizations exercised are the uint8x8_t
    // specializations: the u8 wrapper forwards to arm64_q8, whose narrow uses
    // `vqmovun_s16` (u8 `vqmov`) and store uses `vst1_u8` (u8 `vst1`). Each of
    // the 8 vectorized bytes is produced by these exact ops, so a bug in any of
    // them changes the output (135) and fails the tensor-equality assertion.
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn/test]
    #[test]
    fn test_utils_quantize_per_tensor() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.full(vec![3, 5], 4.0, TensorShapeDynamism::STATIC);
        let scale: f64 = 0.5;
        let zero_point: i32 = 127;
        let tfo = TensorFactory::<quint8>::new();
        let output = tfo.zeros(vec![3, 5], TensorShapeDynamism::STATIC);
        // 4 / 0.5 + 127 = 8 + 127 = 135.
        // PORT-NOTE: The C++ builds `expected` via `at::quantize_per_tensor`
        // (ATen). ATen is not ported; the expected quantized value is computed
        // here directly (round(4/0.5) + 127 = 135), which is exactly what
        // `at::quantize_per_tensor` yields for these inputs.
        let expected = tfo.make(
            vec![3, 5],
            vec![quint8::new(135); 15],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let e = QuantizePerTensor::<u8>(&input, &output, scale, zero_point);
        assert_eq!(e, Error::Ok);
        assert_tensor_eq!(output, expected);
    }

    // T=i8 companion of `test_utils_quantize_per_tensor`. The C++ test suite
    // (test_runtime_utils.cpp) only covers QUInt8, so the int8x8_t
    // specializations were never exercised by a ported test. This numel=15,
    // T=i8 case routes `QuantizePerTensor::<i8>` through (on aarch64) the i8
    // wrapper -> arm64_q8 core, whose narrow uses `vqmovn_s16` (i8 `vqmov`) and
    // store uses `vst1_s8` (i8 `vst1`): the first 8 bytes are produced by the
    // vectorized lane ops and the remaining 7 by the scalar tail. A bug in the
    // i8 narrow/store/wrapper changes the 15 output bytes and fails the
    // assertion. round(4/0.5) + (-1) = 8 - 1 = 7, in-range for i8 [-128, 127].
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn/test]
    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn/test]
    #[test]
    fn test_utils_quantize_per_tensor_int8() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.full(vec![3, 5], 4.0, TensorShapeDynamism::STATIC);
        let scale: f64 = 0.5;
        let zero_point: i32 = -1;
        let tfo = TensorFactory::<qint8>::new();
        let output = tfo.zeros(vec![3, 5], TensorShapeDynamism::STATIC);
        // round(4 / 0.5) + (-1) = 8 - 1 = 7.
        let expected = tfo.make(
            vec![3, 5],
            vec![qint8::new(7); 15],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let e = QuantizePerTensor::<i8>(&input, &output, scale, zero_point);
        assert_eq!(e, Error::Ok);
        assert_tensor_eq!(output, expected);
    }

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.generate-requantization-scale-fn/test]
    #[test]
    fn test_utils_generate_requantizeation_scale() {
        let tf = TensorFactory::<f32>::new();
        let weight_scales = tf.full(vec![3, 5], 4.0, TensorShapeDynamism::STATIC);
        let input_scale: f32 = 2.0;
        let output_scale: f32 = 3.0;
        let mut req_scales: Vec<f32> = vec![0.0; 15];
        let e =
            GenerateRequantizationScale(&weight_scales, input_scale, output_scale, &mut req_scales);
        assert_eq!(e, Error::Ok);
        for m in req_scales.iter() {
            assert_float_eq(*m, 4.0 * 2.0 / 3.0);
        }
    }

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.round-fn/test]
    //
    // `Round` is `std::nearbyint` (round-half-to-even under the default FP mode).
    // No existing suite pins the tie-breaking behavior, so exercise it directly:
    // exact integers pass through, halfway cases round to the even neighbor, and
    // sign is preserved.
    #[test]
    fn test_utils_round_ties_to_even() {
        assert_eq!(Round(0.0), 0.0);
        assert_eq!(Round(1.0), 1.0);
        assert_eq!(Round(1.4), 1.0);
        assert_eq!(Round(1.6), 2.0);
        // Halfway cases: ties to even.
        assert_eq!(Round(0.5), 0.0);
        assert_eq!(Round(1.5), 2.0);
        assert_eq!(Round(2.5), 2.0);
        assert_eq!(Round(3.5), 4.0);
        assert_eq!(Round(-0.5), 0.0);
        assert_eq!(Round(-1.5), -2.0);
        assert_eq!(Round(-2.5), -2.0);
    }

    // [spec:et:sem:utils.executorch.backends.xnnpack.utils.get-min-max-fn/test]
    #[test]
    fn test_utils_get_min_max() {
        let tf = TensorFactory::<f32>::new();

        let val: f32 = 4.12345;
        let ft = tf.full(vec![3, 5], val, TensorShapeDynamism::STATIC);
        let (min, max) = GetMinMax(&ft);
        assert_float_eq(min, val);
        assert_float_eq(max, val);

        // std::numeric_limits<float>::min() is the smallest positive normal
        // (f32::MIN_POSITIVE), and ::max() is f32::MAX.
        let ft_min = tf.make(
            vec![2, 1],
            vec![f32::MIN_POSITIVE, f32::MAX],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let (min, max) = GetMinMax(&ft_min);
        assert_float_eq(min, f32::MIN_POSITIVE);
        assert_float_eq(max, f32::MAX);

        // std::numeric_limits<float>::lowest() is the most negative value
        // (f32::MIN).
        let ft_lowest = tf.make(
            vec![2, 1],
            vec![f32::MIN, f32::MAX],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let (min, max) = GetMinMax(&ft_lowest);
        assert_float_eq(min, f32::MIN);
        assert_float_eq(max, f32::MAX);

        let ft_random = tf.make(
            vec![5, 1],
            vec![-2.2, -1.1, 0.0, 1.1, 2.2],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let (min, max) = GetMinMax(&ft_random);
        assert_float_eq(min, -2.2);
        assert_float_eq(max, 2.2);
    }

    // Mirrors gtest `EXPECT_FLOAT_EQ`, which passes when the values are within
    // 4 ULPs of each other. That tolerance is reproduced here.
    fn assert_float_eq(a: f32, b: f32) {
        if a == b {
            return;
        }
        let ulps = ulp_distance(a, b);
        assert!(
            ulps <= 4,
            "float values not within 4 ULPs: {} vs {} ({} ULPs)",
            a,
            b,
            ulps
        );
    }

    fn ulp_distance(a: f32, b: f32) -> u32 {
        let ai = a.to_bits() as i32;
        let bi = b.to_bits() as i32;
        // Map to a monotonic ordering (two's-complement of sign-magnitude).
        let ai = if ai < 0 {
            i32::MIN.wrapping_sub(ai)
        } else {
            ai
        };
        let bi = if bi < 0 {
            i32::MIN.wrapping_sub(bi)
        } else {
            bi
        };
        (ai.wrapping_sub(bi)).unsigned_abs()
    }
}
