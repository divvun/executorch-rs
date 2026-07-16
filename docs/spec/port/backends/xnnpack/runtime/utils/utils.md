# backends/xnnpack/runtime/utils/utils.cpp, backends/xnnpack/runtime/utils/utils.h

> [spec:et:def:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn]
> Error ChooseQuantizationParams( float min, float max, int32_t qmin, int32_t qmax, QuantizationParams& result, bool preserve_sparsity = false, bool force_scale_power_of_two = false, bool reduce_range = false)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.choose-quantization-params-fn]
> Computes affine quantization parameters (scale as `double`, zero_point as
> `int32_t`) for the real value range `[min, max]` targeting the integer range
> `[qmin, qmax]`, writing them into `result`. Mirrors the fbgemm/PyTorch
> `ChooseQuantizationParams` logic. Signature defaults (from the header):
> `preserve_sparsity=false`, `force_scale_power_of_two=false`,
> `reduce_range=false`.
>
> Steps:
> 1. Require `min <= max`; if violated return `Error::Internal` (message
>    includes both values). No output is written on this failure.
> 2. If `reduce_range`, halve both bounds with integer division:
>    `qmin = qmin / 2`, `qmax = qmax / 2`.
> 3. If `min < 0 && max > 0 && preserve_sparsity` (symmetric case): let
>    `symmetric_qmin = -((qmax - qmin) / 2 + 1)` and
>    `symmetric_qmax = (qmax - qmin) / 2` (integer division). Compute
>    `max_scale = max(|min / symmetric_qmin|, |max / symmetric_qmax|)` as a
>    double, then set `min = max_scale * symmetric_qmin` and
>    `max = max_scale * symmetric_qmax`.
> 4. Extend the interval to include 0: `min = min(min, 0.0f)`,
>    `max = max(max, 0.0f)`.
> 5. Require `qmin < qmax`; if violated return `Error::Internal`.
> 6. Compute `scale = (static_cast<double>(max) - min) / (qmax - qmin)` in
>    double precision.
> 7. If `float(scale) == 0.0f` or `1.0f / float(scale)` is infinite, set
>    `scale = 0.1` (avoids an infinite reciprocal).
> 8. Require `scale > 0`; if violated return `Error::Internal`.
> 9. If `force_scale_power_of_two`, snap scale to a power of two: if
>    `scale < 1`, set `scale = 1.0 / (1 << floor(log(1.0/scale)/log(2)))`; else
>    `scale = 1 << ceil(log(scale)/log(2))` (natural-log-based log2, cast to int).
> 10. Cut off small scale: if `scale < SMALL_SCALE_THRESHOLD` (constant
>     `6.1e-5f`), remember `org_scale = scale`, set `scale =
>     SMALL_SCALE_THRESHOLD`, and rescale the bounds to keep them consistent:
>     - if `min == 0.0f`: `max = SMALL_SCALE_THRESHOLD * (qmax - qmin)`;
>     - else if `max == 0.0f`: `min = -SMALL_SCALE_THRESHOLD * (qmax - qmin)`;
>     - else: `amplifier = SMALL_SCALE_THRESHOLD / org_scale`, `min *= amplifier`,
>       `max *= amplifier`.
> 11. Compute the initial (floating) zero point choosing the numerically better
>     of the two affine solutions:
>     - `zero_point_from_min = qmin - min / scale`;
>     - `zero_point_from_max = qmax - max / scale`;
>     - `zero_point_from_min_error = |qmin| - |min / scale|`;
>     - `zero_point_from_max_error = |qmax| - |max / scale|`;
>     - `initial_zero_point = (zero_point_from_min_error <
>       zero_point_from_max_error) ? zero_point_from_min : zero_point_from_max`.
> 12. If `min < 0 && max > 0 && preserve_sparsity`, override
>     `initial_zero_point = (qmin + qmax) / 2.0` (forced midpoint).
> 13. Nudge to an integer within range: if `initial_zero_point < qmin`,
>     `nudged_zero_point = qmin`; else if `initial_zero_point > qmax`,
>     `nudged_zero_point = qmax`; else `nudged_zero_point =
>     nearbyint(initial_zero_point)` (round-to-nearest, ties to even).
> 14. Store `result.scale = scale` and `result.zero_point = nudged_zero_point`
>     and return `Error::Ok`.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.generate-requantization-scale-fn]
> Error GenerateRequantizationScale( const Tensor& weight_scales, float input_scale, float output_scale, std::vector<float>& requant_scales)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.generate-requantization-scale-fn]
> Computes per-output-channel requantization scales for a quantized conv/linear
> op: `requant_scales[i] = weight_scales[i] * input_scale / output_scale`.
>
> Steps:
> 1. Let `num_output_channels_padded = weight_scales.numel()` (the tensor is
>    allocated with padding, so this is the padded channel count).
> 2. Obtain the weight-scale data via `weight_scales.const_data_ptr<float>()`
>    (input is required to be a Float tensor by the caller).
> 3. If `requant_scales.size()` (as `int64_t`) is less than
>    `num_output_channels_padded`, resize `requant_scales` to that length.
> 4. For each channel `i` from 0 to `num_output_channels_padded - 1`:
>    a. Compute `inverse_output_scale = 1.f / output_scale`.
>    b. Set `requant_scales[i] = (weight_scales_data[i] * input_scale) *
>       inverse_output_scale`.
>    c. Require `requant_scales[i] > 0.0f && std::isnormal(requant_scales[i])`
>       (finite, nonzero, non-subnormal); if this fails return `Error::Internal`
>       with message "failed to create op with requantization scale". Elements
>       written before the failing index are left in `requant_scales`.
> 5. Return `Error::Ok`.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.get-min-max-fn]
> std::pair<float, float> GetMinMax(const Tensor& ft)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.get-min-max-fn]
> Returns the (min, max) pair of the elementwise values of a Float tensor.
>
> Steps:
> 1. Initialize `min = std::numeric_limits<float>::max()` and
>    `max = -std::numeric_limits<float>::max()`.
> 2. Assert (ET_CHECK_MSG; hard failure/abort on violation) that
>    `ft.scalar_type() == ScalarType::Float`; the message prints the offending
>    scalar type as an int8.
> 3. Obtain the data pointer via `ft.const_data_ptr<float>()`.
> 4. For each element index `i` from 0 to `ft.numel() - 1`: update `min` to
>    `d[i]` if `d[i] < min`, and `max` to `d[i]` if `d[i] > max`.
> 5. Return `std::pair<float,float>(min, max)`.
>
> Edge case: for an empty tensor (`numel() == 0`) the loop does nothing and the
> returned pair is `(FLT_MAX, -FLT_MAX)`. NaN elements never satisfy the strict
> `<`/`>` comparisons and are effectively ignored.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantization-params]
> struct QuantizationParams {
>   double scale;
>   int32_t zero_point;
> }

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn]
> executorch::runtime::Error QuantizePerTensor( const executorch::aten::Tensor& rtensor, executorch::aten::Tensor& qtensor, double scale, int zero_point)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-per-tensor-fn]
> Template `QuantizePerTensor<T = uint8_t>`. Quantizes a Float input tensor
> `rtensor` into the integer output tensor `qtensor` using per-tensor affine
> parameters `scale` (double) and `zero_point` (int). `T` must be `uint8_t` or
> `int8_t`.
>
> Steps:
> 1. Get `rdata = rtensor.const_data_ptr<float>()` and `numel = rtensor.numel()`.
> 2. Require `T` is `uint8_t` or `int8_t` (compile-time `std::is_same` check
>    surfaced at runtime); if not, return `Error::Internal` ("Expecting
>    quantized output tensor of dtype uint8_t or int8_t").
> 3. Require `rtensor.numel() <= qtensor.numel()`; if violated return
>    `Error::Internal` (message prints both counts). The output may be larger
>    than the input; only the first `numel` elements are written.
> 4. Get `qdata = qtensor.mutable_data_ptr<T>()`.
> 5. Fill the output:
>    - On `__aarch64__`: call `quantize_tensor_arm64_q8_wrapper<T>(rdata, qdata,
>      numel, static_cast<float>(scale), zero_point)` (see
>      `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn]`),
>      the NEON-accelerated path.
>    - Otherwise (scalar path): for `i` in `0..numel`, set `qdata[i] =
>      quantize_val<T>(scale, zero_point, rdata[i])` (see
>      `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn]`).
> 6. Return `Error::Ok`.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]
> void quantize_tensor_arm64_q8( const float* ET_RESTRICT in, underlying_t* ET_RESTRICT out, const int64_t N, const float scale, const int32_t zero_point)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]
> Template `quantize_tensor_arm64_q8<underlying_t, underlying_x8_t>` (aarch64
> only). NEON-vectorized affine quantization of `N` float inputs into `N`
> 8-bit integers with round-to-nearest-even and saturation, equivalent
> elementwise to `quantize_val<underlying_t>(scale, zero_point, in[k])`.
>
> Steps:
> 1. Compute `inv_scale = 1.0f / scale`.
> 2. Broadcast `inv_scale` into a float32x4 vector `vinv_scale`, and broadcast
>    `zero_point` into an int16x8 vector `vzero_point` as
>    `(int16_t)(uint16_t)zero_point` (low 16 bits of the zero point).
> 3. Process 8 elements per iteration while `i + 8 <= N`:
>    a. Load two float32x4 vectors `vin0123`, `vin4567` from `in` (advancing `in`
>       by 8 floats total).
>    b. Multiply each by `vinv_scale` and convert to int32x4 with
>       round-to-nearest-even (`vcvtnq_s32_f32`), giving `v0123_rounded`,
>       `v4567_rounded`.
>    c. Narrow both int32x4 vectors to a single int16x8 with signed saturation
>       (`vqmovn_s32` + `vqmovn_high_s32`), then saturating-add `vzero_point`
>       (`vqaddq_s16`) → `v01234567_packed`.
>    d. Narrow int16x8 to the 8-bit lane type with saturation via
>       `vqmov<underlying_x8_t>` (see
>       `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-fn]`):
>       `vqmovun_s16` for uint8 (unsigned saturation), `vqmovn_s16` for int8.
>    e. Store the 8 results to the output via `vst1<underlying_t,
>       underlying_x8_t>` (see
>       `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-fn]`),
>       advancing the output pointer by 8.
> 4. Handle the remaining `N mod 8` tail elements scalar-wise: for each, write
>    `quantize_val<underlying_t>(scale, zero_point, *in++)` (see
>    `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn]`).
>
> Note the NEON path adds the zero point with int16 saturation before narrowing,
> whereas the scalar `quantize_val` adds `zero_point` at int64 width; results
> agree for the valid uint8/int8 output ranges targeted here.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn]
> void quantize_tensor_arm64_q8_wrapper( const float* ET_RESTRICT in, T* ET_RESTRICT out, const int64_t N, const float scale, const int32_t zero_point)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-fn]
> Template declaration `quantize_tensor_arm64_q8_wrapper<T>(in, out, N, scale,
> zero_point)` (aarch64 only). This is the primary template with no generic
> body; only the explicit `int8_t` and `uint8_t` specializations are defined
> (see
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn]`
> and
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn]`).
> Each specialization forwards to `quantize_tensor_arm64_q8<T, corresponding
> 8-lane NEON type>`. Instantiating it for any other `T` is a link error.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn]
> void quantize_tensor_arm64_q8_wrapper<int8_t>(

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-int8-t-fn]
> Explicit specialization `quantize_tensor_arm64_q8_wrapper<int8_t>` (aarch64
> only). Forwards directly to `quantize_tensor_arm64_q8<int8_t, int8x8_t>(in,
> out, N, scale, zero_point)` (see
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]`),
> producing signed 8-bit quantized output.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn]
> void quantize_tensor_arm64_q8_wrapper<uint8_t>(

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-wrapper-uint8-t-fn]
> Explicit specialization `quantize_tensor_arm64_q8_wrapper<uint8_t>` (aarch64
> only). Forwards directly to `quantize_tensor_arm64_q8<uint8_t, uint8x8_t>(in,
> out, N, scale, zero_point)` (see
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-tensor-arm64-q8-fn]`),
> producing unsigned 8-bit quantized output.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.quantize-val-fn]
> T quantize_val(double scale, int64_t zero_point, float value)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.quantize-val-fn]
> Template `quantize_val<T>(double scale, int64_t zero_point, float value)`.
> Scalar affine quantization of one float value to integer type `T`.
>
> Steps:
> 1. Let `qmin = std::numeric_limits<T>::min()` and `qmax =
>    std::numeric_limits<T>::max()` (as int64 constants).
> 2. Compute `inv_scale = 1.0f / static_cast<float>(scale)` (single precision).
> 3. Compute `qvalue = static_cast<int64_t>(zero_point + Round(value *
>    inv_scale))`, where `Round` is round-to-nearest, ties-to-even
>    (`std::nearbyint`, per
>    `[spec:et:sem:utils.executorch.backends.xnnpack.utils.round-fn]`). The
>    multiply/round happen in float; the add of `zero_point` is done at int64.
> 4. Clamp: `qvalue = max(qvalue, qmin)`, then `qvalue = min(qvalue, qmax)`.
> 5. Return `static_cast<T>(qvalue)`.
>
> Rounding follows the current FP rounding mode (default nearest-even),
> consistent with SIMD conversions.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.round-fn]
> inline float Round(const float x)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.round-fn]
> Rounds a floating-point value to the nearest integer using the current FP
> rounding mode (default round-half-to-even). Two build variants:
>
> - On old Android without `__NDK_MAJOR__`: a `Round<T>(const float x)` template
>   returning `::nearbyintf(x)` (float), plus a non-template overload
>   `Round(const double x)` returning `::nearbyint(x)` (double).
> - Otherwise: a generic template `Round<T>(const T x)` returning
>   `std::nearbyint(x)`, preserving `T` (float→float, double→double).
>
> In all cases the operation is `nearbyint`: nearest integer, ties to even, no
> `FE_INEXACT` raised.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-fn]
> Tx8 vqmov(int16x8_t vraw)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-fn]
> Template declaration `vqmov<Tx8>(int16x8_t vraw)` (aarch64 only). Primary
> template with no generic body; narrows an int16x8 vector to an 8-bit-lane
> NEON vector with saturation. Only the `int8x8_t` and `uint8x8_t`
> specializations are defined (see
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn]` and
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn]`).

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn]
> int8x8_t vqmov<int8x8_t>(int16x8_t vraw)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-int8x8-t-fn]
> Explicit specialization `vqmov<int8x8_t>(int16x8_t vraw)` (aarch64 only).
> Returns `vqmovn_s16(vraw)`: narrows each of the 8 signed 16-bit lanes to a
> signed 8-bit lane with signed saturation (clamped to [-128, 127]).

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn]
> uint8x8_t vqmov<uint8x8_t>(int16x8_t vraw)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vqmov-uint8x8-t-fn]
> Explicit specialization `vqmov<uint8x8_t>(int16x8_t vraw)` (aarch64 only).
> Returns `vqmovun_s16(vraw)`: narrows each of the 8 signed 16-bit lanes to an
> unsigned 8-bit lane with unsigned saturation (negative values clamp to 0,
> values above 255 clamp to 255).

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-fn]
> void vst1(T* out, Tx8 vout)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-fn]
> Template declaration `vst1<T, Tx8>(T* out, Tx8 vout)` (aarch64 only). Primary
> template with no generic body; stores an 8-lane NEON vector `vout` to the 8
> contiguous elements at `out`. Only the `<int8_t, int8x8_t>` and `<uint8_t,
> uint8x8_t>` specializations are defined (see
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn]`
> and
> `[spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn]`).

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn]
> void vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-int8-t-int8x8-t-fn]
> Explicit specialization `vst1<int8_t, int8x8_t>(int8_t* out, int8x8_t vout)`
> (aarch64 only). Calls `vst1_s8(out, vout)`, storing the 8 signed 8-bit lanes
> contiguously to `out`.

> [spec:et:def:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn]
> void vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t vout)

> [spec:et:sem:utils.executorch.backends.xnnpack.utils.vst1-uint8-t-uint8x8-t-fn]
> Explicit specialization `vst1<uint8_t, uint8x8_t>(uint8_t* out, uint8x8_t
> vout)` (aarch64 only). Calls `vst1_u8(out, vout)`, storing the 8 unsigned
> 8-bit lanes contiguously to `out`.

