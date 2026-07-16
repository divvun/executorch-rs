# kernels/quantized/cpu/op_quantize.cpp

> [spec:et:def:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]
> void check_quantize_per_tensor_args( const Tensor& input, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]
> Validates the quantization parameters against the input and output tensors. Every check below is an `ET_CHECK_MSG` fatal assertion: on failure it aborts the program (there is no non-fatal / context error return path; the function returns `void` and produces no value).
>
> Steps, in order:
> 1. Assert `input.scalar_type()` is a floating-point type (isFloatingType returns true for Half, Float, Double, and BFloat16). If not, abort.
> 2. Initialize two `int32_t` locals `quant_min_lower_bound = 0` and `quant_max_upper_bound = 0`. Read `out_dtype = out.scalar_type()`.
> 3. Assert `out_dtype == dtype` (the requested `dtype` argument must equal the output tensor's actual dtype). If not, abort.
> 4. Select the representable range bounds for the output dtype:
>    - `Byte` (uint8): lower = 0, upper = 255.
>    - `Char` (int8): lower = -128, upper = 127.
>    - `Bits16` or `UInt16`: lower = 0, upper = 65535 (uint16 range).
>    - `Short` (int16): lower = -32768, upper = 32767.
>    - `Int` (int32): lower = INT32_MIN (-2147483648), upper = INT32_MAX (2147483647).
>    - Any other dtype: abort with "Unsupported dtype".
>    Note the branch condition subtlety: the first branch tests `out_dtype == Byte` while all remaining branches test the `dtype` argument; since step 3 already asserted `out_dtype == dtype` these are equivalent.
> 5. Assert `quant_min >= quant_min_lower_bound`. If not, abort.
> 6. Assert `quant_max <= quant_max_upper_bound`. If not, abort.
>
> The function does not check that `quant_min <= quant_max`, nor does it touch tensor shapes despite the misleading comment "Ensure self and out has the same shape".

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t]
> struct NeonQuantizeTraits<int8_t>

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.clamp-scalar-fn]
> static inline int8_t clamp_scalar(int32_t val)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.clamp-scalar-fn]
> Clamps an `int32_t` value to the int8 representable range and narrows it. Returns `static_cast<int8_t>(std::min(127, std::max(-128, val)))`: first raise `val` up to at least -128, then lower it down to at most 127, then truncate to `int8_t`. This is a fixed-range clamp to the full int8 domain and is independent of any user-supplied quant_min/quant_max.

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.narrow-and-saturate-fn]
> static inline int8x8_t narrow_and_saturate(int16x8_t v)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.narrow-and-saturate-fn]
> ARM NEON helper. Narrows a vector of 8 signed 16-bit lanes (`int16x8_t`) into 8 signed 8-bit lanes (`int8x8_t`) with signed saturation, via the intrinsic `vqmovn_s16(v)`: each lane is clamped to [-128, 127] before being truncated to int8. Pure SIMD register operation, no memory access. In a portable Rust port this is: for each of the 8 lanes, saturating-cast the i16 to i8.

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.store-fn]
> static inline void store(int8_t* ptr, int8x8_t v)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.store-fn]
> ARM NEON helper. Writes the 8 signed 8-bit lanes of `v` (`int8x8_t`) contiguously to memory at `ptr` via the intrinsic `vst1_s8(ptr, v)`. Equivalent to storing 8 consecutive `int8_t` values `ptr[0..8]` in lane order. No return value.

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t]
> struct NeonQuantizeTraits<uint8_t>

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.clamp-scalar-fn]
> static inline uint8_t clamp_scalar(int32_t val)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.clamp-scalar-fn]
> Clamps an `int32_t` value to the uint8 representable range and narrows it. Returns `static_cast<uint8_t>(std::min(255, std::max(0, val)))`: first raise `val` up to at least 0, then lower it down to at most 255, then truncate to `uint8_t`. Fixed-range clamp to the full uint8 domain, independent of any user-supplied quant_min/quant_max.

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.narrow-and-saturate-fn]
> static inline uint8x8_t narrow_and_saturate(int16x8_t v)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.narrow-and-saturate-fn]
> ARM NEON helper. Narrows a vector of 8 signed 16-bit lanes (`int16x8_t`) into 8 unsigned 8-bit lanes (`uint8x8_t`) with unsigned saturation, via the intrinsic `vqmovun_s16(v)`: each signed lane is clamped to [0, 255] (negative values saturate to 0, values above 255 saturate to 255) before being truncated to uint8. Pure SIMD register operation, no memory access.

> [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.store-fn]
> static inline void store(uint8_t* ptr, uint8x8_t v)

> [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.store-fn]
> ARM NEON helper. Writes the 8 unsigned 8-bit lanes of `v` (`uint8x8_t`) contiguously to memory at `ptr` via the intrinsic `vst1_u8(ptr, v)`. Equivalent to storing 8 consecutive `uint8_t` values `ptr[0..8]` in lane order. No return value.

> [spec:et:def:op-quantize.torch.executor.native.quantize-arm-fn]
> void quantize_arm( const float* __restrict__ in, T* __restrict__ out, const int64_t N, const float inv_scale, const int32_t zero_point, const int32_t quant_min, const int32_t quant_max)

> [spec:et:sem:op-quantize.torch.executor.native.quantize-arm-fn]
> ARM NEON optimized quantization of a contiguous block of `N` float inputs (`in`) into `N` outputs of type `T` (int8 or uint8) at `out`, all sharing one quantization parameter set. `T` selects the `NeonQuantizeTraits<T>` used for narrowing/storing/clamping. This function is only compiled when NEON is available and is only reached for `input.scalar_type() == Float` with output dtype Byte or Char; other cases fall through to the scalar path. The result MUST be bit-identical to the scalar `quantize_val` path for the same inputs.
>
> Parameters: `inv_scale` is the precomputed reciprocal `1.0f / (float)scale`; `zero_point`, `quant_min`, `quant_max` are `int32_t`.
>
> There are two compile-time variants:
>
> ARMv8 (`__aarch64__`) path:
> 1. Broadcast `inv_scale` into a 4-lane float vector; broadcast `zero_point`, `quant_min`, `quant_max` each into 8-lane int16 vectors (narrowing cast to int16 first).
> 2. Main loop `i = 0`, stepping by 8 while `i + 8 <= N`: load `in[i..i+4)` and `in[i+4..i+8)` as two float4 vectors; multiply each by `inv_scale`; round-to-nearest-even to int32 (`vcvtnq_s32_f32`, IEEE round-half-to-even); pack the two int32x4 down to one int16x8 with signed saturation; add `zero_point` with signed-saturating add (`vqaddq_s16`); clamp lanes to `quant_min` (elementwise max) then to `quant_max` (elementwise min); narrow to `T` via `Traits::narrow_and_saturate` (signed saturate for int8, unsigned saturate for uint8); store 8 lanes via `Traits::store`.
> 3. Tail loop for remaining `i < N`: `val = in[i] * inv_scale`; `qval = (int32)std::nearbyint(val) + zero_point`; `qval = max(quant_min, min(quant_max, qval))`; `out[i] = (T)qval`.
>
> ARMv7 (non-aarch64 NEON) path: same block/tail structure but the SIMD body uses the "magic float" rounding trick: `voffset = zero_point - 0x4B400000`; add magic constant `12582912.0f` to `in*inv_scale`, reinterpret the float bits as int32, add `voffset`, narrow the two int32x4 to int16x8 with signed saturation, then narrow to `T` via `Traits::narrow_and_saturate` and store. Note this SIMD body does NOT apply quant_min/quant_max clamping (only the fixed int8/uint8 saturation of narrow_and_saturate). The scalar tail loop (identical to the ARMv8 tail) DOES apply quant_min/quant_max clamping.
>
> Rounding is round-half-to-even in both variants (matching `std::nearbyint` under the default FE_TONEAREST rounding mode). No return value; writes are in-place into `out`.

> [spec:et:def:op-quantize.torch.executor.native.quantize-val-fn]
> T quantize_val( double scale, int64_t zero_point, K value, int64_t quant_min, int64_t quant_max)

> [spec:et:sem:op-quantize.torch.executor.native.quantize-val-fn]
> Scalar quantization of a single value. Template params: `T` is the output integer type, `K` is the input value type. Given `scale` (double), `zero_point` (int64), `value` (K), `quant_min`, `quant_max` (int64), computes the quantized integer.
>
> Steps:
> 1. `inv_scale = 1.0f / (float)scale` — computed in single precision float (the double `scale` is narrowed to float before the reciprocal).
> 2. Compute `qvalue` as int64: `(int32_t)zero_point + std::nearbyint((float)(inv_scale * value))`. The product `inv_scale * value` and the `nearbyint` are done in `float` (round-half-to-even under default rounding mode); `zero_point` is first narrowed to int32 then the sum is widened to int64.
> 3. `qvalue = std::max<int64_t>(qvalue, quant_min)` then `qvalue = std::min<int64_t>(qvalue, quant_max)` — clamp to the requested range.
> 4. Return `static_cast<T>(qvalue)` (truncating cast into the output integer type).
>
> Note the intermediate arithmetic is float, so precision matches a 32-bit float pipeline (not double), which is what makes the scalar path match the NEON path.

> [spec:et:def:op-quantize.torch.executor.native.quantize-per-channel-out-fn]
> Tensor& quantize_per_channel_out( const Tensor& input, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn]
> Per-channel (per-axis) quantization: each slice along `axis` uses its own scale/zero_point. This is the core implementation without a context arg; it does NOT resize `out` itself (the caller/context wrapper `[spec:et:sem:op-quantize.torch.executor.quantize-per-channel-out-fn]` resizes). All validation uses fatal `ET_CHECK_MSG`.
>
> Steps:
> 1. Assert `tensor_has_dim(input, axis)` (axis must satisfy `-input.dim() <= axis < input.dim()`). Abort otherwise.
> 2. Normalize a negative axis: if `axis < 0`, `axis += nonzero_dim(input)` (nonzero_dim returns `dim()`, or 1 when dim()==0).
> 3. Assert `scale.scalar_type() == Double`. Abort otherwise.
> 4. Assert `scale.numel() == input.size(axis)`. Abort otherwise.
> 5. Assert `zero_point.scalar_type() == Long` (int64). Abort otherwise.
> 6. Assert `zero_point.numel() == input.size(axis)`. Abort otherwise.
> 7. Call `check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out)` per `[spec:et:sem:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]` (validates input is float, out dtype == dtype, and quant_min/quant_max within dtype bounds).
> 8. Get raw pointers: `scale_data` (double*), `zero_point_data` (int64*).
> 9. Compute `axis_block_size` = product of `input.size(i)` for all `i` in `(axis, input.dim())` (the number of contiguous elements sharing one channel index; 1 if axis is the last dim). Compute `axis_size = input.size(axis)`.
>
> Channel-index mapping (used by all paths): for a flat element index `i` into the contiguous (dim-order-0) buffer, its channel is `channel_idx = (i / axis_block_size) % axis_size`. Blocks of `axis_block_size` consecutive elements share one channel; the channel index cycles 0..axis_size-1.
>
> NEON fast path (compiled only with NEON, taken only when `input.scalar_type() == Float` and dtype is Byte or Char):
> - `num_blocks = input.numel() / axis_block_size`; `total_elements = input.numel()`; `use_parallel = total_elements >= 512`.
> - For each block index `block` in `[0, num_blocks)` (optionally parallelized via `parallel_for` with grain size 1 — the split is over blocks and does not change results): `channel_idx = block % axis_size`; `inv_scale = 1.0f / (float)scale_data[channel_idx]`; `zp = (int32)zero_point_data[channel_idx]`; then quantize the `axis_block_size`-long sub-block starting at `block * axis_block_size` via `quantize_arm<uint8_t|int8_t>` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-arm-fn]`. Return `out`.
>   Note: `block % axis_size` equals `((block*axis_block_size)/axis_block_size) % axis_size`, consistent with the flat-index mapping.
>
> Scalar fallback (all other input/output dtype combinations): single loop `i` over `[0, input.numel())`; `channel_idx = (i / axis_block_size) % axis_size`; `_scale = scale_data[channel_idx]`; `_zero_point = zero_point_data[channel_idx]`; `out_data_ptr[i] = quantize_val<CTYPE_OUT, CTYPE_IN>(_scale, _zero_point, input_data_ptr[i], quant_min, quant_max)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-val-fn]`.
> - Accepted input dtypes (ET_FORALL_FLOATH_TYPES): Float, Double, Half. Any other input dtype aborts ("Unhandled input dtype").
> - Accepted output dtypes: the int set {Byte/uint8, Char/int8, Short/int16, Int/int32, Long/int64} plus Bits16 and UInt16 (both handled as uint16). Any other output dtype aborts ("Unhandled output dtype").
>
> Returns `out` (written in place).

> [spec:et:def:op-quantize.torch.executor.native.quantize-per-tensor-out-fn]
> Tensor& quantize_per_tensor_out( const Tensor& input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn]
> Per-tensor quantization: a single scalar `scale`/`zero_point` applied to every element. This is the no-context overload.
>
> Steps:
> 1. `resize_tensor(out, input.sizes())` to make `out` match `input`'s shape; assert the result is `Error::Ok` (fatal `ET_CHECK_MSG` otherwise).
> 2. `check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out)` per `[spec:et:sem:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]`.
> 3. NEON fast path (compiled only with NEON): if `input.scalar_type() == Float` and dtype is Byte or Char, call `quantize_arm<uint8_t|int8_t>(input.const_data_ptr<float>(), out.mutable_data_ptr<...>(), input.numel(), 1.0f/(float)scale, (int32)zero_point, (int32)quant_min, (int32)quant_max)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-arm-fn]`, then return `out`.
> 4. Scalar fallback (all other cases): dispatch on `input.scalar_type()` over ET_FORALL_FLOATH_TYPES = {Float, Double, Half}; for each, dispatch on `out.scalar_type()` over {Byte/uint8, Char/int8, Short/int16, Int/int32, Long/int64} plus Bits16 and UInt16 (both uint16). For each element `i` in `[0, input.numel())`: `out_data_ptr[i] = quantize_val<OUT_CTYPE, IN_CTYPE>(scale, zero_point, input_data_ptr[i], quant_min, quant_max)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-val-fn]`. Unhandled input or output dtype aborts.
> 5. Return `out`.
>
> There is also a context-taking overload `quantize_per_tensor_out(context, ...)` that ignores `context` and forwards to this function unchanged.

> [spec:et:def:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn]
> Tensor& quantize_per_tensor_tensor_args_out( KernelRuntimeContext& context, const Tensor& input, const Tensor& scale, const Tensor& zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn]
> Variant of per-tensor quantize where `scale` and `zero_point` are passed as single-element tensors rather than scalars. This is the context-taking overload.
>
> Steps:
> 1. If `scale.scalar_type() != Double`: call `context.fail(Error::InvalidArgument)` (non-fatal) and return `out` unchanged. (This early non-fatal exit exists to unblock expected-failure tests.)
> 2. Assert `scale.scalar_type() == Double` (fatal `ET_CHECK_MSG`; redundant after step 1 but present).
> 3. Assert `zero_point.scalar_type() == Long` (int64). Abort otherwise.
> 4. Assert `scale.numel() == 1`. Abort otherwise.
> 5. Assert `zero_point.numel() == 1`. Abort otherwise.
> 6. Read `scale.const_data_ptr<double>()[0]` and `zero_point.const_data_ptr<int64_t>()[0]` and forward to `quantize_per_tensor_out(input, scale_val, zero_point_val, quant_min, quant_max, dtype, out)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn]`.
> 7. Return `out`.
>
> There is also a no-context overload that constructs a fresh `KernelRuntimeContext`, calls this function, then asserts `context.failure_state() == Error::Ok` (turning the non-fatal scale-dtype failure into a fatal abort) and returns the result.

> [spec:et:def:op-quantize.torch.executor.quantize-per-channel-out-fn]
> Tensor& quantize_per_channel_out( KernelRuntimeContext& context, const Tensor& input, const Tensor& scale, const Tensor& zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.quantize-per-channel-out-fn]
> Context-taking wrapper for per-channel quantization (registered kernel entry point).
>
> Steps:
> 1. Ignore `context` (`(void)context`).
> 2. `resize_tensor(out, input.sizes())` so `out` matches `input`'s shape; assert result is `Error::Ok` (fatal `ET_CHECK_MSG` otherwise).
> 3. Forward to the `native` per-channel implementation `quantize_per_channel_out(input, scale, zero_point, axis, quant_min, quant_max, dtype, out)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn]` and return its result (`out`).

> [spec:et:def:op-quantize.torch.executor.quantize-per-token-out-fn]
> Tensor& quantize_per_token_out( const Tensor& input, const Tensor& scale, const Tensor& zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, Tensor& out)

> [spec:et:sem:op-quantize.torch.executor.quantize-per-token-out-fn]
> Per-token quantization: treats the input as a 2D `[num_tokens, token_size]` matrix where each "token" (each row = the trailing dimension) has its own scale/zero_point, and delegates to per-channel quantize along axis 0. This is the no-context overload.
>
> Steps:
> 1. `num_tokens = product of input.size(i) for i in [0, input.dim()-1)` — the product of all dimensions except the last (1 if input is 1D).
> 2. Build a logical 2D view `reshaped_input` of the input data with shape `[num_tokens, input.size(input.dim()-1)]`:
>    - Non-ATen (portable) mode: construct a `TensorImpl` of rank 2 over the same data pointer with dim_order `{0, 1}`, sizes `{num_tokens, last_dim}`, strides computed by `dim_order_to_stride_nocheck` (contiguous, i.e. `{last_dim, 1}`), STATIC shape dynamism. Then `resize_tensor(out, input.sizes())` and assert `Error::Ok`.
>    - ATen mode (`USE_ATEN_LIB`): use `at::from_blob(input.mutable_data_ptr(), {num_tokens, last_dim}, input.scalar_type())`; no resize of `out` in this branch.
> 3. Forward to `quantize_per_channel_out(reshaped_input, scale, zero_point, /*axis=*/0, quant_min, quant_max, dtype, out)` per `[spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn]`. Axis 0 means each of the `num_tokens` rows uses `scale[row]`/`zero_point[row]`; consequently `scale` and `zero_point` must each have `num_tokens` elements. Return `out`.
>
> A context-taking overload also exists that ignores `context` and forwards to this function unchanged.

