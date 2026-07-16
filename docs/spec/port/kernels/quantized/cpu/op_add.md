# kernels/quantized/cpu/op_add.cpp

> [spec:et:def:op-add.torch.executor.native.add-tensors-fn]
> void add_tensors( const Tensor& a, float a_scale, int32_t a_zero_point, const Tensor& b, float b_scale, int32_t b_zero_point, Tensor& out, float out_scale, int32_t out_zero_point, int64_t out_quant_min, int64_t out_quant_max)

> [spec:et:sem:op-add.torch.executor.native.add-tensors-fn]
> Elementwise quantized add of two same-shape, same-dtype quantized
> tensors `a` and `b` into `out`, numerically equivalent to
> dequantize(a) + dequantize(b) then requantize. Generic over `CTYPE`,
> the (integer) storage type of `a`, `b`, and `out`. Steps:
> - `n = a.numel()` (all three tensors have identical shape, so the
>   same count; iteration is flat).
> - Get raw pointers `data_a`, `data_b` (const `CTYPE*`) and `data_out`
>   (mutable `CTYPE*`).
> - For each `i` in `[0, n)` in ascending order:
>   - `dqa = dequantize_val<CTYPE, float>(a_scale, a_zero_point,
>     data_a[i])` = `float((data_a[i] - a_zero_point) * a_scale)`
>     (`[spec:et:sem:op-add.torch.executor.native.dequantize-val-fn]`).
>     Note `a_scale`/`b_scale` are `float` here (already downcast by the
>     caller) though the param type is `double`.
>   - `dqb = dequantize_val<CTYPE, float>(b_scale, b_zero_point,
>     data_b[i])`.
>   - `accumulate = dqa + dqb` in float.
>   - `data_out[i] = quantize_val<float, CTYPE>(out_scale,
>     out_zero_point, accumulate, out_quant_min, out_quant_max)`
>     (`[spec:et:sem:op-add.torch.executor.native.quantize-val-fn]`).
> No bounds checks or validation here (the caller validated). Assumes
> the tensors are contiguous / flat-iterable in matching order.

> [spec:et:def:op-add.torch.executor.native.dequantize-val-fn]
> OUTPUT_T dequantize_val(double scale, int64_t zero_point, INPUT_T value)

> [spec:et:sem:op-add.torch.executor.native.dequantize-val-fn]
> Dequantizes a single value. Generic over `INPUT_T` (the quantized
> storage type) and `OUTPUT_T` (the real dtype, e.g. float). Computes
> and returns `(value - zero_point) * scale` cast to `OUTPUT_T`. The
> subtraction `value - zero_point`: `value` (INPUT_T) and `zero_point`
> (int64) — integer arithmetic promotes to int64, giving `value -
> zero_point`, which is then multiplied by `scale` (double). The double
> product is finally narrowed to `OUTPUT_T` (float at the call sites in
> this file). No clamping or rounding.

> [spec:et:def:op-add.torch.executor.native.quantize-val-fn]
> OUTPUT_T quantize_val( double scale, int64_t zero_point, INPUT_T value, int64_t quant_min, int64_t quant_max)

> [spec:et:sem:op-add.torch.executor.native.quantize-val-fn]
> Quantizes a single real value into an integer storage value. Generic
> over `INPUT_T` (real dtype, e.g. float) and `OUTPUT_T` (quantized
> storage type). Steps:
> - `inv_scale = 1.0f / static_cast<float>(scale)` — the reciprocal is
>   computed in FLOAT (the double `scale` is downcast to float first).
> - `qvalue = static_cast<int64_t>(zero_point + std::nearbyint(inv_scale
>   * value))`. The product `inv_scale * value` and the `nearbyint`
>   rounding are done in float; `std::nearbyint` rounds to nearest using
>   the current rounding mode (default round-half-to-even). Adding
>   `zero_point` (int64) promotes the rounded float to a value that is
>   then truncated to int64.
> - Clamp: `qvalue = max(qvalue, quant_min)` then `qvalue =
>   min(qvalue, quant_max)`, both in int64.
> - Return `static_cast<OUTPUT_T>(qvalue)`.
> Uses round-half-to-even (via `nearbyint`), unlike
> `[spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn]` which uses
> `std::round` (half-away-from-zero). NaN input yields an
> implementation-defined int64 conversion before clamping.

> [spec:et:def:op-add.torch.executor.native.quantized-add-out-fn]
> Tensor& quantized_add_out( const Tensor& a, double a_scale_d, int64_t a_zero_point_l, int64_t a_quant_min, int64_t a_quant_max, const Tensor& b, double b_scale_d, int64_t b_zero_point_l, int64_t b_quant_min, int64_t b_quant_max, double o...

> [spec:et:sem:op-add.torch.executor.native.quantized-add-out-fn]
> Quantized elementwise add operator, out variant. Validates, downcasts
> params, dispatches on integer dtype, and returns `out`. Steps
> (validation via `ET_CHECK_*` which ABORT on failure):
> - `ET_CHECK_SAME_SHAPE_AND_DTYPE3(a, b, out)` — `a`, `b`, `out` must
>   share shape and dtype.
> - For each of `a`, `b`, and out: require `quant_min >= 0 &&
>   quant_max <= 255 && quant_min <= quant_max` (the per-tensor
>   quant range must lie within unsigned-8-bit `[0,255]`). Checked for
>   `(a_quant_min,a_quant_max)`, `(b_quant_min,b_quant_max)`, and
>   `(out_quant_min,out_quant_max)`.
> - Downcast to maintain fbgemm numerical parity: `a_scale =
>   float(a_scale_d)`, `b_scale = float(b_scale_d)`, `out_scale =
>   float(out_scale_d)`; `a_zero_point = int32(a_zero_point_l)`,
>   `b_zero_point = int32(b_zero_point_l)`, `out_zero_point =
>   int32(out_zero_point_l)`. `a_quant_min`/`a_quant_max` for a and b
>   are NOT passed into the compute; only the OUTPUT quant range
>   (`out_quant_min`, `out_quant_max`) is used to clamp requantized
>   results.
> - Dispatch on `a.scalar_type()` over `ET_FORALL_INT_TYPES` (the
>   integer scalar types: Byte/uint8, Char/int8, Short/int16, Int/int32,
>   Long/int64). For the matching case call `add_tensors<CTYPE>(a,
>   a_scale, a_zero_point, b, b_scale, b_zero_point, out, out_scale,
>   out_zero_point, out_quant_min, out_quant_max)` per
>   `[spec:et:sem:op-add.torch.executor.native.add-tensors-fn]`. Any
>   non-integer dtype hits the `default` branch: `ET_CHECK_MSG(false,
>   ...)` aborts.
> - Return `out`.
> The `KernelRuntimeContext&`-taking overload ignores the context
> (`(void)context`) and forwards all arguments to this function,
> returning its result.

