# kernels/quantized/cpu/op_choose_qparams.cpp

> [spec:et:def:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn]
> std::tuple<Tensor&, Tensor&> choose_qparams_per_token_asymmetric_out( const Tensor& input, ScalarType dtype, Tensor& scale_out, Tensor& zero_point_out)

> [spec:et:sem:op-choose-qparams.choose-qparams-per-token-asymmetric-out-fn]
> Per-token asymmetric qparams out variant. Hardcodes the quant range
> `quant_min = -128`, `quant_max = 127` (signed int8). Steps (the
> annotated function is the non-context overload; a `RuntimeContext&`
> overload exists that ignores context and forwards here):
> - Build `output_sizes`: for `i` in `[0, input.dim()-1)` set
>   `output_sizes[i] = input.size(i)`; set the last dim
>   `output_sizes[input.dim()-1] = 1`; `output_dim = input.dim()`. So
>   the output shape is the input shape with the last dim collapsed to
>   1 (one qparam per token).
> - `resize_tensor(scale_out, {output_sizes, output_dim})`; abort via
>   `ET_CHECK_MSG` if not `Error::Ok`. Same for `zero_point_out`.
> - `check_quantize_per_tensor_args(input, quant_min, quant_max, dtype,
>   scale_out, zero_point_out, is_per_token=true)`
>   (`[spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]`);
>   aborts on failure.
> - `choose_qparams_per_token(input, quant_min, quant_max, scale_out,
>   zero_point_out)`
>   (`[spec:et:sem:op-choose-qparams.torch.choose-qparams-per-token-fn]`),
>   filling one (scale, zero_point) per token.
> - Return the tuple `{scale_out, zero_point_out}`. Unlike the
>   per-tensor variant, this one resizes the out tensors itself.
> `dtype` is passed to the arg-check only (unused in the math); the
> effective quant range is always [-128, 127] regardless of `dtype`.

> [spec:et:def:op-choose-qparams.choose-qparams-tensor-out-fn]
> ::std::tuple<Tensor&, Tensor&> choose_qparams_tensor_out( KernelRuntimeContext& context, const Tensor& input, int64_t quant_min, int64_t quant_max, double eps, ScalarType dtype, Tensor& scale_out, Tensor& zero_point_out)

> [spec:et:sem:op-choose-qparams.choose-qparams-tensor-out-fn]
> Context-taking out variant computing per-tensor asymmetric qparams.
> `eps` is accepted but unused (marked ET_UNUSED). The context is
> ignored (`(void)context`). Delegates to the non-context overload,
> which:
> - Calls `check_quantize_per_tensor_args(input, quant_min, quant_max,
>   dtype, scale_out, zero_point_out)` (is_per_token defaults false)
>   (`[spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]`);
>   validation aborts on failure.
> - Calls `choose_qparams(input, quant_min, quant_max, scale_out,
>   zero_point_out)`
>   (`[spec:et:sem:op-choose-qparams.torch.choose-qparams-fn]`),
>   writing scale/zero-point into element 0 of each out tensor.
> - Returns the tuple `{scale_out, zero_point_out}` (references to the
>   passed-in out tensors). Does NOT resize the out tensors (caller
>   must pre-size them to 1 element each).

> [spec:et:def:op-choose-qparams.torch.choose-qparams-fn]
> void choose_qparams( const Tensor& input, int32_t qmin, int32_t qmax, Tensor& scale_out, Tensor& zero_point_out)

> [spec:et:sem:op-choose-qparams.torch.choose-qparams-fn]
> Computes a single (scale, zero_point) pair for the whole `input`
> tensor (per-tensor asymmetric) and writes them into `scale_out[0]`
> and `zero_point_out[0]`. Steps:
> - `x_fp32 = input.const_data_ptr<float>()`.
> - `min = vec_minf(x_fp32, input.numel())` and `max = vec_maxf(x_fp32,
>   input.numel())` over all elements
>   (`[spec:et:sem:vec-ops.torch.executor.vec-minf-fn]`,
>   `[spec:et:sem:vec-ops.torch.executor.vec-maxf-fn]`; requires
>   `numel >= 1`).
> - Call the private `calculate_scale_and_zero_point(min, max, qmin,
>   qmax, scale, zero_point)` which produces a `double scale` and
>   `int32_t zero_point`. That helper (described here since it has no
>   separate rule): (1) extends the range to include 0 via `min =
>   min(min, 0)`, `max = max(max, 0)`; (2) `scale = (double(max) - min)
>   / (qmax - qmin)`; (3) if `float(scale) == 0` or `1.0f/float(scale)`
>   is inf, sets `scale = 0.1`; asserts `scale > 0` (ET_CHECK_MSG,
>   abort); (4) if `scale < SMALL_SCALE_THRESHOLD` (6.1e-5f) it raises
>   `scale` to that threshold and rescales `min`/`max` accordingly
>   (if `min==0`: `max = threshold*(qmax-qmin)`; elif `max==0`: `min =
>   -threshold*(qmax-qmin)`; else scale both by `threshold/org_scale`);
>   (5) computes zero point from whichever of the (rmin,qmin)/(rmax,qmax)
>   affine equations has smaller absolute error term
>   (`initial_zero_point = zero_point_from_min` if
>   `|qmin| - |min/scale| < |qmax| - |max/scale|` else
>   `zero_point_from_max`); (6) nudges to int: clamp to `qmin` if below,
>   `qmax` if above, else `nearbyint(float(initial_zero_point))`.
> - Write `scale_out.mutable_data_ptr<double>()[0] = scale` and
>   `zero_point_out.mutable_data_ptr<int64_t>()[0] = zero_point`.

> [spec:et:def:op-choose-qparams.torch.choose-qparams-per-token-fn]
> void choose_qparams_per_token( const Tensor& input, int32_t qmin, int32_t qmax, Tensor& scale_out, Tensor& zero_point_out)

> [spec:et:sem:op-choose-qparams.torch.choose-qparams-per-token-fn]
> Computes one (scale, zero_point) pair per token, where a "token" is a
> contiguous run of `token_dim_size = input.size(input.dim()-1)`
> elements and there are `num_tokens = product of input.size(i) for i in
> [0, input.dim()-1)` tokens. Writes `scale_out[i]` and
> `zero_point_out[i]` for each token `i`. Steps:
> - `x_fp32 = input.const_data_ptr<float>()`.
> - Compute `num_tokens` and `token_dim_size` as above;
>   `total_elements = num_tokens * token_dim_size`.
> - `use_parallel = total_elements >= 512` (MIN_ELEMENTS_FOR_PARALLEL).
> - Parallel path (use_parallel true): via
>   `executorch::extension::parallel_for(0, num_tokens, 1, ...)`, for
>   each token `i` in the assigned `[begin, end)` range: `token_data =
>   x_fp32 + i * token_dim_size`; `min = vec_minf(token_data,
>   token_dim_size)`, `max = vec_maxf(token_data, token_dim_size)`
>   (`[spec:et:sem:vec-ops.torch.executor.vec-minf-fn]`,
>   `[spec:et:sem:vec-ops.torch.executor.vec-maxf-fn]`); compute
>   (scale, zero_point) via the same `calculate_scale_and_zero_point`
>   algorithm described in
>   `[spec:et:sem:op-choose-qparams.torch.choose-qparams-fn]`; write
>   `scale_data[i] = scale`, `zero_point_data[i] = zero_point`.
> - Serial path (use_parallel false): for `i` in `[0, num_tokens)`
>   ascending: `min = vec_minf(x_fp32, token_dim_size)`, `max =
>   vec_maxf(x_fp32, token_dim_size)`; compute (scale, zero_point) the
>   same way; write `scale_out[i]`, `zero_point_out[i]`; then advance
>   `x_fp32 += token_dim_size`. (The serial path walks the pointer;
>   the parallel path indexes by `i` — same tokens either way.)
> Result is identical regardless of the parallel/serial choice; only
> the traversal differs.

> [spec:et:def:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]
> void check_quantize_per_tensor_args( const Tensor& input, int64_t qmin, int64_t qmax, ScalarType dtype, Tensor& scale_out, Tensor& zero_point_out, bool is_per_token = false)

> [spec:et:sem:op-choose-qparams.torch.executor.native.check-quantize-per-tensor-args-fn]
> Validates args for choose_qparams. `dtype` is accepted but unused
> (`(void)dtype`). All checks use `ET_CHECK_MSG` (ABORT on failure);
> returns void, no writes. Checks in order:
> - `qmin < qmax` (strictly less).
> - `input.scalar_type() == Float`.
> - `scale_out.scalar_type() == Double`.
> - `zero_point_out.scalar_type() == Long`.
> - Then branch on `is_per_token` (default false):
>   - If `is_per_token` true: for each dim `i` in `[0, input.dim()-1)`
>     require `scale_out.size(i) == input.size(i)` and
>     `zero_point_out.size(i) == input.size(i)`; and require the last
>     dim `scale_out.size(input.dim()-1) == 1` and
>     `zero_point_out.size(input.dim()-1) == 1` (one qparam per token,
>     tokens being all-but-last dims).
>   - Else (per-tensor): require `scale_out.numel() == 1` and
>     `zero_point_out.numel() == 1`.

