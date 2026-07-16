# kernels/quantized/cpu/op_embedding.cpp

> [spec:et:def:op-embedding.torch.executor.native.check-embedding-byte-args-fn]
> void check_embedding_byte_args( const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const int64_t weight_quant_min, const int64_t weight_quant_max, const Tensor& indices, std::optional<...

> [spec:et:sem:op-embedding.torch.executor.native.check-embedding-byte-args-fn]
> Validates arguments for the byte (8-bit) quantized embedding. All checks are
> ET_CHECK_MSG (abort/fatal on failure). Order:
> 1. `weight.dim() == 2`.
> 2. `weight_scales.dim() == 1 || weight_scales.dim() == 2`.
> 3. `weight_scales.size(0) == weight.size(0)` (one scale row per embedding).
> 4. If `weight_scales.dim() == 2`: let `num_groups = weight_scales.size(1)`;
>    require `weight.size(1) % num_groups == 0` (groups evenly divide the
>    embedding dim).
> 5. `weight.scalar_type()` is `Byte` (uint8) or `Char` (int8).
> 6. `out.scalar_type()` is `Float`, `Half`, or `BFloat16`.
> 7. `weight_scales.scalar_type()` is `Float`, `Half`, or `BFloat16`.
> 8. If `opt_weight_zero_points` has a value:
>    - its `dim()` equals `weight_scales.dim()`;
>    - its scalar_type equals `out.scalar_type()`;
>    - for each `i` in `[0, weight_scales.dim())`, its `size(i)` equals
>      `weight_scales.size(i)`.
> 9. `indices.scalar_type() == ScalarType::Long` (int64) — byte embedding only
>    accepts Long indices.
> 10. `weight_quant_min <= weight_quant_max`.
> 11. If `out_dtype` has a value, `out.scalar_type() == out_dtype.value()`.
>
> quant_min/quant_max and dtype are metadata; only the min<=max relation is
> checked and they are not used in computation.

> [spec:et:def:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]
> void embedding_byte_per_channel( const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const Tensor& indices, Tensor& out)

> [spec:et:sem:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]
> Templated on (CTYPE_WEIGHT, CTYPE_PARAMS, CTYPE_OUT). For each index, gathers
> the corresponding weight row, dequantizes it (per-channel, optionally
> groupwise), and writes it to `out`.
>
> Setup:
> - `embedding_dim = weight.size(1)`.
> - `num_groups_per_channel = weight_scales.dim() == 2 ? weight_scales.size(1) : 1`.
> - `group_size = weight.size(1) / num_groups_per_channel` (integer division;
>    checked divisible by `check_embedding_byte_args`).
> - `out_data = out.mutable_data_ptr<CTYPE_OUT>()`;
>   `indices_ptr = indices.const_data_ptr<int64_t>()`;
>   `scales = weight_scales.const_data_ptr<CTYPE_PARAMS>()`;
>   `zero_points = opt_weight_zero_points ? const_data_ptr<CTYPE_PARAMS>() : nullptr`.
>
> For each `i` in `[0, indices.numel())`:
> 1. `index = indices_ptr[i]`.
> 2. ET_CHECK_MSG `0 <= index < weight.size(0)` (abort otherwise).
> 3. ET_CHECK_MSG `0 <= index < weight_scales.size(0)` (abort otherwise).
> 4. `qparams_index = index * num_groups_per_channel`;
>    `scale_ptr = scales + qparams_index`;
>    `zero_points_ptr = opt_weight_zero_points ? zero_points + qparams_index : nullptr`;
>    `zp = 0`.
> 5. `w_data = weight.const_data_ptr<CTYPE_WEIGHT>() + embedding_dim * index`
>    (row `index` of the weight matrix).
> 6. For each `j` in `[0, embedding_dim)`:
>    - `group_id = j / group_size`;
>    - `scale = scale_ptr[group_id]`;
>    - if zero points present, `zp = zero_points_ptr[group_id]`;
>    - `out_data[j] = CTYPE_OUT((float(w_data[j]) - float(zp)) * float(scale))`.
>      Dequantization is computed entirely in `float` (weight, zp, and scale each
>      promoted to float), then cast to CTYPE_OUT.
> 7. Advance `out_data += embedding_dim`.
>
> So the N output rows are produced in index order, each row being the
> dequantized weight row selected by that index; the flat output layout is
> `[indices.numel(), embedding_dim]`.

> [spec:et:def:op-embedding.torch.executor.native.resize-out-tensor-fn]
> void resize_out_tensor( const Tensor& weight, const Tensor& indices, Tensor& out)

> [spec:et:sem:op-embedding.torch.executor.native.resize-out-tensor-fn]
> Resizes `out` to the shape implied by `indices` and `weight`.
>
> Steps:
> 1. Build `expected_output_size` by copying `indices.size(i)` for each `i` in
>    `[0, indices.dim())`.
> 2. Set the last entry `expected_output_size[out.dim()-1] = weight.size(1)`
>    (the embedding dim). Note the array length used is `out.dim()`; the caller
>    guarantees `out.dim() == indices.dim() + 1` (indices shape followed by
>    embedding dim), so this overwrites/sets the trailing embedding dimension.
> 3. `resize_tensor(out, {expected_output_size, out.dim()})`; ET_CHECK_MSG the
>    result is `Error::Ok` (abort otherwise).

> [spec:et:def:op-embedding.torch.executor.native.quantized-embedding-byte-dtype-out-fn]
> Tensor& quantized_embedding_byte_dtype_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& weight_scale...

> [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-dtype-out-fn]
> Context-taking dtype variant of the byte quantized embedding. Allows the
> weight_scales dtype (CTYPE_PARAMS) to differ from the output dtype. Returns `out`.
>
> Steps:
> 1. `resize_out_tensor(weight, indices, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.resize-out-tensor-fn]`.
> 2. `check_embedding_byte_args(weight, weight_scales, opt_weight_zero_points,
>    weight_quant_min, weight_quant_max, indices, out_dtype, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.check-embedding-byte-args-fn]`.
> 3. Read `weight_type = weight.scalar_type()`, `params_type =
>    weight_scales.scalar_type()`, `out_type = out.scalar_type()`.
> 4. Dispatch: `weight_type` over {Byte, Char}; `params_type` over
>    {Float, Half, BFloat16}; `out_type` over {Float, Half, BFloat16}; using
>    ET_SWITCH macros keyed on `ctx` (an unhandled type routes the kernel-check
>    failure through `ctx`). Call
>    `embedding_byte_per_channel<CTYPE_W, CTYPE_P, CTYPE_OUT>(weight,
>    weight_scales, opt_weight_zero_points, indices, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]`.
> 5. Return `out`.
>
> A non-context overload with the same behavior exists: it constructs a local
> `KernelRuntimeContext`, calls this function, ET_CHECKs
> `context.failure_state() == Error::Ok`, and returns the result.

> [spec:et:def:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn]
> Tensor& quantized_embedding_byte_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& weight_scales, con...

> [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn]
> Context-taking byte quantized embedding (out variant of
> torch.ops.quantized.embedding_byte). Here weight_scales dtype is not dispatched
> separately: CTYPE_PARAMS is forced equal to CTYPE_OUT. Returns `out`.
>
> Steps:
> 1. Read `w_type = weight.scalar_type()`, `out_type = out.scalar_type()`.
> 2. `resize_out_tensor(weight, indices, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.resize-out-tensor-fn]`.
> 3. `check_embedding_byte_args(weight, weight_scales, opt_weight_zero_points,
>    weight_quant_min, weight_quant_max, indices, out_type, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.check-embedding-byte-args-fn]`
>    (passes `out_type` as the `out_dtype` argument).
> 4. Dispatch: `w_type` over {Byte, Char}; `out_type` over {Float, Half, BFloat16}
>    (ET_SWITCH keyed on `ctx`). Call
>    `embedding_byte_per_channel<CTYPE_W, CTYPE_OUT, CTYPE_OUT>(weight,
>    weight_scales, opt_weight_zero_points, indices, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]`
>    — note the params template type is CTYPE_OUT, so weight_scales are read using
>    the output ctype (which check_embedding_byte_args constrains to equal
>    out.scalar_type() only via the zero-point check; weight_scales itself may be
>    any of Float/Half/BFloat16, so in practice callers pass scales matching out).
> 5. Return `out`.
>
> A non-context overload exists that constructs a local `KernelRuntimeContext`,
> forwards, ET_CHECKs `failure_state() == Error::Ok`, and returns the result.

