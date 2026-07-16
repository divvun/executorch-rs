# kernels/quantized/cpu/op_embedding2b.cpp

> [spec:et:def:op-embedding2b.torch.executor.native.quantized-embedding-2bit-dtype-out-fn]
> Tensor& quantized_embedding_2bit_dtype_out( KernelRuntimeContext& context, const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, int64_t weight_quant_min, int64_t weight_quant_max, const ...

> [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-dtype-out-fn]
> Thin wrapper: forwards all arguments to
> `quantized_embedding_xbit_dtype_out(..., out_dtype, out, weight_nbit=2)` per
> `[spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]`,
> returning its result (`out`).
>
> Behavioral summary of the delegated shared kernel with `weight_nbit == 2`
> (packed_values_per_byte = 8/2 = 4): the uint8 `weight` tensor holds 4 packed
> 2-bit values per byte; `embedding_dim = get_embedding_dim(weight.size(1), 2) =
> weight.size(1) * 4`. It resizes `out` (rows = indices shape, last dim =
> embedding_dim), validates args via `check_embedding_xbit_args` (weight must be
> Byte; indices may be Int or Long; scales/out are Float/Half/BFloat16; groups
> divide embedding_dim), dispatches over params/out/indices dtypes, and for each
> index dequantizes each of the `embedding_dim` unpacked values via
> `weight_value(w_data, j, 2)` (extract the 2-bit field, subtract 2 to center),
> applying per-group `scale`/`zero_point` as
> `(value - zp) * scale` in float. quant_min/quant_max/out_dtype are metadata.
> This variant accepts the explicit `out_dtype` and dispatches params dtype
> separately from out dtype. Both a context-taking and a non-context overload
> exist and behave identically.

> [spec:et:def:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn]
> Tensor& quantized_embedding_2bit_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& o...

> [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn]
> Thin wrapper: forwards all arguments to
> `quantized_embedding_xbit_out(..., out, weight_nbit=2)` per
> `[spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]`,
> returning its result (`out`).
>
> Behavioral summary of the delegated shared kernel with `weight_nbit == 2`:
> like the dtype variant above but without an explicit `out_dtype` argument — the
> shared kernel forces the weight_scales ctype equal to the out ctype (CTYPE_P ==
> CTYPE_OUT) and dispatches only over out dtype (Float/Half/BFloat16) and indices
> dtype (Int/Long). `embedding_dim = weight.size(1) * 4`; each output element is
> `(weight_value(w_data, j, 2) - zp) * scale` computed in float, with per-group
> scale/zero_point. `out` is resized and validated exactly as in the dtype
> variant. Both a context-taking and a non-context overload exist and behave
> identically.

