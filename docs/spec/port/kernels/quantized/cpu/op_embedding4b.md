# kernels/quantized/cpu/op_embedding4b.cpp

> [spec:et:def:op-embedding4b.torch.executor.native.quantized-embedding-4bit-dtype-out-fn]
> Tensor& quantized_embedding_4bit_dtype_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output const Tensor& weight, const Tensor& weight_scales, const std::optional<Tens...

> [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-dtype-out-fn]
> Thin wrapper: forwards all arguments to
> `quantized_embedding_xbit_dtype_out(..., out_dtype, out, weight_nbit=4)` per
> `[spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]`,
> returning its result (`out`).
>
> Behavioral summary of the delegated shared kernel with `weight_nbit == 4`
> (packed_values_per_byte = 8/4 = 2): the uint8 `weight` tensor holds 2 packed
> 4-bit values per byte; `embedding_dim = get_embedding_dim(weight.size(1), 4) =
> weight.size(1) * 2`. It resizes `out` (rows = indices shape, last dim =
> embedding_dim), validates args via `check_embedding_xbit_args` (weight must be
> Byte; indices may be Int or Long; scales/out are Float/Half/BFloat16; groups
> divide embedding_dim), dispatches over params/out/indices dtypes, and for each
> index dequantizes each of the `embedding_dim` unpacked values via
> `weight_value(w_data, j, 4)`. Note the 4-bit unpacking order: for element `j`,
> `odd = j & 1`, byte index `j >> 1`; when `odd` it takes the low nibble
> `(byte & 0x0F) - 8`, else the high nibble `((byte >> 4) & 0x0F) - 8`. Each
> output is `(value - zp) * scale` in float with per-group scale/zero_point.
> quant_min/quant_max/out_dtype are metadata. This variant accepts the explicit
> `out_dtype` and dispatches params dtype separately from out dtype. Both a
> context-taking and a non-context overload exist and behave identically.

> [spec:et:def:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn]
> Tensor& quantized_embedding_4bit_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& o...

> [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn]
> Thin wrapper: forwards all arguments to
> `quantized_embedding_xbit_out(..., out, weight_nbit=4)` per
> `[spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]`,
> returning its result (`out`).
>
> Behavioral summary of the delegated shared kernel with `weight_nbit == 4`:
> like the dtype variant above but without an explicit `out_dtype` argument — the
> shared kernel forces the weight_scales ctype equal to the out ctype (CTYPE_P ==
> CTYPE_OUT) and dispatches only over out dtype (Float/Half/BFloat16) and indices
> dtype (Int/Long). `embedding_dim = weight.size(1) * 2`; each output element is
> `(weight_value(w_data, j, 4) - zp) * scale` computed in float, using the 4-bit
> nibble unpacking described in the dtype variant, with per-group
> scale/zero_point. `out` is resized and validated exactly as in the dtype
> variant. Both a context-taking and a non-context overload exist and behave
> identically.

