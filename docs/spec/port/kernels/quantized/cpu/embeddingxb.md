# kernels/quantized/cpu/embeddingxb.cpp

> [spec:et:def:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]
> void check_embedding_xbit_args( const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const int64_t weight_quant_min, const int64_t weight_quant_max, const Tensor& indices, std::optional<...

> [spec:et:sem:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]
> Validates all arguments of the xbit embedding op. Every check uses
> `ET_CHECK_MSG`, which on failure ABORTS the process (fatal, not a
> recoverable context error). Returns void; performs no writes. Checks
> are performed in this order and all must pass:
> - `8 % weight_nbit == 0` (nbit must evenly divide a byte; supports
>   nbit values 1, 2, 4, 8).
> - `weight.dim() == 2` (weight is a 2D packed tensor).
> - `weight_scales.dim() == 1 || weight_scales.dim() == 2`.
> - `weight_scales.size(0) == weight.size(0)` (one scale row per
>   embedding row).
> - If `weight_scales.dim() == 2`: let `num_groups = weight_scales.size(1)`;
>   require `get_embedding_dim(weight.size(1), weight_nbit) % num_groups
>   == 0` (groups must evenly divide the unpacked embedding dimension,
>   per `[spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn]`).
> - `weight.scalar_type() == Byte` (uint8 packed weights).
> - `out.scalar_type()` is one of `Float`, `Half`, `BFloat16`.
> - `weight_scales.scalar_type()` is one of `Float`, `Half`, `BFloat16`.
> - If `opt_weight_zero_points` has a value:
>   - its `dim()` equals `weight_scales.dim()`;
>   - its `scalar_type()` equals `out.scalar_type()`;
>   - for each dim `i` in `[0, weight_scales.dim())`, its `size(i)`
>     equals `weight_scales.size(i)`.
> - `indices.scalar_type()` is `Long` or `Int`.
> - `weight_quant_min <= weight_quant_max`.
> - If `out_dtype` has a value: `out.scalar_type() == out_dtype.value()`.
> `weight_quant_min`, `weight_quant_max`, and `out_dtype` are only
> validated here (they are metadata not used in the dequant math).

> [spec:et:def:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]
> void embedding_xbit_per_channel( const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const Tensor& indices, Tensor& out, int weight_nbit)

> [spec:et:sem:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]
> Gathers the embedding rows named by `indices`, dequantizes each
> sub-bit packed weight, and writes fp results into `out`. Generic over
> `CTYPE_PARAMS` (scale/zero-point element type), `CTYPE_OUT` (output
> element type), and `CTYPE_INDICES` (Int or Long). Steps:
> - `embedding_dim = get_embedding_dim(weight.size(1), weight_nbit)` —
>   the unpacked number of columns per embedding
>   (`[spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn]`).
> - `num_groups_per_channel = 1`, or `weight_scales.size(1)` if
>   `weight_scales.dim() == 2`.
> - `group_size = embedding_dim / num_groups_per_channel` (integer
>   division; the caller has guaranteed exact divisibility via
>   `check_embedding_xbit_args`).
> - Obtain raw pointers: `out_data` (mutable `CTYPE_OUT*`),
>   `indices_ptr` (`CTYPE_INDICES*`), `scales` (`CTYPE_PARAMS*`), and
>   `zero_points` (`CTYPE_PARAMS*` or null when `opt_weight_zero_points`
>   is absent).
> - For each `i` in `[0, indices.numel())` in ascending order (flat
>   iteration over the indices tensor regardless of its rank):
>   - `index = indices_ptr[i]`.
>   - `qparams_index = index * num_groups_per_channel`; set
>     `scale_ptr = scales + qparams_index` and, if zero points present,
>     `zero_points_ptr = zero_points + qparams_index`. `zp` defaults to
>     0.
>   - `w_data = weight.const_data_ptr<uint8_t>() + weight.size(1) * index`
>     points at the packed bytes of row `index` (row stride is the
>     PACKED byte width `weight.size(1)`).
>   - For each output column `j` in `[0, embedding_dim)` in ascending
>     order:
>     - `group_id = j / group_size`; `scale = scale_ptr[group_id]`; if
>       zero points present `zp = zero_points_ptr[group_id]`.
>     - Unpack the quantized value `q = weight_value(w_data, j,
>       weight_nbit)` per
>       `[spec:et:sem:embeddingxb.torch.executor.native.weight-value-fn]`.
>     - `out_data[j] = static_cast<CTYPE_OUT>((float(q) - float(zp)) *
>       float(scale))`. Dequant math (subtract zero point, multiply by
>       scale) is done in `float`, then narrowed to `CTYPE_OUT`.
>   - Advance `out_data += embedding_dim` so successive indices write
>     consecutive contiguous output rows.
> No bounds/validation is done here (the caller runs
> `check_embedding_xbit_args` first). Assumes `out` is contiguous.

> [spec:et:def:embeddingxb.torch.executor.native.get-embedding-dim-fn]
> static inline int32_t get_embedding_dim( int32_t packed_dim, int32_t weight_nbit)

> [spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn]
> Converts a packed byte-width `packed_dim` into the unpacked number of
> quantized values (the true embedding dimension) for a given
> `weight_nbit`. Steps:
> - `ET_CHECK_MSG(8 % weight_nbit == 0, ...)` — aborts if `weight_nbit`
>   does not evenly divide 8.
> - `packed_values_per_byte = 8 / weight_nbit` (int division).
> - Returns `packed_dim * packed_values_per_byte` as int32.
> Example: nbit=2 -> 4 values/byte; nbit=4 -> 2 values/byte. `packed_dim`
> is typically `weight.size(1)` (the packed column count of the weight
> tensor).

> [spec:et:def:embeddingxb.torch.executor.native.resize-out-tensor-fn]
> void resize_out_tensor( const Tensor& weight, const Tensor& indices, Tensor& out, int weight_nbit)

> [spec:et:sem:embeddingxb.torch.executor.native.resize-out-tensor-fn]
> Resizes `out` so its shape is the indices shape with the last
> dimension replaced by the unpacked embedding dimension. Steps:
> - Build `expected_output_size[kTensorDimensionLimit]`: for `i` in
>   `[0, indices.dim())` set `expected_output_size[i] = indices.size(i)`.
> - `embedding_dim = get_embedding_dim(weight.size(1), weight_nbit)`
>   (`[spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn]`).
> - Overwrite the last slot at index `out.dim() - 1` with
>   `embedding_dim`. (The intended output rank equals `indices.dim() +
>   1`, i.e. `out.dim() == indices.dim() + 1`; this writes indices'
>   dims into slots `[0, indices.dim())` and the embedding dim into the
>   final slot.)
> - Form an ArrayRef of length `out.dim()` over `expected_output_size`
>   and call `resize_tensor(out, output_size)`.
> - `ET_CHECK_MSG(err == Error::Ok, ...)` — aborts if the resize fails
>   (e.g. a static/bounded output tensor that cannot hold the shape).
> Mutates only `out`'s size metadata; does not write element data.

> [spec:et:def:embeddingxb.torch.executor.native.weight-value-fn]
> static inline int32_t

> [spec:et:sem:embeddingxb.torch.executor.native.weight-value-fn]
> Unpacks and dequantizes-to-signed-integer the `index`-th sub-byte
> value from packed weight bytes `w_data` for a given `weight_nbit`,
> returning an int32. Packing is little-value-order within each byte
> (lowest bit positions hold the earliest logical value) for the 2-bit
> case, and high-nibble-first for the 4-bit case. Behavior by nbit:
> - `weight_nbit == 2`: `subbyte = index % 4`; `byte = w_data[index >> 2]`
>   (4 values per byte). Extract the 2-bit field by `subbyte`:
>   - 0: `(byte & 0x03)` (bits 0-1)
>   - 1: `((byte & 0x0C) >> 2)` (bits 2-3)
>   - 2: `((byte & 0x30) >> 4)` (bits 4-5)
>   - 3: `((byte & 0xC0) >> 6)` (bits 6-7)
>   Then subtract the symmetric bias 2: result = field - 2 (maps
>   unsigned [0,3] to signed [-2,1]).
> - `weight_nbit == 4`: `odd = index & 1`; `byte = w_data[index >> 1]`
>   (2 values per byte). If `odd` (odd index): take the LOW nibble
>   `(byte & 0x0F)`; if even index: take the HIGH nibble
>   `((byte >> 4) & 0x0F)`. So within a byte the even (first) value is
>   the high nibble and the odd (second) value is the low nibble. Then
>   subtract bias 8: result = nibble - 8 (maps [0,15] to [-8,7]).
> - Any other `weight_nbit`: `ET_CHECK_MSG(false, "invalid weight_nbit")`
>   aborts.
> Only nbit 2 and 4 are handled by this function (despite
> `get_embedding_dim` permitting 1 and 8).

> [spec:et:def:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]
> Tensor& quantized_embedding_xbit_dtype_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& weight_scale...

> [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]
> Context-taking out variant of the xbit embedding op that allows the
> dequant params dtype to differ from the output dtype. Steps:
> - `resize_out_tensor(weight, indices, out, weight_nbit)`
>   (`[spec:et:sem:embeddingxb.torch.executor.native.resize-out-tensor-fn]`).
> - `check_embedding_xbit_args(weight, weight_scales,
>   opt_weight_zero_points, weight_quant_min, weight_quant_max, indices,
>   out_dtype, out, weight_nbit)`
>   (`[spec:et:sem:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]`);
>   any failure aborts.
> - `params_type = weight_scales.scalar_type()`, `out_type =
>   out.scalar_type()`, `indices_type = indices.scalar_type()`.
> - Triple-nested dtype dispatch (op name
>   "quantized_decomposed::embedding_xbit.dtype_out"):
>   `ET_SWITCH_THREE_TYPES(Float, Half, BFloat16, params_type)` selects
>   `CTYPE_P`; inside, `ET_SWITCH_THREE_TYPES(Float, Half, BFloat16,
>   out_type)` selects `CTYPE_OUT`; inside, `ET_SWITCH_TWO_TYPES(Int,
>   Long, indices_type)` selects `CTYPE_IDX`. An unhandled dtype in any
>   switch sets `Error::InvalidArgument` on `ctx` (the ET_SWITCH
>   failure path) and skips the body. Then call
>   `embedding_xbit_per_channel<CTYPE_P, CTYPE_OUT, CTYPE_IDX>(weight,
>   weight_scales, opt_weight_zero_points, indices, out, weight_nbit)`
>   per
>   `[spec:et:sem:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]`.
> - Returns `out`.
> The non-context overload of the same name constructs a local
> `KernelRuntimeContext`, forwards to this function, then
> `ET_CHECK(context.failure_state() == Error::Ok)` (aborting on a
> dispatch failure) before returning `out`.

> [spec:et:def:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]
> Tensor& quantized_embedding_xbit_out( // TODO Evaluate whether this name is appropriate for an operator that takes // non quant input and returns fp output KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& weight_scales, con...

> [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]
> Context-taking out variant of the xbit embedding op where the dequant
> params dtype is forced to equal the output dtype. Steps:
> - `out_type = out.scalar_type()`.
> - `resize_out_tensor(weight, indices, out, weight_nbit)`
>   (`[spec:et:sem:embeddingxb.torch.executor.native.resize-out-tensor-fn]`).
> - `check_embedding_xbit_args(weight, weight_scales,
>   opt_weight_zero_points, weight_quant_min, weight_quant_max, indices,
>   out_type, out, weight_nbit)` — passing `out_type` as the `out_dtype`
>   argument
>   (`[spec:et:sem:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]`);
>   any failure aborts.
> - `indices_type = indices.scalar_type()`.
> - Double-nested dtype dispatch (op name
>   "quantized_decomposed::embedding_xbit.out"):
>   `ET_SWITCH_THREE_TYPES(Float, Half, BFloat16, out_type)` selects
>   `CTYPE_OUT`; inside, `ET_SWITCH_TWO_TYPES(Int, Long, indices_type)`
>   selects `CTYPE_IDX`. An unhandled dtype sets `Error::InvalidArgument`
>   on `ctx` and skips the body. Then call
>   `embedding_xbit_per_channel<CTYPE_OUT, CTYPE_OUT, CTYPE_IDX>(...)`
>   — note the scale/zero-point element type `CTYPE_PARAMS` is set to
>   `CTYPE_OUT`, so scales/zero-points are read as the output dtype
>   (`[spec:et:sem:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]`).
> - Returns `out`.
> The non-context overload constructs a local `KernelRuntimeContext`,
> forwards, then `ET_CHECK(context.failure_state() == Error::Ok)` and
> returns `out`.

