# kernels/portable/cpu/util/kernel_ops_util.cpp, kernels/portable/cpu/util/kernel_ops_util.h

> [spec:et:def:kernel-ops-util.torch.executor.apply-kernel-2d-reduce-then-map-fn-fn]
> void apply_kernel_2d_reduce_then_map_fn( const ReduceOp& reduce_fn, const MapOp& map_fn, const bool include_pad, const Tensor& in, const IntArrayRef kernel_size, const IntArrayRef stride, const IntArrayRef padding, const IntArrayRef dila...

> [spec:et:sem:kernel-ops-util.torch.executor.apply-kernel-2d-reduce-then-map-fn-fn]
> Header-only template function `apply_kernel_2d_reduce_then_map_fn<CTYPE, ReduceOp, MapOp>`. Drives a 2D windowed reduction over every batch and channel of a 3-D {C,H,W} or 4-D {N,C,H,W} tensor, delegating the per-channel window sweep to `[spec:et:sem:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn]`.
>
> Steps:
> 1. Read `in_sizes = in.sizes()` and `out_sizes = out.sizes()` (ArrayRefs of SizesType).
> 2. Read `in_dim_order = in.dim_order()` and `out_dim_order = out.dim_order()`.
> 3. Compute contiguous-per-dim-order strides for the input into a local buffer `in_strides[kTensorDimensionLimit]` via `dim_order_to_stride_nocheck(in_sizes.data(), in_dim_order.data(), in_sizes.size(), in_strides)` (converts a dim_order into the element strides that materialize that memory layout; no validation).
> 4. Likewise compute `out_strides[kTensorDimensionLimit]` from `out_sizes`/`out_dim_order`.
> 5. Obtain `out_ptr = out.mutable_data_ptr<CTYPE>()` and `in_ptr = in.const_data_ptr<CTYPE>()`.
> 6. Set `indices_ptr = nullptr`; if the optional `indices` tensor has a value, set `indices_ptr = indices.value().mutable_data_ptr<int64_t>()`.
> 7. Determine `batch_size`: default 1; if `in.dim() == 4`, `batch_size = in_sizes[0]`.
> 8. For each `batch` in `[0, batch_size)`, and within it for each `channel` in `[0, in_sizes[in.dim() - 3])` (i.e. the channel axis: index 1 for 4-D, index 0 for 3-D), call `kernel_reduction_then_map_2d(reduce_fn, map_fn, include_pad, in_ptr, in_sizes, {in_strides, 4}, kernel_size, stride, padding, dilation, out_ptr, out_sizes, {out_strides, 4}, indices_ptr, batch, channel)`.
>
> Note the strides ArrayRefs are always constructed with length 4 (`{in_strides, 4}` / `{out_strides, 4}`) regardless of actual rank; the callee only indexes strides via `calculate_linear_index` over the true `in_dim`/`out_dim`, so trailing entries are unused. Returns void; results are written in place into `out` (and `indices` if provided). No dtype dispatch here — CTYPE is fixed by the caller; no argument validation is performed (callers must have validated shapes/dtypes beforehand, e.g. via the avg_pool2d/max_pool2d check_* rules).

> [spec:et:def:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]
> void calculate_kernel_output_sizes( const Tensor& in, size_t kernel_ndim, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, executorch::aten::SizesType* out_sizes, bool ceil_mode, bool transposed, In...

> [spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]
> Writes the spatial output extents for an N-dim kernel op into `out_sizes`. Only the last `kernel_ndim` dimensions of `out_sizes` are touched; the leading (batch/channel) dimensions are left unchanged (callers fill those in separately).
>
> For each `i` in `[0, kernel_ndim)`:
> 1. `dim = in.dim() - (kernel_ndim - i)` — the input/output dimension index for this kernel axis (the last `kernel_ndim` axes, in order).
> 2. `k = val_at(kernel_size, i)` — kernel extent for this axis, default 1 (see `[spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]`).
> 3. `s = val_at(stride, i, default_value=k)` — stride, defaulting to `k` when the stride array is empty.
> 4. `d = val_at(dilation, i, default_value=1)` — dilation, default 1.
> 5. `p = val_at(padding, i, default_value=0)` — padding, default 0.
> 6. `op = transposed ? val_at(output_padding, i, default_value=0) : 0` — output padding used only in transposed mode.
> 7. `out_sizes[dim] = _kernel_output_size_helper(in.size(dim), k, p, s, d, ceil_mode, transposed, op)` per `[spec:et:sem:kernel-ops-util.torch.executor.kernel-output-size-helper-fn]`.
>
> Defaults: `ceil_mode=false`, `transposed=false`, `output_padding={}`. Returns void; no validation (assumes the *_is_valid checks already passed).

> [spec:et:def:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]
> bool check_adaptive_avg_pool2d_args( const Tensor& in, const IntArrayRef output_size, const Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]
> Validates arguments for adaptive_avg_pool2d. Returns `bool`; every check uses the log-and-return-false idiom (`ET_LOG_AND_RETURN_IF_FALSE` / `ET_CHECK_OR_RETURN_FALSE`): on the first failing check it logs an Error message and returns `false`; if all pass it returns `true`. Callers (the op kernel) gate on the return value via `ET_KERNEL_CHECK`, which on false sets `Error::InvalidArgument` on the context and returns `out` unchanged.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, out)` — input and output must have identical scalar type.
> 2. `tensor_is_default_or_channels_last_dim_order(in)` — input dim_order must be contiguous (default) or channels-last.
> 3. `tensor_is_default_or_channels_last_dim_order(out)` — same for output.
> 4. Rank/positivity: either (`in.dim() == 3` and `in.size(0)>0 && in.size(1)>0 && in.size(2)>0`) OR (`in.dim() == 4` and `in.size(1)>0 && in.size(2)>0 && in.size(3)>0`). I.e. 3-D {C,H,W} requires all three dims positive; 4-D {N,C,H,W} requires C,H,W positive (batch dim 0 may be 0).
> 5. `output_size.size() == 2` — exactly two target spatial extents.
> 6. `output_size[0] > 0 && output_size[1] > 0` — both target extents strictly positive.
>
> Returns true only if all pass.

> [spec:et:def:kernel-ops-util.torch.executor.check-alpha-type-fn]
> bool check_alpha_type( const ScalarType alpha_type, const ScalarType common_type)

> [spec:et:sem:kernel-ops-util.torch.executor.check-alpha-type-fn]
> Verifies that a scalar `alpha` of ScalarType `alpha_type` is compatible with the computed `common_type` of an op such as add/sub (where the result is `in + alpha * other`). Returns `bool`.
>
> Single check (`ET_LOG_AND_RETURN_IF_FALSE`): passes iff
> `canCast(alpha_type, common_type)` is true, OR (`common_type == ScalarType::Bool` AND `isIntegralType(alpha_type, /*includeBool=*/true)` is true).
>
> `canCast(from, to)` is the standard PyTorch type-promotion castability predicate (true when a value of `from` can be safely represented in `to` under promotion rules). The second clause additionally permits an integral (or bool) alpha when the common type is Bool, since `canCast` would otherwise reject integral→Bool. On failure logs an Error and returns `false`; otherwise returns `true`.

> [spec:et:def:kernel-ops-util.torch.executor.check-arange-args-fn]
> bool check_arange_args(double start, double end, double step, Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-arange-args-fn]
> Validates arguments for the `arange.start_step` op. Returns `bool`.
>
> Checks in order (each `ET_CHECK_OR_RETURN_FALSE`: logs Error and returns `false` on failure):
> 1. `out.dim() == 1` — output must be a 1-D tensor.
> 2. Step/bound consistency: `(step > 0 && end >= start) || (step < 0 && end <= start)`. A positive step requires `end >= start`; a negative step requires `end <= start`. Note `step == 0` fails both clauses and is therefore rejected.
>
> Returns `true` if both hold. Does not compute or check the output length here (only sign/monotonicity consistency).

> [spec:et:def:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn]
> bool check_avg_pool2d_args( const Tensor& in, const IntArrayRef kernel_size, const IntArrayRef stride, const IntArrayRef padding, const bool ceil_mode, const bool count_include_pad, const std::optional<int64_t>& divisor_override, const T...

> [spec:et:sem:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn]
> Validates arguments for avg_pool2d. Returns `bool`; short-circuits false on the first failing check.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_is_default_or_channels_last_dim_order(in)`.
> 3. `tensor_is_default_or_channels_last_dim_order(out)`.
> 4. Rank/positivity, identical to `[spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]`: either (`in.dim()==3` with all of size(0..2) > 0) or (`in.dim()==4` with size(1..3) > 0).
> 5. `kernel_size_is_valid(kernel_size, kernel_ndim=2)` per `[spec:et:sem:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]`.
> 6. Stride validity: `stride_is_valid(kernel_size, kernel_ndim=2, allow_empty=true)` per `[spec:et:sem:kernel-ops-util.torch.executor.stride-is-valid-fn]`. NOTE: the source passes `kernel_size` (not `stride`) as the array argument here — a preexisting C++ quirk that must be reproduced for conformance: the stride array is effectively validated against the kernel_size values, meaning the actual `stride` argument is not shape/min-value checked in this function.
> 7. `padding_is_valid(padding, kernel_size, kernel_ndim=2, enforce_half_kernel=true)` per `[spec:et:sem:kernel-ops-util.torch.executor.padding-is-valid-fn]` (padding must additionally be at most half the kernel size).
> 8. If `divisor_override.has_value()`: `divisor_override.value() != 0` (must be non-zero).
>
> `ceil_mode` and `count_include_pad` are accepted but not validated. Returns `true` if all pass.

> [spec:et:def:kernel-ops-util.torch.executor.check-constant-pad-args-fn]
> bool check_constant_pad_args( const Tensor& in, IntArrayRef pad, const Scalar& value, Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-constant-pad-args-fn]
> Validates arguments for constant_pad_nd. `value` is ignored (cast to void). Returns `bool`.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensors_have_same_rank(in, out)` — input and output must have equal `dim()`.
> 3. `pad.size() % 2 == 0` — padding array length must be even (each dim contributes a low/high pair).
> 4. `pad.size() / 2 <= in.dim()` — no more pad pairs than input dimensions.
> 5. For each `i` in `[0, pad.size())`: `pad[i] >= 0` — all padding values non-negative.
>
> Returns `true` if all pass; on the first failing check logs an Error and returns `false`.

> [spec:et:def:kernel-ops-util.torch.executor.check-convolution-args-fn]
> bool check_convolution_args( const Tensor& in, const Tensor& weight, const std::optional<Tensor>& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, const Ten...

> [spec:et:sem:kernel-ops-util.torch.executor.check-convolution-args-fn]
> Validates arguments for convolution / conv_transpose. Returns `bool`; short-circuits false on first failure.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, weight, out)` — all three share one scalar type.
> 2. `in.dim() == 3 || in.dim() == 4 || in.dim() == 5` (1D/2D/3D convolution).
> 3. `tensor_is_rank(weight, in.dim())` — weight rank equals input rank.
> 4. `tensor_is_rank(out, in.dim())` — output rank equals input rank.
> 5. Dim-order constraints by rank:
>    - If `in.dim() == 5`: `tensor_is_default_dim_order` must hold for `in`, `weight`, and `out` (contiguous only), AND `!transposed` (transposed 3D conv is unsupported on portable; else fail).
>    - Else (3-D or 4-D): `tensor_is_default_or_channels_last_dim_order` must hold for `in`, `weight`, and `out`.
> 6. If `bias.has_value()`: `tensor_is_rank(bias, 1)`, and bias length check. WARNING — reproduce the exact C++ expression for conformance: `bias.value().size(0) == transposed ? groups * weight.size(1) : weight.size(0)`. Due to C operator precedence (`==` binds tighter than `?:`), this parses as `(bias.size(0) == transposed) ? (groups*weight.size(1)) : weight.size(0)`, and the ternary result is discarded — the check effectively passes iff `bias.size(0) == (transposed ? 1 : 0)`. This is a latent bug in the source; a faithful port must replicate this behavior (or a corrected port should be flagged as a deliberate divergence). The intended check is: bias length equals `groups*weight.size(1)` when transposed, else `weight.size(0)`.
> 7. Compute kernel dims: `fill_convolution_kernel_size(weight, kernel_size, &kernel_ndim)` per `[spec:et:sem:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]` (`kernel_ndim = weight.dim()-2`, `kernel_size[i] = weight.size(i+2)`).
> 8. `kernel_size_is_valid({kernel_size, kernel_ndim}, kernel_ndim)` per `[spec:et:sem:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]`.
> 9. `stride_is_valid(stride, kernel_ndim, allow_empty=false)`.
> 10. `padding_is_valid(padding, {kernel_size, kernel_ndim}, kernel_ndim)` (enforce_half_kernel defaults false).
> 11. `dilation_is_valid(dilation, kernel_ndim)`.
> 12. If `transposed`: `output_padding_is_valid(output_padding, stride, dilation, kernel_ndim)` per `[spec:et:sem:kernel-ops-util.torch.executor.output-padding-is-valid-fn]`.
> 13. `weight.size(0) >= groups`.
> 14. `weight.size(0) % groups == 0`.
> 15. Channel consistency:
>     - If not transposed: `in.size(1) == groups * weight.size(1)`.
>     - If transposed: `in.size(1) == weight.size(0)`.
>
> Returns `true` if all pass.

> [spec:et:def:kernel-ops-util.torch.executor.check-cumsum-args-fn]
> bool check_cumsum_args( const Tensor& in, int64_t dim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-cumsum-args-fn]
> Validates arguments for cumsum. Returns `bool`.
>
> Checks in order:
> 1. `dim_is_valid(dim, in.dim())` — `dim` must be a valid axis of `in`, i.e. in `[-in.dim(), in.dim())` (negative indexing allowed; for a 0-dim tensor `dim` in {-1, 0} is accepted).
> 2. If `dtype.has_value()`: `dtype.value() == out.scalar_type()` — the requested output dtype must match the actual output tensor's scalar type.
>
> Returns `true` if all pass; logs an Error and returns `false` on the first failure.

> [spec:et:def:kernel-ops-util.torch.executor.check-embedding-args-fn]
> bool check_embedding_args( const Tensor& weight, const Tensor& indices, const Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-embedding-args-fn]
> Validates arguments for embedding. Returns `bool`.
>
> Checks in order:
> 1. `weight.dim() == 2` — the embedding table is 2-D {num_embeddings, embedding_dim} (may be empty).
> 2. `out.dim() == indices.dim() + 1` — output rank is one more than the indices rank (out's leading dims mirror indices, its last dim is the embedding dim).
> 3. `tensors_have_same_dtype(weight, out)` — weight and out share scalar type.
>
> Returns `true` if all pass; on the first failure logs an Error and returns `false`.

> [spec:et:def:kernel-ops-util.torch.executor.check-masked-fill-args-fn]
> bool check_masked_fill_args( const Tensor& in, const Tensor& mask, const Scalar& value, Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.check-masked-fill-args-fn]
> Validates arguments for masked_fill. `value` is ignored (cast to void). Returns `bool`.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `mask.scalar_type() == ScalarType::Bool` — the mask must be a boolean tensor.
>
> Returns `true` if both pass; on failure logs an Error and returns `false`. Broadcasting compatibility between `in` and `mask` is not checked here (handled elsewhere in the op).

> [spec:et:def:kernel-ops-util.torch.executor.check-max-pool2d-with-indices-args-fn]
> bool check_max_pool2d_with_indices_args( const Tensor& in, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& out, const Tensor& indices)

> [spec:et:sem:kernel-ops-util.torch.executor.check-max-pool2d-with-indices-args-fn]
> Validates arguments for max_pool2d_with_indices. Returns `bool`; short-circuits false on first failure.
>
> Checks in order:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `indices.scalar_type() == ScalarType::Long` — the indices output must be int64.
> 3. `tensor_is_default_or_channels_last_dim_order(in)`.
> 4. `tensor_is_default_or_channels_last_dim_order(out)`.
> 5. Rank/positivity, identical to `[spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]`: either (`in.dim()==3` with size(0..2)>0) or (`in.dim()==4` with size(1..3)>0).
> 6. `kernel_size_is_valid(kernel_size, kernel_ndim=2)`.
> 7. Stride validity: `stride_is_valid(kernel_size, kernel_ndim=2, allow_empty=true)` — same preexisting C++ quirk as in `[spec:et:sem:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn]`: `kernel_size` (not `stride`) is passed as the array, so the real `stride` argument is not shape/min-value checked here.
> 8. `padding_is_valid(padding, kernel_size, kernel_ndim=2, enforce_half_kernel=true)`.
> 9. Dilation validity: `dilation_is_valid(kernel_size, kernel_ndim=2)` — again the source passes `kernel_size` rather than `dilation`; reproduce for conformance (the actual `dilation` argument is not validated here).
>
> `ceil_mode` is accepted but not validated. Returns `true` if all pass.

> [spec:et:def:kernel-ops-util.torch.executor.dilation-is-valid-fn]
> bool dilation_is_valid(IntArrayRef dilation, size_t kernel_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.dilation-is-valid-fn]
> Returns `param_array_is_valid("dilation", dilation, min_val=1, length=kernel_ndim, allow_empty=false)` per `[spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]`.
>
> Concretely: `dilation` must have size 1 or `kernel_ndim` (size 0 not allowed), and every element must be `>= 1`. Returns `bool`.

> [spec:et:def:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]
> void fill_convolution_kernel_size( const Tensor& weight, int64_t* kernel_size, size_t* kernel_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]
> Extracts the spatial kernel extents from a convolution weight tensor. File-local (anonymous namespace) helper.
>
> Steps:
> 1. `*kernel_ndim = weight.dim() - 2` — the number of spatial kernel dims (weight layout is {out_channels, in_channels/groups, *spatial}).
> 2. For each `i` in `[0, *kernel_ndim)`: `kernel_size[i] = weight.size(i + 2)` — copy the trailing spatial sizes into the caller-provided `kernel_size` buffer.
>
> Returns void. Assumes `kernel_size` has capacity `>= *kernel_ndim` (callers use a length-3 stack buffer for up to 3D conv). No validation.

> [spec:et:def:kernel-ops-util.torch.executor.get-adaptive-avg-pool2d-out-target-size-fn]
> void get_adaptive_avg_pool2d_out_target_size( const Tensor& in, const IntArrayRef output_size, executorch::aten::SizesType* const out_sizes, size_t* const out_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.get-adaptive-avg-pool2d-out-target-size-fn]
> Computes the target output shape for adaptive_avg_pool2d, writing into `out_sizes` and setting `*out_ndim`.
>
> Steps:
> 1. `*out_ndim = in.dim()` (3 or 4).
> 2. Copy leading dims: if `in.dim() == 4`, `out_sizes[0] = in.size(0)` (batch) and `out_sizes[1] = in.size(1)` (channels); else (3-D) `out_sizes[0] = in.size(0)` (channels).
> 3. Set the two trailing spatial dims from the requested `output_size`: `out_sizes[*out_ndim - 2] = output_size[0]` (H), `out_sizes[*out_ndim - 1] = output_size[1]` (W).
>
> Returns void. Assumes args already validated by `[spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]`.

> [spec:et:def:kernel-ops-util.torch.executor.get-avg-pool2d-out-target-size-fn]
> void get_avg_pool2d_out_target_size( const Tensor& in, const IntArrayRef kernel_size, const IntArrayRef stride, const IntArrayRef padding, const bool ceil_mode, executorch::aten::SizesType* const out_sizes, size_t* const out_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.get-avg-pool2d-out-target-size-fn]
> Computes the target output shape for avg_pool2d, writing into `out_sizes` and setting `*out_ndim`.
>
> Steps:
> 1. `*out_ndim = in.dim()` (3 or 4).
> 2. Copy leading dims: if `in.dim() == 4`, `out_sizes[0] = in.size(0)`, `out_sizes[1] = in.size(1)`; else `out_sizes[0] = in.size(0)`.
> 3. Fill the trailing two spatial dims via `calculate_kernel_output_sizes(in, kernel_ndim=2, kernel_size, stride, padding, dilation={}, out_sizes, ceil_mode)` per `[spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]`. Dilation is passed empty (so dilation defaults to 1), transposed/output_padding default (false/{}).
>
> Returns void. `count_include_pad`/`divisor_override` do not affect output shape and are not parameters here.

> [spec:et:def:kernel-ops-util.torch.executor.get-convolution-out-target-size-fn]
> void get_convolution_out_target_size( const Tensor& in, const Tensor& weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, executorch::aten::SizesType* out_si...

> [spec:et:sem:kernel-ops-util.torch.executor.get-convolution-out-target-size-fn]
> Computes the target output shape for convolution / conv_transpose, writing into `out_sizes` and setting `*out_ndim`.
>
> Steps:
> 1. `*out_ndim = in.dim()`.
> 2. Batch dim: `out_sizes[0] = in.size(0)`.
> 3. Channel dim (`out_sizes[1]`):
>    - If not transposed: `in.size(1) == 0 ? 0 : weight.size(0)`.
>    - If transposed: `in.size(1) == 0 ? 0 : groups * weight.size(1)`.
>    (When the input channel dim is 0, the output channel dim is 0; otherwise it's the standard conv output-channel count.)
> 4. Compute spatial kernel dims via `fill_convolution_kernel_size(weight, kernel_size, &kernel_ndim)` (`[spec:et:sem:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]`).
> 5. Fill the trailing `kernel_ndim` spatial dims via `calculate_kernel_output_sizes(in, kernel_ndim, {kernel_size, kernel_ndim}, stride, padding, dilation, out_sizes, ceil_mode=false, transposed, output_padding)` per `[spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]`.
>
> Returns void.

> [spec:et:def:kernel-ops-util.torch.executor.get-max-pool2d-with-indices-out-target-size-fn]
> void get_max_pool2d_with_indices_out_target_size( const Tensor& in, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.get-max-pool2d-with-indices-out-target-size-fn]
> Computes the target output shape for max_pool2d_with_indices, writing into `out_sizes` and setting `*out_ndim`.
>
> Steps:
> 1. `*out_ndim = in.dim()` (3 or 4).
> 2. Copy leading dims: if `in.dim() == 4`, `out_sizes[0] = in.size(0)`, `out_sizes[1] = in.size(1)`; else `out_sizes[0] = in.size(0)`.
> 3. Fill the trailing two spatial dims via `calculate_kernel_output_sizes(in, kernel_ndim=2, kernel_size, stride, padding, dilation, out_sizes, ceil_mode)` per `[spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]`. Unlike avg_pool2d, the real `dilation` is passed through.
>
> Returns void. The same shape applies to both the values output and the indices output.

> [spec:et:def:kernel-ops-util.torch.executor.get-unsqueezed-dim-order-fn]
> void get_unsqueezed_dim_order( const Tensor& t, executorch::aten::DimOrderType unsqueeze_dim, executorch::aten::DimOrderType* dim_order_arr)

> [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-dim-order-fn]
> Computes the dim_order of a tensor after inserting a new size-1 dimension at logical position `unsqueeze_dim`, writing `t.dim() + 1` entries into `dim_order_arr`. A dim_order is a permutation of dimension indices describing physical memory ordering.
>
> Algorithm: iterate `i` over `[0, t.dim())` reading `dim = t.dim_order()[i]`, maintaining an `offset` that is 0 before the inserted axis is encountered and 1 after.
> - If `dim == unsqueeze_dim`: this is where the new axis is inserted physically adjacent to it — write `dim_order_arr[i] = dim` and `dim_order_arr[i + 1] = dim + 1`, and set `offset = 1`.
> - Else: write `dim_order_arr[i + offset] = (dim > unsqueeze_dim) ? dim + 1 : dim`. That is, any existing dimension index greater than `unsqueeze_dim` is shifted up by one (to make room for the inserted logical axis), and once the inserted axis has been placed, subsequent writes are offset by one slot.
>
> Returns void. Assumes exactly one entry of `t.dim_order()` equals `unsqueeze_dim` (true since dim_order is a permutation), so `offset` becomes 1 exactly once. `dim_order_arr` must have capacity `t.dim() + 1`.

> [spec:et:def:kernel-ops-util.torch.executor.get-unsqueezed-sizes-fn]
> void get_unsqueezed_sizes( const Tensor& t, int64_t unsqueeze_dim, executorch::aten::SizesType* sizes_arr, size_t& ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-sizes-fn]
> Computes the sizes array of a tensor after inserting a size-1 dimension at logical position `unsqueeze_dim`, and sets `ndim`.
>
> Steps:
> 1. `ndim = t.dim() + 1`.
> 2. For each `d` in `[0, unsqueeze_dim)`: `sizes_arr[d] = t.size(d)` (copy leading dims unchanged).
> 3. `sizes_arr[unsqueeze_dim] = 1` (the inserted axis).
> 4. For each `d` in `[unsqueeze_dim + 1, ndim)`: `sizes_arr[d] = t.size(d - 1)` (copy trailing dims, shifted up by one).
>
> Returns void (`ndim` returned via reference). `sizes_arr` must have capacity `t.dim() + 1`. `unsqueeze_dim` must be in `[0, t.dim()]`.

> [spec:et:def:kernel-ops-util.torch.executor.int-array-all-ge-fn]
> bool int_array_all_ge(IntArrayRef array, int64_t val)

> [spec:et:sem:kernel-ops-util.torch.executor.int-array-all-ge-fn]
> Returns `true` iff every element of `array` is `>= val`. Iterates `i` over `[0, array.size())`; on the first element with `array[i] < val`, logs an Error (reporting the index, `val`, and the offending value) and returns `false`. If no element violates the bound (including the empty-array case), returns `true`.

> [spec:et:def:kernel-ops-util.torch.executor.kernel-output-size-helper-fn]
> int64_t _kernel_output_size_helper( size_t inputSize, int64_t kernelSize, int64_t pad, int64_t stride, int64_t dilation, bool ceil_mode, bool transposed, int64_t output_padding)

> [spec:et:sem:kernel-ops-util.torch.executor.kernel-output-size-helper-fn]
> Computes the output extent along one spatial axis of a pooling/convolution op. Returns `int64_t`. File-local helper. `inputSize` is `size_t`; the rest are `int64_t`/`bool`.
>
> Transposed branch (`transposed == true`): return
> `(inputSize - 1) * stride - 2 * pad + dilation * (kernelSize - 1) + output_padding + 1`.
>
> Non-transposed branch:
> 1. `numerator = inputSize + 2*pad - dilation*(kernelSize - 1) - 1 + (ceil_mode ? stride - 1 : 0)`.
> 2. `outputSize = numerator / stride + 1` (integer division, truncating toward zero; matches floor for the non-negative values expected here).
> 3. If `ceil_mode` and the last pooling window would start in the padding beyond the input: specifically if `(outputSize - 1) * stride >= inputSize + pad`, decrement `outputSize` by 1 (ensures the final window starts inside the image, per PyTorch's ceil-mode convention).
> 4. Return `outputSize`.
>
> No clamping to non-negative is done here; invalid parameter combinations may yield a non-positive result, which callers detect via `[spec:et:sem:kernel-ops-util.torch.executor.output-size-is-valid-fn]`. `output_padding` is ignored in the non-transposed branch.

> [spec:et:def:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn]
> void kernel_reduction_then_map_2d( const ReduceOp& reduce_fn, const MapOp& map_fn, const bool include_pad, const CTYPE* const in_ptr, const executorch::aten::ArrayRef<executorch::aten::SizesType> in_sizes, const executorch::aten::ArrayRe...

> [spec:et:sem:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn]
> Header-only template `kernel_reduction_then_map_2d<CTYPE, ReduceOp, MapOp>`. For one fixed batch and channel, sweeps a 2D kernel window over the full (H, W) output plane, reducing each window then mapping the accumulator, writing values into `out_ptr` and (optionally) window argmax-style indices into `indices_ptr`. `reduce_fn` has signature `(CTYPE in_val, int64_t in_idx, CTYPE accum, int64_t accum_idx) -> tuple<CTYPE,int64_t>`; `map_fn` has signature `(int64_t count, CTYPE accum) -> CTYPE`.
>
> Setup:
> 1. `in_dim = in_sizes.size()`, `out_dim = out_sizes.size()`.
> 2. `out_H = out_sizes[in_dim-2]`, `in_H = in_sizes[in_dim-2]`, `out_W = out_sizes[in_dim-1]`, `in_W = in_sizes[in_dim-1]`.
> 3. Allocate coordinate buffers `in_coord[kTensorDimensionLimit]`, `out_coord[kTensorDimensionLimit]`. If `in_dim == 4`, set `in_coord[0] = out_coord[0] = batch`. Set the channel axis `in_coord[in_dim-3] = out_coord[in_dim-3] = out_c` (index 1 for 4-D, index 0 for 3-D).
> 4. Extract kernel params with broadcast/defaulting via `[spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]`: `k_H=val_at(kernel_size,0)`, `k_W=val_at(kernel_size,1)`, `s_H=val_at(stride,0,default=k_H)`, `s_W=val_at(stride,1,default=k_W)`, `p_H=val_at(padding,0,default=0)`, `p_W=val_at(padding,1,default=0)`, `d_H=val_at(dilation,0,default=1)`, `d_W=val_at(dilation,1,default=1)`.
>
> For each output row `out_y` in `[0, out_H)` (set `out_coord[in_dim-2]=out_y`) and each output col `out_x` in `[0, out_W)` (set `out_coord[in_dim-1]=out_x`):
> 1. Initialize `accum_initialized=false`, `accum=0`, `accum_idx=0`, `count=0`.
> 2. Compute the (padding-inclusive) window bounds: `ih0 = out_y*s_H - p_H`, `iw0 = out_x*s_W - p_W`, `ih1 = min(ih0 + k_H, in_H + p_H)`, `iw1 = min(iw0 + k_W, in_W + p_W)`. `pool_size = (ih1-ih0)*(iw1-iw0)` (the full window size including padding, before clamping — used as `count` when `include_pad`).
> 3. Clamp the actual (in-bounds) window: `ih0=max(ih0,0)`, `iw0=max(iw0,0)`, `ih1=min(ih1,in_H)`, `iw1=min(iw1,in_W)`.
> 4. If `ih0 >= ih1 || iw0 >= iw1` (empty window), `continue` to the next output position — leaving `out_ptr`/`indices_ptr` at that position UNWRITTEN (note: this branch is not reachable for validated pool args, since kernel windows always overlap the input).
> 5. `count = include_pad ? pool_size : (ih1-ih0)*(iw1-iw0)` — divisor/count semantics (avg pool divides by count in `map_fn`).
> 6. Inner reduction over the full kernel extent (NOT the clamped bounds): for `w_y` in `[0, k_H)` and `w_x` in `[0, k_W)`:
>    - `in_y = s_H*out_y + d_H*w_y - p_H`, set `in_coord[in_dim-2]=in_y`; `in_x = s_W*out_x + d_W*w_x - p_W`, set `in_coord[in_dim-1]=in_x`.
>    - `xy_in_bound = (in_x in [0,in_W)) && (in_y in [0,in_H))`.
>    - `in_val = 0`; if `xy_in_bound`, `in_val = in_ptr[calculate_linear_index(in_coord, in_strides.data(), in_dim)]` (linear offset from coords and strides).
>    - Linear window index: `idx = in_y*in_W + in_x`; but if `include_pad`, `idx = in_y + p_H*(in_W + 2*p_W) + (in_x + p_W)` (padded-plane index).
>    - Only when `xy_in_bound`: if not yet initialized, set `accum=in_val`, `accum_idx=idx`, `accum_initialized=true`; otherwise call `reduce_fn(in_val, idx, accum, accum_idx)` and update `accum`, `accum_idx` from its returned tuple. Out-of-bound (padding) positions are skipped in the reduction regardless of `include_pad` — `include_pad` only affects `count` and the `idx` formula, not which elements are reduced.
> 7. After the window: `out_idx = calculate_linear_index(out_coord, out_strides.data(), out_dim)`; `out_ptr[out_idx] = map_fn(count, accum)`; if `indices_ptr != nullptr`, `indices_ptr[out_idx] = accum_idx`.
>
> Returns void. `accum` accumulation type is CTYPE (no separate higher-precision accumulator). Dilation is honored in `in_y`/`in_x` but `k_H`/`k_W` iterate the undilated kernel extent (each tap offset by `d*w`).

> [spec:et:def:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]
> bool kernel_size_is_valid(IntArrayRef kernel_size, size_t kernel_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]
> Returns `param_array_is_valid("kernel_size", kernel_size, min_val=1, length=kernel_ndim, allow_empty=false)` per `[spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]`.
>
> Concretely: `kernel_size` must have size 1 or `kernel_ndim` (size 0 not allowed), and every element must be `>= 1`. Returns `bool`.

> [spec:et:def:kernel-ops-util.torch.executor.output-padding-is-valid-fn]
> bool output_padding_is_valid( IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, size_t kernel_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.output-padding-is-valid-fn]
> Validates `output_padding` for transposed convolution. Returns `bool`.
>
> 1. First: `param_array_is_valid("output_padding", output_padding, min_val=0, length=kernel_ndim, allow_empty=false)` per `[spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]` — size must be 1 or `kernel_ndim`, all elements `>= 0`.
> 2. Then, for each `i` in `[0, kernel_ndim)`: with `op_i=val_at(output_padding,i)`, `s_i=val_at(stride,i)`, `d_i=val_at(dilation,i)` (all default 1 per `[spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]`), require `op_i < s_i || op_i < d_i` — the output padding along each axis must be strictly smaller than either the stride or the dilation for that axis.
>
> Returns `true` if all checks pass; logs an Error and returns `false` on the first failure.

> [spec:et:def:kernel-ops-util.torch.executor.output-size-is-valid-fn]
> bool output_size_is_valid( executorch::aten::ArrayRef<executorch::aten::SizesType> output_size, size_t kernel_ndim)

> [spec:et:sem:kernel-ops-util.torch.executor.output-size-is-valid-fn]
> Validates a computed `output_size` (ArrayRef of SizesType). Returns `bool`. `out_dim = output_size.size()`. The last `kernel_ndim` dims are the spatial dims; the leading `out_dim - kernel_ndim` are batch/channel dims.
>
> 1. `valid = true`.
> 2. For each `i` in `[0, out_dim - kernel_ndim)` (leading dims): if `output_size[i] < 0`, set `valid = false`. Leading dims may be zero but not negative.
> 3. For each `i` in `[out_dim - kernel_ndim, out_dim)` (spatial dims): if `output_size[i] <= 0`, set `valid = false`. Spatial dims must be strictly positive.
> 4. If `!valid`, log an Error header and log each dimension's size for diagnostics.
> 5. Return `valid`.
>
> Unlike the `check_*`/`*_is_valid` functions above, this does not short-circuit — it evaluates all dims. `SizesType` is signed here, so the `< 0`/`<= 0` comparisons detect the negative sizes that `_kernel_output_size_helper` can produce from bad parameters.

> [spec:et:def:kernel-ops-util.torch.executor.padding-is-valid-fn]
> bool padding_is_valid( IntArrayRef padding, IntArrayRef kernel_size, size_t kernel_ndim, bool enforce_half_kernel)

> [spec:et:sem:kernel-ops-util.torch.executor.padding-is-valid-fn]
> Validates `padding`. Returns `bool`. `enforce_half_kernel` defaults to `false`.
>
> 1. First: `param_array_is_valid("padding", padding, min_val=0, length=kernel_ndim, allow_empty=false)` per `[spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]` — size must be 1 or `kernel_ndim`, all elements `>= 0`. If this returns false, return `false` immediately.
> 2. If `enforce_half_kernel`: for each `i` in `[0, padding.size())`, require `padding[i] <= val_at(kernel_size, i) / 2` (integer division). If any `padding[i] > kernel_size_i / 2`, log an Error and return `false`. `val_at` broadcasts a length-1 `kernel_size` per `[spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]`.
>
> Returns `true` if all checks pass.

> [spec:et:def:kernel-ops-util.torch.executor.param-array-is-valid-fn]
> bool param_array_is_valid( const char* name, IntArrayRef array, int64_t min_val, size_t length, bool allow_empty)

> [spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]
> File-local (anonymous namespace) shared validator for kernel-parameter arrays (kernel_size, stride, padding, dilation, output_padding). Returns `bool`. `name` is only used in log messages.
>
> Steps:
> 1. `size = array.size()`.
> 2. Size check:
>    - If `allow_empty`: require `size == 0 || size == 1 || size == length`.
>    - Else: require `size == 1 || size == length`.
>    On failure logs an Error (naming `name`, expected sizes, and actual) and returns `false`. Size 1 means the single value broadcasts across all `length` axes (see `[spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]`).
> 3. `int_array_all_ge(array, min_val)` per `[spec:et:sem:kernel-ops-util.torch.executor.int-array-all-ge-fn]` — every element must be `>= min_val`; on failure returns `false`.
> 4. Returns `true`.

> [spec:et:def:kernel-ops-util.torch.executor.resize-constant-pad-output-fn]
> Error resize_constant_pad_output( const Tensor& in, IntArrayRef pad, Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.resize-constant-pad-output-fn]
> Resizes `out` to the shape produced by constant_pad_nd. Returns `Error`.
>
> The `pad` array is interpreted PyTorch-style: pairs `(pad[0],pad[1])` apply to the LAST input dim, `(pad[2],pad[3])` to the second-to-last, etc. So `pad` covers the trailing `pad.size()/2` dims.
>
> Steps (writing into a local `expected_output_size[kTensorDimensionLimit]`):
> 1. `pad_i = in.dim() - 1` (a countdown pairing input dims to pad pairs from the last dim).
> 2. For each `i` in `[0, in.dim())`:
>    - `expected_output_size[i] = in.size(i)`.
>    - If `pad_i >= 0 && pad_i < pad.size()/2`: add `pad[2*pad_i] + pad[2*pad_i + 1]` to `expected_output_size[i]`.
>    - `--pad_i`.
>    Because `pad_i` starts at the last dim index and decrements while `i` increases, dim `i` receives the pad pair at position `pad_i = in.dim()-1-i`, which correctly maps the first pad pair to the last dim.
> 3. Build `ArrayRef output_size{expected_output_size, in.dim()}` and return `resize_tensor(out, output_size)` (resizes `out` in place, returning `Error::Ok` or a resize error such as when `out` is not resizable to that shape).

> [spec:et:def:kernel-ops-util.torch.executor.resize-embedding-output-fn]
> Error resize_embedding_output( const Tensor& weight, const Tensor& indices, const Tensor& out)

> [spec:et:sem:kernel-ops-util.torch.executor.resize-embedding-output-fn]
> Resizes `out` to the embedding output shape. Returns `Error`.
>
> Steps (writing into a local `expected_output_size[kTensorDimensionLimit]`):
> 1. For each `i` in `[0, indices.dim())`: `expected_output_size[i] = indices.size(i)` — leading dims mirror the indices tensor.
> 2. `embedding_dim = weight.size(1)`; `expected_output_size[out.dim() - 1] = embedding_dim` — the last output dim is the embedding width.
> 3. Build `ArrayRef output_size{expected_output_size, out.dim()}` and return `resize_tensor(out, output_size)`.
>
> Assumes `out.dim() == indices.dim() + 1` (validated by `[spec:et:sem:kernel-ops-util.torch.executor.check-embedding-args-fn]`), so the write at index `out.dim()-1` is exactly the slot after the copied indices dims.

> [spec:et:def:kernel-ops-util.torch.executor.stride-is-valid-fn]
> bool stride_is_valid(IntArrayRef stride, size_t kernel_ndim, bool allow_empty)

> [spec:et:sem:kernel-ops-util.torch.executor.stride-is-valid-fn]
> Returns `param_array_is_valid("stride", stride, min_val=1, length=kernel_ndim, allow_empty)` per `[spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]`.
>
> Concretely: `stride` must have size 1 or `kernel_ndim`, or (when `allow_empty` is true) size 0, and every element must be `>= 1`. Returns `bool`. Pooling callers pass `allow_empty=true` (empty stride means stride defaults to kernel_size); convolution passes `allow_empty=false`.

> [spec:et:def:kernel-ops-util.torch.executor.val-at-fn]
> inline int64_t val_at(IntArrayRef array, size_t i, int64_t default_value = 1)

> [spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]
> Inline accessor that extracts a value at index `i` from an int array, with broadcasting and defaulting. `default_value` defaults to 1. Returns `int64_t`.
>
> - If `array.size() == 1`: return `array[0]` regardless of `i` (a single value broadcasts across all positions).
> - Else if `array.size() > 1`: return `array[i]` (caller must ensure `i < size`).
> - Else (`array.size() == 0`): return `default_value`.
>
> This is the mechanism by which length-1 kernel-parameter arrays broadcast across all spatial axes and empty arrays fall back to per-call defaults (e.g. stride defaulting to kernel size, dilation to 1, padding to 0).

