# kernels/portable/cpu/util/index_util.cpp

> [spec:et:def:index-util.torch.executor.check-gather-args-fn]
> bool check_gather_args( const Tensor& in, int64_t dim, const Tensor& index, bool sparse_grad, Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-gather-args-fn]
> Validates arguments for `gather`. Returns `true` if all checks pass,
> `false` otherwise. Every check is a boolean predicate: `ET_LOG_AND_RETURN_IF_FALSE`
> logs and returns `false` when its predicate is false; `ET_CHECK_OR_RETURN_FALSE`
> logs its formatted message and returns `false` when its condition is false.
> The first failing check short-circuits. The `sparse_grad` parameter is
> accepted but not used by this check.
>
> Checks, in order:
> 1. `tensors_have_same_dtype(in, out)` — input and output must share dtype.
> 2. `tensor_has_dim(in, dim)` — `dim` must be a valid axis of `in`
>    (accepts negatives in range `[-in.dim(), in.dim())`).
> 3. `index.scalar_type() == ScalarType::Long` — the index tensor must be
>    int64.
> 4. If `index.numel() != 0`: `nonzero_dim(in) == nonzero_dim(index)` — self
>    and index must have the same dimensionality (where `nonzero_dim` treats a
>    0-dim tensor as dimensionality 1, so a 0-dim and a 1-dim tensor are
>    treated as compatible).
> 5. Normalize `dim`: if `dim < 0`, set `dim += nonzero_dim(in)`.
> 6. For every axis `d` in `[0, nonzero_dim(in))` with `d != dim`:
>    `nonempty_size(index, d) <= nonempty_size(in, d)` — index must not be
>    larger than input along non-gather axes (`nonempty_size` reports 1 for a
>    0-dim tensor).
> 7. For every element `i` in `[0, index.numel())` read from `index` as
>    `long`/int64: `0 <= index_data[i] < nonempty_size(in, dim)` — each index
>    value must be in bounds along the (normalized) gather axis.
>
> If all pass, return `true`. Performs no resizing or mutation; read-only
> validation.

> [spec:et:def:index-util.torch.executor.check-index-select-args-fn]
> bool check_index_select_args( const Tensor& in, int64_t dim, const Tensor& index, Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-index-select-args-fn]
> Validates arguments for `index_select`. Returns `true` if all checks pass,
> `false` on the first failure (same `ET_LOG_AND_RETURN_IF_FALSE` /
> `ET_CHECK_OR_RETURN_FALSE` semantics as
> `[spec:et:sem:index-util.torch.executor.check-gather-args-fn]`).
>
> Checks, in order:
> 1. `tensor_has_dim(in, dim)` — `dim` must be a valid axis of `in` (negatives
>    allowed in `[-in.dim(), in.dim())`).
> 2. Normalize `dim`: `dim = dim < 0 ? dim + nonzero_dim(in) : dim`.
> 3. `nonempty_size(in, dim) > 0` — the indexing axis must have positive size.
> 4. `tensors_have_same_dtype(in, out)` — input and output share dtype.
> 5. `index.scalar_type() == Long || index.scalar_type() == Int` — index must
>    be int64 or int32.
> 6. `tensor_has_rank_smaller_or_equal_to(index, 1)` — index must be 0-D or
>    1-D.
> 7. If `index.dim() > 0 && in.dim() == 0`: `index.numel() == 1` — indexing a
>    scalar input requires exactly one index value.
> 8. Bounds check every index element against `nonempty_size(in, dim)`,
>    dispatched by the index dtype:
>    - if index is `Long`: read as `int64_t`, require
>      `0 <= index_ptr[i] < nonempty_size(in, dim)` for each `i` in
>      `[0, index.numel())`.
>    - if index is `Int`: read as `int32_t`, same bounds requirement.
>
> If all pass, return `true`. Read-only validation; no resizing/mutation.
> Unlike gather, negative index values are rejected (indices are not
> wrapped).

> [spec:et:def:index-util.torch.executor.check-nonzero-args-fn]
> bool check_nonzero_args(const Tensor& in, const Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-nonzero-args-fn]
> Validates arguments for `nonzero`. The `in` parameter is accepted but
> deliberately unused (`(void)in;`) — no constraint is placed on the input.
> Returns `true` if all checks pass, `false` on the first failure.
>
> Checks, in order:
> 1. `out.scalar_type() == ScalarType::Long` — the output must be an int64
>    tensor (it holds coordinate indices).
> 2. `out.dim() == 2` — the output must be exactly 2-dimensional (shape
>    `[num_nonzero, in.dim()]`).
>
> If both pass, return `true`. Read-only validation; no resizing/mutation.

> [spec:et:def:index-util.torch.executor.check-scatter-add-args-fn]
> bool check_scatter_add_args( const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-scatter-add-args-fn]
> Validates arguments for `scatter_add` (and, via delegation,
> `scatter.src`). Returns `true` if all checks pass, `false` on the first
> failure.
>
> Checks, in order:
> 1. `tensors_have_same_dtype(self, out)` — self and output share dtype.
> 2. `tensors_have_same_dtype(self, src)` — self and src share dtype.
> 3. `index.scalar_type() == ScalarType::Long` — index must be int64.
> 4. `tensor_has_dim(self, dim)` — `dim` must be a valid axis of `self`.
> 5. Early success: if `index.numel() == 0`, return `true` immediately (no
>    further checks; an empty index scatters nothing).
> 6. `nonzero_dim(self) == nonzero_dim(src) && nonzero_dim(self) ==
>    nonzero_dim(index)` — self, src, and index must all have the same
>    dimensionality (0-dim counted as 1 by `nonzero_dim`).
> 7. Normalize `dim`: if `dim < 0`, set `dim += nonzero_dim(self)`.
> 8. For every axis `d` in `[0, nonzero_dim(self))`:
>    - `nonempty_size(index, d) <= nonempty_size(src, d)` — index must not be
>      larger than src along any axis.
>    - additionally, if `d != dim`:
>      `nonempty_size(index, d) <= nonempty_size(self, d)` — index must not be
>      larger than self along non-scatter axes.
> 9. For every element `i` in `[0, index.numel())` read from `index` as
>    `long`/int64: `0 <= index_data[i] < nonempty_size(self, dim)` — each
>    index value must be in bounds along the (normalized) scatter axis.
>
> If all pass, return `true`. Read-only validation; no resizing/mutation.

> [spec:et:def:index-util.torch.executor.check-scatter-src-args-fn]
> bool check_scatter_src_args( const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-scatter-src-args-fn]
> Validates arguments for `scatter.src`. This is a thin delegation: it calls
> and returns `check_scatter_add_args(self, dim, index, src, out)` unchanged,
> so its behavior is exactly that of
> `[spec:et:sem:index-util.torch.executor.check-scatter-add-args-fn]`.
> Returns `true`/`false` accordingly.

> [spec:et:def:index-util.torch.executor.check-scatter-value-args-fn]
> bool check_scatter_value_args( const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value, Tensor& out)

> [spec:et:sem:index-util.torch.executor.check-scatter-value-args-fn]
> Validates arguments for `scatter.value` (scattering a scalar `value` rather
> than a source tensor). It delegates to and returns
> `check_gather_args(self, dim, index, /*sparse_grad=*/false, out)`, so its
> behavior is exactly that of
> `[spec:et:sem:index-util.torch.executor.check-gather-args-fn]` with
> `in = self`. The `value` scalar itself is not inspected (any dtype/value is
> accepted; validation only covers dtype-equality of self/out, `dim` validity,
> index dtype/shape, and index bounds). Returns `true`/`false` accordingly.

> [spec:et:def:index-util.torch.executor.check-select-scatter-args-fn]
> bool check_select_scatter_args( const Tensor& in, const Tensor& src, int64_t dim, int64_t index, Tensor& output)

> [spec:et:sem:index-util.torch.executor.check-select-scatter-args-fn]
> Validates arguments for `select_scatter`, which writes `src` into the slice
> of `in` selected at position `index` along axis `dim`, producing `output`
> with the same shape as `in`. Returns `true` if all checks pass, `false` on
> the first failure. Assumptions: output shape equals input shape; src shape
> equals the selected input slice; `dim`/`index` are valid for `in`.
>
> Checks, in order:
> 1. `tensors_have_same_dtype(in, output)` — input and output share dtype.
> 2. `dim_is_valid(dim, in.dim())` — `dim` must be a valid axis of `in`
>    (negatives allowed in `[-in.dim(), in.dim())`). Note `dim` is NOT
>    normalized to non-negative here; subsequent uses (`in.size(dim)`,
>    the `d < dim` comparison) assume the caller passed a value the tensor
>    accessor handles, and negative `dim` compares as-is against `d`.
> 3. `0 <= index < in.size(dim)` — the selected index must be in bounds along
>    `dim`.
> 4. `in.dim() == src.dim() + 1` — src has exactly one fewer dimension than
>    in (the selected axis is removed).
> 5. Shape match of the remaining axes: for each `d` in `[0, in.dim() - 1)`:
>    - if `d < dim`: `tensors_have_same_size_at_dims(in, d, src, d)` — src
>      axis `d` matches in axis `d`.
>    - else (`d >= dim`): `tensors_have_same_size_at_dims(in, d + 1, src, d)`
>      — src axis `d` matches in axis `d + 1` (skipping the selected axis).
>
> If all pass, return `true`. Read-only validation; no resizing/mutation.

> [spec:et:def:index-util.torch.executor.get-index-select-out-target-size-fn]
> void get_index_select_out_target_size( const Tensor& in, int64_t dim, const Tensor& index, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:index-util.torch.executor.get-index-select-out-target-size-fn]
> Computes the output shape for `index_select` and writes it into the
> caller-provided `out_sizes` buffer and `out_ndim`. Returns nothing.
>
> Algorithm:
> 1. Set `*out_ndim = in.dim()` — the output has the same rank as the input.
> 2. For each axis `i` in `[0, in.dim())`:
>    - if `i == dim`: `out_sizes[i] = index.numel()` — the selected axis takes
>      the number of index elements.
>    - else: `out_sizes[i] = in.size(i)` — all other axes are copied from the
>      input.
>
> `dim` is expected to be the already-normalized (non-negative) axis, matching
> the value validated/normalized by
> `[spec:et:sem:index-util.torch.executor.check-index-select-args-fn]`. The
> caller allocates `out_sizes` with capacity for at least `in.dim()` elements.

