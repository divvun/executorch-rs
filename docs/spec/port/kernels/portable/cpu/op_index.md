# kernels/portable/cpu/op_index.cpp

> [spec:et:def:op-index.torch.executor.native.check-fast-path-args-fn]
> bool check_fast_path_args( KernelRuntimeContext& ctx, const Tensor& in, TensorOptList indices, size_t dim, Tensor& out)

> [spec:et:sem:op-index.torch.executor.native.check-fast-path-args-fn]
> Validates arguments for the single-index fast path (given the already-known
> indexed axis `dim`). Returns true if valid, false otherwise (logging on
> failure). Steps:
> 1. `tensors_have_same_dtype(in, out)` must hold (fast path is a pure copy, no
>    dtype conversion). Return false if not.
> 2. `indices.size() <= in.dim()` ("Indexing too many dimensions"); else false.
> 3. Let `index = indices[dim].value()`. Dispatch over its dtype (Long or Int
>    only) and validate every index value: for each element, apply Python-style
>    wrap-around (`v < 0 ? v + in.size(dim) : v`) and require the wrapped value to
>    be in `[0, in.size(dim))`. If any is out of range, log an "Index out of
>    range" error and mark invalid.
> 4. Require all index values valid ("Some index values are not within bounds of
>    input tensor at indexed dim"); else false.
> 5. Return true.

> [spec:et:def:op-index.torch.executor.native.check-fast-path-conditions-fn]
> bool check_fast_path_conditions( ET_UNUSED const Tensor& in, TensorOptList indices, size_t* dim)

> [spec:et:sem:op-index.torch.executor.native.check-fast-path-conditions-fn]
> Decides whether the fast path applies and, if so, reports the indexed axis via
> `*dim`. Returns true if all fast-path conditions hold. `in` is unused. Steps:
> 1. Iterate `i` over `indices` (in order). For each non-null entry:
>    - Set `*dim = i`.
>    - If a non-null index was already seen (`found_index`), return false (fast
>      path supports exactly one non-null index tensor).
>    - Mark `found_index = true`, let `index = indices[i].value()`.
>    - If `index.scalar_type()` is neither Long nor Int, return false.
>    - If `index.dim() != 1` (must be a 1-D index tensor), return false.
> 2. If no non-null index was found, return false (fast path needs at least one).
> 3. Otherwise return true, with `*dim` set to the position of the single non-null
>    index.

> [spec:et:def:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn]
> void get_fast_path_index_out_target_size( const Tensor& in, TensorOptList indices, size_t dim, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn]
> Computes the output shape for the single-index fast path into `out_sizes` and
> `*out_ndim`. Sets `*out_ndim = in.dim()`. For each axis `d` in `[0, in.dim())`:
> if `d != dim`, `out_sizes[d] = in.size(d)`; if `d == dim`,
> `out_sizes[d] = indices[dim].value().numel()` (the indexed axis is replaced by
> the number of selected indices). The output has the same rank as `in`, with
> only the `dim` axis resized.

> [spec:et:def:op-index.torch.executor.native.fast-path-fn]
> Tensor& fast_path( KernelRuntimeContext& ctx, const Tensor& in, TensorOptList indices, size_t dim, Tensor& out)

> [spec:et:sem:op-index.torch.executor.native.fast-path-fn]
> Executes the single-index gather fast path (one non-null 1-D Long/Int index at
> axis `dim`). Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK `check_fast_path_args(ctx, in, indices, dim, out)` per
>    `[spec:et:sem:op-index.torch.executor.native.check-fast-path-args-fn]`; on
>    failure Error::InvalidArgument, return `out`.
> 2. Let `index = indices[dim].value()`, `index_type = index.scalar_type()`.
> 3. Compute the target output shape into `expected_size`/`expected_ndim` via
>    `get_fast_path_index_out_target_size` per
>    `[spec:et:sem:op-index.torch.executor.native.get-fast-path-index-out-target-size-fn]`,
>    then resize `out` to it (ET_KERNEL_CHECK, Error::InvalidArgument on failure).
> 4. If `out.dim() == 0`: `memcpy` `out.nbytes()` bytes from `in` to `out` and
>    return (scalar select of a 0-d input).
> 5. Let `leading_dims = getLeadingDims(in, dim)` (product of sizes before `dim`),
>    `trailing_dims = getTrailingDims(in, dim)` (product of sizes after `dim`). If
>    either is 0, return `out` unchanged (empty).
> 6. Let `in_dim_length = in.size(dim)`, `out_dim_length = out.size(dim)`,
>    `length_per_step = trailing_dims * in.element_size()` (byte size of one slice
>    along `dim`).
> 7. Dispatch over `index_type` (Long or Int). Let `dim_size = in.size(dim)`. For
>    each leading block `i` in `[0, leading_dims)`:
>    - `src = in_data + i*in_dim_length*length_per_step`,
>      `dest = out_data + i*out_dim_length*length_per_step`.
>    - For each output position `j` in `[0, out_dim_length)`: wrap the index
>      `index_val = index_arr[j] < 0 ? index_arr[j] + dim_size : index_arr[j]`;
>      `memcpy` `length_per_step` bytes from `src + index_val*length_per_step` to
>      `dest + j*length_per_step`.
> 8. Return `out`. (This is a byte-level copy; no dtype conversion.)

> [spec:et:def:op-index.torch.executor.native.index-tensor-out-fn]
> Tensor& index_Tensor_out( KernelRuntimeContext& ctx, const Tensor& in, TensorOptList indices, Tensor& out)

> [spec:et:sem:op-index.torch.executor.native.index-tensor-out-fn]
> Public advanced-indexing op: `out = in[indices...]` where `indices` is a list
> of optional index tensors (some may be null). Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; then
>    `tensor_is_default_dim_order(in)`. Each failure → Error::InvalidArgument,
>    return `out`.
> 2. Check the fast path: `is_fast_path = check_fast_path_conditions(in, indices, &dim)`
>    per `[spec:et:sem:op-index.torch.executor.native.check-fast-path-conditions-fn]`.
>    If true, return `fast_path(ctx, in, indices, dim, out)` per
>    `[spec:et:sem:op-index.torch.executor.native.fast-path-fn]`.
> 3. General path. ET_KERNEL_CHECK `check_index_args(in, indices, out)` per
>    `[spec:et:sem:advanced-index-util.check-index-args]`; on failure
>    Error::InvalidArgument, return `out`.
> 4. Let `in_type = in.scalar_type()`, `block_count = count_index_blocks(indices)`
>    (number of maximal runs of adjacent non-null indices).
> 5. Empty-indexing case: if `block_count == 0` (indices empty or all null),
>    resize `out` to `in.sizes()` (ET_KERNEL_CHECK) and `memcpy` all of `in` into
>    `out` (dispatch over REALHBBF16 dtype for `in.nbytes()`), then return `out`.
> 6. `adjacent = (block_count == 1)` determines output layout (whether the indexed
>    dims collapse into a single contiguous block).
> 7. Compute the output shape into `expected_size`/`expected_ndim` via
>    `get_index_out_target_size(in, indices, adjacent, ...)` per
>    `[spec:et:sem:advanced-index-util.get-index-out-target-size]` (ET_KERNEL_CHECK
>    on failure), then resize `out` to it (ET_KERNEL_CHECK).
> 8. If `out.numel() == 0`, return `out` early.
> 9. Set up the coordinate-translation state:
>    - `start = get_num_leading_null_indices(indices)` if `adjacent`, else 0.
>    - `xdim = get_indices_broadcast_ndim(indices)` (broadcast rank of the index
>      tensors).
>    - `compute_dim_map(in, indices, dim_map, block_count == 1)` and
>      `compute_index_map(in, indices, ix_map)` build the maps used to translate
>      an output coordinate to an input coordinate (per the advanced-index-util
>      rules).
> 10. Dispatch over `in_type` in REALHBBF16. For each flat output index `out_ix`
>     in `[0, out.numel())`: compute the source input flat index and a success
>     flag via `get_in_ix(in, indices, out, out_ix, start, xdim, dim_map, ix_map)`
>     per `[spec:et:sem:advanced-index-util.get-in-ix]`; ET_KERNEL_CHECK the
>     success flag (Error::InvalidArgument, return `out`); then
>     `out_data[out_ix] = in_data[in_ix]`.
> 11. Return `out`.

