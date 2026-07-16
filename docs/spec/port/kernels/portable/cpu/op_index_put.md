# kernels/portable/cpu/op_index_put.cpp

> [spec:et:def:op-index-put.torch.executor.native.check-special-case-in-place-args-fn]
> bool check_special_case_in_place_args( KernelRuntimeContext& ctx, Tensor& in, TensorOptList indices, const Tensor& values, const bool accumulate, size_t* dim)

> [spec:et:sem:op-index-put.torch.executor.native.check-special-case-in-place-args-fn]
> Validates that the in-place `index_put_` "special case" (a single 1-D Long/Int
> index) applies, reporting the indexed axis via `*dim`. Returns true if valid,
> false with a logged reason otherwise. Steps:
> 1. `!accumulate` must hold ("Special case in-place index_put does not support
>    accumulate"); else false.
> 2. `indices.size() <= in.dim()` ("Indexing too many dimensions"); else false.
> 3. Iterate `i` over `indices`; for each non-null entry: set `*dim = i`; require
>    no prior non-null index was seen (single index only); mark found; require
>    `index.scalar_type()` is Long or Int; require `index.dim() == 1`. Any failure
>    → false with the corresponding message.
> 4. Require at least one non-null index was found; else false.
> 5. Let `index = indices[*dim].value()`. Dispatch over Long/Int and validate
>    every index value is in `[0, in.size(*dim))` (NOTE: unlike the op_index fast
>    path, negative indices are NOT wrapped here — a negative value is out of
>    range). Log "Index out of range" and mark invalid on the first violation;
>    require all valid ("Some index values are not within bounds...").
> 6. Require `values.size(*dim) == index.size(0)` (values length along the indexed
>    axis equals the number of indices).
> 7. Build `expected_values_size` = `in`'s shape with the `*dim` axis replaced by
>    `index.size(0)`, and require `tensor_has_expected_size(values, expected_values_size)`
>    — values must match `in`'s shape on every non-indexed axis and equal
>    `index.size(0)` on the indexed axis (message includes the shapes when logging
>    is enabled).
> 8. Return true.

> [spec:et:def:op-index-put.torch.executor.native.index-put-fn]
> Tensor& index_put_( KernelRuntimeContext& ctx, Tensor& in, TensorOptList indices, const Tensor& values, const bool accumulate)

> [spec:et:sem:op-index-put.torch.executor.native.index-put-fn]
> In-place index_put on `in` for the special single-index case: writes rows of
> `values` into `in` at positions given by a single 1-D index tensor along one
> axis. Returns `in` (mutated in place; no separate out tensor). `accumulate` is
> not supported and must be false.
>
> Steps:
> 1. ET_KERNEL_CHECK: `tensors_have_same_dtype(in, values)`;
>    `tensors_have_same_dim_order(in, values)`; `tensor_is_default_dim_order(in)`.
>    Each failure → Error::InvalidArgument, return `in` unchanged.
> 2. ET_KERNEL_CHECK: `check_special_case_in_place_args(ctx, in, indices, values, accumulate, &dim)`
>    per `[spec:et:sem:op-index-put.torch.executor.native.check-special-case-in-place-args-fn]`;
>    on failure Error::InvalidArgument, return `in`. This sets `dim`.
> 3. Let `index = indices[dim].value()`, `index_type = index.scalar_type()`.
> 4. If `in.dim() == 0`: `memcpy` `in.nbytes()` bytes from `values` into `in`,
>    return `in`.
> 5. `leading_dims = getLeadingDims(in, dim)`, `trailing_dims = getTrailingDims(in, dim)`;
>    if either is 0, return `in` unchanged.
> 6. `values_dim_length = values.size(dim)`, `in_dim_length = in.size(dim)`,
>    `length_per_step = trailing_dims * in.element_size()`.
> 7. Dispatch over `index_type` (Long/Int). For each leading block `i` in
>    `[0, leading_dims)`:
>    - `src = values_data + i*values_dim_length*length_per_step`,
>      `dest = in_data + i*in_dim_length*length_per_step`.
>    - For each `j` in `[0, values_dim_length)`: `memcpy` `length_per_step` bytes
>      from `src + j*length_per_step` to `dest + index_arr[j]*length_per_step`
>      (scatter row `j` of values into `in` at index position `index_arr[j]`;
>      indices are pre-validated non-negative in bounds).
> 8. Return `in`. (Byte-level copy, no accumulation.)

> [spec:et:def:op-index-put.torch.executor.native.index-put-out-fn]
> Tensor& index_put_out( KernelRuntimeContext& ctx, const Tensor& in, TensorOptList indices, const Tensor& values, const bool accumulate, Tensor& out)

> [spec:et:sem:op-index-put.torch.executor.native.index-put-out-fn]
> General functional index_put: `out = in`, then scatter `values` into the
> positions selected by `indices` (advanced indexing), optionally accumulating.
> Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_index_args(in, indices, out)`;
>    `tensors_have_same_dtype(in, values)`; `tensors_have_same_dim_order(in, out)`;
>    `tensor_is_default_dim_order(in)`. Each failure → Error::InvalidArgument,
>    return `out`.
> 2. Let `in_type = in.scalar_type()`, `block_count = count_index_blocks(indices)`.
> 3. Whole-tensor case (`block_count == 0`, indices empty or all null): resize
>    `out` to `in.sizes()` (ET_KERNEL_CHECK); ET_KERNEL_CHECK
>    `tensor_is_broadcastable_to(values, out)`; then dispatch over REALHBBF16 and
>    apply the binary elementwise fn over (`in`, `values`) broadcast to `out`: for
>    each element write `accumulate ? in_val + val : val`. Return `out`.
> 4. Otherwise `adjacent = (block_count == 1)`. Compute the implicit indexed-result
>    shape `x_sizes`/`x_dim` via `get_index_out_target_size(in, indices, adjacent, ...)`
>    (ET_KERNEL_CHECK).
> 5. ET_KERNEL_CHECK `tensor_is_broadcastable_to(values.sizes(), {x_sizes, x_dim})`
>    (values must broadcast to the indexed-result shape).
> 6. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK). If `in.numel() == 0`, return
>    `out` early.
> 7. `memcpy` all of `in` into `out` (out starts as a copy of in).
> 8. Set up `x`→`in` coordinate translation for the implicit indexed tensor
>    `x = in[indices]`: `start = get_num_leading_null_indices(indices)` if
>    `adjacent` else 0; `bc_ndim = get_indices_broadcast_ndim(indices)`;
>    `compute_dim_map(in, indices, dim_map, block_count == 1)`;
>    `compute_index_map(in, indices, ix_map)`. Compute `x_numel = product(x_sizes)`.
> 9. Dispatch over `in_type` in REALHBBF16. For each `x_ix` in `[0, x_numel)`:
>    - Delinearize `x_ix` into `x_coord` over `x_sizes`/`x_dim`
>      (`delinearize_index`).
>    - ET_KERNEL_CHECK `get_in_coord(in, indices, start, bc_ndim, dim_map, ix_map, x_coord, in_coord)`
>      per `[spec:et:sem:advanced-index-util.get-in-coord]` (translates the x
>      coordinate to an input coordinate; Error::InvalidArgument on failure).
>    - `in_ix = coordinateToIndex(in, in_coord)` (flat input/out offset).
>    - `val_ix = linearize_access_indexes(x_coord, x_dim, values)` (values flat
>      offset with broadcasting per `[spec:et:sem:broadcast-util.linearize-access-indexes]`).
>    - If `accumulate`: `out_data[in_ix] += values_data[val_ix]`; else
>      `out_data[in_ix] = values_data[val_ix]`.
> 10. Return `out`.

