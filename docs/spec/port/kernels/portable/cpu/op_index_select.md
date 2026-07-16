# kernels/portable/cpu/op_index_select.cpp

> [spec:et:def:op-index-select.torch.executor.native.index-select-out-fn]
> Tensor& index_select_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, const Tensor& index, Tensor& out)

> [spec:et:sem:op-index-select.torch.executor.native.index-select-out-fn]
> Implements `torch.index_select(in, dim, index)`: gathers slices of `in` along
> `dim` at the positions in the 1-D `index` tensor. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_index_select_args(in, dim, index, out)` per
>    `[spec:et:sem:index-util.check-index-select-args]` (validates `dim` is a
>    valid axis, `index` is a 1-D Long/Int tensor with in-range values,
>    `in`/`out` same dtype, output shape consistent); then
>    `tensors_have_same_dim_order(in, out)`; then `tensor_is_default_dim_order(in)`.
>    Each failure → Error::InvalidArgument, return `out`.
> 2. Normalize `dim`: if `dim < 0`, add `nonzero_dim(in)`.
> 3. Compute the output shape into `expected_size`/`expected_ndim` via
>    `get_index_select_out_target_size(in, dim, index, ...)` (same as `in`'s shape
>    with the `dim` axis replaced by `index.numel()`), then resize `out` to it
>    (ET_KERNEL_CHECK, Error::InvalidArgument on failure).
> 4. If `in.dim() == 0`: `memcpy` `in.nbytes()` bytes into `out`, return `out`.
> 5. `leading_dims = getLeadingDims(in, dim)`, `trailing_dims = getTrailingDims(in, dim)`;
>    if either is 0, return `out` unchanged.
> 6. `out_dim_length = out.size(dim)`, `in_dim_length = in.size(dim)`,
>    `length_per_step = trailing_dims * in.element_size()`.
> 7. Dispatch over `index.scalar_type()` (Long/Int). For each leading block `i`:
>    - `src = input_data + i*in_dim_length*length_per_step`,
>      `dest = out_data + i*out_dim_length*length_per_step`.
>    - For each output slot `j` in `[0, out_dim_length)`: `memcpy` `length_per_step`
>      bytes from `src + index_arr[j]*length_per_step` to the current `dest`, then
>      advance `dest` by `length_per_step`. (Indices are used directly without
>      negative wrapping; they are pre-validated in range by
>      `check_index_select_args`.)
> 8. Return `out`. (Byte-level gather copy, no dtype conversion.)

