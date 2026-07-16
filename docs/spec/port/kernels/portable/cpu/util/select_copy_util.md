# kernels/portable/cpu/util/select_copy_util.cpp

> [spec:et:def:select-copy-util.torch.executor.select-copy-util-fn]
> Error select_copy_util( const Tensor& in, int64_t dim, int64_t index, Tensor& out)

> [spec:et:sem:select-copy-util.torch.executor.select-copy-util-fn]
> Copies the slice `in[..., index, ...]` at position `index` along dimension
> `dim` into `out`, dropping that dimension (the shared implementation behind
> `select_copy.int_out`). Returns `Error` (`Ok` on success); errors are returned
> directly (this helper does not use the `KernelRuntimeContext` failure path).
>
> Steps:
> 1. Validate via `check_select_copy_out_args(in, dim, index, out)` per
>    `[spec:et:sem:copy-ops-util.torch.executor.check-select-copy-out-args-fn]`;
>    if it returns false, return `Error::InvalidArgument` (out left unchanged).
> 2. Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)` (the number of
>    non-trivial dims; effectively the rank, treating a scalar as rank 1).
> 3. Compute the output target shape: `get_select_copy_out_target_size(in, dim, ...)`
>    per `[spec:et:sem:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn]`,
>    which yields `in`'s sizes with dimension `dim` removed.
> 4. `resize_tensor(out, target)`; if that does not return `Error::Ok`, return
>    `Error::InvalidArgument`.
> 5. Require `tensors_have_same_dim_order(in, out)`; otherwise return
>    `Error::InvalidArgument`.
> 6. If `in.numel() == 0`, return `Error::Ok` (empty input, nothing to copy;
>    output already resized).
> 7. Normalize `index`: if `index < 0`, `index += in.size(dim)` (Python-style
>    negative indexing).
> 8. Compute copy geometry: `leading_dims = getLeadingDims(in, dim)` (product of
>    sizes before `dim`), `trailing_dims = getTrailingDims(in, dim)` (product of
>    sizes after `dim`), `dim_length = in.size(dim)`.
>    - `copy_size_per_op = trailing_dims * out.element_size()` bytes per memcpy.
>    - `src_step_per_op = dim_length * trailing_dims * in.element_size()` — the
>      byte stride between consecutive selected slices in the input.
>    - `start_offset = index * trailing_dims * in.element_size()` — byte offset of
>      the first selected element.
> 9. Copy: `src = in data + start_offset`, `dest = out data`. For `j` in
>    `[0, leading_dims)`: `memcpy(dest, src, copy_size_per_op)`, then advance
>    `src += src_step_per_op` and `dest += copy_size_per_op`. This gathers the
>    `trailing_dims`-sized block at position `index` from each of the
>    `leading_dims` outer slices, laid out contiguously in `out`.
> 10. Return `Error::Ok`. (Note: input dtype/element_size and output element_size
>     are assumed equal, guaranteed by the same-dtype check in step 1.)
