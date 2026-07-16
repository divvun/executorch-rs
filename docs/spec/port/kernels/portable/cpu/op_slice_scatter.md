# kernels/portable/cpu/op_slice_scatter.cpp

> [spec:et:def:op-slice-scatter.torch.executor.native.slice-scatter-out-fn]
> Tensor& slice_scatter_out( KernelRuntimeContext& ctx, const Tensor& input, const Tensor& src, int64_t dim, std::optional<int64_t> start_val, std::optional<int64_t> end_val, int64_t step, Tensor& out)

> [spec:et:sem:op-slice-scatter.torch.executor.native.slice-scatter-out-fn]
> Returns a copy of `input` with the strided slice along `dim` overwritten by
> `src`. Implements `slice_scatter.out(Tensor self, Tensor src, int dim, int?
> start=None, int? end=None, int step=1, *, Tensor(a!) out)`. Step by step:
>
> - Normalize `dim`: if `dim < 0`, `dim += input.dim()`.
> - Resize `out` to `input.sizes()`; on failure ET_KERNEL_CHECK sets
>   Error::InvalidArgument and returns `out` unchanged.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(input, out)`; else
>   Error::InvalidArgument.
> - If `input.numel() == 0`, return `out` (already resized) immediately.
> - ET_KERNEL_CHECK `dim >= 0 && dim < input.dim()`; else Error::InvalidArgument.
> - Resolve bounds: `end = end_val.has_value() ? end_val.value() : input.size(dim)`;
>   `start = start_val.has_value() ? start_val.value() : 0`.
> - ET_KERNEL_CHECK `step > 0`; else Error::InvalidArgument.
> - `num_values = adjust_slice_indices(input.size(dim), &start, &end, step)` (see
>   `[spec:et:sem:slice-util...adjust-slice-indices-fn]`): clamps start/end,
>   returns the number of sliced positions.
> - ET_KERNEL_CHECK `check_slice_scatter_args(input, src, dim, num_values, step,
>   out)` (see `[spec:et:sem:slice-util...check-slice-scatter-args-fn]`):
>   validates `src` shape matches the slice shape (input shape with `dim` sized
>   `num_values`) and `input`/`out` compatibility; on failure Error::InvalidArgument.
> - Let `dim_length = input.size(dim)`, `leading_dims = getLeadingDims(input, dim)`
>   (product of sizes before `dim`), `trailing_dims = getTrailingDims(input, dim)`
>   (product of sizes after `dim`).
> - First `memcpy` the entire `input` buffer into `out` (byte-for-byte,
>   `input.nbytes()`), so unscattered positions keep `input`'s values.
> - Dtype dispatch: both `input` (in_type) and `src` (src_type) dtypes come from
>   REALHBBF16 = {Byte, Char, Short, Int, Long, Bool, Half, Float, Double,
>   BFloat16}, switched independently as CTYPE and CTYPE_SRC. `out` shares
>   `input`'s dtype (CTYPE).
> - Scatter loop: `src_offset = 0`. For each `i` in `[0, leading_dims)`:
>   `out_offset = (i * dim_length + start) * trailing_dims`. For each of
>   `num_values` steps `j`: for each `k` in `[0, trailing_dims)` set
>   `out_data[out_offset + k] = convert<CTYPE, CTYPE_SRC>(src_data[src_offset + k])`
>   (numeric cast from src dtype to out dtype); then `src_offset += trailing_dims`
>   and `out_offset += step * trailing_dims`. This writes `src` densely
>   (contiguous in src order) into the strided slice positions of `out`.
> - Returns `out`.

