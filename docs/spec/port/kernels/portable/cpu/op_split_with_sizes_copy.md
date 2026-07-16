# kernels/portable/cpu/op_split_with_sizes_copy.cpp

> [spec:et:def:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn]
> void split_with_sizes_copy_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> split_sizes, int64_t dim, TensorList out)

> [spec:et:sem:op-split-with-sizes-copy.torch.executor.native.split-with-sizes-copy-out-fn]
> Splits `in` into chunks of the given `split_sizes` along `dim`, writing each
> chunk into the corresponding `out` tensor. Implements
> `split_with_sizes_copy.out(Tensor self, SymInt[] split_sizes, int dim=0, *,
> Tensor(a!)[] out)`. Returns void (writes through `out`). Step by step:
>
> - Normalize `dim`: if `dim < 0`, `dim += in.dim()`. (Does not accept 0-dim
>   input.)
> - ET_KERNEL_CHECK `check_split_with_sizes_copy_args(in, split_sizes, dim, out)`
>   (see `[spec:et:sem:copy-ops-util...check-split-with-sizes-copy-args-fn]`):
>   validates `out.size() == split_sizes.size()`, each `split_sizes[i] >= 0`,
>   `sum(split_sizes) == in.size(dim)`, `dim` valid, and dtype/shape
>   compatibility. On failure Error::InvalidArgument and returns.
> - For each `i` in `[0, out.size())` ET_KERNEL_CHECK
>   `tensors_have_same_dim_order(in, out[i])`; else Error::InvalidArgument.
> - If `out.size() == 0`, return (valid args imply `in.size(dim) == 0` and empty
>   split_sizes).
> - Build `target_out_sizes` = `in.sizes()` (ndim = `in.dim()`). For each chunk
>   `i` set `target_out_sizes[dim] = split_sizes[i]` and resize `out[i]` to that
>   shape; on resize failure Error::InvalidArgument, returns.
> - Let `leading_dims = getLeadingDims(in, dim)`, `trailing_dims =
>   getTrailingDims(in, dim)`, `step = in.size(dim) * trailing_dims`.
> - Dtype dispatch: in_type and out[0] type both from REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Bool, Half, Float, Double, BFloat16}, as CTYPE_IN /
>   CTYPE_OUT.
> - `in_data` starts at input base. For each chunk `i`:
>   - If `out[i].numel() == 0`, skip (do not advance `in_data`).
>   - `chunk_step = split_sizes[i] * trailing_dims`.
>   - Set `target_out_sizes[dim] = split_sizes[i]` and let `target_shape` be that
>     shape. `is_broadcasted = !out[i].sizes().equals(target_shape)`.
>   - If NOT broadcasted (the common case): for each `j` in `[0, leading_dims)`
>     copy `chunk_step` elements `out_data[k] = convert<CTYPE_OUT,CTYPE_IN>(src[k])`,
>     then advance `src` by `step` and `out_data` by `chunk_step`.
>   - If broadcasted: compute contiguous strides for `target_shape`
>     (`target_out_strides[ndim-1] = 1`, `target_out_strides[d] =
>     target_out_strides[d+1] * target_out_sizes[d+1]`). For each flat index `ix`
>     in `[0, out[i].numel())`: delinearize `ix` into `out_coord` using
>     `out[i]`'s shape, map to `in_linear_index = linearize_access_indexes(
>     out_coord, out[i].dim(), target_shape, target_strides)` (broadcast-aware
>     index mapping per `[spec:et:sem:broadcast-util...linearize-access-indexes-fn]`),
>     and set `out_data[ix] = convert<CTYPE_OUT,CTYPE_IN>(in_data[in_linear_index])`.
>   - After the chunk, advance `in_data` by `chunk_step`.
> - Returns void.

