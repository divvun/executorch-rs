# kernels/portable/cpu/op_select_scatter.cpp

> [spec:et:def:op-select-scatter.torch.executor.native.select-scatter-out-fn]
> Tensor& select_scatter_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& src, int64_t dim, int64_t index, Tensor& out)

> [spec:et:sem:op-select-scatter.torch.executor.native.select-scatter-out-fn]
> `select_scatter.out`: writes `src` (an `in.dim()-1`-dim tensor) into the
> `index`-th slice of a copy of `in` along `dim`. Steps:
>
> - Resize `out` to `in.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - ET_KERNEL_CHECK: `in`, `src`, `out` same dim order; else
>   `Error::InvalidArgument`, return `out`.
> - Normalize `dim`: if `dim < 0`, `dim += in.dim()`. ET_KERNEL_CHECK: `0 <= dim
>   < in.dim()`; else `Error::InvalidArgument`, return `out`.
> - Normalize `index`: if `index < 0`, `index += in.size(dim)`.
> - ET_KERNEL_CHECK: `check_select_scatter_args(in, src, dim, index, out)` (see
>   `[spec:et:sem:index-util.torch.executor.check-select-scatter-args-fn]`, which
>   validates index bounds along `dim`, matching dtypes, and that `src` has the
>   shape of `in` with `dim` removed); on failure `Error::InvalidArgument`,
>   return `out`.
> - If `in.numel() == 0`, return `out` (empty input; nothing to write).
> - `memcpy` `in.nbytes()` bytes from `in` into `out` (copy input through).
> - Compute strides: `leading_dims = getLeadingDims(in, dim)` (product of sizes
>   before `dim`), `trailing_stride = getTrailingDims(in, dim)` (product of sizes
>   after `dim`), `start_offset = index * trailing_stride`, `out_step =
>   in.size(dim) * trailing_stride`.
> - Dispatch on `in.scalar_type()` (CTYPE) and `src.scalar_type()` (CTYPE_SRC),
>   both over REALHBBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double,
>   Bool, BFloat16}. For each `i` in `[0, leading_dims)` and `j` in `[0,
>   trailing_stride)`: `out_data[start_offset + i * out_step + j] =
>   convert<CTYPE, CTYPE_SRC>(src_data[i * trailing_stride + j])`. So the
>   selected slice is filled from `src` with per-element dtype conversion; `src`
>   and `in`/`out` dtypes may differ.
> - Returns `out`.

