# kernels/portable/cpu/op_scatter_add.cpp

> [spec:et:def:op-scatter-add.torch.executor.native.scatter-add-helper-fn]
> void scatter_add_helper( const CTYPE* src_data, const int64_t* index_data, CTYPE* out_data, const Tensor& src, const Tensor& index, Tensor& out, int64_t dim)

> [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-helper-fn]
> Typed helper accumulating `src` values into `out` at index-selected positions,
> for ctype CTYPE. Takes raw pointers `src_data`, `index_data` (int64),
> `out_data`, plus `src`, `index`, `out` tensors and normalized `dim`. Steps:
>
> - For each flat index `ix` in `[0, index.numel())`:
>   - Convert `ix` to `ix_coord` in `index` via `indexToCoordinate` (see
>     `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]`).
>   - `src_ix = coordinateToIndex(src, ix_coord)` (see
>     `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`).
>   - Build `out_coord`: `out_coord[i] = index_data[ix]` for `i == dim`, else
>     `ix_coord[i]`.
>   - `out_ix = coordinateToIndex(out, out_coord)`; accumulate `out_data[out_ix]
>     += src_data[src_ix]`.
> - Unlike scatter (overwrite), collisions on the same `out_ix` accumulate
>   additively. Addition uses CTYPE arithmetic.

> [spec:et:def:op-scatter-add.torch.executor.native.scatter-add-out-fn]
> Tensor& scatter_add_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)

> [spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-out-fn]
> `scatter_add.out`: add values from `src` into a copy of `self` at positions
> given by `index` along `dim`. Steps:
>
> - ET_KERNEL_CHECK: `check_scatter_add_args(self, dim, index, src, out)` (see
>   `[spec:et:sem:index-util.torch.executor.check-scatter-add-args-fn]`); on
>   failure `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `self`, `src`, `out` same dim order; else
>   `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `index` is default dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(self)`.
> - Resize `out` to `self.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - Dispatch on `self.scalar_type()` over REALHBBF16 = {Byte, Char, Short, Int,
>   Long, Half, Float, Double, Bool, BFloat16} as CTYPE. `memcpy` `self.nbytes()`
>   bytes from `self` into `out`. Then, if `index.numel() != 0`:
>   - If `self.dim() == 0` (scalar self/src): `out_data[0] +=
>     static_cast<CTYPE>(nonempty_size(index, 0)) * src_data[0]` — add the single
>     src element as many times as `index` has elements along dim 0.
>   - Else call `scatter_add_helper<CTYPE>(src_data, index_data, out_data, src,
>     index, out, dim)` (see
>     `[spec:et:sem:op-scatter-add.torch.executor.native.scatter-add-helper-fn]`).
> - If `index.numel() == 0`, `out` is just the copy of `self`.
> - Returns `out`.

