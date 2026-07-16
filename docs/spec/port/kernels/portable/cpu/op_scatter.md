# kernels/portable/cpu/op_scatter.cpp

> [spec:et:def:op-scatter.torch.executor.native.scatter-src-helper-fn]
> void scatter_src_helper( const Tensor& in, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)

> [spec:et:sem:op-scatter.torch.executor.native.scatter-src-helper-fn]
> Typed helper implementing scatter with a `src` tensor for ctype CTYPE. `in`,
> `src`, `out` share dtype CTYPE; `index` is int64. Steps:
>
> - `memcpy` `in.nbytes()` bytes from `in` into `out` (start from a copy of the
>   input).
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)` (number of
>   dimensions, treating a 0-dim tensor as having 1 dim for this purpose).
> - For each flat index `ix` in `[0, index.numel())`:
>   - Convert `ix` to coordinates `ix_coord` in `index` via `indexToCoordinate`
>     (see
>     `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]`).
>   - `src_ix = coordinateToIndex(src, ix_coord)` — read the source element at
>     the same coordinate in `src` (see
>     `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`).
>   - Build `out_coord`: for each dim `i` of `out`, `out_coord[i] =
>     index_data[ix]` when `i == dim`, else `ix_coord[i]`. So along `dim` the
>     destination position is taken from the index value; other dims keep the
>     index coordinate.
>   - `out_ix = coordinateToIndex(out, out_coord)`; write `out_data[out_ix] =
>     src_data[src_ix]`.
> - Later index entries targeting the same `out_ix` overwrite earlier ones (last
>   write wins). Index bounds are assumed pre-validated by the caller's
>   `check_scatter_src_args`.

> [spec:et:def:op-scatter.torch.executor.native.scatter-value-helper-fn]
> void scatter_value_helper( const Tensor& in, int64_t dim, const Tensor& index, CTYPE_VAL val, Tensor& out)

> [spec:et:sem:op-scatter.torch.executor.native.scatter-value-helper-fn]
> Typed helper implementing scatter with a single scalar `val` (of type
> CTYPE_VAL) into all index-selected positions, for output ctype CTYPE. Identical
> structure to
> `[spec:et:sem:op-scatter.torch.executor.native.scatter-src-helper-fn]` except
> there is no `src` tensor:
>
> - `memcpy` `in.nbytes()` bytes from `in` into `out`.
> - If `dim < 0`, `dim += nonzero_dim(in)`.
> - For each flat index `ix` in `[0, index.numel())`: convert `ix` to `ix_coord`
>   in `index`; build `out_coord` where `out_coord[i] = index_data[ix]` for `i
>   == dim` else `ix_coord[i]`; `out_ix = coordinateToIndex(out, out_coord)`;
>   write `out_data[out_ix] = static_cast<CTYPE>(val)`.
> - Every selected output position receives the same scalar; last write wins on
>   collisions.

> [spec:et:def:op-scatter.torch.executor.native.scatter-src-out-fn]
> Tensor& scatter_src_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)

> [spec:et:sem:op-scatter.torch.executor.native.scatter-src-out-fn]
> `scatter.src_out`: scatter values from `src` into a copy of `in` at positions
> given by `index` along `dim`. Steps:
>
> - ET_KERNEL_CHECK: `check_scatter_src_args(in, dim, index, src, out)` (see
>   `[spec:et:sem:index-util.torch.executor.check-scatter-src-args-fn]`, which
>   validates dim range, index dtype (Long), index/src shapes relative to `in`,
>   and matching in/src/out dtypes); on failure `Error::InvalidArgument`, return
>   `out`.
> - Resize `out` to `in.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - Dispatch on `in.scalar_type()` over REALHBBF16 = {Byte, Char, Short, Int,
>   Long, Half, Float, Double, Bool, BFloat16} as CTYPE, and call
>   `scatter_src_helper<CTYPE>(in, dim, index, src, out)` (see
>   `[spec:et:sem:op-scatter.torch.executor.native.scatter-src-helper-fn]`).
> - Returns `out`.

> [spec:et:def:op-scatter.torch.executor.native.scatter-value-out-fn]
> Tensor& scatter_value_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, const Tensor& index, const Scalar& value, Tensor& out)

> [spec:et:sem:op-scatter.torch.executor.native.scatter-value-out-fn]
> `scatter.value_out`: scatter a scalar `value` into a copy of `in` at positions
> given by `index` along `dim`. Steps:
>
> - ET_KERNEL_CHECK: `check_scatter_value_args(in, dim, index, value, out)` (see
>   `[spec:et:sem:index-util.torch.executor.check-scatter-value-args-fn]`); on
>   failure `Error::InvalidArgument`, return `out`.
> - Resize `out` to `in.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - Dispatch on `in.scalar_type()` over REALHBBF16 as CTYPE. Cast `value` to
>   CTYPE with overflow checking via
>   `utils::internal::check_overflow_scalar_cast<CTYPE>(value)` (see
>   `[spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]`);
>   ET_KERNEL_CHECK the returned optional has a value (else `Error::InvalidArgument`,
>   return `out`). Then call `scatter_value_helper<CTYPE>(in, dim, index, val,
>   out)` (see
>   `[spec:et:sem:op-scatter.torch.executor.native.scatter-value-helper-fn]`).
> - Returns `out`.

