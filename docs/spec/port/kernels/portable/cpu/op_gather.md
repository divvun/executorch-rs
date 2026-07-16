# kernels/portable/cpu/op_gather.cpp

> [spec:et:def:op-gather.torch.executor.native.gather-helper-fn]
> void gather_helper( const Tensor& in, const Tensor& index, Tensor& out, int64_t dim)

> [spec:et:sem:op-gather.torch.executor.native.gather-helper-fn]
> Templated on `CTYPE` (the input/output element type). Performs the core gather
> copy: for each output position (which mirrors an `index` position), reads the
> element from `in` at the coordinate obtained by replacing the `dim` component
> with the corresponding value from `index`. `dim` is already normalized to a
> non-negative value by the caller.
>
> Steps:
> 1. Obtain `in_data` (const `CTYPE*`), `index_data` (const `int64_t*` — `index`
>    is always Long), and `out_data` (mutable `CTYPE*`).
> 2. Scalar (0-dim index) special case: if `index.dim() == 0`, set
>    `out_data[0] = in_data[index_data[0]]` and return. (Here `index_data[0]` is
>    the flat offset into `in`.)
> 3. Otherwise iterate `ix` over `0 .. index.numel()-1` in flat order:
>    - Delinearize `ix` into per-dimension coordinates `ix_coord` for the `index`
>      tensor via `indexToCoordinate` (a `size_t[kTensorDimensionLimit]` buffer).
>    - Build the input coordinate `in_coord` over `out.dim()` dimensions: for each
>      axis `i`, if `i == dim` set `in_coord[i] = index_data[ix]` (the gathered
>      index value along the gather axis); otherwise `in_coord[i] = ix_coord[i]`
>      (the output/index coordinate is reused for all other axes; `out` and
>      `index` share the same shape).
>    - Compute the flat input offset `in_ix = coordinateToIndex(in, in_coord)`
>      and the flat output offset `out_ix = coordinateToIndex(out, ix_coord)`
>      (both honoring the respective tensor strides).
>    - Copy `out_data[out_ix] = in_data[in_ix]`.
> Note: index-range validity is presumed already checked by
> `check_gather_args` in the caller; this helper does no bounds checking.

> [spec:et:def:op-gather.torch.executor.native.gather-out-fn]
> Tensor& gather_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, const Tensor& index, bool sparse_grad, Tensor& out)

> [spec:et:sem:op-gather.torch.executor.native.gather-out-fn]
> Implements `torch.gather(in, dim, index)` into `out`. `sparse_grad` is
> accepted but unused (inference op). Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_gather_args(in, dim, index, sparse_grad, out)` must
>    hold. This validates (per `[spec:et:sem:index-util.check-gather-args]`):
>    `in` and `out` have the same dtype; `index` is a Long tensor; `dim` is a
>    valid axis of `in`; `index` has the same number of dimensions as `in` (or
>    `in` is 0-dim with a 0-dim/empty index); and each `index` size does not
>    exceed the corresponding `in` size except along `dim`. On failure set
>    Error::InvalidArgument and return `out` unchanged.
> 2. Normalize `dim`: if `dim < 0`, add `nonzero_dim(in)` (the number of
>    dimensions, treating 0-dim as 1) to make it non-negative.
> 3. Resize `out` to `index.sizes()` via `resize_tensor`; on non-Ok set
>    Error::InvalidArgument and return `out`. (Output shape equals index shape.)
> 4. Dispatch over `in.scalar_type()`, which must be in the REALHBBF16 set
>    {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}, and
>    invoke `gather_helper<CTYPE>(in, index, out, dim)` per
>    `[spec:et:sem:op-gather.torch.executor.native.gather-helper-fn]`.
> 5. Return `out`.

