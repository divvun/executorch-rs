# kernels/portable/cpu/op_flip.cpp

> [spec:et:def:op-flip.torch.executor.native.check-flip-args-fn]
> bool check_flip_args(const Tensor& in, IntArrayRef dims, const Tensor& out)

> [spec:et:sem:op-flip.torch.executor.native.check-flip-args-fn]
> Validates `flip` arguments. Returns `true` if valid, else logs and returns `false`:
> 1. `tensors_have_same_dtype(in, out)` (in and out share dtype); on failure return `false`.
> 2. Return `check_dim_list_is_valid(in, dims)` — every entry of `dims` is a valid axis of `in`
>    (in range `[-in.dim(), in.dim())`) with no duplicates.

> [spec:et:def:op-flip.torch.executor.native.unflip-flat-ix-fn]
> size_t unflip_flat_ix(size_t ix, const Tensor& in, ArrayRef<bool> flip_dim)

> [spec:et:sem:op-flip.torch.executor.native.unflip-flat-ix-fn]
> Given an output flat index `ix` (into a tensor with `in`'s shape), returns the flat index in `in`
> of the source element that maps to it under the flip along the dimensions marked `true` in
> `flip_dim`. Since flip is its own inverse, this is used to gather: `out[ix] = in[unflip_flat_ix(ix)]`.
> 1. Decompose `ix` into per-dimension coordinates `ix_coord` via `indexToCoordinate(in, ix, ix_coord)`
>    (row-major / contiguous decomposition using `in`'s shape).
> 2. For each dim `d` in `[0, in.dim())`: if `flip_dim[d]`, `unflip_coord[d] = in.size(d) - ix_coord[d] - 1`
>    (mirror the coordinate along that axis); else `unflip_coord[d] = ix_coord[d]`.
> 3. Return `coordinateToIndex(in, unflip_coord)` (recompose to a flat index using `in`'s strides).

> [spec:et:def:op-flip.torch.executor.native.flip-out-fn]
> Tensor& flip_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef dims, Tensor& out)

> [spec:et:sem:op-flip.torch.executor.native.flip-out-fn]
> Implements `flip.out(in, dims, *, out)`: reverses `in` along each axis in `dims`, writing to `out`.
> Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `resize_tensor(out, in.sizes()) == Error::Ok`, else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `check_flip_args(in, dims, out)` per
>    `[spec:et:sem:op-flip.torch.executor.native.check-flip-args-fn]`, else InvalidArgument.
> 4. Build a boolean per-dimension flip mask `flip_dim` of length `in.dim()`: initialize all `false`;
>    for each entry `d0` in `dims`, normalize `d = d0 < 0 ? d0 + nonzero_dim(in) : d0` and set
>    `flip_dim[d] = true`.
> 5. Dispatch over `in.scalar_type()` in REALHBBF16 (`{Byte, Char, Short, Int, Long, Half, Float,
>    Double, Bool, BFloat16}`). For each flat index `ix` in `[0, in.numel())`:
>    `out[ix] = in[unflip_flat_ix(ix, in, flip_dim)]` per
>    `[spec:et:sem:op-flip.torch.executor.native.unflip-flat-ix-fn]`.
> 6. Return `out`.

