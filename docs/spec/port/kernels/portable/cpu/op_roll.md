# kernels/portable/cpu/op_roll.cpp

> [spec:et:def:op-roll.torch.executor.native.check-roll-args-fn]
> bool check_roll_args( const Tensor& in, IntArrayRef shifts, IntArrayRef dims, const Tensor& out)

> [spec:et:sem:op-roll.torch.executor.native.check-roll-args-fn]
> Validates arguments for `roll.out`. Returns `bool`; each failing check logs
> and returns false. In order:
>
> - `tensor_has_rank_greater_or_equal_to(in, 1)`: `in.dim() >= 1`.
> - If `in.numel() > 0`: every `d` in `dims` must satisfy `dim_is_valid(d,
>   in.dim())`, i.e. `-in.dim() <= d < in.dim()`. (When `in` is empty this dim
>   validity check is skipped.)
> - `!shifts.empty()`: at least one shift.
> - `shifts.size() == dims.size()`: shifts and dims have equal length.
> - `tensors_have_same_dtype(in, out)`.
> - Returns true if all pass.

> [spec:et:def:op-roll.torch.executor.native.unshift-flat-ix-fn]
> size_t unshift_flat_ix(size_t ix, const Tensor& in, IntArrayRef dim_shifts)

> [spec:et:sem:op-roll.torch.executor.native.unshift-flat-ix-fn]
> Given an output flat index `ix` and per-dimension total shifts `dim_shifts`
> (one entry per dim of `in`), returns the input flat index whose value should be
> placed at output position `ix`. Steps:
>
> - Convert `ix` to multi-dim coordinates `ix_coord` for tensor `in` via
>   `indexToCoordinate` (see
>   `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]`).
> - For each dim `d` of `in`, compute the un-shifted (source) coordinate:
>   `shifted_coord[d] = (ix_coord[d] + in.size(d) - (dim_shifts[d] %
>   in.size(d))) % in.size(d)`. The `dim_shifts[d] % in.size(d)` reduces the
>   shift modulo the dim size; adding `in.size(d)` before the outer `%` keeps
>   the result non-negative for positive shifts (rolling forward means reading
>   from an earlier source index).
> - Convert `shifted_coord` back to a flat index for `in` via
>   `coordinateToIndex` (see
>   `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`)
>   and return it.
> - Note: the arithmetic uses `size_t` (unsigned) throughout; callers pass only
>   non-negative accumulated `dim_shifts` (roll_out folds negative user dims/
>   shifts before calling).

> [spec:et:def:op-roll.torch.executor.native.roll-out-fn]
> Tensor& roll_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef shifts, IntArrayRef dims, Tensor& out)

> [spec:et:sem:op-roll.torch.executor.native.roll-out-fn]
> Rolls (circularly shifts) `in` along the given `dims` by `shifts` into `out`.
> Steps:
>
> - Resize `out` to `in.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - ET_KERNEL_CHECK: `check_roll_args(in, shifts, dims, out)` (see
>   `[spec:et:sem:op-roll.torch.executor.native.check-roll-args-fn]`); on failure
>   `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - If `in.numel() == 0`, return `out` unchanged (empty input).
> - Build `dim_shift_array` of length `in.dim()`, initialized to 0. For each `i`
>   over `dims`: normalize the dim `d = dims[i] < 0 ? dims[i] + in.dim() :
>   dims[i]` and accumulate `dim_shift_array[d] += shifts[i]` (multiple shifts on
>   the same dim add together). Form `dim_shifts` as an IntArrayRef over the
>   first `in.dim()` entries.
> - Dispatch on `in.scalar_type()` over REALHBBF16 = {Byte, Char, Short, Int,
>   Long, Half, Float, Double, Bool, BFloat16} as CTYPE. For each output flat
>   index `ix` in `[0, out.numel())`: `out_data[ix] =
>   in_data[unshift_flat_ix(ix, in, dim_shifts)]` (see
>   `[spec:et:sem:op-roll.torch.executor.native.unshift-flat-ix-fn]`).
> - Returns `out`.

