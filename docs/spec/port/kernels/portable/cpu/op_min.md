# kernels/portable/cpu/op_min.cpp

> [spec:et:def:op-min.torch.executor.native.min-out-fn]
> std::tuple<Tensor&, Tensor&> min_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool keepdim, Tensor& min, Tensor& min_indices)

> [spec:et:sem:op-min.torch.executor.native.min-out-fn]
> Reduces `in` along a single dimension `dim`, producing `min` (the minimum
> values) and `min_indices` (the int64 argmin index within `dim`). Returns the
> tuple `{min, min_indices}`. Mirror of
> `[spec:et:sem:op-max.torch.executor.native.max-out-fn]` with the comparison
> reversed.
>
> Steps (every ET_KERNEL_CHECK below, on failure, sets Error::InvalidArgument on
> the context and returns the tuple `{min, min_indices}` unchanged):
> 1. Validate via `check_min_max_args(in, dim, keepdim, min, min_indices)`:
>    `dim` valid for `in`, `min_indices` dtype Long, `min` dtype canCast from
>    `in`'s dtype, shapes consistent with a reduction over `dim` with `keepdim`.
> 2. Resize `min` for the reduction over `dim` with `keepdim` per
>    `[spec:et:sem:reduce-util.torch.executor.native.resize-reduction-out-fn]`.
> 3. Resize `min_indices` to `min.sizes()`.
> 4. Check `in` and `min` share a dim order.
> 5. Check `min_indices` is default (contiguous) dim order.
> 6. Check `in` is default (contiguous) dim order.
> 7. Normalize `dim`: if `dim < 0`, `dim += in.dim()`.
> 8. Dispatch on `in.scalar_type()` over REALHBBF16 {Byte, Char, Short, Int,
>    Long, Half, Float, Double, Bool, BFloat16} (CTYPE). For each output index
>    `out_ix`, reduce over the `dim` slice of `in`: iterate the slice elements
>    with in-slice index `ix`, keeping `(acc_val, acc_ix)`; update rule with NaN
>    propagation — if `acc_val` is not NaN and (candidate `v` is NaN or
>    `v < acc_val`), set `acc_val = v`, `acc_ix = ix`. A NaN anywhere makes the
>    result NaN with the first-NaN index; ties keep the earliest index (strict
>    `<`). Write `acc_val` to `min[out_ix]` and `acc_ix` to
>    `min_indices[out_ix]`.
> 9. If the parallel reduction reports failure, ET_KERNEL_CHECK_MSG sets
>    Error::Internal (message "parallel_for failed").
> 10. Return `{min, min_indices}`.

> [spec:et:def:op-min.torch.executor.native.upper-bound-fn]
> constexpr CTYPE upper_bound()

> [spec:et:sem:op-min.torch.executor.native.upper-bound-fn]
> Compile-time helper returning the largest sentinel value for type `CTYPE`, used
> to initialize a running minimum. If `CTYPE` has an infinity representation
> (floating types), returns `+infinity`; otherwise (integer types) returns
> `std::numeric_limits<CTYPE>::max()` (the largest representable value).

> [spec:et:def:op-min.torch.executor.native.min-unary-out-fn]
> Tensor&

> [spec:et:sem:op-min.torch.executor.native.min-unary-out-fn]
> Reduces `in` over all elements to a single scalar minimum, written into the
> 0-dim tensor `out`. Mirror of
> `[spec:et:sem:op-max.torch.executor.native.max-unary-out-fn]`.
>
> Steps:
> 1. Resize `out` to shape `{}` (0-dim scalar) (ET_KERNEL_CHECK: on failure sets
>    Error::InvalidArgument and returns `out` unchanged).
> 2. Check `in` and `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 3. Require `canCast(in_type, out_type)` (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 4. Dispatch on `in_type` (CTYPE_IN) and `out_type` (CTYPE_OUT), both over
>    REALHBBF16 {Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>    BFloat16}.
> 5. Initialize `out[0]` to `upper_bound<CTYPE_OUT>()` (see
>    `[spec:et:sem:op-min.torch.executor.native.upper-bound-fn]`). Iterate every
>    element `i` in `[0, in.numel())` (row-major): let `val = (CTYPE_OUT)in[i]`;
>    if `val` is NaN, set `out[0] = val` and break (NaN short-circuits and
>    propagates); otherwise if `val < out[0]`, set `out[0] = val`.
> 6. Return `out`. (For an empty `in`, `out[0]` stays at `upper_bound`.)

