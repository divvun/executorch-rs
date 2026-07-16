# kernels/portable/cpu/op_max.cpp

> [spec:et:def:op-max.torch.executor.native.lower-bound-fn]
> constexpr CTYPE lower_bound()

> [spec:et:sem:op-max.torch.executor.native.lower-bound-fn]
> Compile-time helper returning the smallest sentinel value for type `CTYPE`,
> used to initialize a running maximum. If `CTYPE` has an infinity representation
> (floating types), returns `-infinity`; otherwise (integer types) returns
> `std::numeric_limits<CTYPE>::lowest()` (the most negative representable value).

> [spec:et:def:op-max.torch.executor.native.max-out-fn]
> std::tuple<Tensor&, Tensor&> max_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool keepdim, Tensor& max, Tensor& max_indices)

> [spec:et:sem:op-max.torch.executor.native.max-out-fn]
> Reduces `in` along a single dimension `dim`, producing `max` (the maximum
> values) and `max_indices` (the int64 argmax index within `dim`). Returns the
> tuple `{max, max_indices}`.
>
> Steps (every ET_KERNEL_CHECK below, on failure, sets Error::InvalidArgument on
> the context and returns the tuple `{max, max_indices}` unchanged):
> 1. Validate via `check_min_max_args(in, dim, keepdim, max, max_indices)`:
>    `dim` valid for `in`, `max_indices` dtype Long, `max` dtype canCast from
>    `in`'s dtype, shapes consistent with a reduction over `dim` with `keepdim`.
> 2. Resize `max` for the reduction over `dim` with `keepdim` per
>    `[spec:et:sem:reduce-util.torch.executor.native.resize-reduction-out-fn]`.
> 3. Resize `max_indices` to `max.sizes()`.
> 4. Check `in` and `max` share a dim order.
> 5. Check `max_indices` is default (contiguous) dim order.
> 6. Check `in` is default (contiguous) dim order.
> 7. Normalize `dim`: if `dim < 0`, `dim += in.dim()`.
> 8. Dispatch on `in.scalar_type()` over REALHBBF16 {Byte, Char, Short, Int,
>    Long, Half, Float, Double, Bool, BFloat16} (CTYPE). For each output index
>    `out_ix` (iterating output positions, possibly in parallel), reduce over the
>    `dim` slice of `in`: iterate the slice elements with their in-slice index
>    `ix`, keeping `(acc_val, acc_ix)`; update rule with NaN propagation — if
>    `acc_val` is not NaN and (the candidate `v` is NaN or `v > acc_val`), set
>    `acc_val = v`, `acc_ix = ix`. Thus a NaN anywhere in the slice makes the
>    result NaN with the index of the first NaN encountered, and ties keep the
>    earliest index (strict `>`). Write `acc_val` to `max[out_ix]` and `acc_ix`
>    to `max_indices[out_ix]`.
> 9. If the parallel reduction reports failure, ET_KERNEL_CHECK_MSG sets
>    Error::Internal (message "parallel_for failed").
> 10. Return `{max, max_indices}`.

> [spec:et:def:op-max.torch.executor.native.max-unary-out-fn]
> Tensor&

> [spec:et:sem:op-max.torch.executor.native.max-unary-out-fn]
> Reduces `in` over all elements to a single scalar maximum, written into the
> 0-dim tensor `out`.
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
> 5. Initialize `out[0]` to `lower_bound<CTYPE_OUT>()` (see
>    `[spec:et:sem:op-max.torch.executor.native.lower-bound-fn]`). Iterate every
>    element `i` in `[0, in.numel())` (row-major): let `val = (CTYPE_OUT)in[i]`;
>    if `val` is NaN, set `out[0] = val` and break (NaN short-circuits and
>    propagates); otherwise if `val > out[0]`, set `out[0] = val`.
> 6. Return `out`. (For an empty `in`, `out[0]` stays at `lower_bound`.)

