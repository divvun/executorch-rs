# kernels/portable/cpu/op_round.cpp

> [spec:et:def:op-round.torch.executor.native.round-to-even-fn]
> inline CTYPE round_to_even(CTYPE a)

> [spec:et:sem:op-round.torch.executor.native.round-to-even-fn]
> Rounds a value `a` of ctype `CTYPE` to the nearest integer, breaking exact
> `.5` ties toward the nearest even integer (banker's rounding). Algorithm: if
> `a - std::floor(a) == 0.5` (fractional part exactly one half), return
> `std::round(a * 0.5) * 2.0` — halve, round-half-away-from-zero, then double,
> which yields the nearest even integer; otherwise return `std::round(a)`
> (round-half-away-from-zero, though for the non-tie branch the input is never a
> half-integer). Computed in `double` (literals `0.5`, `2.0`) and returned as
> `CTYPE`. NaN/±inf pass through unchanged: `round_to_even(NaN) == NaN`,
> `round_to_even(±inf) == ±inf`.

> [spec:et:def:op-round.torch.executor.native.round-out-fn]
> Tensor& round_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-round.torch.executor.native.round-out-fn]
> Elementwise round into `out`. Steps:
>
> - Resize `out` to `in.sizes()` via `resize_tensor`; on failure
>   `Error::InvalidArgument` (message "Failed to resize output tensor."), return
>   `out`.
> - ET_KERNEL_CHECK: `in`/`out` same shape AND same dtype; else
>   `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `out` dtype realhbf16 (`tensor_is_realhbf16_type`),
>   REALHBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16};
>   else `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Capture `in_scalar_type = in.scalar_type()`. Dispatch on it over REALHBF16.
>   Elementwise over `in.numel()` in flat order at the shared ctype CTYPE: if
>   `isIntegralType(in_scalar_type, includeBool=false)` is true (input is Byte,
>   Char, Short, Int, or Long) copy `val_in` through unchanged; otherwise write
>   `static_cast<CTYPE>(round_to_even<CTYPE>(val_in))` per
>   `[spec:et:sem:op-round.torch.executor.native.round-to-even-fn]` (round to
>   nearest, ties to even). So integer tensors are identity; floating tensors
>   round half-to-even; NaN/±inf pass through.
> - Returns `out`.

