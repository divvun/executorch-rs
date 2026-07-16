# kernels/portable/cpu/op_sign.cpp

> [spec:et:def:op-sign.torch.executor.native.sign-out-fn]
> Tensor& sign_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-sign.torch.executor.native.sign-out-fn]
> Elementwise sign into `out`. Steps:
>
> - Resize `out` to `in.sizes()` via `resize_tensor`; on failure set
>   `Error::InvalidArgument` (message "Failed to resize output tensor.") and
>   return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - ET_KERNEL_CHECK: `in` and `out` must have same shape AND same dtype; else
>   `Error::InvalidArgument`, return `out`.
> - If `in.scalar_type() == Bool`: `memcpy` `in.nbytes()` bytes from `in` to
>   `out` verbatim (sign of a bool is itself: false→false, true→true). Return
>   `out`.
> - Otherwise dispatch on `in.scalar_type()` over REALHBF16 = {Byte, Char,
>   Short, Int, Long, Half, Float, Double, BFloat16}. Elementwise over
>   `in.numel()` in flat order at the shared ctype CTYPE: for each `val_in`, if
>   `utils::isnan_override(val_in)` is true (see
>   `[spec:et:sem:math-util.torch.executor.native.utils.isnan-override]`) write
>   `val_in` unchanged (NaN propagates); else write
>   `static_cast<CTYPE>((val_in > 0) - (val_in < 0))`, i.e. `+1` for positive,
>   `-1` for negative, `0` for zero. For unsigned Byte input the `-1` case
>   cannot occur (values are never `< 0`) so results are `0` or `1`.
> - Returns `out`.

