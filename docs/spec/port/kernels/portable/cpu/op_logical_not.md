# kernels/portable/cpu/op_logical_not.cpp

> [spec:et:def:op-logical-not.torch.executor.native.logical-not-out-fn]
> Tensor&

> [spec:et:sem:op-logical-not.torch.executor.native.logical-not-out-fn]
> Elementwise logical NOT of `in` into `out` (same shape; `out` dtype may differ
> from `in`).
>
> Steps:
> 1. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK_MSG: on failure sets
>    Error::InvalidArgument and returns `out` unchanged, message "Failed to
>    resize output tensor.").
> 2. Check `in` and `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 3. Check `in` and `out` have the same shape (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`). No broadcasting.
> 4. Dispatch on `in.scalar_type()` (CTYPE_IN) and independently on
>    `out.scalar_type()` (CTYPE_OUT), both over REALHBBF16 {Byte, Char, Short,
>    Int, Long, Half, Float, Double, Bool, BFloat16}.
> 5. For each element index `i` in `[0, in.numel())` (row-major): read
>    `in[i]`, convert it to `bool` (nonzero → true), negate it (`!`), and store
>    the result cast to CTYPE_OUT into `out[i]` (true→1, false→0).
> 6. Return `out`.

