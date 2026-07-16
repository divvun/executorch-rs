# kernels/portable/cpu/op_logit.cpp

> [spec:et:def:op-logit.torch.executor.native.logit-out-fn]
> Tensor& logit_out( KernelRuntimeContext& ctx, const Tensor& in, std::optional<double> eps, Tensor& out)

> [spec:et:sem:op-logit.torch.executor.native.logit-out-fn]
> Computes the elementwise logit `log(x / (1 - x))` of `in` into `out`, with an
> optional clamping epsilon `eps`.
>
> Steps:
> 1. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK: on failure sets
>    Error::InvalidArgument and returns `out` unchanged).
> 2. Check `in` and `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 3. Require `out` to be a floating type (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 4. Dispatch on `in.scalar_type()` (CTYPE_IN) over REALHBBF16 {Byte, Char,
>    Short, Int, Long, Half, Float, Double, Bool, BFloat16}, and independently on
>    `out.scalar_type()` (CTYPE_OUT) over FLOAT {Float, Double} only (Half and
>    BFloat16 outputs are not accepted by this op even though step 3 admits them;
>    an unsupported out dtype triggers a kernel error).
> 5. For each element index `i` in `[0, in.numel())` (row-major): let
>    `xi = (CTYPE_OUT)in[i]`. If `eps` has a value: clamp `xi` to `[eps, 1-eps]`
>    (if `xi < eps` set `xi = eps`; else if `xi > 1 - eps` set `xi = 1 - eps`).
>    If `eps` is absent, no clamping. Then compute
>    `out[i] = (CTYPE_OUT)log(xi / (1 - xi))` (natural log). Without clamping,
>    `xi == 1` yields +inf, `xi == 0` yields -inf, `xi` outside `[0,1]` yields
>    NaN, per `std::log` of the ratio.
> 6. Return `out`.

