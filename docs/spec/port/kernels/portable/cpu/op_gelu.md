# kernels/portable/cpu/op_gelu.cpp

> [spec:et:def:op-gelu.torch.executor.native.gelu-out-fn]
> Tensor& gelu_out( KernelRuntimeContext& ctx, const Tensor& in, string_view approximate, Tensor& out)

> [spec:et:sem:op-gelu.torch.executor.native.gelu-out-fn]
> Applies the GELU activation elementwise to `in`, writing to `out`. The
> `approximate` string selects the formula ("none" = exact erf form, "tanh" =
> tanh approximation). Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_gelu_args(in, approximate, out)` must hold. This
>    validates (per `[spec:et:sem:activation-ops-util.check-gelu-args]`) that
>    `in` and `out` have the same dtype and that `approximate` is either "none"
>    or "tanh". On failure set Error::InvalidArgument and return `out`.
> 2. Resize `out` to `in.sizes()` via `resize_tensor`; on non-Ok set
>    Error::InvalidArgument and return `out`.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else
>    Error::InvalidArgument, return `out`.
> 4. Dispatch over `in.scalar_type()`, which must be in FLOATHBF16 =
>    {Half, Float, Double, BFloat16}; math is done in `CTYPE`. Apply the function
>    elementwise over `in.numel()` elements (contiguous, input→output 1:1) via
>    `apply_unary_map_fn`:
>    - Branch on `approximate`:
>      - "tanh": for each `x`: if `x == -inf` return 0; if `x == +inf` return
>        +inf; otherwise let `kBeta = sqrt(2)*2/sqrt(pi)*0.5` (i.e.
>        `sqrt(2/pi)`), `kKappa = 0.044715`, `x_cubed = x*x*x`,
>        `inner = kBeta*(x + kKappa*x_cubed)`, return
>        `0.5 * x * (1 + tanh(inner))`.
>      - "none": for each `x`: if `x == -inf` return 0; if `x == +inf` return
>        +inf; otherwise return `0.5 * x * (1 + erf(x * (1/sqrt(2))))` (using
>        `M_SQRT1_2` for `1/sqrt(2)`).
>      - any other value of `approximate`: unreachable here because
>        `check_gelu_args` already rejected it; the source's else-branch is a hard
>        ET_CHECK_MSG failure "Invalid approximation format: <approximate> for
>        gelu".
> 5. Return `out`.

