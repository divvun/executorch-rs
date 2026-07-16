# kernels/optimized/cpu/op_gelu.cpp

> [spec:et:def:op-gelu.torch.executor.native.gelu-fn]
> void gelu( executorch::runtime::KernelRuntimeContext& context, const Tensor& input, string_view approximate, Tensor& output)

> [spec:et:sem:op-gelu.torch.executor.native.gelu-fn]
> Element-wise Gelu of `input` into `output`, both assumed contiguous, same
> shape, same dtype CTYPE. `lim = input.numel()`. Branch on `approximate`:
> - "tanh": for the vectorized prefix `[0, lim - lim % Vec::size())` step
>   `Vec::size()`, apply `vectorized_gelu_approximated_with_tanh(x)`; scalar tail
>   applies `scalar_gelu_approximated_with_tanh(in_data[i])`. The scalar tanh
>   gelu computes (per ATen Gelu.h): with kBeta = sqrt(2/pi) = M_SQRT2 *
>   M_2_SQRTPI * 0.5, kKappa = 0.044715, `x_cube = x*x*x`,
>   `inner = kBeta * (x + kKappa * x_cube)`, result = 0.5 * x * (1 + tanh(inner)).
> - "none": same loop structure applying `vectorized_gelu` / `scalar_gelu`; the
>   scalar exact gelu computes `0.5 * x * (1 + erf(x * M_SQRT1_2))` where
>   M_SQRT1_2 = 1/sqrt(2).
> - otherwise: fail InvalidArgument via ET_KERNEL_CHECK_MSG (returns from op),
>   message "Invalid approximation format: %.*s for gelu".
> Reduced float types (Half/BFloat16) promote per-operation to float and round
> back, matching ATen's scalar_gelu behavior. The vectorized and scalar paths
> compute the same value; the vector prefix is only an intrinsic acceleration.

> [spec:et:def:op-gelu.torch.executor.native.opt-gelu-out-fn]
> Tensor& opt_gelu_out( KernelRuntimeContext& context, const Tensor& input, string_view approximate, Tensor& out)

> [spec:et:sem:op-gelu.torch.executor.native.opt-gelu-out-fn]
> Out-variant of gelu. Steps:
> 1. ET_KERNEL_CHECK(check_gelu_args(input, approximate, out)) — same dtype,
>    non-Bool, approximate in {"tanh","none"}; else InvalidArgument, return out.
> 2. ET_KERNEL_CHECK(resize_tensor(out, input.sizes()) == Ok); else
>    InvalidArgument, return out.
> 3. Dispatch input dtype over FLOATHBF16 (Float, Double, Half, BFloat16) binding
>    CTYPE, call `gelu<CTYPE>(context, input, approximate, out)`.
> 4. Return out. `(void)context;` — context otherwise unused for logic beyond the
>    checks.
</content>
