# kernels/optimized/cpu/op_exp.cpp

> [spec:et:def:op-exp.torch.executor.native.exp-data-fn]
> void exp_data( const CTYPE_IN* in_data, const size_t numel, CTYPE_OUT* out_data)

> [spec:et:sem:op-exp.torch.executor.native.exp-data-fn]
> Element-wise natural exponential over a flat buffer of `numel` elements.
> Two overloads selected by SFINAE on the (CTYPE_IN, CTYPE_OUT) pair:
> - Fast path (in == out type, and neither is Half/BFloat16): maps
>   `Vectorized<CTYPE_IN>::exp()` over the buffer via `at::vec::map` — i.e.
>   `out_data[i] = exp(in_data[i])` computed with CPU vector intrinsics.
> - Slow path (types differ, or either is Half/BFloat16): a plain loop
>   `for i in 0..numel { out_data[i] = std::exp(static_cast<CTYPE_OUT>(in_data[i])); }`
>   — input element is first cast to CTYPE_OUT, then `std::exp` (the double or
>   float overload matching CTYPE_OUT) is applied.
> Both compute the same mathematical value; the fast path is only an intrinsic
> acceleration for the same-type non-reduced-float case. Assumes contiguous
> buffers of length `numel`.

> [spec:et:def:op-exp.torch.executor.native.opt-exp-out-fn]
> Tensor& opt_exp_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-exp.torch.executor.native.opt-exp-out-fn]
> Out-variant of exp. Steps:
> 1. Resize `out` to `in.sizes()` for dynamic shape; on resize failure record
>    InvalidArgument and return `out` ("Failed to resize output tensor.").
> 2. Check `tensor_is_floating_type(out)`; else InvalidArgument, return `out`.
> 3. Dispatch input dtype over REALHBBF16 (all real + Half + Bool + BFloat16)
>    binding CTYPE_IN, and nested-dispatch output dtype over FLOATHBF16 (Float,
>    Double, Half, BFloat16) binding CTYPE_OUT; then call
>    `exp_data<CTYPE_IN, CTYPE_OUT>(in.const_data_ptr, in.numel(),
>    out.mutable_data_ptr)`.
> 4. Return `out`.
> `ctx` is otherwise unused (`(void)ctx;`).
</content>
