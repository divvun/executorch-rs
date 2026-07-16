# kernels/portable/cpu/op_leaky_relu.cpp

> [spec:et:def:op-leaky-relu.torch.executor.native.leaky-relu-out-fn]
> Tensor& leaky_relu_out( KernelRuntimeContext& ctx, const Tensor& in, const Scalar& negative_slope, Tensor& out)

> [spec:et:sem:op-leaky-relu.torch.executor.native.leaky-relu-out-fn]
> Computes elementwise LeakyReLU of `in` with slope `negative_slope`, writing
> into `out` (same dtype and shape as `in`).
>
> Steps:
> 1. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK_MSG: on failure sets
>    Error::InvalidArgument on the context and returns `out` unchanged, message
>    "Failed to resize output tensor.").
> 2. Check `in` and `out` have the same dim order (ET_KERNEL_CHECK:
>    InvalidArgument, returns `out`).
> 3. Let `in_type = in.scalar_type()`, `out_type = out.scalar_type()`; require
>    `in_type == out_type` (ET_KERNEL_CHECK: InvalidArgument, returns `out`).
> 4. Dispatch on `in_type` over FLOATHBF16 {Half, Float, Double, BFloat16};
>    any other dtype is unsupported (kernel error). Let CTYPE be the dispatched
>    type.
> 5. Cast `negative_slope` to CTYPE via a checked/overflow-aware scalar cast; if
>    the value does not fit (no value), ET_KERNEL_CHECK returns with
>    Error::InvalidArgument (returns `out` unchanged). Call the casted value
>    `slope`.
> 6. For every element index `i` in `[0, in.numel())` (row-major), read
>    `v = in[i]`; if `v >= 0` write `v` unchanged to `out[i]`, else write
>    `v * slope` to `out[i]`.
> 7. Return `out`.

