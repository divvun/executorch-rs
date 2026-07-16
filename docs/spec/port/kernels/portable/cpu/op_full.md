# kernels/portable/cpu/op_full.cpp

> [spec:et:def:op-full.torch.executor.native.full-out-fn]
> Tensor& full_out( KernelRuntimeContext& ctx, const IntArrayRef sizes, const Scalar& fill_value, Tensor& out)

> [spec:et:sem:op-full.torch.executor.native.full-out-fn]
> Fills the output tensor of shape `sizes` with a single scalar `fill_value`.
> Returns `out`. The context is unused for behavior beyond error reporting.
>
> Steps:
> 1. Read `out_type = out.scalar_type()`.
> 2. Resize `out` to `sizes` via `resize_tensor`. On non-Ok, ET_KERNEL_CHECK_MSG
>    fails with Error::InvalidArgument and message "Failed to resize output
>    tensor." and returns `out` unchanged.
> 3. Dispatch over `out_type`, which must be one of the REALHBBF16 set:
>    {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}. An
>    unsupported dtype triggers the switch's failure path (Error::InvalidArgument).
> 4. Cast `fill_value` to the output ctype with overflow checking via
>    `utils::internal::check_overflow_scalar_cast<CTYPE_OUT>`; this returns an
>    optional that is empty when the scalar cannot be represented in the target
>    type (out of range / lossy integer). ET_KERNEL_CHECK: if the optional has no
>    value, set Error::InvalidArgument and return `out`.
> 5. Write the casted value into every element of `out.mutable_data_ptr<CTYPE_OUT>()`
>    for indices `0 .. out.numel()-1` (contiguous fill). A zero-numel output is a
>    valid no-op fill.
> 6. Return `out`.

