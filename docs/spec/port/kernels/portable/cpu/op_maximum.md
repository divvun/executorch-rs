# kernels/portable/cpu/op_maximum.cpp

> [spec:et:def:op-maximum.torch.executor.native.maximum-out-fn]
> Tensor& maximum_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-maximum.torch.executor.native.maximum-out-fn]
> Elementwise `max(a, b)` between two broadcastable tensors, written into `out`
> with NaN-propagating semantics.
>
> Steps:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. Require `canCast(common_type, out.scalar_type())` (ET_KERNEL_CHECK: on
>    failure sets Error::InvalidArgument and returns `out` unchanged).
> 3. Check `a`, `b`, `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 4. Resize `out` to the broadcast of `a` and `b` shapes per
>    `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
>    (ET_KERNEL_CHECK: InvalidArgument, returns `out`).
> 5. `compute_type = get_compute_type(common_type)`, constrained to REALB {Byte,
>    Char, Short, Int, Long, Float, Double, Bool}; dispatch selects CTYPE_COMPUTE.
> 6. For every output element over the broadcast shape, load the mapped `a` and
>    `b` elements (accepted input dtypes REALHBBF16 {Byte, Char, Short, Int,
>    Long, Half, Float, Double, Bool, BFloat16}), promote to CTYPE_COMPUTE, and
>    compute `max_override(val_a, val_b)`: returns the larger of the two, and if
>    either operand is NaN the result is NaN (NaN-propagating max). Store the
>    result into `out` (converted to `out`'s dtype, which is within REALHBBF16).
> 7. Return `out`.

