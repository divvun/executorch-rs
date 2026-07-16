# kernels/portable/cpu/op_logical_and.cpp

> [spec:et:def:op-logical-and.torch.executor.native.logical-and-fn]
> bool logical_and(bool a, bool b)

> [spec:et:sem:op-logical-and.torch.executor.native.logical-and-fn]
> Scalar boolean helper: returns `a && b` (logical AND of two bools). Used as the
> per-element functor for `logical_and_out`.

> [spec:et:def:op-logical-and.torch.executor.native.logical-and-out-fn]
> Tensor& logical_and_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-logical-and.torch.executor.native.logical-and-out-fn]
> Elementwise logical AND of two broadcastable tensors `a` and `b` into `out`.
> Delegates to the shared logical-op pattern with `fn = logical_and` (see
> `[spec:et:sem:op-logical-and.torch.executor.native.logical-and-fn]`) and op
> name "logical_and.out"; behavior is exactly
> `[spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn]`.
>
> Concretely: check `a`, `b`, `out` share a dim order (ET_KERNEL_CHECK: on
> failure sets Error::InvalidArgument and returns `out` unchanged); resize `out`
> to the broadcast of `a` and `b` shapes per
> `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
> (InvalidArgument on failure). For every output element (over the broadcast
> shape), load the mapped `a` and `b` elements (accepted input dtypes REALHBBF16
> {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}), convert
> each to `bool` (nonzero → true), compute `bool_a && bool_b`, and store the
> boolean into `out` (out dtype REALHBBF16; `true`→1). Returns `out`.

