# kernels/portable/cpu/op_logical_xor.cpp

> [spec:et:def:op-logical-xor.torch.executor.native.logical-xor-fn]
> bool logical_xor(bool a, bool b)

> [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-fn]
> Scalar boolean helper: returns `a != b`, i.e. the exclusive-or of two bools
> (true iff exactly one operand is true). Used as the per-element functor for
> `logical_xor_out`.

> [spec:et:def:op-logical-xor.torch.executor.native.logical-xor-out-fn]
> Tensor& logical_xor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-out-fn]
> Elementwise logical XOR of two broadcastable tensors `a` and `b` into `out`.
> Delegates to the shared logical-op pattern with `fn = logical_xor` (see
> `[spec:et:sem:op-logical-xor.torch.executor.native.logical-xor-fn]`) and op
> name "logical_xor.out"; behavior is exactly
> `[spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn]`.
>
> Concretely: check `a`, `b`, `out` share a dim order (ET_KERNEL_CHECK:
> InvalidArgument, returns `out` unchanged); resize `out` to the broadcast of `a`
> and `b` shapes per
> `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`.
> For every output element (over the broadcast shape), load the mapped `a` and
> `b` elements (accepted input dtypes REALHBBF16), convert each to `bool`
> (nonzero → true), compute `bool_a != bool_b`, and store the boolean into `out`
> (out dtype REALHBBF16; `true`→1). Returns `out`.

