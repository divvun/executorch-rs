# kernels/portable/cpu/op_le.cpp

> [spec:et:def:op-le.torch.executor.native.le-scalar-out-fn]
> Tensor& le_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn]
> Elementwise `a <= b` where `a` is a tensor and `b` a `Scalar`, writing a
> boolean-comparison result into `out`. Delegates to the shared scalar
> comparison pattern with the C++ functor `std::less_equal` and op name
> "le.Scalar_out"; behavior is exactly `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`
> instantiated with the `<=` comparison.
>
> Concretely: compute `common_type = promote_type_with_scalar(a.scalar_type(), b)`;
> check `a` and `out` have the same dim order (ET_KERNEL_CHECK: on failure sets
> Error::InvalidArgument on the context and returns `out` unchanged); resize
> `out` to `a.sizes()` (InvalidArgument on failure). Derive `compute_type =
> get_compute_type(common_type)`, which must be one of REALB {Byte, Char, Short,
> Int, Long, Float, Double, Bool}. Cast `b` to the compute ctype, then for every
> element index `i` (row-major over `a`, `a.numel()` elements) load `a[i]`
> converting from `a`'s dtype (accepted input dtypes REALHBBF16 {Byte, Char,
> Short, Int, Long, Half, Float, Double, Bool, BFloat16}) up to the compute type,
> evaluate `a[i] <= val_b`, and store the boolean result into `out[i]` (out dtype
> may be any REALHBBF16 type; `true`→1, `false`→0). Returns `out`.

> [spec:et:def:op-le.torch.executor.native.le-tensor-out-fn]
> Tensor& le_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn]
> Elementwise `a <= b` between two broadcastable tensors, writing a
> boolean-comparison result into `out`. Delegates to the shared tensor
> comparison pattern with the C++ functor `std::less_equal` and op name
> "le.Tensor_out"; behavior is exactly `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`
> instantiated with the `<=` comparison.
>
> Concretely: `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`, and
> if that promoted type is floating and `a` and `b` differ in dtype it is forced
> to Float. Check `a`, `b`, `out` share a dim order (ET_KERNEL_CHECK: on failure
> sets Error::InvalidArgument and returns `out` unchanged); resize `out` to the
> broadcast of `a` and `b` shapes per
> `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
> (InvalidArgument on failure). `compute_type = get_compute_type(common_type)`
> constrained to REALB {Byte, Char, Short, Int, Long, Float, Double, Bool}. For
> every output element (iterating the broadcast output shape), load the
> corresponding `a` and `b` elements (broadcasting index mapping), convert both
> up to the compute type (input dtypes accepted REALHBBF16), evaluate
> `val_a <= val_b`, and store the boolean into `out` (out dtype REALHBBF16;
> `true`→1). Returns `out`.

