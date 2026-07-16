# kernels/portable/cpu/op_lt.cpp

> [spec:et:def:op-lt.torch.executor.native.lt-scalar-out-fn]
> Tensor& lt_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-lt.torch.executor.native.lt-scalar-out-fn]
> Elementwise `a < b` where `a` is a tensor and `b` a `Scalar`, writing a
> boolean-comparison result into `out`. Delegates to the shared scalar
> comparison pattern with the C++ functor `std::less` and op name
> "lt.Scalar_out"; behavior is exactly
> `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`
> instantiated with the strict `<` comparison.
>
> Concretely: `common_type = promote_type_with_scalar(a.scalar_type(), b)`; check
> `a` and `out` share a dim order (ET_KERNEL_CHECK: on failure sets
> Error::InvalidArgument and returns `out` unchanged); resize `out` to
> `a.sizes()` (InvalidArgument on failure); `compute_type =
> get_compute_type(common_type)` constrained to REALB {Byte, Char, Short, Int,
> Long, Float, Double, Bool}. Cast `b` to the compute ctype; for every element
> `i` (row-major, `a.numel()` elements) load `a[i]` (input dtypes accepted
> REALHBBF16), promote to compute type, evaluate `a[i] < val_b`, and store the
> boolean into `out[i]` (out dtype REALHBBF16; `true`→1). Returns `out`.

> [spec:et:def:op-lt.torch.executor.native.lt-tensor-out-fn]
> Tensor& lt_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-lt.torch.executor.native.lt-tensor-out-fn]
> Elementwise `a < b` between two broadcastable tensors, writing a
> boolean-comparison result into `out`. Delegates to the shared tensor
> comparison pattern with the C++ functor `std::less` and op name
> "lt.Tensor_out"; behavior is exactly
> `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`
> instantiated with the strict `<` comparison.
>
> Concretely: `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`,
> forced to Float when the promotion is floating and `a`/`b` dtypes differ. Check
> `a`, `b`, `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument, return
> `out` unchanged); resize `out` to the broadcast of `a` and `b` shapes per
> `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`.
> `compute_type = get_compute_type(common_type)` in REALB. For each output
> element over the broadcast shape, load the mapped `a` and `b` elements (input
> dtypes accepted REALHBBF16), promote to compute type, evaluate `val_a < val_b`,
> store the boolean into `out` (out dtype REALHBBF16; `true`→1). Returns `out`.

