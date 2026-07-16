# kernels/portable/cpu/op_gt.cpp

> [spec:et:def:op-gt.torch.executor.native.gt-scalar-out-fn]
> Tensor& gt_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-gt.torch.executor.native.gt-scalar-out-fn]
> Elementwise `a > b` where `b` is a scalar, writing a boolean result to `out`.
> Delegates to
> `internal::comparison_scalar_out<std::greater, "gt.Scalar_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`,
> which: computes `common_type = promote_type_with_scalar(a.scalar_type(), b)`;
> ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`; resizes `out` to
> `a.sizes()`; selects a compute dtype from the REALB set; converts the scalar
> once; and for each element `val_a` (read over REALHBBF16) computes
> `std::greater(val_a, val_b)` i.e. `val_a > val_b`, storing the boolean into
> `out`. NaN operands compare false. Returns `out`; failed checks return `out`
> with Error::InvalidArgument.

> [spec:et:def:op-gt.torch.executor.native.gt-tensor-out-fn]
> Tensor& gt_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-gt.torch.executor.native.gt-tensor-out-fn]
> Elementwise `a > b` for two tensors with broadcasting, writing a boolean result
> to `out`. Delegates to
> `internal::comparison_tensor_out<std::greater, "gt.Tensor_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`,
> which: computes `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`
> (forced to Float if the promotion is floating and the two input dtypes differ);
> ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`; resizes `out` to the
> broadcast shape via `resize_to_broadcast_target_size`; selects a compute dtype
> from REALB; and for each broadcast pair `(val_a, val_b)` computes `val_a > val_b`,
> storing the boolean into `out`. NaN operands compare false. Returns `out`;
> failed checks return `out` with Error::InvalidArgument.

