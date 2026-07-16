# kernels/portable/cpu/op_ne.cpp

> [spec:et:def:op-ne.torch.executor.native.ne-scalar-out-fn]
> Tensor& ne_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-ne.torch.executor.native.ne-scalar-out-fn]
> Elementwise "not equal" of tensor `a` against scalar `b`, writing 1/0 (per `out`'s dtype) into `out` (same shape as `a`); returns `out`.
>
> Delegates entirely to `internal::comparison_scalar_out<std::not_equal_to, "ne.Scalar_out">(ctx, a, b, out)` per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`, with the comparison functor being `std::not_equal_to` (`val_a != val_b`). That helper: computes `common_type = promote_type_with_scalar(a.scalar_type(), b)`; ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)` (else InvalidArgument, return `out`); resizes `out` to `a.sizes()` (else InvalidArgument); derives `compute_type = get_compute_type(common_type)`; dispatches over REALB; converts `b` to `CTYPE_COMPUTE`; loads each `a` element (REALHBBF16), computes `val_a != val_b` in `CTYPE_COMPUTE`, and stores the boolean result to `out` in the REALHBBF16 store set (1 for true, 0 for false, cast to `out`'s dtype). Returns `out`.

> [spec:et:def:op-ne.torch.executor.native.ne-tensor-out-fn]
> Tensor& ne_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-ne.torch.executor.native.ne-tensor-out-fn]
> Elementwise "not equal" of tensors `a` and `b` with broadcasting, writing 1/0 (per `out`'s dtype) into `out`; returns `out`.
>
> Delegates entirely to `internal::comparison_tensor_out<std::not_equal_to, "ne.Tensor_out">(ctx, a, b, out)` per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`, with the comparison functor `std::not_equal_to` (`val_a != val_b`). That helper: computes `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`, and if `common_type` is floating and `a`/`b` have differing dtypes it forces `common_type = Float`; ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)` (else InvalidArgument, return `out`); resizes `out` to the broadcast shape of `a`,`b` via `resize_to_broadcast_target_size` (`[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`; else InvalidArgument); derives `compute_type = get_compute_type(common_type)`; dispatches over REALB; loads `a` and `b` (REALHBBF16) with broadcasting, computes `val_a != val_b` in `CTYPE_COMPUTE`, and stores the boolean result to `out` (1/0 cast to `out`'s dtype) in the REALHBBF16 store set. Returns `out`.

