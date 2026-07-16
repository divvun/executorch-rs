# kernels/portable/cpu/op_ge.cpp

> [spec:et:def:op-ge.torch.executor.native.ge-scalar-out-fn]
> Tensor& ge_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-ge.torch.executor.native.ge-scalar-out-fn]
> Elementwise `a >= b` where `b` is a scalar, writing a boolean-valued result to
> `out`. Delegates to the shared comparison-scalar pattern
> `internal::comparison_scalar_out<std::greater_equal, "ge.Scalar_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`,
> which: computes `common_type = promote_type_with_scalar(a.scalar_type(), b)`;
> ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`; resizes `out` to
> `a.sizes()`; picks a compute dtype from the REALB set; converts the scalar once
> to the compute type; and for each element `val_a` (read from `a` over the
> REALHBBF16 input set) computes `std::greater_equal(val_a, val_b)` i.e.
> `val_a >= val_b`, storing the boolean into `out` (output over REALHBBF16). NaN
> operands compare false. Returns `out`; failed checks return `out` with
> Error::InvalidArgument on the context.

> [spec:et:def:op-ge.torch.executor.native.ge-tensor-out-fn]
> Tensor& ge_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-ge.torch.executor.native.ge-tensor-out-fn]
> Elementwise `a >= b` for two tensors with broadcasting, writing a boolean
> result to `out`. Delegates to
> `internal::comparison_tensor_out<std::greater_equal, "ge.Tensor_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`,
> which: computes `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`
> (and if the promoted type is floating and the two inputs differ in dtype,
> forces `common_type = Float`); ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`;
> resizes `out` to the broadcast shape via `resize_to_broadcast_target_size`;
> selects a compute dtype from the REALB set; and for each broadcast pair
> `(val_a, val_b)` (both read over REALHBBF16 and converted to compute type)
> computes `val_a >= val_b`, storing the boolean into `out`. NaN operands compare
> false. Returns `out`; failed checks return `out` with Error::InvalidArgument.

