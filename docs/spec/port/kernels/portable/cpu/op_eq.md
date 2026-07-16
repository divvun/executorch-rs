# kernels/portable/cpu/op_eq.cpp

> [spec:et:def:op-eq.torch.executor.native.eq-scalar-out-fn]
> Tensor& eq_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-eq.torch.executor.native.eq-scalar-out-fn]
> Implements `eq.Scalar_out(a, b, *, out)`: elementwise `out = (a == b)`. Delegates to the shared
> comparison pattern `internal::comparison_scalar_out<std::equal_to, "eq.Scalar_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]`.
> That pattern: computes `common_type = promote_type_with_scalar(a.scalar_type(), b)`; ET_KERNEL_CHECK
> same dim order for `(a, out)`; resizes `out` to `a.sizes()`; computes `compute_type =
> get_compute_type(common_type)`; dispatches over REALB (`{Byte, Char, Short, Int, Long, Float,
> Double, Bool}`); casts `val_b = scalar_to<CTYPE_COMPUTE>(b)`; applies elementwise `std::equal_to`
> comparing each `val_a` against `val_b`, `a` loaded as REALHBBF16, writing a boolean-valued result
> to `out` (out dtype set REALHBBF16, typically Bool). Returns `out`; failures set the error on `ctx`
> and return `out` unchanged.

> [spec:et:def:op-eq.torch.executor.native.eq-tensor-out-fn]
> Tensor& eq_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-eq.torch.executor.native.eq-tensor-out-fn]
> Implements `eq.Tensor_out(a, b, *, out)`: elementwise `out = (a == b)` with broadcasting. Delegates
> to the shared comparison pattern `internal::comparison_tensor_out<std::equal_to, "eq.Tensor_out">(ctx, a, b, out)`
> per `[spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]`.
> That pattern: computes `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`, and if it is
> a floating type while `a.scalar_type() != b.scalar_type()` overrides it to Float; ET_KERNEL_CHECK
> same dim order for `(a, b, out)`; resizes `out` to the broadcast target size of `a` and `b`;
> computes `compute_type = get_compute_type(common_type)`; dispatches over REALB (`{Byte, Char,
> Short, Int, Long, Float, Double, Bool}`); applies elementwise `std::equal_to(val_a, val_b)` with
> `a` and `b` loaded as REALHBBF16 and broadcast together, writing boolean results to `out` (out
> dtype set REALHBBF16, typically Bool). Returns `out`; failures set the error on `ctx` and return
> `out` unchanged.

