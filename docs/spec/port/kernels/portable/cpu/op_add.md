# kernels/portable/cpu/op_add.cpp

> [spec:et:def:op-add.torch.executor.native.add-out-fn]
> Tensor& add_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-add.torch.executor.native.add-out-fn]
> Implements `add.out(a, b, alpha, out)`: element-wise `out = a + alpha * b` with
> type promotion and broadcasting.
>
> Setup and validation, in order:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())` (PyTorch type
>    promotion).
> 2. `ET_KERNEL_CHECK` (→ `InvalidArgument`, returns `out`):
>    `canCast(common_type, out.scalar_type())` AND
>    `check_alpha_type(get_scalar_dtype(alpha), common_type)` — the promoted type
>    must be castable to `out`'s dtype, and `alpha`'s dtype must be compatible
>    with the common type.
> 3. `tensors_have_same_dim_order(a, b, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
> 4. `resize_to_broadcast_target_size(a, b, out) == Error::Ok` (`ET_KERNEL_CHECK`
>    → `InvalidArgument`) — resizes `out` to the broadcast of `a` and `b` shapes.
> 5. `compute_type = get_compute_type(common_type)`.
>
> Two dispatch paths:
> - If any of `a`, `b`, `out` is a complex dtype: additionally requires
>   `a.scalar_type() == b.scalar_type() == out.scalar_type()` (`ET_KERNEL_CHECK`
>   → `InvalidArgument`; mixed complex dtypes unsupported). Then
>   `ET_SWITCH_COMPLEXH_TYPES` on `out` dtype: `val_alpha = scalar_to<CTYPE>
>   (alpha)`, and `apply_binary_elementwise_fn<CTYPE,CTYPE,CTYPE>` computes
>   `val_a + val_alpha * val_b` over the broadcast of `a` and `b` into `out`.
> - Otherwise (real path): `ET_SWITCH_REALB_TYPES` on `compute_type` (CTYPE_COMPUTE
>   ∈ {Byte, Char, Short, Int, Long, Float, Double, Bool}). Extracts `val_alpha`
>   via `utils::extract_scalar(alpha, &val_alpha)` (`ET_KERNEL_CHECK` →
>   `InvalidArgument`; note the failure return here returns void from the lambda,
>   leaving `out` as-is). Then `utils::apply_bitensor_elementwise_fn<
>   CTYPE_COMPUTE, op_name, REALHBBF16>` applies `val_a + val_alpha * val_b`,
>   reading `a` and `b` each as `REALHBBF16`-supported dtypes, broadcasting, and
>   writing to `out` (values computed in CTYPE_COMPUTE then cast to `out` dtype).
>   The `REALHBBF16` set is {Byte, Char, Short, Int, Long, Half, Float, Double,
>   Bool, BFloat16}.
>
> Returns `out`.

> [spec:et:def:op-add.torch.executor.native.add-scalar-out-fn]
> Tensor& add_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn]
> Implements `add.Scalar_out(a, b, alpha, out)`: element-wise `out = a + alpha *
> b` where `b` and `alpha` are scalars.
>
> Setup and validation, in order:
> 1. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` —
>    promotes `a`'s dtype against the scalar `b`.
> 2. `ET_KERNEL_CHECK` (→ `InvalidArgument`, returns `out`): `common_type ==
>    out.scalar_type()` (exact equality, not just castable) AND
>    `check_alpha_type(get_scalar_dtype(alpha), common_type)`.
> 3. `tensors_have_same_dim_order(a, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
> 4. `resize_tensor(out, a.sizes()) == Error::Ok` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`) — `out` takes `a`'s shape (no broadcasting; scalar `b`).
> 5. `compute_type = get_compute_type(common_type)`.
>
> Dtype dispatch: `ET_SWITCH_REALB_TYPES` on `compute_type` (CTYPE_COMPUTE ∈
> {Byte, Char, Short, Int, Long, Float, Double, Bool}). Computes `val_b =
> scalar_to<CTYPE_COMPUTE>(b)`, extracts `val_alpha` via
> `utils::extract_scalar(alpha, &val_alpha)` (`ET_KERNEL_CHECK` →
> `InvalidArgument`; failure returns void from the lambda leaving `out` as-is),
> precomputes `val_alpha_times_b = val_alpha * val_b`. Then
> `utils::apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name,
> SAME_AS_COMMON>` applies `val_a + val_alpha_times_b` over each element of `a`
> (read as a `REALHBBF16` dtype), writing results (dtype `SAME_AS_COMMON`, i.e.
> the common/out dtype) into `out`.
>
> Returns `out`.

