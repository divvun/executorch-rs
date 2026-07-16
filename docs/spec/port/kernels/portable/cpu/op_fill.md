# kernels/portable/cpu/op_fill.cpp

> [spec:et:def:op-fill.torch.executor.native.fill-scalar-out-fn]
> Tensor& fill_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn]
> Implements `fill.Scalar_out(a, b, *, out)`: fills every element of `out` (shaped like `a`) with the
> scalar `b`. Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. `a_type = a.scalar_type()`, `out_type = out.scalar_type()`.
> 2. ET_KERNEL_CHECK `a_type == out_type`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK_MSG `resize_tensor(out, a.sizes()) == Error::Ok`, else InvalidArgument (message
>    "Failed to resize output tensor.").
> 5. Dispatch over `a_type` in REALHBBF16 (`{Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>    BFloat16}`) as `CTYPE_A`. Cast the scalar: `opt_b_casted = check_overflow_scalar_cast<CTYPE_A>(b)`;
>    ET_KERNEL_CHECK `opt_b_casted.has_value()` (the scalar fits in `CTYPE_A` without overflow), else
>    InvalidArgument (returns void from the lambda, leaving `out` unchanged); `b_casted = opt_b_casted.value()`.
> 6. Write `b_casted` into every one of `out.numel()` elements (flat order) via `apply_unary_map_fn`
>    (the input `a` values are read but ignored, the mapping returns the constant `b_casted`).
> 7. Return `out`.

> [spec:et:def:op-fill.torch.executor.native.fill-tensor-out-fn]
> Tensor& fill_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn]
> Implements `fill.Tensor_out(a, b, *, out)` where `b` is a scalar (0-dim / single-element) tensor:
> fills every element of `out` (shaped like `a`) with the single value in `b`. Every failure path
> sets the error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `tensor_is_scalar(b)` (b has exactly one element), else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`, else InvalidArgument.
> 3. `a_type = a.scalar_type()`, `b_type = b.scalar_type()`, `out_type = out.scalar_type()`.
> 4. ET_KERNEL_CHECK `a_type == out_type`, else InvalidArgument.
> 5. ET_KERNEL_CHECK_MSG `resize_tensor(out, a.sizes()) == Error::Ok`, else InvalidArgument (message
>    "Failed to resize output tensor.").
> 6. Dispatch over `a_type` in REALHBBF16 as `CTYPE_A`. Nested dispatch over `b_type` in REALHBBF16 as
>    `CTYPE_B`: extract the single value of `b` into a `CTYPE_B` (`ET_EXTRACT_SCALAR_TENSOR`), then
>    `b_casted = static_cast<CTYPE_A>(b_val)`.
> 7. Write `b_casted` into every one of `out.numel()` elements (flat order) via `apply_unary_map_fn`
>    (mapping returns the constant `b_casted`, ignoring `a`).
> 8. Return `out`.

