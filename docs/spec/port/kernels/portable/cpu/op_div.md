# kernels/portable/cpu/op_div.cpp

> [spec:et:def:op-div.torch.executor.native.get-common-type-fn]
> ScalarType get_common_type(ScalarType a_type, ScalarType b_type)

> [spec:et:sem:op-div.torch.executor.native.get-common-type-fn]
> Determines the common (promoted) type for true division of two tensor dtypes `a_type`, `b_type`.
> Because true division always yields a floating result, integer/integer defaults to Float:
> 1. If either type is a complex type: return `promoteTypes(a_type, b_type)`.
> 2. Else if both are floating types: return `promoteTypes(a_type, b_type)`.
> 3. Else if only `a_type` is floating: return `a_type`.
> 4. Else if only `b_type` is floating: return `b_type`.
> 5. Otherwise (both integral/bool): return `ScalarType::Float`.

> [spec:et:def:op-div.torch.executor.native.div-out-fn]
> Tensor& div_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.div-out-fn]
> Elementwise true division `out = a / b` with broadcasting. Every failure path sets the error on
> `ctx` and returns `out` unchanged.
> 1. Compute `common_type = get_common_type(a.scalar_type(), b.scalar_type())` per
>    `[spec:et:sem:op-div.torch.executor.native.get-common-type-fn]`.
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `resize_to_broadcast_target_size(a, b, out) == Error::Ok` (resizes `out` to the
>    broadcast shape of `a` and `b`), else InvalidArgument.
> 4. If `common_type` is a complex type: dispatch over `common_type` in the complex set
>    (`ET_SWITCH_COMPLEX_TYPES`). Requires `a`, `b`, `out` to all be that complex `CTYPE` (direct
>    typed pointers, no broadcasting): for `i` in `[0, out.numel())`, `out[i] = a[i] / b[i]`. (This
>    path assumes matching shapes/elementwise; broadcasting is not applied for complex.)
> 5. Otherwise (real path): `compute_type = utils::get_compute_type(common_type)` (maps reduced
>    floating types to a wider compute float), dispatch over `compute_type` in FLOAT
>    (`{Float, Double}`) via `ET_SWITCH_FLOAT_TYPES`, and apply a two-input elementwise op
>    `(val_a, val_b) => val_a / val_b` in `CTYPE_COMPUTE`, with `a` and `b` loaded as REALHBBF16 and
>    the result written to `out` (out dtype set is FLOATHBF16), broadcasting `a` against `b` per
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`. IEEE division semantics apply
>    (x/0 -> ±inf, 0/0 -> NaN).
> 6. Return `out`.

> [spec:et:def:op-div.torch.executor.native.div-out-mode-fn]
> Tensor& div_out_mode( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, std::optional<std::string_view> mode, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.div-out-mode-fn]
> Elementwise division with rounding `mode` (`"trunc"`, `"floor"`, or none). Every failure path sets
> the error on `ctx` and returns `out` unchanged.
> 1. If `mode` has no value: delegate to `div_out(ctx, a, b, out)` per
>    `[spec:et:sem:op-div.torch.executor.native.div-out-fn]` and return its result.
> 2. ET_KERNEL_CHECK `mode == "trunc" || mode == "floor"`, else InvalidArgument.
> 3. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())` (standard promotion, NOT the
>    true-division get_common_type — integer/integer stays integral here).
> 4. ET_KERNEL_CHECK `canCast(common_type, out.scalar_type()) && common_type != Bool`, else InvalidArgument.
> 5. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`, else InvalidArgument.
> 6. ET_KERNEL_CHECK `resize_to_broadcast_target_size(a, b, out) == Error::Ok`, else InvalidArgument.
> 7. `compute_type = utils::get_compute_type(common_type)`. Set `mode_is_trunc = (mode == "trunc")` and
>    a `div_by_zero_error` flag (initially false).
> 8. Dispatch over `compute_type` in REAL (`{Byte, Char, Short, Int, Long, Float, Double}`) via
>    `ET_SWITCH_REAL_TYPES`. Apply a two-input elementwise op over `(val_a, val_b)` in `CTYPE_COMPUTE`,
>    `a`/`b` loaded as REALHBBF16, output set REALHBF16, broadcasting per
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`:
>    - If `CTYPE_COMPUTE` is an integral type (including bool) and `val_b == 0`: set
>      `div_by_zero_error = true` and return 0 for that element.
>    - `value = val_a / val_b`; if `mode_is_trunc`, `value = std::trunc(value)`; else (floor mode)
>      `value = utils::floor_divide(val_a, val_b)` per `[spec:et:sem:math-util.floor-divide]`.
>    - Return `value`.
> 9. ET_KERNEL_CHECK_MSG `!div_by_zero_error`, else InvalidArgument with message
>    "Div mode operation encountered integer division by zero".
> 10. Return `out`.

> [spec:et:def:op-div.torch.executor.native.div-scalar-mode-out-fn]
> Tensor& div_scalar_mode_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, std::optional<std::string_view> mode, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.div-scalar-mode-out-fn]
> Divides tensor `a` by scalar `b` with rounding `mode` (`"trunc"`, `"floor"`, or none). Every
> failure path sets the error on `ctx` and returns `out` unchanged.
> 1. If `mode` has no value: delegate to `div_scalar_out(ctx, a, b, out)` per
>    `[spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn]` and return its result.
> 2. ET_KERNEL_CHECK `mode == "trunc" || mode == "floor"`, else InvalidArgument.
> 3. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` (promotes the tensor dtype
>    against the scalar's category).
> 4. ET_KERNEL_CHECK `canCast(common_type, out.scalar_type()) && common_type != Bool`, else InvalidArgument.
> 5. ET_KERNEL_CHECK_MSG `!(isIntegralType(common_type, /*includeBool=*/true) && scalar_to<double>(b) == 0)`,
>    else InvalidArgument with message "Div mode operation encountered integer division by zero"
>    (integer division by a zero scalar is rejected up front).
> 6. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`, else InvalidArgument.
> 7. ET_KERNEL_CHECK `resize_tensor(out, a.sizes()) == Error::Ok`, else InvalidArgument.
> 8. `compute_type = utils::get_compute_type(common_type)`; `mode_is_trunc = (mode == "trunc")`.
> 9. Dispatch over `compute_type` in REAL (`{Byte, Char, Short, Int, Long, Float, Double}`) via
>    `ET_SWITCH_REAL_TYPES`. Cast the scalar `val_b = utils::scalar_to<CTYPE_COMPUTE>(b)`. Apply a
>    one-input elementwise op over `val_a` in `CTYPE_COMPUTE`, `a` loaded as REALHBBF16, output set
>    REALHBF16, per `[spec:et:sem:elementwise-util...apply-unitensor-elementwise-fn]`:
>    `value = val_a / val_b`; if `mode_is_trunc`, `value = std::trunc(value)`; else
>    `value = utils::floor_divide(val_a, val_b)`; return `value`.
> 10. Return `out`.

> [spec:et:def:op-div.torch.executor.native.div-scalar-out-fn]
> Tensor& div_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn]
> True division of tensor `a` by scalar `b`, always producing a floating result. Every failure path
> sets the error on `ctx` and returns `out` unchanged.
> 1. `common_type = isFloatingType(a.scalar_type()) ? a.scalar_type() : ScalarType::Float`.
> 2. ET_KERNEL_CHECK `common_type == out.scalar_type()`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `resize_tensor(out, a.sizes()) == Error::Ok`, else InvalidArgument.
> 5. `compute_type = utils::get_compute_type(common_type)`.
> 6. Dispatch over `compute_type` in FLOAT (`{Float, Double}`) via `ET_SWITCH_FLOAT_TYPES`. Cast the
>    scalar `val_b = utils::scalar_to<CTYPE_COMPUTE>(b)`. Apply a one-input elementwise op over
>    `val_a`, returning `val_a / val_b` in `CTYPE_COMPUTE`, with `a` loaded as REALHBBF16 and output
>    dtype set SAME_AS_COMMON (i.e. the compute/common float type == out dtype), per
>    `[spec:et:sem:elementwise-util...apply-unitensor-elementwise-fn]`. IEEE division semantics apply.
> 7. Return `out`.

