# kernels/portable/cpu/op_floor_divide.cpp

> [spec:et:def:op-floor-divide.torch.executor.native.floor-divide-out-fn]
> Tensor& floor_divide_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-floor-divide.torch.executor.native.floor-divide-out-fn]
> Implements `floor_divide.out(a, b, *, out)`: elementwise `out = floor(a / b)` with broadcasting.
> Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())` (standard promotion; integers stay integral).
> 2. ET_KERNEL_CHECK `canCast(common_type, out.scalar_type()) && common_type != Bool`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `resize_to_broadcast_target_size(a, b, out) == Error::Ok`, else InvalidArgument.
> 5. `compute_type = utils::get_compute_type(common_type)`; init `div_by_zero_error = false`.
> 6. Dispatch over `compute_type` in REAL (`{Byte, Char, Short, Int, Long, Float, Double}`) via
>    `ET_SWITCH_REAL_TYPES`. Apply a two-input elementwise op over `(val_a, val_b)` in `CTYPE_COMPUTE`,
>    `a`/`b` loaded as REALHBBF16, output set REALHBF16, broadcasting per
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`:
>    - If `CTYPE_COMPUTE` is integral (including bool) and `val_b == 0`: set `div_by_zero_error = true`
>      and return 0 for that element.
>    - Otherwise return `utils::floor_divide(val_a, val_b)` per `[spec:et:sem:math-util.floor-divide]`
>      (floored division: truncation toward negative infinity; for floating types handles inf/NaN per
>      that util).
> 7. ET_KERNEL_CHECK_MSG `!div_by_zero_error`, else InvalidArgument with message "Floor divide
>    operation encountered integer division by zero".
> 8. Return `out`.

