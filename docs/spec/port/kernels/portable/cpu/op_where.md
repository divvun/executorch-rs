# kernels/portable/cpu/op_where.cpp

> [spec:et:def:op-where.torch.executor.native.where-out-fn]
> Tensor& where_out( KernelRuntimeContext& ctx, const Tensor& cond, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-where.torch.executor.native.where-out-fn]
> Entry point for `where.self_out(cond, a, b, *, out)`. Elementwise select:
> `out = cond ? a : b`, with broadcasting across `cond`, `a`, `b`. Returns
> `out`.
>
> 1. Compute `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`
>    (the promoted dtype of the two value tensors; `cond` does not participate).
> 2. `ET_KERNEL_CHECK(common_type == out.scalar_type(), InvalidArgument)`: the
>    output dtype must equal the promoted type. On failure sets InvalidArgument
>    on `ctx` and returns `out`.
> 3. `ET_KERNEL_CHECK(tensors_have_same_dim_order(cond, a, b, out),
>    InvalidArgument)`.
> 4. `ET_KERNEL_CHECK(resize_to_broadcast_target_size(a, b, cond, out) ==
>    Error::Ok, InvalidArgument)`: resizes `out` to the broadcast shape of `a`,
>    `b`, and `cond`.
> 5. `compute_type = utils::get_compute_type(common_type)` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`.
> 6. Dispatch `CTYPE_COMPUTE` over `compute_type` with `ET_SWITCH_REALB_TYPES`
>    (real numeric types plus Bool: Byte, Char, Short, Int, Long, Float, Double,
>    Bool — excludes Half, BFloat16, complex). For the selected compute type,
>    call
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn]`
>    with the ternary op `(val_a, val_b, val_c) -> val_c ? val_a : val_b`, output
>    dtype policy `SupportedTensorDtypes::SAME_AS_COMMON`, and per-input accepted
>    dtype sets: `a` = `REALHBBF16`, `b` = `REALHBBF16`, `cond` =
>    `BOOL_OR_BYTE`. That util loads each input (broadcasting and converting to
>    `CTYPE_COMPUTE`), applies the op, and writes the result to `out`. Here the
>    predicate is `cond` (`val_c`, nonzero → true selects `a`, else `b`).
> 7. Return `out`.

