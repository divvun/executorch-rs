# kernels/portable/cpu/pattern/comparison_op.h

> [spec:et:def:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]
> Tensor& comparison_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]
> Generic tensor-vs-scalar comparison pattern, templated on a `Comparison<T>`
> functor (e.g. `std::greater`, `std::less`, `std::equal_to`) and a compile-time
> `op_name` string. Applies `Comparison(val_a, b)` elementwise, writing a boolean
> result into `out`. Steps:
>
> 1. Compute `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)`
>    per `[spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]`
>    (half_to_float defaulting false).
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`: on failure set
>    Error::InvalidArgument on `ctx` and return `out` unchanged.
> 3. ET_KERNEL_CHECK `resize_tensor(out, a.sizes()) == Error::Ok` (resize `out`
>    to the shape of `a`); on failure set Error::InvalidArgument and return `out`.
> 4. Compute `compute_type = utils::get_compute_type(common_type)` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`.
> 5. Dispatch over `compute_type` with `ET_SWITCH_REALB_TYPES` — the accepted
>    compute dtypes are REALB = {Byte, Char, Short, Int, Long, Float, Double,
>    Bool}; an unhandled dtype triggers the switch's default failure path
>    (Error::InvalidArgument, return `out`).
> 6. Convert the scalar once via `val_b = utils::scalar_to<CTYPE_COMPUTE>(b)` per
>    `[spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-fn]`.
> 7. Call `utils::apply_unitensor_elementwise_fn<CTYPE_COMPUTE, op_name,
>    SupportedTensorDtypes::REALHBBF16>` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`:
>    reads each element `val_a` of `a` (input accepted from the REALHBBF16 set,
>    loaded/converted to `CTYPE_COMPUTE`), computes `Comparison<CTYPE_COMPUTE>()(
>    val_a, val_b)`, and stores the boolean (0/1) into `out` (output stored as
>    REALHBBF16; `out` is a Bool tensor for comparison ops). NaN operands compare
>    false for ordered comparisons per the C++ functor.
>
> Returns `out`. Any failed ET_KERNEL_CHECK returns `out` unchanged with
> Error::InvalidArgument recorded on `ctx`.

> [spec:et:def:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]
> Tensor& comparison_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]
> Generic tensor-vs-tensor comparison pattern with NumPy-style broadcasting,
> templated on a `Comparison<T>` functor and a compile-time `op_name` string.
> Applies `Comparison(val_a, val_b)` elementwise over the broadcast of `a` and
> `b`, writing a boolean result into `out`. Steps:
>
> 1. Compute `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`. Then
>    if `common_type` is a floating type AND `a.scalar_type() != b.scalar_type()`,
>    force `common_type = ScalarType::Float` (mixed-dtype floating comparisons
>    compute in Float, matching ATen which promotes to Float rather than Double).
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`: on failure set
>    Error::InvalidArgument on `ctx` and return `out` unchanged.
> 3. ET_KERNEL_CHECK `resize_to_broadcast_target_size(a, b, out) == Error::Ok`
>    per `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
>    (resizes `out` to the broadcast shape of `a` and `b`); on failure set
>    Error::InvalidArgument and return `out`.
> 4. Compute `compute_type = utils::get_compute_type(common_type)` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`.
> 5. Dispatch over `compute_type` with `ET_SWITCH_REALB_TYPES` — accepted compute
>    dtypes are REALB = {Byte, Char, Short, Int, Long, Float, Double, Bool};
>    unhandled dtype triggers the switch default failure (Error::InvalidArgument,
>    return `out`).
> 6. Call `utils::apply_bitensor_elementwise_fn<CTYPE_COMPUTE, op_name,
>    SupportedTensorDtypes::REALHBBF16>` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]`:
>    for each output index it loads the broadcast-corresponding elements of `a`
>    and `b` (both accepted from the REALHBBF16 set, converted to `CTYPE_COMPUTE`),
>    computes `Comparison<CTYPE_COMPUTE>()(val_a, val_b)`, and stores the boolean
>    (0/1) into `out` (stored as REALHBBF16; `out` is a Bool tensor). NaN operands
>    compare false for ordered comparisons per the C++ functor.
>
> Returns `out`. Any failed ET_KERNEL_CHECK returns `out` unchanged with
> Error::InvalidArgument recorded on `ctx`.

