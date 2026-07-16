# kernels/portable/cpu/op_fmod.cpp

> [spec:et:def:op-fmod.torch.executor.native.fmod-scalar-out-fn]
> Tensor& fmod_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-fmod.torch.executor.native.fmod-scalar-out-fn]
> Computes the elementwise floating-point remainder `a % b` where `b` is a
> scalar, writing to `out`. Returns `out`.
>
> Steps:
> 1. Compute `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)`:
>    the promotion of the tensor dtype with the scalar's category (a Python
>    integer scalar does not force a floating result; a floating scalar does).
> 2. ET_KERNEL_CHECK: require `canCast(common_type, out.scalar_type())` AND
>    `common_type != Bool`. On failure set Error::InvalidArgument on the context
>    and return `out` unchanged.
> 3. ET_KERNEL_CHECK_MSG (integer-divide-by-zero guard): fail with
>    Error::InvalidArgument and message "Fmod operation encountered integer
>    division by zero" if `common_type` is an integral type (Bool counted as
>    integral) AND the scalar `b`, converted to double, equals 0. Returns `out`
>    unchanged on failure.
> 4. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, out)` must hold; else
>    Error::InvalidArgument, return `out`.
> 5. Resize `out` to `a.sizes()` via `resize_tensor`; on non-Ok set
>    Error::InvalidArgument and return `out`.
> 6. Determine the compute dtype: `compute_type = utils::get_compute_type(common_type)`,
>    then if `compute_type != Float`, force it to Double. So all math happens in
>    Float when the common type is Float, otherwise in Double.
> 7. Dispatch over the compute dtype (Float or Double). Convert the scalar once
>    to the compute type as `val_b`. Apply the unary elementwise function per
>    `[spec:et:sem:elementwise-util...apply-unitensor-elementwise-fn]`: for each
>    element `val_a` of `a` (input read as any of the REALHBBF16 dtypes and
>    converted to the compute type), compute `executorch::math::fmod(val_a, val_b)`
>    (C `fmod`: result has the sign of the dividend `val_a`; if `val_b` is 0 the
>    IEEE fmod yields NaN — no integer guard applies here because the compute type
>    is always floating), and store into `out` (whose dtype must be one of
>    REALHBF16: real integer types plus Half, Float, Double, BFloat16 — note Bool
>    is excluded from the output set).
> 8. Return `out`.

> [spec:et:def:op-fmod.torch.executor.native.fmod-tensor-out-fn]
> Tensor& fmod_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-fmod.torch.executor.native.fmod-tensor-out-fn]
> Computes the elementwise floating-point remainder `a % b` for two tensors with
> broadcasting, writing to `out`. Returns `out`.
>
> Steps:
> 1. Compute `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. ET_KERNEL_CHECK: require `canCast(common_type, out.scalar_type())` AND
>    `common_type != Bool`. On failure set Error::InvalidArgument and return
>    `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)` must hold; else
>    Error::InvalidArgument, return `out`.
> 4. Resize `out` to the broadcast of `a` and `b` shapes via
>    `resize_to_broadcast_target_size` per `[spec:et:sem:broadcast-util.resize-to-broadcast-target-size]`;
>    on non-Ok set Error::InvalidArgument and return `out`.
> 5. Determine compute dtype: `compute_type = utils::get_compute_type(common_type)`,
>    then if it is not Float, force it to Double.
> 6. Initialize `div_by_zero_error = false`.
> 7. Dispatch over the compute dtype (Float or Double). Apply the bitensor
>    elementwise function with broadcasting per
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`: `a` and `b`
>    are each read as any REALHBBF16 dtype and converted to the compute type; for
>    each broadcast pair `(val_a, val_b)`:
>    - Because the compute type is always floating here, the integral branch is
>      never taken (`div_by_zero_error` stays false in practice), but the literal
>      logic is: if `CTYPE_COMPUTE` is an integral type (including Bool) and
>      `val_b == 0`, set `div_by_zero_error = true` and return 0.
>    - Otherwise return `std::fmod(val_a, val_b)` (C `fmod`: sign of the dividend
>      `val_a`; `val_b == 0` yields NaN; `val_a` non-finite propagates per IEEE).
>    Results are written to `out` (output dtype restricted to REALHBF16, i.e.
>    integer reals plus Half/Float/Double/BFloat16, Bool excluded).
> 8. ET_KERNEL_CHECK_MSG: if `div_by_zero_error` is true, fail with
>    Error::InvalidArgument and message "Fmod operation encountered integer
>    division by zero", returning `out`.
> 9. Return `out`.

