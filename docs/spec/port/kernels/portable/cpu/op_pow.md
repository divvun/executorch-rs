# kernels/portable/cpu/op_pow.cpp

> [spec:et:def:op-pow.torch.executor.native.pow-scalar-out-fn]
> Tensor& pow_Scalar_out( KernelRuntimeContext& ctx, const Scalar& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-pow.torch.executor.native.pow-scalar-out-fn]
> Elementwise `a ** b` where base `a` is a scalar and exponent `b` is a tensor; result into `out` (shape of `b`); returns `out`.
>
> Steps:
> 1. `common_type = utils::promote_type_with_scalar(b.scalar_type(), a)` (promote tensor `b`'s dtype against scalar `a`).
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type()) && common_type != Bool`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(b, out)`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: `resize_tensor(out, b.sizes())` == Ok; else InvalidArgument, return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)`; then force floating compute: if `compute_type != Float`, set it to `Double` (so integer/half/bfloat16 promote to Double; Float stays Float).
> 6. Dispatch on `compute_type` over FLOAT types (ET_SWITCH_FLOAT_TYPES = Float, Double only).
> 7. Convert scalar `a` to `CTYPE_COMPUTE` → `val_a`. Apply the unary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]` over `b` (loaded from REALHBBF16), computing `executorch::math::pow(val_a, val_b)` (= `std::pow`, IEEE-754 semantics: `pow(x,0)==1` incl. NaN base, `pow(NaN,y)`=NaN for y!=0, etc.) in `CTYPE_COMPUTE`, storing to `out` in the REALHBF16 store set (result cast to `out`'s dtype).
> 8. Return `out`.

> [spec:et:def:op-pow.torch.executor.native.pow-tensor-scalar-out-fn]
> Tensor& pow_Tensor_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-pow.torch.executor.native.pow-tensor-scalar-out-fn]
> Elementwise `a ** b` where base `a` is a tensor and exponent `b` is a scalar; result into `out` (shape of `a`); returns `out`.
>
> Steps:
> 1. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type()) && common_type != Bool`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, out)`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: `resize_tensor(out, a.sizes())` == Ok; else InvalidArgument, return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)`; if `compute_type != Float`, set it to `Double`.
> 6. Dispatch on `compute_type` over FLOAT types (Float, Double only).
> 7. Convert scalar `b` to `CTYPE_COMPUTE` → `val_b`. Apply the unary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]` over `a` (loaded from REALHBBF16), computing `executorch::math::pow(val_a, (decltype(val_a))(val_b))` (= `std::pow`, IEEE-754 semantics) in `CTYPE_COMPUTE`, storing to `out` in the REALHBF16 store set.
> 8. Return `out`.

> [spec:et:def:op-pow.torch.executor.native.pow-tensor-tensor-out-fn]
> Tensor& pow_Tensor_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-pow.torch.executor.native.pow-tensor-tensor-out-fn]
> Elementwise `a ** b` of tensors `a` (base) and `b` (exponent) with broadcasting; result into `out`; returns `out`.
>
> Steps:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type()) && common_type != Bool`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: resize `out` to the broadcast shape of `a`,`b` via `resize_to_broadcast_target_size` (`[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`) == Ok; else InvalidArgument, return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)`; if `compute_type != Float`, set it to `Double`.
> 6. Dispatch on `compute_type` over FLOAT types (Float, Double only).
> 7. Apply the binary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]` over broadcasted `a`,`b` (each loaded from REALHBBF16), computing `executorch::math::pow(val_a, val_b)` (= `std::pow`, IEEE-754 semantics) in `CTYPE_COMPUTE`, storing to `out` in the REALHBF16 store set.
> 8. Return `out`.

