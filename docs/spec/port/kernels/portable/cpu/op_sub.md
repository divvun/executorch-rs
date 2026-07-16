# kernels/portable/cpu/op_sub.cpp

> [spec:et:def:op-sub.torch.executor.native.sub-out-fn]
> Tensor& sub_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-sub.torch.executor.native.sub-out-fn]
> Computes `out = a - alpha * b` element-wise with broadcasting and type
> promotion. Implements `sub.out(Tensor self, Tensor other, *, Scalar alpha=1,
> Tensor(a!) out)`. Step by step:
>
> - `alpha_type = utils::get_scalar_dtype(alpha)`. ET_KERNEL_CHECK
>   `alpha_type != Bool`; on failure sets Error::InvalidArgument, returns `out`.
> - `common_type = promoteTypes(a.scalar_type(), b.scalar_type())` (PyTorch dtype
>   promotion of the two tensor dtypes).
> - ET_KERNEL_CHECK `canCast(common_type, out.scalar_type()) &&
>   canCast(alpha_type, common_type)`; else Error::InvalidArgument. So `out`'s
>   dtype must be able to hold the promoted type, and `alpha` must be castable
>   into the common type.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`; else
>   Error::InvalidArgument.
> - Resize `out` to the broadcast shape of `a` and `b` via
>   `resize_to_broadcast_target_size` (see `[spec:et:sem:broadcast-util...resize-to-broadcast-target-size-fn]`);
>   on failure Error::InvalidArgument.
> - `compute_type = utils::get_compute_type(common_type)` selects the computation
>   ctype (integers/bool compute in their type; the switch here is
>   ET_SWITCH_REAL_TYPES = {Byte, Char, Short, Int, Long, Float, Double}).
> - Convert `alpha` to `val_alpha` in CTYPE_COMPUTE. Apply the binary elementwise
>   function `val_a - (decltype(val_b))(val_alpha) * val_b` over `a` and `b` via
>   `apply_bitensor_elementwise_fn` (see `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`):
>   inputs `a` and `b` are each loaded/converted from their dtypes (input dtype
>   set REALHBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16})
>   to CTYPE_COMPUTE, the result is computed element-wise over the broadcast
>   shape, and stored into `out` (output dtype set REALHBF16, converting from
>   CTYPE_COMPUTE).
> - Returns `out`.

> [spec:et:def:op-sub.torch.executor.native.sub-scalar-out-fn]
> Tensor& sub_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-sub.torch.executor.native.sub-scalar-out-fn]
> Computes `out = a - alpha * b` element-wise where `b` and `alpha` are scalars.
> Implements `sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *,
> Tensor(a!) out)`. Step by step:
>
> - `alpha_type = utils::get_scalar_dtype(alpha)`. ET_KERNEL_CHECK
>   `alpha_type != Bool`; on failure Error::InvalidArgument, returns `out`.
> - `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` (promotes
>   the tensor dtype with the scalar `b`'s category).
> - ET_KERNEL_CHECK `common_type == out.scalar_type() && canCast(alpha_type,
>   common_type)`; else Error::InvalidArgument. Note `out`'s dtype must EXACTLY
>   equal the common type (stricter than the tensor-tensor variant).
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(a, out)`; else
>   Error::InvalidArgument.
> - Resize `out` to `a.sizes()`; on failure Error::InvalidArgument.
> - `compute_type = utils::get_compute_type(common_type)`; switch is
>   ET_SWITCH_REAL_TYPES = {Byte, Char, Short, Int, Long, Float, Double} giving
>   CTYPE_COMPUTE.
> - Convert `b` to `val_b` and `alpha` to `val_alpha` in CTYPE_COMPUTE, precompute
>   `val_alpha_times_b = val_alpha * val_b`. Apply the unary elementwise function
>   `val_a - (decltype(val_a))(val_alpha_times_b)` over `a` via
>   `apply_unitensor_elementwise_fn` (see `[spec:et:sem:elementwise-util...apply-unitensor-elementwise-fn]`):
>   `a` loaded from its dtype (input dtype set REALHBF16 = {Byte, Char, Short,
>   Int, Long, Half, Float, Double, BFloat16}) to CTYPE_COMPUTE, result stored to
>   `out` with output dtype set SAME_AS_COMMON (i.e. `out` dtype equals the common
>   type, matching the check above).
> - Returns `out`.

