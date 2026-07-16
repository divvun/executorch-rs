# kernels/portable/cpu/op_rsub.cpp

> [spec:et:def:op-rsub.torch.executor.native.rsub-scalar-out-fn]
> Tensor& rsub_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-rsub.torch.executor.native.rsub-scalar-out-fn]
> Reverse subtraction with scalar: elementwise `out = b - alpha * a`, where `a`
> is a tensor and `b`, `alpha` are scalars. Steps:
>
> - `alpha_type = utils::get_scalar_dtype(alpha)`. ET_KERNEL_CHECK: `alpha_type
>   != Bool`; else `Error::InvalidArgument`, return `out`.
> - `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` (see
>   `[spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]`).
> - ET_KERNEL_CHECK: `common_type == out.scalar_type()` AND `canCast(alpha_type,
>   common_type)` (note: output dtype must equal the common type exactly, not
>   merely be castable); else `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `a`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Resize `out` to `a.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - `compute_type = utils::get_compute_type(common_type)`; dispatch on it over
>   REAL = {Byte, Char, Short, Int, Long, Float, Double} as CTYPE_COMPUTE.
> - Read `val_b = utils::scalar_to<CTYPE_COMPUTE>(b)` and `val_alpha =
>   utils::scalar_to<CTYPE_COMPUTE>(alpha)`. Apply the unitensor elementwise fn
>   (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`)
>   over `a` with supported input dtypes REALHBF16 = {Byte, Char, Short, Int,
>   Long, Half, Float, Double, BFloat16} and output supported dtypes
>   SAME_AS_COMMON (the store dtype is the common/compute type). For each `val_a`
>   compute `val_b - val_alpha * val_a` at CTYPE_COMPUTE and store.
> - Returns `out`.

