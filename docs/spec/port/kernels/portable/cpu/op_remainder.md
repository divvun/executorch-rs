# kernels/portable/cpu/op_remainder.cpp

> [spec:et:def:op-remainder.torch.executor.native.remainder-scalar-out-fn]
> Tensor& remainder_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-remainder.torch.executor.native.remainder-scalar-out-fn]
> Elementwise `remainder(a, b)` where the divisor `b` is a scalar. Steps:
>
> - Compute `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)`
>   (see
>   `[spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]`).
> - ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())` AND `common_type
>   != Bool`; else `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK (integer div-by-zero guard): NOT (`common_type` is an
>   integral type including bool AND `utils::scalar_to<double>(b) == 0`); on
>   failure `Error::InvalidArgument` (message "Remainder operation encountered
>   integer division by zero"), return `out`.
> - ET_KERNEL_CHECK: `a`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Resize `out` to `a.sizes()`; on failure `Error::InvalidArgument`, return
>   `out`.
> - `compute_type = utils::get_compute_type(common_type)`; dispatch on it over
>   REAL = {Byte, Char, Short, Int, Long, Float, Double} as CTYPE_COMPUTE.
> - Read `val_b = utils::scalar_to<CTYPE_COMPUTE>(b)`. Apply the unitensor
>   elementwise fn (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`)
>   over `a` with supported input dtypes REALHBBF16 and output supported dtypes
>   REALHBF16: for each `val_a` at CTYPE_COMPUTE return
>   `utils::remainder_override(val_a, val_b)` (see
>   `[spec:et:sem:math-util.torch.executor.native.utils.remainder-override]`),
>   result carrying the sign of `val_b`. (Integer division by zero is already
>   ruled out by the earlier scalar check.)
> - Returns `out`.

> [spec:et:def:op-remainder.torch.executor.native.remainder-tensor-out-fn]
> Tensor& remainder_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-remainder.torch.executor.native.remainder-tensor-out-fn]
> Elementwise `remainder(a, b)` (Python/torch modulo: result has the sign of the
> divisor `b`) with broadcasting into `out`. Steps:
>
> - Compute `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> - ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())` AND `common_type
>   != Bool`; else `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `a`, `b`, `out` all have the same dim order
>   (`tensors_have_same_dim_order`); else `Error::InvalidArgument`, return `out`.
> - Resize `out` to the broadcast shape of `a` and `b`
>   (`resize_to_broadcast_target_size`); on failure `Error::InvalidArgument`,
>   return `out`.
> - Compute type `compute_type = utils::get_compute_type(common_type)` (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`).
>   Dispatch on it over REAL = {Byte, Char, Short, Int, Long, Float, Double} as
>   CTYPE_COMPUTE.
> - Track a `div_by_zero_error` flag, initially false. Apply the bitensor
>   elementwise fn (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]`)
>   with `a` and `b` both having supported input dtypes REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Half, Float, Double, Bool, BFloat16}, output supported
>   dtypes REALHBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double,
>   BFloat16}, broadcasting per the util. For each pair `(val_a, val_b)` at
>   CTYPE_COMPUTE: if CTYPE_COMPUTE is an integral type (including bool) and
>   `val_b == 0`, set `div_by_zero_error = true` and return `0`; otherwise return
>   `utils::remainder_override(val_a, val_b)` (see
>   `[spec:et:sem:math-util.torch.executor.native.utils.remainder-override]`),
>   which is `std::fmod`-based for floats and `((a % b) + b) % b`-style for
>   integers, always yielding a result with the sign of `b`.
> - After the loop, ET_KERNEL_CHECK: `!div_by_zero_error`; on failure
>   `Error::InvalidArgument` (message "Remainder operation encountered integer
>   division by zero"), return `out`.
> - Returns `out`.

