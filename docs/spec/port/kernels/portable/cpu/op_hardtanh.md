# kernels/portable/cpu/op_hardtanh.cpp

> [spec:et:def:op-hardtanh.torch.executor.native.hardtanh-out-fn]
> Tensor& hardtanh_out( KernelRuntimeContext& ctx, const Tensor& in, const Scalar& min, const Scalar& max, Tensor& out)

> [spec:et:sem:op-hardtanh.torch.executor.native.hardtanh-out-fn]
> Clamps each element of `in` to `[min, max]` (both scalars), writing to `out`.
> Equivalent to `clamp(x, min, max)`. Returns `out`.
>
> Steps:
> 1. Resize `out` to `in.sizes()` via `resize_tensor`; on non-Ok fail with
>    ET_KERNEL_CHECK_MSG Error::InvalidArgument and message "Failed to resize
>    output tensor.", returning `out`.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else
>    Error::InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `in.scalar_type() == out.scalar_type()`; else
>    Error::InvalidArgument, return `out`.
> 4. Dispatch over `in.scalar_type()` in REALHBF16 = {Byte, Char, Short, Int,
>    Long, Half, Float, Double, BFloat16} (Bool excluded); math in `CTYPE`.
> 5. Cast `min` and `max` to `CTYPE` with overflow checking via
>    `utils::internal::check_overflow_scalar_cast<CTYPE>`. ET_KERNEL_CHECK each
>    optional has a value (empty → Error::InvalidArgument, return `out`).
> 6. Apply elementwise over `in.numel()` elements (contiguous, 1:1) via
>    `apply_unary_map_fn`: for each `val_in` compute
>    `utils::min_override(utils::max_override(val_in, min_casted), max_casted)`,
>    i.e. lower-clamp to `min` then upper-clamp to `max`. `min_override` /
>    `max_override` are NaN-propagating min/max helpers (per
>    `[spec:et:sem:math-util.min-override]` / `[spec:et:sem:math-util.max-override]`).
> 7. Return `out`.

