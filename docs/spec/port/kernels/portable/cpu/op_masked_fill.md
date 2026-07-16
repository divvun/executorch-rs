# kernels/portable/cpu/op_masked_fill.cpp

> [spec:et:def:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn]
> Tensor& masked_fill_scalar_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mask, const Scalar& value, Tensor& out)

> [spec:et:sem:op-masked-fill.torch.executor.native.masked-fill-scalar-out-fn]
> Elementwise: `out = mask ? value : in`, broadcasting `in` and `mask` together;
> the scalar `value` replaces `in` positions where `mask` is true.
>
> Steps:
> 1. Validate arguments via `check_masked_fill_args(in, mask, value, out)`
>    (ET_KERNEL_CHECK: on failure sets Error::InvalidArgument and returns `out`
>    unchanged). This requires `mask` to be a Bool tensor, `in` and `out` to
>    share the same dtype, and the scalar `value`'s type to be castable into
>    `in`'s dtype.
> 2. Let `in_type = in.scalar_type()` and `val_type = get_scalar_dtype(value)`
>    (the scalar's natural dtype: Bool, Long, or Double).
> 3. Resize `out` to the broadcast of `in` and `mask` shapes per
>    `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
>    (ET_KERNEL_CHECK: InvalidArgument, returns `out`).
> 4. Check `in`, `mask`, `out` share a dim order (ET_KERNEL_CHECK:
>    InvalidArgument, returns `out`).
> 5. Dispatch on `in_type` over REALHBBF16 {Byte, Char, Short, Int, Long, Half,
>    Float, Double, Bool, BFloat16} (CTYPE), and on `val_type` over REAL plus
>    Bool {Byte, Char, Short, Int, Long, Float, Double, Bool} (CTYPE_VAL).
>    Extract the scalar into a CTYPE_VAL, then cast it to CTYPE, giving `val`.
> 6. Apply the binary elementwise map over the broadcast of `in` (as CTYPE) and
>    `mask` (as bool): for each output position, `out = mask_elem ? val :
>    in_elem`. Broadcasting index mapping is per the binary elementwise util.
> 7. Return `out`.

