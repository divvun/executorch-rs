# kernels/portable/cpu/pattern/unary_ufunc_realhbf16.cpp

> [spec:et:def:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
> Tensor& unary_ufunc_realhbf16( float (*fn_float)(float), double (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
> Elementwise unary math function with SAME input/output dtype: reads a tensor
> `in` of any REALHBF16 dtype and writes `out` of the same shape and same dtype,
> where `out[i] = fn(in[i])`. `fn_float`/`fn_double` are the same math function
> specialized for float/double. Steps:
>
> 1. ET_KERNEL_CHECK_MSG `resize_tensor(out, in.sizes()) == Error::Ok` (resize
>    `out` to the shape of `in`); on failure set Error::InvalidArgument on `ctx`,
>    log "Failed to resize output tensor.", and return `out` unchanged.
> 2. ET_KERNEL_CHECK `tensors_have_same_shape_and_dtype(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged. This enforces that `out`
>    has exactly the same dtype as `in` (no output dtype dispatch).
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Dispatch on `in.scalar_type()` with `ET_SWITCH_REALHBF16_TYPES` — accepted
>    dtypes are REALHBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double,
>    BFloat16} (NO Bool); an unhandled dtype triggers the switch default failure
>    (Error::InvalidArgument, return `out`).
> 5. Call `apply_unary_map_fn` per
>    `[spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]` over
>    `in.numel()` elements in flat order: for each element `val_in` (typed
>    `CTYPE`, the shared in/out ctype), if `CTYPE` is `double` compute
>    `fn_double(static_cast<double>(val_in))`, otherwise compute
>    `static_cast<CTYPE>(fn_float(static_cast<float>(val_in)))`. Half, BFloat16,
>    and all integral inputs go through the float path (only Double uses the
>    double path). Reads from `in.const_data_ptr<CTYPE>()`, stores to
>    `out.mutable_data_ptr<CTYPE>()`.
>
> Iteration is over the flat element buffer. Empty tensors apply nothing.
> NaN/inf handling follows the supplied math function. Returns `out`.

