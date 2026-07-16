# kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_bool.cpp

> [spec:et:def:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
> Tensor& unary_ufunc_realhbbf16_to_bool( bool (*fn_float)(float), bool (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
> Elementwise unary predicate: reads a tensor `in` of any REALHBBF16 dtype and
> writes a same-shaped Bool tensor `out`, where `out[i] = predicate(in[i])`.
> `fn_float` and `fn_double` are the same predicate specialized for float/double.
> Steps:
>
> 1. ET_KERNEL_CHECK_MSG `resize_tensor(out, in.sizes()) == Error::Ok` (resize
>    `out` to the shape of `in`); on failure set Error::InvalidArgument on `ctx`,
>    log "Failed to resize output tensor.", and return `out` unchanged.
> 2. ET_KERNEL_CHECK_MSG `out.scalar_type() == ScalarType::Bool`; on failure set
>    Error::InvalidArgument, log the actual dtype, and return `out` unchanged.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Read `in_type = in.scalar_type()` and dispatch with
>    `ET_SWITCH_REALHBBF16_TYPES` — accepted input dtypes are REALHBBF16 =
>    {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}; an
>    unhandled dtype triggers the switch default failure (Error::InvalidArgument,
>    return `out`).
> 5. Call `apply_unary_map_fn` per
>    `[spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]` over
>    `in.numel()` elements in flat order: for each input element `val_in` (typed
>    `CTYPE_IN`), if `CTYPE_IN` is `double` compute `static_cast<bool>(fn_double(
>    static_cast<double>(val_in)))`, otherwise compute `static_cast<bool>(
>    fn_float(static_cast<float>(val_in)))`. Half/BFloat16 inputs go through the
>    float path; all integral and Bool inputs also go through the float path.
>    Results are read from `in.const_data_ptr<CTYPE_IN>()` and stored to
>    `out.mutable_data_ptr<bool>()`.
>
> The iteration is over the flat element buffer (matching dim order, which is
> checked equal). Empty tensors (`numel()==0`) apply nothing and just return
> `out`. NaN handling follows the supplied predicate. Returns `out`.

