# kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp

> [spec:et:def:unary-ufunc-realhbbf16-to-floathbf16.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
> Tensor& unary_ufunc_realhbbf16_to_floathbf16( float (*fn_float)(float), double (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:unary-ufunc-realhbbf16-to-floathbf16.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
> Elementwise unary math function: reads a tensor `in` of any REALHBBF16 dtype
> and writes a same-shaped floating-point tensor `out`, where `out[i] =
> fn(in[i])`. `fn_float` and `fn_double` are the same math function specialized
> for float/double. Steps:
>
> 1. ET_KERNEL_CHECK `tensor_is_floating_type(out)`; on failure set
>    Error::InvalidArgument on `ctx` and return `out` unchanged.
> 2. ET_KERNEL_CHECK_MSG `resize_tensor(out, in.sizes()) == Error::Ok` (resize
>    `out` to the shape of `in`); on failure set Error::InvalidArgument, log
>    "Failed to resize output tensor.", and return `out` unchanged.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Read `in_type = in.scalar_type()`, `out_type = out.scalar_type()`. Dispatch
>    input dtype with `ET_SWITCH_REALHBBF16_TYPES` (REALHBBF16 = {Byte, Char,
>    Short, Int, Long, Half, Float, Double, Bool, BFloat16}) and, nested inside,
>    output dtype with `ET_SWITCH_FLOATHBF16_TYPES` (FLOATHBF16 = {Half, Float,
>    Double, BFloat16}); an unhandled dtype in either switch triggers its default
>    failure (Error::InvalidArgument, return `out`).
> 5. Call `apply_unary_map_fn` per
>    `[spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]` over
>    `in.numel()` elements in flat order: for each input element `val_in` (typed
>    `CTYPE_IN`), if `CTYPE_IN` is `double` compute `static_cast<CTYPE_OUT>(
>    fn_double(static_cast<double>(val_in)))`, otherwise compute
>    `static_cast<CTYPE_OUT>(fn_float(static_cast<float>(val_in)))`. Half,
>    BFloat16, all integral, and Bool inputs go through the float path (only
>    Double inputs use the double path). Reads from
>    `in.const_data_ptr<CTYPE_IN>()`, stores to
>    `out.mutable_data_ptr<CTYPE_OUT>()`.
>
> Iteration is over the flat element buffer. Empty tensors apply nothing.
> NaN/inf handling follows the supplied math function (e.g. std::erf, std::sqrt).
> Returns `out`.

