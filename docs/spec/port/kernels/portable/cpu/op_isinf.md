# kernels/portable/cpu/op_isinf.cpp

> [spec:et:def:op-isinf.torch.executor.native.isinf-double-fn]
> bool isinf_double(double x)

> [spec:et:sem:op-isinf.torch.executor.native.isinf-double-fn]
> Returns `std::isinf(x)` for a `double`: true iff `x` is positive or negative
> infinity; false for finite values and for NaN.

> [spec:et:def:op-isinf.torch.executor.native.isinf-float-fn]
> bool isinf_float(float x)

> [spec:et:sem:op-isinf.torch.executor.native.isinf-float-fn]
> Returns `std::isinf(x)` for a `float`: true iff `x` is positive or negative
> infinity; false for finite values and for NaN.

> [spec:et:def:op-isinf.torch.executor.native.isinf-out-fn]
> Tensor& isinf_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-isinf.torch.executor.native.isinf-out-fn]
> Elementwise `isinf`: writes to `out[i]` a boolean indicating whether `in[i]` is
> ±infinity. Delegates entirely to
> `internal::unary_ufunc_realhbbf16_to_bool(isinf_float, isinf_double, ctx, in, out)`
> per `[spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]`,
> which: accepts `in` of any REALHBBF16 dtype {Byte, Char, Short, Int, Long, Half,
> Float, Double, Bool, BFloat16}; requires `out` to be Bool; resizes `out` to
> `in`'s shape and checks matching dim order; and applies the predicate
> elementwise, using `isinf_float` for the Float compute path and `isinf_double`
> otherwise (integral and boolean inputs are widened to the floating compute type,
> so integer/bool elements always yield false since they can never be infinite).
> Returns `out`.

