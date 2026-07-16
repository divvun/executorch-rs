# kernels/portable/cpu/op_isnan.cpp

> [spec:et:def:op-isnan.torch.executor.native.isnan-double-fn]
> bool isnan_double(double x)

> [spec:et:sem:op-isnan.torch.executor.native.isnan-double-fn]
> Returns `std::isnan(x)` for a `double`: true iff `x` is NaN; false for all
> finite values and for ±infinity.

> [spec:et:def:op-isnan.torch.executor.native.isnan-float-fn]
> bool isnan_float(float x)

> [spec:et:sem:op-isnan.torch.executor.native.isnan-float-fn]
> Returns `std::isnan(x)` for a `float`: true iff `x` is NaN; false for all
> finite values and for ±infinity.

> [spec:et:def:op-isnan.torch.executor.native.isnan-out-fn]
> Tensor& isnan_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-isnan.torch.executor.native.isnan-out-fn]
> Elementwise `isnan`: writes to `out[i]` a boolean indicating whether `in[i]` is
> NaN. Delegates entirely to
> `internal::unary_ufunc_realhbbf16_to_bool(isnan_float, isnan_double, ctx, in, out)`
> per `[spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]`,
> which: accepts `in` of any REALHBBF16 dtype {Byte, Char, Short, Int, Long, Half,
> Float, Double, Bool, BFloat16}; requires `out` to be Bool; resizes `out` to
> `in`'s shape and checks matching dim order; and applies the predicate
> elementwise, using `isnan_float` on the Float compute path and `isnan_double`
> otherwise (integral and boolean inputs are widened to the floating compute type,
> so integer/bool elements always yield false since they can never be NaN).
> Returns `out`.

