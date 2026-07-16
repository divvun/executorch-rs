# kernels/portable/cpu/pattern/pattern.h

> [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
> Tensor& unary_ufunc_realhbbf16_to_bool( bool (*fn_float)(float), bool (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
> Declaration of the unary-ufunc pattern that maps a tensor of any REALHBBF16
> dtype to a same-shaped Bool tensor. It takes two function pointers,
> `fn_float(float)->bool` and `fn_double(double)->bool` (callers pass the same
> predicate for both), plus `ctx`, input tensor `in`, and output `out`. The
> definition (in unary_ufunc_realhbbf16_to_bool.cpp) is specified by
> `[spec:et:sem:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]`:
> it resizes `out` to `in.sizes()`, requires `out` dtype == Bool and same dim
> order as `in`, then for each element casts the input to double (if the input
> ctype is double) or float (otherwise), applies the corresponding predicate,
> and stores the boolean result. The accompanying macro
> `DEFINE_UNARY_UFUNC_REALHBBF16_TO_BOOL(op_name, fn)` expands to a kernel
> `op_name(ctx, in, out)` that calls `internal::unary_ufunc_realhbbf16_to_bool(
> fn, fn, ctx, in, out)` (used e.g. for isnan/isinf-style predicates).

> [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
> Tensor& unary_ufunc_realhbbf16_to_floathbf16( float (*fn_float)(float), double (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
> Declaration of the unary-ufunc pattern that maps a tensor of any REALHBBF16
> dtype to a same-shaped floating-point (FLOATHBF16 = {Half, Float, Double,
> BFloat16}) tensor. It takes `fn_float(float)->float` and
> `fn_double(double)->double` (callers pass the same math function for both),
> plus `ctx`, `in`, `out`. The definition (in
> unary_ufunc_realhbbf16_to_floathbf16.cpp) is specified by
> `[spec:et:sem:unary-ufunc-realhbbf16-to-floathbf16.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]`:
> it requires `out` to be a floating type, resizes `out` to `in.sizes()`,
> requires same dim order, then per element casts the input to double (if input
> ctype is double) or float (otherwise), applies the math function, and casts the
> result to the output ctype. The macro
> `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(op_name, fn)` expands to a kernel
> `op_name(ctx, in, out)` calling `internal::unary_ufunc_realhbbf16_to_floathbf16(
> fn, fn, ctx, in, out)` (used for ops like erf/expm1/sqrt that produce a float
> output regardless of integral input).

> [spec:et:def:pattern.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
> Tensor& unary_ufunc_realhbf16( float (*fn_float)(float), double (*fn_double)(double), KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:pattern.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
> Declaration of the unary-ufunc pattern that maps a tensor of any REALHBF16
> dtype ({Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16}; note NO
> Bool) to an output tensor of the SAME shape and SAME dtype. It takes
> `fn_float(float)->float` and `fn_double(double)->double` (callers pass the same
> math function for both), plus `ctx`, `in`, `out`. The definition (in
> unary_ufunc_realhbf16.cpp) is specified by
> `[spec:et:sem:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn]`:
> it resizes `out` to `in.sizes()`, requires `in` and `out` to have the same
> shape AND same dtype and same dim order, then per element casts the input to
> double (double ctype) or float (otherwise), applies the math function, and
> casts back to the shared ctype. The macro `DEFINE_UNARY_UFUNC_REALHBF16(
> op_name, fn)` expands to a kernel `op_name(ctx, in, out)` calling
> `internal::unary_ufunc_realhbf16(fn, fn, ctx, in, out)` (used for
> same-dtype elementwise math such as ceil/floor/round).

