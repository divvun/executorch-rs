# kernels/portable/cpu/op_abs.cpp

> [spec:et:def:op-abs.torch.executor.native.abs-out-fn]
> Tensor& abs_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-abs.torch.executor.native.abs-out-fn]
> Implements `abs.out(in, out)`: element-wise absolute value (or complex
> magnitude). `ctx` unused for control flow.
>
> Steps:
> 1. `resize_tensor(out, in.sizes()) == Error::Ok` (`ET_KERNEL_CHECK_MSG`; on
>    failure sets context Error to `InvalidArgument`, message "Failed to resize
>    output tensor.", returns `out`).
> 2. Let `in_is_complex = isComplexType(in.scalar_type())`. Check
>    `in_is_complex || tensors_have_same_dtype(in, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`): for real input, `out` must have the same dtype as `in`;
>    for complex input this dtype-equality is not required (output is real).
> 3. `tensors_have_same_dim_order(in, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
>
> Dtype dispatch:
> - Complex input: `ET_SWITCH_COMPLEXH_TYPES` on `in` dtype (CTYPE_IN ∈
>   {ComplexHalf, ComplexFloat, ComplexDouble}) and `ET_SWITCH_FLOATH_TYPES` on
>   `out` dtype (CTYPE_OUT ∈ {Half, Float, Double}). Applies via
>   `apply_unary_map_fn<CTYPE_IN, CTYPE_OUT>` over `in.numel()` elements in flat
>   order, computing magnitude `sqrt(v.real_*v.real_ + v.imag_*v.imag_)` cast to
>   CTYPE_OUT.
> - Real input: `ET_SWITCH_REALHBF16_TYPES` on `in` dtype (CTYPE ∈ {Byte, Char,
>   Short, Int, Long, Half, Float, Double, BFloat16}; note: no Bool), same CTYPE
>   for `out`. Applies `apply_unary_map_fn` over `in.numel()` flat elements: for
>   each `v`, returns `-v` if `v < 0` else `v` (branch on `v < 0`, then
>   `static_cast<CTYPE>`). For unsigned Byte this always returns `v`.
>
> Returns `out`.

