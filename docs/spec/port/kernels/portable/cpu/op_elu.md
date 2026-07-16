# kernels/portable/cpu/op_elu.cpp

> [spec:et:def:op-elu.torch.executor.native.elu-out-fn]
> Tensor& elu_out( KernelRuntimeContext& ctx, const Tensor& in, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, Tensor& out)

> [spec:et:sem:op-elu.torch.executor.native.elu-out-fn]
> Elementwise ELU with parameters `alpha`, `scale`, `input_scale`:
> `elu(x) = x <= 0 ? alpha*scale*(exp(input_scale*x) - 1) : scale*x`. Every failure path sets the
> error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `tensors_have_same_dtype(in, out)`, else InvalidArgument.
> 2. ET_KERNEL_CHECK `resize_tensor(out, in.sizes()) == Error::Ok`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `tensor_is_floating_type(in)`, else InvalidArgument.
> 5. ET_KERNEL_CHECK `tensors_have_same_dtype(in, out)` (checked again), else InvalidArgument.
> 6. Dispatch over `in.scalar_type()` in FLOATHBF16 (`{Half, Float, Double, BFloat16}`) via
>    `ET_SWITCH_FLOATHBF16_TYPES`. Choose math type `MathT`: `float` if `CTYPE` is a reduced
>    floating-point type (Half/BFloat16), else `CTYPE` (i.e. Float or Double). Cast the scalars to
>    `MathT`: `math_alpha`, `math_scale`, `math_input_scale`; precompute `negcoef = math_alpha * math_scale`.
> 7. Apply a one-input elementwise op over `x` (in dtype set FLOATHBF16, out dtype SAME_AS_COMMON ==
>    same dtype), per `[spec:et:sem:elementwise-util...apply-unitensor-elementwise-fn]`, computing in
>    `MathT`: if `MathT(x) <= 0`, result = `std::expm1(MathT(x) * math_input_scale) * negcoef`; else
>    result = `MathT(x) * math_scale`. The `MathT` result is cast back to the output `CTYPE`.
> 8. Return `out`.

