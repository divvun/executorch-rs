# kernels/portable/cpu/op_sigmoid.cpp

> [spec:et:def:op-sigmoid.torch.executor.native.sigmoid-out-fn]
> Tensor& sigmoid_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-sigmoid.torch.executor.native.sigmoid-out-fn]
> Elementwise logistic sigmoid `1 / (1 + exp(-x))` into `out`. Steps:
>
> - ET_KERNEL_CHECK: `out` must be a floating-point type
>   (`tensor_is_floating_type`: Half, Float, Double, BFloat16); else
>   `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Resize `out` to `in.sizes()`; on failure `Error::InvalidArgument` (message
>   "Failed to resize output tensor."), return `out`.
> - Determine the compute type: start from `in.scalar_type()` if it is a
>   floating type, otherwise `Float`; then pass through
>   `utils::get_compute_type` (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`),
>   which promotes Half/BFloat16 to Float for computation. Dispatch on this
>   compute type over FLOAT = {Float, Double} as CTYPE_COMPUTE.
> - Apply the unitensor elementwise fn (see
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`)
>   with input tensor `a = in` whose supported input dtypes are REALHBBF16 =
>   {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16} and
>   output supported dtypes FLOATHBF16 = {Half, Float, Double, BFloat16}. For
>   each element the loaded value is provided at CTYPE_COMPUTE `val_in`; compute
>   `one = (CTYPE_COMPUTE)1.0`, `out_val = one / (one +
>   executorch::math::exp(-val_in))`, and store cast to the output dtype. Inputs
>   are loaded/promoted to the compute type; integer/bool inputs are converted
>   to floating point before the sigmoid. Broadcasting and iteration order are
>   handled by the elementwise util at its cited rule.
> - Numeric edge cases follow `exp`: `sigmoid(+inf) == 1`, `sigmoid(-inf) == 0`,
>   `sigmoid(NaN) == NaN`, `sigmoid(0) == 0.5`.
> - Returns `out`.

