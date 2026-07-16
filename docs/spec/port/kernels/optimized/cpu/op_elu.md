# kernels/optimized/cpu/op_elu.cpp

> [spec:et:def:op-elu.torch.executor.native.elu-fn]
> void elu( KernelRuntimeContext& context, const Tensor& input, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, Tensor& out)

> [spec:et:sem:op-elu.torch.executor.native.elu-fn]
> Element-wise ELU of `input` into `out`, contiguous, same shape/dtype CTYPE.
> MathT = float when CTYPE is a reduced floating type (Half/BFloat16), else
> CTYPE. Convert the three scalars once to MathT:
> `math_alpha = scalar_to<MathT>(alpha)`, `math_scale`, `math_input_scale`.
> Build a scalar elementwise func and a vectorized func (ATen Elu.h):
> negcoef = math_alpha * math_scale; for each element x, with xm = MathT(x):
>   result = xm <= 0 ? expm1(xm * math_input_scale) * negcoef : xm * math_scale,
> cast back to CTYPE. Then run over `[0, out.numel())` via parallel_for with
> GRAIN_SIZE, splitting each [begin,end) chunk into a scalar prologue
> `[begin, vectorized_begin)` (vectorized_begin = begin rounded up to a
> Vec::size() multiple), a main vectorized loop `[vectorized_begin,
> vectorized_end)` (vectorized_end = end rounded down to a Vec::size() multiple),
> and a scalar epilogue `[vectorized_end, end)`. All three apply the identical
> ELU formula; the vector loop is only an intrinsic acceleration.

> [spec:et:def:op-elu.torch.executor.native.opt-elu-out-fn]
> Tensor& opt_elu_out( KernelRuntimeContext& ctx, const Tensor& in, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, Tensor& out)

> [spec:et:sem:op-elu.torch.executor.native.opt-elu-out-fn]
> Out-variant of ELU. Checks (each ET_KERNEL_CHECK InvalidArgument, return out):
> tensors_have_same_dtype(in, out); resize_tensor(out, in.sizes()) == Ok;
> tensors_have_same_dim_order(in, out); tensor_is_floating_type(in);
> tensors_have_same_dtype(in, out) (checked twice, matching the source). Then
> dispatch in.scalar_type() over FLOATHBF16 (Float, Double, Half, BFloat16)
> binding CTYPE and call `elu<CTYPE>(ctx, in, alpha, scale, input_scale, out)`.
> Return out.
</content>
