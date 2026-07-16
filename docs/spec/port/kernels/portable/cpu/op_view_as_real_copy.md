# kernels/portable/cpu/op_view_as_real_copy.cpp

> [spec:et:def:op-view-as-real-copy.torch.executor.native.to-impl-fn]
> inline void _to_impl(const Tensor& self, Tensor& out)

> [spec:et:sem:op-view-as-real-copy.torch.executor.native.to-impl-fn]
> Copies a complex tensor into a real tensor by splitting each complex element
> into its real and imaginary parts. Templated on `SELF_CTYPE` (a complex
> element type with `.real_` and `.imag_` members) and `OUT_CTYPE` (the real
> output element type).
>
> Get `self_data` (as `SELF_CTYPE*`) and `out_data` (as `OUT_CTYPE*`). For each
> `i` in [0, self.numel()): read `val_in = self_data[i]`, then write
> `out_data[2*i] = (OUT_CTYPE)val_in.real_` and
> `out_data[2*i + 1] = (OUT_CTYPE)val_in.imag_`. Iteration is over the flat
> element index in storage order; the output is written contiguously with the
> new innermost size-2 (real, imag) dimension. Returns void.

> [spec:et:def:op-view-as-real-copy.torch.executor.native.view-as-real-copy-out-fn]
> Tensor& view_as_real_copy_out( KernelRuntimeContext& ctx, const Tensor& self, Tensor& out)

> [spec:et:sem:op-view-as-real-copy.torch.executor.native.view-as-real-copy-out-fn]
> Entry point for `view_as_real_copy.out(self) -> out`. Produces a real tensor
> whose shape is `self`'s shape with an extra trailing dimension of size 2
> holding (real, imag). Returns `out`.
>
> 1. `ET_KERNEL_CHECK_MSG((size_t)self.dim() < kTensorDimensionLimit,
>    InvalidArgument, ...)`: the output rank is `self.dim()+1`, which must fit
>    the dimension limit; on failure sets InvalidArgument on `ctx` and returns
>    `out`.
> 2. Compute the target output sizes into a local buffer via
>    `get_view_as_real_copy_out_target_size(self, expected_output_size)` (copies
>    `self.sizes()` then appends a final `2`).
> 3. `ET_KERNEL_CHECK_MSG(resize_tensor(out, {expected_output_size, out.dim()})
>    == Error::Ok, InvalidArgument, "Failed to resize output tensor.")`.
> 4. `ET_KERNEL_CHECK_MSG(isComplexType(self.scalar_type()), InvalidArgument,
>    "Input tensor must be complex type")`.
> 5. `ET_KERNEL_CHECK(tensors_have_same_dim_order(self, out), InvalidArgument)`.
> Each failed check sets the Error on `ctx` and returns `out`.
> 6. Dispatch `CTYPE_IN` over `self.scalar_type()` with `ET_SWITCH_COMPLEXH_TYPES`
>    (complex half/float/double: ComplexHalf, ComplexFloat, ComplexDouble),
>    nested `CTYPE_OUT` over `out.scalar_type()` with `ET_SWITCH_FLOATH_TYPES`
>    (Half, Float, Double); call
>    `[spec:et:sem:op-view-as-real-copy.torch.executor.native.to-impl-fn]` with
>    `<CTYPE_IN, CTYPE_OUT>(self, out)`.
> 7. Return `out`.

