# kernels/portable/cpu/op__to_dim_order_copy.cpp

> [spec:et:def:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn]
> Tensor& _to_dim_order_copy_out( KernelRuntimeContext& ctx, const Tensor& self, bool non_blocking, OptionalArrayRef<int64_t> dim_order, Tensor& out)

> [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn]
> Implements `_to_dim_order_copy.out(self, non_blocking=False, dim_order=None,
> out)`: copies `self` into `out`, converting dtype to `out`'s dtype and
> reordering to the layout given by `dim_order`. `ctx` unused for control flow.
>
> Validation, in order:
> 1. `check__to_dim_order_copy_args(self, non_blocking, dim_order, out)`
>    (`ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`) — validates
>    `dim_order` and that `out`'s dim order matches it (or matches `self` when
>    None).
> 2. `resize_tensor(out, self.sizes()) == Error::Ok` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`, returns `out`).
>
> If `self.numel() == 0`, returns `out` immediately.
>
> Dtype dispatch by complex-ness of input and output
> (`isComplexType(scalar_type())`), four cases:
> - Complex in, complex out: `ET_SWITCH_COMPLEXH_TYPES` on `self` dtype,
>   `_to_dim_order_copy_impl<CTYPE, CTYPE>(self, out)` (same-type element copy,
>   applying the dim-order remap).
> - Real in, complex out: `ET_SWITCH_FLOATH_TYPES` on `self` dtype (CTYPE_IN ∈
>   {Half, Float, Double}) and `ET_SWITCH_COMPLEXH_TYPES` on `out` dtype
>   (CTYPE_OUT complex). Iterates via `BroadcastIndexesRange<2,
>   support_noncontiguous_input_tensors=true>(self, self, out)` — a self→out
>   index walk that maps each output position to its source position honoring dim
>   orders — writing `out[j].real_ = self[i]`, `out[j].imag_ = 0`.
> - Complex in, real out: `ET_SWITCH_COMPLEXH_TYPES` on `self` dtype and
>   `ET_SWITCH_FLOATH_TYPES` on `out` dtype. Same index walk, writes
>   `out[j] = static_cast<CTYPE_OUT>(self[i].real_)` (discards imaginary part).
> - Real in, real out: `ET_SWITCH_REALHBBF16_TYPES` on both `self` and `out`
>   dtypes (each ∈ {Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>   BFloat16}), then `_to_dim_order_copy_impl<CTYPE_IN, CTYPE_OUT>(self, out)`
>   which element-wise copies with `static_cast` conversion and dim-order remap.
>
> Returns `out`. (An overload without `ctx` constructs a default context and
> delegates here.)

