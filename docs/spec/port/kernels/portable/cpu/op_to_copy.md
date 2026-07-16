# kernels/portable/cpu/op_to_copy.cpp

> [spec:et:def:op-to-copy.torch.executor.native.to-impl-fn]
> void _to_impl(const Tensor& self, Tensor& out)

> [spec:et:sem:op-to-copy.torch.executor.native.to-impl-fn]
> Templated helper `_to_impl<SELF_CTYPE, OUT_CTYPE>(self, out)` that performs the
> element-wise dtype cast for `to_copy_out`. It takes typed data pointers
> `self_data` (SELF_CTYPE) and `out_data` (OUT_CTYPE) and, for each flat index
> `i` in `[0, self.numel())`, writes `out_data[i] =
> static_cast<OUT_CTYPE>(self_data[i])`. This is a plain C++ `static_cast`
> conversion in flat/contiguous order (no bounds or dim-order handling — the
> caller guarantees `self` and `out` are contiguous with equal numel). Returns
> void.

> [spec:et:def:op-to-copy.torch.executor.native.to-copy-out-fn]
> Tensor& to_copy_out( KernelRuntimeContext& ctx, const Tensor& self, bool non_blocking, std::optional<executorch::aten::MemoryFormat> memory_format, Tensor& out)

> [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn]
> Copies `self` into `out`, casting to `out`'s dtype element-wise. Implements
> `_to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat?
> memory_format=None, Tensor(a!) out)`. Step by step:
>
> - ET_KERNEL_CHECK `check_to_copy_args(self, non_blocking, memory_format, out)`
>   (see `[spec:et:sem:copy-ops-util...check-to-copy-args-fn]`): validates
>   `self`/`out` have the same shape, `non_blocking == false`, and `memory_format`
>   (if present) is compatible with the tensor's dim order. On failure sets
>   Error::InvalidArgument and returns `out` unchanged.
> - Resize `out` to `self.sizes()`; on failure Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(self, out)`; else
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(self)`; else Error::InvalidArgument.
> - Dtype dispatch: switch `self.scalar_type()` over REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Bool, Half, Float, Double, BFloat16} as CTYPE_IN, and
>   `out.scalar_type()` over the same set as CTYPE_OUT; call
>   `_to_impl<CTYPE_IN, CTYPE_OUT>(self, out)` per
>   `[spec:et:sem:op-to-copy.torch.executor.native.to-impl-fn]`, i.e. cast each
>   element `out[i] = static_cast<CTYPE_OUT>(self[i])` in flat order. Input and
>   output dtypes may differ arbitrarily within the set.
> - Returns `out`.

