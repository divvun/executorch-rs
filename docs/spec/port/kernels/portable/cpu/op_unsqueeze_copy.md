# kernels/portable/cpu/op_unsqueeze_copy.cpp

> [spec:et:def:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn]
> Tensor& unsqueeze_copy_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, Tensor& out)

> [spec:et:sem:op-unsqueeze-copy.torch.executor.native.unsqueeze-copy-out-fn]
> Inserts a new size-1 dimension at position `dim` and copies `self` into `out`.
> Implements `unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out)`. Since
> the underlying data is contiguous and unchanged, this is a flat `memcpy` after
> shape validation. Step by step:
>
> - Normalize `dim`: if `dim < 0`, `dim += out.dim()` (note: normalized against
>   `out.dim()`, which is `self.dim() + 1`), then ET_KERNEL_CHECK `dim >= 0`; on
>   failure sets Error::InvalidArgument and returns `out` unchanged.
> - ET_KERNEL_CHECK `self.dim() + 1 == out.dim()`; else Error::InvalidArgument.
> - ET_KERNEL_CHECK `dim <= self.dim()`; else Error::InvalidArgument.
> - Build `expected_output_size` over `[0, out.dim())`: for `i < dim`
>   `self.size(i)`; for `i > dim` `self.size(i-1)`; for `i == dim` value `1`.
> - Resize `out` to `{expected_output_size, out.dim()}`; on failure
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `check_unsqueeze_copy_args(self, dim, out)` (see
>   `[spec:et:sem:copy-ops-util...check-unsqueeze-copy-args-fn]`): validates
>   `self`/`out` share dtype and the resulting shape is consistent. On failure
>   Error::InvalidArgument.
> - If `self.nbytes() > 0`, `memcpy(out.mutable_data_ptr(), self.const_data_ptr(),
>   self.nbytes())` (byte-for-byte copy; guarded against null pointers when numel
>   is 0).
> - Returns `out`.

