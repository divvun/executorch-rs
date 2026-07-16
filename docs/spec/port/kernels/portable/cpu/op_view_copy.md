# kernels/portable/cpu/op_view_copy.cpp

> [spec:et:def:op-view-copy.torch.executor.native.view-copy-out-fn]
> Tensor& view_copy_out( KernelRuntimeContext& ctx, const Tensor& self, executorch::aten::ArrayRef<int64_t> size_int64_t, Tensor& out)

> [spec:et:sem:op-view-copy.torch.executor.native.view-copy-out-fn]
> Entry point for `view_copy.out(self, size, *, out)`. Copies `self`'s raw
> bytes into `out` after reshaping `out` to the requested `size` (which may
> contain a single `-1` inferred dimension). Returns `out`.
>
> 1. Declare a local `expected_output_size[16]` buffer.
>    `ET_KERNEL_CHECK(get_view_copy_target_size(self, size_int64_t, out.dim(),
>    expected_output_size), InvalidArgument)`: resolves the concrete output
>    sizes from `size_int64_t` (substituting any `-1` with `self.numel() /
>    product(other dims)`) and validates the requested rank matches `out.dim()`.
>    On failure sets InvalidArgument on `ctx` and returns `out`.
> 2. `ET_KERNEL_CHECK_MSG(resize_tensor(out, {expected_output_size, out.dim()})
>    == Error::Ok, InvalidArgument, "Failed to resize output tensor.")`.
> 3. `ET_KERNEL_CHECK(tensors_have_same_dim_order(self, out), InvalidArgument)`.
> 4. `ET_KERNEL_CHECK(tensor_is_default_dim_order(self), InvalidArgument)`.
> 5. `ET_KERNEL_CHECK(check_view_copy_args(self, size_int64_t, out),
>    InvalidArgument)`: verifies at most one `-1`, the requested element count
>    equals `self.numel()`, and `out` has the resolved shape and same dtype.
> Each failed check sets InvalidArgument on `ctx` and returns `out` unchanged.
> 6. If `self.nbytes() > 0`, `memcpy(out.mutable_data_ptr(),
>    self.const_data_ptr(), self.nbytes())` — a straight byte copy (both are
>    contiguous/default dim order, same dtype, same element count). Empty
>    tensors skip the copy.
> 7. Return `out`.

