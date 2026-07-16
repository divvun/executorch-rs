# kernels/portable/cpu/op_squeeze_copy.cpp

> [spec:et:def:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn]
> Tensor& squeeze_copy_dim_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, Tensor& out)

> [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dim-out-fn]
> Removes the size-1 dimension `dim` from `in` (if it is size 1) and copies data
> into `out`. Implements `squeeze_copy.dim_out(Tensor self, int dim, *, Tensor(a!)
> out)`. Since the data is contiguous and only the shape changes, the copy is a
> flat `memcpy`. Step by step:
>
> - ET_KERNEL_CHECK `check_squeeze_copy_dim_args(in, dim, out)` (see
>   `[spec:et:sem:copy-ops-util...check-squeeze-copy-dim-args-fn]`): validates
>   `dim` is a valid dimension of `in`, `in`/`out` share dtype, and `out` has the
>   correct squeezed shape. On failure sets Error::InvalidArgument and returns
>   `out` unchanged.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(in)` (in must be contiguous /
>   default dim order); else Error::InvalidArgument.
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)`.
> - Compute `expected_out_size`/`expected_out_dim` via
>   `get_squeeze_copy_dim_out_target_size`: `out` equals `in`'s shape with `dim`
>   removed if `in.size(dim) == 1`, otherwise unchanged shape. Resize `out` to
>   that shape; on failure Error::InvalidArgument.
> - If `in.nbytes() > 0`, `memcpy(out.mutable_data_ptr(), in.const_data_ptr(),
>   in.nbytes())` (byte-for-byte copy; guarded against null pointers when numel
>   is 0).
> - Returns `out`.

> [spec:et:def:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn]
> Tensor& squeeze_copy_dims_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> dims, Tensor& out)

> [spec:et:sem:op-squeeze-copy.torch.executor.native.squeeze-copy-dims-out-fn]
> Removes each size-1 dimension listed in `dims` from `in` and copies data into
> `out`. Implements `squeeze_copy.dims_out(Tensor self, int[] dim, *, Tensor(a!)
> out)`. Like the single-dim variant, the copy is a flat `memcpy`. Step by step:
>
> - ET_KERNEL_CHECK `check_squeeze_copy_dims_args(in, dims, out)` (see
>   `[spec:et:sem:copy-ops-util...check-squeeze-copy-dims-args-fn]`): validates
>   each entry of `dims` is a valid (possibly negative) dimension of `in` with no
>   duplicates after normalization, `in`/`out` share dtype, and `out` has the
>   correct squeezed shape. On failure Error::InvalidArgument, returns `out`.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(in)`; else Error::InvalidArgument.
> - Compute `expected_out_size`/`expected_out_dim` via
>   `get_squeeze_copy_dims_out_target_size`: `out` equals `in`'s shape with every
>   dimension in `dims` whose size is 1 removed (dims whose size != 1 are kept).
>   Resize `out` to that shape; on failure Error::InvalidArgument.
> - If `in.nbytes() > 0`, `memcpy(out.mutable_data_ptr(), in.const_data_ptr(),
>   in.nbytes())` (guarded against null when numel is 0).
> - Returns `out`.

