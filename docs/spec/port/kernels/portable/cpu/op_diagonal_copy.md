# kernels/portable/cpu/op_diagonal_copy.cpp

> [spec:et:def:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn]
> void diagonal_copy_impl( const Tensor& in, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out)

> [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn]
> Templated on `CTYPE`. Extracts the diagonal of `in` taken between axes `dim1` and `dim2` with the
> given `offset`, materializing it into `out` via a strided copy. Assumes `dim1`/`dim2` are already
> normalized to non-negative axes and `out` is already correctly shaped. No validation.
> 1. If `out.numel() == 0`, return immediately.
> 2. Compute `storage_offset` (element offset into `in`'s buffer where the diagonal starts):
>    let `diag_size = out.size(out.dim() - 1)` (the length of the diagonal, placed as the last dim
>    of `out`). If `diag_size == 0`, leave `storage_offset = 0`. Else if `offset >= 0`,
>    `storage_offset = offset * in.strides()[dim2]`. Else `storage_offset = -offset * in.strides()[dim1]`.
> 3. Build the view sizes `new_sizes` = `out.sizes()` (rank `new_ndim = out.dim()`).
> 4. Build the view strides `new_strides` of rank `in.dim()` positions used as follows: iterate the
>    input dims `d` in `[0, in.dim())` keeping a `shift` counter starting at 0; whenever `d == dim1`
>    or `d == dim2`, increment `shift` (those two axes are collapsed); otherwise set
>    `new_strides[d - shift] = in.strides()[d]` (preserving order of the remaining axes). Finally set
>    `new_strides[in.dim() - 2] = in.strides()[dim1] + in.strides()[dim2]` — the diagonal's stride
>    (stepping one along both collapsed axes), placed as the last stride.
> 5. Call `as_strided_copy<CTYPE>(in, {new_sizes, new_ndim}, {new_strides, new_ndim}, storage_offset, out)`,
>    which copies elements from `in` at the strided/offset view into contiguous `out`.

> [spec:et:def:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn]
> Tensor& diagonal_copy_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t offset, int64_t dim1, int64_t dim2, Tensor& out)

> [spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-out-fn]
> Entry point for `diagonal_copy.out(in, offset, dim1, dim2, *, out)`. Every failure path sets the
> error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `check_diagonal_copy_args(in, dim1, dim2, out)` (validates `dim1`/`dim2` are
>    valid distinct axes and dtypes match), else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensor_is_default_dim_order(in)`, else InvalidArgument.
> 4. Normalize axes: if `dim1 < 0` then `dim1 += nonzero_dim(in)`; if `dim2 < 0` then
>    `dim2 += nonzero_dim(in)` (`nonzero_dim` treats a 0-dim tensor as rank 1).
> 5. Compute the output shape via `get_diagonal_copy_out_target_size(in, offset, dim1, dim2, ...)`
>    into `expected_out_size`/`expected_out_dim` (the remaining axes followed by the diagonal length
>    as the last dim), then ET_KERNEL_CHECK `resize_tensor(out, {expected_out_size, expected_out_dim})
>    == Error::Ok`, else InvalidArgument.
> 6. Dispatch over `in.scalar_type()` across ALL scalar types (`ET_SWITCH_ALL_TYPES`) and call
>    `diagonal_copy_impl<CTYPE>(in, offset, dim1, dim2, out)` per
>    `[spec:et:sem:op-diagonal-copy.torch.executor.native.diagonal-copy-impl-fn]`.
> 7. Return `out`.

