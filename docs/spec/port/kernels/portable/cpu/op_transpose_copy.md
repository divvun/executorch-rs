# kernels/portable/cpu/op_transpose_copy.cpp

> [spec:et:def:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn]
> Tensor& transpose_copy_int_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim0, int64_t dim1, Tensor& out)

> [spec:et:sem:op-transpose-copy.torch.executor.native.transpose-copy-int-out-fn]
> Swaps dimensions `dim0` and `dim1` of `in`, writing the densely-packed
> (contiguous) result into `out`. Implements `transpose_copy.int_out(Tensor self,
> int dim0, int dim1, *, Tensor(a!) out)`. Step by step:
>
> - ET_KERNEL_CHECK `check_transpose_copy_args(in, dim0, dim1, out)` (see
>   `[spec:et:sem:transpose-util...check-transpose-copy-args-fn]`): validates
>   `dim0` and `dim1` are valid (possibly negative) dimensions of `in` and
>   `in`/`out` share dtype. On failure sets Error::InvalidArgument and returns
>   `out` unchanged.
> - Normalize dims: if `dim0 < 0`, `dim0 += nonzero_dim(in)`; if `dim1 < 0`,
>   `dim1 += nonzero_dim(in)`.
> - Compute `expected_out_size` via `get_transpose_out_target_size(in, dim0,
>   dim1, ...)` (same shape as `in` with sizes of `dim0` and `dim1` swapped) and
>   resize `out`; on failure Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument.
> - Switch over ALL types (any ScalarType) and call
>   `transpose_tensors<CTYPE>(in, dim0, dim1, out)` (see
>   `[spec:et:sem:transpose-util...transpose-tensors-fn]`), which for each output
>   element reads the corresponding input element with `dim0`/`dim1` swapped and
>   writes it contiguously into `out`.
> - Returns `out`.

