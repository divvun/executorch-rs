# kernels/portable/cpu/op_tril.cpp

> [spec:et:def:op-tril.torch.executor.native.tril-kernel-fn]
> void tril_kernel( KernelRuntimeContext& ctx, const Tensor& self, int64_t diagonal, const Tensor& out)

> [spec:et:sem:op-tril.torch.executor.native.tril-kernel-fn]
> Templated helper `tril_kernel<CTYPE>(ctx, self, diagonal, out)` that copies the
> lower-triangular part of `self` (a 2-D matrix or batch of matrices) into `out`.
> `out` is assumed already zero-filled by the caller. Step by step:
>
> - `ndim = self.dim()`. ET_KERNEL_CHECK_MSG `ndim < kTensorDimensionLimit`; on
>   failure sets Error::InvalidArgument (message "ndim %zu >= %zu") and returns.
> - Build `sizes[i] = self.size(i)` and `strides[i] = getTrailingDims(self, i)`
>   for each dim `i` (i.e. contiguous strides derived from trailing dims, not the
>   tensor's own strides).
> - `num_rows = sizes[ndim-2]`, `num_cols = sizes[ndim-1]` (last two dims are the
>   matrix dims).
> - `batch_size = getLeadingDims(self, ndim-2)` (product of all leading dims).
>   `self_stride = (self.dim() > 2 && strides[ndim-3] > 0) ? strides[ndim-3] : 1`
>   (per-matrix stride; 1 for a single 2-D matrix).
> - `row_stride = strides[ndim-2]`, `col_stride = strides[ndim-1]`.
> - For each batch `i` in `[0, batch_size)`, apply the lower-triangular copy to
>   the matrix at offset `i * self_stride` in both `data_self` and `data_out`
>   (both offset by the same `self_stride`):
>   - For each row `r` in `[0, num_rows)`, for each col `c` in
>     `[0, min(num_cols, r + diagonal + 1))`: `out[r*row_stride + c*col_stride] =
>     self[r*row_stride + c*col_stride]`. Elements to the right of that per-row
>     limit are left as their pre-zeroed value 0. Thus `diagonal = 0` keeps the
>     main diagonal and below; `diagonal > 0` keeps additional super-diagonals;
>     `diagonal < 0` drops sub-diagonals.
> - Returns void. (CTYPE is the switched dtype from the caller.)

> [spec:et:def:op-tril.torch.executor.native.clear-out-fn]
> Tensor& clear_out(Tensor& out)

> [spec:et:sem:op-tril.torch.executor.native.clear-out-fn]
> Zero-fills the `out` tensor. Gets `out.mutable_data_ptr<uint8_t>()`; if it is
> non-null, `memset(out_data, 0, out.nbytes())` (sets every byte to 0). If the
> data pointer is null (e.g. a 0-numel tensor) does nothing. Returns `out`.

> [spec:et:def:op-tril.torch.executor.native.tril-out-fn]
> Tensor& tril_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t diagonal, Tensor& out)

> [spec:et:sem:op-tril.torch.executor.native.tril-out-fn]
> Returns the lower-triangular part of `self` (a 2-D matrix or batch of matrices)
> in `out`, with all other elements set to 0. Implements `tril.out(Tensor self,
> int diagonal=0, *, Tensor(a!) out)`. `diagonal` controls the retained band:
> `= 0` keeps the main diagonal and below; `> 0` adds super-diagonals; `< 0`
> removes sub-diagonals. Step by step:
>
> - ET_KERNEL_CHECK `check_tril_args(self, out)` (see
>   `[spec:et:sem:copy-ops-util...check-tril-args-fn]`): validates `self.dim() >=
>   2` and `self`/`out` share dtype/shape. On failure sets Error::InvalidArgument
>   and returns `out` unchanged.
> - Resize `out` to `self.sizes()`; on failure Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(self, out)`; else
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(self)`; else Error::InvalidArgument.
> - If `self.numel() == 0`, return `out` (resized, empty).
> - Zero-fill `out` via `clear_out` per
>   `[spec:et:sem:op-tril.torch.executor.native.clear-out-fn]`.
> - Dtype dispatch: switch `out.scalar_type()` over REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Bool, Half, Float, Double, BFloat16} as CTYPE and call
>   `tril_kernel<CTYPE>(ctx, self, diagonal, out)` per
>   `[spec:et:sem:op-tril.torch.executor.native.tril-kernel-fn]`, which copies the
>   selected lower-triangular elements from `self` into the already-zeroed `out`.
> - Returns `out`.

