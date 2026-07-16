# kernels/portable/cpu/op_cumsum.cpp

> [spec:et:def:op-cumsum.torch.executor.native.cumsum-tensors-fn]
> void cumsum_tensors( const Tensor& self, LoadFn load_self, int64_t dim, Tensor& out)

> [spec:et:sem:op-cumsum.torch.executor.native.cumsum-tensors-fn]
> Templated on `CTYPE_OUT` (the output/accumulation element type) and a `load_self` functor that
> reads one input element from a raw byte pointer and returns a `CTYPE_OUT` (this is how input dtype
> is converted to the compute/output dtype). Computes the cumulative sum of `self` along `dim`,
> writing into `out`. Assumes `out` is already correctly shaped (same shape as `self`) and `dim` is
> already normalized to a non-negative axis. No validation.
> 1. If `self.numel() == 0`, return immediately (no writes).
> 2. Let `input_data_base` = raw byte pointer to `self`'s data; `output_data_base` = `CTYPE_OUT*` to `out`.
> 3. If `self.dim() == 0` (scalar tensor): `output_data_base[0] = load_self(&input_data_base[0])`; return.
> 4. Otherwise let `dim_size = self.size(dim)`, `leading_dims = getLeadingDims(self, dim)`
>    (product of sizes of dims before `dim`), `trailing_dims = getTrailingDims(self, dim)`
>    (product of sizes of dims after `dim`). The tensor is treated as a contiguous
>    `[leading_dims, dim_size, trailing_dims]` block (default/contiguous layout).
> 5. For each `i` in `[0, leading_dims)`: let `start_loc = i * trailing_dims * dim_size`.
>    - Initialize the first slice along `dim` (`j = 0`): for each `idx` in `[0, trailing_dims)`,
>      `output[start_loc + idx] = load_self(&input[(start_loc + idx) * self.element_size()])`.
>      (element_size scales the byte offset into the raw input pointer.)
>    - For each `j` in `[1, dim_size)`: `cur = start_loc + j*trailing_dims`,
>      `prev = start_loc + (j-1)*trailing_dims`; for each `idx` in `[0, trailing_dims)`:
>      `output[cur + idx] = load_self(&input[(cur + idx) * self.element_size()]) + output[prev + idx]`.
>      Accumulation is sequential along `dim` and accumulates in `CTYPE_OUT` (so the input is upcast
>      to the output dtype before summation, preventing overflow when a wider output dtype is enforced).

> [spec:et:def:op-cumsum.torch.executor.native.cumsum-out-fn]
> Tensor& cumsum_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, optional<ScalarType> enforced_dtype, Tensor& out)

> [spec:et:sem:op-cumsum.torch.executor.native.cumsum-out-fn]
> Entry point for `cumsum.out(self, dim, dtype?, *, out)`. `enforced_dtype` (the optional `dtype`
> argument) selects the output dtype; the input is loaded/cast to `out.scalar_type()` before summing.
> Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `check_cumsum_args(self, dim, enforced_dtype, out)` (validates `dim` is a valid
>    axis for `self`, and that `out`'s dtype matches `enforced_dtype` when present or `self`'s dtype
>    otherwise), else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(self, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `resize_tensor(out, self.sizes()) == Error::Ok`, else InvalidArgument.
> 4. Normalize `dim`: if `self.dim() == 0` then `dim = 0`; else if `dim < 0` then `dim += self.dim()`.
> 5. Dispatch over `out.scalar_type()` restricted to REALHBBF16
>    (`{Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}`) via
>    `ET_SWITCH_REALHBBF16_TYPES`; unsupported dtype sets error and returns. For the selected
>    `CTYPE_OUT`, obtain `load_self = get_load_to_compute_fn<CTYPE_OUT>(ctx, self, REALHBBF16)`
>    (a functor that loads a `self` element of any REALHBBF16 dtype and returns it as `CTYPE_OUT`),
>    then call `cumsum_tensors<CTYPE_OUT>(self, load_self, dim, out)` per
>    `[spec:et:sem:op-cumsum.torch.executor.native.cumsum-tensors-fn]`.
> 6. Return `out`.

