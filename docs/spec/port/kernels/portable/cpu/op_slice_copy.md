# kernels/portable/cpu/op_slice_copy.cpp

> [spec:et:def:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn]
> Tensor& slice_copy_Tensor_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, std::optional<int64_t> start_val, std::optional<int64_t> end_val, int64_t step, Tensor& out)

> [spec:et:sem:op-slice-copy.torch.executor.native.slice-copy-tensor-out-fn]
> Copies a strided slice of `in` along dimension `dim` into `out`. Implements
> `slice_copy.Tensor_out(Tensor self, int dim, int? start=None, int? end=None,
> int step=1, *, Tensor(a!) out)`. Step by step:
>
> - ET_KERNEL_CHECK `check_slice_copy_args(in, dim, step, out)` (see
>   `[spec:et:sem:slice-util...check-slice-copy-args-fn]`): validates `dim` is a
>   valid dimension of `in` (via tensor_has_dim, before normalization so negative
>   allowed), `step > 0`, and `in`/`out` have the same dtype. On failure sets
>   Error::InvalidArgument and returns `out` unchanged.
> - Normalize `dim`: if `dim < 0`, `dim += in.dim()`.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument, returns `out`.
> - Resolve bounds: `end = end_val.has_value() ? end_val.value() : in.size(dim)`;
>   `start = start_val.has_value() ? start_val.value() : 0`.
> - `length = adjust_slice_indices(in.size(dim), &start, &end, step)` (see
>   `[spec:et:sem:slice-util...adjust-slice-indices-fn]`): clamps `start`/`end`
>   into `[0, in.size(dim)]` handling negatives, and returns the number of
>   sliced elements `length = ceil((end - start) / step)` clamped at >= 0.
> - Compute target sizes via `get_slice_copy_out_target_size`: same shape as `in`
>   except dimension `dim` becomes `length`; resize `out` to that shape. On
>   resize failure Error::InvalidArgument, returns `out`.
> - `compute_slice(ctx, in, dim, start, length, step, out)` (see
>   `[spec:et:sem:slice-util...compute-slice-fn]`) copies the selected elements:
>   iterating over leading dims (product of sizes before `dim`), then over the
>   `length` slice positions `start, start+step, start+2*step, ...`, copying
>   `trailing_dims` (product of sizes after `dim`) contiguous elements each,
>   with the input and output element type identical (byte-wise copy of a
>   trailing block).
> - Returns `out`.

