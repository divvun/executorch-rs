# kernels/portable/cpu/op_softmax.cpp

> [spec:et:def:op-softmax.torch.executor.native.softmax-out-fn]
> Tensor& softmax_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool half_to_float, Tensor& out)

> [spec:et:sem:op-softmax.torch.executor.native.softmax-out-fn]
> Computes numerically-stable softmax of `in` along `dim` into `out`. Implements
> `_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out)`.
> Step by step:
>
> - ET_KERNEL_CHECK `check_softmax_args(in, dim, half_to_float, out)` (see
>   `[spec:et:sem:activation-ops-util...check-softmax-args-fn]`): validates `in`
>   and `out` share dtype, `dim` is a valid dimension of `in`, and the dtype is a
>   supported floating type. On failure sets Error::InvalidArgument and returns
>   `out` unchanged. (`half_to_float` is accepted but not used to alter behavior
>   here.)
> - Resize `out` to `in.sizes()`; on failure Error::InvalidArgument, returns `out`.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument.
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)` (nonzero_dim treats a
>   0-dim tensor as having 1 dimension).
> - Dtype dispatch over FLOATHBF16 = {Half, Float, Double, BFloat16} on
>   `in.scalar_type()` giving CTYPE. Accumulation type ACC is `float` when CTYPE
>   is Half or BFloat16, otherwise ACC = CTYPE (so Float accumulates in float,
>   Double in double). This float accumulation for half precision avoids
>   saturation, matching ATen acc_type.
> - `apply_over_dim(fn, in, dim)` (see `[spec:et:sem:reduce-util...apply-over-dim-fn]`)
>   invokes `fn(size, stride, base)` once per softmax lane, where `size =
>   in.size(dim)`, `stride` is the stride along `dim`, and `base` is the flat
>   start offset of each lane. For each lane:
>   1. `max_in` = max over the lane: `apply_unary_reduce_fn(max, in_data+base,
>      size, stride)` scanning the `size` elements at offsets
>      `base, base+stride, base+2*stride, ...` in CTYPE.
>   2. `temp_sum` (in ACC) = sum over the lane of
>      `exp(ACC(val_in) - ACC(max_in))` via `apply_unary_map_reduce_fn`.
>   3. Write each output element: `out = CTYPE(exp(ACC(val_in) - ACC(max_in)) /
>      temp_sum)`, subtracting the lane max before exp for numerical stability,
>      dividing by the lane sum, and casting back to CTYPE.
> - Returns `out`.

