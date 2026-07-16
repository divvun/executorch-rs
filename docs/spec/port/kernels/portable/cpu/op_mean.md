# kernels/portable/cpu/op_mean.cpp

> [spec:et:def:op-mean.torch.executor.native.mean-dim-out-fn]
> Tensor& mean_dim_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, bool keepdim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn]
> Computes the mean of `in` reduced over the dimensions in `dim_list` (all dims
> if `dim_list` is nullopt/empty), writing to `out`. `keepdim` controls whether
> reduced dims are retained as size-1. `dtype` is the optional output element
> type (validated against `out`).
>
> Steps:
> 1. Validate via `check_mean_dim_args(in, dim_list, keepdim, dtype, out)`
>    (ET_KERNEL_CHECK: on failure sets Error::InvalidArgument and returns `out`
>    unchanged): dims in `dim_list` valid and unique, `out` floating and matching
>    `dtype` if provided, `out` shape consistent with the reduction under
>    `keepdim`.
> 2. Check `in` and `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument).
> 3. Check `in` is default (contiguous) dim order (ET_KERNEL_CHECK:
>    InvalidArgument).
> 4. Resize `out` for the reduction over `dim_list` with `keepdim` per
>    `[spec:et:sem:reduce-util.torch.executor.native.resize-reduction-out-fn]`
>    (ET_KERNEL_CHECK: InvalidArgument).
> 5. Fast path: if `in.numel() > 0` and `dim_list` has exactly one dim and
>    `in.scalar_type() == out.scalar_type()`, normalize `d` (`+= in.dim()` if
>    negative); if `d` is the last dim and `in` is contiguous, dispatch on
>    `in.scalar_type()` over FLOATHBF16 {Half, Float, Double, BFloat16} with
>    accumulation type ACC = float when CTYPE is Half/BFloat16 else CTYPE.
>    Let `reduce_size = in.size(d)`, `outer_size = in.numel() / reduce_size`,
>    `denom = (ACC)reduce_size`. For each outer row `i` in `[0, outer_size)`, sum
>    the `reduce_size` contiguous elements into an ACC accumulator and write
>    `out[i] = (CTYPE)(acc / denom)`. Return `out`.
> 6. General path: dispatch on `in.scalar_type()` over REALHBBF16 (CTYPE_IN) and
>    on `out.scalar_type()` over FLOATHBF16 (CTYPE_OUT), ACC = float when CTYPE_OUT
>    is Half/BFloat16 else CTYPE_OUT. Let `num = get_reduced_dim_product(in,
>    dim_list)` (number of elements collapsed per output position). For each
>    output index `out_ix` (iterating output positions, possibly in parallel):
>    sum (in ACC) the corresponding reduced input elements (converting each
>    CTYPE_IN value to ACC and accumulating), giving `sum`; write
>    `out[out_ix] = (CTYPE_OUT)(sum / (float)num)`. If `in.numel() == 0`, `sum`
>    stays 0 (so the result is `0 / num`). On parallel-reduction failure,
>    ET_KERNEL_CHECK_MSG sets Error::Internal (message "parallel_for failed").
> 7. Return `out`.

> [spec:et:def:op-mean.torch.executor.native.mean-dtype-out-fn]
> Tensor& mean_dtype_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-mean.torch.executor.native.mean-dtype-out-fn]
> Computes the mean of `in` over all dimensions (full reduction to a scalar),
> optionally casting to `dtype`, writing to `out`. Implemented by delegating to
> `[spec:et:sem:op-mean.torch.executor.native.mean-dim-out-fn]` with
> `dim_list = ArrayRef<int64_t>()` (empty → reduce all dims), `keepdim = false`,
> and the same `dtype` and `out`. Returns whatever that call returns (`out`).

