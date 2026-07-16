# kernels/portable/cpu/op_sum.cpp

> [spec:et:def:op-sum.torch.executor.native.sum-dim-out-fn]
> Tensor& sum_dim_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, bool keepdim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-sum.torch.executor.native.sum-dim-out-fn]
> Sums `in` over the dimensions in `dim_list` into `out`. Implements
> `sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType?
> dtype=None, Tensor(a!) out)`. Step by step:
>
> - ET_KERNEL_CHECK `check_reduction_args(in, dim_list, keepdim, dtype, out)`
>   (see `[spec:et:sem:reduce-util...check-reduction-args-fn]`): validates each
>   dim in `dim_list` is valid with no duplicates, `dtype` (if present) matches
>   `out.scalar_type()`, and `out` has the correct reduced shape. On failure sets
>   Error::InvalidArgument and returns `out` unchanged.
> - ET_KERNEL_CHECK `resize_reduction_out(in, dim_list, keepdim, out) == Ok`
>   (resize `out` to the reduced shape: reduced dims removed, or kept as size 1
>   when `keepdim`; a null/empty `dim_list` reduces all dims). On failure
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`; else
>   Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(in)`; else Error::InvalidArgument.
> - Fast path (contiguous innermost single-dim reduction, same dtype): if
>   `in.numel() > 0`, `dim_list` is present with exactly one entry, `in` is not
>   complex, and `in.scalar_type() == out.scalar_type()`: normalize `d`
>   (`d += in.dim()` if negative); if `d` is the last dim (`d == in.dim()-1`) and
>   `in` is contiguous, then for `outer_size = in.numel()/in.size(d)` output
>   rows, accumulate each contiguous run of `reduce_size = in.size(d)` elements.
>   Accumulator ACC is `float` when CTYPE is Half or BFloat16 else CTYPE (float
>   accumulation for half precision to avoid saturation, matching ATen acc_type),
>   starting at 0, `acc += row[j]`; store `out_data[i] = static_cast<CTYPE>(acc)`.
>   Dtype set is REALHBBF16 = {Byte, Char, Short, Int, Long, Bool, Half, Float,
>   Double, BFloat16}. Returns `out`.
> - General path: create a `MapReduceOverDimListPlan(in, dim_list)` iff
>   `in.numel() > 0`.
>   - Complex input: ET_KERNEL_CHECK `in.scalar_type() == out.scalar_type()`;
>     switch over COMPLEXH types. For each output index, if the plan exists sum
>     the mapped input values (identity map, complex add starting from `(0,0)`),
>     else 0, into `out_data[out_ix]`. Uses
>     `parallel_for_each_reduce_over_dim_list_output_index`; on failure
>     ET_KERNEL_CHECK_MSG sets Error::Internal ("parallel_for failed").
>   - Non-complex input: switch input over REALHBBF16 (CTYPE_IN) and output over
>     REALHBBF16 (CTYPE_OUT). ACC is `float` when CTYPE_OUT is Half or BFloat16
>     else CTYPE_OUT. For each output index, sum (via the plan, mapping each
>     CTYPE_IN input to ACC and accumulating in ACC starting at 0) and store
>     `out_data[out_ix] = static_cast<CTYPE_OUT>(sum)`; if no plan (empty input)
>     the sum is 0. Same parallel-for and Internal-error handling.
> - Returns `out`.

