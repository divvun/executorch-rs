# kernels/portable/cpu/op_amin.cpp

> [spec:et:def:op-amin.torch.executor.native.amin-out-fn]
> Tensor& amin_out( KernelRuntimeContext& ctx, const Tensor& in, ArrayRef<int64_t> dim_list, bool keepdim, Tensor& out)

> [spec:et:sem:op-amin.torch.executor.native.amin-out-fn]
> Implements `amin.out(in, dim_list, keepdim, out)`: reduces `in` by taking the
> minimum over the dimensions in `dim_list`, writing into `out`. NaN-propagating.
> `ctx` unused for control flow.
>
> Validation, in order (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns
> `out`):
> 1. `check_amin_amax_args(in, dim_list, keepdim, out)` — validates `dim_list`
>    entries are in range/unique and `out` dtype matches `in`.
> 2. `resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok` — shapes
>    `out` to the reduced shape (reduced dims removed, or kept as size 1 if
>    `keepdim`).
> 3. `tensors_have_same_dim_order(in, out)`.
>
> Builds `ReduceOverDimListPlan plan(in, dim_list)`. Dtype dispatch:
> `ET_SWITCH_REALHBBF16_TYPES` on `in.scalar_type()` (CTYPE ∈ {Byte, Char, Short,
> Int, Long, Half, Float, Double, Bool, BFloat16}); `out` uses the same CTYPE.
>
> For each output index `out_ix` (iterated via
> `parallel_for_each_reduce_over_dim_list_output_index`; output positions may be
> processed in parallel chunks; result order-independent), computes
> `out_data[out_ix] = plan.execute<CTYPE>(reducer, out_ix)` where the reducer,
> given a new element `v` and current running min `min_v`, returns `v` if
> `utils::isnan_override(v) || v < min_v` else `min_v`. NaN inputs propagate. On
> parallel-loop failure sets `Internal` ("parallel_for failed") and returns.
>
> Returns `out`.

