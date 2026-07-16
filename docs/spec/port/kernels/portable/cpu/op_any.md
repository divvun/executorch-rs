# kernels/portable/cpu/op_any.cpp

> [spec:et:def:op-any.torch.executor.native.any-all-out-fn]
> Tensor& any_all_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-any.torch.executor.native.any-all-out-fn]
> Implements `any.all_out(in, out)`: reduces the entire tensor `in` to a scalar
> indicating whether any element is truthy. `ctx` unused for control flow.
>
> Validation, in order (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns
> `out`):
> 1. `resize_tensor(out, {}) == Error::Ok` — resizes `out` to a 0-dim scalar.
> 2. `tensors_have_same_dim_order(in, out)`.
>
> Dtype dispatch: `ET_SWITCH_REALHBBF16_TYPES` on `in.scalar_type()` (CTYPE_IN ∈
> {Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}); output
> dtype via `ET_SWITCH_TWO_TYPES(Bool, Byte, ...)` so CTYPE_OUT is Bool or Byte.
>
> Algorithm: sets `data_out[0] = false`; iterates flat `i` in `[0, in.numel())`,
> and on the first element where `static_cast<bool>(data_in[i])` is true, sets
> `data_out[0] = true` and breaks. (For floats, any nonzero including NaN is
> truthy; +0.0/-0.0 are false.) Empty input yields false. Returns `out`.

> [spec:et:def:op-any.torch.executor.native.any-dims-out-fn]
> Tensor& any_dims_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, bool keepdim, Tensor& out)

> [spec:et:sem:op-any.torch.executor.native.any-dims-out-fn]
> Implements `any.dims_out(in, dim_list, keepdim, out)`: reduces `in` with
> logical-OR over the dimensions in `dim_list` (optional). `ctx` unused for
> control flow.
>
> Validation:
> 1. `check_reduction_args(in, dim_list, keepdim, {}, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
> 2. Resize: if `dim_list` has a value and is empty, `resize_tensor(out,
>    in.sizes())` (element-wise, no reduction — each element mapped to bool);
>    otherwise `resize_reduction_out(in, dim_list, keepdim, out)`. (`ET_KERNEL_CHECK`
>    → `InvalidArgument`.)
> 3. `tensors_have_same_dim_order(in, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
>
> Builds a `MapReduceOverDimListPlan plan(in, dim_list)` only when `(dim_list is
> None OR dim_list non-empty) AND in.numel() > 0`.
>
> Dtype dispatch: `ET_SWITCH_REALHBBF16_TYPES` on `in` (CTYPE_IN ∈ {Byte, Char,
> Short, Int, Long, Half, Float, Double, Bool, BFloat16}); output via
> `ET_SWITCH_TWO_TYPES(Bool, Byte, ...)` (CTYPE_OUT is Bool or Byte).
>
> Two cases:
> - `dim_list` present and empty: no reduction. For each `out_ix` in
>   `[0, out.numel())`, `out_data[out_ix] = static_cast<CTYPE_OUT>(static_cast<
>   bool>(in_data[out_ix]))` — element-wise truthiness copy.
> - Otherwise: for each output index `out_ix` (via
>   `parallel_for_each_reduce_over_dim_list_output_index`), `any = false`; if the
>   plan exists (input non-empty), `any = plan->execute<CTYPE_IN, bool>(map = v →
>   static_cast<bool>(v), reduce = (outv, acc) → acc || outv, out_ix)`; write
>   `out_data[out_ix] = static_cast<CTYPE_OUT>(any)`. When input is empty the
>   plan is absent so results are all false. On parallel-loop failure sets
>   `Internal` ("parallel_for failed").
>
> Returns `out`.

> [spec:et:def:op-any.torch.executor.native.any-out-fn]
> Tensor& any_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool keepdim, Tensor& out)

> [spec:et:sem:op-any.torch.executor.native.any-out-fn]
> Implements `any.out(in, dim, keepdim, out)`: reduces `in` with logical-OR over
> a single dimension `dim`. `ctx` unused for control flow.
>
> Validation (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`):
> 1. `check_reduction_args_single_dim(in, dim, keepdim, {}, out,
>    allow_empty_dim=true)`.
> 2. `resize_reduction_out(in, dim, keepdim, out) == Error::Ok`.
> 3. `tensors_have_same_dim_order(in, out)`.
>
> Dtype dispatch: `ET_SWITCH_REALHBBF16_TYPES` on `in` (CTYPE_IN ∈ {Byte, Char,
> Short, Int, Long, Half, Float, Double, Bool, BFloat16}); output via
> `ET_SWITCH_TWO_TYPES(Bool, Byte, ...)` (CTYPE_OUT is Bool or Byte).
>
> For each output index `out_ix` (via
> `parallel_for_each_reduce_over_dim_output_index`): `any = false`; if
> `in.numel() > 0`, reduce over dimension `dim` at `out_ix` using
> `map_reduce_over_dim<CTYPE_IN, CTYPE_OUT>(map = v → static_cast<bool>(v),
> reduce = (outv, _, acc, _) → {acc || outv, 0}, in, dim, out_ix)` and take the
> boolean component; write `out_data[out_ix] = any`. Empty input yields all
> false. On parallel-loop failure sets `Internal` ("parallel_for failed").
>
> Returns `out`.

