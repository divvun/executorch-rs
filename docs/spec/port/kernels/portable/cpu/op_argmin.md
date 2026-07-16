# kernels/portable/cpu/op_argmin.cpp

> [spec:et:def:op-argmin.torch.executor.native.argmin-out-fn]
> Tensor& argmin_out( KernelRuntimeContext& ctx, const Tensor& in, optional<int64_t> dim, bool keepdim, Tensor& out)

> [spec:et:sem:op-argmin.torch.executor.native.argmin-out-fn]
> Implements `argmin.out(in, dim, keepdim, out)`: index of the minimum element
> over dimension `dim` (or over the whole tensor if `dim` is None). `out` dtype
> is Long (int64). `ctx` unused for control flow.
>
> Validation (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`):
> 1. `check_argmin_argmax_args(in, dim, keepdim, out)`.
> 2. `resize_reduction_out(in, dim, keepdim, out) == Error::Ok`.
> 3. `tensors_have_same_dim_order(in, out)`.
>
> Dtype dispatch: `ET_SWITCH_REALHBF16_TYPES` on `in.scalar_type()` (CTYPE ∈
> {Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16}; no Bool). `out`
> is always `int64_t`.
>
> For each output index `out_ix` (via
> `parallel_for_each_reduce_over_dim_output_index`), reduces over `dim` with
> `reduce_over_dim<CTYPE>`: running `(acc_val, acc_ix)` initialized from the first
> element; for each `(v, ix)`, if `!isnan_override(acc_val) && !(v >= acc_val)`
> then update `acc_val = v`, `acc_ix = ix`. This is equivalent to
> `!isnan(acc_val) && (isnan(v) || v < acc_val)`: NaN values are treated as the
> minimum (first NaN wins and sticks). Ties keep the first (lowest) index. Writes
> `out_data[out_ix] = acc_ix` (the argmin index along `dim`). On parallel-loop
> failure sets `Internal` ("parallel_for failed").
>
> Returns `out`.

