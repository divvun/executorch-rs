# kernels/portable/cpu/op_arange.cpp

> [spec:et:def:op-arange.torch.executor.native.arange-out-fn]
> Tensor& arange_out(KernelRuntimeContext& ctx, const Scalar& end, Tensor& out)

> [spec:et:sem:op-arange.torch.executor.native.arange-out-fn]
> Implements `arange.out(end, out)`: fills `out` with `[0, 1, 2, ..., ceil(end))`
> (start 0, step 1).
>
> Steps (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`):
> 1. `utils::extract_scalar(end, &end_val)` into a `double` `end_val`.
> 2. `check_arange_args(0.0, end_val, 1.0, out)` — validates the range/step and
>    `out` dtype.
> 3. `tensor_is_default_dim_order(out)`.
> 4. `out_length = compute_arange_out_size(0.0, end_val, 1.0)` (number of
>    elements = ceil(end_val) clamped at >= 0), then `resize_tensor(out,
>    {&out_length, 1})` to a 1-D tensor of that length.
> 5. `arange_out_impl(ctx, end_val, out)` fills `out[i] = static_cast<out_dtype>
>    (i)` for `i` in `[0, out_length)`.
>
> Returns `out`.

> [spec:et:def:op-arange.torch.executor.native.arange-start-out-fn]
> Tensor& arange_start_out( KernelRuntimeContext& ctx, const Scalar& start, const Scalar& end, const Scalar& step, Tensor& out)

> [spec:et:sem:op-arange.torch.executor.native.arange-start-out-fn]
> Implements `arange.start_out(start, end, step, out)`: fills `out` with
> `[start, start+step, start+2*step, ...]` up to (excluding) `end`. `ctx` unused
> for control flow.
>
> Steps (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`):
> 1. Extract `d_start`, `d_end`, `d_step` as `double` via
>    `utils::extract_scalar` (three separate checks).
> 2. `check_arange_args(d_start, d_end, d_step, out)` — validates the range/step
>    (e.g. non-zero step, sign consistency) and `out` dtype.
> 3. `tensor_is_default_dim_order(out)`.
> 4. `out_length = compute_arange_out_size(d_start, d_end, d_step)` (=
>    `ceil((d_end - d_start) / d_step)` clamped at >= 0), then `resize_tensor(out,
>    {&out_length, 1})`.
> 5. `arange_out_impl(ctx, d_start, d_end, d_step, out)` fills `out[i] =
>    static_cast<out_dtype>(d_start + i * d_step)` for `i` in `[0, out_length)`.
>
> Returns `out`.

