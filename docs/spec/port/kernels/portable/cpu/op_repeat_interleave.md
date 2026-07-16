# kernels/portable/cpu/op_repeat_interleave.cpp

> [spec:et:def:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn]
> bool check_repeat_interleave_args( const Tensor& repeats, int64_t output_size_value, int64_t repeats_sum, Tensor& out)

> [spec:et:sem:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn]
> Validates arguments for `repeat_interleave.Tensor_out`. Returns `bool`. Checks
> in order, each failing check logs and returns false:
>
> - `repeats.scalar_type()` is Int or Long (message "repeats must be int or
>   long; ...").
> - `repeats.dim() == 1` (repeats must be 1-D).
> - `output_size_value == repeats_sum` (message "output_size, if provided, must
>   be equal to repeats.sum(); ...").
> - `tensors_have_same_dtype(repeats, out)` (out must have the same dtype as
>   repeats).
> - Every element of `repeats` is `>= 0`: iterate over `repeats.numel()` reading
>   as `int64_t` when the dtype is Long, else as `int32_t`; any negative entry
>   fails (message "repeats cannot be negative; ...").
> - Returns true if all checks pass.

> [spec:et:def:op-repeat-interleave.torch.executor.native.repeat-interleave-tensor-out-fn]
> Tensor& repeat_interleave_Tensor_out( KernelRuntimeContext& ctx, const Tensor& repeats, std::optional<int64_t> output_size, Tensor& out)

> [spec:et:sem:op-repeat-interleave.torch.executor.native.repeat-interleave-tensor-out-fn]
> Given a 1-D `repeats` tensor, produces a 1-D index tensor `out` where each
> position `ix` is repeated `repeats[ix]` times, i.e. `out = [0 repeated
> repeats[0] times, 1 repeated repeats[1] times, ...]`. Steps:
>
> - Compute `repeats_sum`: dispatch on `repeats.scalar_type()` over {Int, Long}
>   as CTYPE and sum all `repeats.numel()` elements (accumulated as `int64_t`).
> - `output_size_value = output_size.value()` if the optional is present, else
>   `repeats_sum`.
> - ET_KERNEL_CHECK: `check_repeat_interleave_args(repeats, output_size_value,
>   repeats_sum, out)` (see
>   `[spec:et:sem:op-repeat-interleave.torch.executor.native.check-repeat-interleave-args-fn]`);
>   on failure `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `repeats`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - ET_KERNEL_CHECK: `repeats` is default dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Resize `out` to 1-D shape `{output_size_value}`; on failure
>   `Error::InvalidArgument` (message "Failed to resize output tensor."), return
>   `out`.
> - Dispatch on `repeats.scalar_type()` over {Int, Long} as CTYPE. Walk `ix` over
>   `repeats.numel()`; for each, write the value `static_cast<CTYPE>(ix)` into
>   consecutive `out` positions `repeats_data[ix]` times (inner loop `i` from 0
>   to `repeats_data[ix]`), advancing a running `out_ix`. `out` has the same
>   Int/Long dtype as `repeats`.
> - Returns `out`.

