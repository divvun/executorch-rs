# kernels/portable/cpu/op_log_softmax.cpp

> [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-out-fn]
> Tensor& log_softmax_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool half_to_float, Tensor& out)

> [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-out-fn]
> Computes `log_softmax(in, dim)` into `out` (same shape/dtype as `in`), i.e. for
> each slice along `dim`, `out = in - max - log(sum(exp(in - max)))`.
>
> Steps:
> 1. Validate arguments via `check_log_softmax_args(in, dim, half_to_float, out)`
>    (ET_KERNEL_CHECK: on failure sets Error::InvalidArgument on the context and
>    returns `out` unchanged). This enforces `in` and `out` have the same dtype,
>    `out` is a floating type, `dim` is a valid dimension for `in` (in
>    `[-in.dim(), in.dim())`), and `half_to_float` is false (that mode is
>    unsupported here).
> 2. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK: InvalidArgument, returns
>    `out`).
> 3. Check `in` and `out` share a dim order (ET_KERNEL_CHECK: InvalidArgument,
>    returns `out`).
> 4. Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)` (the number of
>    non-trivial dimensions used for indexing).
> 5. Dispatch on `in.scalar_type()` over FLOATHBF16 {Half, Float, Double,
>    BFloat16}; let CTYPE be the element type. Accumulation type ACC is `float`
>    when CTYPE is Half or BFloat16, otherwise CTYPE (matches ATen acc_type,
>    avoiding saturation of Half/BFloat16 exp-sums).
> 6. Iterate over every slice of `in` along `dim` (via apply_over_dim: for each
>    combination of the other indices, walk `size` elements at `stride` starting
>    at a base offset). For each slice:
>    a. Compute `max_in` = maximum of the CTYPE values in the slice (reduce with
>       `std::max`).
>    b. Compute `exp_sum` (in ACC) = sum over the slice of
>       `exp((ACC)val - (ACC)max_in)`.
>    c. Compute `log_sum = std::log(exp_sum)` (in ACC).
>    d. For each element `val` in the slice, write to the corresponding `out`
>       position `(CTYPE)((ACC)val - (ACC)max_in - log_sum)`.
> 7. Return `out`. The max-subtraction is for numerical stability; results are
>    unchanged mathematically.

