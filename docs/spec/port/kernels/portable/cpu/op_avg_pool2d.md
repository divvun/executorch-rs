# kernels/portable/cpu/op_avg_pool2d.cpp

> [spec:et:def:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn]
> Tensor& avg_pool2d_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override, Tensor& out)

> [spec:et:sem:op-avg-pool2d.torch.executor.native.avg-pool2d-out-fn]
> Computes 2D average pooling into `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_avg_pool2d_args(in, kernel_size, stride, padding,
>    ceil_mode, count_include_pad, divisor_override, out)` must hold (validates
>    input rank of 3 or 4, kernel_size/stride/padding list lengths of 1 or 2,
>    non-negative kernel/stride, padding <= kernel/2, dtype compatibility, etc.).
>    On failure set Error::InvalidArgument on the context and return `out`
>    unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)` (contiguous/NCHW dim
>    order); on failure set Error::InvalidArgument and return `out` unchanged.
> 4. Compute the output sizes with `get_avg_pool2d_out_target_size(in,
>    kernel_size, stride, padding, ceil_mode, output_sizes, &output_ndim)`:
>    output has the same leading dims as input; each pooled spatial dim is
>    `floor` (or `ceil` if `ceil_mode`) of
>    `(in_dim + 2*padding - kernel) / stride + 1`, with the ceil-mode
>    adjustment that a pooling window starting in the right/bottom padding
>    region is dropped.
> 5. ET_KERNEL_CHECK: `output_size_is_valid({output_sizes, output_ndim}, 2)`
>    (spatial dims must be >= 1); on failure set Error::InvalidArgument and
>    return `out` unchanged.
> 6. Resize `out` to `{output_sizes, output_ndim}`; if resize fails set
>    Error::InvalidArgument and return `out` unchanged.
> 7. Dispatch over `in.scalar_type()` restricted to FLOATHBF16 plus Long, i.e.
>    {Half, Float, Double, BFloat16, Long} (ET_SWITCH_FLOATHBF16_TYPES_AND(Long,
>    ...)). Other dtypes set Error::InvalidArgument and return `out` unchanged.
>    The compute type CTYPE equals the input/output dtype.
> 8. Run `apply_kernel_2d_reduce_then_map_fn<CTYPE>` (see the kernel_ops_util
>    reduce-then-map helper) with `count_include_pad`, `in`, `kernel_size`,
>    `stride`, `padding`, empty dilation `{}`, and `out`. For each output
>    position it iterates the pooling window over `in`, accumulating a running
>    sum (reduce lambda returns `in_val + accum`; the tracked index is unused and
>    set to 0). Then a map lambda finalizes each window:
>    - If `divisor_override` has a value, divide the accumulated sum by
>      `static_cast<CTYPE>(divisor_override)` regardless of window/pad counting.
>    - Otherwise divide by `static_cast<CTYPE>(count)`, where `count` is the
>      number of elements averaged: this includes padding cells when
>      `count_include_pad` is true and excludes them when false (the helper
>      determines `count` from the window/pad geometry).
> 9. Return `out`.

