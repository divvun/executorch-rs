# kernels/portable/cpu/op_max_pool2d_with_indices.cpp

> [spec:et:def:op-max-pool2d-with-indices.torch.executor.native.max-pool2d-with-indices-out-fn]
> std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, Tensor& out, Tensor& indices)

> [spec:et:sem:op-max-pool2d-with-indices.torch.executor.native.max-pool2d-with-indices-out-fn]
> 2-D max pooling: for each output spatial location, takes the maximum over the
> corresponding `kernel_size` window of `in` (with `stride`, `padding`,
> `dilation`, and `ceil_mode`), writing the max values to `out` and the flat
> input index of each max to `indices`. Returns the tuple `{out, indices}`.
>
> Steps (every ET_KERNEL_CHECK below, on failure, sets Error::InvalidArgument on
> the context and returns the tuple `{out, indices}` unchanged):
> 1. Validate via `check_max_pool2d_with_indices_args(in, kernel_size, stride,
>    padding, dilation, ceil_mode, out, indices)`: `in` is 3-D (CHW) or 4-D
>    (NCHW); `kernel_size`/`stride`/`padding`/`dilation` have valid lengths (1 or
>    2, stride may be empty meaning it defaults to `kernel_size`) and positive
>    values; padding ≤ half the kernel; `out` dtype equals `in` dtype;
>    `indices` dtype is Long.
> 2. Compute the output spatial size into `output_sizes`/`output_ndim` via
>    `get_max_pool2d_with_indices_out_target_size` (standard pooling formula:
>    `out = floor_or_ceil((in + 2*pad - dilation*(kernel-1) - 1)/stride) + 1`,
>    ceil vs floor per `ceil_mode`, with the last-window-starting-in-padding
>    adjustment).
> 3. Check `output_size_is_valid({output_sizes, output_ndim}, 2)` (each of the 2
>    spatial output dims ≥ 1).
> 4. Resize `out` to `{output_sizes, output_ndim}`; resize `indices` likewise.
> 5. Dispatch on `in.scalar_type()` over REALHBF16 {Byte, Char, Short, Int, Long,
>    Half, Float, Double, BFloat16} (CTYPE). For each output position, reduce
>    over its (dilated, padded) kernel window using the 2-D kernel reduce helper
>    with `include_pad = false` (padding cells are skipped, not counted): the
>    accumulator is `(value, flat_input_index)`; for each in-window input value
>    `in_val` at flat input index `in_idx`, if `in_val > accum` keep
>    `(in_val, in_idx)` else keep the current accumulator (strict `>`, so the
>    earliest max index wins on ties). Max pooling has no post-processing step
>    (the accumulated value is written directly). Write the max value to `out`
>    and the winning flat input index to `indices`.
> 6. Return `{out, indices}`.

