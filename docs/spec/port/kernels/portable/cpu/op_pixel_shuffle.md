# kernels/portable/cpu/op_pixel_shuffle.cpp

> [spec:et:def:op-pixel-shuffle.torch.executor.native.pixel-shuffle-impl-fn]
> void pixel_shuffle_impl(const Tensor& in, int64_t upscale_factor, Tensor& out)

> [spec:et:sem:op-pixel-shuffle.torch.executor.native.pixel-shuffle-impl-fn]
> Dtype-agnostic (byte-copy) worker that rearranges `in` [*, C, H, W] into `out` [*, C/(S*S), H*S, W*S] for `upscale_factor` S. Copies element-by-element with `memcpy` of `elem_size` bytes; no numeric conversion. No return value.
>
> Setup: `elem_size = in.element_size()`; `leading_dims = getLeadingDims(in, in.dim()-3)` (product of all dims before the last 3); `channels = in.size(in.dim()-3)`; `height = in.size(in.dim()-2)`; `width = in.size(in.dim()-1)`; `sub_channels = channels / (S*S)`; `S = upscale_factor`.
> Input strides (input viewed as [n, c=sub_channels, s1=S, s2=S, h=height, w=width]): `stride_n = channels*height*width`, `stride_c = S*S*height*width`, `stride_s1 = S*height*width`, `stride_s2 = height*width`, `stride_h = width`.
>
> Algorithm: output is written contiguously (running index `i` from 0). Six nested loops in output order `n → c(sub_channels) → h(height) → s1(S) → w(width) → s2(S)`; for each combination compute `input_offset = n*stride_n + c*stride_c + s1*stride_s1 + s2*stride_s2 + h*stride_h + w` and `memcpy(out_data + i*elem_size, in_data + input_offset*elem_size, elem_size)`, then `i++`. This realizes the mapping from input logical shape [n, c, s1, s2, h, w] to output logical shape [n, c, h, s1, w, s2] (i.e. the channel's S*S block is interleaved into the H,W spatial grid).

> [spec:et:def:op-pixel-shuffle.torch.executor.native.pixel-shuffle-out-fn]
> Tensor& pixel_shuffle_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t upscale_factor, Tensor& out)

> [spec:et:sem:op-pixel-shuffle.torch.executor.native.pixel-shuffle-out-fn]
> Entry point for `pixel_shuffle.out`: validates, resizes `out`, and calls the byte-copy worker. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_pixel_shuffle_args(in, upscale_factor, out)` per `[spec:et:sem:copy-ops-util.torch.executor.check-pixel-shuffle-args-fn]` (requires `in`/`out` same dtype, `in.dim() >= 3`, `upscale_factor > 0`, and `channels` divisible by `upscale_factor^2`); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)`; else InvalidArgument, return `out`.
> 4. Compute `expected_out_size`/`expected_out_dim` via `get_pixel_shuffle_out_target_size` per `[spec:et:sem:copy-ops-util.torch.executor.get-pixel-shuffle-out-target-size-fn]`; ET_KERNEL_CHECK its boolean result (it may fail); else InvalidArgument, return `out`. (Out shape: leading dims unchanged, channels → `C/(S*S)`, height → `H*S`, width → `W*S`.)
> 5. ET_KERNEL_CHECK: `resize_tensor(out, {expected_out_size, expected_out_dim})` == Ok; else InvalidArgument, return `out`.
> 6. Call `pixel_shuffle_impl(in, upscale_factor, out)` per `[spec:et:sem:op-pixel-shuffle.torch.executor.native.pixel-shuffle-impl-fn]`.
> 7. Return `out`.

