# kernels/portable/cpu/op_pixel_unshuffle.cpp

> [spec:et:def:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn]
> void pixel_unshuffle_impl( const Tensor& in, int64_t downscale_factor, Tensor& out)

> [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn]
> Dtype-agnostic (byte-copy) inverse of pixel-shuffle: rearranges `in` [*, c, H*S, W*S] into `out` [*, c*S*S, H, W] for `downscale_factor` S. Copies element-by-element with `memcpy` of `elem_size` bytes; no numeric conversion. No return value.
>
> Setup: `elem_size = in.element_size()`; `leading_dims = getLeadingDims(in, in.dim()-3)`; `channels = out.size(in.dim()-3)`, `height = out.size(in.dim()-2)`, `width = out.size(in.dim()-1)` (note: shapes read from the OUTPUT tensor); `S = downscale_factor`; `sub_channels = channels / (S*S)`.
> Output strides (output viewed as [n, c=sub_channels, s1=S, s2=S, h=height, w=width]): `stride_n = channels*height*width`, `stride_c = S*S*height*width`, `stride_s1 = S*height*width`, `stride_s2 = height*width`, `stride_h = width`.
>
> Algorithm: input is read contiguously (running index `i` from 0). Six nested loops in the same order as pixel-shuffle `n → c(sub_channels) → h(height) → s1(S) → w(width) → s2(S)`; for each combination compute `output_offset = n*stride_n + c*stride_c + s1*stride_s1 + s2*stride_s2 + h*stride_h + w` and `memcpy(out_data + output_offset*elem_size, in_data + i*elem_size, elem_size)`, then `i++`. This scatters the contiguous input (logical [n, c, h, s1, w, s2]) into the output logical shape [n, c, s1, s2, h, w] — the exact inverse of pixel-shuffle.

> [spec:et:def:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn]
> Tensor& pixel_unshuffle_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t downscale_factor, Tensor& out)

> [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn]
> Entry point for `pixel_unshuffle.out`: validates, resizes `out`, and calls the byte-copy worker. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_pixel_unshuffle_args(in, downscale_factor, out)` per `[spec:et:sem:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn]` (requires `in`/`out` same dtype, `in.dim() >= 3`, `downscale_factor > 0`, and height and width both divisible by `downscale_factor`); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. Compute `expected_out_size`/`expected_out_dim` via `get_pixel_unshuffle_out_target_size` per `[spec:et:sem:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn]` (out shape: leading dims unchanged, channels → `C*S*S`, height → `H/S`, width → `W/S`).
> 3. ET_KERNEL_CHECK: `resize_tensor(out, {expected_out_size, expected_out_dim})` == Ok; else InvalidArgument, return `out`.
> 4. Call `pixel_unshuffle_impl(in, downscale_factor, out)` per `[spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn]`. (Note: unlike pixel_shuffle.out, this op does not separately check dim order / default dim order.)
> 5. Return `out`.

