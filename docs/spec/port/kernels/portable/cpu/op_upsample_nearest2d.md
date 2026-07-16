# kernels/portable/cpu/op_upsample_nearest2d.cpp

> [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn]
> void upsample_nearest2d_kernel_impl( KernelRuntimeContext& ctx, const Tensor& in, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn]
> Dim-order dispatcher templated on element type `CTYPE`. Inspects `in`'s dim
> order:
> - Contiguous (NCHW): calls
>   `[spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn]`
>   with `(in, scale_h, scale_w, out)`.
> - Channels-last (NHWC): calls
>   `[spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn]`.
> - Otherwise (unreachable given arg checks): logs "Unsupported dim order" and
>   calls `ctx.fail(Error::InvalidArgument)`, writing nothing to `out`.
> Returns void.

> [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn]
> void upsample_nearest2d_kernel_impl_nchw( const Tensor& in, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn]
> Nearest-neighbor upsample for an NCHW (contiguous) input, templated on
> `CTYPE`. `scale_h`/`scale_w` are the precomputed source/dest ratios.
>
> Obtain `in_data`/`out_data` pointers; `out_data` advances by 1 per write
> (output written in row-major order). Maintain `in_plane` starting at
> `in_data`. For `n` in [0,out.size(0)), `c` in [0,out.size(1)), `h` in
> [0,out.size(2)), `w` in [0,out.size(3)):
> - `in_h = nearest_neighbor_compute_source_index(scale_h, h, in.sizes()[2])`
>   and `in_w = nearest_neighbor_compute_source_index(scale_w, w,
>   in.sizes()[3])` per
>   `[spec:et:sem:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]`
>   (i.e. `min(floor(dst*scale), input_size-1)`).
> - Write `*out_data = in_plane[in_h*in.strides()[2] + in_w*in.strides()[3]]`;
>   advance `out_data`.
> After each `(n,c)` inner block, advance `in_plane` by `in.strides()[1]`.
> Returns void.

> [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn]
> void upsample_nearest2d_kernel_impl_nhwc( const Tensor& in, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn]
> Nearest-neighbor upsample for an NHWC (channels-last) input, templated on
> `CTYPE`, with the channel loop innermost so output is channels-last
> contiguous. `out_data` advances by 1 per write.
>
> For `n` in [0,out.size(0)): for `h` in [0,out.size(2)) compute
> `in_h = nearest_neighbor_compute_source_index(scale_h, h, in.sizes()[2])`; for
> `w` in [0,out.size(3)) compute `in_w = nearest_neighbor_compute_source_index(
> scale_w, w, in.sizes()[3])` (see
> `[spec:et:sem:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]`);
> then for `c` in [0,out.size(1)) write
> `*out_data = in_data[in_h*in.strides()[2] + in_w*in.strides()[3] +
> c*in.strides()[1]]` and advance `out_data`.
> After each batch `n`, advance `in_data` by `in.strides()[0]`.
> Returns void.

> [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn]
> Tensor& upsample_nearest2d_vec_out( KernelRuntimeContext& ctx, const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t> output_size, const executorch::aten::OptionalArrayRef<double> scale_factors, Tensor& out)

> [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn]
> Entry point for `upsample_nearest2d.vec_out`. Arguments: `in`, optional
> `output_size`, optional `scale_factors`, and preallocated `out`. Returns
> `out`.
>
> 1. `ET_KERNEL_CHECK(check_upsample_nearest2d_args(in, output_size,
>    scale_factors, out), InvalidArgument)` — enforces same dtype, rank 4 with
>    matching dim[0]/dim[1], NHWC-or-NCHW dim order, and exactly one of
>    output_size/scale_factors (size 2). On failure sets InvalidArgument on
>    `ctx` and returns `out` unchanged.
> 2. Declare `double scale_h, scale_w`; call
>    `resize_upsample_2d(in, output_size, scale_factors, scale_h, scale_w, out)`
>    which resizes `out` and fills the scales. `ET_KERNEL_CHECK_MSG` on
>    `Error::Ok` else fail InvalidArgument ("Failed to resize output tensor").
> 3. Compute `kernel_scale_h = area_pixel_compute_scale<double>(in.sizes()[2],
>    out.sizes()[2], false, scale_h)` and `kernel_scale_w` from dim 3 — note
>    `align_corners` is hard-coded `false` here (nearest has no align_corners),
>    so these reduce to `compute_scales_value` (1/scale if a scale was given,
>    else input/output). See
>    `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]`.
> 4. Dispatch on `in.scalar_type()` with `ET_SWITCH_REALHBF16_TYPES` (Byte,
>    Char, Short, Int, Long, Float, Double, Half, BFloat16 — real numeric plus
>    Half and BFloat16, excluding Bool and complex). For the selected `CTYPE`
>    call
>    `[spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn]`
>    with `(ctx, in, kernel_scale_h, kernel_scale_w, out)`.
> 5. Return `out`.

