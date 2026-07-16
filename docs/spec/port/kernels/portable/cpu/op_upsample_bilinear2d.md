# kernels/portable/cpu/op_upsample_bilinear2d.cpp

> [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn]
> void upsample_bilinear2d_kernel_impl( KernelRuntimeContext& ctx, const Tensor& in, bool align_corners, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn]
> Dim-order dispatcher, templated on the element type `CTYPE`. It inspects
> `in`'s dim order (from `in.dim_order()`):
> - If the dim order is the contiguous (NCHW) order, calls
>   `[spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]`
>   with `(in, align_corners, scale_h, scale_w, out)`.
> - Else if the dim order is channels-last (NHWC), calls
>   `[spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn]`
>   with the same arguments.
> - Otherwise (unreachable given the arg checks in the caller), logs an error
>   ("Unsupported dim order") and calls `ctx.fail(Error::InvalidArgument)`,
>   writing nothing to `out`.
> Returns void; the chosen kernel writes results into `out` in place.

> [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]
> void upsample_bilinear2d_kernel_impl_nchw( const Tensor& in, bool align_corners, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]
> Bilinear upsample for an NCHW (contiguous) input tensor, templated on
> element type `CTYPE`. `scale_h`/`scale_w` are the precomputed source/dest
> ratios (see `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]`).
> `in` and `out` are both rank-4 with the same dim[0] (N) and dim[1] (C).
>
> Obtain `in_data = in.const_data_ptr<CTYPE>()` and
> `out_data = out.mutable_data_ptr<CTYPE>()`. Maintain a running plane pointer
> `in_plane` initialized to `in_data`, and a running `out_data` write cursor
> that advances by 1 element per output write (so output is written in
> row-major/contiguous order).
>
> Iterate `n` over [0, out.size(0)), then `c` over [0, out.size(1)); inside
> that, iterate `h` over [0, out.size(2)):
> - Compute source height indices/lambdas via
>   `[spec:et:sem:upsample-util.torch.executor.compute-source-index-and-lambda-fn]`
>   giving `(in_h1, in_h2, weight_h, inv_weight_h)` from
>   `(scale_h, h, in.sizes()[2], out.sizes()[2], align_corners)`. Here `weight_h`
>   is `lambda0` (weight of the top row) and `inv_weight_h` is `lambda1`
>   (weight of the bottom row).
> - Then iterate `w` over [0, out.size(3)):
>   - Compute source width indices/lambdas similarly giving
>     `(in_w1, in_w2, weight_w, inv_weight_w)` from `(scale_w, w, in.sizes()[3],
>     out.sizes()[3], align_corners)`; `weight_w` is the left-column weight and
>     `inv_weight_w` the right-column weight.
>   - Gather the four neighbors from the current plane using `in`'s strides for
>     dims 2 and 3:
>     `top_left    = in_plane[in_h1*strides[2] + in_w1*strides[3]]`,
>     `top_right   = in_plane[in_h1*strides[2] + in_w2*strides[3]]`,
>     `bottom_left = in_plane[in_h2*strides[2] + in_w1*strides[3]]`,
>     `bottom_right= in_plane[in_h2*strides[2] + in_w2*strides[3]]`.
>   - Interpolate in width then height (all arithmetic in `CTYPE`):
>     `top = top_left*weight_w + top_right*inv_weight_w`,
>     `bottom = bottom_left*weight_w + bottom_right*inv_weight_w`,
>     `val = top*weight_h + bottom*inv_weight_h`.
>   - Write `val` to `*out_data`, then advance `out_data` by 1.
> - After finishing all `h`,`w` for a given `(n,c)`, advance `in_plane` by
>   `in.strides()[1]` (move to the next channel plane).
> Returns void. Empty output (any out dim size 0) produces no writes.

> [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn]
> void upsample_bilinear2d_kernel_impl_nhwc( const Tensor& in, bool align_corners, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn]
> Bilinear upsample for an NHWC (channels-last) input tensor, templated on
> element type `CTYPE`. Same math as the NCHW variant
> (`[spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]`)
> but with the channel loop innermost so output is written in channels-last
> contiguous order.
>
> Obtain `in_data = in.const_data_ptr<CTYPE>()` and
> `out_data = out.mutable_data_ptr<CTYPE>()`; `out_data` advances by 1 element
> per write.
>
> Iterate `n` over [0, out.size(0)); inside, iterate `h` over [0, out.size(2)):
> - Compute `(in_h1, in_h2, weight_h, inv_weight_h)` for height exactly as in
>   the NCHW variant via
>   `[spec:et:sem:upsample-util.torch.executor.compute-source-index-and-lambda-fn]`.
> - Iterate `w` over [0, out.size(3)):
>   - Compute `(in_w1, in_w2, weight_w, inv_weight_w)` for width the same way.
>   - Iterate `c` over [0, out.size(1)):
>     - Gather the four neighbors, now including the channel offset via
>       `in.strides()[1]`:
>       `top_left    = in_data[in_h1*strides[2] + in_w1*strides[3] + c*strides[1]]`,
>       `top_right   = in_data[in_h1*strides[2] + in_w2*strides[3] + c*strides[1]]`,
>       `bottom_left = in_data[in_h2*strides[2] + in_w1*strides[3] + c*strides[1]]`,
>       `bottom_right= in_data[in_h2*strides[2] + in_w2*strides[3] + c*strides[1]]`.
>     - Interpolate `top`, `bottom`, `val` identically to the NCHW variant, in
>       `CTYPE` arithmetic, and write `val` to `*out_data`; advance `out_data`.
> - After finishing all `(h,w,c)` for a given `n`, advance `in_data` by
>   `in.strides()[0]` (next batch).
> Returns void.

> [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn]
> Tensor& upsample_bilinear2d_vec_out( KernelRuntimeContext& ctx, const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t> output_size, bool align_corners, const executorch::aten::OptionalArrayRef<double> scale_factors, Tensor& ...

> [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn]
> Entry point for `upsample_bilinear2d.vec_out`. Arguments: input `in`, optional
> `output_size` (target H,W), `align_corners`, optional `scale_factors`, and the
> preallocated `out`. Returns `out`.
>
> Validation and setup (each `ET_KERNEL_CHECK*` on failure sets the given Error
> on `ctx` and returns `out` unchanged):
> 1. `ET_KERNEL_CHECK(check_upsample_bilinear2d_args(in, output_size,
>    align_corners, scale_factors, out), InvalidArgument)`. This enforces (per
>    the upsample_util checks) that `in`/`out` have the same dtype, are rank 4
>    with matching dim[0] and dim[1], share NHWC-or-NCHW dim order, and that
>    exactly one of `output_size`/`scale_factors` is provided with size 2.
> 2. Declare `double scale_h, scale_w`. Call
>    `resize_upsample_2d(in, output_size, scale_factors, scale_h, scale_w, out)`;
>    it resizes `out` to `[N, C, out_H, out_W]` and writes the per-axis scale
>    factors into `scale_h`/`scale_w`. `ET_KERNEL_CHECK_MSG` that it returns
>    `Error::Ok`, else fail InvalidArgument ("Failed to resize output tensor").
> 3. Compute the kernel ratios:
>    `kernel_scale_h = area_pixel_compute_scale<double>(in.sizes()[2],
>    out.sizes()[2], align_corners, scale_h)` and likewise `kernel_scale_w` from
>    dim 3, per `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]`.
> 4. Dispatch on `in.scalar_type()` with `ET_SWITCH_REALHBF16_TYPES` (accepted
>    input dtypes: the real floating and half/bfloat16 set — Byte, Char, Short,
>    Int, Long, Float, Double, Half, BFloat16; i.e. real numeric types plus Half
>    and BFloat16, excluding Bool and complex). For the selected `CTYPE`, call
>    `[spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn]`
>    with `(ctx, in, align_corners, kernel_scale_h, kernel_scale_w, out)`.
>    An unsupported dtype makes the switch fail on `ctx` and return `out`.
> 5. Return `out`.

