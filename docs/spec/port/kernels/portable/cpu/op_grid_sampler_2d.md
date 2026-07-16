# kernels/portable/cpu/op_grid_sampler_2d.cpp

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn]
> void grid_sample_2d_bicubic_kernel_impl_nchw( const Tensor& in, const Tensor& grid, GridSamplerPadding padding_mode, bool align_corners, Tensor& out)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn]
> Templated on `CTYPE`. Bicubic grid sampling, NCHW in/out against a
> [N, H_out, W_out, 2] grid. Math in `CTYPE`. Same dimensions and stride-based
> addressing as the other kernels.
>
> For each `n`, `c`, and output pixel `(h, w)`:
> 1. Read grid `x`, `y`.
> 2. Unlike bilinear/nearest, compute raw unnormalized coordinates WITHOUT padding
>    applied: `ix = grid_sampler_unnormalize(x, inp_W, align_corners)`,
>    `iy = grid_sampler_unnormalize(y, inp_H, align_corners)` per
>    `[spec:et:sem:grid-sampler-2d-util.grid-sampler-unnormalize]`. Padding is
>    applied per-pixel during neighborhood fetch.
> 3. `ix_0 = floor(ix)`, `iy_0 = floor(iy)`; fractional parts `tx = ix - ix_0`,
>    `ty = iy - iy_0`. The 4x4 neighborhood spans column indices
>    `{ix_0-1, ix_0, ix_0+1, ix_0+2}` and row indices `{iy_0-1, iy_0, iy_0+1, iy_0+2}`.
> 4. Define `get_value_bounded(iy, ix)` returning the pixel value with
>    padding-mode-specific handling:
>    - Zeros: return `in[..., iy, ix]` if `within_bounds_2d`, else 0.
>    - Border: clamp `iy` to `[0, inp_H-1]` and `ix` to `[0, inp_W-1]`, return
>      that pixel.
>    - Reflection: reflect the coordinate with `reflect_coordinates` — bounds
>      `[0, 2*(inp_H-1)]`/`[0, 2*(inp_W-1)]` when `align_corners`, else
>      `[-1, 2*inp_H-1]`/`[-1, 2*inp_W-1]` — then clip with `clip_coordinates` for
>      safety, and return that pixel.
> 5. For each of the 4 rows (iy_m1, iy_0, iy_p1, iy_p2) fetch the 4 pixels across
>    the columns and interpolate in x with `cubic_interp1d(p0,p1,p2,p3,tx)` per
>    `[spec:et:sem:grid-sampler-2d-util.cubic-interp1d]`, producing `coefficients[0..3]`.
> 6. Interpolate those 4 row results in y: `out_val = cubic_interp1d(coefficients[0..3], ty)`.
> 7. Write `out_val` to `out` at `out_channel_offset + h*out.strides[2] + w*out.strides[3]`.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn]
> void grid_sample_2d_bilinear_kernel_impl_nchw( const Tensor& in, const Tensor& grid, GridSamplerPadding padding_mode, bool align_corners, Tensor& out)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn]
> Templated on `CTYPE`. Bilinear grid sampling of a NCHW input against a
> [N, H_out, W_out, 2] grid, writing NCHW output. Half/BFloat16 inputs do all
> internal math in fp32 (`ACC = float`); other types compute in `CTYPE` directly.
> Loads/stores stay in `CTYPE`.
>
> Dimensions: `N=in.size(0)`, `C=in.size(1)`, `inp_H=in.size(2)`,
> `inp_W=in.size(3)`; `out_H=out.size(2)`, `out_W=out.size(3)`. All addressing
> uses each tensor's own strides.
>
> For each `n` in `[0,N)`, each `c` in `[0,C)`, each output pixel `(h, w)` with
> `h` in `[0,out_H)`, `w` in `[0,out_W)`:
> 1. Read the normalized grid coordinate: at flat index
>    `grid_offset + h*grid.strides[1] + w*grid.strides[2]`, `x = grid[...]` and
>    `y = grid[... + grid.strides[3]]`, cast to `ACC`.
> 2. Convert to source pixel coordinates: `ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners)`
>    and `iy` analogously with `inp_H`, per
>    `[spec:et:sem:grid-sampler-2d-util.grid-sampler-compute-source-index]`
>    (applies unnormalize + padding-mode clipping/reflection).
> 3. Compute the four surrounding integer corners from `ix_nw=floor(ix)`,
>    `iy_nw=floor(iy)`: NE = (ix_nw+1, iy_nw), SW = (ix_nw, iy_nw+1),
>    SE = (ix_nw+1, iy_nw+1).
> 4. Bilinear weights (each in ACC): `nw = (ix_se-ix)*(iy_se-iy)`,
>    `ne = (ix-ix_sw)*(iy_sw-iy)`, `sw = (ix_ne-ix)*(iy-iy_ne)`,
>    `se = (ix-ix_nw)*(iy-iy_nw)`.
> 5. Accumulate `out_val` (ACC, starts 0):
>    - If `padding_mode == Zeros`: for each corner, only add its weighted pixel
>      value if `within_bounds_2d(iy_*, ix_*, inp_H, inp_W)` (i.e. within
>      [0,inp_H)×[0,inp_W)); out-of-bounds corners contribute nothing.
>    - Otherwise (Border/Reflection): clip each corner's ix/iy to the valid range
>      with `clip_coordinates` (a corner may fall outside even after source-index
>      clipping because of the +1), then add the weighted value for all four
>      corners unconditionally.
> 6. Write `out_val` cast back to `CTYPE` at
>    `out_channel_offset + h*out.strides[2] + w*out.strides[3]`.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn]
> void grid_sample_2d_nearest_kernel_impl_nchw( const Tensor& in, const Tensor& grid, GridSamplerPadding padding_mode, bool align_corners, Tensor& out)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn]
> Templated on `CTYPE`. Nearest-neighbor grid sampling, NCHW in/out against a
> [N, H_out, W_out, 2] grid. All math is in `CTYPE` (no fp32 promotion here).
> Same dimension extraction and stride-based addressing as the bilinear kernel.
>
> For each `n`, `c`, and output pixel `(h, w)`:
> 1. Read grid `x`, `y` (as in the bilinear kernel, no cast).
> 2. `ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners)`,
>    `iy` analogously with `inp_H`.
> 3. Round to the nearest pixel using `std::nearbyint` (round-to-even under the
>    current rounding mode, matching ATen — deliberately NOT `std::round`):
>    `ix_nearest = (int64_t)nearbyint(ix)`, `iy_nearest = (int64_t)nearbyint(iy)`.
> 4. `out_val` starts at 0. If `padding_mode == Zeros`: sample
>    `in[..., iy_nearest, ix_nearest]` only if
>    `within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)`, else leave 0.
>    Otherwise (Border/Reflection): clip `ix_nearest`, `iy_nearest` with
>    `clip_coordinates` (rounding can push a coordinate out of bounds) and sample
>    the clipped location unconditionally.
> 5. Write `out_val` to `out` at
>    `out_channel_offset + h*out.strides[2] + w*out.strides[3]`.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn]
> Tensor& grid_sampler_2d_out( KernelRuntimeContext& ctx, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, Tensor& out)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn]
> Public entry point for 2D grid sampling. `interpolation_mode` and
> `padding_mode` are integer enum codes; `align_corners` toggles corner
> alignment. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK_MSG: `check_grid_sampler_2d_args_and_resize_out(input, grid, out) == Error::Ok`
>    per `[spec:et:sem:grid-sampler-2d-util.check-grid-sampler-2d-args-and-resize-out]`,
>    which validates shapes/dtypes (input is 4D NCHW, grid is [N,H_out,W_out,2]
>    with matching batch and dtype) and resizes `out` to [N, C, H_out, W_out]. On
>    failure set Error::InvalidArgument with message "Failed to validate arguments
>    and resize output tensor" and return `out`.
> 2. Convert `interpolation_mode` to `GridSamplerInterpolation` and `padding_mode`
>    to `GridSamplerPadding` by static_cast of the integer code.
> 3. ET_KERNEL_CHECK: interpolation mode must be Bilinear, Nearest, or Bicubic;
>    else Error::InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: padding mode must be Zeros, Border, or Reflection; else
>    Error::InvalidArgument, return `out`.
> 5. Dispatch over `input.scalar_type()` in REALHBF16 = {Byte, Char, Short, Int,
>    Long, Half, Float, Double, BFloat16} (note: Bool excluded), then switch on
>    the interpolation mode and call the matching kernel with `<CTYPE>`:
>    - Bilinear → `[spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn]`
>    - Nearest → `[spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn]`
>    - Bicubic → `[spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn]`
>    each passed `(input, grid, padding, align_corners, out)`.
> 6. Return `out`.

