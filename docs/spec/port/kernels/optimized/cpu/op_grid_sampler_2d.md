# kernels/optimized/cpu/op_grid_sampler_2d.cpp

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f16-sw-fn]
> inline void bilinear_all_channels_f16_sw( const c10::Half* input_n, c10::Half* output_n, int C, int H_in, int W_in, int H_out, int W_out, int h_out, int w_out, float gx, float gy)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f16-sw-fn]
> Software-conversion fp16 bilinear sample for one output spatial location
> `(h_out, w_out)`, all `C` channels, for one batch. `input_n`/`output_n` point
> at the start of the current batch's channel-0 plane. Given already-unnormalized
> source coordinates `gx`, `gy` (float):
> 1. `x0 = (int)floor(gx)`, `y0 = (int)floor(gy)`, `x1 = x0+1`, `y1 = y0+1`.
>    `fx = gx - (float)x0`, `fy = gy - (float)y0` (the fractional offsets).
> 2. Bounds flags for each of the 4 bilinear corners, using the C++ idiom
>    `(unsigned)coord < (unsigned)limit` — a signed-negative index wraps to a
>    huge unsigned value and so tests false (out of bounds):
>    - `tl_v = (unsigned)x0 < (unsigned)W_in && (unsigned)y0 < (unsigned)H_in`
>    - `tr_v = (unsigned)x1 < (unsigned)W_in && (unsigned)y0 < (unsigned)H_in`
>    - `bl_v = (unsigned)x0 < (unsigned)W_in && (unsigned)y1 < (unsigned)H_in`
>    - `br_v = (unsigned)x1 < (unsigned)W_in && (unsigned)y1 < (unsigned)H_in`
> 3. Per-plane element offsets: `off_tl = y0*W_in + x0`, `off_tr = y0*W_in + x1`,
>    `off_bl = y1*W_in + x0`, `off_br = y1*W_in + x1`. `spatial_in = H_in*W_in`,
>    `spatial_out = H_out*W_out`, `out_off = h_out*W_out + w_out`.
> 4. Bilinear weights in fp32: `w_tl = (1-fx)*(1-fy)`, `w_tr = fx*(1-fy)`,
>    `w_bl = (1-fx)*fy`, `w_br = fx*fy`.
> 5. Process channels in blocks of 4 (`for c; c+3 < C; c += 4`): for each of the
>    4 planes `p0..p3` at `input_n + (c+k)*spatial_in`, gather the 4 corner
>    values (fp16->fp32 via c10::Half's portable conversion) into fp32 lane
>    arrays `tl/tr/bl/br` (0.0 where the corresponding `*_v` flag is false), then
>    `result = w_tl*tl + w_tr*tr + w_bl*bl + w_br*br` per lane (FMA), and store
>    each lane fp32->fp16 to `output_n[(c+k)*spatial_out + out_off]`.
> 6. Scalar tail (`for ; c < C; ++c`): `v = 0; if(tl_v) v += w_tl*p[off_tl]; ...`
>    (each addend gated by its `*_v`, fp16->fp32 on read), store `Half(v)` to
>    `output_n[c*spatial_out + out_off]`.
> The Rust port collapses the NEON f32x4 lanes to a scalar 4-iteration inner
> compute (DEVIATION: Vectorized -> scalar loop); numerically identical because
> the C++ math is already all fp32.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f32-fn]
> inline void bilinear_all_channels_f32( const float* input_n, float* output_n, int C, int H_in, int W_in, int H_out, int W_out, int h_out, int w_out, float gx, float gy)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f32-fn]
> Identical algorithm to `bilinear_all_channels_f16_sw` but with `float`
> input/output (no fp16<->fp32 conversion). Given unnormalized `gx`, `gy`:
> compute `x0,y0,x1,y1,fx,fy`; the four `tl_v/tr_v/bl_v/br_v` bounds flags via
> the `(unsigned)coord < (unsigned)limit` idiom; the four plane offsets
> `off_tl/off_tr/off_bl/off_br`; `spatial_in`, `spatial_out`, `out_off`; the four
> weights `w_tl=(1-fx)*(1-fy)`, `w_tr=fx*(1-fy)`, `w_bl=(1-fx)*fy`, `w_br=fx*fy`.
> Blocked-by-4 channel loop gathers 4 corner values per channel (0.0 where the
> flag is false) and accumulates `w_tl*tl + w_tr*tr + w_bl*bl + w_br*br`, storing
> each lane. Scalar tail does the same per-channel with `v` gated by the flags.
> DEVIATION: NEON f32x4 -> scalar 4-lane loop; identical results.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn]
> void grid_sampler_2d_neon( const SCALAR* input, const SCALAR* grid, SCALAR* output, int N, int C, int H_in, int W_in, int H_out, int W_out, bool align_corners, SampleFn sample_fn)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn]
> Generic driver over the fast (bilinear + zeros padding) path for a contiguous
> NCHW input, `[N,H_out,W_out,2]` grid, NCHW output. `spatial_in = H_in*W_in`,
> `spatial_out = H_out*W_out`. For each batch `n` in `0..N`: set
> `input_n = input + n*C*spatial_in`, `output_n = output + n*C*spatial_out`,
> `grid_n = grid + n*H_out*W_out*2`. For each output row `h` in `0..H_out`
> (prefetch of the next grid row is a no-op in the port), for each column `w` in
> `0..W_out`: read `gx = (float)grid_n[(h*W_out+w)*2]`,
> `gy = (float)grid_n[(h*W_out+w)*2 + 1]`, unnormalize to source pixel space:
>   - align_corners: `gx = (gx+1)*(W_in-1)*0.5`, `gy = (gy+1)*(H_in-1)*0.5`;
>   - else:          `gx = (gx+1)*W_in*0.5 - 0.5`, `gy = (gy+1)*H_in*0.5 - 0.5`.
> Then call `sample_fn(input_n, output_n, C, H_in, W_in, H_out, W_out, h, w,
> gx, gy)`. `SCALAR` is `float` or `c10::Half`; `sample_fn` is the matching
> `bilinear_all_channels_*`. The Rust port is generic over `SCALAR: NeonScalar`
> (providing `to_f32`) and takes the sample fn as an `Fn` pointer/closure.

> [spec:et:def:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn]
> Tensor& opt_grid_sampler_2d_out( KernelRuntimeContext& ctx, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, Tensor& out)

> [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn]
> Optimized `grid_sampler_2d.out`. Decides eligibility for the fast NEON path and
> otherwise delegates to the portable `grid_sampler_2d_out`.
> 1. `fast_eligible` = `input.dim()==4 && grid.dim()==4 && grid.size(3)==2 &&
>    input.size(0)==grid.size(0) && tensor_is_default_dim_order(input) &&
>    tensor_is_default_dim_order(grid) && tensor_is_default_dim_order(out) &&
>    tensor_is_contiguous(input) && tensor_is_contiguous(grid) &&
>    tensor_is_contiguous(out)`. The fast paths index buffers directly assuming
>    contiguous default-dim-order NCHW; anything else falls back.
> 2. `dtypes_match` = `input.scalar_type()==grid.scalar_type() &&
>    input.scalar_type()==out.scalar_type()`. Mixed-dtype calls are rejected up
>    front because the fast paths do unchecked pointer casts to a single dtype.
> 3. If `interpolation_mode != 0` (not bilinear) OR `padding_mode != 0` (not
>    zeros) OR `!fast_eligible` OR `!dtypes_match`: return
>    `grid_sampler_2d_out(ctx, input, grid, interpolation_mode, padding_mode,
>    align_corners, out)` (portable fallback).
> 4. On non-aarch64 targets: always return the portable fallback (same call).
> 5. On aarch64: read `N=input.size(0)`, `C=input.size(1)`, `H_in=input.size(2)`,
>    `W_in=input.size(3)`, `H_out=grid.size(1)`, `W_out=grid.size(2)` as ints.
>    - `Float` dtype: call `grid_sampler_2d_neon::<float>(...)` with
>      `bilinear_all_channels_f32`; return out.
>    - `Half` dtype: if `cpuinfo_initialize() && cpuinfo_has_arm_neon_fp16()`
>      (Rust: `std::arch::is_aarch64_feature_detected!("fp16")`), call the
>      hardware path `grid_sampler_2d_bilinear_fp16_hw(input.const_data_ptr(),
>      grid.const_data_ptr(), out.mutable_data_ptr(), N,C,H_in,W_in,H_out,W_out,
>      align_corners)` on the raw void buffers; else call
>      `grid_sampler_2d_neon::<c10::Half>(...)` with `bilinear_all_channels_f16_sw`.
>      return out.
>    - Any other dtype: return the portable fallback.
> DEVIATION: cpuinfo runtime dispatch -> `is_aarch64_feature_detected!`; the whole
> aarch64 block is `#[cfg(target_arch = "aarch64")]`, with the non-aarch64 arm
> unconditionally delegating to portable.
