# kernels/optimized/cpu/op_grid_sampler_2d_fp16_hw.cpp

> [spec:et:def:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn]
> inline void bilinear_all_channels_fp16_hw_sample( const __fp16* input_n, __fp16* output_n, int C, int H_in, int W_in, int H_out, int W_out, int h_out, int w_out, float gx, float gy)

> [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn]
> Hardware-fp16 bilinear sample for one output spatial location `(h_out, w_out)`,
> all `C` channels, one batch. Byte-for-byte the same algorithm as
> `bilinear_all_channels_f16_sw` (op_grid_sampler_2d.cpp) — the only difference is
> that fp16<->fp32 conversion uses the ARMv8.2 `+fp16` hardware NEON instructions
> (`vld1_f16`/`vcvt_f32_f16` on load, `vcvt_f16_f32`/`vst1_f16` on store) instead
> of the portable software conversion. Given unnormalized `gx`, `gy`:
> 1. `x0=(int)floor(gx)`, `y0=(int)floor(gy)`, `x1=x0+1`, `y1=y0+1`,
>    `fx=gx-(float)x0`, `fy=gy-(float)y0`.
> 2. Corner bounds flags via `(unsigned)coord < (unsigned)limit`:
>    `tl_v`(x0,y0), `tr_v`(x1,y0), `bl_v`(x0,y1), `br_v`(x1,y1).
> 3. Offsets `off_tl=y0*W_in+x0`, `off_tr=y0*W_in+x1`, `off_bl=y1*W_in+x0`,
>    `off_br=y1*W_in+x1`; `spatial_in=H_in*W_in`, `spatial_out=H_out*W_out`,
>    `out_off=h_out*W_out+w_out`.
> 4. fp32 weights `w_tl=(1-fx)*(1-fy)`, `w_tr=fx*(1-fy)`, `w_bl=(1-fx)*fy`,
>    `w_br=fx*fy`.
> 5. Blocked-by-4 channels: gather 4 corner fp16 values per channel into `tl/tr/
>    bl/br` (0 where the flag is false), convert to fp32, accumulate
>    `w_tl*tl + w_tr*tr + w_bl*bl + w_br*br` in fp32, convert back to fp16, store
>    each lane to `output_n[(c+k)*spatial_out + out_off]`.
> 6. Scalar tail per channel: `v=0; if(tl_v) v += w_tl*(float)p[off_tl]; ...`,
>    store `(__fp16)v`.
> Rust DEVIATION: on aarch64 with the `fp16` feature at build time this could use
> `core::arch::aarch64` fp16 intrinsics, but per the porting note it is ported as
> a scalar `f16` (half::f16) fallback computing all math in fp32 — numerically
> identical to the C++ (which also does all math in fp32). The whole file is gated
> `#[cfg(target_arch = "aarch64")]`.

> [spec:et:def:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn]
> void grid_sampler_2d_bilinear_fp16_hw( const void* input, const void* grid, void* output, int N, int C, int H_in, int W_in, int H_out, int W_out, bool align_corners)

> [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn]
> Entry point called by op_grid_sampler_2d's dispatcher only after a runtime
> `cpuinfo_has_arm_neon_fp16()` check succeeds. `input`/`grid`/`output` are raw
> void buffers reinterpreted as `__fp16*` (in Rust: `*const/*mut half::f16`).
> `spatial_in=H_in*W_in`, `spatial_out=H_out*W_out`. For each batch `n` in `0..N`:
> `input_n = in + n*C*spatial_in`, `output_n = out + n*C*spatial_out`,
> `grid_n = gd + n*H_out*W_out*2`. For each output row `h` (prefetch of next grid
> row is a no-op in the port), each column `w`: read
> `gx=(float)grid_n[(h*W_out+w)*2]`, `gy=(float)grid_n[(h*W_out+w)*2+1]`,
> unnormalize (align_corners: `(g+1)*(size-1)*0.5`; else `(g+1)*size*0.5-0.5`),
> then call `bilinear_all_channels_fp16_hw_sample(input_n, output_n, C, H_in,
> W_in, H_out, W_out, h, w, gx, gy)`. Same structure as `grid_sampler_2d_neon`
> but hardwired to the fp16 hardware sample fn and raw pointer casts.
> DEVIATION: `reinterpret_cast<__fp16*>` -> `as *const/*mut half::f16`; whole file
> `#[cfg(target_arch = "aarch64")]`.
