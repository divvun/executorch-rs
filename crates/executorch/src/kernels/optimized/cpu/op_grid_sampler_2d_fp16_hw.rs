//! Literal port of kernels/optimized/cpu/op_grid_sampler_2d_fp16_hw.cpp.
//!
//! Hardware-fp16 variant of the NEON grid_sampler_2d.out bilinear + zeros-
//! padding fast path. In C++ this TU is compiled with `-march=armv8.2-a+fp16`
//! and emits hardware fp16 load/store/convert intrinsics; the dispatcher in
//! op_grid_sampler_2d only invokes it after a runtime `cpuinfo_has_arm_neon_fp16`
//! check. Math happens in fp32 regardless: load fp16, convert to fp32, do the
//! weighted-sum FMA chain in fp32, convert back to fp16 on store.
//!
//! DEVIATION (rust/PORTING.md optimized-kernels): the C++ NEON fp16 hardware
//! intrinsics (vld1_f16 / vcvt_f32_f16 / vst1_f16 / vcvt_f16_f32) are ported as a
//! scalar `half::f16` implementation computing every intermediate in fp32. The
//! C++ already performs all arithmetic in fp32, so this is numerically identical;
//! only the SIMD widening/narrowing collapses to scalar conversions. The whole
//! module is gated `#[cfg(target_arch = "aarch64")]` to mirror the C++ `#ifdef
//! __aarch64__`. See PORT-NOTE below re: not using std::arch fp16 intrinsics.

#![cfg(target_arch = "aarch64")]

use crate::runtime::core::portable_type::Half;

// PORT-NOTE: the C++ path exists to be *fast* via hardware fp16 NEON. std::arch
// aarch64 fp16 conversion intrinsics (e.g. vcvt_f32_f16) require the `fp16`
// target feature and `#[target_feature(enable = "fp16")]` unsafe fns; with only
// a runtime `is_aarch64_feature_detected!("fp16")` guard (no build-time feature)
// they are not stably callable here. Since TTS is unlikely to hit this path and
// half::f16 conversions already round-trip through fp32 identically, the port
// uses the scalar fallback. Swapping in the intrinsics is a future optimization.

// One output spatial location, all channels.
// [spec:et:def:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn]
// [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn]
#[inline]
unsafe fn bilinear_all_channels_fp16_hw_sample(
    input_n: *const Half,
    output_n: *mut Half,
    c_channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    h_out_idx: i32,
    w_out_idx: i32,
    gx: f32,
    gy: f32,
) {
    unsafe {
        let x0 = gx.floor() as i32;
        let y0 = gy.floor() as i32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;

        let tl_v = (x0 as u32) < (w_in as u32) && (y0 as u32) < (h_in as u32);
        let tr_v = (x1 as u32) < (w_in as u32) && (y0 as u32) < (h_in as u32);
        let bl_v = (x0 as u32) < (w_in as u32) && (y1 as u32) < (h_in as u32);
        let br_v = (x1 as u32) < (w_in as u32) && (y1 as u32) < (h_in as u32);

        let off_tl = y0 * w_in + x0;
        let off_tr = y0 * w_in + x1;
        let off_bl = y1 * w_in + x0;
        let off_br = y1 * w_in + x1;
        let spatial_in = h_in * w_in;
        let spatial_out = h_out * w_out;
        let out_off = h_out_idx * w_out + w_out_idx;

        let w_tl = (1.0f32 - fx) * (1.0f32 - fy);
        let w_tr = fx * (1.0f32 - fy);
        let w_bl = (1.0f32 - fx) * fy;
        let w_br = fx * fy;

        let mut c = 0i32;
        while c + 3 < c_channels {
            let p0 = input_n.offset(((c + 0) * spatial_in) as isize);
            let p1 = input_n.offset(((c + 1) * spatial_in) as isize);
            let p2 = input_n.offset(((c + 2) * spatial_in) as isize);
            let p3 = input_n.offset(((c + 3) * spatial_in) as isize);

            let mut tl = [0.0f32; 4];
            let mut tr = [0.0f32; 4];
            let mut bl = [0.0f32; 4];
            let mut br = [0.0f32; 4];
            if tl_v {
                tl[0] = (*p0.offset(off_tl as isize)).to_f32();
                tl[1] = (*p1.offset(off_tl as isize)).to_f32();
                tl[2] = (*p2.offset(off_tl as isize)).to_f32();
                tl[3] = (*p3.offset(off_tl as isize)).to_f32();
            }
            if tr_v {
                tr[0] = (*p0.offset(off_tr as isize)).to_f32();
                tr[1] = (*p1.offset(off_tr as isize)).to_f32();
                tr[2] = (*p2.offset(off_tr as isize)).to_f32();
                tr[3] = (*p3.offset(off_tr as isize)).to_f32();
            }
            if bl_v {
                bl[0] = (*p0.offset(off_bl as isize)).to_f32();
                bl[1] = (*p1.offset(off_bl as isize)).to_f32();
                bl[2] = (*p2.offset(off_bl as isize)).to_f32();
                bl[3] = (*p3.offset(off_bl as isize)).to_f32();
            }
            if br_v {
                br[0] = (*p0.offset(off_br as isize)).to_f32();
                br[1] = (*p1.offset(off_br as isize)).to_f32();
                br[2] = (*p2.offset(off_br as isize)).to_f32();
                br[3] = (*p3.offset(off_br as isize)).to_f32();
            }

            let mut res = [0.0f32; 4];
            for lane in 0..4 {
                res[lane] = w_tl * tl[lane];
                res[lane] = res[lane] + w_tr * tr[lane];
                res[lane] = res[lane] + w_bl * bl[lane];
                res[lane] = res[lane] + w_br * br[lane];
            }
            *output_n.offset(((c + 0) * spatial_out + out_off) as isize) = Half::from_f32(res[0]);
            *output_n.offset(((c + 1) * spatial_out + out_off) as isize) = Half::from_f32(res[1]);
            *output_n.offset(((c + 2) * spatial_out + out_off) as isize) = Half::from_f32(res[2]);
            *output_n.offset(((c + 3) * spatial_out + out_off) as isize) = Half::from_f32(res[3]);

            c += 4;
        }

        // Scalar tail.
        while c < c_channels {
            let p = input_n.offset((c * spatial_in) as isize);
            let mut v = 0.0f32;
            if tl_v {
                v += w_tl * (*p.offset(off_tl as isize)).to_f32();
            }
            if tr_v {
                v += w_tr * (*p.offset(off_tr as isize)).to_f32();
            }
            if bl_v {
                v += w_bl * (*p.offset(off_bl as isize)).to_f32();
            }
            if br_v {
                v += w_br * (*p.offset(off_br as isize)).to_f32();
            }
            *output_n.offset((c * spatial_out + out_off) as isize) = Half::from_f32(v);
            c += 1;
        }
    }
}

// Exposed entry point. Called by op_grid_sampler_2d's dispatcher only when
// cpuinfo_has_arm_neon_fp16() reports true. Input/output data are raw buffers
// interpreted as fp16; N/C/H/W/grid come pre-computed from the dispatcher.
// [spec:et:def:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn]
// [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn]
pub unsafe fn grid_sampler_2d_bilinear_fp16_hw(
    input: *const core::ffi::c_void,
    grid: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    n_batches: i32,
    c_channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    align_corners: bool,
) {
    unsafe {
        let in_ = input as *const Half;
        let gd = grid as *const Half;
        let out = output as *mut Half;

        let spatial_in = h_in * w_in;
        let spatial_out = h_out * w_out;

        for n in 0..n_batches {
            let input_n = in_.offset((n * c_channels * spatial_in) as isize);
            let output_n = out.offset((n * c_channels * spatial_out) as isize);
            let grid_n = gd.offset((n * h_out * w_out * 2) as isize);

            for h in 0..h_out {
                // __builtin_prefetch of the next grid row is a no-op in the port.
                for w in 0..w_out {
                    let grid_off = (h * w_out + w) * 2;
                    let mut gx = (*grid_n.offset(grid_off as isize)).to_f32();
                    let mut gy = (*grid_n.offset((grid_off + 1) as isize)).to_f32();
                    if align_corners {
                        gx = (gx + 1.0f32) * (w_in - 1) as f32 * 0.5f32;
                        gy = (gy + 1.0f32) * (h_in - 1) as f32 * 0.5f32;
                    } else {
                        gx = (gx + 1.0f32) * w_in as f32 * 0.5f32 - 0.5f32;
                        gy = (gy + 1.0f32) * h_in as f32 * 0.5f32 - 0.5f32;
                    }
                    bilinear_all_channels_fp16_hw_sample(
                        input_n, output_n, c_channels, h_in, w_in, h_out, w_out, h, w, gx, gy,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // input[c][y][x] = c*10 + y*2 + x over a 2x2 spatial grid (exactly
    // representable in fp16), for `c_channels` channels.
    fn make_input(c_channels: usize) -> Vec<Half> {
        (0..c_channels * 4)
            .map(|i| {
                let c = i / 4;
                let p = i % 4;
                Half::from_f32((c * 10 + p) as f32)
            })
            .collect()
    }

    // One output location, all channels: grid center of a 2x2 input has
    // gx = gy = 0.5, so every channel is the mean of its four pixels
    // (c*10 + 1.5). 5 channels exercise the 4-lane block plus the scalar tail.
    // [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn/test]
    #[test]
    fn bilinear_all_channels_fp16_hw_sample_center() {
        let input = make_input(5);
        let mut out = vec![Half::from_f32(-1.0); 5];
        unsafe {
            bilinear_all_channels_fp16_hw_sample(
                input.as_ptr(),
                out.as_mut_ptr(),
                5,
                2,
                2,
                1,
                1,
                0,
                0,
                0.5,
                0.5,
            );
        }
        for c in 0..5 {
            let expected = (c * 10) as f32 + 1.5;
            assert_eq!(out[c].to_f32(), expected, "channel {c}");
        }
    }

    // Zeros padding: a fully out-of-bounds sample writes 0 for every channel.
    // [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn/test]
    #[test]
    fn bilinear_all_channels_fp16_hw_sample_out_of_bounds() {
        let input = make_input(5);
        let mut out = vec![Half::from_f32(-1.0); 5];
        unsafe {
            bilinear_all_channels_fp16_hw_sample(
                input.as_ptr(),
                out.as_mut_ptr(),
                5,
                2,
                2,
                1,
                1,
                0,
                0,
                -3.0,
                -3.0,
            );
        }
        for c in 0..5 {
            assert_eq!(out[c].to_f32(), 0.0, "channel {c}");
        }
    }

    // Driver test: the identity grid maps the four output locations exactly
    // onto the four input pixels (align_corners=false unnormalize:
    // gx = (g + 1) * 2 * 0.5 - 0.5 lands on integer coordinates).
    // [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn/test]
    #[test]
    fn grid_sampler_2d_bilinear_fp16_hw_identity_grid() {
        let input = make_input(5);
        let grid: Vec<Half> = [-0.5f32, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
            .iter()
            .map(|&v| Half::from_f32(v))
            .collect();
        let mut out = vec![Half::from_f32(-1.0); 5 * 4];

        unsafe {
            grid_sampler_2d_bilinear_fp16_hw(
                input.as_ptr() as *const core::ffi::c_void,
                grid.as_ptr() as *const core::ffi::c_void,
                out.as_mut_ptr() as *mut core::ffi::c_void,
                1,
                5,
                2,
                2,
                2,
                2,
                false,
            );
        }
        for (i, (got, want)) in out.iter().zip(input.iter()).enumerate() {
            assert_eq!(got.to_f32(), want.to_f32(), "element {i}");
        }
    }

    // Driver test: center sample + align_corners=true corner samples.
    // [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.grid-sampler-2d-bilinear-fp16-hw-fn/test]
    // [spec:et:sem:op-grid-sampler-2d-fp16-hw.torch.executor.native.opt-grid-sampler-2d-internal.bilinear-all-channels-fp16-hw-sample-fn/test]
    #[test]
    fn grid_sampler_2d_bilinear_fp16_hw_center_and_align_corners() {
        let input = make_input(5);

        // align_corners = false, single center point -> per-channel mean.
        let grid_center: Vec<Half> = [0.0f32, 0.0].iter().map(|&v| Half::from_f32(v)).collect();
        let mut out = vec![Half::from_f32(-1.0); 5];
        unsafe {
            grid_sampler_2d_bilinear_fp16_hw(
                input.as_ptr() as *const core::ffi::c_void,
                grid_center.as_ptr() as *const core::ffi::c_void,
                out.as_mut_ptr() as *mut core::ffi::c_void,
                1,
                5,
                2,
                2,
                1,
                1,
                false,
            );
        }
        for c in 0..5 {
            assert_eq!(out[c].to_f32(), (c * 10) as f32 + 1.5, "channel {c}");
        }

        // align_corners = true, grid corners land exactly on the pixels.
        let grid_corners: Vec<Half> = [-1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
            .iter()
            .map(|&v| Half::from_f32(v))
            .collect();
        let mut out4 = vec![Half::from_f32(-1.0); 5 * 4];
        unsafe {
            grid_sampler_2d_bilinear_fp16_hw(
                input.as_ptr() as *const core::ffi::c_void,
                grid_corners.as_ptr() as *const core::ffi::c_void,
                out4.as_mut_ptr() as *mut core::ffi::c_void,
                1,
                5,
                2,
                2,
                2,
                2,
                true,
            );
        }
        for (i, (got, want)) in out4.iter().zip(input.iter()).enumerate() {
            assert_eq!(got.to_f32(), want.to_f32(), "element {i}");
        }
    }
}
