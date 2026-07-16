//! Literal port of kernels/optimized/cpu/op_grid_sampler_2d.cpp.
//!
//! Optimized grid_sampler_2d.out for CPU. On aarch64 this is a NEON-vectorized
//! implementation for the common (bilinear + zeros padding) case. fp16 inputs are
//! promoted to fp32 for weight computation and accumulation and cast back on
//! store. Non-aarch64 targets, and any unsupported interpolation/padding/layout
//! combination, delegate to the portable kernel.
//!
//! DEVIATION (rust/PORTING.md optimized-kernels): the C++ NEON `float32x4_t`
//! intrinsics collapse to scalar 4-lane loops (all arithmetic is fp32, so the
//! results are identical). cpuinfo runtime dispatch becomes
//! `std::arch::is_aarch64_feature_detected!`. The aarch64 fast paths live behind
//! `#[cfg(target_arch = "aarch64")]`, mirroring the C++ `#ifdef __aarch64__`.

use crate::kernels::portable::cpu::op_grid_sampler_2d::grid_sampler_2d_out;
use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_is_contiguous, tensor_is_default_dim_order,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

#[cfg(target_arch = "aarch64")]
use crate::runtime::core::portable_type::Half;

// -------------------- aarch64 NEON fast paths --------------------

// SCALAR abstraction for the generic NEON driver: `to_f32` reads a source
// element into fp32 (the C++ `static_cast<float>` on `float` / `c10::Half`).
#[cfg(target_arch = "aarch64")]
trait NeonScalar: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

#[cfg(target_arch = "aarch64")]
impl NeonScalar for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

#[cfg(target_arch = "aarch64")]
impl NeonScalar for Half {
    #[inline]
    fn to_f32(self) -> f32 {
        Half::to_f32(self)
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        Half::from_f32(v)
    }
}

// -------------------- fp32 (plain ARMv8 NEON) --------------------
//
// The C++ fp16 software-convert path (bilinear_all_channels_f16_sw) and this
// fp32 path (bilinear_all_channels_f32) are byte-for-byte identical apart from
// the element type and the fp16<->fp32 conversion; the port unifies them into a
// single generic over `NeonScalar` whose `to_f32`/`from_f32` are the identity for
// `f32` and c10::Half's portable conversion for `Half`.
//
// [spec:et:def:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f32-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f32-fn]
// [spec:et:def:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f16-sw-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f16-sw-fn]
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bilinear_all_channels<SCALAR: NeonScalar>(
    input_n: *const SCALAR,
    output_n: *mut SCALAR,
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
            *output_n.offset(((c + 0) * spatial_out + out_off) as isize) = SCALAR::from_f32(res[0]);
            *output_n.offset(((c + 1) * spatial_out + out_off) as isize) = SCALAR::from_f32(res[1]);
            *output_n.offset(((c + 2) * spatial_out + out_off) as isize) = SCALAR::from_f32(res[2]);
            *output_n.offset(((c + 3) * spatial_out + out_off) as isize) = SCALAR::from_f32(res[3]);

            c += 4;
        }

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
            *output_n.offset((c * spatial_out + out_off) as isize) = SCALAR::from_f32(v);
            c += 1;
        }
    }
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn]
#[cfg(target_arch = "aarch64")]
unsafe fn grid_sampler_2d_neon<SCALAR>(
    input: *const SCALAR,
    grid: *const SCALAR,
    output: *mut SCALAR,
    n_batches: i32,
    c_channels: i32,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    align_corners: bool,
    sample_fn: unsafe fn(*const SCALAR, *mut SCALAR, i32, i32, i32, i32, i32, i32, i32, f32, f32),
) where
    SCALAR: NeonScalar,
{
    unsafe {
        let spatial_in = h_in * w_in;
        let spatial_out = h_out * w_out;

        for n in 0..n_batches {
            let input_n = input.offset((n * c_channels * spatial_in) as isize);
            let output_n = output.offset((n * c_channels * spatial_out) as isize);
            let grid_n = grid.offset((n * h_out * w_out * 2) as isize);

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
                    sample_fn(
                        input_n, output_n, c_channels, h_in, w_in, h_out, w_out, h, w, gx, gy,
                    );
                }
            }
        }
    }
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn]
pub fn opt_grid_sampler_2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    grid: &Tensor,
    interpolation_mode: i64,
    padding_mode: i64,
    align_corners: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // The NEON paths index input/grid/out directly assuming a contiguous NCHW
    // default-dim-order layout — no use of .strides() or .dim_order(). Fall back
    // to portable for anything else.
    let fast_eligible = input.dim() == 4
        && grid.dim() == 4
        && grid.size(3) == 2
        && input.size(0) == grid.size(0)
        && tensor_is_default_dim_order(input)
        && tensor_is_default_dim_order(grid)
        && tensor_is_default_dim_order(out)
        && tensor_is_contiguous(input)
        && tensor_is_contiguous(grid)
        && tensor_is_contiguous(out);

    // The fast paths read input/grid and write out as a single dtype: reject any
    // mixed-dtype call up front so none of the unchecked pointer casts can be
    // reached with a mismatched buffer.
    let dtypes_match =
        input.scalar_type() == grid.scalar_type() && input.scalar_type() == out.scalar_type();

    if interpolation_mode != 0 || padding_mode != 0 || !fast_eligible || !dtypes_match {
        return grid_sampler_2d_out(
            ctx,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            out,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        grid_sampler_2d_out(
            ctx,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            out,
        )
    }

    #[cfg(target_arch = "aarch64")]
    {
        let n_batches = input.size(0) as i32;
        let c_channels = input.size(1) as i32;
        let h_in = input.size(2) as i32;
        let w_in = input.size(3) as i32;
        let h_out = grid.size(1) as i32;
        let w_out = grid.size(2) as i32;

        if input.scalar_type() == ScalarType::Float {
            unsafe {
                grid_sampler_2d_neon::<f32>(
                    input.const_data_ptr::<f32>(),
                    grid.const_data_ptr::<f32>(),
                    out.mutable_data_ptr::<f32>(),
                    n_batches,
                    c_channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    align_corners,
                    bilinear_all_channels::<f32>,
                );
            }
            return out;
        }
        if input.scalar_type() == ScalarType::Half {
            if std::arch::is_aarch64_feature_detected!("fp16") {
                // Hardware fp16 path — safe because the CPU supports the +fp16
                // extension. Defined in op_grid_sampler_2d_fp16_hw.rs.
                unsafe {
                    super::op_grid_sampler_2d_fp16_hw::grid_sampler_2d_bilinear_fp16_hw(
                        input.const_data_ptr_typed(),
                        grid.const_data_ptr_typed(),
                        out.mutable_data_ptr_typed(),
                        n_batches,
                        c_channels,
                        h_in,
                        w_in,
                        h_out,
                        w_out,
                        align_corners,
                    );
                }
                return out;
            }
            // Software fp16<->fp32 conversion path. Works on any ARMv8.
            unsafe {
                grid_sampler_2d_neon::<Half>(
                    input.const_data_ptr::<Half>(),
                    grid.const_data_ptr::<Half>(),
                    out.mutable_data_ptr::<Half>(),
                    n_batches,
                    c_channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    align_corners,
                    bilinear_all_channels::<Half>,
                );
            }
            return out;
        }
        // Any other dtype: let portable handle it.
        grid_sampler_2d_out(
            ctx,
            input,
            grid,
            interpolation_mode,
            padding_mode,
            align_corners,
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // OpGridSampler2dTest.BilinearSimple: the four grid points map exactly onto
    // the four input pixels (bilinear, zeros padding, align_corners=false).
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_bilinear_simple() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        assert_tensor_close!(out, input);
    }

    // grid (0,0) -> the center of the 2x2 input: exact bilinear mean 2.5.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_bilinear_center() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        assert_tensor_close!(out, tf.make_default(vec![1, 1, 1, 1], vec![2.5]));
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, true, &out);
        assert_tensor_eq!(out, input);
    }

    // Zeros padding: a fully out-of-bounds sample contributes 0.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_zeros_padding_out_of_bounds() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(vec![1, 1, 2, 2], vec![-3.0, -3.0, 3.0, 3.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 2]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        assert_tensor_eq!(out, tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]));
    }

    // Multi-channel (5 channels: 4-lane block + scalar tail) against the
    // portable kernel on the same inputs.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_multi_channel_matches_portable() {
        let tf = TensorFactory::<f32>::new();
        let input_data: Vec<f32> = (0..45).map(|i| (i % 16) as f32 * 0.25).collect();
        let grid_data = vec![0.0f32, 0.0, -1.0, -1.0, 1.0, 1.0, 0.5, -0.5];

        let input = tf.make_default(vec![1, 5, 3, 3], input_data);
        let grid = tf.make_default(vec![1, 2, 2, 2], grid_data);
        let out_opt = tf.zeros_default(vec![1, 5, 2, 2]);
        let out_portable = tf.zeros_default(vec![1, 5, 2, 2]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out_opt);
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out_portable);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close!(out_opt, out_portable);
    }

    // Nearest interpolation (mode 1) is not fast-eligible: delegates to the
    // portable kernel.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.opt-grid-sampler-2d-out-fn/test]
    #[test]
    fn opt_grid_sampler_2d_out_nearest_falls_back_to_portable() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-0.6, -0.6, 0.4, -0.4, -0.3, 0.3, 0.6, 0.6],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        opt_grid_sampler_2d_out(&mut ctx, &input, &grid, 1, 0, false, &out);
        assert_tensor_eq!(out, input);
    }

    #[cfg(target_arch = "aarch64")]
    fn reference_bilinear_zeros(
        input: &[f32],
        grid: &[f32],
        c_channels: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        align_corners: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; c_channels * h_out * w_out];
        for p in 0..h_out * w_out {
            let mut gx = grid[p * 2];
            let mut gy = grid[p * 2 + 1];
            if align_corners {
                gx = (gx + 1.0) * (w_in as f32 - 1.0) * 0.5;
                gy = (gy + 1.0) * (h_in as f32 - 1.0) * 0.5;
            } else {
                gx = (gx + 1.0) * w_in as f32 * 0.5 - 0.5;
                gy = (gy + 1.0) * h_in as f32 * 0.5 - 0.5;
            }
            let x0 = gx.floor() as i64;
            let y0 = gy.floor() as i64;
            let fx = gx - x0 as f32;
            let fy = gy - y0 as f32;
            for c in 0..c_channels {
                let mut acc = 0.0f32;
                for (dy, dx, w) in [
                    (0i64, 0i64, (1.0 - fx) * (1.0 - fy)),
                    (0, 1, fx * (1.0 - fy)),
                    (1, 0, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ] {
                    let x = x0 + dx;
                    let y = y0 + dy;
                    if x >= 0 && (x as usize) < w_in && y >= 0 && (y as usize) < h_in {
                        acc += w * input[c * h_in * w_in + y as usize * w_in + x as usize];
                    }
                }
                out[c * h_out * w_out + p] = acc;
            }
        }
        out
    }

    // Direct NEON-driver test (fp32): 5 channels exercise the 4-lane block of
    // bilinear_all_channels plus its scalar tail; compared against a scalar
    // reference of the bilinear + zeros-padding formula.
    #[cfg(target_arch = "aarch64")]
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn/test]
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f32-fn/test]
    #[test]
    fn grid_sampler_2d_neon_f32_direct() {
        let input: Vec<f32> = (0..45).map(|i| (i % 16) as f32 * 0.25).collect();
        let grid = vec![0.0f32, 0.0, -1.0, -1.0, 1.0, 1.0, 0.5, -0.5];
        let mut out = vec![0.0f32; 20];

        for align_corners in [false, true] {
            unsafe {
                grid_sampler_2d_neon::<f32>(
                    input.as_ptr(),
                    grid.as_ptr(),
                    out.as_mut_ptr(),
                    1,
                    5,
                    3,
                    3,
                    2,
                    2,
                    align_corners,
                    bilinear_all_channels::<f32>,
                );
            }
            let expected = reference_bilinear_zeros(&input, &grid, 5, 3, 3, 2, 2, align_corners);
            for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-5,
                    "lane {i}: got {got}, want {want} (align_corners={align_corners})"
                );
            }
        }
    }

    // Direct NEON-driver test for the software fp16 path: Half in/out with all
    // arithmetic in fp32; results equal the fp32 reference rounded to Half.
    #[cfg(target_arch = "aarch64")]
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-neon-fn/test]
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.bilinear-all-channels-f16-sw-fn/test]
    #[test]
    fn grid_sampler_2d_neon_f16_sw_direct() {
        let input_f32: Vec<f32> = (0..45).map(|i| (i % 16) as f32 * 0.25).collect();
        let grid_f32 = vec![0.0f32, 0.0, -1.0, -1.0, 1.0, 1.0, 0.5, -0.5];
        let input: Vec<Half> = input_f32.iter().map(|&v| Half::from_f32(v)).collect();
        let grid: Vec<Half> = grid_f32.iter().map(|&v| Half::from_f32(v)).collect();
        let mut out = vec![Half::from_f32(0.0); 20];

        unsafe {
            grid_sampler_2d_neon::<Half>(
                input.as_ptr(),
                grid.as_ptr(),
                out.as_mut_ptr(),
                1,
                5,
                3,
                3,
                2,
                2,
                false,
                bilinear_all_channels::<Half>,
            );
        }
        let expected = reference_bilinear_zeros(&input_f32, &grid_f32, 5, 3, 3, 2, 2, false);
        for (i, (got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.to_f32() - want).abs() <= 1e-2,
                "lane {i}: got {}, want {want}",
                got.to_f32()
            );
        }
    }
}
