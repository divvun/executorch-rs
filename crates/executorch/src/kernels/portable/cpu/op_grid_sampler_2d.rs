//! Literal port of kernels/portable/cpu/op_grid_sampler_2d.cpp.

use crate::kernels::portable::cpu::util::grid_sampler_2d_util::{
    GridFloat, GridSamplerInterpolation, GridSamplerPadding,
    check_grid_sampler_2d_args_and_resize_out, clip_coordinates, cubic_interp1d,
    grid_sampler_compute_source_index, grid_sampler_unnormalize, reflect_coordinates,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE (util gap for the fixer): the ported `grid_sampler_2d_util` only
// implements `GridFloat` (and thus `grid_sampler_compute_source_index`,
// `grid_sampler_unnormalize`, `clip_coordinates`, `reflect_coordinates`,
// `cubic_interp1d`) for `f32`/`f64`, and `within_bounds_2d` takes a `GridFloat`
// argument. The C++ kernels instantiate these helpers for:
//   - bilinear: `AccType<CTYPE>` = `float` for Half/BFloat16, else CTYPE — always
//     f32/f64 for the FLOATHBF16 arms, but CTYPE (an integer) for the integer
//     REALHBF16 arms;
//   - nearest / bicubic: CTYPE directly, i.e. also Half/BFloat16 and the integer
//     types;
// and `within_bounds_2d` is instantiated with `int64_t` corner indices.
//
// To keep the util untouched (per the wave-2 no-cross-module-redesign rule) this
// file models the per-CTYPE compute type via a local `GridScalar` trait: `Acc`
// (a `GridFloat`) is `f32` for the reduced floats and the small integer types,
// `f64` for `Int`/`Long`/`Double`, `f32` for `Float`. This reproduces the C++
// `AccType` exactly for the FLOATHBF16 arms of bilinear. For nearest/bicubic and
// for the integer arms of bilinear the C++ computes in CTYPE (integer arithmetic
// for the integer types); routing them through `Acc` (f32/f64) is a DELIBERATE
// DEVIATION where CTYPE is an integer type — a fully bug-for-bug integer port
// needs the util to provide `GridFloat` for the integer/reduced-float ctypes.
// `within_bounds_2d(int64, int64, ...)` is reproduced by the local integer
// `within_bounds_2d_i64` below (pure integer comparison, matching the C++
// `within_bounds_2d<int64_t>` instantiation).
trait GridScalar: Copy {
    type Acc: GridFloat;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(v: Self::Acc) -> Self;
}

macro_rules! impl_grid_scalar {
    ($t:ty, $acc:ty, $to:expr, $from:expr) => {
        impl GridScalar for $t {
            type Acc = $acc;
            fn to_acc(self) -> Self::Acc {
                $to(self)
            }
            fn from_acc(v: Self::Acc) -> Self {
                $from(v)
            }
        }
    };
}
impl_grid_scalar!(f32, f32, |v: f32| v, |v: f32| v);
impl_grid_scalar!(f64, f64, |v: f64| v, |v: f64| v);
impl_grid_scalar!(Half, f32, |v: Half| v.to_f32(), |v: f32| Half::from_f32(v));
impl_grid_scalar!(BFloat16, f32, |v: BFloat16| v.to_f32(), |v: f32| {
    BFloat16::from_f32(v)
});
impl_grid_scalar!(u8, f32, |v: u8| v as f32, |v: f32| v as u8);
impl_grid_scalar!(i8, f32, |v: i8| v as f32, |v: f32| v as i8);
impl_grid_scalar!(i16, f32, |v: i16| v as f32, |v: f32| v as i16);
impl_grid_scalar!(i32, f64, |v: i32| v as f64, |v: f64| v as i32);
impl_grid_scalar!(i64, f64, |v: i64| v as f64, |v: f64| v as i64);

// within_bounds_2d<int64_t>: pure integer comparison (see util-gap note).
fn within_bounds_2d_i64(h: i64, w: i64, big_h: i64, big_w: i64) -> bool {
    h >= 0 && h < big_h && w >= 0 && w < big_w
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn]
fn grid_sample_2d_bilinear_kernel_impl_nchw<CTYPE: GridScalar>(
    in_: &Tensor,
    grid: &Tensor,
    padding_mode: GridSamplerPadding,
    align_corners: bool,
    out: &Tensor,
) {
    type Acc<CTYPE> = <CTYPE as GridScalar>::Acc;
    let in_data = in_.const_data_ptr::<CTYPE>();
    let out_data = out.mutable_data_ptr::<CTYPE>();

    // Grid has shape [N, H_out, W_out, 2]
    let grid_data = grid.const_data_ptr::<CTYPE>();

    let n_batches: i64 = in_.size(0) as i64;
    let c_channels: i64 = in_.size(1) as i64;
    let inp_h: i64 = in_.size(2) as i64;
    let inp_w: i64 = in_.size(3) as i64;

    let out_h: i64 = out.size(2) as i64;
    let out_w: i64 = out.size(3) as i64;

    for n in 0..n_batches {
        let grid_offset = n * *grid.strides().at(0) as i64;
        let in_batch_offset = n * *in_.strides().at(0) as i64;
        let out_batch_offset = n * *out.strides().at(0) as i64;

        for c in 0..c_channels {
            let in_channel_offset = in_batch_offset + c * *in_.strides().at(1) as i64;
            let out_channel_offset = out_batch_offset + c * *out.strides().at(1) as i64;

            for h in 0..out_h {
                for w in 0..out_w {
                    let grid_idx: i64 = grid_offset
                        + h * *grid.strides().at(1) as i64
                        + w * *grid.strides().at(2) as i64;
                    let x: Acc<CTYPE> = unsafe { *grid_data.offset(grid_idx as isize) }.to_acc();
                    let y: Acc<CTYPE> = unsafe {
                        *grid_data.offset((grid_idx + *grid.strides().at(3) as i64) as isize)
                    }
                    .to_acc();

                    let ix: Acc<CTYPE> = grid_sampler_compute_source_index::<Acc<CTYPE>>(
                        x,
                        inp_w,
                        padding_mode,
                        align_corners,
                    );
                    let iy: Acc<CTYPE> = grid_sampler_compute_source_index::<Acc<CTYPE>>(
                        y,
                        inp_h,
                        padding_mode,
                        align_corners,
                    );

                    let ix_nw: i64 = ix.floor().to_f64() as i64;
                    let iy_nw: i64 = iy.floor().to_f64() as i64;
                    let ix_ne: i64 = ix_nw + 1;
                    let iy_ne: i64 = iy_nw;
                    let ix_sw: i64 = ix_nw;
                    let iy_sw: i64 = iy_nw + 1;
                    let ix_se: i64 = ix_nw + 1;
                    let iy_se: i64 = iy_nw + 1;

                    // Interpolation weights (computed in ACC precision).
                    // (ix_se - ix): int64 - Acc under usual arithmetic
                    // conversions -> Acc.
                    let nw_weight: Acc<CTYPE> = Acc::<CTYPE>::from_i64(ix_se)
                        .sub(ix)
                        .mul(Acc::<CTYPE>::from_i64(iy_se).sub(iy));
                    let ne_weight: Acc<CTYPE> = ix
                        .sub(Acc::<CTYPE>::from_i64(ix_sw))
                        .mul(Acc::<CTYPE>::from_i64(iy_sw).sub(iy));
                    let sw_weight: Acc<CTYPE> = Acc::<CTYPE>::from_i64(ix_ne)
                        .sub(ix)
                        .mul(iy.sub(Acc::<CTYPE>::from_i64(iy_ne)));
                    let se_weight: Acc<CTYPE> = ix
                        .sub(Acc::<CTYPE>::from_i64(ix_nw))
                        .mul(iy.sub(Acc::<CTYPE>::from_i64(iy_nw)));

                    let mut out_val: Acc<CTYPE> = Acc::<CTYPE>::from_i64(0);

                    if padding_mode == GridSamplerPadding::Zeros {
                        if within_bounds_2d_i64(iy_nw, ix_nw, inp_h, inp_w) {
                            let px = unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_nw * *in_.strides().at(2) as i64
                                        + ix_nw * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            };
                            out_val = out_val.add(px.to_acc().mul(nw_weight));
                        }
                        if within_bounds_2d_i64(iy_ne, ix_ne, inp_h, inp_w) {
                            let px = unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_ne * *in_.strides().at(2) as i64
                                        + ix_ne * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            };
                            out_val = out_val.add(px.to_acc().mul(ne_weight));
                        }
                        if within_bounds_2d_i64(iy_sw, ix_sw, inp_h, inp_w) {
                            let px = unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_sw * *in_.strides().at(2) as i64
                                        + ix_sw * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            };
                            out_val = out_val.add(px.to_acc().mul(sw_weight));
                        }
                        if within_bounds_2d_i64(iy_se, ix_se, inp_h, inp_w) {
                            let px = unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_se * *in_.strides().at(2) as i64
                                        + ix_se * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            };
                            out_val = out_val.add(px.to_acc().mul(se_weight));
                        }
                    } else {
                        // For border/reflection padding, clip corner indices.
                        // clip_coordinates<Acc>(corner, limit) -> Acc, then to i64.
                        let ix_nw_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(ix_nw), inp_w)
                                .to_f64() as i64;
                        let iy_nw_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(iy_nw), inp_h)
                                .to_f64() as i64;
                        let ix_ne_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(ix_ne), inp_w)
                                .to_f64() as i64;
                        let iy_ne_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(iy_ne), inp_h)
                                .to_f64() as i64;
                        let ix_sw_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(ix_sw), inp_w)
                                .to_f64() as i64;
                        let iy_sw_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(iy_sw), inp_h)
                                .to_f64() as i64;
                        let ix_se_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(ix_se), inp_w)
                                .to_f64() as i64;
                        let iy_se_safe: i64 =
                            clip_coordinates::<Acc<CTYPE>>(Acc::<CTYPE>::from_i64(iy_se), inp_h)
                                .to_f64() as i64;
                        let p_nw = unsafe {
                            *in_data.offset(
                                (in_channel_offset
                                    + iy_nw_safe * *in_.strides().at(2) as i64
                                    + ix_nw_safe * *in_.strides().at(3) as i64)
                                    as isize,
                            )
                        };
                        let p_ne = unsafe {
                            *in_data.offset(
                                (in_channel_offset
                                    + iy_ne_safe * *in_.strides().at(2) as i64
                                    + ix_ne_safe * *in_.strides().at(3) as i64)
                                    as isize,
                            )
                        };
                        let p_sw = unsafe {
                            *in_data.offset(
                                (in_channel_offset
                                    + iy_sw_safe * *in_.strides().at(2) as i64
                                    + ix_sw_safe * *in_.strides().at(3) as i64)
                                    as isize,
                            )
                        };
                        let p_se = unsafe {
                            *in_data.offset(
                                (in_channel_offset
                                    + iy_se_safe * *in_.strides().at(2) as i64
                                    + ix_se_safe * *in_.strides().at(3) as i64)
                                    as isize,
                            )
                        };
                        out_val = p_nw
                            .to_acc()
                            .mul(nw_weight)
                            .add(p_ne.to_acc().mul(ne_weight))
                            .add(p_sw.to_acc().mul(sw_weight))
                            .add(p_se.to_acc().mul(se_weight));
                    }

                    let out_idx: i64 = out_channel_offset
                        + h * *out.strides().at(2) as i64
                        + w * *out.strides().at(3) as i64;
                    unsafe {
                        *out_data.offset(out_idx as isize) = CTYPE::from_acc(out_val);
                    }
                }
            }
        }
    }
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn]
//
// PORT-NOTE: C++ computes in CTYPE. This port computes the source-index math in
// `CTYPE::Acc` (see the util-gap note above); for the FLOATHBF16 arms this is a
// faithful reproduction (Acc = f32/f64), for the integer arms it is the noted
// deviation. Storage load/store stay in CTYPE.
fn grid_sample_2d_nearest_kernel_impl_nchw<CTYPE: GridScalar>(
    in_: &Tensor,
    grid: &Tensor,
    padding_mode: GridSamplerPadding,
    align_corners: bool,
    out: &Tensor,
) {
    type Acc<CTYPE> = <CTYPE as GridScalar>::Acc;
    let in_data = in_.const_data_ptr::<CTYPE>();
    let out_data = out.mutable_data_ptr::<CTYPE>();

    let grid_data = grid.const_data_ptr::<CTYPE>();

    let n_batches: i64 = in_.size(0) as i64;
    let c_channels: i64 = in_.size(1) as i64;
    let inp_h: i64 = in_.size(2) as i64;
    let inp_w: i64 = in_.size(3) as i64;

    let out_h: i64 = out.size(2) as i64;
    let out_w: i64 = out.size(3) as i64;

    for n in 0..n_batches {
        let grid_offset = n * *grid.strides().at(0) as i64;
        let in_batch_offset = n * *in_.strides().at(0) as i64;
        let out_batch_offset = n * *out.strides().at(0) as i64;

        for c in 0..c_channels {
            let in_channel_offset = in_batch_offset + c * *in_.strides().at(1) as i64;
            let out_channel_offset = out_batch_offset + c * *out.strides().at(1) as i64;

            for h in 0..out_h {
                for w in 0..out_w {
                    let grid_idx: i64 = grid_offset
                        + h * *grid.strides().at(1) as i64
                        + w * *grid.strides().at(2) as i64;
                    let x: Acc<CTYPE> = unsafe { *grid_data.offset(grid_idx as isize) }.to_acc();
                    let y: Acc<CTYPE> = unsafe {
                        *grid_data.offset((grid_idx + *grid.strides().at(3) as i64) as isize)
                    }
                    .to_acc();

                    let ix: Acc<CTYPE> = grid_sampler_compute_source_index::<Acc<CTYPE>>(
                        x,
                        inp_w,
                        padding_mode,
                        align_corners,
                    );
                    let iy: Acc<CTYPE> = grid_sampler_compute_source_index::<Acc<CTYPE>>(
                        y,
                        inp_h,
                        padding_mode,
                        align_corners,
                    );

                    // std::nearbyint: round to nearest, ties to even (default
                    // rounding mode), matching ATen — NOT std::round.
                    let ix_nearest: i64 = ix.to_f64().round_ties_even() as i64;
                    let iy_nearest: i64 = iy.to_f64().round_ties_even() as i64;

                    let mut out_val: CTYPE = CTYPE::from_acc(Acc::<CTYPE>::from_i64(0));

                    if padding_mode == GridSamplerPadding::Zeros {
                        if within_bounds_2d_i64(iy_nearest, ix_nearest, inp_h, inp_w) {
                            out_val = unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_nearest * *in_.strides().at(2) as i64
                                        + ix_nearest * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            };
                        }
                    } else {
                        let ix_clipped: i64 = clip_coordinates::<Acc<CTYPE>>(
                            Acc::<CTYPE>::from_i64(ix_nearest),
                            inp_w,
                        )
                        .to_f64() as i64;
                        let iy_clipped: i64 = clip_coordinates::<Acc<CTYPE>>(
                            Acc::<CTYPE>::from_i64(iy_nearest),
                            inp_h,
                        )
                        .to_f64() as i64;
                        out_val = unsafe {
                            *in_data.offset(
                                (in_channel_offset
                                    + iy_clipped * *in_.strides().at(2) as i64
                                    + ix_clipped * *in_.strides().at(3) as i64)
                                    as isize,
                            )
                        };
                    }

                    let out_idx: i64 = out_channel_offset
                        + h * *out.strides().at(2) as i64
                        + w * *out.strides().at(3) as i64;
                    unsafe {
                        *out_data.offset(out_idx as isize) = out_val;
                    }
                }
            }
        }
    }
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn]
//
// PORT-NOTE: as with nearest, the coordinate/interpolation math runs in
// `CTYPE::Acc` (see util-gap note); faithful for FLOATHBF16 arms, noted deviation
// for integer arms. `cubic_interp1d` is the util helper over `Acc`.
fn grid_sample_2d_bicubic_kernel_impl_nchw<CTYPE: GridScalar>(
    in_: &Tensor,
    grid: &Tensor,
    padding_mode: GridSamplerPadding,
    align_corners: bool,
    out: &Tensor,
) {
    type Acc<CTYPE> = <CTYPE as GridScalar>::Acc;
    let in_data = in_.const_data_ptr::<CTYPE>();
    let out_data = out.mutable_data_ptr::<CTYPE>();

    let grid_data = grid.const_data_ptr::<CTYPE>();

    let n_batches: i64 = in_.size(0) as i64;
    let c_channels: i64 = in_.size(1) as i64;
    let inp_h: i64 = in_.size(2) as i64;
    let inp_w: i64 = in_.size(3) as i64;

    let out_h: i64 = out.size(2) as i64;
    let out_w: i64 = out.size(3) as i64;

    for n in 0..n_batches {
        let grid_offset = n * *grid.strides().at(0) as i64;
        let in_batch_offset = n * *in_.strides().at(0) as i64;
        let out_batch_offset = n * *out.strides().at(0) as i64;

        for c in 0..c_channels {
            let in_channel_offset = in_batch_offset + c * *in_.strides().at(1) as i64;
            let out_channel_offset = out_batch_offset + c * *out.strides().at(1) as i64;

            for h in 0..out_h {
                for w in 0..out_w {
                    let grid_idx: i64 = grid_offset
                        + h * *grid.strides().at(1) as i64
                        + w * *grid.strides().at(2) as i64;
                    let x: Acc<CTYPE> = unsafe { *grid_data.offset(grid_idx as isize) }.to_acc();
                    let y: Acc<CTYPE> = unsafe {
                        *grid_data.offset((grid_idx + *grid.strides().at(3) as i64) as isize)
                    }
                    .to_acc();

                    // Raw unnormalized coordinates without padding applied.
                    let ix: Acc<CTYPE> =
                        grid_sampler_unnormalize::<Acc<CTYPE>>(x, inp_w, align_corners);
                    let iy: Acc<CTYPE> =
                        grid_sampler_unnormalize::<Acc<CTYPE>>(y, inp_h, align_corners);

                    let ix_0: i64 = ix.floor().to_f64() as i64;
                    let iy_0: i64 = iy.floor().to_f64() as i64;
                    let tx: Acc<CTYPE> = ix.sub(Acc::<CTYPE>::from_i64(ix_0));
                    let ty: Acc<CTYPE> = iy.sub(Acc::<CTYPE>::from_i64(iy_0));

                    let ix_m1: i64 = ix_0 - 1;
                    let ix_p1: i64 = ix_0 + 1;
                    let ix_p2: i64 = ix_0 + 2;

                    let iy_m1: i64 = iy_0 - 1;
                    let iy_p1: i64 = iy_0 + 1;
                    let iy_p2: i64 = iy_0 + 2;

                    // Helper: safely get pixel value with bounds/padding handling.
                    let get_value_bounded = |iy: i64, ix: i64| -> Acc<CTYPE> {
                        if padding_mode == GridSamplerPadding::Zeros {
                            if within_bounds_2d_i64(iy, ix, inp_h, inp_w) {
                                return unsafe {
                                    *in_data.offset(
                                        (in_channel_offset
                                            + iy * *in_.strides().at(2) as i64
                                            + ix * *in_.strides().at(3) as i64)
                                            as isize,
                                    )
                                }
                                .to_acc();
                            }
                            CTYPE::from_acc(Acc::<CTYPE>::from_i64(0)).to_acc()
                        } else if padding_mode == GridSamplerPadding::Border {
                            let iy_safe: i64 = 0i64.max(iy.min(inp_h - 1));
                            let ix_safe: i64 = 0i64.max(ix.min(inp_w - 1));
                            unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_safe * *in_.strides().at(2) as i64
                                        + ix_safe * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            }
                            .to_acc()
                        } else {
                            let mut iy_reflected: Acc<CTYPE> = Acc::<CTYPE>::from_i64(iy);
                            let mut ix_reflected: Acc<CTYPE> = Acc::<CTYPE>::from_i64(ix);

                            if align_corners {
                                iy_reflected = reflect_coordinates::<Acc<CTYPE>>(
                                    iy_reflected,
                                    0,
                                    2 * (inp_h - 1),
                                );
                                ix_reflected = reflect_coordinates::<Acc<CTYPE>>(
                                    ix_reflected,
                                    0,
                                    2 * (inp_w - 1),
                                );
                            } else {
                                iy_reflected = reflect_coordinates::<Acc<CTYPE>>(
                                    iy_reflected,
                                    -1,
                                    2 * inp_h - 1,
                                );
                                ix_reflected = reflect_coordinates::<Acc<CTYPE>>(
                                    ix_reflected,
                                    -1,
                                    2 * inp_w - 1,
                                );
                            }

                            let iy_safe: i64 =
                                clip_coordinates::<Acc<CTYPE>>(iy_reflected, inp_h).to_f64() as i64;
                            let ix_safe: i64 =
                                clip_coordinates::<Acc<CTYPE>>(ix_reflected, inp_w).to_f64() as i64;

                            unsafe {
                                *in_data.offset(
                                    (in_channel_offset
                                        + iy_safe * *in_.strides().at(2) as i64
                                        + ix_safe * *in_.strides().at(3) as i64)
                                        as isize,
                                )
                            }
                            .to_acc()
                        }
                    };

                    let mut coefficients: [Acc<CTYPE>; 4] = [Acc::<CTYPE>::from_i64(0); 4];

                    // Row -1
                    let mut p0 = get_value_bounded(iy_m1, ix_m1);
                    let mut p1 = get_value_bounded(iy_m1, ix_0);
                    let mut p2 = get_value_bounded(iy_m1, ix_p1);
                    let mut p3 = get_value_bounded(iy_m1, ix_p2);
                    coefficients[0] = cubic_interp1d::<Acc<CTYPE>>(p0, p1, p2, p3, tx);

                    // Row 0
                    p0 = get_value_bounded(iy_0, ix_m1);
                    p1 = get_value_bounded(iy_0, ix_0);
                    p2 = get_value_bounded(iy_0, ix_p1);
                    p3 = get_value_bounded(iy_0, ix_p2);
                    coefficients[1] = cubic_interp1d::<Acc<CTYPE>>(p0, p1, p2, p3, tx);

                    // Row +1
                    p0 = get_value_bounded(iy_p1, ix_m1);
                    p1 = get_value_bounded(iy_p1, ix_0);
                    p2 = get_value_bounded(iy_p1, ix_p1);
                    p3 = get_value_bounded(iy_p1, ix_p2);
                    coefficients[2] = cubic_interp1d::<Acc<CTYPE>>(p0, p1, p2, p3, tx);

                    // Row +2
                    p0 = get_value_bounded(iy_p2, ix_m1);
                    p1 = get_value_bounded(iy_p2, ix_0);
                    p2 = get_value_bounded(iy_p2, ix_p1);
                    p3 = get_value_bounded(iy_p2, ix_p2);
                    coefficients[3] = cubic_interp1d::<Acc<CTYPE>>(p0, p1, p2, p3, tx);

                    let out_val: Acc<CTYPE> = cubic_interp1d::<Acc<CTYPE>>(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                        ty,
                    );

                    let out_idx: i64 = out_channel_offset
                        + h * *out.strides().at(2) as i64
                        + w * *out.strides().at(3) as i64;
                    unsafe {
                        *out_data.offset(out_idx as isize) = CTYPE::from_acc(out_val);
                    }
                }
            }
        }
    }
}

// [spec:et:def:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn]
// [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn]
pub fn grid_sampler_2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    grid: &Tensor,
    interpolation_mode: i64,
    padding_mode: i64,
    align_corners: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Check arguments and resize output tensor
    crate::et_kernel_check_msg!(
        ctx,
        check_grid_sampler_2d_args_and_resize_out(input, grid, out) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to validate arguments and resize output tensor"
    );

    // Convert integer mode parameters to enums (static_cast of the int code).
    let mode: GridSamplerInterpolation = match interpolation_mode {
        0 => GridSamplerInterpolation::Bilinear,
        1 => GridSamplerInterpolation::Nearest,
        2 => GridSamplerInterpolation::Bicubic,
        _ => GridSamplerInterpolation::Bilinear, // placeholder; rejected below
    };
    let padding: GridSamplerPadding = match padding_mode {
        0 => GridSamplerPadding::Zeros,
        1 => GridSamplerPadding::Border,
        2 => GridSamplerPadding::Reflection,
        _ => GridSamplerPadding::Zeros, // placeholder; rejected below
    };

    // Validate mode and padding values
    crate::et_kernel_check!(
        ctx,
        (interpolation_mode == 0 || interpolation_mode == 1 || interpolation_mode == 2),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        (padding_mode == 0 || padding_mode == 1 || padding_mode == 2),
        InvalidArgument,
        out
    );

    crate::et_switch_realhbf16_types!(input.scalar_type(), ctx, "grid_sampler_2d.out", CTYPE, {
        match mode {
            GridSamplerInterpolation::Bilinear => {
                grid_sample_2d_bilinear_kernel_impl_nchw::<CTYPE>(
                    input,
                    grid,
                    padding,
                    align_corners,
                    out,
                );
            }
            GridSamplerInterpolation::Nearest => {
                grid_sample_2d_nearest_kernel_impl_nchw::<CTYPE>(
                    input,
                    grid,
                    padding,
                    align_corners,
                    out,
                );
            }
            GridSamplerInterpolation::Bicubic => {
                grid_sample_2d_bicubic_kernel_impl_nchw::<CTYPE>(
                    input,
                    grid,
                    padding,
                    align_corners,
                    out,
                );
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::{assert_tensor_close, assert_tensor_close_with_tol, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    // enable_if_t<is_floating_point<CTYPE>>: floating dtypes run the body; the
    // integer/reduced-float `!is_floating_point` overload is a no-op.
    trait GridSamplerDType: Copy {
        const IS_FLOAT: bool;
    }
    macro_rules! impl_grid_dtype {
        ($t:ty, $is_float:expr) => {
            impl GridSamplerDType for $t {
                const IS_FLOAT: bool = $is_float;
            }
        };
    }
    impl_grid_dtype!(u8, false);
    impl_grid_dtype!(i8, false);
    impl_grid_dtype!(i16, false);
    impl_grid_dtype!(i32, false);
    impl_grid_dtype!(i64, false);
    impl_grid_dtype!(f32, true);
    impl_grid_dtype!(f64, true);
    impl_grid_dtype!(Half, false);
    impl_grid_dtype!(BFloat16, false);

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    // Non-float dtypes never reach `from_f64` (the body returns early), but the
    // trait bound requires an impl; these are unreachable stubs.
    macro_rules! impl_from_f64_stub {
        ($($t:ty),*) => {$(impl FromF64 for $t {
            fn from_f64(_v: f64) -> Self { unreachable!() }
        })*};
    }
    impl_from_f64_stub!(u8, i8, i16, i32, i64);
    impl FromF64 for Half {
        fn from_f64(_v: f64) -> Self {
            unreachable!()
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(_v: f64) -> Self {
            unreachable!()
        }
    }

    fn test_grid_sampler_2d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + GridSamplerDType + FromF64,
    {
        if !T::IS_FLOAT {
            // not supported (enable_if false overload returns immediately)
            return;
        }
        let tf = TensorFactory::<T>::new();

        let input = tf.make_default(
            vec![1, 1, 2, 2],
            vec![
                T::from_f64(1.0),
                T::from_f64(2.0),
                T::from_f64(3.0),
                T::from_f64(4.0),
            ],
        );
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![
                T::from_f64(-0.5),
                T::from_f64(-0.5),
                T::from_f64(0.5),
                T::from_f64(-0.5),
                T::from_f64(-0.5),
                T::from_f64(0.5),
                T::from_f64(0.5),
                T::from_f64(0.5),
            ],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        assert_tensor_close!(out, input);
    }

    fn f(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // also verifies check_grid_sampler_2d_args_and_resize_out (valid path: resizes
    // out to [N,C,H_out,W_out]); the _dies tests below exercise its error branches.
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.check-grid-sampler-2d-args-and-resize-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_bilinear_simple() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        let expected = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_tensor_close!(out, expected);
    }

    // grid (0,0) -> center of the 2x2 input; the 2.5 result pins the coordinate
    // unnormalize + source-index math these helpers compute.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn/test]
    // also verifies grid_sample_2d_bilinear_kernel_impl_nchw: mode 0 samples the 2x2
    // input center to the exact bilinear mean 2.5.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bilinear-kernel-impl-nchw-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_bilinear_interpolation() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        let expected = tf.make_default(vec![1, 1, 1, 1], vec![2.5]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_bilinear_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, true, &out);
        let expected = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_tensor_eq!(out, expected);
    }

    // also verifies grid_sample_2d_nearest_kernel_impl_nchw: mode 1 rounds each grid
    // coordinate to the nearest source pixel, reproducing the 2x2 input exactly.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-nearest-kernel-impl-nchw-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_nearest_simple() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-0.6, -0.6, 0.4, -0.4, -0.3, 0.3, 0.6, 0.6],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 1, 0, false, &out);
        let expected = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_tensor_eq!(out, expected);
    }

    // bicubic path drives get_cubic_upsample_coefficients -> cubic_convolution1/2
    // and cubic_interp1d; the 8.5 result pins their combined coefficient math.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-interp1d-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn/test]
    // also verifies grid_sample_2d_bicubic_kernel_impl_nchw: mode 2 combines the cubic
    // coefficients over the 4x4 input to the exact center value.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sample-2d-bicubic-kernel-impl-nchw-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_bicubic_simple() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 1, 4, 4],
            f(&[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ]),
        );
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 2, 0, false, &out);
        let expected = tf.make_default(vec![1, 1, 1, 1], vec![8.5]);
        assert_tensor_close_with_tol!(out, expected, 0.0, 0.5);
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_zeros_padding_out_of_bounds() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(
            vec![1, 2, 2, 2],
            vec![-2.0, -2.0, 2.0, 2.0, -0.5, -0.5, 0.5, 0.5],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        let expected = tf.make_default(vec![1, 1, 2, 2], vec![0.0, 0.0, 1.0, 4.0]);
        assert_tensor_close!(out, expected);
    }

    // border padding clamps out-of-bounds corners via clip_coordinates; the
    // [1.0, 4.0] result pins that clamp to [0, size-1].
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.clip-coordinates-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_border_padding_out_of_bounds() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let grid = tf.make_default(vec![1, 1, 2, 2], vec![-2.0, -2.0, 2.0, 2.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 2]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 1, false, &out);
        let expected = tf.make_default(vec![1, 1, 1, 2], vec![1.0, 4.0]);
        assert_tensor_close!(out, expected);
    }

    // reflection padding routes coordinates through reflect_coordinates.
    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_reflection_padding() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 3, 3], f(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]));
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 2, false, &out);
        let expected = tf.make_default(vec![1, 1, 1, 1], vec![5.0]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_multi_channel() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 2, 2, 2], f(&[1., 2., 3., 4., 5., 6., 7., 8.]));
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 2, 1, 1]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        let expected = tf.make_default(vec![1, 2, 1, 1], vec![2.5, 6.5]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_multi_batch() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![2, 1, 2, 2], f(&[1., 2., 3., 4., 5., 6., 7., 8.]));
        let grid = tf.make_default(vec![2, 1, 1, 2], vec![0.0, 0.0, 0.0, 0.0]);
        let out = tf.zeros_default(vec![2, 1, 1, 1]);

        let mut ctx = context();
        grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out);
        let expected = tf.make_default(vec![2, 1, 1, 1], vec![2.5, 6.5]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_dtype() {
        // ET_FORALL_REALHBF16_TYPES
        test_grid_sampler_2d_dtype::<u8>();
        test_grid_sampler_2d_dtype::<i8>();
        test_grid_sampler_2d_dtype::<i16>();
        test_grid_sampler_2d_dtype::<i32>();
        test_grid_sampler_2d_dtype::<i64>();
        test_grid_sampler_2d_dtype::<f32>();
        test_grid_sampler_2d_dtype::<f64>();
        test_grid_sampler_2d_dtype::<Half>();
        test_grid_sampler_2d_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_invalid_input_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 2, 2]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_invalid_grid_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.make_default(vec![1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_grid_last_dim_must_be2_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.ones_default(vec![1, 1, 1, 3]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_batch_size_mismatch_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.make_default(vec![2, 1, 1, 2], vec![0.0, 0.0, 0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_mismatched_dtype_dies() {
        let tf = TensorFactory::<f32>::new();
        let tf_long = TensorFactory::<i64>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf_long.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_grid_dtype_mismatch_dies() {
        let tf = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf_double.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_invalid_interpolation_mode_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 3, 0, false, &out)
        );
    }

    // [spec:et:sem:op-grid-sampler-2d.torch.executor.native.grid-sampler-2d-out-fn/test]
    #[test]
    fn op_grid_sampler_2d_out_test_invalid_padding_mode_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2, 2]);
        let grid = tf.make_default(vec![1, 1, 1, 2], vec![0.0, 0.0]);
        let out = tf.zeros_default(vec![1, 1, 1, 1]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            grid_sampler_2d_out(&mut ctx, &input, &grid, 0, 3, false, &out)
        );
    }
}
