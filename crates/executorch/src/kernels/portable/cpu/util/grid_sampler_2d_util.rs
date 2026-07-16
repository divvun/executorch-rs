//! Literal port of kernels/portable/cpu/util/grid_sampler_2d_util.cpp + kernels/portable/cpu/util/grid_sampler_2d_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_default_dim_order, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;

// Ported from aten/src/ATen/native/GridSampler.h
// note that these need to be in the SAME ORDER as the enum in GridSampler.h
// as they are mapped to integer values (0, 1, 2) in this order
// [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-interpolation]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GridSamplerInterpolation {
    Bilinear,
    Nearest,
    Bicubic,
}
// [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-padding]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GridSamplerPadding {
    Zeros,
    Border,
    Reflection,
}

// PORT-NOTE: the header-inline helpers are function templates over `scalar_t`
// (the grid/coord float type). Ported over a `GridFloat` trait providing the
// float ops the C++ uses (`fabs`, `fmod`, `floor`, `min`, `max`, and casts).
pub trait GridFloat: Copy {
    fn from_f64(v: f64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn to_f64(self) -> f64;
    fn fabs(self) -> Self;
    fn fmod(self, other: Self) -> Self;
    fn floor(self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn lt(self, other: Self) -> bool;
}

macro_rules! impl_grid_float {
    ($t:ty) => {
        impl GridFloat for $t {
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn from_i64(v: i64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn fabs(self) -> Self {
                <$t>::abs(self)
            }
            fn fmod(self, other: Self) -> Self {
                self % other
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
            fn div(self, other: Self) -> Self {
                self / other
            }
            // std::min(a, b): returns first arg when comparison is unordered
            // (returns `a` unless `b < a`), matching C++ NaN semantics.
            fn min(self, other: Self) -> Self {
                if other < self { other } else { self }
            }
            // std::max(a, b): returns `a` unless `a < b`.
            fn max(self, other: Self) -> Self {
                if self < other { other } else { self }
            }
            fn lt(self, other: Self) -> bool {
                self < other
            }
        }
    };
}
impl_grid_float!(f32);
impl_grid_float!(f64);

// Ported from aten/src/ATen/native/GridSampler.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn]
pub fn grid_sampler_unnormalize<S: GridFloat>(coord: S, size: i64, align_corners: bool) -> S {
    if align_corners {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        coord
            .add(S::from_i64(1))
            .div(S::from_i64(2))
            .mul(S::from_i64(size - 1))
    } else {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        coord
            .add(S::from_i64(1))
            .mul(S::from_i64(size))
            .sub(S::from_i64(1))
            .div(S::from_i64(2))
    }
}

// Ported from aten/src/ATen/native/GridSampler.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.clip-coordinates-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.clip-coordinates-fn]
pub fn clip_coordinates<S: GridFloat>(in_: S, clip_limit: i64) -> S {
    // std::min(clip_limit - 1, std::max(in, 0)), with clip_limit-1 in int64.
    S::from_i64(clip_limit - 1).min(in_.max(S::from_i64(0)))
}

// Ported from aten/src/ATen/native/GridSampler.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn]
pub fn reflect_coordinates<S: GridFloat>(mut in_: S, twice_low: i64, twice_high: i64) -> S {
    if twice_low == twice_high {
        return S::from_i64(0);
    }
    let min: S = S::from_i64(twice_low).div(S::from_i64(2));
    let span: S = S::from_i64(twice_high - twice_low).div(S::from_i64(2));
    in_ = in_.sub(min).fabs();
    // `fmod` returns same sign as `in`, which is positive after the `fabs`.
    let extra: S = in_.fmod(span);
    let flips: i32 = in_.div(span).floor().to_f64() as i32;
    if flips % 2 == 0 {
        extra.add(min)
    } else {
        span.sub(extra).add(min)
    }
}

// Ported from aten/src/ATen/native/GridSampler.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]
pub fn grid_sampler_compute_source_index<S: GridFloat>(
    mut coord: S,
    size: i64,
    padding_mode: GridSamplerPadding,
    align_corners: bool,
) -> S {
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    if padding_mode == GridSamplerPadding::Border {
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    } else if padding_mode == GridSamplerPadding::Reflection {
        // reflect coordinates by image borders
        if align_corners {
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        } else {
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        coord = clip_coordinates(coord, size);
    }
    coord
}

// Ported from aten/src/ATen/native/GridSampler.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.within-bounds-2d-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.within-bounds-2d-fn]
pub fn within_bounds_2d<S: GridFloat>(h: S, w: S, big_h: i64, big_w: i64) -> bool {
    // h >= 0 && h < H && w >= 0 && w < W, comparisons mixing scalar_t with
    // int64_t under the usual arithmetic conversions (compare in f64).
    h.to_f64() >= 0.0
        && h.to_f64() < (big_h as f64)
        && w.to_f64() >= 0.0
        && w.to_f64() < (big_w as f64)
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn]
pub fn cubic_convolution1<S: GridFloat>(x: S, big_a: S) -> S {
    // ((A + 2) * x - (A + 3)) * x * x + 1
    big_a
        .add(S::from_i64(2))
        .mul(x)
        .sub(big_a.add(S::from_i64(3)))
        .mul(x)
        .mul(x)
        .add(S::from_i64(1))
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn]
pub fn cubic_convolution2<S: GridFloat>(x: S, big_a: S) -> S {
    // ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
    big_a
        .mul(x)
        .sub(S::from_i64(5).mul(big_a))
        .mul(x)
        .add(S::from_i64(8).mul(big_a))
        .mul(x)
        .sub(S::from_i64(4).mul(big_a))
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]
pub fn get_cubic_upsample_coefficients<S: GridFloat>(coeffs: &mut [S; 4], t: S) {
    // Standard bicubic interpolation uses alpha = -0.75
    let big_a: S = S::from_f64(-0.75);

    let x1: S = t;
    coeffs[0] = cubic_convolution2::<S>(x1.add(S::from_f64(1.0)), big_a);
    coeffs[1] = cubic_convolution1::<S>(x1, big_a);

    let x2: S = S::from_f64(1.0).sub(t);
    coeffs[2] = cubic_convolution1::<S>(x2, big_a);
    coeffs[3] = cubic_convolution2::<S>(x2.add(S::from_f64(1.0)), big_a);
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-interp1d-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-interp1d-fn]
pub fn cubic_interp1d<S: GridFloat>(x0: S, x1: S, x2: S, x3: S, t: S) -> S {
    let mut coeffs: [S; 4] = [S::from_i64(0); 4];
    get_cubic_upsample_coefficients::<S>(&mut coeffs, t);

    x0.mul(coeffs[0])
        .add(x1.mul(coeffs[1]))
        .add(x2.mul(coeffs[2]))
        .add(x3.mul(coeffs[3]))
}

// PORT-NOTE: `ET_CHECK_OR_RETURN_ERROR` is the crate macro. `exec_aten::SizesType`
// is the ported `SizesType` (i32).

// Argument checking and output tensor resizing for grid_sampler_2d
// [spec:et:def:grid-sampler-2d-util.torch.executor.check-grid-sampler-2d-args-and-resize-out-fn]
// [spec:et:sem:grid-sampler-2d-util.torch.executor.check-grid-sampler-2d-args-and-resize-out-fn]
pub fn check_grid_sampler_2d_args_and_resize_out(
    input: &Tensor,
    grid: &Tensor,
    out: &Tensor,
) -> Error {
    // Input must be 4D (N, C, H, W)
    crate::et_check_or_return_error!(
        input.dim() == 4,
        InvalidArgument,
        "Input must be 4D, got {} dimensions",
        input.dim() as usize
    );

    crate::et_check_or_return_error!(
        tensor_is_default_dim_order(input),
        InvalidArgument,
        "Input must be in NCHW format"
    );

    // Grid must be 4D (N, H_out, W_out, 2)
    crate::et_check_or_return_error!(
        grid.dim() == 4,
        InvalidArgument,
        "Grid must be 4D, got {} dimensions",
        grid.dim() as usize
    );

    crate::et_check_or_return_error!(
        grid.size(3) == 2,
        InvalidArgument,
        "Grid last dimension must be 2, got {}",
        grid.size(3) as i64
    );

    // Batch sizes must match
    crate::et_check_or_return_error!(
        input.size(0) == grid.size(0),
        InvalidArgument,
        "Input and grid batch sizes must match, got input={}, grid={}",
        input.size(0) as i64,
        grid.size(0) as i64
    );

    // Input and grid must have same dtype
    crate::et_check_or_return_error!(
        tensors_have_same_dtype2(input, grid),
        InvalidArgument,
        "Input and grid must have same dtype"
    );

    // Input and output must have the same dtype
    crate::et_check_or_return_error!(
        tensors_have_same_dtype2(input, out),
        InvalidArgument,
        "Input and output must have the same dtype"
    );

    // Resize output tensor to [N, C, H_out, W_out]
    let out_sizes: [SizesType; 4] = [
        input.size(0) as SizesType,
        input.size(1) as SizesType,
        grid.size(1) as SizesType,
        grid.size(2) as SizesType,
    ];

    let err: Error = resize_tensor_same_type(out, ArrayRef::from_raw_parts(out_sizes.as_ptr(), 4));
    crate::et_check_or_return_error!(
        err == Error::Ok,
        InvalidArgument,
        "Failed to resize output tensor"
    );

    Error::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    // PORT-NOTE: the generic `within_bounds_2d<scalar_t>` header helper is not
    // reached by op_grid_sampler_2d (which uses a local integer specialization);
    // this focused test pins the pure `h >= 0 && h < H && w >= 0 && w < W`
    // predicate against the C++ semantics.
    // [spec:et:sem:grid-sampler-2d-util.torch.executor.within-bounds-2d-fn/test]
    #[test]
    fn within_bounds_2d_predicate() {
        // interior point
        assert!(within_bounds_2d::<f32>(1.0, 2.0, 3, 4));
        // corners of the valid range
        assert!(within_bounds_2d::<f32>(0.0, 0.0, 3, 4));
        assert!(within_bounds_2d::<f32>(2.0, 3.0, 3, 4));
        // h at the upper bound (== H) is out
        assert!(!within_bounds_2d::<f32>(3.0, 0.0, 3, 4));
        // w at the upper bound (== W) is out
        assert!(!within_bounds_2d::<f32>(0.0, 4.0, 3, 4));
        // negative h / w are out
        assert!(!within_bounds_2d::<f32>(-1.0, 0.0, 3, 4));
        assert!(!within_bounds_2d::<f32>(0.0, -1.0, 3, 4));
        // fractional coordinate below H is in (compared in f64)
        assert!(within_bounds_2d::<f64>(2.5, 3.5, 3, 4));
        // fractional coordinate at/above the bound is out
        assert!(!within_bounds_2d::<f64>(3.5, 0.0, 3, 4));
    }
}
