//! Literal port of kernels/portable/cpu/util/upsample_util.cpp + kernels/portable/cpu/util/upsample_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type,
    tensor_is_default_or_channels_last_dim_order, tensors_have_same_dim_order2,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;

// PORT-NOTE: `executorch::aten::OptionalArrayRef<T>` has no ported target yet.
// Per PORTING.md `optional<T>` maps to `Option<T>`, and an `OptionalArrayRef<T>`
// is an optional non-owning array view, so it is modeled as
// `Option<ArrayRef<T>>` here: `.has_value()` -> `.is_some()`,
// `.value()` -> `.unwrap()`, indexing `v[i]` -> `*v.at(i)`. Unresolved
// cross-module reference: replace with the real `OptionalArrayRef` once ported.
pub type OptionalArrayRef<T> = Option<ArrayRef<T>>;

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` logs and returns false; ported
// as a local macro over `et_log!` mirroring tensor_util.rs, dropping the empty
// message. `ET_CHECK_OR_RETURN_ERROR` / `ET_LOG` are the crate macros.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        if !($cond) {
            return false;
        }
    };
}

// [spec:et:def:upsample-util.torch.executor.check-upsample-2d-common-args-fn]
// [spec:et:sem:upsample-util.torch.executor.check-upsample-2d-common-args-fn]
pub fn check_upsample_2d_common_args(
    in_: &Tensor,
    output_size: &OptionalArrayRef<i64>,
    scale_factors: &OptionalArrayRef<f64>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensors_have_same_dim_order2(in_, out));
    et_log_and_return_if_false!(in_.dim() == 4);
    et_log_and_return_if_false!(out.dim() == 4);
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));
    et_log_and_return_if_false!(output_size.is_some() ^ scale_factors.is_some());
    if scale_factors.is_some() {
        let sf = scale_factors.unwrap();
        et_log_and_return_if_false!(sf.size() == 2);
        et_log_and_return_if_false!(*sf.at(0) > 0.0);
        et_log_and_return_if_false!(*sf.at(1) > 0.0);
    } else if output_size.is_some() {
        let os = output_size.unwrap();
        et_log_and_return_if_false!(os.size() == 2);
        et_log_and_return_if_false!(*os.at(0) > 0);
        et_log_and_return_if_false!(*os.at(1) > 0);
    }

    true
}

// [spec:et:def:upsample-util.torch.executor.check-upsample-bilinear2d-args-fn]
// [spec:et:sem:upsample-util.torch.executor.check-upsample-bilinear2d-args-fn]
pub fn check_upsample_bilinear2d_args(
    in_: &Tensor,
    output_size: &OptionalArrayRef<i64>,
    _align_corners: bool,
    scale_factors: &OptionalArrayRef<f64>,
    out: &Tensor,
) -> bool {
    check_upsample_2d_common_args(in_, output_size, scale_factors, out)
}

// [spec:et:def:upsample-util.torch.executor.check-upsample-nearest2d-args-fn]
// [spec:et:sem:upsample-util.torch.executor.check-upsample-nearest2d-args-fn]
pub fn check_upsample_nearest2d_args(
    in_: &Tensor,
    output_size: &OptionalArrayRef<i64>,
    scale_factors: &OptionalArrayRef<f64>,
    out: &Tensor,
) -> bool {
    check_upsample_2d_common_args(in_, output_size, scale_factors, out)
}

// [spec:et:def:upsample-util.torch.executor.resize-upsample-2d-fn]
// [spec:et:sem:upsample-util.torch.executor.resize-upsample-2d-fn]
pub fn resize_upsample_2d(
    in_: &Tensor,
    output_size: &OptionalArrayRef<i64>,
    scale_factors: &OptionalArrayRef<f64>,
    scale_h_out: &mut f64,
    scale_w_out: &mut f64,
    out: &Tensor,
) -> Error {
    // Either output_size or scale_factors are provided, not both. This
    // is checked in check_..._args.
    let mut target_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    let dim = in_.dim();
    // std::copy(in.sizes().cbegin(), in.sizes().cend(), target_size.begin());
    let sizes = in_.sizes();
    for i in 0..sizes.size() {
        target_size[i] = *sizes.at(i);
    }

    if scale_factors.is_some() {
        let sf = scale_factors.unwrap();
        *scale_h_out = *sf.at(0);
        *scale_w_out = *sf.at(1);

        target_size[(dim - 2) as usize] =
            (*sizes.at((dim - 2) as usize) as f64 * *scale_h_out) as SizesType;
        target_size[(dim - 1) as usize] =
            (*sizes.at((dim - 1) as usize) as f64 * *scale_w_out) as SizesType;
    } else if output_size.is_some() {
        let os = output_size.unwrap();
        *scale_h_out = (*os.at(0) as f64) / (*sizes.at((dim - 2) as usize) as f64);
        *scale_w_out = (*os.at(1) as f64) / (*sizes.at((dim - 1) as usize) as f64);

        target_size[(dim - 2) as usize] = *os.at(0) as SizesType;
        target_size[(dim - 1) as usize] = *os.at(1) as SizesType;
    } else {
        crate::et_log!(Error, "Invalid output_size or scale_factors");
        return Error::InvalidArgument;
    }

    crate::et_check_or_return_error!(
        target_size[(dim - 2) as usize] > 0 && target_size[(dim - 1) as usize] > 0,
        InvalidArgument,
        "Upsampled output size must be non-empty, but was {} x {}.",
        target_size[(dim - 2) as usize] as i64,
        target_size[(dim - 1) as usize] as i64
    );

    resize_tensor_same_type(
        out,
        ArrayRef::from_raw_parts(target_size.as_ptr(), dim as usize),
    )
}

// PORT-NOTE: the header-inline template helpers below are generic over
// `scalar_t` / `opmath_t` (float compute types). Ported over an `UpsampleFloat`
// trait providing the float ops the C++ uses (`floor`, `min`, `max`, cast
// to/from int64). The `std::optional<double>` maps to `Option<f64>`.

pub trait UpsampleFloat: Copy {
    fn from_f64(v: f64) -> Self;
    fn from_i64(v: i64) -> Self;
    fn to_i64_floor(self) -> i64;
    fn floor(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn lt(self, other: Self) -> bool;
}

macro_rules! impl_upsample_float {
    ($t:ty) => {
        impl UpsampleFloat for $t {
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn from_i64(v: i64) -> Self {
                v as $t
            }
            fn to_i64_floor(self) -> i64 {
                self as i64
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn min(self, other: Self) -> Self {
                if other < self { other } else { self }
            }
            fn max(self, other: Self) -> Self {
                if self < other { other } else { self }
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
            fn lt(self, other: Self) -> bool {
                self < other
            }
        }
    };
}
impl_upsample_float!(f32);
impl_upsample_float!(f64);

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.compute-scales-value-fn]
// [spec:et:sem:upsample-util.torch.executor.compute-scales-value-fn]
pub fn compute_scales_value<S: UpsampleFloat>(
    scale: &Option<f64>,
    input_size: i64,
    output_size: i64,
) -> S {
    match scale {
        Some(v) => S::from_f64(1.0 / *v),
        None => S::from_i64(input_size).div(S::from_i64(output_size)),
    }
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.area-pixel-compute-scale-fn]
// [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]
pub fn area_pixel_compute_scale<S: UpsampleFloat>(
    input_size: i64,
    output_size: i64,
    align_corners: bool,
    scale: &Option<f64>,
) -> S {
    // see Note [area_pixel_compute_scale]
    if align_corners {
        if output_size > 1 {
            S::from_i64(input_size - 1).div(S::from_i64(output_size - 1))
        } else {
            S::from_f64(0.0)
        }
    } else {
        compute_scales_value::<S>(scale, input_size, output_size)
    }
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.area-pixel-compute-source-index-fn]
// [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-source-index-fn]
pub fn area_pixel_compute_source_index<S: UpsampleFloat>(
    scale: S,
    dst_index: i64,
    align_corners: bool,
    cubic: bool,
) -> S {
    if align_corners {
        scale.mul(S::from_i64(dst_index))
    } else {
        let src_idx = scale
            .mul(S::from_i64(dst_index).add(S::from_f64(0.5)))
            .sub(S::from_f64(0.5));
        if !cubic && src_idx.lt(S::from_f64(0.0)) {
            S::from_f64(0.0)
        } else {
            src_idx
        }
    }
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.guard-index-and-lambda-fn]
// [spec:et:sem:upsample-util.torch.executor.guard-index-and-lambda-fn]
pub fn guard_index_and_lambda<S: UpsampleFloat, O: UpsampleFloat + ToF64Lossy>(
    real_input_index: O,
    input_size: i64,
    input_index: &mut i64,
    lambda: &mut S,
) {
    *input_index = core::cmp::min(real_input_index.floor().to_i64_floor(), input_size - 1);
    // lambda = min(max(real_input_index - input_index, 0), 1)
    let l = real_input_index
        .sub(O::from_i64(*input_index))
        .max(O::from_f64(0.0))
        .min(O::from_f64(1.0));
    // Assign into scalar_t lambda (opmath_t -> scalar_t narrowing).
    *lambda = S::from_f64(l.to_f64_lossy());
}

// PORT-NOTE: helper to reproduce the `opmath_t -> scalar_t` reference
// assignment in `guard_index_and_lambda`/`compute_source_index_and_lambda`
// without exposing a raw `as` at every site.
pub trait ToF64Lossy {
    fn to_f64_lossy(self) -> f64;
}
impl ToF64Lossy for f32 {
    fn to_f64_lossy(self) -> f64 {
        self as f64
    }
}
impl ToF64Lossy for f64 {
    fn to_f64_lossy(self) -> f64 {
        self
    }
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.compute-source-index-and-lambda-fn]
// [spec:et:sem:upsample-util.torch.executor.compute-source-index-and-lambda-fn]
#[allow(clippy::too_many_arguments)]
pub fn compute_source_index_and_lambda<S: UpsampleFloat, O: UpsampleFloat + ToF64Lossy>(
    input_index0: &mut i64,
    input_index1: &mut i64,
    lambda0: &mut S,
    lambda1: &mut S,
    ratio: O,
    output_index: i64,
    input_size: i64,
    output_size: i64,
    align_corners: bool,
) {
    if output_size == input_size {
        // scale_factor = 1, simply copy
        *input_index0 = output_index;
        *input_index1 = output_index;
        *lambda0 = S::from_f64(1.0);
        *lambda1 = S::from_f64(0.0);
    } else {
        let real_input_index = area_pixel_compute_source_index::<O>(
            ratio,
            output_index,
            align_corners,
            /*cubic=*/ false,
        );
        guard_index_and_lambda::<S, O>(real_input_index, input_size, input_index0, lambda1);
        let offset: i64 = if *input_index0 < input_size - 1 { 1 } else { 0 };
        *input_index1 = *input_index0 + offset;
        *lambda0 = S::from_f64(1.0).sub(*lambda1);
    }
}

// Ported from aten/src/ATen/native/UpSample.h
// [spec:et:def:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]
// [spec:et:sem:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]
pub fn nearest_neighbor_compute_source_index(scale: f32, dst_index: i64, input_size: i64) -> i64 {
    // Index computation matching OpenCV INTER_NEAREST (buggy, kept for BC).
    core::cmp::min(((dst_index as f32) * scale).floor() as i64, input_size - 1)
}
