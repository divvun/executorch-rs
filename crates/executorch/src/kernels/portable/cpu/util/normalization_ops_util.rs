//! Literal port of kernels/portable/cpu/util/normalization_ops_util.cpp + kernels/portable/cpu/util/normalization_ops_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, tensor_has_expected_size, tensor_is_rank, tensors_have_same_dtype2,
    tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: local check macros mirroring the C++ `ET_LOG_AND_RETURN_IF_FALSE`
// and `ET_CHECK_OR_RETURN_FALSE`; the crate-level `et_check_or_return_false!`
// drops caller format args, so (like tensor_util.rs) this module carries its own.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

// [spec:et:def:normalization-ops-util.torch.executor.layer-norm-scalar-fn]
// [spec:et:sem:normalization-ops-util.torch.executor.layer-norm-scalar-fn]
// PORT-NOTE: templated on the element type `CTYPE`. The C++ accumulates
// statistics in `float`, then evaluates `(x - mean) / std * w + b` under the
// usual arithmetic conversions (CTYPE and float mixed) and stores back to
// CTYPE. Modeled via the local `LayerNormCtype` trait exposing the `float`
// conversions and CTYPE store used, implemented for the real dtypes.
///
/// # Safety
/// The pointer arguments must be valid contiguous buffers of the documented
/// lengths (`input_data`/`out_data`: `M*N`; `weight_data`/`bias_data`: `N` or
/// null; `mean_data`/`rstd_data`: `M`).
pub unsafe fn layer_norm_scalar<CTYPE: LayerNormCtype>(
    input_data: *const CTYPE,
    weight_data: *const CTYPE, // nullable
    bias_data: *const CTYPE,   // nullable
    out_data: *mut CTYPE,
    mean_data: *mut CTYPE,
    rstd_data: *mut CTYPE,
    m: usize,
    n: usize,
    eps: f32,
) {
    for i in 0..m {
        let x: *const CTYPE = unsafe { input_data.add(i * n) };
        let y: *mut CTYPE = unsafe { out_data.add(i * n) };

        // compute E[X] and Var[x] = E[x^2] - E[x]^2
        let mut sum: f32 = 0.0f32;
        for j in 0..n {
            sum += CTYPE::to_f32(unsafe { *x.add(j) });
        }
        let mut sq_sum: f32 = 0.0;
        for j in 0..n {
            let xj: f32 = CTYPE::to_f32(unsafe { *x.add(j) });
            sq_sum += xj * xj;
        }
        let mean_value: f32 = sum / n as f32;
        let variance: f32 = sq_sum / n as f32 - mean_value * mean_value;
        let std: f32 = (variance + eps).sqrt();

        // Calculate the elements of output
        for j in 0..n {
            let w: CTYPE = if !weight_data.is_null() {
                unsafe { *weight_data.add(j) }
            } else {
                CTYPE::from_i32(1)
            };
            let b: CTYPE = if !bias_data.is_null() {
                unsafe { *bias_data.add(j) }
            } else {
                CTYPE::from_i32(0)
            };
            let xj: CTYPE = unsafe { *x.add(j) };
            unsafe {
                *y.add(j) = CTYPE::normalize(xj, mean_value, std, w, b);
            }
        }

        unsafe {
            *mean_data.add(i) = CTYPE::from_f32(mean_value);
            *rstd_data.add(i) = CTYPE::from_f64(1.0f64 / std as f64);
        }
    }
}

// PORT-NOTE: models the `float`/CTYPE mixed arithmetic and the CTYPE
// static_casts used by `layer_norm_scalar`. `normalize` reproduces
// `(x - mean_value) / std * w + b` where `mean_value`/`std` are `float`,
// `x`/`w`/`b` are CTYPE; for integer CTYPE the C++ promotes to `int` for the
// mixed float ops then narrows on store — modeled per-type below.
pub trait LayerNormCtype: Copy {
    fn to_f32(v: Self) -> f32;
    fn from_i32(v: i32) -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_f64(v: f64) -> Self;
    fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self;
}

macro_rules! impl_layer_norm_ctype_int {
    ($($t:ty),*) => {$(
        impl LayerNormCtype for $t {
            fn to_f32(v: Self) -> f32 { v as f32 }
            fn from_i32(v: i32) -> Self { v as $t }
            fn from_f32(v: f32) -> Self { v as $t }
            fn from_f64(v: f64) -> Self { v as $t }
            fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self {
                // usual arithmetic conversions promote the integer operands to
                // float for the (x - mean)/std*w term, then the `+ b` and store
                // narrow back to CTYPE.
                (((x as f32 - mean_value) / std * (w as f32)) as $t) + b
            }
        }
    )*};
}
impl_layer_norm_ctype_int!(u8, i8, i16, i32, i64);

impl LayerNormCtype for f32 {
    fn to_f32(v: Self) -> f32 {
        v
    }
    fn from_i32(v: i32) -> Self {
        v as f32
    }
    fn from_f32(v: f32) -> Self {
        v
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self {
        (x - mean_value) / std * w + b
    }
}

impl LayerNormCtype for f64 {
    fn to_f32(v: Self) -> f32 {
        v as f32
    }
    fn from_i32(v: i32) -> Self {
        v as f64
    }
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self {
        // float mean_value/std widen to double under usual arithmetic
        // conversions when combined with the double operands.
        (x - mean_value as f64) / std as f64 * w + b
    }
}

impl LayerNormCtype for crate::runtime::core::portable_type::Half {
    fn to_f32(v: Self) -> f32 {
        v.to_f32()
    }
    fn from_i32(v: i32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v as f32)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v)
    }
    fn from_f64(v: f64) -> Self {
        crate::runtime::core::portable_type::Half::from_f64(v)
    }
    fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self {
        // Half promotes to float for arithmetic then narrows to Half on store.
        crate::runtime::core::portable_type::Half::from_f32(
            (x.to_f32() - mean_value) / std * w.to_f32() + b.to_f32(),
        )
    }
}

impl LayerNormCtype for crate::runtime::core::portable_type::BFloat16 {
    fn to_f32(v: Self) -> f32 {
        v.to_f32()
    }
    fn from_i32(v: i32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v as f32)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v)
    }
    fn from_f64(v: f64) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f64(v)
    }
    fn normalize(x: Self, mean_value: f32, std: f32, w: Self, b: Self) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(
            (x.to_f32() - mean_value) / std * w.to_f32() + b.to_f32(),
        )
    }
}

// [spec:et:def:normalization-ops-util.torch.executor.check-batch-norm-args-fn]
// [spec:et:sem:normalization-ops-util.torch.executor.check-batch-norm-args-fn]
pub fn check_batch_norm_args(
    in_: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    _momentum: f64,
    _eps: f64,
    out: &Tensor,
    mean_out: &Tensor,
    var_out: &Tensor,
) -> bool {
    // All tensors must be the same dtype
    if let Some(weight) = weight {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, weight));
    }
    if let Some(bias) = bias {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, bias));
    }
    if let Some(running_mean) = running_mean {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, running_mean));
    }
    if let Some(running_var) = running_var {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, running_var));
    }
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, mean_out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, var_out));

    let c_dim: usize = if in_.dim() >= 1 { 1 } else { 0 };
    // All parameter tensors must be of dim 1 and have length equal to the
    // channels dim of in
    if let Some(weight) = weight {
        et_log_and_return_if_false!(tensor_is_rank(weight, 1));
        et_log_and_return_if_false!(tensors_have_same_size_at_dims(weight, 0, in_, c_dim));
    }
    if let Some(bias) = bias {
        et_log_and_return_if_false!(tensor_is_rank(bias, 1));
        et_log_and_return_if_false!(tensors_have_same_size_at_dims(bias, 0, in_, c_dim));
    }
    if let Some(running_mean) = running_mean {
        et_log_and_return_if_false!(tensor_is_rank(running_mean, 1));
        et_log_and_return_if_false!(tensors_have_same_size_at_dims(running_mean, 0, in_, c_dim));
    }
    if let Some(running_var) = running_var {
        et_log_and_return_if_false!(tensor_is_rank(running_var, 1));
        et_log_and_return_if_false!(tensors_have_same_size_at_dims(running_var, 0, in_, c_dim));
    }

    true
}

// [spec:et:def:normalization-ops-util.torch.executor.check-layer-norm-args-fn]
// [spec:et:sem:normalization-ops-util.torch.executor.check-layer-norm-args-fn]
pub fn check_layer_norm_args(
    in_: &Tensor,
    normalized_shape: ArrayRef<i64>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    out: &Tensor,
    mean_out: &Tensor,
    rstd_out: &Tensor,
) -> bool {
    let ndim: usize = normalized_shape.size();
    et_check_or_return_false!(
        ndim >= 1,
        "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element; ndim = {}",
        ndim
    );
    et_check_or_return_false!(
        in_.dim() >= ndim as ssize_t,
        "Expected input tensor to have rank >= the length of normalized_shape; in.dim() = {}, ndim = {}",
        in_.dim(),
        ndim
    );
    et_check_or_return_false!(
        ndim <= K_TENSOR_DIMENSION_LIMIT,
        "Expected normalized shape to have at most {} dimensions but it had {}",
        K_TENSOR_DIMENSION_LIMIT,
        ndim
    );
    let shift: usize = in_.dim() as usize - ndim;
    for d in 0..ndim {
        et_check_or_return_false!(
            in_.size((d + shift) as ssize_t) as i64 == *normalized_shape.at(d),
            "Expected normalized_shape to match the sizes of input's rightmost dimensions; in.size({}) = {}, normalized_shape[{}] = {}",
            d + shift,
            in_.size((d + shift) as ssize_t),
            d,
            *normalized_shape.at(d)
        );
    }
    let mut shape: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..ndim {
        shape[i] = *normalized_shape.at(i) as SizesType;
    }

    if let Some(weight) = weight {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, weight));
        et_log_and_return_if_false!(tensor_has_expected_size(
            weight,
            ArrayRef::from_raw_parts(shape.as_ptr(), ndim)
        ));
    }
    if let Some(bias) = bias {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, bias));
        et_log_and_return_if_false!(tensor_has_expected_size(
            bias,
            ArrayRef::from_raw_parts(shape.as_ptr(), ndim)
        ));
    }
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, mean_out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, rstd_out));
    true
}

// [spec:et:def:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn]
// [spec:et:sem:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn]
///
/// # Safety
/// `mean_rstd_sizes` must point to at least `in_.dim()` valid `SizesType`
/// elements and `mean_rstd_ndim` to a valid `usize`.
pub unsafe fn get_layer_norm_out_target_size(
    in_: &Tensor,
    normalized_shape: ArrayRef<i64>,
    mean_rstd_sizes: *mut SizesType,
    mean_rstd_ndim: *mut usize,
) {
    unsafe {
        *mean_rstd_ndim = in_.dim() as usize;

        for d in 0..in_.dim() {
            if (d as i64) < (in_.dim() - normalized_shape.size() as ssize_t) as i64 {
                *mean_rstd_sizes.add(d as usize) = in_.size(d) as SizesType;
            } else {
                *mean_rstd_sizes.add(d as usize) = 1;
            }
        }
    }
}

// [spec:et:def:normalization-ops-util.torch.executor.check-group-norm-args-fn]
// [spec:et:sem:normalization-ops-util.torch.executor.check-group-norm-args-fn]
pub fn check_group_norm_args(
    in_: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    n: i64,
    c: i64,
    hxw: i64,
    group: i64,
    out: &Tensor,
    mean_out: &Tensor,
    rstd_out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(in_.size(0) as i64 == n);
    et_log_and_return_if_false!(in_.size(1) as i64 == c);
    et_log_and_return_if_false!(in_.numel() as i64 == n * c * hxw);
    et_check_or_return_false!(
        group > 0,
        "Expected number of groups to be greater than 0; group = {}",
        group
    );
    et_check_or_return_false!(
        c % group == 0,
        "Expected number of channels in input to be divisible by number of groups; C = {}, group = {}, C % group = {}",
        c,
        group,
        c % group
    );
    et_check_or_return_false!(
        weight.is_none() || (weight.unwrap().dim() == 1 && weight.unwrap().size(0) as i64 == c),
        "Expected weight to be a vector of size equal to the number of channels in input; weight.has_value() = {}, weight.dim() = {}, weight.size(0) = {}, C = {}",
        weight.is_some(),
        if let Some(weight) = weight {
            weight.dim()
        } else {
            -1
        },
        if let Some(weight) = weight {
            weight.size(0)
        } else {
            -1
        },
        c
    );
    et_check_or_return_false!(
        bias.is_none() || (bias.unwrap().dim() == 1 && bias.unwrap().size(0) as i64 == c),
        "Expected bias to be a vector of size equal to the number of channels in input; bias.has_value() = {}, bias.dim() = {}, bias.size(0) = {}, C = {}",
        bias.is_some(),
        if let Some(bias) = bias {
            bias.dim()
        } else {
            -1
        },
        if let Some(bias) = bias {
            bias.size(0)
        } else {
            -1
        },
        c
    );

    if let Some(weight) = weight {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, weight));
    }
    if let Some(bias) = bias {
        et_log_and_return_if_false!(tensors_have_same_dtype2(in_, bias));
    }
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, mean_out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, rstd_out));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    // PORT-NOTE: `check_batch_norm_args` and `check_group_norm_args` have no op test
    // exercising their rejection branches (the ported op tests only drive the accept
    // path or die at the dtype switch), and `get_layer_norm_out_target_size`'s mean/
    // rstd sizes are never asserted by the op tests. These focused tests pin their
    // C++ semantics directly (normalization_ops_util.cpp).

    // [spec:et:sem:normalization-ops-util.torch.executor.check-batch-norm-args-fn/test]
    #[test]
    fn normalization_ops_util_test_check_batch_norm_args() {
        crate::runtime::platform::platform::pal_init();
        let tf = TensorFactory::<f32>::new();
        // in: [N, C, H, W]; C = 3 -> parameter tensors are rank-1, length C.
        let in_ = tf.zeros_default(vec![2, 3, 4, 4]);
        let weight = tf.zeros_default(vec![3]);
        let bias = tf.zeros_default(vec![3]);
        let running_mean = tf.zeros_default(vec![3]);
        let running_var = tf.zeros_default(vec![3]);
        let out = tf.zeros_default(vec![2, 3, 4, 4]);
        let mean_out = tf.zeros_default(vec![3]);
        let var_out = tf.zeros_default(vec![3]);

        assert!(check_batch_norm_args(
            &in_,
            Some(&weight),
            Some(&bias),
            Some(&running_mean),
            Some(&running_var),
            1e-3,
            1e-5,
            &out,
            &mean_out,
            &var_out,
        ));

        // All optional tensors may be absent.
        assert!(check_batch_norm_args(
            &in_, None, None, None, None, 1e-3, 1e-5, &out, &mean_out, &var_out,
        ));

        // A parameter tensor whose length != channels (C=3) is rejected.
        let weight_wrong = tf.zeros_default(vec![5]);
        assert!(!check_batch_norm_args(
            &in_,
            Some(&weight_wrong),
            None,
            None,
            None,
            1e-3,
            1e-5,
            &out,
            &mean_out,
            &var_out,
        ));

        // A parameter tensor of rank != 1 is rejected.
        let weight_2d = tf.zeros_default(vec![3, 1]);
        assert!(!check_batch_norm_args(
            &in_,
            Some(&weight_2d),
            None,
            None,
            None,
            1e-3,
            1e-5,
            &out,
            &mean_out,
            &var_out,
        ));
    }

    // [spec:et:sem:normalization-ops-util.torch.executor.check-group-norm-args-fn/test]
    #[test]
    fn normalization_ops_util_test_check_group_norm_args() {
        crate::runtime::platform::platform::pal_init();
        let tf = TensorFactory::<f32>::new();
        // in: [N=2, C=4, H*W=6]; group=2 divides C.
        let n = 2i64;
        let c = 4i64;
        let hxw = 6i64;
        let group = 2i64;
        let in_ = tf.zeros_default(vec![2, 4, 6]);
        let weight = tf.zeros_default(vec![4]);
        let bias = tf.zeros_default(vec![4]);
        let out = tf.zeros_default(vec![2, 4, 6]);
        let mean_out = tf.zeros_default(vec![2, 2]);
        let rstd_out = tf.zeros_default(vec![2, 2]);

        assert!(check_group_norm_args(
            &in_,
            Some(&weight),
            Some(&bias),
            n,
            c,
            hxw,
            group,
            &out,
            &mean_out,
            &rstd_out,
        ));

        // group must be > 0.
        assert!(!check_group_norm_args(
            &in_,
            Some(&weight),
            Some(&bias),
            n,
            c,
            hxw,
            0,
            &out,
            &mean_out,
            &rstd_out,
        ));

        // C must be divisible by group.
        assert!(!check_group_norm_args(
            &in_,
            Some(&weight),
            Some(&bias),
            n,
            c,
            hxw,
            3,
            &out,
            &mean_out,
            &rstd_out,
        ));

        // weight must be a vector of length C.
        let weight_wrong = tf.zeros_default(vec![5]);
        assert!(!check_group_norm_args(
            &in_,
            Some(&weight_wrong),
            Some(&bias),
            n,
            c,
            hxw,
            group,
            &out,
            &mean_out,
            &rstd_out,
        ));

        // in.size(0) must equal N.
        assert!(!check_group_norm_args(
            &in_,
            Some(&weight),
            Some(&bias),
            3,
            c,
            hxw,
            group,
            &out,
            &mean_out,
            &rstd_out,
        ));
    }

    // [spec:et:sem:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn/test]
    #[test]
    fn normalization_ops_util_test_get_layer_norm_out_target_size() {
        let tf = TensorFactory::<f32>::new();
        // in: [4, 5, 6], normalized_shape = [6] -> mean/rstd sizes = [4, 5, 1].
        let in_ = tf.zeros_default(vec![4, 5, 6]);
        let ns_vec: [i64; 1] = [6];
        let normalized_shape = ArrayRef::from_raw_parts(ns_vec.as_ptr(), ns_vec.len());

        let mut sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let mut ndim: usize = 0;
        unsafe {
            get_layer_norm_out_target_size(&in_, normalized_shape, sizes.as_mut_ptr(), &mut ndim);
        }
        assert_eq!(ndim, 3);
        assert_eq!(&sizes[0..3], &[4, 5, 1]);

        // normalized_shape spanning the trailing two dims -> [4, 1, 1].
        let ns2_vec: [i64; 2] = [5, 6];
        let normalized_shape2 = ArrayRef::from_raw_parts(ns2_vec.as_ptr(), ns2_vec.len());
        let mut sizes2: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let mut ndim2: usize = 0;
        unsafe {
            get_layer_norm_out_target_size(
                &in_,
                normalized_shape2,
                sizes2.as_mut_ptr(),
                &mut ndim2,
            );
        }
        assert_eq!(ndim2, 3);
        assert_eq!(&sizes2[0..3], &[4, 1, 1]);
    }
}
