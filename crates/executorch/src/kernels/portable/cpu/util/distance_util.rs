//! Literal port of kernels/portable/cpu/util/distance_util.cpp + kernels/portable/cpu/util/distance_util.h.

use crate::runtime::core::exec_aten::util::tensor_util::{
    tensor_has_rank_greater_or_equal_to, tensor_is_rank, tensors_have_same_dtype2,
    tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};

// PORT-NOTE: the crate-level `et_check_or_return_false!` drops all
// caller-supplied format arguments after the leading literal (see the analogous
// note in tensor_util.rs), so any message with `{}` placeholders fails to
// compile. This local macro mirrors the C++ `ET_CHECK_OR_RETURN_FALSE`
// faithfully (prepend "Check failed (cond): " then forward the full message +
// args). Unresolved cross-module reference.
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

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to
// `ET_CHECK_OR_RETURN_FALSE(cond, "")` in the C++ header.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// [spec:et:def:distance-util.torch.executor.check-pdist-args-fn]
// [spec:et:sem:distance-util.torch.executor.check-pdist-args-fn]
pub fn check_pdist_args(in_: &Tensor, p: f64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_is_rank(in_, 2));
    et_check_or_return_false!(
        p >= 0.0,
        "pdist only supports non-negative p values; p = {:.6}",
        p
    );
    true
}

// [spec:et:def:distance-util.torch.executor.get-pdist-out-target-size-fn]
// [spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least one valid `TensorSizesType` element and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_pdist_out_target_size(
    in_: &Tensor,
    out_sizes: *mut TensorSizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = 1;
    }
    let n: usize = in_.size(0) as usize;
    unsafe {
        *out_sizes.add(0) = (n.wrapping_mul(n.wrapping_sub(1)) / 2) as TensorSizesType;
    }
}

// [spec:et:def:distance-util.torch.executor.check-cdist-args-fn]
// [spec:et:sem:distance-util.torch.executor.check-cdist-args-fn]
pub fn check_cdist_args(
    x1: &Tensor,
    x2: &Tensor,
    p: f64,
    compute_mode: Option<i64>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(x1, x2));
    et_log_and_return_if_false!(tensors_have_same_dtype2(x1, out));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(x1, 2));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(x2, 2));
    et_log_and_return_if_false!(tensors_have_same_size_at_dims(
        x1,
        (x1.dim() - 1) as usize,
        x2,
        (x2.dim() - 1) as usize
    ));
    et_check_or_return_false!(
        p >= 0.0,
        "cdist only supports non-negative p values; p = {:.6}",
        p
    );
    if let Some(mode) = compute_mode {
        et_check_or_return_false!(
            mode >= 0 && mode <= 2,
            "possible modes: 0, 1, 2, but was: {}",
            mode
        );
    }
    true
}

//
// Norm policy structs.
//
// PORT-NOTE: the C++ norm structs are templated on `CTYPE` with `static inline`
// member functions. Rust models the "policy struct" as a trait `Norm<CTYPE>`
// whose associated functions carry the same map/reduce/finish signatures; each
// norm is a unit struct implementing it. `CTYPE` must be a float type (`Copy`,
// arithmetic, `sqrt`/`pow`/`max`/`abs`), captured by the `NormCtype` bound.

/// The element-type bound required by the norm policies: mirrors the operations
/// `Norm::map`/`reduce`/`finish` and `pdist` perform on `CTYPE`.
pub trait NormCtype:
    Copy
    + PartialEq
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn recip_pow(agg: Self, p: Self) -> Self;
    fn from_f64(v: f64) -> Self;
}

impl NormCtype for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn abs(self) -> Self {
        f32::abs(self)
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn powf(self, p: Self) -> Self {
        f32::powf(self, p)
    }
    fn recip_pow(agg: Self, p: Self) -> Self {
        // std::pow(agg, 1.0 / p) with the exponent computed in double.
        f64::powf(agg as f64, 1.0 / p as f64) as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl NormCtype for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn powf(self, p: Self) -> Self {
        f64::powf(self, p)
    }
    fn recip_pow(agg: Self, p: Self) -> Self {
        f64::powf(agg, 1.0 / p)
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

// PORT-NOTE: op_{cdist,pdist}_forward switch over ET_SWITCH_FLOATHBF16_TYPES,
// which instantiates the norm templates for Half/BFloat16 as well. c10's
// Half/BFloat16 arithmetic and `std::{abs,sqrt,pow}` overloads promote to float
// (recip_pow's exponent is computed in double, mirroring the f32 impl); results
// narrow back on store. These impls reproduce that: compute in f32/f64, narrow.
macro_rules! impl_norm_ctype_narrow_float {
    ($t:ty) => {
        impl NormCtype for $t {
            const ZERO: Self = <$t>::from_f32_const(0.0);
            const ONE: Self = <$t>::from_f32_const(1.0);
            fn abs(self) -> Self {
                <$t>::from_f32(f32::abs(self.to_f32()))
            }
            fn sqrt(self) -> Self {
                <$t>::from_f32(f32::sqrt(self.to_f32()))
            }
            fn powf(self, p: Self) -> Self {
                <$t>::from_f32(f32::powf(self.to_f32(), p.to_f32()))
            }
            fn recip_pow(agg: Self, p: Self) -> Self {
                <$t>::from_f64(f64::powf(agg.to_f64(), 1.0 / p.to_f64()))
            }
            fn from_f64(v: f64) -> Self {
                <$t>::from_f64(v)
            }
        }
    };
}
impl_norm_ctype_narrow_float!(crate::runtime::core::portable_type::Half);
impl_norm_ctype_narrow_float!(crate::runtime::core::portable_type::BFloat16);

/// The map/reduce/finish policy for a norm, mirroring the C++ `Norm` template
/// parameter.
pub trait Norm<CTYPE: NormCtype> {
    fn map(diff: CTYPE, p: CTYPE) -> CTYPE;
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE;
    fn finish(agg: CTYPE, p: CTYPE) -> CTYPE;
}

// [spec:et:def:distance-util.torch.executor.l0]
pub struct L0;
impl<CTYPE: NormCtype> Norm<CTYPE> for L0 {
    // [spec:et:def:distance-util.torch.executor.l0.map-fn]
    // [spec:et:sem:distance-util.torch.executor.l0.map-fn]
    fn map(diff: CTYPE, _p: CTYPE) -> CTYPE {
        if diff == CTYPE::ZERO {
            CTYPE::ZERO
        } else {
            CTYPE::ONE
        }
    }
    // [spec:et:def:distance-util.torch.executor.l0.reduce-fn]
    // [spec:et:sem:distance-util.torch.executor.l0.reduce-fn]
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE {
        agg + up
    }
    // [spec:et:def:distance-util.torch.executor.l0.finish-fn]
    // [spec:et:sem:distance-util.torch.executor.l0.finish-fn]
    fn finish(agg: CTYPE, _p: CTYPE) -> CTYPE {
        agg
    }
}

// [spec:et:def:distance-util.torch.executor.l1]
pub struct L1;
impl<CTYPE: NormCtype> Norm<CTYPE> for L1 {
    // [spec:et:def:distance-util.torch.executor.l1.map-fn]
    // [spec:et:sem:distance-util.torch.executor.l1.map-fn]
    fn map(diff: CTYPE, _p: CTYPE) -> CTYPE {
        diff
    }
    // [spec:et:def:distance-util.torch.executor.l1.reduce-fn]
    // [spec:et:sem:distance-util.torch.executor.l1.reduce-fn]
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE {
        agg + up
    }
    // [spec:et:def:distance-util.torch.executor.l1.finish-fn]
    // [spec:et:sem:distance-util.torch.executor.l1.finish-fn]
    fn finish(agg: CTYPE, _p: CTYPE) -> CTYPE {
        agg
    }
}

// [spec:et:def:distance-util.torch.executor.l2]
pub struct L2;
impl<CTYPE: NormCtype> Norm<CTYPE> for L2 {
    // [spec:et:def:distance-util.torch.executor.l2.map-fn]
    // [spec:et:sem:distance-util.torch.executor.l2.map-fn]
    fn map(diff: CTYPE, _p: CTYPE) -> CTYPE {
        diff * diff
    }
    // [spec:et:def:distance-util.torch.executor.l2.reduce-fn]
    // [spec:et:sem:distance-util.torch.executor.l2.reduce-fn]
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE {
        agg + up
    }
    // [spec:et:def:distance-util.torch.executor.l2.finish-fn]
    // [spec:et:sem:distance-util.torch.executor.l2.finish-fn]
    fn finish(agg: CTYPE, _p: CTYPE) -> CTYPE {
        agg.sqrt()
    }
}

// [spec:et:def:distance-util.torch.executor.lp]
pub struct Lp;
impl<CTYPE: NormCtype> Norm<CTYPE> for Lp {
    // [spec:et:def:distance-util.torch.executor.lp.map-fn]
    // [spec:et:sem:distance-util.torch.executor.lp.map-fn]
    fn map(diff: CTYPE, p: CTYPE) -> CTYPE {
        diff.powf(p)
    }
    // [spec:et:def:distance-util.torch.executor.lp.reduce-fn]
    // [spec:et:sem:distance-util.torch.executor.lp.reduce-fn]
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE {
        agg + up
    }
    // [spec:et:def:distance-util.torch.executor.lp.finish-fn]
    // [spec:et:sem:distance-util.torch.executor.lp.finish-fn]
    fn finish(agg: CTYPE, p: CTYPE) -> CTYPE {
        CTYPE::recip_pow(agg, p)
    }
}

// [spec:et:def:distance-util.torch.executor.linf]
pub struct Linf;
impl<CTYPE: NormCtype> Norm<CTYPE> for Linf {
    // [spec:et:def:distance-util.torch.executor.linf.map-fn]
    // [spec:et:sem:distance-util.torch.executor.linf.map-fn]
    fn map(diff: CTYPE, _p: CTYPE) -> CTYPE {
        diff
    }
    // [spec:et:def:distance-util.torch.executor.linf.reduce-fn]
    // [spec:et:sem:distance-util.torch.executor.linf.reduce-fn]
    fn reduce(agg: CTYPE, up: CTYPE) -> CTYPE {
        // std::max(agg, up) — returns the first argument on a tie.
        if up > agg { up } else { agg }
    }
    // [spec:et:def:distance-util.torch.executor.linf.finish-fn]
    // [spec:et:sem:distance-util.torch.executor.linf.finish-fn]
    fn finish(agg: CTYPE, _p: CTYPE) -> CTYPE {
        agg
    }
}

// [spec:et:def:distance-util.torch.executor.pdist-fn]
// [spec:et:sem:distance-util.torch.executor.pdist-fn]
pub fn pdist_norm<CTYPE: NormCtype, N: Norm<CTYPE>>(in_: &Tensor, out: &Tensor, p: f64) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let n: usize = in_.size(0) as usize;
    let m: usize = in_.size(1) as usize;

    let p_ct = CTYPE::from_f64(p);

    let mut out_ix: usize = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let row_i: *const CTYPE = unsafe { in_data.add(i * m) };
            let row_j: *const CTYPE = unsafe { in_data.add(j * m) };
            let mut agg: CTYPE = CTYPE::ZERO;
            for k in 0..m {
                let diff: CTYPE = (unsafe { *row_i.add(k) } - unsafe { *row_j.add(k) }).abs();
                agg = N::reduce(agg, N::map(diff, p_ct));
            }
            unsafe {
                *out_data.add(out_ix) = N::finish(agg, p_ct);
            }
            out_ix += 1;
        }
    }
}

/// Norm-selecting `pdist<CTYPE>(in, out, p)` overload that dispatches on the
/// runtime `p` (exact floating-point equality) and wraps the worker.
pub fn pdist<CTYPE: NormCtype>(in_: &Tensor, out: &Tensor, p: f64) {
    if p == 0.0 {
        pdist_norm::<CTYPE, L0>(in_, out, p);
    } else if p == 1.0 {
        pdist_norm::<CTYPE, L1>(in_, out, p);
    } else if p == 2.0 {
        pdist_norm::<CTYPE, L2>(in_, out, p);
    } else if p == f64::INFINITY {
        pdist_norm::<CTYPE, Linf>(in_, out, p);
    } else {
        pdist_norm::<CTYPE, Lp>(in_, out, p);
    }
}
