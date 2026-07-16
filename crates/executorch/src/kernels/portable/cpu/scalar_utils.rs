//! Literal port of kernels/portable/cpu/scalar_utils.h.

use crate::runtime::core::exec_aten::util::scalar_type_util::{
    is_bits_type, is_complex_type, is_floating_type, is_integral_type, is_qint_type, to_string,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;

// PORT-NOTE: `ET_CHECK_MSG` / `ET_CHECK` are the fatal-abort macros from
// runtime/platform/assert.h, which has no ported target yet. Local macros
// mirror their semantics (abort via the PAL path), matching scalar.rs /
// tensor_util.rs. Message formatting is dropped since a fatal abort follows.
// Unresolved cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `ET_CHECK_SCALAR_SAME_TYPE` and `ET_EXTRACT_SCALAR` are convenience
// macros used by scalar-variant kernels. `ET_EXTRACT_SCALAR(scalar, out_val)`
// wraps `extract_scalar` in an `ET_CHECK_MSG`. Ported as macros below;
// `ET_CHECK_SCALAR_SAME_TYPE` mirrors the three category-equality checks.

// #define ET_CHECK_SCALAR_SAME_TYPE(a__, b__)
#[macro_export]
macro_rules! et_check_scalar_same_type {
    ($a:expr, $b:expr) => {{
        let a__ = $a;
        let b__ = $b;
        if a__.is_boolean() != b__.is_boolean()
            || a__.is_integral(false) != b__.is_integral(false)
            || a__.is_floating_point() != b__.is_floating_point()
        {
            $crate::runtime::platform::abort::runtime_abort();
        }
    }};
}

/// Returns the dtype associated with a Scalar that reflects the category
/// of value stored by the Scalar.
// [spec:et:def:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn]
pub fn get_scalar_dtype(scalar: Scalar) -> ScalarType {
    if scalar.is_boolean() {
        return ScalarType::Bool;
    }
    if scalar.is_integral(false) {
        return ScalarType::Long;
    }
    if scalar.is_floating_point() {
        return ScalarType::Double;
    }
    et_check_msg!(false, "Scalar must be Boolean, Integral or Floating.");
    // PORT-NOTE: `ET_CHECK_MSG(false, ...)` never returns; unreachable after abort.
    unreachable!()
}

// [spec:et:def:scalar-utils.torch.executor.native.utils.scalars-have-same-dtype-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.scalars-have-same-dtype-fn]
pub fn scalars_have_same_dtype(a: Scalar, b: Scalar) -> bool {
    let a_dtype: ScalarType = get_scalar_dtype(a);
    let b_dtype: ScalarType = get_scalar_dtype(b);
    if a_dtype == b_dtype {
        return true;
    }
    crate::et_log!(
        Error,
        "Expected scalars to have the same dtype, but found {} and {}",
        to_string(a_dtype),
        to_string(b_dtype)
    );
    false
}

// PORT-NOTE: `promote_type_with_scalar_type` is a compile-time-only C++ struct
// (a nested `std::conditional` tree yielding `::type`). It performs pure
// type-level metaprogramming with no runtime body, so there is no faithful
// literal runtime translation; the equivalent value-level computation is the
// runtime `promote_type_with_scalar` below. Ported as a documentation-only note.
// [spec:et:def:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-type]

/// Implement type promotion between a tensor's ScalarType and a Scalar.
// [spec:et:def:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]
pub fn promote_type_with_scalar(
    mut t: ScalarType,
    scalar: Scalar,
    half_to_float: bool,
) -> ScalarType {
    if half_to_float && t == ScalarType::Half {
        t = ScalarType::Float;
    }

    // QInt, and Bits types not supported
    et_check!(!is_qint_type(t));
    et_check!(!is_bits_type(t));

    if is_complex_type(t) {
        return t;
    }
    if scalar.is_floating_point() {
        if is_floating_type(t) {
            return t;
        } else {
            // ATen will promote to Float instead of Double
            return ScalarType::Float;
        }
    }
    if scalar.is_integral(false) {
        if is_floating_type(t) || is_integral_type(t, false) {
            return t;
        } else {
            return ScalarType::Long;
        }
    }
    if scalar.is_boolean() {
        return t;
    }
    et_check_msg!(false, "Scalar must be Boolean, Integral or Floating.");
    unreachable!()
}

// ---------------------------------------------------------------------------
// extract_scalar
// ---------------------------------------------------------------------------

// PORT-NOTE: `extract_scalar` is a set of three enable_if-selected overloads on
// the destination type. Ported as one trait `ExtractScalar` with a per-type
// impl carrying each overload's body verbatim.
pub trait ExtractScalar: Sized {
    fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool;
}

// Integer overload (INT_T integral and not bool).
macro_rules! impl_extract_scalar_int {
    ($t:ty) => {
        impl ExtractScalar for $t {
            fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
                if !scalar.is_integral(/*include_bool=*/ false) {
                    return false;
                }
                let val: i64 = scalar.to_i64();
                if val < (<$t>::MIN as i64) || val > (<$t>::MAX as i64) {
                    // PyTorch's clamp() raises if min/max cannot be represented
                    // as the dtype, so we fail too.
                    return false;
                }
                *out_val = val as $t;
                true
            }
        }
    };
}
impl_extract_scalar_int!(u8);
impl_extract_scalar_int!(i8);
impl_extract_scalar_int!(i16);
impl_extract_scalar_int!(i32);
impl_extract_scalar_int!(i64);

// Floating overload (FLOAT_T floating point).
// [spec:et:def:scalar-utils.torch.executor.native.utils.extract-scalar-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.extract-scalar-fn]
macro_rules! impl_extract_scalar_float {
    ($t:ty) => {
        impl ExtractScalar for $t {
            fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
                let val: f64;
                if scalar.is_floating_point() {
                    val = scalar.to_f64();
                    // Finite out-of-range fails; infinite/NaN are allowed
                    // through (float can represent them).
                    if val.is_finite() && (val < (<$t>::MIN as f64) || val > (<$t>::MAX as f64)) {
                        return false;
                    }
                } else if scalar.is_integral(/*include_bool=*/ false) {
                    val = scalar.to_i64() as f64;
                } else {
                    // Not a numeric Scalar.
                    return false;
                }
                *out_val = val as $t;
                true
            }
        }
    };
}
impl_extract_scalar_float!(f32);
impl_extract_scalar_float!(f64);

// Bool overload (BOOL_T == bool).
impl ExtractScalar for bool {
    fn extract_scalar(scalar: Scalar, out_val: &mut Self) -> bool {
        if scalar.is_integral(false) {
            *out_val = scalar.to_i64() != 0;
            return true;
        }
        if scalar.is_boolean() {
            *out_val = scalar.to_bool_val();
            return true;
        }
        false
    }
}

/// Extracts a value from a Scalar into `out_val`; returns `true` on success.
pub fn extract_scalar<T: ExtractScalar>(scalar: Scalar, out_val: &mut T) -> bool {
    T::extract_scalar(scalar, out_val)
}

// #define ET_EXTRACT_SCALAR(scalar, out_val)
#[macro_export]
macro_rules! et_extract_scalar {
    ($scalar:expr, $out_val:expr) => {
        if !$crate::kernels::portable::cpu::scalar_utils::extract_scalar($scalar, &mut $out_val) {
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// ---------------------------------------------------------------------------
// scalar_to
// ---------------------------------------------------------------------------

// PORT-NOTE: primary `scalar_to<T>` plus full specializations for `double` and
// `int64_t`. Ported as a trait `ScalarTo` with the primary body as the default
// impl (via macro) and specialized bodies for f64/i64.
pub trait ScalarTo: Sized {
    fn scalar_to(s: &Scalar) -> Self;
}

// Primary template: dispatch on category, static_cast to T.
// [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-fn]
macro_rules! impl_scalar_to_primary {
    ($t:ty) => {
        impl ScalarTo for $t {
            fn scalar_to(s: &Scalar) -> Self {
                if s.is_boolean() {
                    s.to_bool_val() as u8 as $t
                } else if s.is_floating_point() {
                    s.to_f64() as $t
                } else {
                    s.to_i64() as $t
                }
            }
        }
    };
}
impl_scalar_to_primary!(f32);
impl_scalar_to_primary!(i8);
impl_scalar_to_primary!(i16);
impl_scalar_to_primary!(i32);
impl_scalar_to_primary!(u8);

// PORT-NOTE: primary-template `scalar_to<T>` for `c10::Half`/`BFloat16`, whose
// `static_cast<T>(double)` maps to `from_f64`. Added for kernels dispatching over
// REALHBF16 (e.g. addmm) that request `scalar_to<Half/BFloat16>`.
impl ScalarTo for crate::runtime::core::portable_type::Half {
    fn scalar_to(s: &Scalar) -> Self {
        use crate::runtime::core::portable_type::Half;
        if s.is_boolean() {
            Half::from_f64(s.to_bool_val() as u8 as f64)
        } else if s.is_floating_point() {
            Half::from_f64(s.to_f64())
        } else {
            Half::from_f64(s.to_i64() as f64)
        }
    }
}
impl ScalarTo for crate::runtime::core::portable_type::BFloat16 {
    fn scalar_to(s: &Scalar) -> Self {
        use crate::runtime::core::portable_type::BFloat16;
        if s.is_boolean() {
            BFloat16::from_f64(s.to_bool_val() as u8 as f64)
        } else if s.is_floating_point() {
            BFloat16::from_f64(s.to_f64())
        } else {
            BFloat16::from_f64(s.to_i64() as f64)
        }
    }
}

// Full specialization scalar_to<double>.
// [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-double-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-double-fn]
impl ScalarTo for f64 {
    fn scalar_to(s: &Scalar) -> Self {
        if s.is_floating_point() {
            s.to_f64()
        } else {
            s.to_i64() as f64
        }
    }
}

// Full specialization scalar_to<int64_t>.
// [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn]
impl ScalarTo for i64 {
    fn scalar_to(s: &Scalar) -> Self {
        if s.is_floating_point() {
            s.to_f64() as i64
        } else {
            s.to_i64()
        }
    }
}

// Primary-template instantiation for bool. `u8 as bool` is not a valid Rust
// cast, so the `static_cast<bool>(...)` of the C++ primary template is expressed
// as a `!= 0` truncation of the same category branches (nonzero -> true).
// [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-fn]
// [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-fn]
impl ScalarTo for bool {
    fn scalar_to(s: &Scalar) -> Self {
        if s.is_boolean() {
            s.to_bool_val()
        } else if s.is_floating_point() {
            s.to_f64() != 0.0
        } else {
            s.to_i64() != 0
        }
    }
}

pub fn scalar_to<T: ScalarTo>(s: &Scalar) -> T {
    T::scalar_to(s)
}

// ---------------------------------------------------------------------------
// internal::check_overflow_cast / check_overflow_scalar_cast
// ---------------------------------------------------------------------------

pub mod internal {
    use super::Scalar;

    // PORT-NOTE: `c10::overflows<To, From>(in)` implements PyTorch's
    // representability check. The vendored c10 header is out of scope; this
    // trait reproduces the range/finiteness checks for the (To, From) pairs
    // used by ExecuTorch: integer From (i64), floating From (f64), and bool
    // From. For integer destinations from a floating source, a value overflows
    // if it is non-finite or lies outside the destination's integer range; for
    // integer-to-integer, if outside the destination range; casts to a floating
    // destination never overflow here (matching c10 for the float targets used).
    pub trait Overflows<From> {
        fn overflows(v: From) -> bool;
    }

    macro_rules! impl_overflows_int_from_i64 {
        ($t:ty) => {
            impl Overflows<i64> for $t {
                fn overflows(v: i64) -> bool {
                    v < (<$t>::MIN as i64) || v > (<$t>::MAX as i64)
                }
            }
        };
    }
    impl_overflows_int_from_i64!(i8);
    impl_overflows_int_from_i64!(i16);
    impl_overflows_int_from_i64!(i32);
    impl Overflows<i64> for i64 {
        fn overflows(_v: i64) -> bool {
            false
        }
    }
    impl Overflows<i64> for u8 {
        fn overflows(v: i64) -> bool {
            v < (u8::MIN as i64) || v > (u8::MAX as i64)
        }
    }

    macro_rules! impl_overflows_int_from_f64 {
        ($t:ty) => {
            impl Overflows<f64> for $t {
                fn overflows(v: f64) -> bool {
                    !v.is_finite() || v < (<$t>::MIN as f64) || v > (<$t>::MAX as f64)
                }
            }
        };
    }
    impl_overflows_int_from_f64!(i8);
    impl_overflows_int_from_f64!(i16);
    impl_overflows_int_from_f64!(i32);
    impl_overflows_int_from_f64!(i64);
    impl Overflows<f64> for u8 {
        fn overflows(v: f64) -> bool {
            !v.is_finite() || v < (u8::MIN as f64) || v > (u8::MAX as f64)
        }
    }

    // Floating destinations: representable range spans; only guard against
    // finite values that exceed the narrower float's finite range (f64 -> f32).
    impl Overflows<f64> for f32 {
        fn overflows(v: f64) -> bool {
            v.is_finite() && (v < (f32::MIN as f64) || v > (f32::MAX as f64))
        }
    }
    impl Overflows<f64> for f64 {
        fn overflows(_v: f64) -> bool {
            false
        }
    }
    // From bool (0/1) never overflows an arithmetic destination.
    macro_rules! impl_overflows_from_bool {
        ($t:ty) => {
            impl Overflows<bool> for $t {
                fn overflows(_v: bool) -> bool {
                    false
                }
            }
        };
    }
    impl_overflows_from_bool!(i8);
    impl_overflows_from_bool!(i16);
    impl_overflows_from_bool!(i32);
    impl_overflows_from_bool!(i64);
    impl_overflows_from_bool!(u8);
    impl_overflows_from_bool!(f32);
    impl_overflows_from_bool!(f64);

    // PORT-NOTE: floating destinations from an i64 source never overflow their
    // (much wider) finite range, matching c10::overflows for these pairs. Needed
    // because check_overflow_scalar_cast<To> requires Overflows<i64> for every To
    // in the REALHBBF16 switch, including f32/f64.
    impl Overflows<i64> for f32 {
        fn overflows(_v: i64) -> bool {
            false
        }
    }
    impl Overflows<i64> for f64 {
        fn overflows(_v: i64) -> bool {
            false
        }
    }

    // PORT-NOTE: bool is a destination in the REALHBBF16 set. check_overflow_cast
    // short-circuits on To::IS_BOOL before ever calling overflows(), so these
    // bodies are never reached; they exist only to satisfy the trait bounds of
    // check_overflow_scalar_cast<bool>.
    impl Overflows<bool> for bool {
        fn overflows(_v: bool) -> bool {
            false
        }
    }
    impl Overflows<f64> for bool {
        fn overflows(_v: f64) -> bool {
            false
        }
    }
    impl Overflows<i64> for bool {
        fn overflows(_v: i64) -> bool {
            false
        }
    }

    // PORT-NOTE: models the `static_cast<To>(in)` in check_overflow_cast for
    // each (To, From) pair used.
    pub trait CastFrom<From> {
        fn cast_from(v: From) -> Self;
        const IS_BOOL: bool;
    }
    macro_rules! impl_cast_from {
        ($to:ty, $from:ty) => {
            impl CastFrom<$from> for $to {
                fn cast_from(v: $from) -> Self {
                    v as $to
                }
                const IS_BOOL: bool = false;
            }
        };
    }
    impl_cast_from!(i8, i64);
    impl_cast_from!(i16, i64);
    impl_cast_from!(i32, i64);
    impl_cast_from!(i64, i64);
    impl_cast_from!(u8, i64);
    impl_cast_from!(f32, i64);
    impl_cast_from!(f64, i64);
    impl_cast_from!(i8, f64);
    impl_cast_from!(i16, f64);
    impl_cast_from!(i32, f64);
    impl_cast_from!(i64, f64);
    impl_cast_from!(u8, f64);
    impl_cast_from!(f32, f64);
    impl_cast_from!(f64, f64);
    // bool destination: `To == bool` is excluded from the overflow check.
    impl CastFrom<i64> for bool {
        fn cast_from(v: i64) -> Self {
            v != 0
        }
        const IS_BOOL: bool = true;
    }
    impl CastFrom<f64> for bool {
        fn cast_from(v: f64) -> Self {
            v != 0.0
        }
        const IS_BOOL: bool = true;
    }
    // From bool source casts.
    macro_rules! impl_cast_from_bool {
        ($to:ty) => {
            impl CastFrom<bool> for $to {
                fn cast_from(v: bool) -> Self {
                    v as u8 as $to
                }
                const IS_BOOL: bool = false;
            }
        };
    }
    impl_cast_from_bool!(i8);
    impl_cast_from_bool!(i16);
    impl_cast_from_bool!(i32);
    impl_cast_from_bool!(i64);
    impl_cast_from_bool!(u8);
    impl_cast_from_bool!(f32);
    impl_cast_from_bool!(f64);
    impl CastFrom<bool> for bool {
        fn cast_from(v: bool) -> Self {
            v
        }
        const IS_BOOL: bool = true;
    }

    // PORT-NOTE: Half/BFloat16 destinations mirror the floating-destination
    // behavior of `c10::overflows`/`static_cast<To>` used by check_overflow_cast
    // over the REALHBBF16 switch set. Overflow guards convert to f64 and compare
    // against the narrow float's finite range (same shape as the `f32` impls
    // above); casts route through the half crate's f64 conversions.
    use crate::runtime::core::portable_type::{BFloat16, Half};

    macro_rules! impl_overflows_narrow_float {
        ($t:ty) => {
            impl Overflows<f64> for $t {
                fn overflows(v: f64) -> bool {
                    v.is_finite() && (v < <$t>::MIN.to_f64() || v > <$t>::MAX.to_f64())
                }
            }
            impl Overflows<i64> for $t {
                fn overflows(v: i64) -> bool {
                    (v as f64) < <$t>::MIN.to_f64() || (v as f64) > <$t>::MAX.to_f64()
                }
            }
            impl Overflows<bool> for $t {
                fn overflows(_v: bool) -> bool {
                    false
                }
            }
            impl CastFrom<i64> for $t {
                fn cast_from(v: i64) -> Self {
                    <$t>::from_f64(v as f64)
                }
                const IS_BOOL: bool = false;
            }
            impl CastFrom<f64> for $t {
                fn cast_from(v: f64) -> Self {
                    <$t>::from_f64(v)
                }
                const IS_BOOL: bool = false;
            }
            impl CastFrom<bool> for $t {
                fn cast_from(v: bool) -> Self {
                    <$t>::from_f64(v as u8 as f64)
                }
                const IS_BOOL: bool = false;
            }
        };
    }
    impl_overflows_narrow_float!(Half);
    impl_overflows_narrow_float!(BFloat16);

    // [spec:et:def:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn]
    // [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn]
    pub fn check_overflow_cast<To, From>(in_: From) -> Option<To>
    where
        To: CastFrom<From> + Overflows<From>,
        From: Copy,
    {
        // Converting to bool can't overflow so we exclude that case.
        if !To::IS_BOOL && To::overflows(in_) {
            return None;
        }
        Some(To::cast_from(in_))
    }

    // [spec:et:def:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]
    // [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]
    pub fn check_overflow_scalar_cast<To>(in_: &Scalar) -> Option<To>
    where
        To: CastFrom<bool>
            + Overflows<bool>
            + CastFrom<f64>
            + Overflows<f64>
            + CastFrom<i64>
            + Overflows<i64>,
    {
        if in_.is_boolean() {
            check_overflow_cast::<To, bool>(in_.to_bool_val())
        } else if in_.is_floating_point() {
            check_overflow_cast::<To, f64>(in_.to_f64())
        } else {
            check_overflow_cast::<To, i64>(in_.to_i64())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::internal::{check_overflow_cast, check_overflow_scalar_cast};
    use super::*;

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn/test]
    #[test]
    fn get_scalar_dtype_categories() {
        // Ordering: boolean classified before integral.
        assert_eq!(get_scalar_dtype(Scalar::from_bool(true)), ScalarType::Bool);
        assert_eq!(get_scalar_dtype(Scalar::from_i64(7)), ScalarType::Long);
        assert_eq!(
            get_scalar_dtype(Scalar::from_double(3.5)),
            ScalarType::Double
        );
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.scalars-have-same-dtype-fn/test]
    #[test]
    fn scalars_have_same_dtype_category_match() {
        // The mismatch branch logs at Error level, which routes through the PAL.
        crate::runtime::platform::platform::pal_init();
        assert!(scalars_have_same_dtype(
            Scalar::from_i64(1),
            Scalar::from_i64(9)
        ));
        assert!(scalars_have_same_dtype(
            Scalar::from_double(1.0),
            Scalar::from_double(2.0)
        ));
        assert!(scalars_have_same_dtype(
            Scalar::from_bool(false),
            Scalar::from_bool(true)
        ));
        // Different categories: Long vs Double, Long vs Bool.
        assert!(!scalars_have_same_dtype(
            Scalar::from_i64(1),
            Scalar::from_double(1.0)
        ));
        assert!(!scalars_have_same_dtype(
            Scalar::from_i64(1),
            Scalar::from_bool(true)
        ));
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn/test]
    #[test]
    fn promote_type_with_scalar_semantics() {
        let f = Scalar::from_double(1.0);
        let i = Scalar::from_i64(1);
        let b = Scalar::from_bool(true);

        // Float scalar: floating tensor preserved, else promotes to Float (not Double).
        assert_eq!(
            promote_type_with_scalar(ScalarType::Double, f, false),
            ScalarType::Double
        );
        assert_eq!(
            promote_type_with_scalar(ScalarType::Int, f, false),
            ScalarType::Float
        );

        // Integer scalar: floating or integral tensor preserved; Bool tensor -> Long.
        assert_eq!(
            promote_type_with_scalar(ScalarType::Float, i, false),
            ScalarType::Float
        );
        assert_eq!(
            promote_type_with_scalar(ScalarType::Int, i, false),
            ScalarType::Int
        );
        assert_eq!(
            promote_type_with_scalar(ScalarType::Bool, i, false),
            ScalarType::Long
        );

        // Boolean scalar: tensor dtype unchanged.
        assert_eq!(
            promote_type_with_scalar(ScalarType::Int, b, false),
            ScalarType::Int
        );

        // half_to_float promotes Half -> Float before the rest of the logic runs.
        assert_eq!(
            promote_type_with_scalar(ScalarType::Half, i, true),
            ScalarType::Float
        );
        assert_eq!(
            promote_type_with_scalar(ScalarType::Half, i, false),
            ScalarType::Half
        );
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.extract-scalar-fn]
    // [spec:et:sem:scalar-utils.torch.executor.native.utils.extract-scalar-fn/test]
    #[test]
    fn extract_scalar_overloads() {
        // Integer overload: in-range succeeds, out-of-range fails, non-integral fails.
        let mut i: i8 = 0;
        assert!(extract_scalar(Scalar::from_i64(5), &mut i));
        assert_eq!(i, 5);
        assert!(!extract_scalar(Scalar::from_i64(200), &mut i)); // > i8::MAX
        assert!(!extract_scalar(Scalar::from_double(1.5), &mut i)); // not integral

        // Floating overload: float and integer sources succeed; finite out-of-range
        // fails; infinite passes through.
        let mut f: f32 = 0.0;
        assert!(extract_scalar(Scalar::from_double(2.5), &mut f));
        assert_eq!(f, 2.5);
        assert!(extract_scalar(Scalar::from_i64(3), &mut f));
        assert_eq!(f, 3.0);
        assert!(!extract_scalar(Scalar::from_double(1e40), &mut f)); // finite, > f32::MAX
        assert!(extract_scalar(Scalar::from_double(f64::INFINITY), &mut f)); // inf allowed
        assert!(f.is_infinite());

        // Bool overload: integral nonzero -> true; boolean passes; float fails.
        let mut b: bool = false;
        assert!(extract_scalar(Scalar::from_i64(0), &mut b));
        assert!(!b);
        assert!(extract_scalar(Scalar::from_i64(2), &mut b));
        assert!(b);
        assert!(extract_scalar(Scalar::from_bool(true), &mut b));
        assert!(b);
        assert!(!extract_scalar(Scalar::from_double(1.0), &mut b));
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-fn/test]
    // [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-double-fn/test]
    // [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn/test]
    #[test]
    fn scalar_to_specializations() {
        // Primary template (i32): dispatch on category, static_cast.
        assert_eq!(scalar_to::<i32>(&Scalar::from_bool(true)), 1);
        assert_eq!(scalar_to::<i32>(&Scalar::from_double(2.9)), 2); // trunc toward zero
        assert_eq!(scalar_to::<i32>(&Scalar::from_i64(7)), 7);

        // double specialization: float read as-is, else via to<int64_t>.
        assert_eq!(scalar_to::<f64>(&Scalar::from_double(3.25)), 3.25);
        assert_eq!(scalar_to::<f64>(&Scalar::from_bool(true)), 1.0);
        assert_eq!(scalar_to::<f64>(&Scalar::from_i64(4)), 4.0);

        // int64 specialization: float truncates toward zero; bool false->0.
        assert_eq!(scalar_to::<i64>(&Scalar::from_double(-2.9)), -2);
        assert_eq!(scalar_to::<i64>(&Scalar::from_bool(false)), 0);
        assert_eq!(scalar_to::<i64>(&Scalar::from_i64(9)), 9);
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn/test]
    #[test]
    fn check_overflow_cast_semantics() {
        // In-range int->int succeeds.
        assert_eq!(check_overflow_cast::<i8, i64>(100), Some(100i8));
        // Out-of-range int->int returns None.
        assert_eq!(check_overflow_cast::<i8, i64>(200), None);
        // Float->int: finite out-of-range and non-finite both overflow.
        assert_eq!(check_overflow_cast::<i8, f64>(5.0), Some(5i8));
        assert_eq!(check_overflow_cast::<i8, f64>(1000.0), None);
        assert_eq!(check_overflow_cast::<i8, f64>(f64::NAN), None);
        // Casting to bool is exempt from the overflow check: any value maps.
        assert_eq!(check_overflow_cast::<bool, i64>(999_999), Some(true));
        assert_eq!(check_overflow_cast::<bool, i64>(0), Some(false));
    }

    // [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn/test]
    #[test]
    fn check_overflow_scalar_cast_dispatch() {
        // Integral source within range.
        assert_eq!(
            check_overflow_scalar_cast::<i8>(&Scalar::from_i64(10)),
            Some(10i8)
        );
        // Integral source out of range -> None.
        assert_eq!(
            check_overflow_scalar_cast::<i8>(&Scalar::from_i64(300)),
            None
        );
        // Floating source dispatch, out of range -> None.
        assert_eq!(
            check_overflow_scalar_cast::<i8>(&Scalar::from_double(500.0)),
            None
        );
        // Boolean source never overflows a bool destination.
        assert_eq!(
            check_overflow_scalar_cast::<bool>(&Scalar::from_bool(true)),
            Some(true)
        );
    }
}
