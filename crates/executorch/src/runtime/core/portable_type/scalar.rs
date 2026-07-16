//! Literal port of runtime/core/portable_type/scalar.h.

use crate::runtime::core::tag::Tag;
use half::{bf16, f16};

// PORT-NOTE: `ET_CHECK_MSG` is defined in runtime/platform/assert.h, which has
// no ported `assert.rs` target yet. This local macro mirrors its semantics
// (emit message, then abort via the PAL abort path), matching the pattern used
// in runtime/platform/profiler.rs. Should be replaced by the shared
// `et_check_msg!` once the assert module is ported. Unresolved cross-module
// reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:scalar.executorch.runtime.etensor.scalar.v-t]
// PORT-NOTE: C++ untagged `union v_t` modeled as a Rust `union`. The empty
// `v_t()` default constructor (v-t-fn) has no direct analog; the payload is
// always written together with `tag` by the enclosing `Scalar` constructor, so
// its markers collapse onto this union definition.
// [spec:et:def:scalar.executorch.runtime.etensor.scalar.v-t.v-t-fn]
// [spec:et:sem:scalar.executorch.runtime.etensor.scalar.v-t.v-t-fn]
#[derive(Clone, Copy)]
union VT {
    as_double: f64,
    as_int: i64,
    as_bool: bool,
}

/// Represents a scalar value.
///
/// The API is a source-compatible subset of c10::Scalar, and the
/// semantics/behavior should also match the c10 version.
// [spec:et:def:scalar.executorch.runtime.etensor.scalar]
#[derive(Clone, Copy)]
pub struct Scalar {
    tag: Tag,
    v: VT,
}

impl Scalar {
    /// `Scalar() : Scalar(int64_t(0)) {}`
    pub fn new() -> Self {
        Scalar::from_i64(0i64)
    }

    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.scalar-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.scalar-fn]
    // PORT-NOTE: templated `Scalar(T val)` over integral `T`. The C++ SFINAE
    // set (all integral types except bool) is represented here by the concrete
    // `i64` entry point; callers widen/sign-extend to `i64` per C++ integral
    // conversion at the call site, matching `static_cast<int64_t>(val)`.
    pub fn from_i64(val: i64) -> Self {
        Scalar {
            tag: Tag::Int,
            v: VT { as_int: val as i64 },
        }
    }

    /// `Scalar(bool val) : tag(Tag::Bool)`
    pub fn from_bool(val: bool) -> Self {
        Scalar {
            tag: Tag::Bool,
            v: VT { as_bool: val },
        }
    }

    /// `Scalar(double val) : tag(Tag::Double)`
    pub fn from_double(val: f64) -> Self {
        Scalar {
            tag: Tag::Double,
            v: VT { as_double: val },
        }
    }

    /// `Scalar(BFloat16 val) : Scalar((double)(float)val) {}`
    pub fn from_bfloat16(val: bf16) -> Self {
        Scalar::from_double(val.to_f32() as f64)
    }

    /// `Scalar(Half val) : Scalar((double)(float)val) {}`
    pub fn from_half(val: f16) -> Self {
        Scalar::from_double(val.to_f32() as f64)
    }

    /// Returns the concrete scalar value stored within.
    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-fn]
    // PORT-NOTE: `template <typename T> T to() const` is defined (via
    // ET_DEFINE_SCALAR_TO_METHOD) only for double, int64_t, and bool. Modeled
    // here as the three concrete conversion methods `to_double`, `to_int`,
    // `to_bool` below rather than an open generic.

    /// Returns true if the scalar is integral, false otherwise.
    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-integral-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-integral-fn]
    pub fn is_integral(&self, include_bool: bool) -> bool {
        Tag::Int == self.tag || (include_bool && self.is_boolean())
    }

    /// Returns true if the scalar is a floating point, false otherwise.
    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-floating-point-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-floating-point-fn]
    pub fn is_floating_point(&self) -> bool {
        self.tag == Tag::Double
    }

    /// Returns true if the scalar is a boolean, false otherwise.
    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]
    pub fn is_boolean(&self) -> bool {
        self.tag == Tag::Bool
    }

    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-int-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn]
    fn to_int(&self) -> i64 {
        if self.is_integral(/*include_bool=*/ false) {
            unsafe { self.v.as_int }
        } else if self.is_boolean() {
            unsafe { self.v.as_bool as i64 }
        } else {
            et_check_msg!(false, "Scalar is not an int nor a Boolean.");
            // PORT-NOTE: `ET_CHECK_MSG(false, ...)` never returns; the abort
            // above diverges so this line is unreachable.
            unreachable!()
        }
    }

    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn]
    fn to_floating_point(&self) -> f64 {
        et_check_msg!(self.is_floating_point(), "Scalar is not a Double.");
        unsafe { self.v.as_double }
    }

    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-double-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-double-fn]
    fn to_double(&self) -> f64 {
        et_check_msg!(self.is_floating_point(), "Scalar is not a Double.");
        unsafe { self.v.as_double }
    }

    // [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-bool-fn]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-bool-fn]
    fn to_bool(&self) -> bool {
        et_check_msg!(self.is_boolean(), "Scalar is not a Boolean.");
        unsafe { self.v.as_bool }
    }

    // ET_DEFINE_SCALAR_TO_METHOD(double, Double)
    pub fn to_f64(&self) -> f64 {
        self.to_double()
    }

    // ET_DEFINE_SCALAR_TO_METHOD(int64_t, Int)
    pub fn to_i64(&self) -> i64 {
        self.to_int()
    }

    // ET_DEFINE_SCALAR_TO_METHOD(bool, Bool)
    pub fn to_bool_val(&self) -> bool {
        self.to_bool()
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // also verifies is_floating_point (to_double checks it), is_boolean (to_bool
    // checks it), and the VT union storage (each constructor writes a variant
    // read back here).
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.scalar-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-double-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-bool-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-floating-point-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-boolean-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.v-t.v-t-fn/test]
    #[test]
    fn scalar_test_to_scalar_type() {
        let s_d = Scalar::from_double(3.141);
        assert_eq!(s_d.to_f64(), 3.141);
        assert!(s_d.is_floating_point());
        assert!(!s_d.is_boolean());
        let s_i = Scalar::from_i64(3);
        assert_eq!(s_i.to_i64(), 3);
        assert!(!s_i.is_floating_point());
        let s_b = Scalar::from_bool(true);
        assert_eq!(s_b.to_bool_val(), true);
        assert!(s_b.is_boolean());
        assert!(!s_b.is_floating_point());
    }

    // PORT-NOTE: `to_floating_point` (ET_DEFINE_SCALAR_TO_METHOD is wired to
    // `to_double`, so no public path reaches it). Focused unit test pinning it
    // directly: on a Double scalar it returns the stored double, matching
    // to_double per the sem rule.
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn/test]
    #[test]
    fn scalar_test_to_floating_point() {
        let s_d = Scalar::from_double(2.5);
        assert_eq!(s_d.to_floating_point(), 2.5);
        assert_eq!(s_d.to_floating_point(), s_d.to_double());

        let s_h = Scalar::from_half(f16::from_f32(1.5));
        assert_eq!(s_h.to_floating_point(), 1.5);
    }

    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-integral-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn/test]
    #[test]
    fn scalar_test_to_int_for_false_scalar_passes() {
        let s_b = Scalar::from_bool(false);
        assert!(!s_b.is_integral(/*include_bool=*/ false));
        assert!(s_b.is_integral(/*include_bool=*/ true));
        assert_eq!(s_b.to_i64(), 0);
    }

    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-integral-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn/test]
    #[test]
    fn scalar_test_to_int_for_true_scalar_passes() {
        let s_b = Scalar::from_bool(true);
        assert!(!s_b.is_integral(/*include_bool=*/ false));
        assert!(s_b.is_integral(/*include_bool=*/ true));
        assert_eq!(s_b.to_i64(), 1);
    }

    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.scalar-fn/test]
    // [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn/test]
    // PORT-NOTE: the C++ test builds from `int`, `int32_t`, and `int64_t` to
    // exercise the integral-template ctor for each width. The Rust port collapses
    // all integral widths onto `from_i64` (per the scalar-fn PORT-NOTE), so the
    // three constructions are the same `i64` entry point with widening at the
    // call site, matching `static_cast<int64_t>(val)`.
    #[test]
    fn scalar_test_int_constructor() {
        let int_val: i32 = 1;
        let s_int = Scalar::from_i64(int_val as i64);
        let int32_val: i32 = 1;
        let s_int32 = Scalar::from_i64(int32_val as i64);
        let int64_val: i64 = 1;
        let s_int64 = Scalar::from_i64(int64_val);
        assert_eq!(s_int.to_i64(), s_int32.to_i64());
        assert_eq!(s_int32.to_i64(), s_int64.to_i64());
    }
}
