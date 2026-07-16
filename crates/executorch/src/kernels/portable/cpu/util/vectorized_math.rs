//! Literal port of kernels/portable/cpu/util/vectorized_math.h.

// PORT-NOTE: This header has two layers. The `ET_USE_PYTORCH_HEADERS` layer
// provides SIMD (`at::vec::Vectorized`) variants of a large set of float ops,
// built on `convert_to_vectorized_n_of_float`
// ([spec:et:sem:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn]).
// SIMD is out of scope for this port, so that layer — including
// `convert_to_vectorized_n_of_float`, `ET_INTERNAL_VECTORIZED_FLOAT_UNARY_FUNC`,
// `ET_INTERNAL_VECTORIZED_FLOAT_BINARY_FUNC`, and the vectorized `rsqrt` overload
// — is not translated. Only the scalar fallback path is ported.
//
// The `_ET_INTERNAL_STD_MATH_FUNC(name)` macro, in the non-PyTorch build, simply
// does `using std::name;` inside `executorch::math`, i.e. makes the C++ standard
// library scalar function available under that namespace. The full set (abs,
// acos, asin, atan, ceil, cos, cosh, erf, erfc, exp, expm1, floor, log, log10,
// log1p, log2, sin, sinh, sqrt, round, tan, tanh, trunc, lgamma, atan2, fmod,
// pow) is provided below as scalar free functions over a `Float` trait,
// mirroring the `executorch::math::name` scalar entry points.

// ---------------------------------------------------------------------------
// scalar Float trait: the scalar fallback set of std math functions
// ---------------------------------------------------------------------------

// PORT-NOTE: In C++ these are `using std::name;` re-exports resolved by overload
// for `float`/`double` (and reduced-precision via implicit widen). Here they are
// trait methods implemented for `f32`/`f64`, forwarding to the corresponding
// Rust std float method (which is the platform libm, matching `std::name`).
// [spec:et:def:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn]
// [spec:et:sem:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn]
pub trait Float: Copy {
    fn abs(self) -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn atan(self) -> Self;
    fn ceil(self) -> Self;
    fn cos(self) -> Self;
    fn cosh(self) -> Self;
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn exp(self) -> Self;
    fn expm1(self) -> Self;
    fn floor(self) -> Self;
    fn log(self) -> Self;
    fn log10(self) -> Self;
    fn log1p(self) -> Self;
    fn log2(self) -> Self;
    fn sin(self) -> Self;
    fn sinh(self) -> Self;
    fn sqrt(self) -> Self;
    fn round(self) -> Self;
    fn tan(self) -> Self;
    fn tanh(self) -> Self;
    fn trunc(self) -> Self;
    fn lgamma(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn fmod(self, other: Self) -> Self;
    fn pow(self, other: Self) -> Self;
    fn one() -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn acos(self) -> Self {
                <$t>::acos(self)
            }
            fn asin(self) -> Self {
                <$t>::asin(self)
            }
            fn atan(self) -> Self {
                <$t>::atan(self)
            }
            fn ceil(self) -> Self {
                <$t>::ceil(self)
            }
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }
            fn erf(self) -> Self {
                // PORT-NOTE: `erf`/`erfc` are not in Rust std; they map to C
                // `erf`/`erfc` from libm via libc, matching `std::erf`.
                unsafe { erf_libc(self) }
            }
            fn erfc(self) -> Self {
                unsafe { erfc_libc(self) }
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn expm1(self) -> Self {
                <$t>::exp_m1(self)
            }
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            fn log(self) -> Self {
                <$t>::ln(self)
            }
            fn log10(self) -> Self {
                <$t>::log10(self)
            }
            fn log1p(self) -> Self {
                <$t>::ln_1p(self)
            }
            fn log2(self) -> Self {
                <$t>::log2(self)
            }
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn round(self) -> Self {
                <$t>::round(self)
            }
            fn tan(self) -> Self {
                <$t>::tan(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            fn trunc(self) -> Self {
                <$t>::trunc(self)
            }
            fn lgamma(self) -> Self {
                unsafe { lgamma_libc(self) }
            }
            fn atan2(self, other: Self) -> Self {
                <$t>::atan2(self, other)
            }
            fn fmod(self, other: Self) -> Self {
                self % other
            }
            fn pow(self, other: Self) -> Self {
                <$t>::powf(self, other)
            }
            fn one() -> Self {
                1 as $t
            }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

// PORT-NOTE: `std::erf`, `std::erfc`, `std::lgamma` have no Rust std analog;
// bind to the C library entry points via `libc`. `lgammaf`/`lgamma` write the
// sign into a global (`signgam`) which is unused here, matching the plain
// `std::lgamma(x)` value-only usage.
mod libm_bindings {
    unsafe extern "C" {
        pub fn erff(x: f32) -> f32;
        pub fn erf(x: f64) -> f64;
        pub fn erfcf(x: f32) -> f32;
        pub fn erfc(x: f64) -> f64;
        pub fn lgammaf(x: f32) -> f32;
        pub fn lgamma(x: f64) -> f64;
    }
}
trait Libm {
    unsafe fn erf_c(self) -> Self;
    unsafe fn erfc_c(self) -> Self;
    unsafe fn lgamma_c(self) -> Self;
}
impl Libm for f32 {
    unsafe fn erf_c(self) -> Self {
        unsafe { libm_bindings::erff(self) }
    }
    unsafe fn erfc_c(self) -> Self {
        unsafe { libm_bindings::erfcf(self) }
    }
    unsafe fn lgamma_c(self) -> Self {
        unsafe { libm_bindings::lgammaf(self) }
    }
}
impl Libm for f64 {
    unsafe fn erf_c(self) -> Self {
        unsafe { libm_bindings::erf(self) }
    }
    unsafe fn erfc_c(self) -> Self {
        unsafe { libm_bindings::erfc(self) }
    }
    unsafe fn lgamma_c(self) -> Self {
        unsafe { libm_bindings::lgamma(self) }
    }
}
unsafe fn erf_libc<T: Libm>(x: T) -> T {
    unsafe { x.erf_c() }
}
unsafe fn erfc_libc<T: Libm>(x: T) -> T {
    unsafe { x.erfc_c() }
}
unsafe fn lgamma_libc<T: Libm>(x: T) -> T {
    unsafe { x.lgamma_c() }
}

pub fn abs<T: Float>(x: T) -> T {
    x.abs()
}
pub fn acos<T: Float>(x: T) -> T {
    x.acos()
}
pub fn asin<T: Float>(x: T) -> T {
    x.asin()
}
pub fn atan<T: Float>(x: T) -> T {
    x.atan()
}
pub fn ceil<T: Float>(x: T) -> T {
    x.ceil()
}
pub fn cos<T: Float>(x: T) -> T {
    x.cos()
}
pub fn cosh<T: Float>(x: T) -> T {
    x.cosh()
}
pub fn erf<T: Float>(x: T) -> T {
    x.erf()
}
pub fn erfc<T: Float>(x: T) -> T {
    x.erfc()
}
pub fn exp<T: Float>(x: T) -> T {
    x.exp()
}
pub fn expm1<T: Float>(x: T) -> T {
    x.expm1()
}
pub fn floor<T: Float>(x: T) -> T {
    x.floor()
}
pub fn log<T: Float>(x: T) -> T {
    x.log()
}
pub fn log10<T: Float>(x: T) -> T {
    x.log10()
}
pub fn log1p<T: Float>(x: T) -> T {
    x.log1p()
}
pub fn log2<T: Float>(x: T) -> T {
    x.log2()
}
pub fn sin<T: Float>(x: T) -> T {
    x.sin()
}
pub fn sinh<T: Float>(x: T) -> T {
    x.sinh()
}
pub fn sqrt<T: Float>(x: T) -> T {
    x.sqrt()
}
pub fn round<T: Float>(x: T) -> T {
    x.round()
}
pub fn tan<T: Float>(x: T) -> T {
    x.tan()
}
pub fn tanh<T: Float>(x: T) -> T {
    x.tanh()
}
pub fn trunc<T: Float>(x: T) -> T {
    x.trunc()
}
pub fn lgamma<T: Float>(x: T) -> T {
    x.lgamma()
}
pub fn atan2<T: Float>(a: T, b: T) -> T {
    a.atan2(b)
}
pub fn fmod<T: Float>(a: T, b: T) -> T {
    a.fmod(b)
}
pub fn pow<T: Float>(a: T, b: T) -> T {
    a.pow(b)
}

// ---------------------------------------------------------------------------
// rsqrt (scalar)
// ---------------------------------------------------------------------------

// template <typename T, std::enable_if_t<std::is_floating_point_v<T>>>
// T rsqrt(T x) { return T(1) / std::sqrt(x); }
// [spec:et:def:vectorized-math.executorch.math.rsqrt-fn]
// [spec:et:sem:vectorized-math.executorch.math.rsqrt-fn]
pub fn rsqrt<T: Float + core::ops::Div<Output = T>>(x: T) -> T {
    T::one() / x.sqrt()
}

#[cfg(test)]
mod tests {
    // PORT-NOTE: kernels/portable/cpu/util/test/vectorized_math_test.cpp is
    // guarded by `#error "This test requires ET_USE_PYTORCH_HEADERS!"` and every
    // one of its cases (BasicUnary, UnaryInt16/32/64ToFloat, BasicBinary,
    // BinaryInt16/32/64ToFloat) exercises the SIMD `at::vec::Vectorized<T>` layer:
    // it builds `Vectorized<T>::arange(...)` vectors and calls
    // `executorch::math::exp`/`pow` on those *vector* values, then compares each
    // lane. As documented at the top of this module, the `ET_USE_PYTORCH_HEADERS`
    // SIMD layer (`at::vec::Vectorized`, `convert_to_vectorized_n_of_float`, the
    // `ET_INTERNAL_VECTORIZED_FLOAT_{UNARY,BINARY}_FUNC` families) is deliberately
    // NOT translated in this port; only the scalar `executorch::math::*` fallback
    // exists here. There is therefore no Rust surface for these tests to bind to,
    // so the suite is not portable as written. It is recorded here rather than
    // ported; if/when the SIMD layer is translated, these cases should be added.
    //
    // The scalar `rsqrt` entry point (the only non-SIMD symbol) has no ported
    // caller, so it is pinned directly against its C++ sem rule `T(1)/std::sqrt(x)`.
    use super::rsqrt;

    // [spec:et:sem:vectorized-math.executorch.math.rsqrt-fn/test]
    #[test]
    fn rsqrt_scalar_matches_one_over_sqrt() {
        assert_eq!(rsqrt::<f64>(4.0), 0.5);
        assert_eq!(rsqrt::<f64>(0.25), 2.0);
        assert_eq!(rsqrt::<f64>(1.0), 1.0);
        assert_eq!(rsqrt::<f32>(4.0), 0.5);
        assert_eq!(rsqrt::<f32>(0.25), 2.0);
        // Literal `T(1) / std::sqrt(x)` (not a fused reciprocal-sqrt intrinsic).
        let x: f64 = 2.0;
        assert_eq!(rsqrt::<f64>(x), 1.0 / x.sqrt());
    }

    fn close64(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
    }

    fn close32(a: f32, b: f32) {
        assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
    }

    // Exercises the ported stand-in for the vectorized float-conversion layer:
    // in this scalar-only port the `convert_to_vectorized_n_of_float` sem
    // facet is anchored on the `Float` trait, which carries every
    // `executorch::math::*` scalar entry point. This pins the non-obvious
    // mappings (log -> ln, fmod -> `%` with C truncated-remainder semantics,
    // erf/erfc/lgamma -> libc bindings) for both f32 and f64.
    // [spec:et:sem:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn/test]
    #[test]
    fn scalar_math_entry_points_match_std() {
        use core::f64::consts::{E, FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, PI};

        close64(super::exp(1.0f64), E);
        close64(super::log(E), 1.0);
        close64(super::log2(8.0f64), 3.0);
        close64(super::log10(1000.0f64), 3.0);
        assert_eq!(super::log1p(0.0f64), 0.0);
        assert_eq!(super::expm1(0.0f64), 0.0);
        close64(super::expm1(core::f64::consts::LN_2), 1.0);

        assert_eq!(super::abs(-3.5f64), 3.5);
        assert_eq!(super::sqrt(2.25f64), 1.5);
        assert_eq!(super::floor(2.7f64), 2.0);
        assert_eq!(super::ceil(2.3f64), 3.0);
        // std::round rounds halfway cases away from zero.
        assert_eq!(super::round(2.5f64), 3.0);
        assert_eq!(super::round(-2.5f64), -3.0);
        assert_eq!(super::trunc(-2.7f64), -2.0);

        close64(super::sin(FRAC_PI_2), 1.0);
        assert_eq!(super::cos(0.0f64), 1.0);
        close64(super::tan(FRAC_PI_4), 1.0);
        close64(super::asin(0.5f64), FRAC_PI_6);
        close64(super::acos(0.5f64), PI / 3.0);
        close64(super::atan(1.0f64), FRAC_PI_4);
        close64(super::atan2(1.0f64, 1.0), FRAC_PI_4);
        close64(super::atan2(-1.0f64, -1.0), -3.0 * FRAC_PI_4);

        assert_eq!(super::sinh(0.0f64), 0.0);
        assert_eq!(super::cosh(0.0f64), 1.0);
        assert_eq!(super::tanh(0.0f64), 0.0);
        close64(super::sinh(1.0f64), (E - 1.0 / E) / 2.0);
        close64(super::cosh(1.0f64), (E + 1.0 / E) / 2.0);

        // libc-bound special functions.
        assert_eq!(super::erf(0.0f64), 0.0);
        assert_eq!(super::erfc(0.0f64), 1.0);
        close64(super::erf(1.0f64), 0.8427007929497149);
        close64(super::erf(1.0f64) + super::erfc(1.0), 1.0);
        assert_eq!(super::lgamma(1.0f64), 0.0);
        assert_eq!(super::lgamma(2.0f64), 0.0);
        close64(super::lgamma(5.0f64), 24.0f64.ln());

        // C fmod: truncated remainder, sign follows the dividend.
        assert_eq!(super::fmod(7.5f64, 2.0), 1.5);
        assert_eq!(super::fmod(-7.5f64, 2.0), -1.5);
        assert_eq!(super::pow(2.0f64, 10.0), 1024.0);

        // f32 instantiations of the same trait surface.
        close32(super::exp(1.0f32), core::f32::consts::E);
        close32(super::log(core::f32::consts::E), 1.0);
        close32(super::erf(1.0f32), 0.842_700_8);
        assert_eq!(super::erfc(0.0f32), 1.0);
        close32(super::lgamma(5.0f32), 24.0f32.ln());
        assert_eq!(super::fmod(-7.5f32, 2.0), -1.5);
        assert_eq!(super::pow(2.0f32, 10.0), 1024.0);
        assert_eq!(super::round(-2.5f32), -3.0);
    }
}
