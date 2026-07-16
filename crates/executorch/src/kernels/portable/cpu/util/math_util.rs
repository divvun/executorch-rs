//! Literal port of kernels/portable/cpu/util/math_util.h.

use crate::runtime::core::portable_type::{BFloat16, Half};

// PORT-NOTE: the C++ header is a set of function templates selected by
// `std::enable_if` on `std::is_integral` / `std::is_floating_point` /
// `std::is_same<T, Half|BFloat16>`. Rust has no SFINAE-overloading over a single
// free-function name, so each C++ template name is ported as one generic free
// function `name<T: Trait>(...)` forwarding to a per-type trait impl. The trait
// impls carry the enable_if-selected bodies verbatim; the disjoint C++ overload
// sets become disjoint trait impls. The `at::vec` vectorized overloads
// (guarded by `ET_USE_PYTORCH_HEADERS`) are out of scope (scalar fallback only).

// ---------------------------------------------------------------------------
// floor_divide
// ---------------------------------------------------------------------------

pub trait FloorDivide {
    fn floor_divide(self, b: Self) -> Self;
}

// Integral overload:
//   template <typename INT_T, is_integral<INT_T>> INT_T floor_divide(INT_T a, INT_T b)
macro_rules! impl_floor_divide_int {
    ($t:ty) => {
        impl FloorDivide for $t {
            fn floor_divide(self, b: Self) -> Self {
                let a = self;
                let quot = a / b;
                // MSVC does not like signbit on integral types.
                if (a < 0) == (b < 0) {
                    return quot;
                }
                let rem = a % b;
                if rem != 0 { quot - 1 } else { quot }
            }
        }
    };
}
impl_floor_divide_int!(i8);
impl_floor_divide_int!(i16);
impl_floor_divide_int!(i32);
impl_floor_divide_int!(i64);
// PORT-NOTE: the C++ `is_integral<INT_T>` overload also selects unsigned
// integers; ET_SWITCH_REAL_TYPES instantiates floor_divide<uint8_t>. For u8 the
// `(a < 0) == (b < 0)` branch is always `true == true`, so the signed-remainder
// adjustment is never taken (correct for unsigned), mirroring the C++ template
// body verbatim.
#[allow(unused_comparisons)]
impl FloorDivide for u8 {
    fn floor_divide(self, b: Self) -> Self {
        let a = self;
        let quot = a / b;
        // MSVC does not like signbit on integral types.
        if (a < 0) == (b < 0) {
            return quot;
        }
        let rem = a % b;
        if rem != 0 { quot - 1 } else { quot }
    }
}

// Floating-point overload:
//   template <typename FLOAT_T, is_floating_point<FLOAT_T>> FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b)
// [spec:et:def:math-util.torch.executor.native.utils.floor-divide-fn]
// [spec:et:sem:math-util.torch.executor.native.utils.floor-divide-fn]
macro_rules! impl_floor_divide_float {
    ($t:ty) => {
        impl FloorDivide for $t {
            fn floor_divide(self, b: Self) -> Self {
                let a = self;
                if b == 0 as $t {
                    return if a.is_sign_negative() {
                        <$t>::NEG_INFINITY
                    } else {
                        <$t>::INFINITY
                    };
                }
                let mod_ = a % b;
                let div = (a - mod_) / b;
                if (mod_ != 0 as $t) && (b.is_sign_negative() != mod_.is_sign_negative()) {
                    return div - 1 as $t;
                }
                div
            }
        }
    };
}
impl_floor_divide_float!(f32);
impl_floor_divide_float!(f64);

pub fn floor_divide<T: FloorDivide>(a: T, b: T) -> T {
    a.floor_divide(b)
}

// ---------------------------------------------------------------------------
// isnan_override
// ---------------------------------------------------------------------------

pub trait IsnanOverride {
    fn isnan_override(self) -> bool;
}

// [spec:et:def:math-util.torch.executor.native.utils.isnan-override-fn]
// [spec:et:sem:math-util.torch.executor.native.utils.isnan-override-fn]
macro_rules! impl_isnan_int {
    ($t:ty) => {
        impl IsnanOverride for $t {
            fn isnan_override(self) -> bool {
                false
            }
        }
    };
}
macro_rules! impl_isnan_float {
    ($t:ty) => {
        impl IsnanOverride for $t {
            fn isnan_override(self) -> bool {
                self.is_nan()
            }
        }
    };
}
impl_isnan_int!(u8);
impl_isnan_int!(i8);
impl_isnan_int!(i16);
impl_isnan_int!(i32);
impl_isnan_int!(i64);
impl_isnan_int!(bool);
impl_isnan_float!(f32);
impl_isnan_float!(f64);
impl IsnanOverride for Half {
    fn isnan_override(self) -> bool {
        self.is_nan()
    }
}
impl IsnanOverride for BFloat16 {
    fn isnan_override(self) -> bool {
        self.is_nan()
    }
}

pub fn isnan_override<T: IsnanOverride>(a: T) -> bool {
    a.isnan_override()
}

// ---------------------------------------------------------------------------
// min_override / max_override
// ---------------------------------------------------------------------------

pub trait MinMaxOverride {
    fn min_override(self, b: Self) -> Self;
    fn max_override(self, b: Self) -> Self;
}

// Floating-point overloads:
//   FLOAT_T min_override/max_override(FLOAT_T a, FLOAT_T b)
macro_rules! impl_minmax_float {
    ($t:ty) => {
        impl MinMaxOverride for $t {
            fn min_override(self, b: Self) -> Self {
                let a = self;
                if a.is_nan() {
                    a
                } else if b.is_nan() {
                    b
                } else {
                    a.min(b)
                }
            }
            fn max_override(self, b: Self) -> Self {
                let a = self;
                if a.is_nan() {
                    a
                } else if b.is_nan() {
                    b
                } else {
                    a.max(b)
                }
            }
        }
    };
}
impl_minmax_float!(f32);
impl_minmax_float!(f64);

// Integral overloads:
//   INT_T min_override/max_override(INT_T a, INT_T b)
macro_rules! impl_minmax_int {
    ($t:ty) => {
        impl MinMaxOverride for $t {
            fn min_override(self, b: Self) -> Self {
                core::cmp::min(self, b)
            }
            fn max_override(self, b: Self) -> Self {
                core::cmp::max(self, b)
            }
        }
    };
}
impl_minmax_int!(u8);
impl_minmax_int!(i8);
impl_minmax_int!(i16);
impl_minmax_int!(i32);
impl_minmax_int!(i64);
// PORT-NOTE: bool reaches `max_override`/`min_override` via `ET_SWITCH_REALB_TYPES`
// (e.g. op_maximum with both inputs Bool); C++ routes it through the integral
// `std::max`/`std::min<bool>` overload, mirrored here (bool is `Ord`).
impl_minmax_int!(bool);

// Half / BFloat16 overloads:
//   T min_override/max_override(T a, T b) for T in {Half, BFloat16}
// [spec:et:def:math-util.torch.executor.native.utils.min-override-fn]
// [spec:et:sem:math-util.torch.executor.native.utils.min-override-fn]
// [spec:et:def:math-util.torch.executor.native.utils.max-override-fn]
// [spec:et:sem:math-util.torch.executor.native.utils.max-override-fn]
macro_rules! impl_minmax_reduced_float {
    ($t:ty) => {
        impl MinMaxOverride for $t {
            fn min_override(self, b: Self) -> Self {
                let a = self;
                let float_a = a.to_f32();
                if float_a.is_nan() {
                    return a;
                }
                let float_b = b.to_f32();
                if float_b.is_nan() {
                    return b;
                }
                if float_a < float_b { a } else { b }
            }
            fn max_override(self, b: Self) -> Self {
                let a = self;
                let float_a = a.to_f32();
                if float_a.is_nan() {
                    return a;
                }
                let float_b = b.to_f32();
                if float_b.is_nan() {
                    return b;
                }
                if float_a > float_b { a } else { b }
            }
        }
    };
}
impl_minmax_reduced_float!(Half);
impl_minmax_reduced_float!(BFloat16);

pub fn min_override<T: MinMaxOverride>(a: T, b: T) -> T {
    a.min_override(b)
}

pub fn max_override<T: MinMaxOverride>(a: T, b: T) -> T {
    a.max_override(b)
}

// ---------------------------------------------------------------------------
// remainder_override
// ---------------------------------------------------------------------------

pub trait RemainderOverride {
    fn remainder_override(self, b: Self) -> Self;
}

// Floating-point overload:
//   CTYPE remainder_override(CTYPE a, CTYPE b)
// [spec:et:def:math-util.torch.executor.native.utils.remainder-override-fn]
// [spec:et:sem:math-util.torch.executor.native.utils.remainder-override-fn]
macro_rules! impl_remainder_float {
    ($t:ty) => {
        impl RemainderOverride for $t {
            fn remainder_override(self, b: Self) -> Self {
                let a = self;
                // PORT-NOTE: the C++ declares the intermediate `rem` as `float`
                // regardless of CTYPE, so for `double` inputs there is a
                // narrowing to `float` before the final conversion back to
                // CTYPE. Reproduced bit-for-bit: compute `fmod` in f64, narrow
                // to f32, then widen back to the target type.
                let mut rem: f32 = ((a as f64) % (b as f64)) as f32;
                if (((a < 0 as $t) as i32) ^ ((b < 0 as $t) as i32)) != 0 && rem != 0.0f32 {
                    rem += b as f32;
                }
                rem as $t
            }
        }
    };
}
impl_remainder_float!(f32);
impl_remainder_float!(f64);

// Integral overload:
//   CTYPE remainder_override(CTYPE a, CTYPE b) { return a % b; }
macro_rules! impl_remainder_int {
    ($t:ty) => {
        impl RemainderOverride for $t {
            fn remainder_override(self, b: Self) -> Self {
                self % b
            }
        }
    };
}
impl_remainder_int!(u8);
impl_remainder_int!(i8);
impl_remainder_int!(i16);
impl_remainder_int!(i32);
impl_remainder_int!(i64);

pub fn remainder_override<T: RemainderOverride>(a: T, b: T) -> T {
    a.remainder_override(b)
}
