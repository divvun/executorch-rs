//! Literal port of kernels/optimized/utils/math_utils.h.

use crate::runtime::core::portable_type::{BFloat16, Half};

/// Trait analogue of the C++ `ComputeDTypeTraits<scalar_t>::type` alias
/// (`compute_dtype<T>`): the type an op should perform internal math in for a
/// given element type. Default is the type itself; small int types widen to a
/// 32-bit int, and 16-bit floats widen to `f32`.
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits]
pub trait ComputeDTypeTraits {
    type Type;
}

macro_rules! compute_dtype_traits_default {
    ($($t:ty),*) => {
        $(
            impl ComputeDTypeTraits for $t {
                type Type = $t;
            }
        )*
    };
}

// The generic (identity) case in C++ is a template default; here it is spelled
// out for the concrete element types that actually flow through the optimized
// kernels, since Rust has no primary-template fallback.
compute_dtype_traits_default!(f32, f64, i32, i64, u32, u64);

// For 16 bit int types, ops should perform internal math in int32_t.
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-uint16-t]
impl ComputeDTypeTraits for u16 {
    type Type = u32;
}
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-int16-t]
impl ComputeDTypeTraits for i16 {
    type Type = i32;
}
// For 8 bit int types, ops should perform internal math in int32_t.
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-uint8-t]
impl ComputeDTypeTraits for u8 {
    type Type = u32;
}
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-int8-t]
impl ComputeDTypeTraits for i8 {
    type Type = i32;
}
// For 16 bit float types, ops should perform internal math in float32.
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-c10-b-float16]
impl ComputeDTypeTraits for BFloat16 {
    type Type = f32;
}
// [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-c10-half]
impl ComputeDTypeTraits for Half {
    type Type = f32;
}

/// `using compute_dtype<T> = typename ComputeDTypeTraits<T>::type;`
pub type ComputeDtype<T> = <T as ComputeDTypeTraits>::Type;

// [spec:et:def:math-utils.executorch.utils.divup-fn]
// [spec:et:sem:math-utils.executorch.utils.divup-fn]
#[inline]
pub fn divup(x: i64, y: i64) -> i64 {
    (x + y - 1) / y
}

// [spec:et:def:math-utils.executorch.utils.ceil-log2-fn]
// [spec:et:sem:math-utils.executorch.utils.ceil-log2-fn]
// PORT-NOTE: the C++ `CeilLog2` is templated on `T`, but its only instantiation
// (moments_utils.h) passes an `int64_t`. The template collapses to this single
// monomorphic `i64` form.
pub fn ceil_log2(x: i64) -> i64 {
    if x <= 2 {
        return 1;
    }
    // Last set bit is floor(log2(x)), floor + 1 is ceil
    // except when x is an exact powers of 2, so subtract 1 first
    // DEVIATION: llvmMathExtras `findLastSet(v)` for a nonzero unsigned value is
    // `63 - v.leading_zeros()` for u64; use Rust std bit intrinsics directly
    // rather than calling the (pending) llvmMathExtras port.
    (find_last_set_u64((x as u64) - 1) as i64) + 1
}

// DEVIATION: local literal stand-in for `executorch::llvm::findLastSet` over
// u64 with ZeroBehavior::ZB_Max, implemented via std `leading_zeros`. For a
// nonzero value this equals `floor(log2(v))`; for 0 it returns u64::MAX (the
// ZB_Max contract), matching llvmMathExtras.
#[inline]
fn find_last_set_u64(val: u64) -> u64 {
    if val == 0 {
        return u64::MAX;
    }
    (u64::BITS - 1 - val.leading_zeros()) as u64
}

#[cfg(test)]
mod tests {
    use super::{ceil_log2, divup};

    // [spec:et:sem:math-utils.executorch.utils.divup-fn/test]
    #[test]
    fn divup_rounds_up() {
        // Exact divisions stay exact.
        assert_eq!(divup(0, 5), 0);
        assert_eq!(divup(10, 5), 2);
        assert_eq!(divup(9, 3), 3);
        // Any remainder rounds up: (x + y - 1) / y.
        assert_eq!(divup(1, 5), 1);
        assert_eq!(divup(10, 3), 4);
        assert_eq!(divup(11, 3), 4);
        assert_eq!(divup(12, 3), 4);
        assert_eq!(divup(13, 3), 5);
        assert_eq!(divup(7, 8), 1);
        assert_eq!(divup(8, 7), 2);
        // Large values well below the x + y - 1 overflow boundary.
        assert_eq!(divup(i64::MAX - 8, 8), (i64::MAX - 8 + 7) / 8);
    }

    // [spec:et:sem:math-utils.executorch.utils.ceil-log2-fn/test]
    #[test]
    fn ceil_log2_matches_cpp() {
        // x <= 2 clamps to 1 (including the degenerate 0 and 1 inputs).
        assert_eq!(ceil_log2(0), 1);
        assert_eq!(ceil_log2(1), 1);
        assert_eq!(ceil_log2(2), 1);
        // Exact powers of two: the subtract-1 trick keeps them exact.
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(1024), 10);
        assert_eq!(ceil_log2(1 << 40), 40);
        // Non-powers round up: findLastSet(x - 1) + 1.
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(7), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(1025), 11);
        assert_eq!(ceil_log2((1i64 << 40) + 1), 41);
        // One below a power of two rounds up to that power's exponent.
        assert_eq!(ceil_log2(15), 4);
        assert_eq!(ceil_log2(1023), 10);
    }
}
