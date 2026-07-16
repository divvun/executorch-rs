//! Literal port of kernels/optimized/utils/llvmMathExtras.h.
//!
//! Port of a subset of LLVM's `llvm/Support/MathExtras.h`. The C++ leans on
//! `__builtin_ctz`/`__builtin_clz`/`__builtin_popcount` (with portable bisection
//! fallbacks) and the MSVC `_BitScan*` intrinsics; the Rust port substitutes the
//! equivalent `std` bit methods (`trailing_zeros`, `leading_zeros`,
//! `count_ones`, `reverse_bits`, `to_bits`/`from_bits`). See the
//! `// DEVIATION:` notes at each substitution.

#![allow(non_snake_case, non_upper_case_globals)]

/// The behavior an operation has on an input of 0.
// [spec:et:def:llvm-math-extras.executorch.llvm.zero-behavior]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZeroBehavior {
    /// The returned value is undefined.
    ZbUndefined,
    /// The returned value is numeric_limits<T>::max()
    ZbMax,
    /// The returned value is numeric_limits<T>::digits
    ZbWidth,
}

pub use ZeroBehavior::{ZbMax, ZbUndefined, ZbWidth};

// The MSVC `_BitScan*` intrinsics are declared (not defined) in the C++ header
// under `#ifdef _MSC_VER` and are only used by the `_MSC_VER` branches of the
// counter specializations below. On the Rust target the counters use the std
// bit methods directly, so these have no bodies to port; their sem rules record
// the intrinsic contract.
// [spec:et:def:llvm-math-extras.bit-scan-forward-fn]
// [spec:et:sem:llvm-math-extras.bit-scan-forward-fn]
// [spec:et:def:llvm-math-extras.bit-scan-forward64-fn]
// [spec:et:sem:llvm-math-extras.bit-scan-forward64-fn]
// [spec:et:def:llvm-math-extras.bit-scan-reverse-fn]
// [spec:et:sem:llvm-math-extras.bit-scan-reverse-fn]
// [spec:et:def:llvm-math-extras.bit-scan-reverse64-fn]
// [spec:et:sem:llvm-math-extras.bit-scan-reverse64-fn]

/// Unsigned integral element type accepted by the counting helpers. Mirrors the
/// C++ `static_assert(is_integer && !is_signed)` constraint and the
/// `numeric_limits<T>::digits` / `sizeof(T)`-based dispatch.
pub trait UnsignedInt: Copy + Eq {
    /// numeric_limits<T>::digits (bit width).
    const DIGITS: usize;
    /// sizeof(T).
    const SIZE: usize;
    const ZERO: Self;
    /// numeric_limits<T>::max().
    const MAX: Self;
    fn ctz(self) -> u32;
    fn clz(self) -> u32;
    fn popcount(self) -> u32;
    fn is_zero(self) -> bool;
    fn not(self) -> Self;
    fn to_u64(self) -> u64;
}

macro_rules! impl_unsigned_int {
    ($($t:ty),*) => {
        $(
            impl UnsignedInt for $t {
                const DIGITS: usize = <$t>::BITS as usize;
                const SIZE: usize = core::mem::size_of::<$t>();
                const ZERO: Self = 0;
                const MAX: Self = <$t>::MAX;
                fn ctz(self) -> u32 { self.trailing_zeros() }
                fn clz(self) -> u32 { self.leading_zeros() }
                fn popcount(self) -> u32 { self.count_ones() }
                fn is_zero(self) -> bool { self == 0 }
                fn not(self) -> Self { !self }
                fn to_u64(self) -> u64 { self as u64 }
            }
        )*
    };
}

impl_unsigned_int!(u8, u16, u32, u64, usize);

pub mod detail {
    use super::{UnsignedInt, ZbUndefined, ZeroBehavior};

    // The C++ has a generic bisection `TrailingZerosCounter` plus 32-bit and
    // 64-bit specializations that call `__builtin_ctz`/`__builtin_ctzll`. Since
    // Rust's `trailing_zeros()` already lowers to those builtins for every width,
    // one generic implementation over `UnsignedInt` covers all three.
    // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter]
    pub struct TrailingZerosCounter;

    impl TrailingZerosCounter {
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter.count-fn]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4.count-fn]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8.count-fn]
        pub fn count<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> usize {
            // DEVIATION: C++ uses a bisection loop (generic) or __builtin_ctz
            // (4/8-byte specializations); Rust `trailing_zeros()` is the builtin
            // for every width. On Val == 0 the builtins are UB while `ctz`
            // returns DIGITS, so we replicate the specializations' explicit
            // "return width when ZB != ZB_Undefined && Val == 0" guard.
            if zb != ZbUndefined && val.is_zero() {
                return T::DIGITS;
            }
            val.ctz() as usize
        }
    }

    // Same collapse as above for leading zeros: generic bisection + 32/64-bit
    // `__builtin_clz` specializations all map onto `leading_zeros()`.
    // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter]
    pub struct LeadingZerosCounter;

    impl LeadingZerosCounter {
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter.count-fn]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4.count-fn]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8.count-fn]
        pub fn count<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> usize {
            // DEVIATION: bisection / __builtin_clz collapse onto
            // `leading_zeros()`; guard Val == 0 exactly as the C++
            // specializations do (return width unless ZB_Undefined).
            if zb != ZbUndefined && val.is_zero() {
                return T::DIGITS;
            }
            val.clz() as usize
        }
    }

    // Generic (<=32-bit forward) + 64-bit `__builtin_popcount(ll)`
    // specializations collapse onto `count_ones()`.
    // [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter]
    pub struct PopulationCounter;

    impl PopulationCounter {
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter.count-fn]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter-t-8]
        // [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter-t-8.count-fn]
        // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter-t-8.count-fn]
        pub fn count<T: UnsignedInt>(value: T) -> u32 {
            // DEVIATION: __builtin_popcount(ll) / SWAR fallback -> `count_ones()`.
            value.popcount()
        }
    }
}

/// Count number of 0's from the least significant bit to the most, stopping at
/// the first 1. Only unsigned integral types are allowed.
// [spec:et:def:llvm-math-extras.executorch.llvm.count-trailing-zeros-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-zeros-fn]
pub fn count_trailing_zeros<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> usize {
    detail::TrailingZerosCounter::count(val, zb)
}

/// Count number of 0's from the most significant bit to the least, stopping at
/// the first 1. Only unsigned integral types are allowed.
// [spec:et:def:llvm-math-extras.executorch.llvm.count-leading-zeros-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-zeros-fn]
pub fn count_leading_zeros<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> usize {
    detail::LeadingZerosCounter::count(val, zb)
}

/// Get the index of the first set bit starting from the least significant bit.
// [spec:et:def:llvm-math-extras.executorch.llvm.find-first-set-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.find-first-set-fn]
pub fn find_first_set<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> T {
    if zb == ZbMax && val.is_zero() {
        return T::MAX;
    }
    let idx = count_trailing_zeros(val, ZbUndefined);
    // C++ implicitly narrows the size_t result to T.
    truncate_to::<T>(idx)
}

/// Get the index of the last set bit starting from the least significant bit.
// [spec:et:def:llvm-math-extras.executorch.llvm.find-last-set-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.find-last-set-fn]
pub fn find_last_set<T: UnsignedInt>(val: T, zb: ZeroBehavior) -> T {
    if zb == ZbMax && val.is_zero() {
        return T::MAX;
    }
    // Use ^ instead of - to mirror the C++ (both gcc and llvm fold it into clz).
    let r = count_leading_zeros(val, ZbUndefined) ^ (T::DIGITS - 1);
    truncate_to::<T>(r)
}

/// Narrow a `usize` bit index/count to the unsigned element type `T`, matching
/// the C++ implicit `size_t`→`T` conversion at the `findFirstSet`/`findLastSet`
/// return sites.
fn truncate_to<T: UnsignedInt>(v: usize) -> T {
    // The valid results (bit indices, all-ones sentinels) always fit; mask to
    // T's width to reproduce C++ narrowing conversion semantics.
    let masked = if T::DIGITS >= usize::BITS as usize {
        v as u64
    } else {
        (v as u64) & ((1u64 << T::DIGITS) - 1)
    };
    // Reconstruct T from its u64 image via the primitive casts each impl uses.
    from_u64::<T>(masked)
}

/// Inverse of `UnsignedInt::to_u64` restricted to the low `DIGITS` bits.
fn from_u64<T: UnsignedInt>(v: u64) -> T {
    // Only the small set of concrete unsigned types implement `UnsignedInt`.
    // Dispatch by width; the values passed here always fit.
    match T::SIZE {
        1 => u8_to::<T>(v as u8),
        2 => u16_to::<T>(v as u16),
        4 => u32_to::<T>(v as u32),
        _ => u64_to::<T>(v),
    }
}

// These helper conversions launder a concrete primitive back into the generic
// `T`. They are safe because the caller has already matched `T::SIZE`.
fn u8_to<T: UnsignedInt>(v: u8) -> T {
    unsafe { core::mem::transmute_copy(&v) }
}
fn u16_to<T: UnsignedInt>(v: u16) -> T {
    unsafe { core::mem::transmute_copy(&v) }
}
fn u32_to<T: UnsignedInt>(v: u32) -> T {
    unsafe { core::mem::transmute_copy(&v) }
}
fn u64_to<T: UnsignedInt>(v: u64) -> T {
    unsafe { core::mem::transmute_copy(&v) }
}

/// Create a bitmask with the N right-most bits set to 1, and all other bits 0.
// [spec:et:def:llvm-math-extras.executorch.llvm.mask-trailing-ones-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-ones-fn]
pub fn mask_trailing_ones<T: UnsignedInt>(n: u32) -> T {
    let bits = (8 * T::SIZE) as u32;
    debug_assert!(n <= bits, "Invalid bit index");
    if n == 0 {
        T::ZERO
    } else {
        // T(-1) >> (Bits - N)
        from_u64::<T>((T::MAX.to_u64() & width_mask::<T>()) >> (bits - n))
    }
}

/// Create a bitmask with the N left-most bits set to 1, and all other bits 0.
// [spec:et:def:llvm-math-extras.executorch.llvm.mask-leading-ones-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-ones-fn]
pub fn mask_leading_ones<T: UnsignedInt>(n: u32) -> T {
    let inner: T = mask_trailing_ones::<T>((8 * T::SIZE) as u32 - n);
    // ~maskTrailingOnes<T>(...)
    from_u64::<T>((!inner.to_u64()) & width_mask::<T>())
}

/// Create a bitmask with the N right-most bits set to 0, and all other bits 1.
// [spec:et:def:llvm-math-extras.executorch.llvm.mask-trailing-zeros-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-zeros-fn]
pub fn mask_trailing_zeros<T: UnsignedInt>(n: u32) -> T {
    mask_leading_ones::<T>((8 * T::SIZE) as u32 - n)
}

/// Create a bitmask with the N left-most bits set to 0, and all other bits 1.
// [spec:et:def:llvm-math-extras.executorch.llvm.mask-leading-zeros-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-zeros-fn]
pub fn mask_leading_zeros<T: UnsignedInt>(n: u32) -> T {
    mask_trailing_ones::<T>((8 * T::SIZE) as u32 - n)
}

/// Mask of T's valid width within a u64 (all ones for 64-bit types).
fn width_mask<T: UnsignedInt>() -> u64 {
    if T::DIGITS >= 64 {
        u64::MAX
    } else {
        (1u64 << T::DIGITS) - 1
    }
}

/// Reverse the bits in Val.
// [spec:et:def:llvm-math-extras.executorch.llvm.reverse-bits-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.reverse-bits-fn]
pub fn reverse_bits<T: ReverseBits>(val: T) -> T {
    // DEVIATION: C++ uses a 256-entry byte-reversal table + memcpy byte swap;
    // Rust's `reverse_bits()` performs the identical full-width bit reversal.
    val.reverse_bits_impl()
}

/// Element types supporting `reverseBits`.
pub trait ReverseBits {
    fn reverse_bits_impl(self) -> Self;
}
macro_rules! impl_reverse_bits {
    ($($t:ty),*) => { $( impl ReverseBits for $t { fn reverse_bits_impl(self) -> Self { self.reverse_bits() } } )* };
}
impl_reverse_bits!(u8, u16, u32, u64, usize);

/// Return the high 32 bits of a 64 bit value.
// [spec:et:def:llvm-math-extras.executorch.llvm.hi-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.hi-32-fn]
pub const fn Hi_32(value: u64) -> u32 {
    (value >> 32) as u32
}

/// Return the low 32 bits of a 64 bit value.
// [spec:et:def:llvm-math-extras.executorch.llvm.lo-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.lo-32-fn]
pub const fn Lo_32(value: u64) -> u32 {
    value as u32
}

/// Make a 64-bit integer from a high / low pair of 32-bit integers.
// [spec:et:def:llvm-math-extras.executorch.llvm.make-64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.make-64-fn]
pub const fn Make_64(high: u32, low: u32) -> u64 {
    ((high as u64) << 32) | (low as u64)
}

/// Checks if an integer fits into the given bit width N.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-int-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-fn]
pub const fn is_int(n: u32, x: i64) -> bool {
    n >= 64 || (-(1i64 << (n - 1)) <= x && x < (1i64 << (n - 1)))
}

/// isInt<8> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-int-8-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-8-fn]
pub const fn is_int_8(x: i64) -> bool {
    (x as i8) as i64 == x
}

/// isInt<16> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-int-16-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-16-fn]
pub const fn is_int_16(x: i64) -> bool {
    (x as i16) as i64 == x
}

/// isInt<32> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-int-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-32-fn]
pub const fn is_int_32(x: i64) -> bool {
    (x as i32) as i64 == x
}

/// Checks if a signed integer is an N bit number shifted left by S.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-int-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-int-fn]
pub const fn is_shifted_int(n: u32, s: u32, x: i64) -> bool {
    // static_assert N > 0 && N + S <= 64 in C++ (template params); enforced by callers.
    is_int(n + s, x) && ((x as u64) % (1u64 << s) == 0)
}

/// Checks if an unsigned integer fits into the given bit width N.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-fn]
pub const fn is_u_int(n: u32, x: u64) -> bool {
    // C++ splits into N < 64 (X < (1 << N)) and N >= 64 (true) overloads.
    if n >= 64 { true } else { x < (1u64 << n) }
}

/// isUInt<8> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-8-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-8-fn]
pub const fn is_u_int_8(x: u64) -> bool {
    (x as u8) as u64 == x
}

/// isUInt<16> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-16-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-16-fn]
pub const fn is_u_int_16(x: u64) -> bool {
    (x as u16) as u64 == x
}

/// isUInt<32> specialization.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-32-fn]
pub const fn is_u_int_32(x: u64) -> bool {
    (x as u32) as u64 == x
}

/// Checks if a unsigned integer is an N bit number shifted left by S.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-u-int-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-u-int-fn]
pub const fn is_shifted_u_int(n: u32, s: u32, x: u64) -> bool {
    is_u_int(n + s, x) && (x % (1u64 << s) == 0)
}

/// Gets the maximum value for a N-bit unsigned integer.
// [spec:et:def:llvm-math-extras.executorch.llvm.max-u-int-n-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.max-u-int-n-fn]
pub fn max_u_int_n(n: u64) -> u64 {
    debug_assert!(n > 0 && n <= 64, "integer width out of range");
    // UINT64_MAX >> (64 - N)
    u64::MAX >> (64 - n)
}

/// Gets the minimum value for a N-bit signed integer.
// [spec:et:def:llvm-math-extras.executorch.llvm.min-int-n-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.min-int-n-fn]
pub fn min_int_n(n: i64) -> i64 {
    debug_assert!(n > 0 && n <= 64, "integer width out of range");
    // -(UINT64_C(1) << (N - 1)) computed in u64 then cast to i64.
    (1u64 << (n - 1)).wrapping_neg() as i64
}

/// Gets the maximum value for a N-bit signed integer.
// [spec:et:def:llvm-math-extras.executorch.llvm.max-int-n-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.max-int-n-fn]
pub fn max_int_n(n: i64) -> i64 {
    debug_assert!(n > 0 && n <= 64, "integer width out of range");
    // (UINT64_C(1) << (N - 1)) - 1, relying on two's-complement wrap at N == 64.
    ((1u64 << (n - 1)).wrapping_sub(1)) as i64
}

/// Checks if an unsigned integer fits into the given (dynamic) bit width.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-n-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-n-fn]
pub fn is_u_int_n(n: u32, x: u64) -> bool {
    n >= 64 || x <= max_u_int_n(n as u64)
}

/// Checks if an signed integer fits into the given (dynamic) bit width.
// [spec:et:def:llvm-math-extras.executorch.llvm.is-int-n-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-n-fn]
pub fn is_int_n(n: u32, x: i64) -> bool {
    n >= 64 || (min_int_n(n as i64) <= x && x <= max_int_n(n as i64))
}

/// Return true if the argument is a non-empty sequence of ones starting at the
/// least significant bit with the remainder zero (32 bit version).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-mask-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-32-fn]
pub const fn is_mask_32(value: u32) -> bool {
    value != 0 && (value.wrapping_add(1) & value) == 0
}

/// Return true if the argument is a non-empty sequence of ones starting at the
/// least significant bit with the remainder zero (64 bit version).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-mask-64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-64-fn]
pub const fn is_mask_64(value: u64) -> bool {
    value != 0 && (value.wrapping_add(1) & value) == 0
}

/// Return true if the argument contains a non-empty sequence of ones with the
/// remainder zero (32 bit version).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-mask-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-32-fn]
pub const fn is_shifted_mask_32(value: u32) -> bool {
    value != 0 && is_mask_32((value.wrapping_sub(1)) | value)
}

/// Return true if the argument contains a non-empty sequence of ones with the
/// remainder zero (64 bit version).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-mask-64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-64-fn]
pub const fn is_shifted_mask_64(value: u64) -> bool {
    value != 0 && is_mask_64((value.wrapping_sub(1)) | value)
}

/// Return true if the argument is a power of two > 0 (32 bit edition).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-power-of2-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-32-fn]
pub const fn is_power_of2_32(value: u32) -> bool {
    value != 0 && (value & (value.wrapping_sub(1))) == 0
}

/// Return true if the argument is a power of two > 0 (64 bit edition).
// [spec:et:def:llvm-math-extras.executorch.llvm.is-power-of2-64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-64-fn]
pub const fn is_power_of2_64(value: u64) -> bool {
    value != 0 && (value & (value.wrapping_sub(1))) == 0
}

/// Count the number of ones from the most significant bit to the first zero bit.
// [spec:et:def:llvm-math-extras.executorch.llvm.count-leading-ones-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-ones-fn]
pub fn count_leading_ones<T: UnsignedInt>(value: T, zb: ZeroBehavior) -> usize {
    count_leading_zeros(value.not(), zb)
}

/// Count the number of ones from the least significant bit to the first zero bit.
// [spec:et:def:llvm-math-extras.executorch.llvm.count-trailing-ones-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-ones-fn]
pub fn count_trailing_ones<T: UnsignedInt>(value: T, zb: ZeroBehavior) -> usize {
    count_trailing_zeros(value.not(), zb)
}

/// Count the number of set bits in a value. Returns 0 if the word is zero.
// [spec:et:def:llvm-math-extras.executorch.llvm.count-population-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.count-population-fn]
pub fn count_population<T: UnsignedInt>(value: T) -> u32 {
    detail::PopulationCounter::count(value)
}

/// Return the log base 2 of the specified value.
// [spec:et:def:llvm-math-extras.executorch.llvm.log2-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.log2-fn]
pub fn Log2(value: f64) -> f64 {
    // DEVIATION: C++ calls libm `log2` (or log/log(2) on old Android);
    // `f64::log2` is the same operation.
    value.log2()
}

/// Return the floor log base 2 of the specified value, -1 (0xFFFFFFFF) if zero
/// (32 bit edition).
// [spec:et:def:llvm-math-extras.executorch.llvm.log2-32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-fn]
pub fn Log2_32(value: u32) -> u32 {
    // 31 - countLeadingZeros(Value); for Value == 0, clz == 32 so this wraps to
    // 0xFFFFFFFF (== (unsigned)-1), matching C++.
    31u32.wrapping_sub(count_leading_zeros(value, ZbWidth) as u32)
}

/// Return the floor log base 2 of the specified value, -1 if zero (64 bit).
// [spec:et:def:llvm-math-extras.executorch.llvm.log2-64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-fn]
pub fn Log2_64(value: u64) -> u32 {
    63u32.wrapping_sub(count_leading_zeros(value, ZbWidth) as u32)
}

/// Return the ceil log base 2 of the specified value, 32 if zero (32 bit).
// [spec:et:def:llvm-math-extras.executorch.llvm.log2-32-ceil-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-ceil-fn]
pub fn Log2_32_Ceil(value: u32) -> u32 {
    32u32.wrapping_sub(count_leading_zeros(value.wrapping_sub(1), ZbWidth) as u32)
}

/// Return the ceil log base 2 of the specified value, 64 if zero (64 bit).
// [spec:et:def:llvm-math-extras.executorch.llvm.log2-64-ceil-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-ceil-fn]
pub fn Log2_64_Ceil(value: u64) -> u32 {
    64u32.wrapping_sub(count_leading_zeros(value.wrapping_sub(1), ZbWidth) as u32)
}

/// Return the greatest common divisor of the values using Euclid's algorithm.
// [spec:et:def:llvm-math-extras.executorch.llvm.greatest-common-divisor64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.greatest-common-divisor64-fn]
pub fn GreatestCommonDivisor64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Takes a 64-bit integer and returns the bit equivalent double.
// [spec:et:def:llvm-math-extras.executorch.llvm.bits-to-double-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-double-fn]
pub fn BitsToDouble(bits: u64) -> f64 {
    // DEVIATION: memcpy reinterpret -> `f64::from_bits`.
    f64::from_bits(bits)
}

/// Takes a 32-bit integer and returns the bit equivalent float.
// [spec:et:def:llvm-math-extras.executorch.llvm.bits-to-float-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-float-fn]
pub fn BitsToFloat(bits: u32) -> f32 {
    // DEVIATION: memcpy reinterpret -> `f32::from_bits`.
    f32::from_bits(bits)
}

/// Takes a double and returns the bit equivalent 64-bit integer.
// [spec:et:def:llvm-math-extras.executorch.llvm.double-to-bits-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.double-to-bits-fn]
pub fn DoubleToBits(double: f64) -> u64 {
    // DEVIATION: memcpy reinterpret -> `f64::to_bits`.
    double.to_bits()
}

/// Takes a float and returns the bit equivalent 32-bit integer.
// [spec:et:def:llvm-math-extras.executorch.llvm.float-to-bits-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.float-to-bits-fn]
pub fn FloatToBits(float: f32) -> u32 {
    // DEVIATION: memcpy reinterpret -> `f32::to_bits`.
    float.to_bits()
}

/// A and B are either alignments or offsets. Return the minimum alignment that
/// may be assumed after adding the two together.
// [spec:et:def:llvm-math-extras.executorch.llvm.min-align-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.min-align-fn]
pub const fn MinAlign(a: u64, b: u64) -> u64 {
    // (A | B) & (1 + ~(A | B))  -- isolates the lowest set bit.
    (a | b) & (1u64.wrapping_add(!(a | b)))
}

/// Aligns Addr to Alignment bytes, rounding up. Alignment must be a power of two.
// [spec:et:def:llvm-math-extras.executorch.llvm.align-addr-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.align-addr-fn]
pub fn alignAddr(addr: *const core::ffi::c_void, alignment: usize) -> usize {
    debug_assert!(
        alignment != 0 && is_power_of2_64(alignment as u64),
        "Alignment is not a power of two!"
    );
    let addr = addr as usize;
    debug_assert!(addr.wrapping_add(alignment - 1) >= addr);
    (addr.wrapping_add(alignment - 1)) & !(alignment - 1)
}

/// Returns the necessary adjustment for aligning Ptr to Alignment bytes.
// [spec:et:def:llvm-math-extras.executorch.llvm.alignment-adjustment-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.alignment-adjustment-fn]
pub fn alignmentAdjustment(ptr: *const core::ffi::c_void, alignment: usize) -> usize {
    alignAddr(ptr, alignment) - (ptr as usize)
}

/// Returns the next power of two (in 64-bits) strictly greater than A. Zero on
/// overflow.
// [spec:et:def:llvm-math-extras.executorch.llvm.next-power-of2-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.next-power-of2-fn]
pub fn NextPowerOf2(mut a: u64) -> u64 {
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a |= a >> 8;
    a |= a >> 16;
    a |= a >> 32;
    a.wrapping_add(1)
}

/// Returns the power of two which is less than or equal to the given value.
// [spec:et:def:llvm-math-extras.executorch.llvm.power-of2-floor-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-floor-fn]
pub fn PowerOf2Floor(a: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    1u64 << (63 - count_leading_zeros(a, ZbUndefined))
}

/// Returns the power of two which is greater than or equal to the given value.
// [spec:et:def:llvm-math-extras.executorch.llvm.power-of2-ceil-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-ceil-fn]
pub fn PowerOf2Ceil(a: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    NextPowerOf2(a - 1)
}

/// Returns the next integer (mod 2**64) that is >= Value and equals
/// Align * N + Skew for some integer N. Align must be non-zero.
// [spec:et:def:llvm-math-extras.executorch.llvm.align-to-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.align-to-fn]
pub fn alignTo(value: u64, align: u64, mut skew: u64) -> u64 {
    debug_assert!(align != 0, "Align can't be 0.");
    skew %= align;
    // (Value + Align - 1 - Skew) / Align * Align + Skew, wrapping to match the
    // documented alignTo(~0LL, 8) == 0 behavior.
    (value.wrapping_add(align).wrapping_sub(1).wrapping_sub(skew) / align)
        .wrapping_mul(align)
        .wrapping_add(skew)
}

/// `AlignTo<Align>` constant-expression helper.
// [spec:et:def:llvm-math-extras.executorch.llvm.align-to]
pub struct AlignTo<const ALIGN: u64>;

impl<const ALIGN: u64> AlignTo<ALIGN> {
    // [spec:et:def:llvm-math-extras.executorch.llvm.align-to.from-value]
    /// from_value<Value>::value = (Value + Align - 1) / Align * Align
    pub const fn from_value<const VALUE: u64>() -> u64 {
        // static_assert(Align != 0u) is a caller obligation.
        (VALUE + ALIGN - 1) / ALIGN * ALIGN
    }
}

/// Returns the integer ceil(Numerator / Denominator).
// [spec:et:def:llvm-math-extras.executorch.llvm.divide-ceil-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.divide-ceil-fn]
pub fn divideCeil(numerator: u64, denominator: u64) -> u64 {
    alignTo(numerator, denominator, 0) / denominator
}

/// Returns the largest uint64_t <= Value that is Skew mod Align. Align non-zero.
// [spec:et:def:llvm-math-extras.executorch.llvm.align-down-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.align-down-fn]
pub fn alignDown(value: u64, align: u64, mut skew: u64) -> u64 {
    debug_assert!(align != 0, "Align can't be 0.");
    skew %= align;
    (value.wrapping_sub(skew) / align)
        .wrapping_mul(align)
        .wrapping_add(skew)
}

/// Returns the offset to the next integer (mod 2**64) that is >= Value and a
/// multiple of Align.
// [spec:et:def:llvm-math-extras.executorch.llvm.offset-to-alignment-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.offset-to-alignment-fn]
pub fn OffsetToAlignment(value: u64, align: u64) -> u64 {
    alignTo(value, align, 0).wrapping_sub(value)
}

/// Sign-extend the number in the bottom B bits of X to a 32-bit integer.
/// Requires 0 < B <= 32.
// [spec:et:def:llvm-math-extras.executorch.llvm.sign-extend32-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend32-fn]
pub fn SignExtend32(x: u32, b: u32) -> i32 {
    debug_assert!(b > 0, "Bit width can't be 0.");
    debug_assert!(b <= 32, "Bit width out of range.");
    // int32_t(X << (32 - B)) >> (32 - B)
    ((x << (32 - b)) as i32) >> (32 - b)
}

/// Sign-extend the number in the bottom B bits of x to a 64-bit integer.
/// Requires 0 < B <= 64.
// [spec:et:def:llvm-math-extras.executorch.llvm.sign-extend64-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend64-fn]
pub fn SignExtend64(x: u64, b: u32) -> i64 {
    debug_assert!(b > 0, "Bit width can't be 0.");
    debug_assert!(b <= 64, "Bit width out of range.");
    ((x << (64 - b)) as i64) >> (64 - b)
}

/// Subtract two unsigned integers, X and Y, and return |X - Y|.
// [spec:et:def:llvm-math-extras.executorch.llvm.absolute-difference-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.absolute-difference-fn]
pub fn AbsoluteDifference<T: Ord + core::ops::Sub<Output = T> + Copy>(x: T, y: T) -> T {
    core::cmp::max(x, y) - core::cmp::min(x, y)
}

/// Add two unsigned integers, clamping to the maximum on overflow.
// [spec:et:def:llvm-math-extras.executorch.llvm.saturating-add-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-add-fn]
pub fn SaturatingAdd<T: SaturatingInt>(x: T, y: T, result_overflowed: Option<&mut bool>) -> T {
    let mut dummy = false;
    let overflowed: &mut bool = result_overflowed.unwrap_or(&mut dummy);
    // Hacker's Delight, p. 29
    let z = x.wrapping_add(y);
    *overflowed = z < x || z < y;
    if *overflowed { T::MAX_VALUE } else { z }
}

/// Multiply two unsigned integers, clamping to the maximum on overflow.
// [spec:et:def:llvm-math-extras.executorch.llvm.saturating-multiply-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-fn]
pub fn SaturatingMultiply<T: SaturatingInt>(x: T, y: T, result_overflowed: Option<&mut bool>) -> T {
    let mut dummy = false;
    let overflowed: &mut bool = result_overflowed.unwrap_or(&mut dummy);

    *overflowed = false;

    // Log2(Z) would be either Log2Z or Log2Z + 1. Special case: if X or Y is 0,
    // Log2_64 gives -1 (0xFFFFFFFF), and Log2Z will necessarily be less than
    // Log2Max as desired. Match the C++ `int` arithmetic (wrapping i32 sum).
    let log2z: i32 = (Log2_64(x.to_u64()) as i32).wrapping_add(Log2_64(y.to_u64()) as i32);
    let max = T::MAX_VALUE;
    let log2max: i32 = Log2_64(max.to_u64()) as i32;
    if log2z < log2max {
        return x.wrapping_mul(y);
    }
    if log2z > log2max {
        *overflowed = true;
        return max;
    }

    // We're going to use the top bit, and maybe overflow one bit past it.
    // Multiply all but the bottom bit then add that on at the end.
    let mut z = (x.shr1()).wrapping_mul(y);
    if !(z & max.shr1().not_bits()).is_zero() {
        *overflowed = true;
        return max;
    }
    z = z.shl1();
    if x.bit0() {
        return SaturatingAdd(z, y, Some(overflowed));
    }
    z
}

/// Multiply X and Y and add A, clamping to the maximum on overflow.
// [spec:et:def:llvm-math-extras.executorch.llvm.saturating-multiply-add-fn]
// [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-add-fn]
pub fn SaturatingMultiplyAdd<T: SaturatingInt>(
    x: T,
    y: T,
    a: T,
    result_overflowed: Option<&mut bool>,
) -> T {
    let mut dummy = false;
    let overflowed: &mut bool = result_overflowed.unwrap_or(&mut dummy);

    let product = SaturatingMultiply(x, y, Some(overflowed));
    if *overflowed {
        return product;
    }

    SaturatingAdd(a, product, Some(overflowed))
}

/// Unsigned integer element type supporting the `Saturating*` helpers.
pub trait SaturatingInt: Copy + Ord + core::ops::BitAnd<Output = Self> {
    const MAX_VALUE: Self;
    fn wrapping_add(self, o: Self) -> Self;
    fn wrapping_mul(self, o: Self) -> Self;
    fn shr1(self) -> Self;
    fn shl1(self) -> Self;
    fn not_bits(self) -> Self;
    fn bit0(self) -> bool;
    fn is_zero(self) -> bool;
    fn to_u64(self) -> u64;
}

macro_rules! impl_saturating_int {
    ($($t:ty),*) => {
        $(
            impl SaturatingInt for $t {
                const MAX_VALUE: Self = <$t>::MAX;
                fn wrapping_add(self, o: Self) -> Self { <$t>::wrapping_add(self, o) }
                fn wrapping_mul(self, o: Self) -> Self { <$t>::wrapping_mul(self, o) }
                fn shr1(self) -> Self { self >> 1 }
                fn shl1(self) -> Self { self << 1 }
                fn not_bits(self) -> Self { !self }
                fn bit0(self) -> bool { (self & 1) != 0 }
                fn is_zero(self) -> bool { self == 0 }
                fn to_u64(self) -> u64 { self as u64 }
            }
        )*
    };
}

impl_saturating_int!(u8, u16, u32, u64, usize);

/// Use this rather than HUGE_VALF; the latter causes warnings on MSVC.
pub const huge_valf: f32 = f32::INFINITY;

#[cfg(test)]
mod tests {
    use super::*;

    // The MSVC `_BitScan*` intrinsics have no Rust body: the counter
    // specializations use `trailing_zeros`/`leading_zeros` directly. The
    // observable contract of `_BitScanForward{,64}` (index of least-significant
    // set bit) is exactly `count_trailing_zeros`, and `_BitScanReverse{,64}`
    // (index of most-significant set bit) is `Log2_*`; we exercise those
    // equivalences here so the intrinsic contract is genuinely covered.
    // [spec:et:sem:llvm-math-extras.bit-scan-forward-fn/test]
    // [spec:et:sem:llvm-math-extras.bit-scan-forward64-fn/test]
    // [spec:et:sem:llvm-math-extras.bit-scan-reverse-fn/test]
    // [spec:et:sem:llvm-math-extras.bit-scan-reverse64-fn/test]
    #[test]
    fn bit_scan_contract_matches_std_bit_ops() {
        // _BitScanForward(mask) -> index of lowest set bit.
        assert_eq!(count_trailing_zeros(0x0000_0010u32, ZbUndefined), 4);
        assert_eq!(
            count_trailing_zeros(0x8000_0000_0000_0000u64, ZbUndefined),
            63
        );
        // _BitScanReverse(mask) -> index of highest set bit == Log2 (floor).
        assert_eq!(Log2_32(0x0000_0010u32), 4);
        assert_eq!(Log2_64(0x8000_0000_0000_0000u64), 63);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-zeros-fn/test]
    #[test]
    fn count_trailing_zeros_basic_and_zero_behavior() {
        // Generic (<4-byte) specialization via u8/u16.
        assert_eq!(count_trailing_zeros(0b0000_1000u8, ZbUndefined), 3);
        assert_eq!(count_trailing_zeros(0x0100u16, ZbUndefined), 8);
        // 4-byte specialization.
        assert_eq!(count_trailing_zeros(0x0000_0001u32, ZbUndefined), 0);
        assert_eq!(count_trailing_zeros(0x8000_0000u32, ZbUndefined), 31);
        // 8-byte specialization.
        assert_eq!(
            count_trailing_zeros(0x0000_0000_0001_0000u64, ZbUndefined),
            16
        );
        // Val == 0: non-Undefined behaviors return the type width (digits).
        assert_eq!(count_trailing_zeros(0u8, ZbWidth), 8);
        assert_eq!(count_trailing_zeros(0u16, ZbWidth), 16);
        assert_eq!(count_trailing_zeros(0u32, ZbWidth), 32);
        assert_eq!(count_trailing_zeros(0u64, ZbMax), 64);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-zeros-fn/test]
    #[test]
    fn count_leading_zeros_basic_and_zero_behavior() {
        assert_eq!(count_leading_zeros(0x01u8, ZbUndefined), 7);
        assert_eq!(count_leading_zeros(0x80u8, ZbUndefined), 0);
        assert_eq!(count_leading_zeros(0x0001u16, ZbUndefined), 15);
        assert_eq!(count_leading_zeros(0x0000_0001u32, ZbUndefined), 31);
        assert_eq!(count_leading_zeros(0x8000_0000u32, ZbUndefined), 0);
        assert_eq!(
            count_leading_zeros(0x0000_0000_0000_0001u64, ZbUndefined),
            63
        );
        // Val == 0 returns width for non-Undefined behaviors.
        assert_eq!(count_leading_zeros(0u8, ZbWidth), 8);
        assert_eq!(count_leading_zeros(0u32, ZbWidth), 32);
        assert_eq!(count_leading_zeros(0u64, ZbWidth), 64);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.find-first-set-fn/test]
    #[test]
    fn find_first_set_returns_index_or_max() {
        assert_eq!(find_first_set(0x0000_0018u32, ZbMax), 3);
        assert_eq!(find_first_set(0x0000_0001u32, ZbMax), 0);
        assert_eq!(find_first_set(0x8000_0000u32, ZbMax), 31);
        // ZB_Max: zero input yields all-ones sentinel.
        assert_eq!(find_first_set(0u32, ZbMax), u32::MAX);
        assert_eq!(find_first_set(0u8, ZbMax), u8::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.find-last-set-fn/test]
    #[test]
    fn find_last_set_returns_index_or_max() {
        assert_eq!(find_last_set(0x0000_0018u32, ZbMax), 4);
        assert_eq!(find_last_set(0x0000_0001u32, ZbMax), 0);
        assert_eq!(find_last_set(0x8000_0000u32, ZbMax), 31);
        assert_eq!(find_last_set(0x01u8, ZbMax), 0);
        assert_eq!(find_last_set(0x80u8, ZbMax), 7);
        assert_eq!(find_last_set(0u32, ZbMax), u32::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-ones-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-ones-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-zeros-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-zeros-fn/test]
    #[test]
    fn mask_helpers() {
        assert_eq!(mask_trailing_ones::<u8>(0), 0x00);
        assert_eq!(mask_trailing_ones::<u8>(3), 0x07);
        assert_eq!(mask_trailing_ones::<u8>(8), 0xFF);
        assert_eq!(mask_trailing_ones::<u32>(12), 0x0000_0FFF);
        assert_eq!(mask_trailing_ones::<u64>(64), u64::MAX);

        assert_eq!(mask_leading_ones::<u8>(3), 0xE0);
        assert_eq!(mask_leading_ones::<u32>(4), 0xF000_0000);

        assert_eq!(mask_trailing_zeros::<u8>(3), 0xF8);
        assert_eq!(mask_leading_zeros::<u8>(3), 0x1F);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.reverse-bits-fn/test]
    #[test]
    fn reverse_bits_full_width() {
        assert_eq!(reverse_bits(0x01u8), 0x80);
        assert_eq!(reverse_bits(0x80u8), 0x01);
        assert_eq!(reverse_bits(0x0000_0001u32), 0x8000_0000);
        assert_eq!(
            reverse_bits(0x8000_0000_0000_0000u64),
            0x0000_0000_0000_0001
        );
        // Involution: reversing twice is the identity.
        let v = 0x1234_5678u32;
        assert_eq!(reverse_bits(reverse_bits(v)), v);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.hi-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.lo-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.make-64-fn/test]
    #[test]
    fn hi_lo_make_64() {
        let v = 0xDEAD_BEEF_1234_5678u64;
        assert_eq!(Hi_32(v), 0xDEAD_BEEF);
        assert_eq!(Lo_32(v), 0x1234_5678);
        assert_eq!(Make_64(0xDEAD_BEEF, 0x1234_5678), v);
        assert_eq!(Make_64(Hi_32(v), Lo_32(v)), v);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-8-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-16-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-32-fn/test]
    #[test]
    fn is_int_and_specializations() {
        // isInt<4>: valid range [-8, 7].
        assert!(is_int(4, 7));
        assert!(is_int(4, -8));
        assert!(!is_int(4, 8));
        assert!(!is_int(4, -9));
        // N >= 64 is always true.
        assert!(is_int(64, i64::MAX));
        assert!(is_int(64, i64::MIN));

        assert!(is_int_8(127));
        assert!(is_int_8(-128));
        assert!(!is_int_8(128));
        assert!(is_int_16(32767));
        assert!(!is_int_16(32768));
        assert!(is_int_32(i32::MAX as i64));
        assert!(!is_int_32(i32::MAX as i64 + 1));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-int-fn/test]
    #[test]
    fn is_shifted_int_checks() {
        // N=4, S=2: value must fit in isInt<6> ([-32,31]) and be a multiple of 4.
        assert!(is_shifted_int(4, 2, 12));
        assert!(!is_shifted_int(4, 2, 13));
        assert!(!is_shifted_int(4, 2, 32));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-8-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-16-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-32-fn/test]
    #[test]
    fn is_u_int_and_specializations() {
        // isUInt<4>: valid range [0, 15].
        assert!(is_u_int(4, 15));
        assert!(!is_u_int(4, 16));
        // N >= 64 always true.
        assert!(is_u_int(64, u64::MAX));

        assert!(is_u_int_8(255));
        assert!(!is_u_int_8(256));
        assert!(is_u_int_16(65535));
        assert!(!is_u_int_16(65536));
        assert!(is_u_int_32(u32::MAX as u64));
        assert!(!is_u_int_32(u32::MAX as u64 + 1));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-u-int-fn/test]
    #[test]
    fn is_shifted_u_int_checks() {
        // N=4, S=2: value must fit in isUInt<6> ([0,63]) and be a multiple of 4.
        assert!(is_shifted_u_int(4, 2, 12));
        assert!(!is_shifted_u_int(4, 2, 13));
        assert!(!is_shifted_u_int(4, 2, 64));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.max-u-int-n-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.min-int-n-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.max-int-n-fn/test]
    #[test]
    fn min_max_int_n() {
        assert_eq!(max_u_int_n(8), 255);
        assert_eq!(max_u_int_n(1), 1);
        assert_eq!(max_u_int_n(64), u64::MAX);

        assert_eq!(min_int_n(8), -128);
        assert_eq!(max_int_n(8), 127);
        assert_eq!(min_int_n(64), i64::MIN);
        assert_eq!(max_int_n(64), i64::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-n-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-n-fn/test]
    #[test]
    fn is_u_int_n_and_is_int_n() {
        assert!(is_u_int_n(8, 255));
        assert!(!is_u_int_n(8, 256));
        assert!(is_u_int_n(64, u64::MAX));

        assert!(is_int_n(8, 127));
        assert!(is_int_n(8, -128));
        assert!(!is_int_n(8, 128));
        assert!(!is_int_n(8, -129));
        assert!(is_int_n(64, i64::MIN));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-64-fn/test]
    #[test]
    fn is_mask() {
        assert!(is_mask_32(0x0000_FFFF));
        assert!(is_mask_32(0x0000_0001));
        assert!(is_mask_32(u32::MAX));
        assert!(!is_mask_32(0));
        assert!(!is_mask_32(0x0000_FF01));

        assert!(is_mask_64(0x0000_0000_FFFF_FFFF));
        assert!(is_mask_64(u64::MAX));
        assert!(!is_mask_64(0));
        assert!(!is_mask_64(0x0000_0000_0000_0002));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-64-fn/test]
    #[test]
    fn is_shifted_mask() {
        assert!(is_shifted_mask_32(0x0000_FF00));
        assert!(is_shifted_mask_32(0x0000_FFFF));
        assert!(!is_shifted_mask_32(0));
        assert!(!is_shifted_mask_32(0x0000_FF01));

        assert!(is_shifted_mask_64(0x00FF_FF00_0000_0000));
        assert!(!is_shifted_mask_64(0));
        assert!(!is_shifted_mask_64(0x0000_0000_0000_0101));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-64-fn/test]
    #[test]
    fn is_power_of2() {
        assert!(is_power_of2_32(0x0010_0000));
        assert!(is_power_of2_32(1));
        assert!(!is_power_of2_32(0));
        assert!(!is_power_of2_32(3));

        assert!(is_power_of2_64(1u64 << 40));
        assert!(!is_power_of2_64(0));
        assert!(!is_power_of2_64(6));
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-ones-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-ones-fn/test]
    #[test]
    fn count_leading_and_trailing_ones() {
        assert_eq!(count_leading_ones(0xFF0F_FF00u32, ZbWidth), 8);
        assert_eq!(count_leading_ones(0xFFFF_FFFFu32, ZbWidth), 32);
        assert_eq!(count_leading_ones(0x0000_0000u32, ZbWidth), 0);

        assert_eq!(count_trailing_ones(0x00FF_00FFu32, ZbWidth), 8);
        assert_eq!(count_trailing_ones(0xFFFF_FFFFu32, ZbWidth), 32);
        assert_eq!(count_trailing_ones(0x0000_0000u32, ZbWidth), 0);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter-t-8.count-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.count-population-fn/test]
    #[test]
    fn count_population() {
        // <=4-byte generic specialization.
        assert_eq!(super::count_population(0xF000_F000u32), 8);
        assert_eq!(super::count_population(0u32), 0);
        assert_eq!(super::count_population(0xFFu8), 8);
        // 8-byte specialization.
        assert_eq!(super::count_population(u64::MAX), 64);
        assert_eq!(super::count_population(0xF000_0000_0000_000Fu64), 8);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.log2-fn/test]
    #[test]
    fn log2_double() {
        assert_eq!(Log2(8.0), 3.0);
        assert_eq!(Log2(1.0), 0.0);
        assert!((Log2(1024.0) - 10.0).abs() < 1e-12);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-fn/test]
    #[test]
    fn log2_floor() {
        assert_eq!(Log2_32(32), 5);
        assert_eq!(Log2_32(1), 0);
        assert_eq!(Log2_32(6), 2);
        // Zero returns (unsigned)-1.
        assert_eq!(Log2_32(0), u32::MAX);

        assert_eq!(Log2_64(1u64 << 40), 40);
        assert_eq!(Log2_64(1), 0);
        assert_eq!(Log2_64(0), u32::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-ceil-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-ceil-fn/test]
    #[test]
    fn log2_ceil() {
        assert_eq!(Log2_32_Ceil(32), 5);
        assert_eq!(Log2_32_Ceil(1), 0);
        assert_eq!(Log2_32_Ceil(6), 3);
        // Zero returns the width.
        assert_eq!(Log2_32_Ceil(0), 32);

        assert_eq!(Log2_64_Ceil(1u64 << 40), 40);
        assert_eq!(Log2_64_Ceil(6), 3);
        assert_eq!(Log2_64_Ceil(0), 64);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.greatest-common-divisor64-fn/test]
    #[test]
    fn gcd64() {
        assert_eq!(GreatestCommonDivisor64(48, 36), 12);
        assert_eq!(GreatestCommonDivisor64(17, 5), 1);
        assert_eq!(GreatestCommonDivisor64(0, 9), 9);
        assert_eq!(GreatestCommonDivisor64(9, 0), 9);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-double-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.double-to-bits-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-float-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.float-to-bits-fn/test]
    #[test]
    fn bit_reinterpret_round_trips() {
        // 1.0f64 == 0x3FF0000000000000.
        assert_eq!(DoubleToBits(1.0), 0x3FF0_0000_0000_0000);
        assert_eq!(BitsToDouble(0x3FF0_0000_0000_0000), 1.0);
        assert_eq!(BitsToDouble(DoubleToBits(3.14159)), 3.14159);

        // 1.0f32 == 0x3F800000.
        assert_eq!(FloatToBits(1.0), 0x3F80_0000);
        assert_eq!(BitsToFloat(0x3F80_0000), 1.0);
        assert_eq!(BitsToFloat(FloatToBits(2.5)), 2.5);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.min-align-fn/test]
    #[test]
    fn min_align() {
        // The largest power of two dividing both operands.
        assert_eq!(MinAlign(8, 12), 4);
        assert_eq!(MinAlign(2, 4), 2);
        assert_eq!(MinAlign(16, 16), 16);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.align-addr-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.alignment-adjustment-fn/test]
    #[test]
    fn align_addr_and_adjustment() {
        assert_eq!(alignAddr(7 as *const core::ffi::c_void, 4), 8);
        assert_eq!(alignAddr(8 as *const core::ffi::c_void, 4), 8);
        assert_eq!(alignAddr(9 as *const core::ffi::c_void, 8), 16);

        assert_eq!(alignmentAdjustment(7 as *const core::ffi::c_void, 4), 1);
        assert_eq!(alignmentAdjustment(8 as *const core::ffi::c_void, 4), 0);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.next-power-of2-fn/test]
    #[test]
    fn next_power_of2() {
        assert_eq!(NextPowerOf2(0), 1);
        assert_eq!(NextPowerOf2(7), 8);
        assert_eq!(NextPowerOf2(8), 16);
        // Strictly greater than A; overflow returns 0.
        assert_eq!(NextPowerOf2(u64::MAX), 0);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-floor-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-ceil-fn/test]
    #[test]
    fn power_of2_floor_ceil() {
        assert_eq!(PowerOf2Floor(0), 0);
        assert_eq!(PowerOf2Floor(6), 4);
        assert_eq!(PowerOf2Floor(8), 8);

        assert_eq!(PowerOf2Ceil(0), 0);
        assert_eq!(PowerOf2Ceil(6), 8);
        assert_eq!(PowerOf2Ceil(8), 8);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.align-to-fn/test]
    #[test]
    fn align_to() {
        assert_eq!(alignTo(5, 8, 0), 8);
        assert_eq!(alignTo(17, 8, 0), 24);
        assert_eq!(alignTo(321, 255, 0), 510);
        // Documented wrap: alignTo(~0LL, 8) == 0.
        assert_eq!(alignTo(u64::MAX, 8, 0), 0);
        // With skew.
        assert_eq!(alignTo(5, 8, 7), 7);
        assert_eq!(alignTo(17, 8, 1), 17);
        assert_eq!(alignTo(321, 255, 42), 552);
    }

    // [spec:et:def:llvm-math-extras.executorch.llvm.align-to]
    // [spec:et:def:llvm-math-extras.executorch.llvm.align-to.from-value]
    #[test]
    fn align_to_const_helper() {
        assert_eq!(AlignTo::<8>::from_value::<5>(), 8);
        assert_eq!(AlignTo::<8>::from_value::<17>(), 24);
        assert_eq!(AlignTo::<255>::from_value::<321>(), 510);
        assert_eq!(AlignTo::<4>::from_value::<8>(), 8);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.divide-ceil-fn/test]
    #[test]
    fn divide_ceil() {
        assert_eq!(divideCeil(10, 3), 4);
        assert_eq!(divideCeil(9, 3), 3);
        assert_eq!(divideCeil(1, 1), 1);
        assert_eq!(divideCeil(0, 5), 0);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.align-down-fn/test]
    #[test]
    fn align_down() {
        assert_eq!(alignDown(5, 8, 0), 0);
        assert_eq!(alignDown(17, 8, 0), 16);
        assert_eq!(alignDown(15, 8, 0), 8);
        // With skew: (17 - 1)/8*8 + 1 == 17.
        assert_eq!(alignDown(17, 8, 1), 17);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.offset-to-alignment-fn/test]
    #[test]
    fn offset_to_alignment() {
        assert_eq!(OffsetToAlignment(5, 8), 3);
        assert_eq!(OffsetToAlignment(8, 8), 0);
        assert_eq!(OffsetToAlignment(17, 8), 7);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend32-fn/test]
    // [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend64-fn/test]
    #[test]
    fn sign_extend() {
        // Bottom 4 bits: 0xF -> -1, 0x8 -> -8, 0x7 -> 7.
        assert_eq!(SignExtend32(0x0F, 4), -1);
        assert_eq!(SignExtend32(0x08, 4), -8);
        assert_eq!(SignExtend32(0x07, 4), 7);
        // B == 32 is the identity reinterpret.
        assert_eq!(SignExtend32(0xFFFF_FFFF, 32), -1);

        assert_eq!(SignExtend64(0x0F, 4), -1);
        assert_eq!(SignExtend64(0x07, 4), 7);
        assert_eq!(SignExtend64(u64::MAX, 64), -1);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.absolute-difference-fn/test]
    #[test]
    fn absolute_difference() {
        assert_eq!(AbsoluteDifference(10u32, 4u32), 6);
        assert_eq!(AbsoluteDifference(4u32, 10u32), 6);
        assert_eq!(AbsoluteDifference(7u8, 7u8), 0);
        assert_eq!(AbsoluteDifference(0u64, u64::MAX), u64::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-add-fn/test]
    #[test]
    fn saturating_add() {
        let mut overflowed = false;
        assert_eq!(SaturatingAdd(3u8, 4u8, Some(&mut overflowed)), 7);
        assert!(!overflowed);

        assert_eq!(SaturatingAdd(200u8, 100u8, Some(&mut overflowed)), 255);
        assert!(overflowed);

        assert_eq!(SaturatingAdd(255u8, 0u8, Some(&mut overflowed)), 255);
        assert!(!overflowed);

        // None sink still returns the clamped result.
        assert_eq!(SaturatingAdd(u32::MAX, 1u32, None), u32::MAX);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-fn/test]
    #[test]
    fn saturating_multiply() {
        let mut overflowed = false;
        // No overflow: exact product.
        assert_eq!(SaturatingMultiply(3u8, 4u8, Some(&mut overflowed)), 12);
        assert!(!overflowed);
        // A zero operand short-circuits (Log2_64(0) path) with no overflow.
        assert_eq!(SaturatingMultiply(0u8, 5u8, Some(&mut overflowed)), 0);
        assert!(!overflowed);
        // Boundary product that still fits.
        assert_eq!(SaturatingMultiply(127u8, 2u8, Some(&mut overflowed)), 254);
        assert!(!overflowed);
        // Overflow clamps to max, sets flag.
        assert_eq!(SaturatingMultiply(16u8, 16u8, Some(&mut overflowed)), 255);
        assert!(overflowed);
        assert_eq!(SaturatingMultiply(255u8, 255u8, Some(&mut overflowed)), 255);
        assert!(overflowed);
        // Odd-X top-bit path that overflows through SaturatingAdd.
        assert_eq!(SaturatingMultiply(128u8, 2u8, Some(&mut overflowed)), 255);
        assert!(overflowed);
    }

    // [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-add-fn/test]
    #[test]
    fn saturating_multiply_add() {
        let mut overflowed = false;
        // 3*4 + 5 = 17, no overflow.
        assert_eq!(
            SaturatingMultiplyAdd(3u8, 4u8, 5u8, Some(&mut overflowed)),
            17
        );
        assert!(!overflowed);
        // Multiply overflows -> returns product (max), overflow set.
        assert_eq!(
            SaturatingMultiplyAdd(255u8, 255u8, 1u8, Some(&mut overflowed)),
            255
        );
        assert!(overflowed);
        // Multiply fits but the add saturates.
        assert_eq!(
            SaturatingMultiplyAdd(100u8, 2u8, 100u8, Some(&mut overflowed)),
            255
        );
        assert!(overflowed);
    }
}
