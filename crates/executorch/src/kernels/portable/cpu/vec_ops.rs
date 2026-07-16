//! Literal port of kernels/portable/cpu/vec_ops.h.
//!
//! This header defines common, low-level operations that can often be
//! vectorized/accelerated on hardware targets.

// PORT-NOTE: the C++ functions take `float* ET_RESTRICT` etc. — raw,
// non-owning, non-aliasing pointers with explicit `size`. Ported as raw Rust
// pointers with the arithmetic in `unsafe` blocks, mirroring the tensor-aliasing
// boundary convention in the runtime port. `ET_RESTRICT` (no-alias) is a
// documented precondition, not enforced by the type system here.
//
// The templated matmul-family functions (`vec_matmul`,
// `vec_quantized_matmul_int8`, `vec_quantized_matmul_transb_int8`, `vec_addmm`,
// `vec_softmax`) are generic over the element/scalar types. Their arithmetic is
// expressed via the small `VecScalar` / `VecMatScalar` traits so the `sum += ...`
// mixed-type accumulation stays literal.

use crate::runtime::core::portable_type::{BFloat16, Half};

/// Returns the minimum element of the array at `x`, which must have `size`
/// elements.
///
/// # Safety
/// `x` must point to at least `size` valid `f32` elements, and `size >= 1`.
// [spec:et:def:vec-ops.torch.executor.vec-minf-fn]
// [spec:et:sem:vec-ops.torch.executor.vec-minf-fn]
pub unsafe fn vec_minf(x: *const f32, size: usize) -> f32 {
    // *std::min_element(x, x + size): first minimal element under operator<.
    // PRECONDITION size >= 1 (empty range is UB in C++).
    let mut best: f32 = unsafe { *x };
    let mut i: usize = 1;
    while i < size {
        let v = unsafe { *x.add(i) };
        // std::min_element replaces best only when the new element is strictly
        // less than the current best (`v < best`), keeping the FIRST minimum on
        // ties and never displacing on unordered (NaN) comparisons.
        if v < best {
            best = v;
        }
        i += 1;
    }
    best
}

/// Returns the maximum element of the array at `x`, which must have `size`
/// elements.
///
/// # Safety
/// `x` must point to at least `size` valid `f32` elements, and `size >= 1`.
// [spec:et:def:vec-ops.torch.executor.vec-maxf-fn]
// [spec:et:sem:vec-ops.torch.executor.vec-maxf-fn]
pub unsafe fn vec_maxf(x: *const f32, size: usize) -> f32 {
    // *std::max_element(x, x + size): first maximal element under operator<.
    let mut best: f32 = unsafe { *x };
    let mut i: usize = 1;
    while i < size {
        let v = unsafe { *x.add(i) };
        // std::max_element replaces best only when best is strictly less than
        // the new element (`best < v`), keeping the FIRST maximum on ties.
        if best < v {
            best = v;
        }
        i += 1;
    }
    best
}

/// Add each element of `x` and `y` into the corresponding element of `z`. All
/// arrays must have `size` elements.
///
/// # Safety
/// `z`, `x`, `y` must each point to at least `size` valid `f32` elements and
/// must not alias.
// [spec:et:def:vec-ops.torch.executor.vec-addf-fn]
// [spec:et:sem:vec-ops.torch.executor.vec-addf-fn]
pub unsafe fn vec_addf(z: *mut f32, x: *const f32, y: *const f32, size: usize) {
    for i in 0..size {
        unsafe {
            *z.add(i) = *x.add(i) + *y.add(i);
        }
    }
}

/// Multiplies every element of `x` by `scale`, and writes the result into the
/// corresponding element of `y`. `x` and `y` must have `size` elements.
///
/// # Safety
/// `y` and `x` must each point to at least `size` valid `f32` elements and must
/// not alias.
// [spec:et:def:vec-ops.torch.executor.vec-scalef-fn]
// [spec:et:sem:vec-ops.torch.executor.vec-scalef-fn]
pub unsafe fn vec_scalef(y: *mut f32, x: *const f32, scale: f32, size: usize) {
    for i in 0..size {
        unsafe {
            *y.add(i) = *x.add(i) * scale;
        }
    }
}

// PORT-NOTE: mixed-type multiply-accumulate `sum += x * y`. In C++ the
// accumulator type `T` and operand type `U` are template params; the product
// promotes per usual arithmetic conversions. Modeled by `VecScalar<U>`: convert
// each operand to the accumulator type and multiply/add there. For the concrete
// instantiations used by ExecuTorch (float/double, plus int8 operands widened to
// U) this matches the C++ promotion.
pub trait VecScalar: Copy {
    fn zero() -> Self;
    fn add_assign(&mut self, other: Self);
    fn mul(self, other: Self) -> Self;
    fn from_u(u: Self) -> Self {
        u
    }
}

macro_rules! impl_vec_scalar {
    ($t:ty) => {
        impl VecScalar for $t {
            fn zero() -> Self {
                0 as $t
            }
            fn add_assign(&mut self, other: Self) {
                *self += other;
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
        }
    };
}
impl_vec_scalar!(f32);
impl_vec_scalar!(f64);
impl_vec_scalar!(i8);
impl_vec_scalar!(i16);
impl_vec_scalar!(i32);
impl_vec_scalar!(i64);
impl_vec_scalar!(u8);

// PORT-NOTE: `c10::Half`/`BFloat16` participate in `vec_matmul`/`vec_addmm` over
// the REALHBF16 set (e.g. addmm). Their `+`/`*` promote through `float`; the
// accumulator stays in the element type between operations, matching the C++
// `sum += x[..] * y[..]` on `CTYPE`.
impl VecScalar for crate::runtime::core::portable_type::Half {
    fn zero() -> Self {
        crate::runtime::core::portable_type::Half::from_f32(0.0)
    }
    fn add_assign(&mut self, other: Self) {
        *self = crate::runtime::core::portable_type::Half::from_f32(self.to_f32() + other.to_f32());
    }
    fn mul(self, other: Self) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(self.to_f32() * other.to_f32())
    }
}
impl VecScalar for crate::runtime::core::portable_type::BFloat16 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(0.0)
    }
    fn add_assign(&mut self, other: Self) {
        *self =
            crate::runtime::core::portable_type::BFloat16::from_f32(self.to_f32() + other.to_f32());
    }
    fn mul(self, other: Self) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(self.to_f32() * other.to_f32())
    }
}

/// x: m * n, y: n * p, z: m * p.
/// z\[i]\[j] = sum(x\[i]\[k] * y\[k]\[j])
///
/// # Safety
/// `z` (m*p), `x` (m*n), `y` (n*p) must be valid and non-aliasing.
pub unsafe fn vec_matmul<T>(z: *mut T, x: *const T, y: *const T, m: i64, n: i64, p: i64)
where
    T: VecScalar,
{
    for i in 0..m {
        for j in 0..p {
            let mut sum: T = T::zero();
            for k in 0..n {
                unsafe {
                    let a = *x.offset((i * n + k) as isize);
                    let b = *y.offset((k * p + j) as isize);
                    sum.add_assign(a.mul(b));
                }
            }
            unsafe {
                *z.offset((i * p + j) as isize) = sum;
            }
        }
    }
}

// PORT-NOTE: `sum += x[..] * static_cast<U>(y[..]) * s[..]` where `y` is int8
// and `U` is the tensor/scalar type. Modeled with a `FromI8` conversion into the
// accumulator type.
pub trait FromI8 {
    fn from_i8(v: i8) -> Self;
}
impl FromI8 for f32 {
    fn from_i8(v: i8) -> Self {
        v as f32
    }
}
impl FromI8 for f64 {
    fn from_i8(v: i8) -> Self {
        v as f64
    }
}
impl FromI8 for i32 {
    fn from_i8(v: i8) -> Self {
        v as i32
    }
}
impl FromI8 for i64 {
    fn from_i8(v: i8) -> Self {
        v as i64
    }
}
// PORT-NOTE: `static_cast<U>(y[..])` with `U = c10::Half`/`BFloat16` — the int8
// weight is cast to the reduced-precision float type (via float) so the
// quantized matmul family (`vec_quantized_matmul_int8`,
// `vec_quantized_matmul_transb_int8`) can instantiate over {Float, Half} exactly
// as op_mixed_mm.cpp / op_mixed_linear.cpp require. Added to fill the gap left by
// the initial port, which only covered f32/f64/i32/i64.
impl FromI8 for Half {
    fn from_i8(v: i8) -> Self {
        Half::from_f32(v as f32)
    }
}
impl FromI8 for BFloat16 {
    fn from_i8(v: i8) -> Self {
        BFloat16::from_f32(v as f32)
    }
}

/// # Safety
/// `z`, `x`, `y`, `s` must be valid for the m/n/p extents and non-aliasing.
pub unsafe fn vec_quantized_matmul_int8<T>(
    z: *mut T,
    x: *const T,
    y: *const i8,
    s: *const T,
    m: i64,
    n: i64,
    p: i64,
) where
    T: VecScalar + FromI8,
{
    for i in 0..m {
        for j in 0..p {
            let mut sum: T = T::zero();
            for k in 0..n {
                unsafe {
                    let xv = *x.offset((i * n + k) as isize);
                    let yv = T::from_i8(*y.offset((k * p + j) as isize));
                    let sv = *s.offset(k as isize);
                    sum.add_assign(xv.mul(yv).mul(sv));
                }
            }
            unsafe {
                *z.offset((i * p + j) as isize) = sum;
            }
        }
    }
}

// [spec:et:def:vec-ops.torch.executor.bounds-min-fn]
// [spec:et:sem:vec-ops.torch.executor.bounds-min-fn]
fn bounds_min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}

/// x: m * n, y: p * n, z: m * p, s: p * groups
/// z\[i]\[j] = sum(x\[i]\[k] * y\[j]\[k] * s\[j]\[k/g])
///
/// # Safety
/// `z`, `x`, `y`, `s` must be valid for the m/n/p/g extents and non-aliasing.
pub unsafe fn vec_quantized_matmul_transb_int8<T>(
    z: *mut T,
    x: *const T,
    y: *const i8,
    s: *const T,
    m: i64,
    n: i64,
    p: i64,
    g: i64,
) where
    T: VecScalar + FromI8,
{
    let n_over_g: i64 = (n + g - 1) / g;

    for i in 0..m {
        for j in 0..p {
            let mut sum: T = T::zero();
            let mut k: i64 = 0;
            while k < n {
                let mut psum: T = T::zero();
                // the last group may have fewer than g elements
                let hi = bounds_min((k + g) as usize, n as usize);
                for k2 in (k as usize)..hi {
                    unsafe {
                        let xv = *x.offset((i * n) as isize + k2 as isize);
                        let yv = T::from_i8(*y.offset((j * n) as isize + k2 as isize));
                        psum.add_assign(xv.mul(yv));
                    }
                }
                unsafe {
                    let sv = *s.offset((j * n_over_g + k / g) as isize);
                    sum.add_assign(psum.mul(sv));
                }
                k += g;
            }
            unsafe {
                *z.offset((i * p + j) as isize) = sum;
            }
        }
    }
}

// mat1 (m x n), mat2 (n x p), out (m, p), self (m x p)
// z[i][j] = sum(x[i][k] * y[k][j]), for k in range(n)
// T for tensor dtype, U for scalar type
/// # Safety
/// `out_data`, `self_data`, `mat1_data`, `mat2_data` must be valid for the
/// m/n/p extents and non-aliasing.
pub unsafe fn vec_addmm<T>(
    out_data: *mut T,
    self_data: *const T,
    mat1_data: *const T,
    mat2_data: *const T,
    m: i64,
    n: i64,
    p: i64,
    beta: T,
    alpha: T,
) where
    T: VecScalar,
{
    for i in 0..m {
        for j in 0..p {
            let mut sum: T = T::zero();
            for k in 0..n {
                unsafe {
                    let a = *mat1_data.offset((i * n + k) as isize);
                    let b = *mat2_data.offset((k * p + j) as isize);
                    sum.add_assign(a.mul(b));
                }
            }
            unsafe {
                let s = *self_data.offset((i * p + j) as isize);
                *out_data.offset((i * p + j) as isize) = sum.mul(alpha).add_result(s.mul(beta));
            }
        }
    }
}

// PORT-NOTE: helper for `sum * alpha + self * beta`; `VecScalar::mul` covers the
// products, this covers the final `+`.
trait AddResult {
    fn add_result(self, other: Self) -> Self;
}
impl<T: VecScalar> AddResult for T {
    fn add_result(self, other: Self) -> Self {
        let mut acc = self;
        acc.add_assign(other);
        acc
    }
}

// PORT-NOTE: `reduce_add` / `vec_powerf` accumulate into a `float` regardless of
// the element type `T`. `ReduceToF32` promotes each element to `f32` (matching
// the C++ `float`-accumulator promotion) for the running sum.
pub trait ReduceToF32: Copy {
    fn to_f32(self) -> f32;
}
macro_rules! impl_reduce_to_f32 {
    ($t:ty) => {
        impl ReduceToF32 for $t {
            fn to_f32(self) -> f32 {
                self as f32
            }
        }
    };
}
impl_reduce_to_f32!(f32);
impl_reduce_to_f32!(f64);
impl_reduce_to_f32!(u8);
impl_reduce_to_f32!(i8);
impl_reduce_to_f32!(i16);
impl_reduce_to_f32!(i32);
impl_reduce_to_f32!(i64);
impl ReduceToF32 for Half {
    fn to_f32(self) -> f32 {
        Half::to_f32(self)
    }
}
impl ReduceToF32 for BFloat16 {
    fn to_f32(self) -> f32 {
        BFloat16::to_f32(self)
    }
}

/// Returns the sum of all elements in `x`, which must have `size` elements.
///
/// # Safety
/// `x` must point to at least `size` valid `T` elements.
// [spec:et:def:vec-ops.torch.executor.reduce-add-fn]
// [spec:et:sem:vec-ops.torch.executor.reduce-add-fn]
pub unsafe fn reduce_add<T: ReduceToF32>(x: *const T, size: usize) -> f32 {
    // std::accumulate(x, x + size, 0.f): float accumulator, left-to-right.
    let mut acc: f32 = 0.0f32;
    for i in 0..size {
        acc = acc + unsafe { *x.add(i) }.to_f32();
    }
    acc
}

/// Returns the sum of the squares of all elements in `x`, which must have
/// `size` elements.
///
/// # Safety
/// `x` must point to at least `size` valid `T` elements.
// [spec:et:def:vec-ops.torch.executor.vec-powerf-fn]
// [spec:et:sem:vec-ops.torch.executor.vec-powerf-fn]
pub unsafe fn vec_powerf<T: ReduceToF32>(x: *const T, size: usize) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..size {
        // Note only the LEFT operand is explicitly cast to float in the C++;
        // the right `x[i]` promotes to float via the multiplication, so each
        // term is `f32(x[i]) * f32(x[i])`.
        let xi = unsafe { *x.add(i) };
        sum += xi.to_f32() * xi.to_f32();
    }
    sum
}

// PORT-NOTE: `vec_softmax` is templated on T, U constrained to float/double by
// SFINAE. Modeled with a `SoftmaxFloat` trait providing exp/div and f32/f64
// widening. `std::max_element` uses operator< (first maximal on ties).
pub trait SoftmaxFloat: Copy {
    fn zero() -> Self;
    fn exp(self) -> Self;
    fn add_assign(&mut self, other: Self);
    fn div_assign(&mut self, other: Self);
    fn sub(self, other: Self) -> Self;
    fn lt(self, other: Self) -> bool;
}
macro_rules! impl_softmax_float {
    ($t:ty) => {
        impl SoftmaxFloat for $t {
            fn zero() -> Self {
                0 as $t
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn add_assign(&mut self, other: Self) {
                *self += other;
            }
            fn div_assign(&mut self, other: Self) {
                *self /= other;
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn lt(self, other: Self) -> bool {
                self < other
            }
        }
    };
}
impl_softmax_float!(f32);
impl_softmax_float!(f64);

/// Computes the result of softmax(x, x+n), write into y.
/// y = e ^ (x - max(x)) / sum(e^(x - max(x)))
///
/// # Safety
/// `y` and `x` must each point to at least `n` valid elements and must not
/// alias; `n >= 1`.
pub unsafe fn vec_softmax<T, U>(y: *mut T, x: *const U, n: i32)
where
    T: SoftmaxFloat,
    U: SoftmaxFloat + Into<T>,
{
    // U max_x = *std::max_element(x, x + n);
    let mut max_x: U = unsafe { *x };
    {
        let mut i: i32 = 1;
        while i < n {
            let v = unsafe { *x.offset(i as isize) };
            if max_x.lt(v) {
                max_x = v;
            }
            i += 1;
        }
    }
    let mut sum: T = T::zero();

    for i in 0..n {
        unsafe {
            let xi = *x.offset(i as isize);
            let val: T = xi.sub(max_x).exp().into();
            *y.offset(i as isize) = val;
            sum.add_assign(*y.offset(i as isize));
        }
    }

    for i in 0..n {
        unsafe {
            (*y.offset(i as isize)).div_assign(sum);
        }
    }
}

mod internal {
    // [spec:et:def:vec-ops.torch.executor.internal.clamp-fn]
    // [spec:et:sem:vec-ops.torch.executor.internal.clamp-fn]
    // PORT-NOTE: the C++ selects `std::clamp` when `__cpp_lib_clamp` is
    // available, else the `v < lo ? lo : (hi < v ? hi : v)` fallback; both are
    // behaviorally identical. The fallback form is ported here (it also
    // reproduces the documented NaN behavior: a NaN `v` returns `v` unchanged
    // since both `v < lo` and `hi < v` are false).
    pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
        if v < lo {
            lo
        } else if hi < v {
            hi
        } else {
            v
        }
    }
}

/// Quantizes the elements of `x` into `y`, both of which must have `size`
/// elements. Inverse of `dequantize_i8_f32()`.
///
/// # Safety
/// `y` and `x` must each point to at least `size` valid elements and must not
/// alias.
// [spec:et:def:vec-ops.torch.executor.quantize-i8-f32-fn]
// [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn]
pub unsafe fn quantize_i8_f32(y: *mut i8, x: *const f32, scale: f32, zero_point: i32, size: usize) {
    for i in 0..size {
        // std::round rounds half away from zero, in float.
        let tmp: f32 = (unsafe { *x.add(i) } * scale + zero_point as f32).round();
        unsafe {
            *y.add(i) = internal::clamp(tmp, -128.0f32, 127.0f32) as i8;
        }
    }
}

/// Dequantizes the elements of `x` into `y`, both of which must have `size`
/// elements. Inverse of `quantize_i8_f32()`.
///
/// # Safety
/// `y` and `x` must each point to at least `size` valid elements and must not
/// alias.
// [spec:et:def:vec-ops.torch.executor.dequantize-i8-f32-fn]
// [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn]
pub unsafe fn dequantize_i8_f32(
    y: *mut f32,
    x: *const i8,
    scale: f32,
    zero_point: i32,
    size: usize,
) {
    for i in 0..size {
        // `x[i] - zero_point` in int32, then to float and multiply by scale.
        let diff: i32 = unsafe { *x.add(i) } as i32 - zero_point;
        unsafe {
            *y.add(i) = scale * (diff as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from kernels/portable/cpu/test/vec_ops_test.cpp.

    // [spec:et:sem:vec-ops.torch.executor.vec-minf-fn/test]
    #[test]
    fn vec_minf_test_smoke() {
        let x: [f32; 5] = [1.1, -2.2, 0.0, -1234.5, 10.0];
        assert_eq!(unsafe { vec_minf(x.as_ptr(), x.len()) }, -1234.5);
    }

    // [spec:et:sem:vec-ops.torch.executor.vec-maxf-fn/test]
    #[test]
    fn vec_maxf_test_smoke() {
        let x: [f32; 5] = [1.1, -2.2, 0.0, -1234.5, 10.0];
        assert_eq!(unsafe { vec_maxf(x.as_ptr(), x.len()) }, 10.0);
    }

    // [spec:et:sem:vec-ops.torch.executor.vec-addf-fn/test]
    #[test]
    fn vec_addf_test_smoke() {
        let in1: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let in2: [f32; 5] = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out: [f32; 5] = [0.0; 5];

        unsafe { vec_addf(out.as_mut_ptr(), in1.as_ptr(), in2.as_ptr(), 5) };

        assert_eq!(out[0], 11.0);
        assert_eq!(out[1], 22.0);
        assert_eq!(out[2], 33.0);
        assert_eq!(out[3], 44.0);
        assert_eq!(out[4], 55.0);
    }

    // [spec:et:sem:vec-ops.torch.executor.vec-scalef-fn/test]
    #[test]
    fn vec_scalef_test_smoke() {
        let input: [f32; 5] = [4.0, 8.0, 16.0, 32.0, 64.0];
        let mut out: [f32; 5] = [0.0; 5];

        unsafe { vec_scalef(out.as_mut_ptr(), input.as_ptr(), 0.5, 5) };

        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 4.0);
        assert_eq!(out[2], 8.0);
        assert_eq!(out[3], 16.0);
        assert_eq!(out[4], 32.0);
    }

    // [spec:et:sem:vec-ops.torch.executor.vec-powerf-fn/test]
    #[test]
    fn vec_powerf_test_smoke() {
        let input: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert_eq!(
            unsafe { vec_powerf(input.as_ptr(), 5) },
            (-2.0 * -2.0) + (-1.0 * -1.0) + (0.0 * 0.0) + (1.0 * 1.0) + (2.0 * 2.0)
        );
    }

    // reduce_add is not exercised by the C++ vec_ops_test.cpp suite; this is a
    // focused unit test for the float-accumulator left-to-right sum described by
    // the sem rule.
    // [spec:et:sem:vec-ops.torch.executor.reduce-add-fn/test]
    #[test]
    fn reduce_add_test_smoke() {
        // size == 0 returns 0.0f.
        let empty: [f32; 0] = [];
        assert_eq!(unsafe { reduce_add(empty.as_ptr(), 0) }, 0.0f32);

        // float accumulator over floats.
        let xf: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.5];
        assert_eq!(unsafe { reduce_add(xf.as_ptr(), 5) }, 0.5f32);

        // int8 elements promote to float before adding.
        let xi: [i8; 4] = [-128, 0, 64, 127];
        assert_eq!(
            unsafe { reduce_add(xi.as_ptr(), 4) },
            (-128i32 + 0 + 64 + 127) as f32
        );
    }

    // bounds_min is a private helper only reached through
    // vec_quantized_matmul_transb_int8; test it directly against the
    // `(a < b) ? a : b` semantics (equal args return b, numerically identical).
    // [spec:et:sem:vec-ops.torch.executor.bounds-min-fn/test]
    #[test]
    fn bounds_min_test_smoke() {
        assert_eq!(bounds_min(3, 7), 3);
        assert_eq!(bounds_min(7, 3), 3);
        assert_eq!(bounds_min(5, 5), 5);
        assert_eq!(bounds_min(0, 9), 0);
    }

    // clamp is private to `internal`; exercised end-to-end through the quantize
    // clamping cases below, but also pinned here directly for the lo/hi/NaN
    // branches described by the sem rule.
    // [spec:et:sem:vec-ops.torch.executor.internal.clamp-fn/test]
    #[test]
    fn internal_clamp_test() {
        assert_eq!(internal::clamp(-200.0, -128.0, 127.0), -128.0);
        assert_eq!(internal::clamp(200.0, -128.0, 127.0), 127.0);
        assert_eq!(internal::clamp(5.0, -128.0, 127.0), 5.0);
        // NaN passes through unchanged: both `v < lo` and `hi < v` are false.
        assert!(internal::clamp(f32::NAN, -128.0, 127.0).is_nan());
    }

    fn quantize_inputs() -> Vec<f32> {
        let inf = f32::INFINITY;
        vec![
            -inf, -512.0, -256.0, -128.0, -64.0, 0.0, 64.0, 128.0, 256.0, 512.0, inf,
        ]
    }

    // [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn/test]
    // also verifies internal::clamp (min/max saturation branches)
    #[test]
    fn quantize_i8_f32_test_identity() {
        let inputs = quantize_inputs();
        let mut outputs = vec![0i8; inputs.len()];
        unsafe { quantize_i8_f32(outputs.as_mut_ptr(), inputs.as_ptr(), 1.0, 0, inputs.len()) };
        assert_eq!(
            outputs,
            vec![-128, -128, -128, -128, -64, 0, 64, 127, 127, 127, 127]
        );
    }

    // [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn/test]
    #[test]
    fn quantize_i8_f32_test_rounding() {
        // roundf() semantics (half away from zero), not ceil/floor.
        let input = vec![-1.9f32, -1.1, 1.1, 1.9];
        let mut out = vec![0i8; input.len()];
        unsafe { quantize_i8_f32(out.as_mut_ptr(), input.as_ptr(), 1.0, 0, input.len()) };
        assert_eq!(out, vec![-2, -1, 1, 2]);
    }

    // [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn/test]
    #[test]
    fn quantize_i8_f32_test_scaled_down() {
        let inputs = quantize_inputs();
        let mut outputs = vec![0i8; inputs.len()];
        unsafe { quantize_i8_f32(outputs.as_mut_ptr(), inputs.as_ptr(), 0.5, 0, inputs.len()) };
        assert_eq!(
            outputs,
            vec![-128, -128, -128, -64, -32, 0, 32, 64, 127, 127, 127]
        );
    }

    // [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn/test]
    #[test]
    fn quantize_i8_f32_test_shifted_zero_point() {
        let inputs = quantize_inputs();
        let mut outputs = vec![0i8; inputs.len()];
        unsafe { quantize_i8_f32(outputs.as_mut_ptr(), inputs.as_ptr(), 1.0, 32, inputs.len()) };
        assert_eq!(
            outputs,
            vec![-128, -128, -128, -96, -32, 32, 96, 127, 127, 127, 127]
        );
    }

    // [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn/test]
    #[test]
    fn quantize_i8_f32_test_scaled_down_with_shifted_zero_point() {
        let inputs = quantize_inputs();
        let mut outputs = vec![0i8; inputs.len()];
        unsafe { quantize_i8_f32(outputs.as_mut_ptr(), inputs.as_ptr(), 0.5, 32, inputs.len()) };
        // Zero point adjustment happens after scaling.
        assert_eq!(
            outputs,
            vec![-128, -128, -96, -32, 0, 32, 64, 96, 127, 127, 127]
        );
    }

    fn dequantize_inputs() -> Vec<i8> {
        vec![-128, -64, -32, 0, 32, 64, 127]
    }

    // [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn/test]
    #[test]
    fn dequantize_i8_f32_test_identity() {
        let inputs = dequantize_inputs();
        let mut outputs = vec![0.0f32; inputs.len()];
        unsafe { dequantize_i8_f32(outputs.as_mut_ptr(), inputs.as_ptr(), 1.0, 0, inputs.len()) };
        assert_eq!(outputs, vec![-128.0, -64.0, -32.0, 0.0, 32.0, 64.0, 127.0]);
    }

    // [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn/test]
    #[test]
    fn dequantize_i8_f32_test_scaled_up() {
        let inputs = dequantize_inputs();
        let scale = 2.0f32;
        let mut outputs = vec![0.0f32; inputs.len()];
        unsafe {
            dequantize_i8_f32(
                outputs.as_mut_ptr(),
                inputs.as_ptr(),
                scale,
                0,
                inputs.len(),
            )
        };
        let expected: Vec<f32> = inputs.iter().map(|&v| v as f32 * scale).collect();
        assert_eq!(outputs, expected);
    }

    // [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn/test]
    #[test]
    fn dequantize_i8_f32_test_shifted_zero_point() {
        let inputs = dequantize_inputs();
        let zero_point = 32i32;
        let mut outputs = vec![0.0f32; inputs.len()];
        unsafe {
            dequantize_i8_f32(
                outputs.as_mut_ptr(),
                inputs.as_ptr(),
                1.0,
                zero_point,
                inputs.len(),
            )
        };
        let expected: Vec<f32> = inputs
            .iter()
            .map(|&v| (v as i32 - zero_point) as f32)
            .collect();
        assert_eq!(outputs, expected);
    }

    // [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn/test]
    #[test]
    fn dequantize_i8_f32_test_scaled_up_with_shifted_zero_point() {
        let inputs = dequantize_inputs();
        let scale = 2.0f32;
        let zero_point = 32i32;
        let mut outputs = vec![0.0f32; inputs.len()];
        unsafe {
            dequantize_i8_f32(
                outputs.as_mut_ptr(),
                inputs.as_ptr(),
                scale,
                zero_point,
                inputs.len(),
            )
        };
        // Zero point adjustment happens before scaling.
        let expected: Vec<f32> = inputs
            .iter()
            .map(|&v| (v as i32 - zero_point) as f32 * scale)
            .collect();
        assert_eq!(outputs, expected);
    }
}
