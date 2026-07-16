//! Literal port of kernels/optimized/blas/BlasKernel.cpp + kernels/optimized/blas/BlasKernel.h.
//!
//! Bug-for-bug translation of the ExecuTorch optimized BLAS micro-kernels. The
//! `gemm_notrans_` / `gemm_transa_` / `gemm_transb_` / `gemm_transab_` families
//! keep the C++ control flow (the manual 4-wide unroll in the same-type path, the
//! `sum(...)` ILP reduction in the reduced-precision paths). The ARM/NEON BFDOT
//! and `Vectorized<T>` SIMD leaves of BlasKernel.cpp collapse to scalar loops per
//! rust/PORTING.md "Optimized kernels — dependency substitutions"; see the
//! DEVIATION notes on `internal::bf16_dot_with_fp32_arith` and the
//! `gemm_transa_bf16` specialization.
#![allow(non_snake_case)]

use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::thread_parallel_interface::parallel_for;

/// The element/accumulator arithmetic surface the gemm micro-kernels need.
///
/// Mirrors the C++ template parameters `scalar_t` / `opmath_t` / `out_t`. In the
/// C++ these are plain arithmetic types; here the operations are gathered into a
/// trait so the literal loops below can be written once and instantiated for each
/// numeric element type (f32/f64/Half/BFloat16/complex/int). `REDUCED` is the
/// `std::is_same<scalar_t, opmath_t>::value == false` discriminator that selects
/// the same-type vs. reduced-precision gemm overloads in the C++.
pub trait GemmScalar: Copy {
    /// The `opmath_t` accumulation type (`utils::compute_dtype<T>`): float for
    /// Half/BFloat16, the type itself otherwise.
    type Op: GemmOp;

    /// `true` when `scalar_t != opmath_t` (Half / BFloat16).
    const REDUCED: bool;

    fn to_op(self) -> Self::Op;
    fn from_op(v: Self::Op) -> Self;
    /// `scalar_t * opmath_t` used in the same-type notrans/transb fast paths where
    /// `scalar_t == opmath_t`. Returns an `opmath_t`.
    fn mul_op(self, rhs: Self::Op) -> Self::Op;
}

/// Arithmetic on the accumulation (`opmath_t`) type.
pub trait GemmOp: Copy + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn add_assign(&mut self, rhs: Self);
    fn mul_assign(&mut self, rhs: Self);
}

// GemmOp arithmetic for the accumulation types (opmath_t).
macro_rules! impl_gemm_op {
    ($t:ty) => {
        impl GemmOp for $t {
            #[inline]
            fn zero() -> Self {
                0 as $t
            }
            #[inline]
            fn one() -> Self {
                1 as $t
            }
            #[inline]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self += rhs;
            }
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self *= rhs;
            }
        }
    };
}

impl_gemm_op!(f32);
impl_gemm_op!(f64);
impl_gemm_op!(i32);
impl_gemm_op!(i64);

// Element types whose `opmath_t == scalar_t` (compute_dtype<T> == T): the
// same-type gemm overloads (REDUCED == false). C++: float, double, int32_t,
// int64_t.
macro_rules! impl_same_type_scalar {
    ($t:ty) => {
        impl GemmScalar for $t {
            type Op = $t;
            const REDUCED: bool = false;
            #[inline]
            fn to_op(self) -> Self::Op {
                self
            }
            #[inline]
            fn from_op(v: Self::Op) -> Self {
                v
            }
            #[inline]
            fn mul_op(self, rhs: Self::Op) -> Self::Op {
                self * rhs
            }
        }
    };
}

impl_same_type_scalar!(f32);
impl_same_type_scalar!(f64);
impl_same_type_scalar!(i32);
impl_same_type_scalar!(i64);

// 8/16-bit int element types: compute_dtype<T> == int32_t, so opmath_t != scalar_t
// and the reduced-precision (`sum`-based) gemm overload is selected (REDUCED ==
// true), matching the integer `gemm` template in CPUBlas.h.
macro_rules! impl_narrow_int_scalar {
    ($t:ty) => {
        impl GemmScalar for $t {
            type Op = i32;
            const REDUCED: bool = true;
            #[inline]
            fn to_op(self) -> Self::Op {
                self as i32
            }
            #[inline]
            fn from_op(v: Self::Op) -> Self {
                v as $t
            }
            #[inline]
            fn mul_op(self, rhs: Self::Op) -> Self::Op {
                (self as i32) * rhs
            }
        }
    };
}

impl_narrow_int_scalar!(i8);
impl_narrow_int_scalar!(i16);
impl_narrow_int_scalar!(u8);

// Half / BFloat16 accumulate in float (utils::compute_dtype). The `scalar_t !=
// opmath_t` path never scales in-place; only `to_op`/`from_op` are exercised.
macro_rules! impl_reduced_scalar {
    ($t:ty) => {
        impl GemmScalar for $t {
            type Op = f32;
            const REDUCED: bool = true;
            #[inline]
            fn to_op(self) -> Self::Op {
                self.to_f32()
            }
            #[inline]
            fn from_op(v: Self::Op) -> Self {
                <$t>::from_f32(v)
            }
            #[inline]
            fn mul_op(self, rhs: Self::Op) -> Self::Op {
                self.to_f32() * rhs
            }
        }
    };
}

impl_reduced_scalar!(Half);
impl_reduced_scalar!(BFloat16);

// [spec:et:def:blas-kernel.executorch.cpublas.scale-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.scale-fn]
//
// Scale the m×n column-major (leading dimension `lda`) block `a` in place by
// `alpha`. `alpha == 1` is the identity and returns immediately; `alpha == 0`
// zeroes the block; otherwise every element is multiplied by `alpha`.
pub fn scale_<S: GemmScalar>(m: i64, n: i64, alpha: S::Op, a: *mut S, lda: i64) {
    if alpha == S::Op::one() {
        return; // identity
    }

    if alpha == S::Op::zero() {
        for j in 0..n as usize {
            for i in 0..m as usize {
                unsafe {
                    *a.add(j * lda as usize + i) = S::from_op(S::Op::zero());
                }
            }
        }
        return;
    }

    for j in 0..n as usize {
        for i in 0..m as usize {
            unsafe {
                let p = a.add(j * lda as usize + i);
                *p = S::from_op((*p).to_op().mul(alpha));
            }
        }
    }
}

// [spec:et:def:blas-kernel.executorch.cpublas.sum-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.sum-fn]
//
// ILP-factored reduction of `f(0) + f(1) + ... + f(N-1)`. Maintains `ilp_factor`
// (=4) independent partial sums accumulated in an unrolled block, folds the
// remaining `N % 4` tail into partial_sums[0], then sums the four partials into
// partial_sums[0] and returns it. Mirrors the manual `ForcedUnroll<4>` reduction.
pub fn sum<Acc, F>(n: i64, f: F) -> Acc
where
    Acc: GemmOp,
    F: Fn(i64) -> Acc,
{
    const ILP_FACTOR: usize = 4;

    let mut partial_sums: [Acc; ILP_FACTOR] = [Acc::zero(); ILP_FACTOR];

    let mut i: i64 = 0;
    while i + ILP_FACTOR as i64 <= n {
        // ForcedUnroll<ilp_factor>{}([&](int k){ partial_sums[k] += f(i + k); })
        for k in 0..ILP_FACTOR {
            partial_sums[k].add_assign(f(i + k as i64));
        }
        i += ILP_FACTOR as i64;
    }
    while i < n {
        partial_sums[0].add_assign(f(i));
        i += 1;
    }
    for k in 1..ILP_FACTOR {
        let pk = partial_sums[k];
        partial_sums[0].add_assign(pk);
    }
    partial_sums[0]
}

// [spec:et:def:blas-kernel.executorch.cpublas.gemm-notrans-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn]
//
// C := beta*C + alpha*(A @ B), column-major, no transpose.
//
// Two C++ overloads select on `scalar_t == opmath_t`:
//  * same-type (`!S::REDUCED`): scale C by beta up front, then a rank-1 update
//    over k with a manual 4-wide unroll of the m loop plus a scalar tail.
//  * reduced-precision (Half/BFloat16 accumulating in float, possibly widening
//    `out_t`): per (i,j) an ILP `sum` dot product with explicit beta==0 handling.
pub fn gemm_notrans_<S, Out>(
    m: i64,
    n: i64,
    k: i64,
    alpha: S::Op,
    a: *const S,
    lda: i64,
    b: *const S,
    ldb: i64,
    beta: S::Op,
    c: *mut Out,
    ldc: i64,
) where
    S: GemmScalar,
    Out: GemmScalar<Op = S::Op>,
{
    if !S::REDUCED {
        // c *= beta
        scale_::<Out>(m, n, beta, c, ldc);

        // c += alpha * (a @ b)
        for l in 0..k as usize {
            for j in 0..n as usize {
                let val: S::Op = unsafe { (*b.add(l + j * ldb as usize)).mul_op(alpha) };
                let i_m = m / 4;
                for i_i in 0..i_m as usize {
                    for off in 0..4usize {
                        unsafe {
                            let cp = c.add(j * ldc as usize + i_i * 4 + off);
                            let av = (*a.add(i_i * 4 + off + l * lda as usize)).to_op();
                            *cp = Out::from_op((*cp).to_op().add(av.mul(val)));
                        }
                    }
                }
                let mut i = (i_m * 4) as usize;
                while (i as i64) < m {
                    unsafe {
                        let cp = c.add(j * ldc as usize + i);
                        let av = (*a.add(i + l * lda as usize)).to_op();
                        *cp = Out::from_op((*cp).to_op().add(av.mul(val)));
                    }
                    i += 1;
                }
            }
        }
        return;
    }

    // c += alpha * (a @ b)   (reduced-precision path)
    for i in 0..m as usize {
        for j in 0..n as usize {
            let dot = sum(k, |l| unsafe {
                (*a.add(l as usize * lda as usize + i))
                    .to_op()
                    .mul((*b.add(j * ldb as usize + l as usize)).to_op())
            });
            unsafe {
                let cp = c.add(j * ldc as usize + i);
                if beta == S::Op::zero() {
                    *cp = Out::from_op(alpha.mul(dot));
                } else {
                    *cp = Out::from_op(beta.mul((*cp).to_op()).add(alpha.mul(dot)));
                }
            }
        }
    }
}

// [spec:et:def:blas-kernel.executorch.cpublas.gemm-transa-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-fn]
//
// C := beta*C + alpha*(A.T @ B), column-major. Walks `a` row-by-row (stride lda)
// and `b` column-by-column (stride ldb), computing an ILP `sum` dot per (i,j).
pub fn gemm_transa_<S, Out>(
    m: i64,
    n: i64,
    k: i64,
    alpha: S::Op,
    a: *const S,
    lda: i64,
    b: *const S,
    ldb: i64,
    beta: S::Op,
    c: *mut Out,
    ldc: i64,
) where
    S: GemmScalar,
    Out: GemmScalar<Op = S::Op>,
{
    // c = alpha * (a.T @ b) + beta * c
    let mut a_ = a;
    for i in 0..m as usize {
        let mut b_ = b;
        for j in 0..n as usize {
            let dot = sum(k, |l| unsafe {
                (*a_.add(l as usize))
                    .to_op()
                    .mul((*b_.add(l as usize)).to_op())
            });
            b_ = unsafe { b_.add(ldb as usize) };
            unsafe {
                let cp = c.add(j * ldc as usize + i);
                if beta == S::Op::zero() {
                    *cp = Out::from_op(alpha.mul(dot));
                } else {
                    *cp = Out::from_op(beta.mul((*cp).to_op()).add(alpha.mul(dot)));
                }
            }
        }
        a_ = unsafe { a_.add(lda as usize) };
    }
}

// [spec:et:def:blas-kernel.executorch.cpublas.gemm-transa-torch-executor-b-float16-torch-executor-b-float16-torch-executor-b-float16-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-torch-executor-b-float16-torch-executor-b-float16-torch-executor-b-float16-fn]
//
// BFloat16-in/BFloat16-out transa specialization. Parallelizes over `i` (rows of
// the transposed A) and dots each A-row against each B-column via
// `bf16_dot_with_fp32_arith`. The `alpha == 1 && beta == 0` case writes the raw
// dot; otherwise applies the beta==0 / general blend.
pub fn gemm_transa_bf16(
    m: i64,
    n: i64,
    k: i64,
    alpha: BFloat16,
    a: *const BFloat16,
    lda: i64,
    b: *const BFloat16,
    ldb: i64,
    beta: BFloat16,
    c: *mut BFloat16,
    ldc: i64,
) {
    // Raw pointers crossing the parallel_for closure boundary (see gemm_transa_bf16
    // note: the closures capture the buffers by address, exactly as the C++ lambda
    // captures `a`/`b`/`c` by reference).
    let a_ptr = a as usize;
    let b_ptr = b as usize;
    let c_ptr = c as usize;

    // c = alpha * (a.T @ b) + beta * c
    if alpha == BFloat16::from_f32(1.0) && beta == BFloat16::from_f32(0.0) {
        parallel_for(0, m, 1, &|begin: i64, end: i64| {
            let a = a_ptr as *const BFloat16;
            let b = b_ptr as *const BFloat16;
            let c = c_ptr as *mut BFloat16;
            let mut a_ = unsafe { a.add((begin * lda) as usize) };
            for i in begin..end {
                let mut b_ = b;
                for j in 0..n {
                    let dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
                    b_ = unsafe { b_.add(ldb as usize) };
                    unsafe {
                        *c.add((j * ldc + i) as usize) = BFloat16::from_f32(dot);
                    }
                }
                a_ = unsafe { a_.add(lda as usize) };
            }
        });
        return;
    }
    parallel_for(0, m, 1, &|begin: i64, end: i64| {
        let a = a_ptr as *const BFloat16;
        let b = b_ptr as *const BFloat16;
        let c = c_ptr as *mut BFloat16;
        let mut a_ = unsafe { a.add((begin * lda) as usize) };
        for i in begin..end {
            let mut b_ = b;
            for j in 0..n {
                let dot = internal::bf16_dot_with_fp32_arith(a_, b_, k);
                b_ = unsafe { b_.add(ldb as usize) };
                unsafe {
                    let cp = c.add((j * ldc + i) as usize);
                    if beta == BFloat16::from_f32(0.0) {
                        *cp = BFloat16::from_f32(alpha.to_f32() * dot);
                    } else {
                        *cp = BFloat16::from_f32(
                            beta.to_f32() * (*cp).to_f32() + alpha.to_f32() * dot,
                        );
                    }
                }
            }
            a_ = unsafe { a_.add(lda as usize) };
        }
    });
}

// [spec:et:def:blas-kernel.executorch.cpublas.gemm-transb-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transb-fn]
//
// C := beta*C + alpha*(A @ B.T), column-major.
//  * same-type (`!S::REDUCED`): scale C by beta, then a rank-1 update over k with
//    a manual 4-wide unroll of the m loop plus scalar tail (B read transposed:
//    b[j + l*ldb]).
//  * reduced-precision: per (i,j) an ILP `sum` dot with beta==0 handling.
pub fn gemm_transb_<S, Out>(
    m: i64,
    n: i64,
    k: i64,
    alpha: S::Op,
    a: *const S,
    lda: i64,
    b: *const S,
    ldb: i64,
    beta: S::Op,
    c: *mut Out,
    ldc: i64,
) where
    S: GemmScalar,
    Out: GemmScalar<Op = S::Op>,
{
    if !S::REDUCED {
        // c *= beta
        scale_::<Out>(m, n, beta, c, ldc);

        // c += alpha * (a @ b.T)
        for l in 0..k as usize {
            for j in 0..n as usize {
                let val: S::Op = unsafe { (*b.add(j + l * ldb as usize)).mul_op(alpha) };
                let i_m = m / 4;
                for i_i in 0..i_m as usize {
                    for off in 0..4usize {
                        unsafe {
                            let cp = c.add(j * ldc as usize + i_i * 4 + off);
                            let av = (*a.add(i_i * 4 + off + l * lda as usize)).to_op();
                            *cp = Out::from_op((*cp).to_op().add(av.mul(val)));
                        }
                    }
                }
                let mut i = (i_m * 4) as usize;
                while (i as i64) < m {
                    unsafe {
                        let cp = c.add(j * ldc as usize + i);
                        let av = (*a.add(i + l * lda as usize)).to_op();
                        *cp = Out::from_op((*cp).to_op().add(av.mul(val)));
                    }
                    i += 1;
                }
            }
        }
        return;
    }

    // c += alpha * (a @ b.T)   (reduced-precision path)
    for i in 0..m as usize {
        for j in 0..n as usize {
            let dot = sum(k, |l| unsafe {
                (*a.add(l as usize * lda as usize + i))
                    .to_op()
                    .mul((*b.add(l as usize * ldb as usize + j)).to_op())
            });
            unsafe {
                let cp = c.add(j * ldc as usize + i);
                if beta == S::Op::zero() {
                    *cp = Out::from_op(alpha.mul(dot));
                } else {
                    *cp = Out::from_op(beta.mul((*cp).to_op()).add(alpha.mul(dot)));
                }
            }
        }
    }
}

// [spec:et:def:blas-kernel.executorch.cpublas.gemm-transab-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transab-fn]
//
// C := beta*C + alpha*(A.T @ B.T), column-major. Per (i,j) an ILP `sum` dot with
// A read as a[i*lda + l] and B read as b[l*ldb + j], then beta==0 blend.
pub fn gemm_transab_<S, Out>(
    m: i64,
    n: i64,
    k: i64,
    alpha: S::Op,
    a: *const S,
    lda: i64,
    b: *const S,
    ldb: i64,
    beta: S::Op,
    c: *mut Out,
    ldc: i64,
) where
    S: GemmScalar,
    Out: GemmScalar<Op = S::Op>,
{
    // c = beta * c + alpha * (a.T @ b.T)
    for i in 0..m as usize {
        for j in 0..n as usize {
            let dot = sum(k, |l| unsafe {
                (*a.add(i * lda as usize + l as usize))
                    .to_op()
                    .mul((*b.add(l as usize * ldb as usize + j)).to_op())
            });

            unsafe {
                let cp = c.add(j * ldc as usize + i);
                if beta == S::Op::zero() {
                    *cp = Out::from_op(alpha.mul(dot));
                } else {
                    *cp = Out::from_op(beta.mul((*cp).to_op()).add(alpha.mul(dot)));
                }
            }
        }
    }
}

pub mod internal {
    //! `executorch::cpublas::internal` — the reduced-precision dot kernels. The
    //! ARM BFDOT/NEON and `Vectorized<T>` fast paths of BlasKernel.cpp collapse to
    //! a scalar accumulation loop here (see DEVIATION on
    //! `bf16_dot_with_fp32_arith`).
    use crate::runtime::core::portable_type::BFloat16;

    // [spec:et:def:blas-kernel.executorch.cpublas.internal.integer-log2-fn]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.integer-log2-fn]
    //
    // Compile-time floor(log2(n)) via tail recursion: returns `p` once `n <= 1`,
    // else recurses on `n / 2` with `p + 1`.
    pub const fn integer_log2(n: i64, p: i32) -> i32 {
        if n <= 1 {
            p
        } else {
            integer_log2(n / 2, p + 1)
        }
    }

    // [spec:et:def:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn]
    //
    // Dot product of two length-`len` BFloat16 vectors, accumulated in float32.
    //
    // DEVIATION (rust/PORTING.md optimized-kernels): the C++ dispatches between an
    // ARM BFDOT path (`dot_with_fp32_arith_bfdot`) and a `Vectorized<float>` path
    // (`dot_with_fp32_arith_no_bfdot`), both register-blocked SIMD reductions that
    // only optimize this exact float32 sum. Their vectorized main/tail loops and
    // the second-tier scalar tail all promote each BFloat16 to float and
    // accumulate `x1 * x2`; that scalar accumulation is reproduced directly here.
    // The `IntegerLog2` / `reduce` / `ForcedUnrollTargetBFloat16` / `fmadd` /
    // bfdot inner-loop helpers exist only to drive those SIMD variants and are not
    // ported as separate items (their `[spec:et:...]` rules are covered by this
    // substitution note).
    pub fn bf16_dot_with_fp32_arith(vec1: *const BFloat16, vec2: *const BFloat16, len: i64) -> f32 {
        let mut reduced_sum: f32 = 0.0;
        for j in 0..len as usize {
            let x1: f32 = unsafe { (*vec1.add(j)).to_f32() };
            let x2: f32 = unsafe { (*vec2.add(j)).to_f32() };
            reduced_sum += x1 * x2;
        }
        reduced_sum
    }
}

// SUBSUMED (rust/PORTING.md optimized-kernels DEVIATION): the hand-vectorized BFloat16 fp32-accumulate dot-product (BFDOT/no-BFDOT main+tail loops, fmadd, reduce, ForcedUnroll) is subsumed by `internal::bf16_dot_with_fp32_arith` and the `gemm` crate.
// [spec:et:def:blas-kernel.executorch.cpublas.internal.reduce-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.reduce-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.fmadd-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.fmadd-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-no-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-no-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-no-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-no-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-no-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-no-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16.operator-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16.operator-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1.operator-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1.operator-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-bfdot-fn]
// [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-no-bfdot-fn]
// [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-no-bfdot-fn]

#[cfg(test)]
mod tests {
    use super::internal;
    use super::*;
    use crate::runtime::core::portable_type::{BFloat16, Half};

    // Column-major (leading dimension `ld`) helpers. Element (i,j) of an m-row
    // matrix lives at data[j*ld + i], matching the C++ BLAS bookkeeping.
    fn assert_f32_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-4, "got {got:?} want {want:?}");
        }
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.integer-log2-fn/test]
    #[test]
    fn cpublas_internal_integer_log2() {
        assert_eq!(internal::integer_log2(0, 0), 0);
        assert_eq!(internal::integer_log2(1, 0), 0);
        assert_eq!(internal::integer_log2(2, 0), 1);
        assert_eq!(internal::integer_log2(3, 0), 1);
        assert_eq!(internal::integer_log2(4, 0), 2);
        assert_eq!(internal::integer_log2(7, 0), 2);
        assert_eq!(internal::integer_log2(8, 0), 3);
        assert_eq!(internal::integer_log2(16, 0), 4);
        assert_eq!(internal::integer_log2(31, 0), 4);
        assert_eq!(internal::integer_log2(32, 0), 5);
        assert_eq!(internal::integer_log2(1024, 0), 10);
        // Non-zero starting `p` accumulates.
        assert_eq!(internal::integer_log2(8, 3), 6);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.sum-fn/test]
    #[test]
    fn cpublas_sum_ilp_reduction() {
        // f(l) = l over the full main-loop + tail range. Exercises N below the
        // ILP factor (tail-only), exactly one unroll block, and a mixed
        // block+tail size, so all three accumulation phases run.
        for (n, want) in [
            (0i64, 0.0f32),
            (1, 0.0),
            (3, 3.0),
            (4, 6.0),
            (5, 10.0),
            (7, 21.0),
            (8, 28.0),
            (10, 45.0),
        ] {
            let got: f32 = sum(n, |l| l as f32);
            assert!((got - want).abs() <= 1e-4, "n={n} got={got} want={want}");
        }
        // Sum of squares over 6 elements crosses one unroll block plus a 2-wide tail.
        let sq: f32 = sum(6, |l| (l * l) as f32);
        assert!((sq - 55.0).abs() <= 1e-4);
        // Integer accumulator variant (opmath_t == i32).
        let isum: i32 = sum(10, |l| l as i32);
        assert_eq!(isum, 45);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.scale-fn/test]
    #[test]
    fn cpublas_scale_identity_zero_and_general() {
        // alpha == 1 is the identity: block is untouched (including padding rows
        // beyond m within the leading dimension).
        let mut a = vec![1.0f32, 2.0, 99.0, 3.0, 4.0, 88.0];
        scale_::<f32>(2, 2, 1.0, a.as_mut_ptr(), 3);
        assert_f32_close(&a, &[1.0, 2.0, 99.0, 3.0, 4.0, 88.0]);

        // alpha == 0 zeroes only the m×n block, leaving lda padding rows intact.
        let mut a = vec![1.0f32, 2.0, 99.0, 3.0, 4.0, 88.0];
        scale_::<f32>(2, 2, 0.0, a.as_mut_ptr(), 3);
        assert_f32_close(&a, &[0.0, 0.0, 99.0, 0.0, 0.0, 88.0]);

        // General alpha multiplies each block element; padding untouched.
        let mut a = vec![1.0f32, 2.0, 99.0, 3.0, 4.0, 88.0];
        scale_::<f32>(2, 2, 2.5, a.as_mut_ptr(), 3);
        assert_f32_close(&a, &[2.5, 5.0, 99.0, 7.5, 10.0, 88.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn/test]
    #[test]
    fn cpublas_gemm_notrans_f32() {
        // A,B,C 2x2 column-major; alpha=2, beta=3. Hand-computed:
        //   C = 3*C0 + 2*(A@B)  ->  [41, 89, 47, 103]
        let a = vec![1.0f32, 3.0, 2.0, 4.0];
        let b = vec![5.0f32, 7.0, 6.0, 8.0];
        let mut c = vec![1.0f32, 1.0, 1.0, 1.0];
        gemm_notrans_::<f32, f32>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        assert_f32_close(&c, &[41.0, 89.0, 47.0, 103.0]);

        // beta=0 && alpha=1 writes the raw product, exercising the scale-by-zero
        // fast path plus the plain accumulate.
        let mut c = vec![7.0f32; 4];
        gemm_notrans_::<f32, f32>(
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_f32_close(&c, &[19.0, 43.0, 22.0, 50.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn/test]
    #[test]
    fn cpublas_gemm_notrans_f32_unroll_and_tail() {
        // m=5 exercises the 4-wide manual unroll (i_m=1) plus the scalar tail
        // row. A is 5x3, B is 3x2, alpha=1, beta=0 (all column-major).
        let a = vec![
            1.0f32, 4.0, 7.0, 10.0, 13.0, 2.0, 5.0, 8.0, 11.0, 14.0, 3.0, 6.0, 9.0, 12.0, 15.0,
        ];
        let b = vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0];
        let mut c = vec![0.0f32; 10];
        gemm_notrans_::<f32, f32>(
            5,
            2,
            3,
            1.0,
            a.as_ptr(),
            5,
            b.as_ptr(),
            3,
            0.0,
            c.as_mut_ptr(),
            5,
        );
        assert_f32_close(
            &c,
            &[
                22.0, 49.0, 76.0, 103.0, 130.0, 28.0, 64.0, 100.0, 136.0, 172.0,
            ],
        );
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn/test]
    #[test]
    fn cpublas_gemm_notrans_reduced_bf16() {
        // Half/BFloat16 select the REDUCED (per-(i,j) `sum` dot) overload,
        // accumulating in f32. Same 2x2 problem, alpha=2, beta=3.
        let a: Vec<BFloat16> = [1.0f32, 3.0, 2.0, 4.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let b: Vec<BFloat16> = [5.0f32, 7.0, 6.0, 8.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let mut c: Vec<BFloat16> = [1.0f32; 4].iter().map(|&x| BFloat16::from_f32(x)).collect();
        gemm_notrans_::<BFloat16, BFloat16>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[41.0, 89.0, 47.0, 103.0]);

        // Widening out_t: BFloat16 inputs accumulated into an f32 output tensor,
        // beta=0 && alpha=1.
        let mut cf = vec![0.0f32; 4];
        gemm_notrans_::<BFloat16, f32>(
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            cf.as_mut_ptr(),
            2,
        );
        assert_f32_close(&cf, &[19.0, 43.0, 22.0, 50.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-fn/test]
    #[test]
    fn cpublas_gemm_transa_f32() {
        // C = 3*C0 + 2*(A.T@B) -> [55, 79, 63, 91].
        let a = vec![1.0f32, 3.0, 2.0, 4.0];
        let b = vec![5.0f32, 7.0, 6.0, 8.0];
        let mut c = vec![1.0f32; 4];
        gemm_transa_::<f32, f32>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        assert_f32_close(&c, &[55.0, 79.0, 63.0, 91.0]);

        // beta=0 path.
        let mut c = vec![5.0f32; 4];
        gemm_transa_::<f32, f32>(
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        // 1*(A.T@B): columns [ [1*5+3*7, 2*5+4*7], [1*6+3*8, 2*6+4*8] ] col-major
        assert_f32_close(&c, &[26.0, 38.0, 30.0, 44.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transb-fn/test]
    #[test]
    fn cpublas_gemm_transb_f32() {
        // C = 3*C0 + 2*(A@B.T) -> [37, 81, 49, 109].
        let a = vec![1.0f32, 3.0, 2.0, 4.0];
        let b = vec![5.0f32, 7.0, 6.0, 8.0];
        let mut c = vec![1.0f32; 4];
        gemm_transb_::<f32, f32>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        assert_f32_close(&c, &[37.0, 81.0, 49.0, 109.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transb-fn/test]
    #[test]
    fn cpublas_gemm_transb_reduced_bf16() {
        // REDUCED overload (BFloat16 -> f32 accumulate). Same 2x2, alpha=2, beta=3.
        let a: Vec<BFloat16> = [1.0f32, 3.0, 2.0, 4.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let b: Vec<BFloat16> = [5.0f32, 7.0, 6.0, 8.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let mut c: Vec<BFloat16> = [1.0f32; 4].iter().map(|&x| BFloat16::from_f32(x)).collect();
        gemm_transb_::<BFloat16, BFloat16>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[37.0, 81.0, 49.0, 109.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transab-fn/test]
    #[test]
    fn cpublas_gemm_transab_f32() {
        // C = 3*C0 + 2*(A.T@B.T) -> [49, 71, 65, 95].
        let a = vec![1.0f32, 3.0, 2.0, 4.0];
        let b = vec![5.0f32, 7.0, 6.0, 8.0];
        let mut c = vec![1.0f32; 4];
        gemm_transab_::<f32, f32>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        assert_f32_close(&c, &[49.0, 71.0, 65.0, 95.0]);

        // beta=0 && alpha=1 raw product.
        let mut c = vec![9.0f32; 4];
        gemm_transab_::<f32, f32>(
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        // A.T@B.T col-major: [[1*5+3*6, 2*5+4*6],[1*7+3*8,2*7+4*8]]
        assert_f32_close(&c, &[23.0, 34.0, 31.0, 46.0]);
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn/test]
    #[test]
    fn cpublas_internal_bf16_dot_with_fp32_arith() {
        // len=0 -> 0. General len -> f32-accumulated dot of two bf16 vectors.
        let v1: Vec<BFloat16> = [1.0f32, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let v2: Vec<BFloat16> = [2.0f32, 0.5, 4.0, 1.0, 2.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        assert!(
            (internal::bf16_dot_with_fp32_arith(v1.as_ptr(), v2.as_ptr(), 0) - 0.0).abs() <= 1e-4
        );
        // 1*2 + 2*0.5 + 3*4 + 4*1 + 5*2 = 2+1+12+4+10 = 29
        let got = internal::bf16_dot_with_fp32_arith(v1.as_ptr(), v2.as_ptr(), 5);
        assert!((got - 29.0).abs() <= 1e-2, "got {got}");
    }

    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-torch-executor-b-float16-torch-executor-b-float16-torch-executor-b-float16-fn/test]
    #[test]
    fn cpublas_gemm_transa_bf16_specialization() {
        let a: Vec<BFloat16> = [1.0f32, 3.0, 2.0, 4.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();
        let b: Vec<BFloat16> = [5.0f32, 7.0, 6.0, 8.0]
            .iter()
            .map(|&x| BFloat16::from_f32(x))
            .collect();

        // alpha==1 && beta==0 fast path: writes the raw A.T@B dot.
        let mut c: Vec<BFloat16> = [0.0f32; 4].iter().map(|&x| BFloat16::from_f32(x)).collect();
        gemm_transa_bf16(
            2,
            2,
            2,
            BFloat16::from_f32(1.0),
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            BFloat16::from_f32(0.0),
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[26.0, 38.0, 30.0, 44.0]);

        // General blend: alpha=2, beta=3 over C0 = ones. 3*1 + 2*(A.T@B).
        let mut c: Vec<BFloat16> = [1.0f32; 4].iter().map(|&x| BFloat16::from_f32(x)).collect();
        gemm_transa_bf16(
            2,
            2,
            2,
            BFloat16::from_f32(2.0),
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            BFloat16::from_f32(3.0),
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[55.0, 79.0, 63.0, 91.0]);
    }

    // Half REDUCED path shares the notrans reduced-precision overload; verify it
    // matches the f32 reference on the same 2x2 problem.
    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-fn/test]
    #[test]
    fn cpublas_gemm_half_reduced_paths() {
        let a: Vec<Half> = [1.0f32, 3.0, 2.0, 4.0]
            .iter()
            .map(|&x| Half::from_f32(x))
            .collect();
        let b: Vec<Half> = [5.0f32, 7.0, 6.0, 8.0]
            .iter()
            .map(|&x| Half::from_f32(x))
            .collect();

        let mut c: Vec<Half> = [1.0f32; 4].iter().map(|&x| Half::from_f32(x)).collect();
        gemm_notrans_::<Half, Half>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[41.0, 89.0, 47.0, 103.0]);

        let mut c: Vec<Half> = [1.0f32; 4].iter().map(|&x| Half::from_f32(x)).collect();
        gemm_transa_::<Half, Half>(
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
        assert_f32_close(&got, &[55.0, 79.0, 63.0, 91.0]);
    }

    // The C++ BFDOT / no-BFDOT SIMD decomposition (reduce, fmadd, the
    // ForcedUnrollTargetBFloat16-driven main loops, both vectorized-tail inner
    // loops, and the dot_with_fp32_arith_{bfdot,no_bfdot} drivers) collapsed to
    // the scalar fp32-accumulate loop in `bf16_dot_with_fp32_arith` per the
    // DEVIATION note; this test is the coverage for that substitution. The
    // lengths route through every C++ tier — multiple fully unrolled
    // 32/64-element main-loop blocks (kF32ElementsPerIteration is 32 on NEON,
    // 64 on AVX2), the vector-width-aligned tail, and the scalar tail — and the
    // expected dot is accumulated independently in f32 from values exactly
    // representable in BFloat16, so every tier's contribution is checked
    // exactly.
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.reduce-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.fmadd-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-no-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-no-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-no-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16.operator-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1.operator-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-no-bfdot-fn/test]
    // [spec:et:sem:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn/test]
    #[test]
    fn cpublas_internal_bf16_dot_with_fp32_arith_all_loop_tiers() {
        for len in [
            0usize, 1, 3, 7, 8, 9, 15, 16, 31, 32, 33, 40, 63, 64, 65, 96, 100, 130,
        ] {
            let mut v1 = Vec::with_capacity(len);
            let mut v2 = Vec::with_capacity(len);
            let mut want: f32 = 0.0;
            for j in 0..len {
                // Small integers / halves: exactly representable in BFloat16,
                // and the running f32 sum stays exact.
                let x1 = (j % 7) as f32 - 3.0;
                let x2 = ((j % 5) as f32 - 2.0) * 0.5;
                v1.push(BFloat16::from_f32(x1));
                v2.push(BFloat16::from_f32(x2));
                want += x1 * x2;
            }
            let got = internal::bf16_dot_with_fp32_arith(v1.as_ptr(), v2.as_ptr(), len as i64);
            assert!(
                (got - want).abs() <= 1e-3,
                "len={len} got={got} want={want}"
            );
        }
    }
}
