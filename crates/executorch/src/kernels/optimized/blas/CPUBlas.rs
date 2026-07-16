//! Literal port of kernels/optimized/blas/CPUBlas.cpp + kernels/optimized/blas/CPUBlas.h.
//!
//! `executorch::cpublas` — the GEMM entry points that the optimized
//! linear/mm/bmm ops call. The `TransposeType` / `to_blas` / `normalize_last_dims`
//! bookkeeping and the per-dtype `gemm` overloads are ported literally. At the
//! inner compute leaf the C++ chooses between an Eigen/CBLAS FFI (`ET_BUILD_WITH_BLAS`)
//! and the hand-written `gemm_impl` fallback; per rust/PORTING.md "Optimized
//! kernels — dependency substitutions" the crate-supported float/complex overloads
//! call the `gemm` crate (DEVIATION: Eigen/CBLAS FFI → `gemm` crate), and the
//! remaining dtypes go through `gemm_impl` → the BlasKernel micro-kernels.
#![allow(non_snake_case)]

use gemm::Parallelism;

use crate::runtime::core::portable_type::{BFloat16, Complex, Half};

use super::BlasKernel::{
    GemmScalar, gemm_notrans_, gemm_transa_, gemm_transa_bf16, gemm_transab_, gemm_transb_,
};

// [spec:et:def:cpu-blas.executorch.cpublas.transpose-type]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransposeType {
    NoTranspose,
    Transpose,
    ConjTranspose,
}

// [spec:et:def:cpu-blas.executorch.cpublas.to-blas-fn]
// [spec:et:sem:cpu-blas.executorch.cpublas.to-blas-fn]
//
// Map a `TransposeType` to its LAPACK/BLAS transpose character: Transpose→'T',
// NoTranspose→'N', ConjTranspose→'C'; defaults to 'N'.
#[inline]
// SUBSUMED: BLAS entry point routed to the gemm crate.
// [spec:et:def:cpu-blas.executorch.cpublas.to-cblas-transpose-fn]
// [spec:et:sem:cpu-blas.executorch.cpublas.to-cblas-transpose-fn]
pub fn to_blas(trans: TransposeType) -> u8 {
    match trans {
        TransposeType::Transpose => b'T',
        TransposeType::NoTranspose => b'N',
        TransposeType::ConjTranspose => b'C',
    }
}

// DEVIATION: `to_cblas_transpose` (CBLAS_TRANSPOSE, ET_BUILD_FOR_APPLE Accelerate
// path) and the `dgemm_`/`sgemm_`/`cgemm_`/`zgemm_` extern "C" FFI declarations
// (ET_BUILD_WITH_BLAS non-Apple path) are the Eigen/CBLAS dependency leaf. Per
// rust/PORTING.md they are replaced by the `gemm` crate below, so the FFI shims
// are not ported. Their `[spec:et:...]` rules (cpu-blas.{c,d,s,z}gemm-fn,
// cpu-blas.executorch.cpublas.to-cblas-transpose-fn) are covered by this note.

// [spec:et:def:cpu-blas.executorch.cpublas.normalize-last-dims-fn]
// [spec:et:sem:cpu-blas.executorch.cpublas.normalize-last-dims-fn]
//
// Patch the leading dimensions for degenerate (size-1) trailing dims so a
// column-major GEMM stays valid: if n==1, ldc=m. For A, when transposed and m==1,
// lda=k; when not transposed and k==1, lda=m. For B, when transposed and k==1,
// ldb=n; when not transposed and n==1, ldb=k.
pub fn normalize_last_dims(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    lda: &mut i64,
    ldb: &mut i64,
    ldc: &mut i64,
) {
    if n == 1 {
        *ldc = m;
    }

    if transa != TransposeType::NoTranspose {
        if m == 1 {
            *lda = k;
        }
    } else if k == 1 {
        *lda = m;
    }

    if transb != TransposeType::NoTranspose {
        if k == 1 {
            *ldb = n;
        }
    } else if n == 1 {
        *ldb = k;
    }
}

// [spec:et:def:cpu-blas.executorch.cpublas.gemm-impl-fn]
// [spec:et:sem:cpu-blas.executorch.cpublas.gemm-impl-fn]
//
// Dispatch on the (transa, transb) pair to the four BlasKernel micro-kernels:
//  (No, No)   → gemm_notrans_
//  (T,  !T)   → gemm_transa_   (transa transposed, transb anything but Transpose)
//  (No, T)    → gemm_transb_
//  else (T,T) → gemm_transab_
// The C++ `gemm_impl<scalar_t, opmath_t, out_t>` is generic over the element,
// accumulator, and (widening) output types; here `S`/`Out` carry the same
// (`S::Op` is the accumulator `opmath_t`).
pub fn gemm_impl<S, Out>(
    transa: TransposeType,
    transb: TransposeType,
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
    if transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose {
        gemm_notrans_::<S, Out>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if transa == TransposeType::Transpose && transb != TransposeType::Transpose {
        gemm_transa_::<S, Out>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if transa == TransposeType::NoTranspose && transb == TransposeType::Transpose {
        gemm_transb_::<S, Out>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        // transa == Transpose && transb == Transpose
        gemm_transab_::<S, Out>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

/// Column-major (transa/transb, alpha/beta, lda/ldb/ldc) → `gemm` crate call.
///
/// DEVIATION (rust/PORTING.md optimized-kernels): stands in for the Eigen/CBLAS
/// `{s,d,c,z}gemm` FFI leaf. The BLAS convention `C := alpha*op(A)*op(B) + beta*C`
/// is expressed in the crate's `dst := alpha_g*dst + beta_g*lhs*rhs` form, so the
/// crate's `alpha` receives BLAS `beta` (the dst scale) and the crate's `beta`
/// receives BLAS `alpha` (the product scale). op(A)/op(B) transposition is encoded
/// in the row/col strides; ConjTranspose additionally sets the conj flag.
#[inline]
unsafe fn gemm_crate<T: 'static + Copy>(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: T,
    a: *const T,
    lda: i64,
    b: *const T,
    ldb: i64,
    beta: T,
    c: *mut T,
    ldc: i64,
    beta_is_zero: bool,
) {
    // op(A): m×k. NoTranspose => A is column-major m×k (rs=1, cs=lda).
    // (Conj)Transpose => underlying A is k×m column-major, op(A) reads it
    // transposed (rs=lda, cs=1).
    let (lhs_rs, lhs_cs, conj_lhs) = match transa {
        TransposeType::NoTranspose => (1isize, lda as isize, false),
        TransposeType::Transpose => (lda as isize, 1isize, false),
        TransposeType::ConjTranspose => (lda as isize, 1isize, true),
    };
    // op(B): k×n. NoTranspose => column-major k×n (rs=1, cs=ldb).
    let (rhs_rs, rhs_cs, conj_rhs) = match transb {
        TransposeType::NoTranspose => (1isize, ldb as isize, false),
        TransposeType::Transpose => (ldb as isize, 1isize, false),
        TransposeType::ConjTranspose => (ldb as isize, 1isize, true),
    };

    unsafe {
        gemm::gemm::<T>(
            m as usize,
            n as usize,
            k as usize,
            c,
            ldc as isize,  // dst_cs (column stride)
            1,             // dst_rs (row stride, column-major)
            !beta_is_zero, // read_dst: skip reading uninitialized C when beta == 0
            a,
            lhs_cs,
            lhs_rs,
            b,
            rhs_cs,
            rhs_rs,
            beta,  // alpha_g == BLAS beta (scale of dst)
            alpha, // beta_g == BLAS alpha (scale of product)
            false, // conj_dst
            conj_lhs,
            conj_rhs,
            Parallelism::None,
        );
    }
}

// gemm(double) — CPUBlas.cpp double overload.
// SUBSUMED: BLAS entry point routed to the gemm crate.
// [spec:et:def:cpu-blas.dgemm-fn]
// [spec:et:sem:cpu-blas.dgemm-fn]
pub fn gemm_f64(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: f64,
    a: *const f64,
    mut lda: i64,
    b: *const f64,
    mut ldb: i64,
    beta: f64,
    c: *mut f64,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // DEVIATION: Eigen/CBLAS dgemm → gemm crate.
    unsafe {
        gemm_crate::<f64>(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            beta == 0.0,
        );
    }
}

// [spec:et:def:cpu-blas.executorch.cpublas.gemm-fn]
// [spec:et:sem:cpu-blas.executorch.cpublas.gemm-fn]
//
// Single-precision GEMM: normalize the trailing leading-dims, then compute
// `C := alpha*op(A)*op(B) + beta*C` column-major. DEVIATION: the Eigen/CBLAS
// `sgemm` (or hand-written `gemm_impl`) compute is replaced by a `gemm` crate call.
// SUBSUMED: BLAS entry point routed to the gemm crate.
// [spec:et:def:cpu-blas.sgemm-fn]
// [spec:et:sem:cpu-blas.sgemm-fn]
pub fn gemm_f32(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const f32,
    mut lda: i64,
    b: *const f32,
    mut ldb: i64,
    beta: f32,
    c: *mut f32,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // DEVIATION: Eigen/CBLAS sgemm → gemm crate.
    unsafe {
        gemm_crate::<f32>(
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            beta == 0.0,
        );
    }
}

// gemm(Half) — always the hand-written gemm_impl path in the C++.
pub fn gemm_half(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: Half,
    a: *const Half,
    mut lda: i64,
    b: *const Half,
    mut ldb: i64,
    beta: Half,
    c: *mut Half,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // using acc_type = utils::compute_dtype<Half> == float
    gemm_impl::<Half, Half>(
        transa,
        transb,
        m,
        n,
        k,
        alpha.to_f32(),
        a,
        lda,
        b,
        ldb,
        beta.to_f32(),
        c,
        ldc,
    );
}

// gemm(BFloat16) — hand-written gemm_impl path.
pub fn gemm_bf16(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: BFloat16,
    a: *const BFloat16,
    mut lda: i64,
    b: *const BFloat16,
    mut ldb: i64,
    beta: BFloat16,
    c: *mut BFloat16,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // using acc_type = utils::compute_dtype<BFloat16> == float
    //
    // PORT-NOTE: the C++ BFloat16-in/out transa specialization
    // (gemm_transa_<BFloat16,BFloat16,BFloat16>) that dispatches to
    // bf16_dot_with_fp32_arith is a template specialization selected inside
    // gemm_impl. Rust monomorphization can't specialize on the concrete type
    // inside the generic gemm_impl, so the (Transpose, !Transpose) case is routed
    // to gemm_transa_bf16 explicitly here to preserve that behavior; the other
    // three cases match gemm_impl's dispatch exactly.
    if transa == TransposeType::Transpose && transb != TransposeType::Transpose {
        gemm_transa_bf16(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        gemm_impl::<BFloat16, BFloat16>(
            transa,
            transb,
            m,
            n,
            k,
            alpha.to_f32(),
            a,
            lda,
            b,
            ldb,
            beta.to_f32(),
            c,
            ldc,
        );
    }
}

// gemm(BFloat16 in, float out) — reduced-precision inputs accumulated into float.
pub fn gemm_bf16_f32(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const BFloat16,
    mut lda: i64,
    b: *const BFloat16,
    mut ldb: i64,
    beta: f32,
    c: *mut f32,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // gemm_impl<BFloat16, float, float>(...)
    gemm_impl::<BFloat16, f32>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// gemm(Half in, float out) — reduced-precision inputs accumulated into float.
pub fn gemm_half_f32(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: f32,
    a: *const Half,
    mut lda: i64,
    b: *const Half,
    mut ldb: i64,
    beta: f32,
    c: *mut f32,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // gemm_impl<Half, float, float>(...)
    gemm_impl::<Half, f32>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// gemm(complex<double>) — DEVIATION: zgemm FFI → gemm crate (c64).
// SUBSUMED: BLAS entry point routed to the gemm crate.
// [spec:et:def:cpu-blas.zgemm-fn]
// [spec:et:sem:cpu-blas.zgemm-fn]
pub fn gemm_c64(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: Complex<f64>,
    a: *const Complex<f64>,
    mut lda: i64,
    b: *const Complex<f64>,
    mut ldb: i64,
    beta: Complex<f64>,
    c: *mut Complex<f64>,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // DEVIATION: zgemm → gemm crate. Complex<f64> and gemm::c64 (num_complex
    // Complex64) share the same {re, im} f64 layout, so the buffers are
    // reinterpreted in place.
    let zero = beta.real == 0.0 && beta.imag == 0.0;
    unsafe {
        gemm_crate::<gemm::c64>(
            transa,
            transb,
            m,
            n,
            k,
            gemm::c64::new(alpha.real, alpha.imag),
            a as *const gemm::c64,
            lda,
            b as *const gemm::c64,
            ldb,
            gemm::c64::new(beta.real, beta.imag),
            c as *mut gemm::c64,
            ldc,
            zero,
        );
    }
}

// gemm(complex<float>) — DEVIATION: cgemm FFI → gemm crate (c32).
// SUBSUMED: BLAS entry point routed to the gemm crate.
// [spec:et:def:cpu-blas.cgemm-fn]
// [spec:et:sem:cpu-blas.cgemm-fn]
pub fn gemm_c32(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: Complex<f32>,
    a: *const Complex<f32>,
    mut lda: i64,
    b: *const Complex<f32>,
    mut ldb: i64,
    beta: Complex<f32>,
    c: *mut Complex<f32>,
    mut ldc: i64,
) {
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    // DEVIATION: cgemm → gemm crate. Complex<f32> and gemm::c32 (num_complex
    // Complex32) share the same {re, im} f32 layout.
    let zero = beta.real == 0.0 && beta.imag == 0.0;
    unsafe {
        gemm_crate::<gemm::c32>(
            transa,
            transb,
            m,
            n,
            k,
            gemm::c32::new(alpha.real, alpha.imag),
            a as *const gemm::c32,
            lda,
            b as *const gemm::c32,
            ldb,
            gemm::c32::new(beta.real, beta.imag),
            c as *mut gemm::c32,
            ldc,
            zero,
        );
    }
}

// gemm(complex<Half>) — hand-written gemm_impl path (no BLAS overload in C++).
//
// PORT-NOTE: complex<Half> is not exercised by the ported ops and the portable
// Complex<T> stand-in carries no arithmetic (portable_type/mod.rs), so a
// GemmScalar impl over Complex<Half> would have no meaning yet. The C++ overload
// is recorded here for completeness; wiring it needs a complex arithmetic type.

// Integer gemm (CPUBlas.h templated `gemm<T, is_integral>`): normalize dims then
// gemm_impl with acc_type == compute_dtype<T> (int32 for 8/16-bit, T for 32/64).
// `alpha`/`beta` are the accumulator type (`T::Op`), matching the C++
// `static_cast<const acc_type>(alpha)`.
pub fn gemm_int<T>(
    transa: TransposeType,
    transb: TransposeType,
    m: i64,
    n: i64,
    k: i64,
    alpha: <T as GemmScalar>::Op,
    a: *const T,
    mut lda: i64,
    b: *const T,
    mut ldb: i64,
    beta: <T as GemmScalar>::Op,
    c: *mut T,
    mut ldc: i64,
) where
    T: GemmScalar,
{
    normalize_last_dims(transa, transb, m, n, k, &mut lda, &mut ldb, &mut ldc);
    gemm_impl::<T, T>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#[cfg(test)]
mod tests {
    //! Ports kernels/optimized/test/libblas_test.cpp (BlasTest.MatmulOnes) plus
    //! fresh hand-computed small GEMMs that pin the column-major transpose /
    //! alpha-beta bookkeeping, `to_blas`, `normalize_last_dims`, and `gemm_impl`
    //! dispatch. Column-major convention throughout: logical element (i, j) of an
    //! `ld`-leading-dimension buffer lives at `buf[i + j*ld]`.
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * (1.0 + a.abs().max(b.abs()))
    }

    fn assert_close_slice(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(close(g, w), "index {i}: got {g}, want {w}");
        }
    }

    // Logical row-major 2x2 [[a,b],[c,d]] -> column-major buffer [a,c,b,d].
    const A: [f32; 4] = [1.0, 3.0, 2.0, 4.0]; // [[1,2],[3,4]]
    const B: [f32; 4] = [5.0, 7.0, 6.0, 8.0]; // [[5,6],[7,8]]

    // C++ BlasTest.MatmulOnes: N*N all-ones @ all-ones (NoTrans, alpha=1, beta=0)
    // yields every output element == N. Ported for the crate-backed float/double
    // overloads and the reduced-precision Half/BFloat16 overloads.
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    // [spec:et:sem:cpu-blas.executorch.cpublas.gemm-fn/test]
    // [spec:et:sem:cpu-blas.dgemm-fn/test]
    #[test]
    fn blas_test_matmul_ones() {
        const N: i64 = 25;
        let nn = (N * N) as usize;

        let in1 = vec![1.0f32; nn];
        let in2 = vec![1.0f32; nn];
        let mut out = vec![0.0f32; nn];
        gemm_f32(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            N,
            N,
            N,
            1.0,
            in1.as_ptr(),
            N,
            in2.as_ptr(),
            N,
            0.0,
            out.as_mut_ptr(),
            N,
        );
        assert!(out.iter().all(|&v| close(v, N as f32)));

        let in1d = vec![1.0f64; nn];
        let in2d = vec![1.0f64; nn];
        let mut outd = vec![0.0f64; nn];
        gemm_f64(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            N,
            N,
            N,
            1.0,
            in1d.as_ptr(),
            N,
            in2d.as_ptr(),
            N,
            0.0,
            outd.as_mut_ptr(),
            N,
        );
        assert!(outd.iter().all(|&v| (v - N as f64).abs() < 1e-6));

        let in1h = vec![Half::from_f32(1.0); nn];
        let in2h = vec![Half::from_f32(1.0); nn];
        let mut outh = vec![Half::from_f32(0.0); nn];
        gemm_half(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            N,
            N,
            N,
            Half::from_f32(1.0),
            in1h.as_ptr(),
            N,
            in2h.as_ptr(),
            N,
            Half::from_f32(0.0),
            outh.as_mut_ptr(),
            N,
        );
        assert!(outh.iter().all(|&v| close(v.to_f32(), N as f32)));

        let in1b = vec![BFloat16::from_f32(1.0); nn];
        let in2b = vec![BFloat16::from_f32(1.0); nn];
        let mut outb = vec![BFloat16::from_f32(0.0); nn];
        gemm_bf16(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            N,
            N,
            N,
            BFloat16::from_f32(1.0),
            in1b.as_ptr(),
            N,
            in2b.as_ptr(),
            N,
            BFloat16::from_f32(0.0),
            outb.as_mut_ptr(),
            N,
        );
        // BFloat16 has only 8 mantissa bits; N == 25 is exactly representable.
        assert!(outb.iter().all(|&v| (v.to_f32() - N as f32).abs() < 0.5));
    }

    // NoTranspose x NoTranspose, alpha=1 beta=0: C = A@B = [[19,22],[43,50]]
    // column-major -> [19,43,22,50]. Exercises sgemm entry + crate leaf.
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    // [spec:et:sem:cpu-blas.executorch.cpublas.gemm-fn/test]
    #[test]
    fn gemm_f32_notrans_notrans() {
        let mut c = [0.0f32; 4];
        gemm_f32(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            1.0,
            A.as_ptr(),
            2,
            B.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_close_slice(&c, &[19.0, 43.0, 22.0, 50.0]);
    }

    // Transpose A: C = A^T @ B = [[26,30],[38,44]] col-major -> [26,38,30,44].
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    #[test]
    fn gemm_f32_transa() {
        let mut c = [0.0f32; 4];
        gemm_f32(
            TransposeType::Transpose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            1.0,
            A.as_ptr(),
            2,
            B.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_close_slice(&c, &[26.0, 38.0, 30.0, 44.0]);
    }

    // Transpose B: C = A @ B^T = [[17,23],[39,53]] col-major -> [17,39,23,53].
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    #[test]
    fn gemm_f32_transb() {
        let mut c = [0.0f32; 4];
        gemm_f32(
            TransposeType::NoTranspose,
            TransposeType::Transpose,
            2,
            2,
            2,
            1.0,
            A.as_ptr(),
            2,
            B.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_close_slice(&c, &[17.0, 39.0, 23.0, 53.0]);
    }

    // Transpose A and B: C = A^T @ B^T = [[23,31],[34,46]] col-major -> [23,34,31,46].
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    #[test]
    fn gemm_f32_transab() {
        let mut c = [0.0f32; 4];
        gemm_f32(
            TransposeType::Transpose,
            TransposeType::Transpose,
            2,
            2,
            2,
            1.0,
            A.as_ptr(),
            2,
            B.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_close_slice(&c, &[23.0, 34.0, 31.0, 46.0]);
    }

    // alpha/beta blend: C0 = [[10,20],[30,40]], alpha=2 beta=3, NoTrans:
    // 2*[[19,22],[43,50]] + 3*[[10,20],[30,40]] = [[68,104],[176,220]]
    // col-major -> [68,176,104,220]. Pins the crate alpha/beta swap bookkeeping.
    // [spec:et:sem:cpu-blas.sgemm-fn/test]
    // [spec:et:sem:cpu-blas.executorch.cpublas.gemm-fn/test]
    #[test]
    fn gemm_f32_alpha_beta() {
        let mut c = [10.0f32, 30.0, 20.0, 40.0];
        gemm_f32(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            2.0,
            A.as_ptr(),
            2,
            B.as_ptr(),
            2,
            3.0,
            c.as_mut_ptr(),
            2,
        );
        assert_close_slice(&c, &[68.0, 176.0, 104.0, 220.0]);
    }

    // f64 identity: I @ B = B. Identity (2x2) col-major = [1,0,0,1].
    // [spec:et:sem:cpu-blas.dgemm-fn/test]
    #[test]
    fn gemm_f64_identity() {
        let id = [1.0f64, 0.0, 0.0, 1.0];
        let b = [5.0f64, 7.0, 6.0, 8.0];
        let mut c = [0.0f64; 4];
        gemm_f64(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            1.0,
            id.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
        assert_eq!(c, b);
    }

    // complex<f32> GEMM. A=[[1+1i,0],[0,1]], B=[[i,0],[0,2]] -> A@B=[[i-1,0],[0,2]].
    // In column-major: A buf = [1+1i,0, 0,1]; B buf = [i,0, 0,2].
    // C(0,0)=(1+i)*i = -1+i; C(1,1)=1*2=2; off-diagonals 0.
    // [spec:et:sem:cpu-blas.cgemm-fn/test]
    #[test]
    fn gemm_c32_diagonal() {
        let a = [
            Complex {
                real: 1.0f32,
                imag: 1.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 1.0,
                imag: 0.0,
            },
        ];
        let b = [
            Complex {
                real: 0.0f32,
                imag: 1.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 2.0,
                imag: 0.0,
            },
        ];
        let mut c = [Complex {
            real: 0.0f32,
            imag: 0.0,
        }; 4];
        gemm_c32(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            Complex {
                real: 1.0,
                imag: 0.0,
            },
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            c.as_mut_ptr(),
            2,
        );
        assert!(close(c[0].real, -1.0) && close(c[0].imag, 1.0));
        assert!(close(c[1].real, 0.0) && close(c[1].imag, 0.0));
        assert!(close(c[2].real, 0.0) && close(c[2].imag, 0.0));
        assert!(close(c[3].real, 2.0) && close(c[3].imag, 0.0));
    }

    // complex<f64> ConjTranspose on A: op(A)=conj(A)^T. With A=[[1+2i,0],[0,1]],
    // conj(A)^T = [[1-2i,0],[0,1]]; times B=[[1,0],[0,1]] (identity) -> [[1-2i,0],[0,1]].
    // [spec:et:sem:cpu-blas.zgemm-fn/test]
    #[test]
    fn gemm_c64_conj_transpose() {
        let a = [
            Complex {
                real: 1.0f64,
                imag: 2.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 1.0,
                imag: 0.0,
            },
        ];
        let id = [
            Complex {
                real: 1.0f64,
                imag: 0.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            Complex {
                real: 1.0,
                imag: 0.0,
            },
        ];
        let mut c = [Complex {
            real: 0.0f64,
            imag: 0.0,
        }; 4];
        gemm_c64(
            TransposeType::ConjTranspose,
            TransposeType::NoTranspose,
            2,
            2,
            2,
            Complex {
                real: 1.0,
                imag: 0.0,
            },
            a.as_ptr(),
            2,
            id.as_ptr(),
            2,
            Complex {
                real: 0.0,
                imag: 0.0,
            },
            c.as_mut_ptr(),
            2,
        );
        assert!(close(c[0].real as f32, 1.0) && close(c[0].imag as f32, -2.0));
        assert!(close(c[3].real as f32, 1.0) && close(c[3].imag as f32, 0.0));
    }

    // to_blas: Transpose->'T', NoTranspose->'N', ConjTranspose->'C'.
    // (to_cblas_transpose is SUBSUMED into to_blas by the wave-2 DEVIATION note,
    // so its facet is exercised by the same transpose-character mapping.)
    // [spec:et:sem:cpu-blas.executorch.cpublas.to-blas-fn/test]
    // [spec:et:sem:cpu-blas.executorch.cpublas.to-cblas-transpose-fn/test]
    #[test]
    fn to_blas_maps_transpose_chars() {
        assert_eq!(to_blas(TransposeType::Transpose), b'T');
        assert_eq!(to_blas(TransposeType::NoTranspose), b'N');
        assert_eq!(to_blas(TransposeType::ConjTranspose), b'C');
    }

    // normalize_last_dims degenerate-dimension patches (CPUBlas.cpp lines 63-86):
    //  - n==1        -> ldc = m
    //  - transa!=No, m==1 -> lda = k ; else (transa==No) k==1 -> lda = m
    //  - transb!=No, k==1 -> ldb = n ; else (transb==No) n==1 -> ldb = k
    // [spec:et:sem:cpu-blas.executorch.cpublas.normalize-last-dims-fn/test]
    #[test]
    fn normalize_last_dims_patches() {
        // n == 1: ldc patched to m; transb==NoTranspose & n==1 -> ldb patched to k.
        let (mut lda, mut ldb, mut ldc) = (99i64, 99, 99);
        normalize_last_dims(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            7,
            1,
            5,
            &mut lda,
            &mut ldb,
            &mut ldc,
        );
        assert_eq!(ldc, 7); // n==1 -> ldc = m
        assert_eq!(ldb, 5); // transb==No && n==1 -> ldb = k
        assert_eq!(lda, 99); // transa==No, k!=1 -> untouched

        // transa==No & k==1: lda = m. n!=1 so ldc/ldb untouched.
        let (mut lda, mut ldb, mut ldc) = (11i64, 22, 33);
        normalize_last_dims(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            4,
            6,
            1,
            &mut lda,
            &mut ldb,
            &mut ldc,
        );
        assert_eq!(lda, 4); // k==1 -> lda = m
        assert_eq!(ldb, 22);
        assert_eq!(ldc, 33);

        // transa==Transpose & m==1: lda = k. transb==Transpose & k==1: ldb = n.
        let (mut lda, mut ldb, mut ldc) = (0i64, 0, 44);
        normalize_last_dims(
            TransposeType::Transpose,
            TransposeType::Transpose,
            1,
            9,
            1,
            &mut lda,
            &mut ldb,
            &mut ldc,
        );
        assert_eq!(lda, 1); // m==1 -> lda = k
        assert_eq!(ldb, 9); // transb!=No && k==1 -> ldb = n
        assert_eq!(ldc, 44); // n!=1 -> untouched
    }

    // gemm_impl dispatch: drive the four (transa, transb) branches directly on
    // f32 (same-type, non-reduced) and confirm each routes to the right micro-
    // kernel by matching the hand-computed products above. Pre-zeroed C, alpha=1.
    // [spec:et:sem:cpu-blas.executorch.cpublas.gemm-impl-fn/test]
    #[test]
    fn gemm_impl_dispatches_all_transpose_pairs() {
        let run = |ta, tb| {
            let mut c = [0.0f32; 4];
            gemm_impl::<f32, f32>(
                ta,
                tb,
                2,
                2,
                2,
                1.0,
                A.as_ptr(),
                2,
                B.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );
            c
        };
        assert_close_slice(
            &run(TransposeType::NoTranspose, TransposeType::NoTranspose),
            &[19.0, 43.0, 22.0, 50.0],
        );
        assert_close_slice(
            &run(TransposeType::Transpose, TransposeType::NoTranspose),
            &[26.0, 38.0, 30.0, 44.0],
        );
        assert_close_slice(
            &run(TransposeType::NoTranspose, TransposeType::Transpose),
            &[17.0, 39.0, 23.0, 53.0],
        );
        assert_close_slice(
            &run(TransposeType::Transpose, TransposeType::Transpose),
            &[23.0, 34.0, 31.0, 46.0],
        );
    }
}
