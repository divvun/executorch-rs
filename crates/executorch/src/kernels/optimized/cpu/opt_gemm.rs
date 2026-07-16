//! Per-element-type GEMM dispatch for the optimized linear/mm/bmm ops.
//!
//! PORT-NOTE (codegen/overload deviation): in C++ the optimized ops call the
//! overloaded free function `executorch::cpublas::gemm(...)`, and the compiler
//! selects the correct per-dtype overload inside the `ET_SWITCH_*` body. Rust has
//! no overload resolution inside a monomorphized dtype-switch arm, so this module
//! collapses that overload set into the `OptGemm` trait: one method per element
//! type forwarding to the matching `CPUBlas::gemm_*` entry point, plus `one()` /
//! `zero()` producers for the `static_cast<CTYPE>(1)` / `static_cast<CTYPE>(0)`
//! alpha/beta the callers pass. This is a thin dispatch shim, not new behavior —
//! the column-major / transpose / alpha / beta bookkeeping stays in CPUBlas.

use crate::kernels::optimized::blas::BlasKernel::GemmScalar;
use crate::kernels::optimized::blas::CPUBlas::{
    TransposeType, gemm_bf16, gemm_f32, gemm_f64, gemm_half, gemm_int,
};
use crate::runtime::core::portable_type::{BFloat16, Half};

/// Element-type GEMM dispatch. `opt_gemm` mirrors the C++ overloaded
/// `executorch::cpublas::gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb,
/// beta, c, ldc)` for a single `Self` element type; `alpha`/`beta` are the
/// CTYPE-valued scale factors the ops construct with `static_cast<CTYPE>(1|0)`.
pub trait OptGemm: Copy {
    /// `static_cast<CTYPE>(1)`.
    fn one() -> Self;
    /// `static_cast<CTYPE>(0)`.
    fn zero() -> Self;

    /// # Safety
    /// `a`/`b`/`c` must point to valid column-major m×k / k×n / m×n blocks with
    /// the given leading dimensions, exactly as the C++ `cpublas::gemm` requires.
    #[allow(clippy::too_many_arguments)]
    unsafe fn opt_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    );
}

impl OptGemm for f32 {
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
    unsafe fn opt_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) {
        gemm_f32(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl OptGemm for f64 {
    fn one() -> Self {
        1.0
    }
    fn zero() -> Self {
        0.0
    }
    unsafe fn opt_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) {
        gemm_f64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl OptGemm for Half {
    fn one() -> Self {
        Half::from_f32(1.0)
    }
    fn zero() -> Self {
        Half::from_f32(0.0)
    }
    unsafe fn opt_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) {
        gemm_half(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

impl OptGemm for BFloat16 {
    fn one() -> Self {
        BFloat16::from_f32(1.0)
    }
    fn zero() -> Self {
        BFloat16::from_f32(0.0)
    }
    unsafe fn opt_gemm(
        transa: TransposeType,
        transb: TransposeType,
        m: i64,
        n: i64,
        k: i64,
        alpha: Self,
        a: *const Self,
        lda: i64,
        b: *const Self,
        ldb: i64,
        beta: Self,
        c: *mut Self,
        ldc: i64,
    ) {
        gemm_bf16(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

// Integer element types (u8/i8/i16/i32/i64) route through the templated integer
// `cpublas::gemm<T>`, which takes the accumulator-typed (`T::Op`) alpha/beta; the
// CTYPE-valued `static_cast<CTYPE>(1|0)` is widened via `GemmScalar::to_op`,
// matching the C++ `static_cast<const acc_type>(alpha)`.
macro_rules! impl_opt_gemm_int {
    ($($t:ty),*) => {$(
        impl OptGemm for $t {
            fn one() -> Self { 1 as $t }
            fn zero() -> Self { 0 as $t }
            unsafe fn opt_gemm(
                transa: TransposeType,
                transb: TransposeType,
                m: i64,
                n: i64,
                k: i64,
                alpha: Self,
                a: *const Self,
                lda: i64,
                b: *const Self,
                ldb: i64,
                beta: Self,
                c: *mut Self,
                ldc: i64,
            ) {
                gemm_int::<$t>(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    <$t as GemmScalar>::to_op(alpha),
                    a,
                    lda,
                    b,
                    ldb,
                    <$t as GemmScalar>::to_op(beta),
                    c,
                    ldc,
                );
            }
        }
    )*};
}
impl_opt_gemm_int!(u8, i8, i16, i32, i64);
