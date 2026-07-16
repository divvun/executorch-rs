# kernels/optimized/blas/CPUBlas.cpp, kernels/optimized/blas/CPUBlas.h

> [spec:et:def:cpu-blas.cgemm-fn]
> void cgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc)

> [spec:et:sem:cpu-blas.cgemm-fn]
> Fortran BLAS single-precision-complex GEMM (`extern "C" cgemm_`). All
> arguments are passed by pointer (Fortran calling convention). Computes
> C := alpha*op(A)*op(B) + beta*C on column-major complex<float> matrices,
> where op(X) is X, X.T, or conj(X.T) per the transa/transb characters
> ('N'/'T'/'C'). m,n,k are the op dimensions; lda,ldb,ldc the leading
> dimensions. DEVIATION (rust/PORTING.md optimized-kernels): the external
> BLAS FFI is not ported; the complex<float> gemm entry point instead calls
> the `gemm` crate (`gemm::c32`).

> [spec:et:def:cpu-blas.dgemm-fn]
> void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, const double *a, int *lda, const double *b, int *ldb, double *beta, double *c, int *ldc)

> [spec:et:sem:cpu-blas.dgemm-fn]
> Fortran BLAS double-precision GEMM (`extern "C" dgemm_`), all args by
> pointer. Computes C := alpha*op(A)*op(B) + beta*C on column-major f64
> matrices; op per the transa/transb characters. DEVIATION: not ported; the
> f64 gemm entry point calls the `gemm` crate.

> [spec:et:def:cpu-blas.executorch.cpublas.gemm-fn]
> void gemm( TransposeType transa, TransposeType transb, int64_t m, int64_t n, int64_t k, const float alpha, const float *a, int64_t lda, const float *b, int64_t ldb, const float beta, float *c, int64_t ldc)

> [spec:et:sem:cpu-blas.executorch.cpublas.gemm-fn]
> The public single-precision (float) GEMM. First calls
> `normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc)` to patch
> the leading dimensions for degenerate size-1 trailing dims. Then computes
> C := alpha*op(A)*op(B) + beta*C on column-major float matrices, where
> op(A)/op(B) apply the transa/transb transpose. In the C++ the inner compute
> is either a CBLAS/Eigen `sgemm` (when built with BLAS) or the hand-written
> `gemm_impl` (with acc_type = compute_dtype<float> = float). DEVIATION
> (rust/PORTING.md optimized-kernels): the inner compute is replaced by a
> single `gemm` crate call. The overload set covers f64, f32, Half, BFloat16,
> Half/BFloat16-in-float-out, complex<float>, complex<double>, complex<Half>,
> and an integral template; each first normalizes dims then computes the same
> GEMM with acc_type = compute_dtype<scalar_t>.

> [spec:et:def:cpu-blas.executorch.cpublas.gemm-impl-fn]
> void gemm_impl( TransposeType transa, TransposeType transb, int64_t m, int64_t n, int64_t k, opmath_t alpha, const scalar_t *a, int64_t lda, const scalar_t *b, int64_t ldb, opmath_t beta, out_t *c, int64_t ldc)

> [spec:et:sem:cpu-blas.executorch.cpublas.gemm-impl-fn]
> Template<scalar_t, opmath_t, out_t> dispatcher over the (transa, transb)
> pair to the four BlasKernel micro-kernels:
>   - (NoTranspose, NoTranspose) -> gemm_notrans_  (and returns its result)
>   - (Transpose, transb != Transpose) -> gemm_transa_
>   - (NoTranspose, Transpose) -> gemm_transb_
>   - else (Transpose, Transpose) -> gemm_transab_
> Forwards m,n,k,alpha,a,lda,b,ldb,beta,c,ldc unchanged. No normalization is
> done here (the caller already ran normalize_last_dims).

> [spec:et:def:cpu-blas.executorch.cpublas.normalize-last-dims-fn]
> void normalize_last_dims( TransposeType transa, TransposeType transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)

> [spec:et:sem:cpu-blas.executorch.cpublas.normalize-last-dims-fn]
> Patches lda/ldb/ldc in place so a column-major GEMM stays valid when a
> trailing dimension is 1:
>   - if n == 1: *ldc = m.
>   - for A: if transa != NoTranspose and m == 1: *lda = k;
>            else if (transa == NoTranspose and) k == 1: *lda = m.
>   - for B: if transb != NoTranspose and k == 1: *ldb = n;
>            else if (transb == NoTranspose and) n == 1: *ldb = k.
> (The `else if` guards mean the k==1/n==1 branches only apply in the
> NoTranspose case.)

> [spec:et:def:cpu-blas.executorch.cpublas.to-blas-fn]
> inline char to_blas(TransposeType trans)

> [spec:et:sem:cpu-blas.executorch.cpublas.to-blas-fn]
> Map a TransposeType to its BLAS transpose character: Transpose -> 'T',
> NoTranspose -> 'N', ConjTranspose -> 'C'. Returns 'N' by default.

> [spec:et:def:cpu-blas.executorch.cpublas.to-cblas-transpose-fn]
> inline CBLAS_TRANSPOSE to_cblas_transpose(TransposeType trans)

> [spec:et:sem:cpu-blas.executorch.cpublas.to-cblas-transpose-fn]
> Apple-Accelerate-only (ET_BUILD_FOR_APPLE) helper mapping a TransposeType
> to a CBLAS_TRANSPOSE: Transpose -> CblasTrans, NoTranspose -> CblasNoTrans,
> ConjTranspose -> CblasConjTrans; defaults to CblasNoTrans. DEVIATION: not
> ported (the CBLAS FFI path is replaced by the `gemm` crate).

> [spec:et:def:cpu-blas.executorch.cpublas.transpose-type]
> enum class TransposeType {
>   NoTranspose;
>   Transpose;
>   ConjTranspose;
> }

> [spec:et:def:cpu-blas.sgemm-fn]
> void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, const float *a, int *lda, const float *b, int *ldb, float *beta, float *c, int *ldc)

> [spec:et:sem:cpu-blas.sgemm-fn]
> Fortran BLAS single-precision GEMM (`extern "C" sgemm_`), all args by
> pointer. Computes C := alpha*op(A)*op(B) + beta*C on column-major f32
> matrices; op per the transa/transb characters. DEVIATION: not ported; the
> f32 gemm entry point calls the `gemm` crate.

> [spec:et:def:cpu-blas.zgemm-fn]
> void zgemm_(char *transa, char *transb, int *m, int *n, int *k, void *alpha, const void *a, int *lda, const void *b, int *ldb, void *beta, void *c, int *ldc)

> [spec:et:sem:cpu-blas.zgemm-fn]
> Fortran BLAS double-precision-complex GEMM (`extern "C" zgemm_`), all args
> by pointer. Computes C := alpha*op(A)*op(B) + beta*C on column-major
> complex<double> matrices; op ('N'/'T'/'C') per transa/transb. DEVIATION:
> not ported; the complex<double> gemm entry point calls the `gemm` crate
> (`gemm::c64`).

