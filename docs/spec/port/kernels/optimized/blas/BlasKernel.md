# kernels/optimized/blas/BlasKernel.cpp, kernels/optimized/blas/BlasKernel.h

> [spec:et:def:blas-kernel.executorch.cpublas.gemm-notrans-fn]
> typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type gemm_notrans_( int64_t m, int64_t n, int64_t k, opmath_t alpha, const scalar_t* a, int64_t lda, const scalar_t* b, int64_t ldb, opmath_t beta, scalar_t* c, int6...

> [spec:et:sem:blas-kernel.executorch.cpublas.gemm-notrans-fn]
> Column-major GEMM with no transpose: C := beta*C + alpha*(A @ B), A is m×k
> (col stride lda), B is k×n (col stride ldb), C is m×n (col stride ldc). Two
> overloads select on std::is_same<scalar_t, opmath_t>:
>   - same-type (scalar_t == opmath_t): first `scale_(m, n, beta, c, ldc)`
>     scales C by beta in place. Then a rank-1 update: for l in 0..k, for j in
>     0..n, compute val = b[l + j*ldb] * alpha (opmath_t); update column j of C
>     with a manual 4-wide unroll over i (i_m = m/4 blocks of
>     c[j*ldc + i*4+r] += a[i*4+r + l*lda] * val for r in 0..4) plus a scalar
>     tail for the remaining m%4 rows.
>   - reduced-precision (scalar_t != opmath_t; Half/BFloat16, out_t may widen
>     to float): for i in 0..m, for j in 0..n, dot = sum over l of
>     (opmath_t)a[l*lda + i] * (opmath_t)b[j*ldb + l] (via the ILP `sum`
>     helper); if beta == 0 then c[j*ldc+i] = (out_t)(alpha*dot) else
>     c[j*ldc+i] = (out_t)(beta*(opmath_t)c[j*ldc+i] + alpha*dot).

> [spec:et:def:blas-kernel.executorch.cpublas.gemm-transa-fn]
> void gemm_transa_( int64_t m, int64_t n, int64_t k, opmath_t alpha, const scalar_t *a, int64_t lda, const scalar_t *b, int64_t ldb, opmath_t beta, out_t *c, int64_t ldc)

> [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-fn]
> Column-major GEMM with A transposed: C := beta*C + alpha*(A.T @ B). Walks A
> row-by-row via a pointer a_ advanced by lda each outer i (0..m) and B
> column-by-column via a pointer b_ advanced by ldb each inner j (0..n). For
> each (i,j): dot = sum over l in 0..k of (opmath_t)a_[l] * (opmath_t)b_[l]
> (ILP `sum` helper); if beta == 0 then c[j*ldc+i] = (out_t)(alpha*dot) else
> c[j*ldc+i] = (out_t)(beta*(opmath_t)c[j*ldc+i] + alpha*dot).

> [spec:et:def:blas-kernel.executorch.cpublas.gemm-transa-torch-executor-b-float16-torch-executor-b-float16-torch-executor-b-float16-fn]
> inline void gemm_transa_<torch::executor::BFloat16, torch::executor::BFloat16, torch::executor::BFloat16>( int64_t m, int64_t n, int64_t k, torch::executor::BFloat16 alpha, const torch::executor::BFloat16 *a, int64_t lda, const torch::ex...

> [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transa-torch-executor-b-float16-torch-executor-b-float16-torch-executor-b-float16-fn]
> BFloat16-in/BFloat16-out template specialization of gemm_transa_. Computes
> C := beta*C + alpha*(A.T @ B), parallelizing over rows i via
> extension::parallel_for(0, m, 1, ...): each chunk [begin,end) walks a_ = a +
> begin*lda (advanced by lda per i) and, per i, b_ = b (advanced by ldb per
> j in 0..n) computing dot = internal::bf16_dot_with_fp32_arith(a_, b_, k)
> (float32 accumulation). Fast case alpha == 1 && beta == 0: c[j*ldc+i] = dot.
> Otherwise (second parallel_for): if beta == 0 then c[j*ldc+i] = alpha*dot
> else c[j*ldc+i] = beta*c[j*ldc+i] + alpha*dot. DEVIATION: the SIMD dot leaf
> is a scalar float loop (see bf16-dot-with-fp32-arith-fn).

> [spec:et:def:blas-kernel.executorch.cpublas.gemm-transab-fn]
> void gemm_transab_( int64_t m, int64_t n, int64_t k, opmath_t alpha, const scalar_t *a, int64_t lda, const scalar_t *b, int64_t ldb, opmath_t beta, out_t *c, int64_t ldc)

> [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transab-fn]
> Column-major GEMM with both transposed: C := beta*C + alpha*(A.T @ B.T).
> For i in 0..m, for j in 0..n: dot = sum over l in 0..k of
> (opmath_t)a[i*lda + l] * (opmath_t)b[l*ldb + j] (ILP `sum` helper); if
> beta == 0 then c[j*ldc+i] = (out_t)(alpha*dot) else
> c[j*ldc+i] = (out_t)(beta*(opmath_t)c[j*ldc+i] + alpha*dot).

> [spec:et:def:blas-kernel.executorch.cpublas.gemm-transb-fn]
> typename std::enable_if<std::is_same<scalar_t, opmath_t>::value, void>::type gemm_transb_( int64_t m, int64_t n, int64_t k, opmath_t alpha, const scalar_t* a, int64_t lda, const scalar_t* b, int64_t ldb, opmath_t beta, scalar_t* c, int64...

> [spec:et:sem:blas-kernel.executorch.cpublas.gemm-transb-fn]
> Column-major GEMM with B transposed: C := beta*C + alpha*(A @ B.T). Two
> overloads select on std::is_same<scalar_t, opmath_t>:
>   - same-type: `scale_(m, n, beta, c, ldc)` then a rank-1 update: for l in
>     0..k, for j in 0..n, val = b[j + l*ldb] * alpha (B read transposed);
>     column-j update of C with a 4-wide unroll over i (c[j*ldc + i*4+r] +=
>     a[i*4+r + l*lda] * val) plus scalar tail for m%4.
>   - reduced-precision: for i in 0..m, for j in 0..n, dot = sum over l of
>     (opmath_t)a[l*lda + i] * (opmath_t)b[l*ldb + j]; if beta == 0 then
>     c[j*ldc+i] = (out_t)(alpha*dot) else c[j*ldc+i] =
>     (out_t)(beta*(opmath_t)c[j*ldc+i] + alpha*dot).

> [spec:et:def:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn]
> float bf16_dot_with_fp32_arith( const at::BFloat16* vec1, const at::BFloat16* vec2, int64_t len)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.bf16-dot-with-fp32-arith-fn]
> Dot product of two length-`len` BFloat16 vectors accumulated in float32.
> C++ dispatch: if COMPILER_SUPPORTS_BF16_TARGET and cpuinfo_has_arm_bf16(),
> use dot_with_fp32_arith_bfdot (ARM BFDOT), else dot_with_fp32_arith_no_bfdot
> (Vectorized<float> reduction). Both perform a register-blocked main loop
> plus first-tier vectorized tail plus a second-tier scalar tail that promotes
> each element to float and accumulates x1*x2; the mathematical result is the
> float32 sum over j of (float)vec1[j] * (float)vec2[j]. DEVIATION
> (rust/PORTING.md optimized-kernels): the Rust port implements exactly that
> scalar float accumulation loop, collapsing the SIMD variants (the IntegerLog2
> / reduce / ForcedUnrollTargetBFloat16 / fmadd / bfdot inner-loop helpers exist
> only to drive them and are not ported separately).

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-bfdot-fn]
> TARGET_ARM_BF16_ATTRIBUTE float dot_with_fp32_arith_bfdot( const BFloat16* vec1, const BFloat16* vec2, int64_t len)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-bfdot-fn]
> ARM-BF16 (bfdot) variant of the bf16 dot product: reduced_sum =
> dot_with_fp32_arith_main_loop_bfdot(vec1, vec2, len), then the shared tail
> macro (first-tier vectorized tail over Vectorized<BFloat16>::size()-aligned
> remainder, reduce into reduced_sum, then second-tier scalar tail promoting
> each remaining element to float and adding x1*x2). Returns reduced_sum.
> DEVIATION: subsumed by bf16_dot_with_fp32_arith's scalar port; not ported
> as a separate item.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-bfdot-fn]
> TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_bfdot( const BFloat16* vec1, const BFloat16* vec2, vec::VectorizedN<float, kF32RegistersPerIteration>& sum, int registerPairIndex)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-bfdot-fn]
> One unrolled register-pair step of the bfdot main loop: loads 8 BFloat16
> lanes from vec1 and vec2 at offset registerPairIndex*8 (via vld1q_bf16) and
> accumulates sum[registerPairIndex] = vbfdotq_f32(sum[registerPairIndex],
> temp_vec1, temp_vec2) (ARM BFDOT into 4 float32 lanes). DEVIATION: subsumed
> by the scalar bf16 dot port; not ported separately.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-no-bfdot-fn]
> C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot( const T* vec1, const T* vec2, vec::VectorizedN<float, kF32RegistersPerIteration>& sum, int registerPairIndex)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-inner-loop-no-bfdot-fn]
> Non-bfdot unrolled register-pair step (static_assert T == BFloat16): loads
> one Vectorized<BFloat16> from vec1 and vec2 at offset registerPairIndex*size,
> then fmadd converts each to a low/high pair of Vectorized<float> and does
> sum[2*idx], sum[2*idx+1] += a_lo*b_lo, a_hi*b_hi. DEVIATION: subsumed by the
> scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-bfdot-fn]
> C10_ALWAYS_INLINE TARGET_ARM_BF16_ATTRIBUTE auto dot_with_fp32_arith_main_loop_bfdot( const BFloat16* vec1, const BFloat16* vec2, int64_t len)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-bfdot-fn]
> bfdot main loop: sum = VectorizedN<float, kF32RegistersPerIteration>(0);
> len_aligned = len & ~(kF32ElementsPerIteration - 1); for j from 0 to
> len_aligned step kF32ElementsPerIteration, ForcedUnrollTargetBFloat16<
> kF32RegisterPairsPerIteration> runs dot_with_fp32_arith_main_inner_loop_bfdot
> over vec1+j, vec2+j. Returns reduce(sum). DEVIATION: subsumed by the scalar
> bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-no-bfdot-fn]
> C10_ALWAYS_INLINE auto dot_with_fp32_arith_main_loop_no_bfdot( const T* vec1, const T* vec2, int64_t len)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-main-loop-no-bfdot-fn]
> Non-bfdot main loop: sum = VectorizedN<float, kF32RegistersPerIteration>(0);
> len_aligned = len & ~(kF32ElementsPerIteration - 1); for j from 0 to
> len_aligned step kF32ElementsPerIteration, ForcedUnroll<
> kF32RegisterPairsPerIteration> runs dot_with_fp32_arith_main_inner_loop_no_bfdot
> over vec1+j, vec2+j. Returns reduce(sum). DEVIATION: subsumed by the scalar
> bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-no-bfdot-fn]
> C10_ALWAYS_INLINE float dot_with_fp32_arith_no_bfdot(const T* vec1, const T* vec2, int64_t len)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-no-bfdot-fn]
> Non-bfdot bf16 dot: reduced_sum = dot_with_fp32_arith_main_loop_no_bfdot(
> vec1, vec2, len), then the shared tail macro (first-tier vectorized tail,
> reduce into reduced_sum, second-tier scalar tail adding float x1*x2 per
> remaining element). Returns reduced_sum. DEVIATION: subsumed by the scalar
> bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-bfdot-fn]
> TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_bfdot( const at::BFloat16* vec1, const at::BFloat16* vec2, vec::Vectorized<float>* tail_sum, int idx)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-bfdot-fn]
> One first-tier tail step (bfdot): loads 8 BFloat16 lanes from vec1[idx],
> vec2[idx] (vld1q_bf16) and *tail_sum = vbfdotq_f32(*tail_sum, temp_vec1,
> temp_vec2). DEVIATION: subsumed by the scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-no-bfdot-fn]
> C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot( const T* vec1, const T* vec2, vec::Vectorized<float>* tail_sum, int idx)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.dot-with-fp32-arith-vectorized-tail-inner-loop-no-bfdot-fn]
> One first-tier tail step (non-bfdot): loads a Vectorized<T> from vec1[idx],
> vec2[idx] and *tail_sum = fmadd(*tail_sum, temp_vec1, temp_vec2) (convert
> bf16->float pairs and accumulate). DEVIATION: subsumed by the scalar bf16
> dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.fmadd-fn]
> [[maybe_unused]] std::pair<vec::Vectorized<float>, vec::Vectorized<float>> fmadd( const vec::Vectorized<c10::BFloat16>& a, const vec::Vectorized<c10::BFloat16>& b, const vec::Vectorized<float>& acc_low, const vec::Vectorized<float>& acc_...

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.fmadd-fn]
> Two overloads. (1) fmadd(a:BFloat16vec, b:BFloat16vec, acc_low, acc_high) ->
> pair<Vectorized<float>,Vectorized<float>>: convert a,b to (low,high) float
> pairs via convert_bfloat16_float, return (fmadd(a_lo,b_lo,acc_low),
> fmadd(a_hi,b_hi,acc_high)). (2) fmadd(acc:float, a:BFloat16vec, b:BFloat16vec)
> -> Vectorized<float>: convert a,b to float pairs, return fmadd(a_hi,b_hi,
> fmadd(a_lo,b_lo,acc)). DEVIATION: subsumed by the scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16]
> struct ForcedUnrollTargetBFloat16

> [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1]
> struct ForcedUnrollTargetBFloat16<1>

> [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1.operator-fn]
> TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()( const Func& f) const

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16-1.operator-fn]
> Base case of the bf16-target forced unroll: operator()(f) calls f(0). (The
> ARM-BF16 attributed analogue of utils::ForcedUnroll<1>.) DEVIATION: subsumed
> by the scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16.operator-fn]
> TARGET_ARM_BF16_ATTRIBUTE C10_ALWAYS_INLINE void operator()( const Func& f) const

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.forced-unroll-target-b-float16.operator-fn]
> Recursive case of the bf16-target forced unroll: operator()(f) calls
> ForcedUnrollTargetBFloat16<n-1>{}(f) then f(n-1), i.e. f(0); f(1); ...;
> f(n-1). ARM-BF16-attributed analogue of utils::ForcedUnroll<n>. DEVIATION:
> subsumed by the scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.integer-log2-fn]
> constexpr int IntegerLog2(T n, int p = 0)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.integer-log2-fn]
> constexpr floor(log2(n)) by tail recursion: returns p when n <= 1, else
> IntegerLog2(n / 2, p + 1). Called with p defaulting to 0.

> [spec:et:def:blas-kernel.executorch.cpublas.internal.reduce-fn]
> float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x)

> [spec:et:sem:blas-kernel.executorch.cpublas.internal.reduce-fn]
> Horizontal reduction of a VectorizedN<float, kF32RegistersPerIteration> to a
> scalar float. offset = kF32RegistersPerIteration; ForcedUnroll<IntegerLog2(
> kF32RegistersPerIteration)> repeatedly halves offset and does x[i] += x[
> offset+i] for i in 0..offset (tree reduction of the register lanes), then
> returns reduce(x[0]) (the single-Vectorized<float> overload: vaddvq_f32 on
> aarch64, else vec_reduce_all with std::plus). DEVIATION: subsumed by the
> scalar bf16 dot port.

> [spec:et:def:blas-kernel.executorch.cpublas.scale-fn]
> void scale_(int64_t m, int64_t n, opmath_t alpha, scalar_t* a, int64_t lda)

> [spec:et:sem:blas-kernel.executorch.cpublas.scale-fn]
> Scale the m×n column-major block `a` (leading dimension lda) in place by the
> opmath_t scalar alpha. If alpha == 1: return immediately (identity). If
> alpha == 0: set a[j*lda + i] = scalar_t(0) for all i in 0..m, j in 0..n and
> return. Otherwise: a[j*lda + i] *= alpha for all i, j (column-major order,
> j outer / i inner).

> [spec:et:def:blas-kernel.executorch.cpublas.sum-fn]
> auto sum(int64_t N, Func f)

> [spec:et:sem:blas-kernel.executorch.cpublas.sum-fn]
> ILP-factored reduction of f(0)+f(1)+...+f(N-1) with acc_t = decltype(f(0)).
> ilp_factor = 4. partial_sums is a zero-initialized array of 4 acc_t. Main
> loop: while i + 4 <= N, ForcedUnroll<4> does partial_sums[k] += f(i+k) for
> k in 0..4, then i += 4. Tail: while i < N, partial_sums[0] += f(i), i++.
> Fold: for k in 1..4, partial_sums[0] += partial_sums[k]. Return
> partial_sums[0].

