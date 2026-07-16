# kernels/optimized/cpu/moments_utils.h

> [spec:et:def:moments-utils.torch.executor.native.add-moments-fn]
> void AddMoments( int64_t m0_add, const T& m1_add, const T& m2_add, int64_t& m0, T& m1, T& m2)

> [spec:et:sem:moments-utils.torch.executor.native.add-moments-fn]
> Chan-style parallel merge of two Welford moment accumulators, scalar variant.
> Given an existing accumulator `(m0, m1, m2)` (count, mean, sum-of-squared-
> deviations) and a second accumulator `(m0_add, m1_add, m2_add)`, merge the
> second into the first in place. Compute `n = m0 + m0_add`. Compute the blend
> weight `c = (n == 0) ? 0 : m0_add / n` (all arithmetic in `T`). Compute
> `delta = m1_add - m1`. Update the mean `m1 += c * delta`. Update the M2
> `m2 += m2_add + delta*delta*c*m0` (using the OLD `m0`). Finally set `m0 = n`.

> [spec:et:def:moments-utils.torch.executor.native.add-moments-vec-fn]
> ET_INLINE void AddMomentsVec( int64_t m0_add, const at::vec::Vectorized<T>& m1_add, const at::vec::Vectorized<T>& m2_add, int64_t& m0, at::vec::Vectorized<T>& m1, at::vec::Vectorized<T>& m2)
>
> DEVIATION: `at::vec::Vectorized<T>` collapses to scalar `T` in the Rust port
> (SIMD lane count 1); the vector merge becomes the scalar merge.

> [spec:et:sem:moments-utils.torch.executor.native.add-moments-vec-fn]
> Vectorized (lanewise) form of AddMoments: identical algebra, but `m1_add`,
> `m2_add`, `m1`, `m2` are SIMD vectors and the scalars `c` and `m0` are
> broadcast to vectors. `n = m0 + m0_add`; `c = (n == 0) ? 0 : m0_add / n`
> (scalar in `T`); broadcast `c` to `c_vec`. `delta = m1_add - m1` (vector).
> `m1 += c_vec * delta`. `m2 += m2_add + delta*delta*c_vec*Vec(m0)` (m0 is the
> OLD count, broadcast). `m0 = n`. In the scalar-lane Rust port each vector is a
> single `T`, so this is the same computation as AddMoments.

> [spec:et:def:moments-utils.torch.executor.native.rowwise-moments-fn]
> std::pair<acc_t<T>, acc_t<T>> RowwiseMoments(const T* X, int64_t N, int64_t ddof = 0)

> [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-fn]
> Public entry. Computes the (mean, variance) of the N elements at X using a
> parallel Welford accumulation with cascade (pairwise) summation for numerical
> stability. Selects a compile-time recursion/stack depth bound `kMaxDepth` from
> the data size and dispatches to RowwiseMomentsImpl<T, kMaxDepth>. Compute
> `kVecSize = Vectorized<T>::size()` (1 in the scalar-lane port), `n = N /
> kVecSize`, `m = divup(n, kChunkSize)` with `kChunkSize = 16`, `depth =
> CeilLog2(m)`. If `depth <= 4` call impl with kMaxDepth=4; else if `<= 8` use 8;
> else if `<= 16` use 16; else if `<= 32` use 32; else use 64. Returns the pair
> from the impl. `ddof` defaults to 0 and is forwarded.

> [spec:et:def:moments-utils.torch.executor.native.rowwise-moments-impl-fn]
> std::pair<acc_t<T>, acc_t<T>> RowwiseMomentsImpl(const T* X, int64_t N, int64_t ddof = 0)

> [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-impl-fn]
> Core Welford + cascade-sum implementation. `T_ACC = acc_t<T>` (compute dtype:
> f32 for 16-bit floats, else T). `kVecSize = Vectorized<T>::size()`,
> `kAccVecSize = Vectorized<T_ACC>::size()` (both 1 in the scalar-lane port).
> `n = N / kVecSize`, `m = divup(n, kChunkSize)`, `depth = CeilLog2(m)`.
> Allocate three fixed stacks of length kMaxDepth: `m0_stk` (int64), `m1_stk`,
> `m2_stk` (vectors of T_ACC), all zero-initialized.
> Main loop over `i in 0..m`: `X_ptr = X + i*kChunkSize*kVecSize`;
> `m0 = min(kChunkSize, n - i*kChunkSize)`. Build (once, statically) `c_vecs[j]
> = 1/(j+1)` for `j in 0..kChunkSize`. Call UpdateMomentsVec(m0, X_ptr, c_vecs,
> m0_stk[0], m1_stk[0], m2_stk[0]) to fold this chunk into stack slot 0. Then
> cascade: `mask = i+1`; for `j in 1..depth` while `(mask & 1) == 0`: merge slot
> j-1 into slot j via AddMomentsVec, then zero slot j-1, then `mask >>= 1`.
> After the loop, drain the remaining stack: for `i in 1..depth` merge slot i
> into slot 0 via AddMomentsVec.
> Store slot-0 vectors `m1_stk[0]`, `m2_stk[0]` into `m1_arr`, `m2_arr` (length
> kAccVecSize). Handle the scalar tail beyond `n*kVecSize`: init `m0=0, m1=0,
> m2=0`; for `i in n*kVecSize..N`: `x = X[i]`; `delta = x - m1`; `++m0`;
> `m1 += delta/m0`; `m2 += delta*(x - m1)`. Then fold the per-lane accumulators:
> `m0_add = n*kVecSize / kAccVecSize`; for each of the kAccVecSize lanes call
> AddMoments(m0_add, m1_arr[lane], m2_arr[lane], m0, m1, m2).
> Return `(m1, m2 / (N - ddof))`.

> [spec:et:def:moments-utils.torch.executor.native.update-moments-vec-fn]
> inline void UpdateMomentsVec( int64_t m0, const T* X_ptr, const std::array<at::vec::Vectorized<acc_t<T>>, kChunkSize>& c_vecs, int64_t& m0_stk0, at::vec::Vectorized<acc_t<T>>& m1_stk0, at::vec::Vectorized<acc_t<T>>& m2_stk0)

> [spec:et:sem:moments-utils.torch.executor.native.update-moments-vec-fn]
> Fold up to `m0` consecutive vector-loads from X_ptr into a fresh Welford
> accumulator, then merge that accumulator into stack slot 0. Init local
> `m1_vec = 0`, `m2_vec = 0`. For `j in 0..m0`: load `x_vec = X_ptr[j*Vec::size()
> ..]`; `delta_vec = x_vec - m1_vec`; `m1_vec += delta_vec * c_vecs[j]` (where
> `c_vecs[j] = 1/(j+1)`); `m2_vec += delta_vec * (x_vec - m1_vec)` (using the
> UPDATED m1_vec). After the loop call AddMomentsVec(m0, m1_vec, m2_vec,
> m0_stk0, m1_stk0, m2_stk0) to merge the count-`m0` accumulator into slot 0.
> In the scalar-lane port each vector is a single `T_ACC`.
