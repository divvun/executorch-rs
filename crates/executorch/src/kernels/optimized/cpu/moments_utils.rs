//! Literal port of kernels/optimized/cpu/moments_utils.h.
//!
//! Slightly modified version of caffe2/aten/src/ATen/native/cpu/moments_utils.h
//! for use in optimized ExecuTorch ops. Template specializations of BFloat16
//! are excluded.
//!
//! DEVIATION: the C++ leans on `at::vec::Vectorized<T>` (a cross-arch SIMD lane
//! type). Per PORTING.md's optimized-kernels rule, the SIMD lane type collapses
//! to the scalar element type (lane count 1). Every `Vectorized<T>` here becomes
//! a scalar `T`/`T_ACC`; `Vec::size()` becomes `1`. The blocked/cascade loop
//! STRUCTURE is preserved bug-for-bug; only the lane width changes. With lane
//! width 1 the `RowwiseMomentsImpl` main loop consumes `N` scalars, the vector
//! tail loop (`i in n*kVecSize..N`) is empty, and `kAccVecSize == 1`.

use crate::kernels::optimized::utils::math_utils::{ComputeDtype, ceil_log2, divup};
use crate::runtime::core::portable_type::{BFloat16, Half};

// `template <typename T> using acc_t = executorch::utils::compute_dtype<T>;`
pub type AccT<T> = ComputeDtype<T>;

pub const K_CHUNK_SIZE: i64 = 16;

/// Float accumulation type used by the moment recurrences (`T_ACC`, always a
/// real float — `f32` for 16-bit-float `T`, else `T`). Carries the scalar
/// arithmetic the C++ performs on `Vectorized<T_ACC>` lanes plus construction
/// from integers/reciprocals.
///
/// DEVIATION: a `Vectorized<T_ACC>` is one `T_ACC` lane here.
pub trait AccFloat:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    fn zero() -> Self;
    fn from_i64(v: i64) -> Self;
    fn recip_of_index(j: i64) -> Self; // 1 / (j + 1)
    fn sqrt(self) -> Self;
}

impl AccFloat for f32 {
    fn zero() -> Self {
        0.0
    }
    fn from_i64(v: i64) -> Self {
        v as f32
    }
    fn recip_of_index(j: i64) -> Self {
        1.0f32 / ((j + 1) as f32)
    }
    fn sqrt(self) -> Self {
        // DEVIATION: scalar libm sqrt in place of the vectorized transcendental.
        libm::sqrtf(self)
    }
}

impl AccFloat for f64 {
    fn zero() -> Self {
        0.0
    }
    fn from_i64(v: i64) -> Self {
        v as f64
    }
    fn recip_of_index(j: i64) -> Self {
        1.0f64 / ((j + 1) as f64)
    }
    fn sqrt(self) -> Self {
        // DEVIATION: scalar libm sqrt in place of the vectorized transcendental.
        libm::sqrt(self)
    }
}

/// Element type `T` flowing into RowwiseMoments: real float / 16-bit float. Its
/// `Acc` is the compute dtype and `to_acc` reproduces `static_cast<T_ACC>(X[i])`.
pub trait MomentScalar: Copy {
    type Acc: AccFloat;
    fn to_acc(self) -> Self::Acc;
}

impl MomentScalar for f32 {
    type Acc = f32;
    fn to_acc(self) -> f32 {
        self
    }
}
impl MomentScalar for f64 {
    type Acc = f64;
    fn to_acc(self) -> f64 {
        self
    }
}
impl MomentScalar for Half {
    type Acc = f32;
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}
impl MomentScalar for BFloat16 {
    type Acc = f32;
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
}

// [spec:et:def:moments-utils.torch.executor.native.add-moments-fn]
// [spec:et:sem:moments-utils.torch.executor.native.add-moments-fn]
pub fn add_moments<T: AccFloat>(
    m0_add: i64,
    m1_add: T,
    m2_add: T,
    m0: &mut i64,
    m1: &mut T,
    m2: &mut T,
) {
    let n: i64 = *m0 + m0_add;
    let c: T = if n == 0 {
        T::zero()
    } else {
        T::from_i64(m0_add) / T::from_i64(n)
    };
    let delta: T = m1_add - *m1;
    *m1 = *m1 + c * delta;
    *m2 = *m2 + m2_add + delta * delta * c * T::from_i64(*m0);
    *m0 = n;
}

// [spec:et:def:moments-utils.torch.executor.native.add-moments-vec-fn]
// [spec:et:sem:moments-utils.torch.executor.native.add-moments-vec-fn]
// DEVIATION: `at::vec::Vectorized<T>` collapses to scalar `T` (lane count 1),
// so `AddMomentsVec` is `AddMoments` on a single lane.
pub fn add_moments_vec<T: AccFloat>(
    m0_add: i64,
    m1_add: T,
    m2_add: T,
    m0: &mut i64,
    m1: &mut T,
    m2: &mut T,
) {
    let n: i64 = *m0 + m0_add;
    let c: T = if n == 0 {
        T::zero()
    } else {
        T::from_i64(m0_add) / T::from_i64(n)
    };
    let c_vec: T = c;
    let delta: T = m1_add - *m1;
    *m1 = *m1 + c_vec * delta;
    *m2 = *m2 + m2_add + delta * delta * c_vec * T::from_i64(*m0);
    *m0 = n;
}

// [spec:et:def:moments-utils.torch.executor.native.update-moments-vec-fn]
// [spec:et:sem:moments-utils.torch.executor.native.update-moments-vec-fn]
// DEVIATION: `Vec::size()` is 1; each `c_vecs[j]` and each loaded `x_vec` is one
// `T_ACC` lane. The C++ `std::array<Vectorized<acc_t<T>>, kChunkSize>` becomes
// `[T_ACC; kChunkSize]`.
///
/// # Safety
/// `x_ptr` must point to at least `m0 * Vec::size()` (= `m0`) valid `T` elements.
pub unsafe fn update_moments_vec<T: MomentScalar>(
    m0: i64,
    x_ptr: *const T,
    c_vecs: &[T::Acc; K_CHUNK_SIZE as usize],
    m0_stk0: &mut i64,
    m1_stk0: &mut T::Acc,
    m2_stk0: &mut T::Acc,
) {
    // Vec::size() == 1 in the scalar-lane port.
    const VEC_SIZE: i64 = 1;
    let mut m1_vec: T::Acc = T::Acc::zero();
    let mut m2_vec: T::Acc = T::Acc::zero();
    for j in 0..m0 {
        let x_vec: T::Acc = unsafe { (*x_ptr.offset((j * VEC_SIZE) as isize)).to_acc() };
        let delta_vec: T::Acc = x_vec - m1_vec;
        m1_vec = m1_vec + delta_vec * c_vecs[j as usize];
        m2_vec = m2_vec + delta_vec * (x_vec - m1_vec);
    }
    add_moments_vec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
}

// Compute rowwise moments by parallel Welford algorithm and cascade sum to
// improve numerical stability.
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
// https://en.wikipedia.org/wiki/Pairwise_summation
// [spec:et:def:moments-utils.torch.executor.native.rowwise-moments-impl-fn]
// [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-impl-fn]
// PORT-NOTE: the C++ selects the stack depth via a template value `kMaxDepth`
// (4/8/16/32/64). The Rust port passes it as an ordinary `usize` parameter
// (`k_max_depth`) driving the runtime-sized stacks — the stacks are never
// larger than `depth`, so a heap `Vec` mirrors the fixed `std::array<_,
// kMaxDepth>` without a const generic.
///
/// # Safety
/// `x` must point to at least `n` valid `T` elements.
pub unsafe fn rowwise_moments_impl<T: MomentScalar>(
    x: *const T,
    n_len: i64,
    ddof: i64,
    k_max_depth: usize,
) -> (T::Acc, T::Acc) {
    // constexpr int64_t kVecSize = at::vec::Vectorized<T>::size();     -> 1
    // constexpr int64_t kAccVecSize = at::vec::Vectorized<T_ACC>::size(); -> 1
    const K_VEC_SIZE: i64 = 1;
    const K_ACC_VEC_SIZE: i64 = 1;
    let n: i64 = n_len / K_VEC_SIZE;
    let m: i64 = divup(n, K_CHUNK_SIZE);
    let depth: i64 = ceil_log2(m);

    let k_zero: T::Acc = T::Acc::zero();
    let mut m0_stk: Vec<i64> = vec![0; k_max_depth];
    let mut m1_stk: Vec<T::Acc> = vec![k_zero; k_max_depth];
    let mut m2_stk: Vec<T::Acc> = vec![k_zero; k_max_depth];

    // static std::array<Vec, kChunkSize> c_vecs, with c_vecs[j] = 1/(j+1).
    let mut c_vecs: [T::Acc; K_CHUNK_SIZE as usize] = [k_zero; K_CHUNK_SIZE as usize];
    for j in 0..K_CHUNK_SIZE {
        c_vecs[j as usize] = <T::Acc as AccFloat>::recip_of_index(j);
    }

    for i in 0..m {
        let x_ptr: *const T = unsafe { x.offset((i * K_CHUNK_SIZE * K_VEC_SIZE) as isize) };
        let m0: i64 = core::cmp::min(K_CHUNK_SIZE, n - i * K_CHUNK_SIZE);
        unsafe {
            update_moments_vec(
                m0,
                x_ptr,
                &c_vecs,
                &mut m0_stk[0],
                &mut m1_stk[0],
                &mut m2_stk[0],
            );
        }

        let mut mask: i64 = i + 1;
        let mut j: i64 = 1;
        while j < depth && (mask & 1) == 0 {
            let (m0_prev, m1_prev, m2_prev) = (
                m0_stk[(j - 1) as usize],
                m1_stk[(j - 1) as usize],
                m2_stk[(j - 1) as usize],
            );
            add_moments_vec(
                m0_prev,
                m1_prev,
                m2_prev,
                &mut m0_stk[j as usize],
                &mut m1_stk[j as usize],
                &mut m2_stk[j as usize],
            );
            m0_stk[(j - 1) as usize] = 0;
            m1_stk[(j - 1) as usize] = k_zero;
            m2_stk[(j - 1) as usize] = k_zero;
            mask >>= 1;
            j += 1;
        }
    }
    for i in 1..depth {
        let (m0_i, m1_i, m2_i) = (m0_stk[i as usize], m1_stk[i as usize], m2_stk[i as usize]);
        add_moments_vec(
            m0_i,
            m1_i,
            m2_i,
            &mut m0_stk[0],
            &mut m1_stk[0],
            &mut m2_stk[0],
        );
    }

    // std::array<T_ACC, kAccVecSize> m1_arr / m2_arr; m1_stk[0].store(...).
    let mut m1_arr: [T::Acc; K_ACC_VEC_SIZE as usize] = [k_zero; K_ACC_VEC_SIZE as usize];
    let mut m2_arr: [T::Acc; K_ACC_VEC_SIZE as usize] = [k_zero; K_ACC_VEC_SIZE as usize];
    // store() of a 1-lane vector copies the single lane.
    m1_arr[0] = m1_stk[0];
    m2_arr[0] = m2_stk[0];

    let mut m0: i64 = 0;
    let mut m1: T::Acc = T::Acc::zero();
    let mut m2: T::Acc = T::Acc::zero();
    for i in (n * K_VEC_SIZE)..n_len {
        let x_val: T::Acc = unsafe { (*x.offset(i as isize)).to_acc() };
        let delta: T::Acc = x_val - m1;
        m0 += 1;
        m1 = m1 + delta / T::Acc::from_i64(m0);
        m2 = m2 + delta * (x_val - m1);
    }
    // for BFloat16, each vector in m1_arr/m2_arr holds 2*n accumulated result
    let m0_add: i64 = n * K_VEC_SIZE / K_ACC_VEC_SIZE;
    for i in 0..K_ACC_VEC_SIZE {
        add_moments(
            m0_add,
            m1_arr[i as usize],
            m2_arr[i as usize],
            &mut m0,
            &mut m1,
            &mut m2,
        );
    }

    (m1, m2 / T::Acc::from_i64(n_len - ddof))
}

// [spec:et:def:moments-utils.torch.executor.native.rowwise-moments-fn]
// [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-fn]
///
/// # Safety
/// `x` must point to at least `n` valid `T` elements.
pub unsafe fn rowwise_moments<T: MomentScalar>(
    x: *const T,
    n_len: i64,
    ddof: i64,
) -> (T::Acc, T::Acc) {
    // constexpr int64_t kVecSize = Vec::size();  -> 1
    const K_VEC_SIZE: i64 = 1;
    let n: i64 = n_len / K_VEC_SIZE;
    let m: i64 = divup(n, K_CHUNK_SIZE);
    let depth: i64 = ceil_log2(m);
    if depth <= 4 {
        unsafe { rowwise_moments_impl::<T>(x, n_len, ddof, 4) }
    } else if depth <= 8 {
        unsafe { rowwise_moments_impl::<T>(x, n_len, ddof, 8) }
    } else if depth <= 16 {
        unsafe { rowwise_moments_impl::<T>(x, n_len, ddof, 16) }
    } else if depth <= 32 {
        unsafe { rowwise_moments_impl::<T>(x, n_len, ddof, 32) }
    } else {
        unsafe { rowwise_moments_impl::<T>(x, n_len, ddof, 64) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // moments_utils_test.cpp `is_close<T>(val, ref, tol = 1e-5)`.
    fn is_close(val: f64, reference: f64, tol: f64) -> bool {
        (val - reference).abs() <= tol
    }

    fn test_calc_moments<T: MomentScalar>(data: &[T])
    where
        T::Acc: Into<f64>,
    {
        let (mean, variance) = unsafe {
            rowwise_moments::<T>(data.as_ptr(), data.len() as i64, /*ddof=*/ 0)
        };
        let mean_f: f64 = mean.into();
        let variance_f: f64 = variance.into();
        assert!(is_close(mean_f, 7.25, 1e-5), "mean = {}", mean_f);
        assert!(
            is_close(variance_f, 15.9375, 1e-5),
            "variance = {}",
            variance_f
        );
    }

    // Port of moments_utils_test.cpp TEST(MomentsUtilTest, CalculateMoments).
    // PORT-NOTE: the C++ TEST_FORALL_FLOAT_CTYPES list also instantiates
    // RowwiseMoments<short> (acc type int32); MomentScalar deliberately only
    // covers the float element types the optimized kernels dispatch
    // (f32/f64/Half/BFloat16), so the `short` instantiation is unrepresentable
    // here. Half/BFloat16 (f32 accumulator) are covered instead.
    // [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-fn/test]
    // [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-impl-fn/test]
    #[test]
    fn moments_util_test_calculate_moments() {
        let data_f64: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0, 9.0, 10.0, 12.0, 13.0];
        test_calc_moments::<f64>(&data_f64);

        let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();
        test_calc_moments::<f32>(&data_f32);

        let data_half: Vec<Half> = data_f64.iter().map(|&v| Half::from_f64(v)).collect();
        test_calc_moments::<Half>(&data_half);

        let data_bf16: Vec<BFloat16> = data_f64.iter().map(|&v| BFloat16::from_f64(v)).collect();
        test_calc_moments::<BFloat16>(&data_bf16);
    }

    // Merging Welford summaries of [1,2,3] (m0=3, m1=2, m2=2) and [4,6]
    // (m0=2, m1=5, m2=2) must equal the moments of [1,2,3,4,6]:
    // mean 3.2, M2 = sum((x-3.2)^2) = 14.8.
    // [spec:et:sem:moments-utils.torch.executor.native.add-moments-fn/test]
    #[test]
    fn add_moments_merges_welford_summaries() {
        let mut m0: i64 = 3;
        let mut m1: f64 = 2.0;
        let mut m2: f64 = 2.0;
        add_moments::<f64>(2, 5.0, 2.0, &mut m0, &mut m1, &mut m2);
        assert_eq!(m0, 5);
        assert!(is_close(m1, 3.2, 1e-12));
        assert!(is_close(m2, 14.8, 1e-12));
    }

    // n == 0 selects c = 0 (no division by zero) and leaves everything zero.
    // [spec:et:sem:moments-utils.torch.executor.native.add-moments-fn/test]
    #[test]
    fn add_moments_zero_counts_stay_zero() {
        let mut m0: i64 = 0;
        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;
        add_moments::<f32>(0, 0.0, 0.0, &mut m0, &mut m1, &mut m2);
        assert_eq!(m0, 0);
        assert_eq!(m1, 0.0);
        assert_eq!(m2, 0.0);
    }

    // AddMomentsVec is AddMoments on a single scalar lane (lane count 1); it
    // must produce identical results on the same summaries.
    // [spec:et:sem:moments-utils.torch.executor.native.add-moments-vec-fn/test]
    #[test]
    fn add_moments_vec_matches_add_moments() {
        let mut m0_a: i64 = 3;
        let mut m1_a: f64 = 2.0;
        let mut m2_a: f64 = 2.0;
        add_moments::<f64>(2, 5.0, 2.0, &mut m0_a, &mut m1_a, &mut m2_a);

        let mut m0_b: i64 = 3;
        let mut m1_b: f64 = 2.0;
        let mut m2_b: f64 = 2.0;
        add_moments_vec::<f64>(2, 5.0, 2.0, &mut m0_b, &mut m1_b, &mut m2_b);

        assert_eq!(m0_a, m0_b);
        assert_eq!(m1_a, m1_b);
        assert_eq!(m2_a, m2_b);
        assert_eq!(m0_b, 5);
        assert!(is_close(m1_b, 3.2, 1e-12));
        assert!(is_close(m2_b, 14.8, 1e-12));

        // m0_add == 0 with empty stack: n == 0 -> c == 0, all zero.
        let mut m0: i64 = 0;
        let mut m1: f64 = 0.0;
        let mut m2: f64 = 0.0;
        add_moments_vec::<f64>(0, 0.0, 0.0, &mut m0, &mut m1, &mut m2);
        assert_eq!((m0, m1, m2), (0, 0.0, 0.0));
    }

    // One chunk [2,3,4,5] with c_vecs[j] = 1/(j+1) accumulates the exact
    // Welford summary (m0=4, mean=3.5, M2=5) into the level-0 stack slot.
    // [spec:et:sem:moments-utils.torch.executor.native.update-moments-vec-fn/test]
    #[test]
    fn update_moments_vec_accumulates_chunk() {
        let x: [f64; 4] = [2.0, 3.0, 4.0, 5.0];
        let mut c_vecs: [f64; K_CHUNK_SIZE as usize] = [0.0; K_CHUNK_SIZE as usize];
        for j in 0..K_CHUNK_SIZE {
            c_vecs[j as usize] = <f64 as AccFloat>::recip_of_index(j);
        }
        let mut m0_stk0: i64 = 0;
        let mut m1_stk0: f64 = 0.0;
        let mut m2_stk0: f64 = 0.0;
        unsafe {
            update_moments_vec::<f64>(
                x.len() as i64,
                x.as_ptr(),
                &c_vecs,
                &mut m0_stk0,
                &mut m1_stk0,
                &mut m2_stk0,
            );
        }
        assert_eq!(m0_stk0, 4);
        assert!(is_close(m1_stk0, 3.5, 1e-12));
        assert!(is_close(m2_stk0, 5.0, 1e-12));
    }

    // Direct RowwiseMomentsImpl call: ddof rescales only the second moment,
    // m2 / (N - ddof): 127.5/8 = 15.9375 (ddof 0), 127.5/7 (ddof 1).
    // [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-impl-fn/test]
    #[test]
    fn rowwise_moments_impl_applies_ddof() {
        let data: [f64; 8] = [2.0, 3.0, 4.0, 5.0, 9.0, 10.0, 12.0, 13.0];
        let (mean0, var0) = unsafe {
            rowwise_moments_impl::<f64>(data.as_ptr(), 8, /*ddof=*/ 0, 4)
        };
        assert!(is_close(mean0, 7.25, 1e-12));
        assert!(is_close(var0, 15.9375, 1e-12));

        let (mean1, var1) = unsafe {
            rowwise_moments_impl::<f64>(data.as_ptr(), 8, /*ddof=*/ 1, 4)
        };
        assert!(is_close(mean1, 7.25, 1e-12));
        assert!(is_close(var1, 127.5 / 7.0, 1e-12));
    }

    // N = 1000 forces multiple chunks (m = 63, depth = 6 -> kMaxDepth = 8
    // dispatch arm) so the cascade merge loops actually run; compare against
    // the naive two-pass mean/variance.
    // [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-fn/test]
    // [spec:et:sem:moments-utils.torch.executor.native.rowwise-moments-impl-fn/test]
    #[test]
    fn rowwise_moments_large_input_matches_naive() {
        let n: usize = 1000;
        let data: Vec<f64> = (0..n).map(|i| ((i * 37) % 100) as f64 / 10.0).collect();

        let naive_mean: f64 = data.iter().sum::<f64>() / n as f64;
        let naive_var: f64 = data
            .iter()
            .map(|&x| (x - naive_mean) * (x - naive_mean))
            .sum::<f64>()
            / n as f64;

        let (mean, variance) = unsafe {
            rowwise_moments::<f64>(data.as_ptr(), n as i64, /*ddof=*/ 0)
        };
        assert!(
            is_close(mean, naive_mean, 1e-9),
            "{} vs {}",
            mean,
            naive_mean
        );
        assert!(
            is_close(variance, naive_var, 1e-9),
            "{} vs {}",
            variance,
            naive_var
        );
    }
}
