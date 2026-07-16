//! Literal port of kernels/optimized/utils/unroll.h.
//!
//! Utility to guarantee complete unrolling of a loop where the bounds are known
//! at compile time. Various pragmas achieve similar effects, but are not as
//! portable across compilers.
//!
//! Example: `forced_unroll::<4>(f);` is equivalent to `f(0); f(1); f(2); f(3);`

// DEVIATION: Rust has no `ForcedUnroll<n>` template-recursion type. The C++
// recursion (`ForcedUnroll<n>` calls `ForcedUnroll<n-1>` then `f(n-1)`, with
// `ForcedUnroll<1>` as the base calling `f(0)`) unwinds to the ascending
// sequence `f(0); f(1); ...; f(N-1)`. Here that is a plain loop over
// `0..N`; LLVM unrolls it. The compile-time count `N` stays a const generic to
// preserve the "bounds known at compile time" intent.

// [spec:et:def:unroll.executorch.utils.forced-unroll]
// [spec:et:def:unroll.executorch.utils.forced-unroll-1]
// [spec:et:def:unroll.executorch.utils.forced-unroll.operator-fn]
// [spec:et:sem:unroll.executorch.utils.forced-unroll.operator-fn]
// [spec:et:def:unroll.executorch.utils.forced-unroll-1.operator-fn]
// [spec:et:sem:unroll.executorch.utils.forced-unroll-1.operator-fn]
#[inline(always)]
pub fn forced_unroll<const N: usize>(mut f: impl FnMut(usize)) {
    for i in 0..N {
        f(i);
    }
}

#[cfg(test)]
mod tests {
    use super::forced_unroll;

    // The recursive case ForcedUnroll<n> unwinds to f(0); f(1); ...; f(n-1)
    // in ascending order (recurse first, then call f(n - 1)).
    // [spec:et:sem:unroll.executorch.utils.forced-unroll.operator-fn/test]
    #[test]
    fn forced_unroll_calls_ascending() {
        let mut calls = Vec::new();
        forced_unroll::<4>(|i| calls.push(i));
        assert_eq!(calls, vec![0, 1, 2, 3]);

        let mut calls = Vec::new();
        forced_unroll::<7>(|i| calls.push(i));
        assert_eq!(calls, (0..7).collect::<Vec<_>>());

        // Header example: ForcedUnroll<4>{}(f) accumulating a sum.
        let mut sum = 0usize;
        forced_unroll::<4>(|i| sum += i + 1);
        assert_eq!(sum, 1 + 2 + 3 + 4);
    }

    // The ForcedUnroll<1> base case invokes f(0) exactly once.
    // [spec:et:sem:unroll.executorch.utils.forced-unroll-1.operator-fn/test]
    #[test]
    fn forced_unroll_base_case_single_call() {
        let mut calls = Vec::new();
        forced_unroll::<1>(|i| calls.push(i));
        assert_eq!(calls, vec![0]);
    }
}
