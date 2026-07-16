//! Literal port of runtime/kernel/thread_parallel_interface.h.
//!
//! This is the no-threadpool build variant (compiled when `ET_USE_THREADPOOL`
//! is not defined). The threadpool variant is out of scope.

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal, so any
// message with `{}` placeholders fails to compile. This local override mirrors
// the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully (prepend "Check failed (cond): "
// then forward the full message + args). Unresolved cross-module reference.
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

pub mod internal {
    // [spec:et:def:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn]
    // [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn]
    //
    // PORT-NOTE: C++ takes `const Func& f`; ported as a generic `F: Fn(i64,
    // i64)`. The debug-only reverse single-element iteration is preserved under
    // `debug_assertions` (the Rust equivalent of `#ifndef NDEBUG`).
    pub fn parallel_for_no_threadpool<F: Fn(i64, i64)>(
        begin: i64,
        end: i64,
        grain_size: i64,
        f: &F,
    ) -> bool {
        et_check_or_return_false!(
            begin >= 0 && end >= 0 && end >= begin,
            "begin = {}, end = {}",
            begin,
            end
        );
        et_check_or_return_false!(grain_size > 0, "grain_size = {}", grain_size);
        #[cfg(debug_assertions)]
        {
            // Go backwards through the range elementwise to catch code that
            // assumes parallel_for is in order like a regular for loop.
            for i in begin..end {
                let offset = i - begin;
                let idx = end - offset - 1;
                f(idx, idx + 1);
            }
        }
        #[cfg(not(debug_assertions))]
        {
            f(begin, end);
        }
        true
    }

    // Match GRAIN_SIZE from PyTorch core.
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h#L78
    pub const GRAIN_SIZE: i64 = 32768;
}

/// A helper to run a function in parallel.
///
/// This is the no-threadpool build variant: a thin forwarder to
/// `internal::parallel_for_no_threadpool`.
// [spec:et:def:thread-parallel-interface.executorch.extension.parallel-for-fn]
// [spec:et:sem:thread-parallel-interface.executorch.extension.parallel-for-fn]
pub fn parallel_for<F: Fn(i64, i64)>(begin: i64, end: i64, grain_size: i64, func: &F) -> bool {
    internal::parallel_for_no_threadpool(begin, end, grain_size, func)
}

// [spec:et:def:thread-parallel-interface.executorch.extension.get-thread-num-fn]
// [spec:et:sem:thread-parallel-interface.executorch.extension.get-thread-num-fn]
pub fn get_thread_num() -> i64 {
    0
}

// [spec:et:def:thread-parallel-interface.executorch.extension.set-thread-num-fn]
// [spec:et:sem:thread-parallel-interface.executorch.extension.set-thread-num-fn]
//
// PORT-NOTE: C++ fires `ET_DCHECK_MSG(false, ...)` (a debug-only fatal check).
// `ET_DCHECK_MSG` has no ported shared macro (unresolved cross-module
// reference); its abort-in-debug / no-op-in-release semantics are mirrored with
// a `debug_assertions`-gated abort.
pub fn set_thread_num(_thread_num: i64) {
    #[cfg(debug_assertions)]
    {
        crate::et_log!(Fatal, "cannot set_thread_num without threading support!");
        crate::runtime::platform::abort::runtime_abort();
    }
}

// No C++ test file ports here: the upstream thread_parallel_test.cpp exercises
// only the threadpool variant (out of scope, see the module header). These are
// focused unit tests for the no-threadpool build variant, with the sem rules
// under docs/spec/port/runtime/kernel/thread_parallel_interface.md as referee.
#[cfg(test)]
mod tests {
    use super::internal::parallel_for_no_threadpool;
    use super::{get_thread_num, parallel_for};
    use core::cell::Cell;

    // Collects every (chunk_begin, chunk_end) pair the callback was invoked with,
    // so the test can assert coverage of the range regardless of visitation order
    // (debug builds visit in reverse single-element chunks; release visits the
    // whole range once). Initializes the PAL first (the gtest `SetUp()`
    // `runtime_init()` convention): the validation-failure path logs via the
    // PAL, which aborts if uninitialized.
    fn collect_chunks(begin: i64, end: i64, grain_size: i64) -> (bool, Vec<(i64, i64)>) {
        crate::runtime::platform::runtime::runtime_init();
        let chunks: Cell<Vec<(i64, i64)>> = Cell::new(Vec::new());
        let ret = parallel_for_no_threadpool(begin, end, grain_size, &|b, e| {
            let mut v = chunks.take();
            v.push((b, e));
            chunks.set(v);
        });
        (ret, chunks.into_inner())
    }

    // A successful run covers [begin, end): the union of all visited chunks is
    // exactly the half-open range, and the function returns true.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn/test]
    #[test]
    fn parallel_for_no_threadpool_covers_range() {
        let (ret, chunks) = collect_chunks(2, 7, 32768);
        assert!(ret);
        let mut covered: Vec<i64> = Vec::new();
        for (b, e) in chunks {
            for i in b..e {
                covered.push(i);
            }
        }
        covered.sort_unstable();
        assert_eq!(covered, vec![2, 3, 4, 5, 6]);
    }

    // Empty range [begin, begin): no callback invocation, returns true.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn/test]
    #[test]
    fn parallel_for_no_threadpool_empty_range_no_calls() {
        let (ret, chunks) = collect_chunks(5, 5, 1);
        assert!(ret);
        assert!(chunks.is_empty());
    }

    // Validation: begin < 0, end < 0, or end < begin returns false without
    // invoking the callback.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn/test]
    #[test]
    fn parallel_for_no_threadpool_rejects_bad_bounds() {
        let (ret_neg_begin, c1) = collect_chunks(-1, 5, 1);
        assert!(!ret_neg_begin);
        assert!(c1.is_empty());

        let (ret_neg_end, c2) = collect_chunks(0, -1, 1);
        assert!(!ret_neg_end);
        assert!(c2.is_empty());

        let (ret_end_lt_begin, c3) = collect_chunks(5, 2, 1);
        assert!(!ret_end_lt_begin);
        assert!(c3.is_empty());
    }

    // Validation: grain_size <= 0 returns false without invoking the callback.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.internal.parallel-for-no-threadpool-fn/test]
    #[test]
    fn parallel_for_no_threadpool_rejects_non_positive_grain_size() {
        let (ret_zero, c1) = collect_chunks(0, 5, 0);
        assert!(!ret_zero);
        assert!(c1.is_empty());

        let (ret_neg, c2) = collect_chunks(0, 5, -4);
        assert!(!ret_neg);
        assert!(c2.is_empty());
    }

    // parallel_for is a thin forwarder to internal::parallel_for_no_threadpool:
    // same range coverage, same validation, same return value.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.parallel-for-fn/test]
    #[test]
    fn parallel_for_forwards_to_no_threadpool() {
        crate::runtime::platform::runtime::runtime_init();
        let chunks: Cell<Vec<(i64, i64)>> = Cell::new(Vec::new());
        let ret = parallel_for(0, 4, 32768, &|b, e| {
            let mut v = chunks.take();
            v.push((b, e));
            chunks.set(v);
        });
        assert!(ret);
        let mut covered: Vec<i64> = Vec::new();
        for (b, e) in chunks.into_inner() {
            for i in b..e {
                covered.push(i);
            }
        }
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 1, 2, 3]);

        // Forwards the failure path too.
        assert!(!parallel_for(0, 4, 0, &|_, _| {}));
    }

    // get_thread_num() unconditionally returns 0 (index of the only, main
    // thread) with no side effects, in the no-threadpool variant.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.get-thread-num-fn/test]
    #[test]
    fn get_thread_num_returns_zero() {
        assert_eq!(get_thread_num(), 0);
    }

    // set_thread_num() fires ET_DCHECK_MSG(false, ...): a debug-only fatal
    // check that aborts the process (no unwinding), and a no-op in release.
    // Re-exec death test per the convention in runtime/platform/abort.rs: the
    // parent re-runs this test binary filtered to this one test with a marker
    // env var set; the child branch calls set_thread_num(1) and must terminate
    // abnormally (SIGABRT on unix hosts) in debug builds, or exit cleanly in
    // release builds where the DCHECK compiles away.
    // [spec:et:sem:thread-parallel-interface.executorch.extension.set-thread-num-fn/test]
    #[test]
    fn set_thread_num_dies_without_threading_support() {
        const ENV_KEY: &str = "ET_SET_THREAD_NUM_DEATH_TEST_CHILD";
        if std::env::var_os(ENV_KEY).is_some() {
            crate::runtime::platform::runtime::runtime_init();
            super::set_thread_num(1);
            // Release (NDEBUG) builds: the DCHECK is a no-op and set_thread_num
            // returns normally.
            std::process::exit(0);
        }

        let exe = std::env::current_exe().expect("current_exe");
        let output = std::process::Command::new(exe)
            .args([
                "set_thread_num_dies_without_threading_support",
                "--test-threads=1",
            ])
            .env(ENV_KEY, "1")
            .output()
            .expect("failed to spawn death-test child");

        if cfg!(debug_assertions) {
            assert!(
                !output.status.success(),
                "set_thread_num() child exited successfully in a debug build: {:?}",
                output.status
            );
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                assert_eq!(output.status.signal(), Some(libc::SIGABRT));
            }
        } else {
            assert!(
                output.status.success(),
                "set_thread_num() child terminated abnormally in a release build: {:?}",
                output.status
            );
        }
    }
}
