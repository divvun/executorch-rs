//! Literal port of runtime/platform/default/posix.cpp.
//!
//! Fallback PAL implementations for POSIX-compatible systems.
//!
//! Note that this assumes that the platform defines the symbols used in this
//! file (like fprintf()), because this file will still be built even if the
//! functions are later overridden. When building for a platform that does not
//! provide the necessary symbols, clients can use minimal.rs instead, but they
//! will need to override all of the functions.

// Build-system selection: POSIX default on unix (excluding android/zephyr, which
// have their own PAL files). See the parent module's gating.
#![cfg(all(unix, not(feature = "android"), not(feature = "zephyr")))]

use core::ffi::{c_char, c_void};

use crate::runtime::platform::platform::{et_pal_log_level_t, et_tick_ratio_t};
use crate::runtime::platform::types::et_timestamp_t;

// PORT-NOTE: The C++ file defines these as weak symbols (`ET_WEAK` /
// `#pragma weak`) so client code can strongly override them. Rust has no stable
// weak-symbol attribute; these are exported as ordinary `#[unsafe(no_mangle)]` C symbols.
// If a client override is linked in, the duplicate-symbol resolution differs from
// the C++ weak-symbol behavior. Construct deviation.

// The FILE* to write logs to. (stderr)

// Start time of the system (used to zero the system timestamp).
// PORT-NOTE: std::chrono::steady_clock is ported to std::time::Instant. The C++
// default-constructs `systemStartTime` as the steady_clock epoch; there is no
// default Instant, so an Option is used, `None` standing in for the unset epoch.
static mut SYSTEM_START_TIME: Option<std::time::Instant> = None;

// Flag set to true if the PAL has been successfully initialized.
static mut INITIALIZED: bool = false;

/// On debug builds, ensure that `et_pal_init` has been called before other PAL
/// functions which depend on initialization.
///
/// In release builds (`NDEBUG` / not `debug_assertions`) this is a no-op.
macro_rules! _assert_pal_initialized {
    () => {
        #[cfg(debug_assertions)]
        unsafe {
            if !INITIALIZED {
                // ET_FUNCTION expands to the enclosing function name in C++; there
                // is no portable equivalent at the macro expansion site, so a
                // placeholder is used (construct deviation).
                libc::fprintf(
                    et_log_output_file(),
                    c"ExecuTorch PAL must be initialized before call to %s()".as_ptr(),
                    c"<function>".as_ptr(),
                );
                libc::fflush(et_log_output_file());
                et_pal_abort();
            }
        }
    };
}

// ET_LOG_OUTPUT_FILE stderr
// PORT-NOTE: libc does not expose `stderr` uniformly across targets (on POSIX it
// is typically a macro over a global). We resolve it via an extern static.
// PORT-NOTE (cross-module fix): on Darwin the C `stderr` macro expands to the
// global `__stderrp`; there is no linkable `stderr` symbol, so the plain
// `extern "C" { static mut stderr }` below fails to link every test binary on
// macOS. Bind the correctly-named symbol per platform via `#[link_name]`. This
// belongs to the platform layer, not the testing_util port, but is required for
// any `cargo test` to link on Darwin.
#[cfg(target_vendor = "apple")]
#[inline]
unsafe fn et_log_output_file() -> *mut libc::FILE {
    unsafe extern "C" {
        #[link_name = "__stderrp"]
        static mut stderr: *mut libc::FILE;
    }
    unsafe { stderr }
}

#[cfg(not(target_vendor = "apple"))]
#[inline]
unsafe fn et_log_output_file() -> *mut libc::FILE {
    unsafe extern "C" {
        static mut stderr: *mut libc::FILE;
    }
    unsafe { stderr }
}

/// Initialize the platform abstraction layer.
///
/// This function should be called before any other function provided by the PAL
/// to initialize any global state. Typically overridden by PAL implementer.
// [spec:et:def:posix.et-pal-init-fn]
// [spec:et:sem:posix.et-pal-init-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_init() {
    unsafe {
        if INITIALIZED {
            return;
        }

        SYSTEM_START_TIME = Some(std::time::Instant::now());
        INITIALIZED = true;
    }
}

/// Immediately abort execution, setting the device into an error state, if
/// available.
// [spec:et:def:posix.et-pal-abort-fn]
// [spec:et:sem:posix.et-pal-abort-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_abort() -> ! {
    unsafe {
        libc::abort();
    }
}

/// Return a monotonically non-decreasing timestamp in system ticks.
// [spec:et:def:posix.et-pal-current-ticks-fn]
// [spec:et:sem:posix.et-pal-current-ticks-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_current_ticks() -> et_timestamp_t {
    _assert_pal_initialized!();
    unsafe {
        let system_current_time = std::time::Instant::now();
        // duration_cast<nanoseconds>(systemCurrentTime - systemStartTime).count()
        match SYSTEM_START_TIME {
            Some(start) => system_current_time.duration_since(start).as_nanos() as et_timestamp_t,
            None => 0,
        }
    }
}

/// Return the conversion rate from system ticks to nanoseconds, as a fraction.
/// To convert an interval from system ticks to nanoseconds, multiply the tick
/// count by the numerator and then divide by the denominator:
///   nanoseconds = ticks * numerator / denominator
// [spec:et:def:posix.et-pal-ticks-to-ns-multiplier-fn]
// [spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
    // The system tick interval is 1 nanosecond, so the conversion factor is 1.
    et_tick_ratio_t {
        numerator: 1,
        denominator: 1,
    }
}

/// Emit a log message via platform output (serial port, console, etc).
// [spec:et:def:posix.et-pal-emit-log-message-fn]
// [spec:et:sem:posix.et-pal-emit-log-message-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_emit_log_message(
    mut timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    filename: *const c_char,
    _function: *const c_char,
    line: usize,
    message: *const c_char,
    _length: usize,
) {
    _assert_pal_initialized!();

    // Not all platforms have ticks == nanoseconds, but this one does.
    timestamp /= 1000; // To microseconds
    let us: core::ffi::c_ulong = (timestamp % 1000000) as core::ffi::c_ulong;
    timestamp /= 1000000; // To seconds
    let sec: core::ffi::c_uint = (timestamp % 60) as core::ffi::c_uint;
    timestamp /= 60; // To minutes
    let min: core::ffi::c_uint = (timestamp % 60) as core::ffi::c_uint;
    timestamp /= 60; // To hours
    let hour: core::ffi::c_uint = timestamp as core::ffi::c_uint;

    // Use a format similar to glog and folly::logging, except:
    // - Print time since et_pal_init since we don't have wall time
    // - Don't include the thread ID, to avoid adding a threading dependency
    // - Add the string "executorch:" to make the logs more searchable
    //
    // Clients who want to change the format or add other fields can override this
    // weak implementation of et_pal_emit_log_message.
    unsafe {
        libc::fprintf(
            et_log_output_file(),
            c"%c %02u:%02u:%02u.%06lu executorch:%s:%zu] %s\n".as_ptr(),
            level as core::ffi::c_int,
            hour,
            min,
            sec,
            us,
            filename,
            line,
            message,
        );
        libc::fflush(et_log_output_file());
    }
}

/// NOTE: Core runtime code must not call this directly. It may only be called by
/// a MemoryAllocator wrapper.
///
/// Allocates size bytes of memory via malloc.
// [spec:et:def:posix.et-pal-allocate-fn]
// [spec:et:sem:posix.et-pal-allocate-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_allocate(size: usize) -> *mut c_void {
    unsafe { libc::malloc(size) }
}

/// Frees memory allocated by et_pal_allocate().
// [spec:et:def:posix.et-pal-free-fn]
// [spec:et:sem:posix.et-pal-free-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_free(ptr: *mut c_void) {
    unsafe {
        libc::free(ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PORT-NOTE: The C++ posix.cpp has no dedicated *_test.cpp; the weak defaults
    // are exercised indirectly by the shared platform suite (executor_pal_test.cpp
    // etc.) once the real symbols are linked. These focused tests pin the strong
    // posix hooks' behavior directly, since register_pal only rewrites the PalImpl
    // table and never displaces the strong `et_pal_*` symbols called here.

    // [spec:et:sem:posix.et-pal-allocate-fn/test]
    // [spec:et:sem:posix.et-pal-free-fn/test]
    #[test]
    fn posix_allocate_and_free() {
        // malloc-backed: a positive size returns a usable, writable block.
        let ptr = et_pal_allocate(64) as *mut u8;
        assert!(!ptr.is_null());
        unsafe {
            for i in 0..64usize {
                ptr.add(i).write(i as u8);
            }
            for i in 0..64usize {
                assert_eq!(ptr.add(i).read(), i as u8);
            }
        }
        et_pal_free(ptr as *mut c_void);

        // free(nullptr) is a no-op and must not fault.
        et_pal_free(core::ptr::null_mut());
    }

    // [spec:et:sem:posix.et-pal-emit-log-message-fn/test]
    #[test]
    fn posix_emit_log_message_smoke() {
        // The debug-build _ASSERT_PAL_INITIALIZED gate requires init first.
        et_pal_init();
        // Drives the real timestamp decomposition + fprintf-to-stderr path; the
        // function returns void, so this pins that a well-formed call completes
        // without faulting under the strong posix emitter.
        et_pal_emit_log_message(
            3_723_000_456, // 1h 02m 03s 000456us worth of ns
            et_pal_log_level_t::kInfo,
            c"posix_test.rs".as_ptr(),
            core::ptr::null(),
            42,
            c"posix emit smoke".as_ptr(),
            "posix emit smoke".len(),
        );
    }

    // Death test in the gtest EXPECT_DEATH style: fork, call et_pal_abort() in
    // the child, and assert the child terminates via SIGABRT (the posix default
    // aborts through libc::abort). Only async-signal-safe calls happen in the
    // child after the fork.
    // [spec:et:sem:posix.et-pal-abort-fn/test]
    #[test]
    fn posix_abort_death() {
        unsafe {
            let pid = libc::fork();
            assert!(pid >= 0);
            if pid == 0 {
                et_pal_abort();
            }
            let mut status: core::ffi::c_int = 0;
            assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
            assert!(libc::WIFSIGNALED(status));
            assert_eq!(libc::WTERMSIG(status), libc::SIGABRT);
        }
    }
}
