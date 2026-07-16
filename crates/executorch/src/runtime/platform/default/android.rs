//! Literal port of runtime/platform/default/android.cpp.
//!
//! Default PAL implementations for Android system.
//!
//! PORT-NOTE(wave-3): no tests — this module is only compiled with the
//! `android` feature and links against liblog's `__android_log_print`, which
//! does not exist on non-Android hosts. Coverage of the `android.et-pal-*`
//! symbols is a recorded platform gap until tests run on an Android target.

// Build-system selection: the Android PAL is selected via the `android` feature.
#![cfg(feature = "android")]

use core::ffi::{c_char, c_int, c_void};

use crate::runtime::platform::platform::{et_pal_log_level_t, et_tick_ratio_t};
use crate::runtime::platform::types::et_timestamp_t;

// PORT-NOTE: The C++ file defines these as weak symbols (`ET_WEAK`) so client code
// can strongly override them. Rust has no stable weak-symbol attribute; these are
// exported as ordinary `#[unsafe(no_mangle)]` C symbols. Construct deviation.

// android/log.h constants and function.
// PORT-NOTE: these are provided by liblog on Android; declared here as externs
// since libc does not expose them uniformly.
const ANDROID_LOG_UNKNOWN: c_int = 0;
const ANDROID_LOG_DEBUG: c_int = 3;
const ANDROID_LOG_INFO: c_int = 4;
const ANDROID_LOG_ERROR: c_int = 6;
const ANDROID_LOG_FATAL: c_int = 7;

unsafe extern "C" {
    fn __android_log_print(prio: c_int, tag: *const c_char, fmt: *const c_char, ...) -> c_int;
}

// Start time of the system (used to zero the system timestamp).
// PORT-NOTE: std::chrono::steady_clock ported to std::time::Instant; `None`
// stands in for the default-constructed steady_clock epoch.
static mut SYSTEM_START_TIME: Option<std::time::Instant> = None;

// Flag set to true if the PAL has been successfully initialized.
static mut INITIALIZED: bool = false;

/// On debug builds, ensure that `et_pal_init` has been called before other PAL
/// functions which depend on initialization. Unlike the POSIX variant, this only
/// logs and does not abort. No-op in release builds.
macro_rules! _assert_pal_initialized {
    () => {
        #[cfg(debug_assertions)]
        unsafe {
            if !INITIALIZED {
                // ET_FUNCTION placeholder (construct deviation, see posix.rs).
                __android_log_print(
                    ANDROID_LOG_FATAL,
                    c"ExecuTorch".as_ptr(),
                    c"ExecuTorch PAL must be initialized before call to %s()".as_ptr(),
                    c"<function>".as_ptr(),
                );
            }
        }
    };
}

/// Initialize the platform abstraction layer.
// [spec:et:def:android.et-pal-init-fn]
// [spec:et:sem:android.et-pal-init-fn]
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
// [spec:et:def:android.et-pal-abort-fn]
// [spec:et:sem:android.et-pal-abort-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_abort() -> ! {
    unsafe {
        libc::abort();
    }
}

/// Return a monotonically non-decreasing timestamp in system ticks.
// [spec:et:def:android.et-pal-current-ticks-fn]
// [spec:et:sem:android.et-pal-current-ticks-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_current_ticks() -> et_timestamp_t {
    _assert_pal_initialized!();
    unsafe {
        let system_current_time = std::time::Instant::now();
        match SYSTEM_START_TIME {
            Some(start) => system_current_time.duration_since(start).as_nanos() as et_timestamp_t,
            None => 0,
        }
    }
}

/// Return the conversion rate from system ticks to nanoseconds, as a fraction.
// [spec:et:def:android.et-pal-ticks-to-ns-multiplier-fn]
// [spec:et:sem:android.et-pal-ticks-to-ns-multiplier-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
    // The system tick interval is 1 nanosecond, so the conversion factor is 1.
    et_tick_ratio_t {
        numerator: 1,
        denominator: 1,
    }
}

/// Emit a log message to adb logcat.
// [spec:et:def:android.et-pal-emit-log-message-fn]
// [spec:et:sem:android.et-pal-emit-log-message-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_emit_log_message(
    _timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    _filename: *const c_char,
    _function: *const c_char,
    _line: usize,
    message: *const c_char,
    _length: usize,
) {
    _assert_pal_initialized!();

    let mut android_log_level: c_int = ANDROID_LOG_UNKNOWN;
    if level as u8 == b'D' {
        android_log_level = ANDROID_LOG_DEBUG;
    } else if level as u8 == b'I' {
        android_log_level = ANDROID_LOG_INFO;
    } else if level as u8 == b'E' {
        android_log_level = ANDROID_LOG_ERROR;
    } else if level as u8 == b'F' {
        android_log_level = ANDROID_LOG_FATAL;
    }

    unsafe {
        __android_log_print(
            android_log_level,
            c"ExecuTorch".as_ptr(),
            c"%s".as_ptr(),
            message,
        );
    }
}

/// NOTE: Core runtime code must not call this directly. It may only be called by
/// a MemoryAllocator wrapper.
///
/// Allocates size bytes of memory via malloc.
// [spec:et:def:android.et-pal-allocate-fn]
// [spec:et:sem:android.et-pal-allocate-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_allocate(size: usize) -> *mut c_void {
    unsafe { libc::malloc(size) }
}

/// Frees memory allocated by et_pal_allocate().
// [spec:et:def:android.et-pal-free-fn]
// [spec:et:sem:android.et-pal-free-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_free(ptr: *mut c_void) {
    unsafe {
        libc::free(ptr);
    }
}
