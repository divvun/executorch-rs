//! Literal port of runtime/platform/default/minimal.cpp.
//!
//! Fallback PAL implementations that do not depend on any assumptions about
//! capabililties of the system.

// Build-system selection: minimal is the fallback used when the target is not
// POSIX and neither the android nor zephyr PAL is selected.
#![cfg(all(not(unix), not(feature = "android"), not(feature = "zephyr")))]

// PORT-NOTE: No `#[cfg(test)] mod tests` is added here. This entire module is
// gated behind `#![cfg(all(not(unix), ...))]`, so on a POSIX/unix host (the
// test host is darwin) none of it — including any tests module — compiles or
// runs; the POSIX PAL in `posix.rs` is used instead. Adding a tests module here
// would be a fake never-run test. These symbols are platform-gated on this host
// and are recorded as gaps by the porting harness.

use core::ffi::{c_char, c_void};

use crate::runtime::platform::platform::{et_pal_log_level_t, et_tick_ratio_t};
use crate::runtime::platform::types::et_timestamp_t;

// PORT-NOTE: The C++ file defines these as weak symbols (`ET_WEAK`) so client code
// can strongly override them. Rust has no stable weak-symbol attribute; these are
// exported as ordinary `#[unsafe(no_mangle)]` C symbols. Construct deviation.

// [spec:et:def:minimal.et-pal-init-fn]
// [spec:et:sem:minimal.et-pal-init-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_init() {}

// [spec:et:def:minimal.et-pal-abort-fn]
// [spec:et:sem:minimal.et-pal-abort-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_abort() -> ! {
    // PORT-NOTE: C++ calls `__builtin_trap()`, which emits an illegal/trap
    // instruction (typically SIGILL). Rust has no stable trap-instruction
    // intrinsic; `std::process::abort()` is used instead (SIGABRT). The
    // "immediate abnormal termination, no cleanup" behavior matches; the exact
    // signal differs. Construct deviation.
    std::process::abort();
}

// [spec:et:def:minimal.et-pal-current-ticks-fn]
// [spec:et:sem:minimal.et-pal-current-ticks-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_current_ticks() -> et_timestamp_t {
    // This file cannot make any assumptions about the presence of functions that
    // return the current time, so all users should provide a strong override for
    // it. To help make it more obvious when this weak version is being used,
    // return a number that should be easier to search for than 0.
    11223344
}

// [spec:et:def:minimal.et-pal-ticks-to-ns-multiplier-fn]
// [spec:et:sem:minimal.et-pal-ticks-to-ns-multiplier-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
    // Since we don't define a tick rate, return a conversion ratio of 1.
    et_tick_ratio_t {
        numerator: 1,
        denominator: 1,
    }
}

// [spec:et:def:minimal.et-pal-emit-log-message-fn]
// [spec:et:sem:minimal.et-pal-emit-log-message-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_emit_log_message(
    _timestamp: et_timestamp_t,
    _level: et_pal_log_level_t,
    _filename: *const c_char,
    _function: *const c_char,
    _line: usize,
    _message: *const c_char,
    _length: usize,
) {
}

// [spec:et:def:minimal.et-pal-allocate-fn]
// [spec:et:sem:minimal.et-pal-allocate-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_allocate(_size: usize) -> *mut c_void {
    core::ptr::null_mut()
}

// [spec:et:def:minimal.et-pal-free-fn]
// [spec:et:sem:minimal.et-pal-free-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_free(_ptr: *mut c_void) {}
