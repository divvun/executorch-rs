//! Literal port of runtime/platform/default/zephyr.cpp.

// Build-system selection: the Zephyr PAL is selected via the `zephyr` feature.
#![cfg(feature = "zephyr")]

// PORT-NOTE: No `#[cfg(test)] mod tests` is added here. This entire module is
// gated behind `#![cfg(feature = "zephyr")]` and its symbols require the Zephyr
// kernel (`k_uptime_ticks`/`k_malloc`/`k_free`) at link time; on the darwin
// posix test host none of it compiles, links, or runs. Adding a tests module
// here would be a fake never-run test. These symbols are platform-gated on this
// host and are recorded as gaps by the porting harness.

use core::ffi::{c_char, c_int, c_void};

use crate::runtime::platform::platform::{et_pal_log_level_t, et_tick_ratio_t};
use crate::runtime::platform::types::et_timestamp_t;

// PORT-NOTE: The C++ file defines these as weak-overridable PAL symbols. Rust has
// no stable weak-symbol attribute; these are exported as ordinary `#[unsafe(no_mangle)]`
// C symbols. Construct deviation.

// Zephyr kernel externs (<zephyr/kernel.h>).
// PORT-NOTE: declared here as externs; provided by the Zephyr kernel at link time.
unsafe extern "C" {
    fn k_uptime_ticks() -> i64;
    fn k_malloc(size: usize) -> *mut c_void;
    fn k_free(ptr: *mut c_void);
    // <cstdlib> _Exit
    fn _Exit(status: c_int) -> !;
    static mut stderr: *mut libc::FILE;
}

// [spec:et:def:zephyr.et-pal-init-fn]
// [spec:et:sem:zephyr.et-pal-init-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_init() {}

// [spec:et:def:zephyr.et-pal-abort-fn]
// [spec:et:sem:zephyr.et-pal-abort-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_abort() -> ! {
    unsafe {
        _Exit(-1);
    }
}

// [spec:et:def:zephyr.et-pal-current-ticks-fn]
// [spec:et:sem:zephyr.et-pal-current-ticks-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_current_ticks() -> et_timestamp_t {
    unsafe { k_uptime_ticks() as et_timestamp_t }
}

// [spec:et:def:zephyr.et-pal-ticks-to-ns-multiplier-fn]
// [spec:et:sem:zephyr.et-pal-ticks-to-ns-multiplier-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
    // Since we don't know the CPU freq for your target and just cycles in the
    // FVP for et_pal_current_ticks() we return a conversion ratio of 1
    et_tick_ratio_t {
        numerator: 1,
        denominator: 1,
    }
}

/// Emit a log message via platform output (serial port, console, etc).
// [spec:et:def:zephyr.et-pal-emit-log-message-fn]
// [spec:et:sem:zephyr.et-pal-emit-log-message-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_emit_log_message(
    _timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    filename: *const c_char,
    function: *const c_char,
    line: usize,
    message: *const c_char,
    _length: usize,
) {
    unsafe {
        libc::fprintf(
            stderr,
            c"%c [executorch:%s:%zu %s()] %s\n".as_ptr(),
            level as c_int,
            filename,
            line,
            function,
            message,
        );
    }
}

// [spec:et:def:zephyr.et-pal-allocate-fn]
// [spec:et:sem:zephyr.et-pal-allocate-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_allocate(size: usize) -> *mut c_void {
    unsafe { k_malloc(size) }
}

// [spec:et:def:zephyr.et-pal-free-fn]
// [spec:et:sem:zephyr.et-pal-free-fn]
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_free(ptr: *mut c_void) {
    unsafe {
        k_free(ptr);
    }
}
