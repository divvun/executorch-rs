//! Literal port of runtime/platform/log.cpp + runtime/platform/log.h.
//!
//! ExecuTorch logging API.

use core::ffi::c_char;

use crate::runtime::platform::platform::{
    et_pal_log_level_t, pal_current_ticks, pal_emit_log_message,
};
use crate::runtime::platform::types::et_timestamp_t;

// Set minimum log severity if compiler option is not provided.
//
// PORT-NOTE: C++ `ET_MIN_LOG_LEVEL` defaults to `Info`; the `et_log!` macro
// compares against `LogLevel::Info as u32`. Exposed as a const for the macro.
pub const ET_MIN_LOG_LEVEL: LogLevel = LogLevel::Info;

/*
 * Enable logging by default if compiler option is not provided.
 * This should facilitate less confusion for those developing ExecuTorch.
 */
pub const ET_LOG_ENABLED: bool = true;

/// Severity level of a log message. Must be ordered from lowest to highest
/// severity.
// [spec:et:def:log.executorch.runtime.log-level]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Log messages provided for highly granular debuggability.
    ///
    /// Log messages using this severity are unlikely to be compiled by default
    /// into most debug builds.
    Debug,

    /// Log messages providing information about the state of the system
    /// for debuggability.
    Info,

    /// Log messages about errors within ExecuTorch during runtime.
    Error,

    /// Log messages that precede a fatal error. However, logging at this level
    /// does not perform the actual abort, something else needs to.
    Fatal,

    /// Number of supported log levels, with values in [0, NumLevels).
    NumLevels,
}

pub mod internal {
    use super::*;

    /// Get the current timestamp to construct a log event.
    ///
    /// @retval Monotonically non-decreasing timestamp in system ticks.
    // [spec:et:def:log.executorch.runtime.internal.get-log-timestamp-fn]
    // [spec:et:sem:log.executorch.runtime.internal.get-log-timestamp-fn]
    pub fn get_log_timestamp() -> et_timestamp_t {
        pal_current_ticks()
    }

    // Double-check that the log levels are ordered from lowest to highest
    // severity.
    const _: () = assert!((LogLevel::Debug as u8) < (LogLevel::Info as u8));
    const _: () = assert!((LogLevel::Info as u8) < (LogLevel::Error as u8));
    const _: () = assert!((LogLevel::Error as u8) < (LogLevel::Fatal as u8));
    const _: () = assert!((LogLevel::Fatal as u8) < (LogLevel::NumLevels as u8));

    /// Maps LogLevel values to et_pal_log_level_t values.
    ///
    /// We don't share values because LogLevel values need to be ordered by
    /// severity, and et_pal_log_level_t values need to be printable characters.
    static K_LEVEL_TO_PAL: [et_pal_log_level_t; LogLevel::NumLevels as usize] = [
        et_pal_log_level_t::kDebug,
        et_pal_log_level_t::kInfo,
        et_pal_log_level_t::kError,
        et_pal_log_level_t::kFatal,
    ];

    /// Maximum length of a log message.
    const K_MAX_LOG_MESSAGE_LENGTH: usize = 256;

    // [spec:et:def:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn]
    // [spec:et:sem:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn]
    fn get_valid_utf8_prefix_length(bytes: &[u8], length: usize) -> usize {
        if bytes.is_empty() || length == 0 {
            return 0;
        }
        let data = bytes;
        let mut i = length;
        while i > 0 && (data[i - 1] & 0xC0) == 0x80 {
            i -= 1;
        }
        if i == 0 {
            return 0;
        }
        let lead_pos = i - 1;
        let lead = data[lead_pos];
        let need: usize;

        if lead < 0x80 {
            need = 1;
        } else if (lead & 0xE0) == 0xC0 {
            need = 2;
        } else if (lead & 0xF0) == 0xE0 {
            need = 3;
        } else if (lead & 0xF8) == 0xF0 {
            need = 4;
        } else {
            return lead_pos;
        }
        if length - lead_pos == need {
            length
        } else {
            lead_pos
        }
    }

    /// Log a string message.
    ///
    /// Note: This is an internal function. Use the `et_log!` macro instead.
    // [spec:et:def:log.executorch.runtime.internal.vlogf-fn]
    // [spec:et:sem:log.executorch.runtime.internal.vlogf-fn]
    //
    // PORT-NOTE: C++ `vlogf` takes a `va_list` and renders via `vsnprintf` into
    // a 256-byte stack buffer. Rust has no `va_list`; the `et_log!` macro
    // formats the message itself and passes the rendered bytes here. `level`,
    // `timestamp`, `filename`, `line` are forwarded to the emitter; the
    // formatting/truncation logic below mirrors the C++ verbatim.
    pub fn vlogf(
        level: LogLevel,
        timestamp: et_timestamp_t,
        filename: *const c_char,
        function: *const c_char,
        line: usize,
        rendered: &[u8],
    ) {
        if !ET_LOG_ENABLED {
            return;
        }

        let mut buffer = [0u8; K_MAX_LOG_MESSAGE_LENGTH];

        // Emulate vsnprintf: copy up to 255 bytes, reserving room for the NUL.
        let write_count = rendered.len();
        let used_length = if write_count >= K_MAX_LOG_MESSAGE_LENGTH {
            K_MAX_LOG_MESSAGE_LENGTH - 1
        } else {
            write_count
        };
        buffer[..used_length].copy_from_slice(&rendered[..used_length]);

        let valid_length = get_valid_utf8_prefix_length(&buffer, used_length);
        buffer[valid_length] = b'\0';

        let pal_level = if (level as u8) < (LogLevel::NumLevels as u8) {
            K_LEVEL_TO_PAL[level as usize]
        } else {
            et_pal_log_level_t::kUnknown
        };

        pal_emit_log_message(
            timestamp,
            pal_level,
            filename,
            function,
            line,
            buffer.as_ptr() as *const c_char,
            valid_length,
        );
    }

    /// Log a string message.
    ///
    /// Note: This is an internal function. Use the `et_log!` macro instead.
    // [spec:et:def:log.executorch.runtime.internal.logf-fn]
    // [spec:et:sem:log.executorch.runtime.internal.logf-fn]
    //
    // PORT-NOTE: the C++ `logf` captures the variadic args into a `va_list` and
    // forwards to `vlogf`. Here the `et_log!` macro has already rendered the
    // message; `logf` simply forwards the rendered bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn logf(
        level: LogLevel,
        timestamp: et_timestamp_t,
        filename: *const c_char,
        function: *const c_char,
        line: usize,
        rendered: &[u8],
    ) {
        if ET_LOG_ENABLED {
            vlogf(level, timestamp, filename, function, line, rendered);
        }
    }
}

/// Log a message at the given log severity level.
//
// PORT-NOTE: mirrors the C++ `ET_LOG` macro. Marked `#[macro_export]` so it is
// reachable as `crate::et_log!`. The compile-time `ET_LOG_ENABLED` /
// `ET_MIN_LOG_LEVEL` gating is preserved as runtime `if` guards (const-folded).
// `ET_SHORT_FILENAME`/`ET_FUNCTION`/`ET_LINE` map to `file!()`/`""`/`line!()`;
// the C-string filename is synthesized from `file!()`.
#[macro_export]
macro_rules! et_log {
    ($level:ident, $($arg:tt)*) => {{
        if $crate::runtime::platform::log::ET_LOG_ENABLED {
            let _log_level = $crate::runtime::platform::log::LogLevel::$level;
            if (_log_level as u32)
                >= ($crate::runtime::platform::log::ET_MIN_LOG_LEVEL as u32)
            {
                let _timestamp =
                    $crate::runtime::platform::log::internal::get_log_timestamp();
                let _rendered = ::std::format!($($arg)*);
                let _filename = ::std::concat!(::std::file!(), "\0");
                $crate::runtime::platform::log::internal::logf(
                    _log_level,
                    _timestamp,
                    _filename.as_ptr() as *const ::core::ffi::c_char,
                    ::core::ptr::null(),
                    ::std::line!() as usize,
                    _rendered.as_bytes(),
                );
            }
        }
    }};
}

/// Check a condition and log an error message if the condition is false.
#[macro_export]
macro_rules! et_check_or_log_error {
    ($condition:expr, $($arg:tt)*) => {{
        if !($condition) {
            $crate::et_log!(Error, $($arg)*);
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::runtime::platform::platform::test_spy;

    // logging_test.cpp uses a PalSpy intercept to capture emit_log_message args.
    // PORT-NOTE: as with the platform tests, the C++ weak-symbol/InterceptWith
    // mechanism is replaced by the shared `test_spy` register_pal harness. Each
    // test holds PAL_TEST_LOCK for the duration and reads the spy's captured
    // message, exactly mirroring the C++ `spy.last_log_message_args`.

    // Emit at `level` through the same internal path `et_log!` expands to
    // (timestamp + rendered bytes + logf), bypassing the macro's compile-time
    // `ET_MIN_LOG_LEVEL` gate.
    //
    // PORT-NOTE: The C++ `logging_test.cpp` target is compiled with
    // `-DET_MIN_LOG_LEVEL=Debug` (see runtime/platform/test/CMakeLists.txt /
    // targets.bzl), so `ET_LOG(Debug, ...)` emits there. The Rust `et_log!` macro
    // reads the crate-level `ET_MIN_LOG_LEVEL` const (Info) and const-folds the
    // Debug call away; there is no per-test way to lower that const. To reproduce
    // the `ET_MIN_LOG_LEVEL=Debug` build's observable emit-at-Debug behavior
    // without weakening the assertion, drive `internal::logf` directly here (the
    // exact code the macro runs once the gate passes).
    fn emit_log(level: crate::runtime::platform::log::LogLevel, message: &str) {
        let timestamp = crate::runtime::platform::log::internal::get_log_timestamp();
        crate::runtime::platform::log::internal::logf(
            level,
            timestamp,
            c"logging_test.rs".as_ptr(),
            core::ptr::null(),
            0,
            message.as_bytes(),
        );
    }

    // [spec:et:sem:log.executorch.runtime.internal.logf-fn/test]
    // [spec:et:sem:log.executorch.runtime.internal.vlogf-fn/test]
    // TEST_F(LoggingTest, LogLevels)
    #[test]
    fn logging_test_log_levels() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        use crate::runtime::platform::log::LogLevel;

        emit_log(LogLevel::Debug, "Debug log.");
        assert_eq!(spy.last_log_message_args.message, "Debug log.");

        emit_log(LogLevel::Info, "Info log.");
        assert_eq!(spy.last_log_message_args.message, "Info log.");

        emit_log(LogLevel::Error, "Error log.");
        assert_eq!(spy.last_log_message_args.message, "Error log.");

        emit_log(LogLevel::Fatal, "Fatal log.");
        assert_eq!(spy.last_log_message_args.message, "Fatal log.");
    }

    // [spec:et:sem:log.executorch.runtime.internal.vlogf-fn/test]
    // TEST_F(LoggingTest, LogFormatting)
    #[test]
    fn logging_test_log_formatting() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        crate::et_log!(Info, "Sample log with integer: {}", 100u32);
        assert_eq!(
            spy.last_log_message_args.message,
            "Sample log with integer: 100"
        );
    }

    // Mirror of the C++ helper `get_prefix`.
    fn get_prefix(length: usize, use_multibyte: bool) -> Vec<u8> {
        if !use_multibyte {
            return vec![b'A'; length];
        }
        let mut result: Vec<u8> = vec![b'A'; length % 4];
        let mut remaining = length - (length % 4);
        while remaining > 0 {
            result.extend_from_slice(b"\xF0\x9F\x91\x8D");
            remaining -= 4;
        }
        result
    }

    // [spec:et:sem:log.executorch.runtime.internal.vlogf-fn/test]
    // [spec:et:sem:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn/test]
    // TEST_F(LoggingTest, Utf8Truncation)
    #[test]
    fn logging_test_utf8_truncation() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        let euro: &[u8] = b"\xE2\x82\xAC";
        let thumbs_up: &[u8] = b"\xF0\x9F\x91\x8D";
        let e_acute: &[u8] = b"\xC3\xA9";
        let capital_a_tilde: &[u8] = b"\xC3\x83";

        struct TruncCase {
            prefix_length: usize,
            codepoint: &'static [u8],
        }
        let cases = [
            TruncCase {
                prefix_length: 253,
                codepoint: euro,
            },
            TruncCase {
                prefix_length: 252,
                codepoint: thumbs_up,
            },
            TruncCase {
                prefix_length: 254,
                codepoint: e_acute,
            },
            TruncCase {
                prefix_length: 254,
                codepoint: capital_a_tilde,
            },
        ];
        for use_multibyte_prefix in [false, true] {
            for c in &cases {
                let prefix = get_prefix(c.prefix_length, use_multibyte_prefix);
                let suffix = b"_SHOULD_BE_CUT";

                // ET_LOG(Info, "%s%s%s", prefix, codepoint, suffix)
                //
                // PORT-NOTE: the et_log! macro formats via Rust `format!`, which
                // requires the interpolated args be valid UTF-8. The prefix (when
                // multibyte) and codepoint are valid UTF-8 byte sequences; we build
                // the full message as a byte string and feed it through the same
                // logf path the macro uses so the vsnprintf-buffer truncation +
                // UTF-8 prefix walk-back is exercised identically.
                let mut full: Vec<u8> = Vec::new();
                full.extend_from_slice(&prefix);
                full.extend_from_slice(c.codepoint);
                full.extend_from_slice(suffix);

                let timestamp = crate::runtime::platform::log::internal::get_log_timestamp();
                crate::runtime::platform::log::internal::logf(
                    crate::runtime::platform::log::LogLevel::Info,
                    timestamp,
                    c"log_test.rs".as_ptr(),
                    core::ptr::null(),
                    0,
                    &full,
                );

                let expected = String::from_utf8(prefix.clone()).expect("prefix is valid UTF-8");
                assert_eq!(spy.last_log_message_args.message, expected);
                assert_eq!(spy.last_log_message_args.length, prefix.len());
            }
        }
    }
}
