# runtime/platform/log.cpp, runtime/platform/log.h

> [spec:et:def:log.executorch.runtime.internal.get-log-timestamp-fn]
> et_timestamp_t get_log_timestamp()

> [spec:et:sem:log.executorch.runtime.internal.get-log-timestamp-fn]
> Returns the current timestamp used to stamp a log event.
>
> Behavior: returns the value of `pal_current_ticks()` verbatim (see
> `[spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn]`), i.e. the
> active PAL's monotonically non-decreasing timestamp in system ticks. Takes no
> arguments, performs no validation, and has no side effects. The result type is
> `et_timestamp_t` (a 64-bit tick count).

> [spec:et:def:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn]
> static size_t get_valid_utf8_prefix_length(const char* bytes, size_t length)

> [spec:et:sem:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn]
> Given a byte buffer `bytes` of `length` bytes, returns the length (in bytes)
> of the longest prefix that ends on a complete UTF-8 code point boundary, so
> that truncating the buffer at that length never leaves a partial multi-byte
> sequence. This function is only compiled when `ET_LOG_ENABLED` is nonzero.
>
> Steps:
> 1. If `bytes` is null OR `length == 0`, return 0.
> 2. Interpret the bytes as unsigned bytes. Set `i = length`. While `i > 0` and
>    the byte at index `i - 1` is a UTF-8 continuation byte (top two bits equal
>    `10`, i.e. `(data[i-1] & 0xC0) == 0x80`), decrement `i`. This walks
>    backward over any trailing continuation bytes to find the last lead byte.
> 3. If `i == 0` (the whole tail was continuation bytes with no lead byte),
>    return 0.
> 4. Let `lead_pos = i - 1` and `lead = data[lead_pos]` (the last lead byte).
>    Determine how many bytes `need` the code point starting at `lead` requires:
>    - `lead < 0x80` (ASCII, `0xxxxxxx`): need = 1.
>    - `(lead & 0xE0) == 0xC0` (`110xxxxx`): need = 2.
>    - `(lead & 0xF0) == 0xE0` (`1110xxxx`): need = 3.
>    - `(lead & 0xF8) == 0xF0` (`11110xxx`): need = 4.
>    - otherwise (an invalid lead byte, e.g. a stray continuation-pattern or
>      `0xF8`-and-above): return `lead_pos` (drop the invalid trailing byte(s)).
> 5. Let `available = length - lead_pos` be the number of bytes present from the
>    lead byte to the end. If `available == need`, the final code point is
>    complete: return `length` (keep the whole buffer). Otherwise the final code
>    point is truncated: return `lead_pos` (drop the incomplete tail).
>
> Note: the function only validates the length of the final (possibly partial)
> sequence and the class of its lead byte; it does not validate that
> intermediate continuation bytes are well-formed or that the code point is in a
> valid range.

> [spec:et:def:log.executorch.runtime.internal.logf-fn]
> inline void logf( LogLevel level, et_timestamp_t timestamp, const char* filename, const char* function, size_t line, const char* format, ...)

> [spec:et:sem:log.executorch.runtime.internal.logf-fn]
> Variadic front-end to the logging path; the `ET_LOG` macro ultimately calls
> this. Internal function — callers use `ET_LOG` rather than calling directly.
>
> Signature: `logf(level, timestamp, filename, function, line, format, ...)`.
> Behavior:
> 1. If `ET_LOG_ENABLED` is zero, the entire body is compiled out: the call is a
>    no-op and the variadic arguments are ignored.
> 2. Otherwise, capture the variadic arguments following `format` into a
>    `va_list` and forward them, together with `level`, `timestamp`, `filename`,
>    `function`, `line`, and `format`, to `vlogf` (see
>    `[spec:et:sem:log.executorch.runtime.internal.vlogf-fn]`), then release the
>    `va_list`.
>
> No validation is performed here and no value is returned. Note the caller-side
> `ET_LOG` macro (not this function) is responsible for skipping the call
> entirely when `level` is below the compile-time `ET_MIN_LOG_LEVEL` threshold;
> `logf` itself does not re-check the level. The `ET_PRINTFLIKE(6,7)` attribute
> requests printf-style format/argument checking against `format`.

> [spec:et:def:log.executorch.runtime.internal.vlogf-fn]
> void vlogf( ET_UNUSED LogLevel level, et_timestamp_t timestamp, const char* filename, ET_UNUSED const char* function, size_t line, const char* format, va_list args)

> [spec:et:sem:log.executorch.runtime.internal.vlogf-fn]
> Formats a log message from a `va_list` and emits it through the PAL. Internal
> function — callers use `ET_LOG` (via `logf`) rather than calling directly.
>
> Signature: `vlogf(level, timestamp, filename, function, line, format, args)`.
> `level` and `function` are unused when logging is disabled; `filename`, `line`
> and `timestamp` are passed through to the emitter.
>
> Behavior:
> 1. If `ET_LOG_ENABLED` is zero, the entire body is compiled out: no-op, no
>    output, no return value.
> 2. Otherwise:
>    a. Allocate a fixed stack buffer of `kMaxLogMessageLength = 256` bytes.
>    b. Call `vsnprintf(buffer, 256, format, args)` to render the message,
>       obtaining `write_count` (the number of characters that would have been
>       written excluding the NUL, or negative on encoding error).
>    c. Compute `used_length`:
>       - if `write_count < 0` (formatting error): `used_length = 0`;
>       - else if `write_count >= 256` (output was truncated by vsnprintf):
>         `used_length = 255` (256 - 1);
>       - else: `used_length = write_count`.
>    d. Compute `valid_length = get_valid_utf8_prefix_length(buffer,
>       used_length)` (see
>       `[spec:et:sem:log.executorch.runtime.internal.get-valid-utf8-prefix-length-fn]`)
>       so that a mid-character truncation at byte 255 does not emit a partial
>       UTF-8 sequence.
>    e. Write a NUL terminator at `buffer[valid_length]`.
>    f. Map `level` to an `et_pal_log_level_t`: if `level < LogLevel::NumLevels`,
>       use the fixed table `kLevelToPal` which maps
>       `Debug -> kDebug ('D')`, `Info -> kInfo ('I')`, `Error -> kError ('E')`,
>       `Fatal -> kFatal ('F')`; otherwise use `kUnknown ('?')`.
>    g. Call `pal_emit_log_message(timestamp, pal_level, filename, function,
>       line, buffer, valid_length)` (see
>       `[spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn]`),
>       passing the NUL-terminated buffer and its `valid_length` byte count.
>
> Returns nothing. No error is propagated on formatting failure; a failed format
> simply produces an empty (`valid_length == 0`) message.

> [spec:et:def:log.executorch.runtime.log-level]
> enum class LogLevel : uint8_t {
>   Debug;
>   Info;
>   Error;
>   Fatal;
>   NumLevels;
> }

