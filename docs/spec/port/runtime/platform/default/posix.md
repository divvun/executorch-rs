# runtime/platform/default/posix.cpp

> [spec:et:def:posix.et-pal-abort-fn]
> ET_NORETURN void et_pal_abort(void)

> [spec:et:sem:posix.et-pal-abort-fn]
> Weak default POSIX abort hook (may be overridden; declared `#pragma weak` under
> MSVC). Takes no arguments and never returns (`ET_NORETURN`). Unconditionally
> calls C++ `std::abort()`, which raises `SIGABRT` and terminates the process
> abnormally with no cleanup (no destructors, no atexit handlers). No argument
> validation, no logging, no return path.

> [spec:et:def:posix.et-pal-allocate-fn]
> void* et_pal_allocate(size_t size)

> [spec:et:sem:posix.et-pal-allocate-fn]
> Weak default POSIX allocator hook (may be overridden; `#pragma weak` under MSVC).
> Allocates `size` bytes via C `malloc(size)` and returns the resulting pointer.
> Returns `nullptr` on allocation failure. The returned memory is uninitialized and
> must be released with `et_pal_free()` (`[spec:et:sem:posix.et-pal-free-fn]`). No
> argument validation; `size == 0` behaves as C `malloc(0)` (implementation-defined:
> returns either `nullptr` or a unique freeable pointer). Core runtime code must not
> call this directly — only a MemoryAllocator wrapper may. No `_ASSERT_PAL_INITIALIZED`
> check.

> [spec:et:def:posix.et-pal-current-ticks-fn]
> et_timestamp_t et_pal_current_ticks(void)

> [spec:et:sem:posix.et-pal-current-ticks-fn]
> Weak default POSIX timestamp hook (may be overridden; `#pragma weak` under MSVC).
> Returns a monotonically non-decreasing timestamp in system ticks, where one tick
> equals one nanosecond (see `[spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn]`),
> measured relative to `et_pal_init` time.
>
> Steps:
> 1. Invoke `_ASSERT_PAL_INITIALIZED()`. In debug builds (`!NDEBUG`), if the
>    module-static `initialized` flag is false, print "ExecuTorch PAL must be
>    initialized before call to <function>()" (function name from `ET_FUNCTION`) to
>    stderr, `fflush` stderr, then call `et_pal_abort()`
>    (`[spec:et:sem:posix.et-pal-abort-fn]`), which terminates the process. In
>    release builds (`NDEBUG`) this macro is a no-op.
> 2. Read the current `std::chrono::steady_clock::now()`.
> 3. Compute the elapsed duration since the module-static `systemStartTime`
>    (captured in `et_pal_init`, `[spec:et:sem:posix.et-pal-init-fn]`),
>    `duration_cast` it to `std::chrono::nanoseconds`, and return its integer
>    `.count()` as an `et_timestamp_t` (`uint64_t`).
>
> The steady_clock is monotonic, so successive calls are non-decreasing. If
> `et_pal_init` was never called (only reachable in release builds, since debug
> aborts), `systemStartTime` is the default-constructed steady_clock epoch and the
> return is the raw reading since that epoch.

> [spec:et:def:posix.et-pal-emit-log-message-fn]
> void et_pal_emit_log_message( et_timestamp_t timestamp, et_pal_log_level_t level, const char* filename, ET_UNUSED const char* function, size_t line, const char* message, ET_UNUSED size_t length)

> [spec:et:sem:posix.et-pal-emit-log-message-fn]
> Weak default POSIX log sink (may be overridden; `#pragma weak` under MSVC). Emits
> one formatted log line to stderr. Parameters: `timestamp` (system ticks since
> `et_pal_init`; here 1 tick == 1 ns), `level` (an `et_pal_log_level_t` ASCII
> letter code), `filename`, `function` (unused, `ET_UNUSED`), `line`, `message`
> (the NUL-terminated string to log), `length` (unused, `ET_UNUSED`).
>
> Steps:
> 1. Invoke `_ASSERT_PAL_INITIALIZED()` (see
>    `[spec:et:sem:posix.et-pal-current-ticks-fn]` step 1: debug-only; logs to
>    stderr and calls `et_pal_abort()` if `initialized` is false; no-op in release).
> 2. Decompose `timestamp` (nanoseconds) into a time-since-init clock, mutating a
>    local copy via successive integer divisions/moduli:
>    - divide `timestamp` by 1000 -> microseconds;
>    - `us = timestamp % 1000000` (an `unsigned long int`, the fractional
>      microseconds within the current second);
>    - divide `timestamp` by 1000000 -> seconds;
>    - `sec = timestamp % 60` (`unsigned int`);
>    - divide `timestamp` by 60 -> minutes;
>    - `min = timestamp % 60` (`unsigned int`);
>    - divide `timestamp` by 60 -> hours;
>    - `hour = timestamp` (`unsigned int`; not wrapped modulo anything, so hours can
>      grow without bound).
>    All divisions are truncating integer division; sub-microsecond precision is
>    dropped.
> 3. `fprintf` to stderr with the format
>    `"%c %02u:%02u:%02u.%06lu executorch:%s:%zu] %s\n"` and arguments, in order:
>    `level` (the raw byte, e.g. 'D'/'I'/'E'/'F'), `hour`, `min`, `sec` (each
>    zero-padded to width 2), `us` (zero-padded to width 6), `filename`, `line`
>    (a `size_t`, `%zu`), then `message`, followed by a trailing newline.
> 4. `fflush` stderr so the line is not buffered.
>
> Returns void. The timestamp is time since `et_pal_init`, not wall-clock time; no
> thread ID is printed. The literal "executorch:" prefix is included to make logs
> searchable.

> [spec:et:def:posix.et-pal-free-fn]
> void et_pal_free(void* ptr)

> [spec:et:sem:posix.et-pal-free-fn]
> Weak default POSIX deallocator hook (may be overridden; `#pragma weak` under
> MSVC). Frees the memory at `ptr` via C `free(ptr)`. `ptr` may be `nullptr`, in
> which case `free` is a no-op. `ptr` must have been returned by `et_pal_allocate()`
> (`[spec:et:sem:posix.et-pal-allocate-fn]`). Returns void. No PAL-init check.

> [spec:et:def:posix.et-pal-init-fn]
> void et_pal_init(void)

> [spec:et:sem:posix.et-pal-init-fn]
> Weak default POSIX PAL initializer (may be overridden; `#pragma weak` under MSVC).
> Idempotent one-time setup of the timing baseline. Takes no arguments, returns void.
>
> Steps:
> 1. If the module-static `initialized` flag is already true, return immediately
>    (idempotent; a second call does not reset the start time).
> 2. Otherwise set the module-static `systemStartTime` to
>    `std::chrono::steady_clock::now()`, establishing the zero point for
>    `et_pal_current_ticks()` (`[spec:et:sem:posix.et-pal-current-ticks-fn]`).
> 3. Set `initialized = true`.
>
> Not thread-safe (plain non-atomic `bool` and `time_point`). Should be called
> before any other PAL function; the debug `_ASSERT_PAL_INITIALIZED` checks in the
> ticks/log functions enforce this in `!NDEBUG` builds.

> [spec:et:def:posix.et-pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)

> [spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn]
> Weak default POSIX tick-to-nanosecond ratio (may be overridden; `#pragma weak`
> under MSVC). Takes no arguments; returns an `et_tick_ratio_t` value
> `{numerator = 1, denominator = 1}`, i.e. one system tick equals one nanosecond,
> consistent with `et_pal_current_ticks` returning nanosecond-resolution timestamps
> (`[spec:et:sem:posix.et-pal-current-ticks-fn]`). Consumed by
> `[spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]`. No side effects, no
> PAL-init check.

