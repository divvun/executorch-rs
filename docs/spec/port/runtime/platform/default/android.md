# runtime/platform/default/android.cpp

> [spec:et:def:android.et-pal-abort-fn]
> ET_NORETURN void et_pal_abort(void)

> [spec:et:sem:android.et-pal-abort-fn]
> Weak default Android implementation of the PAL abort hook (may be overridden by
> a strong client definition). Takes no arguments and never returns
> (`ET_NORETURN`). Unconditionally calls `std::abort()`, terminating the process
> abnormally (raises `SIGABRT`) with no cleanup. No argument validation, no
> logging, no return path.

> [spec:et:def:android.et-pal-allocate-fn]
> void* et_pal_allocate(size_t size)

> [spec:et:sem:android.et-pal-allocate-fn]
> Weak default Android allocator hook (may be overridden). Allocates `size` bytes
> via the C `malloc(size)` and returns the resulting pointer. Returns `nullptr` on
> allocation failure. The returned memory is uninitialized and must be released
> with `et_pal_free()` (see `[spec:et:sem:android.et-pal-free-fn]`). No argument
> validation; `size == 0` behaves as C `malloc(0)` (implementation-defined: returns
> either `nullptr` or a unique freeable pointer). Not called directly by core
> runtime code — only via a MemoryAllocator wrapper.

> [spec:et:def:android.et-pal-current-ticks-fn]
> et_timestamp_t et_pal_current_ticks(void)

> [spec:et:sem:android.et-pal-current-ticks-fn]
> Weak default Android timestamp hook (may be overridden). Returns a monotonically
> non-decreasing timestamp in system ticks, where one tick equals one nanosecond
> (see `[spec:et:sem:android.et-pal-ticks-to-ns-multiplier-fn]`).
>
> Steps:
> 1. In non-`NDEBUG` (debug) builds, assert PAL initialization: if the module-static
>    `initialized` flag is false, emit an `ANDROID_LOG_FATAL` logcat message with
>    tag "ExecuTorch" reading "ExecuTorch PAL must be initialized before call to
>    <function>()" (function name from `ET_FUNCTION`). This assertion only logs and
>    does not abort or return early. In `NDEBUG` (release) builds the check is a
>    no-op.
> 2. Read the current `std::chrono::steady_clock::now()`.
> 3. Compute the elapsed duration since the module-static `systemStartTime`
>    (captured in `et_pal_init`), cast it to `std::chrono::nanoseconds`, and return
>    its integer `.count()` as an `et_timestamp_t` (`uint64_t`).
>
> If `et_pal_init` was never called, `systemStartTime` is default-constructed (the
> steady_clock epoch), so the returned value is the raw steady_clock reading since
> epoch. The clock is monotonic, so successive calls are non-decreasing.

> [spec:et:def:android.et-pal-emit-log-message-fn]
> void et_pal_emit_log_message( ET_UNUSED et_timestamp_t timestamp, et_pal_log_level_t level, ET_UNUSED const char* filename, ET_UNUSED const char* function, ET_UNUSED size_t line, const char* message, ET_UNUSED size_t length)

> [spec:et:sem:android.et-pal-emit-log-message-fn]
> Weak default Android log sink (may be overridden). Emits `message` to the Android
> logcat via `__android_log_print`. Parameters `timestamp`, `filename`, `function`,
> `line`, and `length` are all unused (`ET_UNUSED`); only `level` and `message` are
> consumed. `message` is a NUL-terminated C string; `length` is ignored.
>
> Steps:
> 1. In debug builds, assert PAL initialization as in
>    `[spec:et:sem:android.et-pal-current-ticks-fn]` (logs a FATAL logcat message if
>    `initialized` is false; no-op in `NDEBUG`).
> 2. Map the ExecuTorch severity `level` (an `et_pal_log_level_t`, whose values are
>    ASCII letter codes) to an Android log priority, defaulting to
>    `ANDROID_LOG_UNKNOWN`: `'D'` (kDebug) -> `ANDROID_LOG_DEBUG`, `'I'` (kInfo) ->
>    `ANDROID_LOG_INFO`, `'E'` (kError) -> `ANDROID_LOG_ERROR`, `'F'` (kFatal) ->
>    `ANDROID_LOG_FATAL`. Any other value (including `'?'`/kUnknown) leaves the
>    priority as `ANDROID_LOG_UNKNOWN`.
> 3. Call `__android_log_print(android_log_level, "ExecuTorch", "%s", message)` —
>    tag "ExecuTorch", the message passed through a "%s" format so it is logged
>    verbatim.
>
> Returns void. No timestamp formatting is performed (logcat supplies its own
> timestamps), unlike the POSIX variant `[spec:et:sem:posix.et-pal-emit-log-message-fn]`.

> [spec:et:def:android.et-pal-free-fn]
> void et_pal_free(void* ptr)

> [spec:et:sem:android.et-pal-free-fn]
> Weak default Android deallocator hook (may be overridden). Frees the memory at
> `ptr` via C `free(ptr)`. `ptr` may be `nullptr`, in which case `free` is a no-op.
> `ptr` must have been returned by `et_pal_allocate()`
> (`[spec:et:sem:android.et-pal-allocate-fn]`). Returns void.

> [spec:et:def:android.et-pal-init-fn]
> void et_pal_init(void)

> [spec:et:sem:android.et-pal-init-fn]
> Weak default Android PAL initializer (may be overridden). Idempotent one-time
> setup of the timing baseline. Takes no arguments, returns void.
>
> Steps:
> 1. If the module-static `initialized` flag is already true, return immediately
>    (idempotent; a second call does not reset the start time).
> 2. Otherwise set the module-static `systemStartTime` to
>    `std::chrono::steady_clock::now()`, establishing the zero point for
>    `et_pal_current_ticks()` (`[spec:et:sem:android.et-pal-current-ticks-fn]`).
> 3. Set `initialized = true`.
>
> Not thread-safe (plain non-atomic bool and time_point). Should be called before
> any other PAL function.

> [spec:et:def:android.et-pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)

> [spec:et:sem:android.et-pal-ticks-to-ns-multiplier-fn]
> Weak default Android tick-to-nanosecond ratio (may be overridden). Takes no
> arguments; returns an `et_tick_ratio_t` value `{numerator = 1, denominator = 1}`,
> i.e. one system tick equals one nanosecond. Consumed by
> `[spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]` and consistent with
> `et_pal_current_ticks` returning nanosecond-resolution timestamps. No side
> effects, no `_ASSERT_PAL_INITIALIZED` check.

