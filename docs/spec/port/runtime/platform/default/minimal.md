# runtime/platform/default/minimal.cpp

> [spec:et:def:minimal.et-pal-abort-fn]
> ET_NORETURN void et_pal_abort(void)

> [spec:et:sem:minimal.et-pal-abort-fn]
> Weak default minimal/fallback abort hook (may be overridden by a strong client
> definition). Takes no arguments and never returns (`ET_NORETURN`).
> Unconditionally calls the compiler builtin `__builtin_trap()`, which emits an
> illegal/trap instruction causing immediate abnormal termination (typically
> `SIGILL` or a hardware trap) with no cleanup and no dependency on any libc abort
> facility. Chosen because this fallback file makes no assumptions about the
> availability of `std::abort()` or similar. No argument validation, no return path.

> [spec:et:def:minimal.et-pal-allocate-fn]
> void* et_pal_allocate(ET_UNUSED size_t size)

> [spec:et:sem:minimal.et-pal-allocate-fn]
> Weak default minimal/fallback allocator hook (may be overridden). The `size`
> parameter is unused (`ET_UNUSED`). Performs no allocation and always returns
> `nullptr`. This fallback assumes no heap is available; clients that need
> allocation must provide a strong override. Returns void* (always null). No side
> effects.

> [spec:et:def:minimal.et-pal-current-ticks-fn]
> et_timestamp_t et_pal_current_ticks(void)

> [spec:et:sem:minimal.et-pal-current-ticks-fn]
> Weak default minimal/fallback timestamp hook (may be overridden). This fallback
> makes no assumptions about the availability of any wall-clock or monotonic-clock
> facility, so it does not read any real clock. Takes no arguments and returns a
> fixed sentinel constant: the `et_timestamp_t` (`uint64_t`) value `11223344`. The
> constant is deliberately a distinctive, easily grep-able number (rather than 0)
> so that its appearance in profiling/timing output signals that this weak stub is
> in use and a strong override was expected. No side effects, no PAL-init check.
> Because the value is constant, successive calls are equal (trivially
> non-decreasing) and any interval computed from it is 0.

> [spec:et:def:minimal.et-pal-emit-log-message-fn]
> void et_pal_emit_log_message( ET_UNUSED et_timestamp_t timestamp, ET_UNUSED et_pal_log_level_t level, ET_UNUSED const char* filename, ET_UNUSED const char* function, ET_UNUSED size_t line, ET_UNUSED const char* message, ET_UNUSED size_t ...

> [spec:et:sem:minimal.et-pal-emit-log-message-fn]
> Weak default minimal/fallback log sink (may be overridden). All seven parameters
> (`timestamp`, `level`, `filename`, `function`, `line`, `message`, `length`) are
> unused (`ET_UNUSED`). The body is empty: it performs no output, no formatting,
> and has no side effects. Log messages are silently discarded because this
> fallback assumes no console/serial/stderr facility is available. Returns void.

> [spec:et:def:minimal.et-pal-free-fn]
> void et_pal_free(ET_UNUSED void* ptr)

> [spec:et:sem:minimal.et-pal-free-fn]
> Weak default minimal/fallback deallocator hook (may be overridden). The `ptr`
> parameter is unused (`ET_UNUSED`). The body is empty and performs no operation.
> This is the counterpart to `[spec:et:sem:minimal.et-pal-allocate-fn]`, which
> never allocates (always returns `nullptr`), so there is nothing to free. Returns
> void, no side effects.

> [spec:et:def:minimal.et-pal-init-fn]
> void et_pal_init(void)

> [spec:et:sem:minimal.et-pal-init-fn]
> Weak default minimal/fallback PAL initializer (may be overridden). Takes no
> arguments, returns void. The body is empty: it performs no initialization and
> has no side effects. This fallback maintains no timing baseline or other global
> state (unlike the POSIX/Android variants), so there is nothing to set up. Safe to
> call any number of times (trivially idempotent).

> [spec:et:def:minimal.et-pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)

> [spec:et:sem:minimal.et-pal-ticks-to-ns-multiplier-fn]
> Weak default minimal/fallback tick-to-nanosecond ratio (may be overridden). Takes
> no arguments; returns an `et_tick_ratio_t` value `{numerator = 1, denominator =
> 1}`. Because this fallback defines no real tick rate, it returns the identity
> conversion ratio (1 tick == 1 ns). Consumed by
> `[spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]`. No side effects, no
> PAL-init check.

