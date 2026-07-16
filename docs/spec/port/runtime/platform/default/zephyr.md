# runtime/platform/default/zephyr.cpp

> [spec:et:def:zephyr.et-pal-abort-fn]
> ET_NORETURN void et_pal_abort(void)

> [spec:et:sem:zephyr.et-pal-abort-fn]
> Weak default Zephyr abort hook (may be overridden). Takes no arguments and never
> returns (`ET_NORETURN`). Unconditionally calls `_Exit(-1)`, terminating the
> process/thread immediately with exit status `-1` (a `_Exit` performs no atexit
> handlers, no stream flushing, no destructors). No argument validation, no logging,
> no return path. Unlike the POSIX variant (`[spec:et:sem:posix.et-pal-abort-fn]`,
> which raises `SIGABRT` via `std::abort()`), this uses `_Exit` for a clean
> immediate exit.

> [spec:et:def:zephyr.et-pal-allocate-fn]
> void* et_pal_allocate(size_t size)

> [spec:et:sem:zephyr.et-pal-allocate-fn]
> Weak default Zephyr allocator hook (may be overridden). Allocates `size` bytes
> from the Zephyr kernel heap via `k_malloc(size)` and returns the resulting
> pointer. Returns `nullptr` on allocation failure (or if the system heap is not
> configured). The returned memory is uninitialized and must be released with
> `et_pal_free()` (`[spec:et:sem:zephyr.et-pal-free-fn]`, which calls `k_free`). No
> argument validation; behavior for `size == 0` follows Zephyr `k_malloc`
> semantics. Not called directly by core runtime code — only via a MemoryAllocator
> wrapper.

> [spec:et:def:zephyr.et-pal-current-ticks-fn]
> et_timestamp_t et_pal_current_ticks(void)

> [spec:et:sem:zephyr.et-pal-current-ticks-fn]
> Weak default Zephyr timestamp hook (may be overridden). Takes no arguments;
> returns the current system uptime in Zephyr kernel ticks by calling
> `k_uptime_ticks()`, cast to `et_timestamp_t` (`uint64_t`). This is a monotonic
> counter of ticks elapsed since kernel boot (the Zephyr system-clock tick rate,
> `CONFIG_SYS_CLOCK_TICKS_PER_SEC`, not necessarily nanoseconds). No PAL-init check,
> no `systemStartTime` baseline subtraction (unlike POSIX/Android). Successive calls
> are non-decreasing.
>
> Note the tick-to-nanosecond ratio reported by
> `[spec:et:sem:zephyr.et-pal-ticks-to-ns-multiplier-fn]` is a placeholder `{1, 1}`
> and does not reflect the real Zephyr tick rate, so nanosecond conversions of this
> value are not physically accurate on real hardware.

> [spec:et:def:zephyr.et-pal-emit-log-message-fn]
> void et_pal_emit_log_message( et_timestamp_t timestamp, et_pal_log_level_t level, const char* filename, const char* function, size_t line, const char* message, size_t length)

> [spec:et:sem:zephyr.et-pal-emit-log-message-fn]
> Weak default Zephyr log sink (may be overridden). Emits one formatted log line to
> stderr via `fprintf`. Parameters `timestamp`, `length` are accepted but unused;
> `level` (an `et_pal_log_level_t` ASCII letter code), `filename`, `function`,
> `line`, and `message` (NUL-terminated) are consumed.
>
> Steps:
> 1. `fprintf` to stderr with the format `"%c [executorch:%s:%zu %s()] %s\n"` and
>    arguments, in order: `level` (the raw byte, e.g. 'D'/'I'/'E'/'F'), `filename`,
>    `line` (a `size_t`, `%zu`), `function`, then `message`, followed by a trailing
>    newline.
>
> Returns void. Unlike the POSIX variant (`[spec:et:sem:posix.et-pal-emit-log-message-fn]`),
> no timestamp decomposition or formatting is performed (the `timestamp` argument is
> ignored), the source `function` name is included, and there is no
> `_ASSERT_PAL_INITIALIZED` check and no explicit `fflush`.

> [spec:et:def:zephyr.et-pal-free-fn]
> void et_pal_free(void* ptr)

> [spec:et:sem:zephyr.et-pal-free-fn]
> Weak default Zephyr deallocator hook (may be overridden). Frees the memory at
> `ptr` back to the Zephyr kernel heap via `k_free(ptr)`. `ptr` may be `nullptr`,
> in which case `k_free` is a no-op. `ptr` must have been returned by
> `et_pal_allocate()` (`[spec:et:sem:zephyr.et-pal-allocate-fn]`, i.e. `k_malloc`).
> Returns void.

> [spec:et:def:zephyr.et-pal-init-fn]
> void et_pal_init(void)

> [spec:et:sem:zephyr.et-pal-init-fn]
> Weak default Zephyr PAL initializer (may be overridden). Takes no arguments,
> returns void. The body is empty: it performs no initialization and has no side
> effects. No timing baseline is established because
> `[spec:et:sem:zephyr.et-pal-current-ticks-fn]` reads kernel uptime directly
> rather than an init-relative reference. Trivially idempotent.

> [spec:et:def:zephyr.et-pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)

> [spec:et:sem:zephyr.et-pal-ticks-to-ns-multiplier-fn]
> Weak default Zephyr tick-to-nanosecond ratio (may be overridden). Takes no
> arguments; returns an `et_tick_ratio_t` value `{numerator = 1, denominator = 1}`.
> This is a placeholder identity ratio: the real target CPU frequency / Zephyr tick
> rate is unknown here (the implementation notes it just returns cycles from the FVP
> for `et_pal_current_ticks`), so the {1,1} ratio does NOT correctly convert the
> kernel ticks from `[spec:et:sem:zephyr.et-pal-current-ticks-fn]` into physical
> nanoseconds. Consumed by `[spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]`.
> No side effects, no PAL-init check.

