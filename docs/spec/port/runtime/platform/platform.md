# runtime/platform/platform.cpp, runtime/platform/platform.h

> [spec:et:def:platform.et-pal-allocate-fn]
> void* et_pal_allocate(size_t size)

> [spec:et:sem:platform.et-pal-allocate-fn]
> Declares the weak PAL hook `et_pal_allocate(size_t size)`. This is the C-ABI
> platform-abstraction-layer entry point (marked with
> `ET_INTERNAL_PLATFORM_WEAKNESS`, i.e. a weak symbol), not an
> `executorch::runtime` dispatch wrapper. Core runtime code must not call it
> directly; only a MemoryAllocator wrapper may, and the runtime normally reaches
> allocation through `pal_allocate()` (see
> `[spec:et:sem:platform.executorch.runtime.pal-allocate-fn]`), which dispatches
> to whatever function pointer currently occupies the PAL table's `allocate`
> slot.
>
> Contract of any implementation of this hook: allocate `size` bytes of memory
> and return a pointer to it, or return `nullptr` on failure. The returned block
> must be releasable via `et_pal_free()`. The reference default implementation
> (POSIX, `[spec:et:sem:posix.et-pal-allocate-fn]`) simply returns
> `malloc(size)`. Because the symbol is weak, a client may override it with a
> strong definition; alternatively an override may be installed at runtime via
> `register_pal` (see
> `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`).

> [spec:et:def:platform.et-pal-current-ticks-fn]
> et_timestamp_t et_pal_current_ticks(void)

> [spec:et:sem:platform.et-pal-current-ticks-fn]
> Declares the weak PAL hook `et_pal_current_ticks(void)`. This is the C-ABI
> platform-abstraction-layer entry point (weak symbol via
> `ET_INTERNAL_PLATFORM_WEAKNESS`), not the `executorch::runtime` dispatch
> wrapper `pal_current_ticks()` (see
> `[spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn]`).
>
> Contract of any implementation: return a monotonically non-decreasing
> timestamp of type `et_timestamp_t` (a 64-bit integer), measured in system
> ticks. The tick-to-nanosecond ratio for these values is given by
> `et_pal_ticks_to_ns_multiplier()` (see
> `[spec:et:sem:platform.et-pal-ticks-to-ns-multiplier-fn]`). The reference
> default implementation (POSIX, `[spec:et:sem:posix.et-pal-current-ticks-fn]`)
> returns the number of nanoseconds elapsed since `et_pal_init()` measured with
> a steady clock. Being weak, the symbol may be overridden at link time, or the
> active implementation may be replaced at runtime via `register_pal`.

> [spec:et:def:platform.et-pal-current-ticks-t-void]
> typedef et_timestamp_t (*et_pal_current_ticks_t)(void)

> [spec:et:def:platform.et-pal-emit-log-message-fn]
> void et_pal_emit_log_message( et_timestamp_t timestamp, et_pal_log_level_t level, const char* filename, const char* function, size_t line, const char* message, size_t length)

> [spec:et:sem:platform.et-pal-emit-log-message-fn]
> Declares the weak PAL hook
> `et_pal_emit_log_message(timestamp, level, filename, function, line, message,
> length)`. This is the C-ABI platform-abstraction-layer entry point (weak
> symbol via `ET_INTERNAL_PLATFORM_WEAKNESS`), not the `executorch::runtime`
> dispatch wrapper `pal_emit_log_message()` (see
> `[spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn]`).
>
> Contract of any implementation: emit the log record to the platform's output
> (serial port, console, etc.). Parameters: `timestamp` is the event time in
> system ticks since boot; `level` is an `et_pal_log_level_t` whose value is a
> printable 7-bit ASCII character (`'D'`, `'I'`, `'E'`, `'F'`, or `'?'`);
> `filename` and `function` are source location strings; `line` is the source
> line; `message` is the (NUL-terminated) message text and `length` is its byte
> length. The reference default implementation (POSIX,
> `[spec:et:sem:posix.et-pal-emit-log-message-fn]`) formats a line of the form
> `"<level> HH:MM:SS.uuuuuu executorch:<filename>:<line>] <message>\n"` to
> stderr (deriving H/M/S/us from `timestamp` interpreted as nanoseconds) and
> flushes; it ignores `function` and `length`. Being weak, the symbol may be
> overridden at link time or replaced at runtime via `register_pal`.

> [spec:et:def:platform.et-pal-free-fn]
> void et_pal_free(void* ptr)

> [spec:et:sem:platform.et-pal-free-fn]
> Declares the weak PAL hook `et_pal_free(void* ptr)`. This is the C-ABI
> platform-abstraction-layer entry point (weak symbol via
> `ET_INTERNAL_PLATFORM_WEAKNESS`), not the `executorch::runtime` dispatch
> wrapper `pal_free()` (see `[spec:et:sem:platform.executorch.runtime.pal-free-fn]`).
>
> Contract of any implementation: free the memory block `ptr` that was
> previously returned by `et_pal_allocate()` (see
> `[spec:et:sem:platform.et-pal-allocate-fn]`). `ptr` may be `nullptr`, in which
> case the call must be a no-op. Returns nothing. The reference default
> implementation (POSIX, `[spec:et:sem:posix.et-pal-free-fn]`) simply calls
> `free(ptr)`. Being weak, the symbol may be overridden at link time, or the
> active implementation replaced at runtime via `register_pal` (see
> `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`).

> [spec:et:def:platform.et-pal-init-fn]
> void et_pal_init(void)

> [spec:et:sem:platform.et-pal-init-fn]
> Declares the weak PAL hook `et_pal_init(void)`. This is the C-ABI
> platform-abstraction-layer entry point (weak symbol via
> `ET_INTERNAL_PLATFORM_WEAKNESS`), not the `executorch::runtime` dispatch
> wrapper `pal_init()` (see `[spec:et:sem:platform.executorch.runtime.pal-init-fn]`).
>
> Contract of any implementation: initialize any global platform state. This
> function is intended to be called once before any other PAL function is used;
> it is idempotent-safe in the reference implementation. Takes no arguments and
> returns nothing. The reference default implementation (POSIX,
> `[spec:et:sem:posix.et-pal-init-fn]`) records a steady-clock start instant used
> as the zero point for `et_pal_current_ticks()`. Being weak, the symbol may be
> overridden at link time, or the active implementation replaced at runtime via
> `register_pal` (see `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`).

> [spec:et:def:platform.et-pal-log-level-t]
> typedef enum

> [spec:et:def:platform.et-pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)

> [spec:et:sem:platform.et-pal-ticks-to-ns-multiplier-fn]
> Declares the weak PAL hook `et_pal_ticks_to_ns_multiplier(void)`. This is the
> C-ABI platform-abstraction-layer entry point (weak symbol via
> `ET_INTERNAL_PLATFORM_WEAKNESS`), not the `executorch::runtime` dispatch
> wrapper `pal_ticks_to_ns_multiplier()` (see
> `[spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]`).
>
> Contract of any implementation: return an `et_tick_ratio_t` (a struct of two
> `uint64_t` fields `{ numerator, denominator }`) describing the conversion from
> system ticks to nanoseconds, such that
> `nanoseconds = ticks * numerator / denominator`. The reference default
> implementation (POSIX, `[spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn]`)
> returns `{ 1, 1 }` because its `et_pal_current_ticks()` already measures
> nanoseconds. Takes no arguments and has no side effects. Being weak, the symbol
> may be overridden at link time, or the active implementation replaced at
> runtime via `register_pal` (see
> `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`).

> [spec:et:def:platform.et-tick-ratio-t]
> typedef struct

> [spec:et:def:platform.executorch.runtime.get-pal-impl-fn]
> const PalImpl* get_pal_impl()

> [spec:et:sem:platform.executorch.runtime.get-pal-impl-fn]
> Returns a pointer to the singleton `PalImpl` function table currently in
> effect. Behavior: returns `&pal_impl`, the address of the file-local static
> `PalImpl` singleton defined in this translation unit. The singleton is
> zero-cost, statically initialized (no global constructor) to the default table
> `{ et_pal_init, et_pal_abort, et_pal_current_ticks,
> et_pal_ticks_to_ns_multiplier, et_pal_emit_log_message, et_pal_allocate,
> et_pal_free, __FILE__ }`, where each function pointer is the weak `et_pal_*`
> hook and `source_filename` is this source file's path. Takes no arguments,
> performs no validation, has no side effects, and never returns null. The
> returned pointer aliases the live table, so any subsequent `register_pal` (see
> `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`) is observable
> through it.

> [spec:et:def:platform.executorch.runtime.pal-abort-fn]
> ET_NORETURN void pal_abort()

> [spec:et:sem:platform.executorch.runtime.pal-abort-fn]
> `executorch::runtime` dispatch wrapper that aborts execution. Marked
> `ET_NORETURN` (must never return to its caller).
>
> Steps:
> 1. Call the currently-installed abort function via the PAL table:
>    `pal_impl.abort()` (the `abort` slot of the singleton `PalImpl`; default is
>    the weak `et_pal_abort`).
> 2. As a safety net in case the installed implementation does return (violating
>    its `ET_NORETURN` contract), call `std::abort()` to force process
>    termination.
>
> Takes no arguments, returns nothing (control never returns). No validation is
> performed; the `abort` slot is assumed non-null (it is non-null in the default
> table and `register_pal` never clears it to null).

> [spec:et:def:platform.executorch.runtime.pal-allocate-fn]
> void* pal_allocate(size_t size)

> [spec:et:sem:platform.executorch.runtime.pal-allocate-fn]
> `executorch::runtime` dispatch wrapper for allocation. Behavior: returns
> `pal_impl.allocate(size)` verbatim — i.e. forwards `size` to the currently
> installed `allocate` slot of the singleton `PalImpl` (default is the weak
> `et_pal_allocate`, POSIX `malloc`) and returns its result. Returns a pointer to
> `size` bytes of memory, or `nullptr` on failure; the block must later be
> released via `pal_free` (see
> `[spec:et:sem:platform.executorch.runtime.pal-free-fn]`). No validation is
> performed here; the `allocate` slot is assumed non-null. As noted in the
> header, core runtime code must not call this directly — only a MemoryAllocator
> wrapper may.

> [spec:et:def:platform.executorch.runtime.pal-current-ticks-fn]
> et_timestamp_t pal_current_ticks()

> [spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn]
> `executorch::runtime` dispatch wrapper for the current timestamp. Behavior:
> returns `pal_impl.current_ticks()` verbatim — i.e. calls the currently
> installed `current_ticks` slot of the singleton `PalImpl` (default is the weak
> `et_pal_current_ticks`) and returns its `et_timestamp_t` (64-bit) result, a
> monotonically non-decreasing count of system ticks. Takes no arguments,
> performs no validation, has no side effects. The `current_ticks` slot is
> assumed non-null. Interpret ticks as nanoseconds using the ratio from
> `pal_ticks_to_ns_multiplier` (see
> `[spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]`).

> [spec:et:def:platform.executorch.runtime.pal-emit-log-message-fn]
> void pal_emit_log_message( et_timestamp_t timestamp, et_pal_log_level_t level, const char* filename, const char* function, size_t line, const char* message, size_t length)

> [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn]
> `executorch::runtime` dispatch wrapper for log emission. Signature:
> `pal_emit_log_message(timestamp, level, filename, function, line, message,
> length)`. Behavior: forwards all seven arguments unchanged, in the same order,
> to the currently installed `emit_log_message` slot of the singleton `PalImpl`
> (default is the weak `et_pal_emit_log_message`):
> `pal_impl.emit_log_message(timestamp, level, filename, function, line, message,
> length)`. Returns nothing. No validation, transformation, or filtering is
> performed here; the `emit_log_message` slot is assumed non-null. This is the
> function that `vlogf` calls to actually output a formatted log line (see
> `[spec:et:sem:log.executorch.runtime.internal.vlogf-fn]`).

> [spec:et:def:platform.executorch.runtime.pal-free-fn]
> void pal_free(void* ptr)

> [spec:et:sem:platform.executorch.runtime.pal-free-fn]
> `executorch::runtime` dispatch wrapper for freeing memory. Behavior: calls
> `pal_impl.free(ptr)` — forwards `ptr` to the currently installed `free` slot of
> the singleton `PalImpl` (default is the weak `et_pal_free`, POSIX `free`).
> Returns nothing. `ptr` may be `nullptr` (the installed implementation must
> treat that as a no-op). No validation is performed here; the `free` slot is
> assumed non-null. Frees a block previously obtained from `pal_allocate` (see
> `[spec:et:sem:platform.executorch.runtime.pal-allocate-fn]`).

> [spec:et:def:platform.executorch.runtime.pal-impl]
> struct PalImpl {
>   pal_init_method init = nullptr;
>   pal_abort_method abort = nullptr;
>   pal_current_ticks_method current_ticks = nullptr;
>   pal_ticks_to_ns_multiplier_method ticks_to_ns_multiplier = nullptr;
>   pal_emit_log_message_method emit_log_message = nullptr;
>   pal_allocate_method allocate = nullptr;
>   pal_free_method free = nullptr;
>   const char* source_filename;
> }

> [spec:et:def:platform.executorch.runtime.pal-impl.create-fn]
> PalImpl PalImpl::create( pal_init_method init, pal_abort_method abort, pal_current_ticks_method current_ticks, pal_ticks_to_ns_multiplier_method ticks_to_ns_multiplier, pal_emit_log_message_method emit_log_message, pal_allocate_method al...

> [spec:et:sem:platform.executorch.runtime.pal-impl.create-fn]
> Static factory that constructs a `PalImpl` function table from explicit slot
> values. This is the 8-argument overload:
> `PalImpl::create(init, abort, current_ticks, ticks_to_ns_multiplier,
> emit_log_message, allocate, free, source_filename)`.
>
> Behavior: performs no validation and no side effects; simply aggregate-
> initializes and returns a `PalImpl` whose fields are set, in declaration order,
> to the eight arguments: `{ init, abort, current_ticks, ticks_to_ns_multiplier,
> emit_log_message, allocate, free, source_filename }`. Any of the function-
> pointer arguments may be `nullptr`; a null slot means "no override" when the
> result is later passed to `register_pal` (see
> `[spec:et:sem:platform.executorch.runtime.register-pal-fn]`), which leaves the
> corresponding default slot unchanged.
>
> There is also a convenience 2-argument overload
> `PalImpl::create(emit_log_message, source_filename)` that delegates to this one
> with all other function-pointer slots passed as `nullptr` (i.e. it overrides
> only the log emitter). The struct deliberately has no constructors so the
> default singleton can be statically initialized without a global constructor;
> `create` exists as the sanctioned way to build an override table.

> [spec:et:def:platform.executorch.runtime.pal-init-fn]
> void pal_init()

> [spec:et:sem:platform.executorch.runtime.pal-init-fn]
> `executorch::runtime` dispatch wrapper that initializes the platform. Behavior:
> calls `pal_impl.init()` — invokes the currently installed `init` slot of the
> singleton `PalImpl` (default is the weak `et_pal_init`). Takes no arguments,
> returns nothing. No validation is performed; the `init` slot is assumed
> non-null. This is invoked by `runtime_init` (see
> `[spec:et:sem:runtime.executorch.runtime.runtime-init-fn]`) to bring up global
> platform state before other PAL functions are used.

> [spec:et:def:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]
> et_tick_ratio_t pal_ticks_to_ns_multiplier()

> [spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]
> `executorch::runtime` dispatch wrapper for the tick-to-nanosecond ratio.
> Behavior: returns `pal_impl.ticks_to_ns_multiplier()` verbatim — calls the
> currently installed `ticks_to_ns_multiplier` slot of the singleton `PalImpl`
> (default is the weak `et_pal_ticks_to_ns_multiplier`) and returns its
> `et_tick_ratio_t` result (`{ numerator, denominator }`). To convert a tick
> count, compute `ticks * numerator / denominator` to obtain nanoseconds. Takes
> no arguments, performs no validation, has no side effects. The
> `ticks_to_ns_multiplier` slot is assumed non-null.

> [spec:et:def:platform.executorch.runtime.register-pal-fn]
> bool register_pal(PalImpl impl)

> [spec:et:sem:platform.executorch.runtime.register-pal-fn]
> Installs user-provided PAL function pointers into the singleton `PalImpl`
> table, overriding only the non-null slots. Returns `true` (always, in the
> reference implementation).
>
> State: a file-local `static bool is_pal_overridden` (initially `false`) tracks
> whether any prior registration has occurred.
>
> Steps:
> 1. If `is_pal_overridden` is already `true`, emit a warning via
>    `ET_LOG(Error, ...)` noting that `register_pal()` was called multiple times
>    and subsequent calls override the previous implementation, including
>    `impl.source_filename` in the message (or the literal string `"unknown"`
>    when `impl.source_filename` is null). (Note: the message text refers to the
>    "previous implementation" but the source actually interpolates the incoming
>    `impl.source_filename`.)
> 2. Set `is_pal_overridden = true`.
> 3. For each of `abort`, `current_ticks`, `ticks_to_ns_multiplier`,
>    `emit_log_message`, `allocate`, and `free`: if the corresponding field of
>    `impl` is non-null, copy it into the matching slot of the singleton
>    `pal_impl`; if it is null, leave that slot unchanged (keeping the current
>    default/previous value). These six are handled in that order.
> 4. `init` is handled last and specially: if `impl.init` is non-null, copy it
>    into `pal_impl.init` and then immediately call `pal_impl.init()` (running the
>    newly installed initializer). If `impl.init` is null, `pal_impl.init` is left
>    unchanged and no init call is made.
> 5. Return `true`.
>
> No thread-safety or ordering guarantees are provided; `impl` is passed by value
> (a copy of the caller's table). A typical override table is built with
> `PalImpl::create` (see
> `[spec:et:sem:platform.executorch.runtime.pal-impl.create-fn]`).

