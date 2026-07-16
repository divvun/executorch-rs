# runtime/core/event_tracer_hooks.h

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn]
> inline EventTracerEntry event_tracer_begin_profiling_event( EventTracer* event_tracer, char const* name)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn]
> Free function that conditionally begins a profiling event on the given
> tracer. Behavior depends on the compile-time flag `ET_EVENT_TRACER_ENABLED`.
>
> When event tracing is disabled (flag not defined): `event_tracer` and
> `name` are ignored and a default-constructed `EventTracerEntry` is
> returned. The port may model this as: if the tracing feature is compiled
> out, return the default `EventTracerEntry` value (whose fields are
> ignored by callers).
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->start_profiling(name)` and return its `EventTracerEntry`
> result. If `event_tracer` is null, return a default-constructed
> `EventTracerEntry`.
>
> `name` is a human-readable C-string identifying the event; the callee
> (`start_profiling`) is responsible for copying it if needed. The returned
> entry must be kept by the caller and passed to
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn]`
> to close the event.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-create-event-block-fn]
> inline void event_tracer_create_event_block( EventTracer* event_tracer, char const* name)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-create-event-block-fn]
> Free function that conditionally starts a new named event block on the
> tracer. Any events logged afterward are associated with this block.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `name` are ignored; the function does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->create_event_block(name)`. If null, do nothing. Returns
> nothing.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-enabled-fn]
> inline bool event_tracer_enabled()

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-enabled-fn]
> Free function that reports whether event-tracer code is compiled in.
> Returns the constant `true` when `ET_EVENT_TRACER_ENABLED` is defined at
> compile time, and the constant `false` otherwise. Takes no arguments and
> has no side effects. In the port this maps to a compile-time / build-time
> feature flag returning a boolean.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn]
> inline void event_tracer_end_profiling_event( EventTracer* event_tracer, EventTracerEntry event)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn]
> Free function that conditionally closes a profiling event previously
> opened by
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn]`.
> The `event` argument is the `EventTracerEntry` token returned by that
> begin call (passed by value).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `event` are ignored; the function does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->end_profiling(event)`. If null, do nothing. Returns
> nothing.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-fn]
> inline void event_tracer_log_evalue(EventTracer* event_tracer, EValue& evalue)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-fn]
> Free function that conditionally logs an intermediate `EValue` (passed by
> mutable reference) to the tracer.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `evalue` are ignored; the function does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, read the
> tracer's current debug-log level via
> `event_tracer->event_tracer_debug_level()`. Only if that level is greater
> than or equal to `EventTracerDebugLogLevel::kIntermediateOutputs` (i.e.
> intermediate-output logging is requested), call
> `event_tracer->log_evalue(evalue, LoggedEValueType::kIntermediateOutput)`.
> Otherwise, or if `event_tracer` is null, do nothing. Returns nothing.
> Contrast with
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn]`,
> which logs program outputs at the lower `kProgramOutputs` threshold.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn]
> inline void event_tracer_log_evalue_output( EventTracer* event_tracer, const EValue& evalue)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn]
> Free function that conditionally logs a program-output `EValue` (passed by
> const reference) to the tracer.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `evalue` are ignored; the function does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, read the
> tracer's current debug-log level via
> `event_tracer->event_tracer_debug_level()`. Only if that level is greater
> than or equal to `EventTracerDebugLogLevel::kProgramOutputs`, call
> `event_tracer->log_evalue(evalue, LoggedEValueType::kProgramOutput)`.
> Otherwise, or if `event_tracer` is null, do nothing. Returns nothing.
> Because `kProgramOutputs` is a lower threshold than
> `kIntermediateOutputs`, program outputs are logged even when intermediate
> output logging is disabled.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope]
> class EventTracerProfileInstructionScope final

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope.event-tracer-profile-instruction-scope-fn]
> EventTracerProfileInstructionScope( EventTracer* event_tracer, ChainID chain_idx, DebugHandle debug_handle)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope.event-tracer-profile-instruction-scope-fn]
> Constructor of the RAII `EventTracerProfileInstructionScope` class. It sets
> the tracer's current chain id and debug handle for the duration of the
> scope and resets them to defaults on destruction. Arguments: `event_tracer`
> (pointer, may be null), `chain_idx` (`ChainID`), `debug_handle`
> (`DebugHandle`).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined): the
> class holds no state, the constructor ignores all three arguments, and the
> destructor does nothing.
>
> When event tracing is enabled:
> - Constructor: store `event_tracer` in the member `event_tracer_`. If it is
>   null, return immediately (do nothing further). Otherwise call
>   `event_tracer_->set_chain_debug_handle(chain_idx, debug_handle)`.
> - Destructor: if `event_tracer_` is null, do nothing; otherwise call
>   `event_tracer_->set_chain_debug_handle(kUnsetChainId, kUnsetDebugHandle)`
>   to reset the chain id and debug handle to their unset defaults.
>
> A Rust port models this as a guard object whose `new`/constructor sets the
> chain/debug handle and whose `Drop` resets them to `kUnsetChainId` /
> `kUnsetDebugHandle`, both no-ops when the tracer is absent.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope]
> class EventTracerProfileMethodScope final

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn]
> EventTracerProfileMethodScope(EventTracer* event_tracer, const char* name)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn]
> Constructor of the RAII `EventTracerProfileMethodScope` class, used to
> profile a method for the lifetime of the scope object. Arguments:
> `event_tracer` (pointer, may be null) and `name` (human-readable C-string).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined): the
> class holds no state, the constructor ignores both arguments, and the
> destructor does nothing.
>
> When event tracing is enabled:
> - Constructor: store `event_tracer` in member `event_tracer_`. If it is
>   null, return immediately. Otherwise call
>   `event_entry_ = event_tracer->start_profiling(name)`, saving the returned
>   `EventTracerEntry` in member `event_entry_`.
> - Destructor: if `event_tracer_` is null, do nothing; otherwise call
>   `event_tracer_->end_profiling(event_entry_)`.
>
> Unlike
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn]`,
> this scope profiles unconditionally (no profiling-level gate) since it is
> intended for method-level profiling. A Rust port models it as a guard whose
> constructor starts profiling and whose `Drop` ends it, both no-ops when the
> tracer is absent.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope]
> class EventTracerProfileOpScope final

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn]
> EventTracerProfileOpScope(EventTracer* event_tracer, const char* name)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn]
> Constructor of the RAII `EventTracerProfileOpScope` class (aliased as
> `EventTracerProfileScope`), used to profile a single operator for the
> lifetime of the scope object. Arguments: `event_tracer` (pointer, may be
> null) and `name` (human-readable C-string). Profiling only occurs when the
> tracer's profiling level exceeds method-only; otherwise the object is a
> no-op.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined): the
> class holds no state, the constructor ignores both arguments, and the
> destructor does nothing.
>
> When event tracing is enabled:
> - Constructor: store `event_tracer` in member `event_tracer_`. If it is
>   null, return immediately. Otherwise read
>   `event_tracer_->event_tracer_profiling_level()`; only if that level is
>   strictly greater than
>   `executorch::runtime::EventTracerProfilingLevel::kProfileMethodOnly`
>   (i.e. operator-level profiling is enabled), call
>   `event_entry_ = event_tracer->start_profiling(name)` and store the
>   returned `EventTracerEntry` in member `event_entry_`. If the level is not
>   above `kProfileMethodOnly`, `event_entry_` is left uninitialized/default
>   and no profiling is started.
> - Destructor: if `event_tracer_` is null, do nothing. Otherwise re-check
>   the profiling level the same way; only if it is strictly greater than
>   `kProfileMethodOnly` call `event_tracer_->end_profiling(event_entry_)`.
>
> A Rust port models this as a guard that conditionally starts op profiling
> in its constructor and conditionally ends it in `Drop`, gated on the same
> profiling-level check in both places, and a full no-op when the tracer is
> null or op-level profiling is disabled.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-set-bundled-input-index-fn]
> inline void event_tracer_set_bundled_input_index( EventTracer* event_tracer, int bundled_input_index)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-set-bundled-input-index-fn]
> Free function that conditionally records which bundled input the method is
> currently using. Argument `bundled_input_index` is an `int`.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `bundled_input_index` are ignored; does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->set_bundled_input_index(bundled_input_index)`. If null, do
> nothing. Returns nothing.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn]
> inline void event_tracer_track_allocation( EventTracer* event_tracer, AllocatorID id, size_t size)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn]
> Free function that conditionally logs a single allocation performed by a
> previously registered allocator. Arguments: `id` (`AllocatorID` returned by
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn]`)
> and `size` (`size_t`, the number of bytes allocated).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer`, `id`, and `size` are ignored; does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->track_allocation(id, size)`. If null, do nothing. Returns
> nothing.

> [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn]
> inline AllocatorID event_tracer_track_allocator( EventTracer* event_tracer, const char* name)

> [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn]
> Free function that conditionally registers a named allocator with the
> tracer and returns an `AllocatorID` used to attribute subsequent
> allocations (see
> `[spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn]`).
> Argument `name` is a C-string identifying the allocator.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer` and `name` are ignored and the constant `0` is returned
> (this value will be ignored by callers).
>
> When event tracing is enabled: if `event_tracer` is non-null, return
> `event_tracer->track_allocator(name)`. If null, return `0`.

