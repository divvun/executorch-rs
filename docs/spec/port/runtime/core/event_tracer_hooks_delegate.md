# runtime/core/event_tracer_hooks_delegate.h

> [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn]
> inline void event_tracer_end_profiling_delegate( EventTracer* event_tracer, EventTracerEntry event_tracer_entry, const void* metadata = nullptr, size_t metadata_len = 0)

> [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn]
> Free function that conditionally signals the end of a delegate profiling
> event opened by
> `[spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn]`.
> Arguments: `event_tracer` (pointer, may be null), `event_tracer_entry` (the
> `EventTracerEntry` returned by the start call, passed by value), `metadata`
> (optional opaque `const void*`, default `nullptr`), and `metadata_len`
> (`size_t`, default `0`). The metadata bytes are opaque to the tracer; the
> pointer need not remain valid after the call (the tracer copies what it
> needs).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `event_tracer_entry`, `metadata`, and `metadata_len` are ignored; does
> nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->end_profiling_delegate(event_tracer_entry, metadata,
> metadata_len)`. If null, do nothing. Returns nothing.

> [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn]
> inline void event_tracer_log_output_delegate( EventTracer* event_tracer, const char* name, DebugHandle delegate_debug_id, const T& output)

> [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn]
> Function template (over `T`) that conditionally logs a delegate
> intermediate output value. Arguments: `event_tracer` (pointer, may be
> null), `name` (C-string name of the delegate event, or null when the
> delegate identifies ops by index), `delegate_debug_id` (`DebugHandle`; pass
> the unset handle / `-1` when using string names), and `output` (const
> reference to the value of type `T`).
>
> The type `T` is restricted at compile time (via a `static_assert`) to
> exactly one of: `int`, `bool`, `double`, `executorch::aten::Tensor`, or
> `ArrayRef<executorch::aten::Tensor>`. Any other type is a compile error
> ("Unsupported type for intermediate output"). A Rust port should accept
> only these output kinds (an integer, a boolean, a double, a single tensor,
> or a slice/array of tensors).
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `name`, `delegate_debug_id`, and `output` are ignored; does nothing. Note
> the compile-time type restriction is only enforced in the enabled build.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->log_intermediate_output_delegate(name, delegate_debug_id,
> output)`. If null, do nothing. Returns nothing.

> [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-profiling-delegate-fn]
> inline void event_tracer_log_profiling_delegate( EventTracer* event_tracer, const char* name, DebugHandle delegate_debug_id, et_timestamp_t start_time, et_timestamp_t end_time, const void* metadata = nullptr, size_t metadata_len = 0)

> [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-profiling-delegate-fn]
> Free function that conditionally logs a fully-specified delegate profiling
> event in a single call, for delegates that only obtain timing after the
> whole graph has run. Arguments: `event_tracer` (pointer, may be null),
> `name` (C-string name, or null when the delegate identifies ops by index),
> `delegate_debug_id` (`DebugHandle`; pass `-1` / the unset handle when using
> string names), `start_time` and `end_time` (`et_timestamp_t` start/end
> timestamps of the event), `metadata` (optional opaque `const void*`,
> default `nullptr`), and `metadata_len` (`size_t`, default `0`). Metadata is
> opaque and need not remain valid after the call.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `name`, `delegate_debug_id`, `start_time`, `end_time`, `metadata`, and
> `metadata_len` are ignored; does nothing.
>
> When event tracing is enabled: if `event_tracer` is non-null, call
> `event_tracer->log_profiling_delegate(name, delegate_debug_id, start_time,
> end_time, metadata, metadata_len)`. If null, do nothing. Returns nothing.

> [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn]
> inline EventTracerEntry event_tracer_start_profiling_delegate( EventTracer* event_tracer, const char* name, DebugHandle delegate_debug_id)

> [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn]
> Free function that conditionally begins a delegate profiling event and
> returns an `EventTracerEntry` token to be passed to
> `[spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn]`.
> Arguments: `event_tracer` (pointer, may be null), `name` (C-string name of
> the delegate event, or null when the delegate identifies ops by index and
> uses `delegate_debug_id` instead), and `delegate_debug_id` (`DebugHandle`;
> pass the unset debug handle when using string names). The tracer copies
> `name` internally if needed.
>
> When event tracing is disabled (`ET_EVENT_TRACER_ENABLED` not defined):
> `name` and `delegate_debug_id` are ignored and a default-constructed
> `EventTracerEntry` is returned (its value will be ignored by callers).
>
> When event tracing is enabled: if `event_tracer` is non-null, return
> `event_tracer->start_profiling_delegate(name, delegate_debug_id)`. If null,
> return a default-constructed `EventTracerEntry`.

