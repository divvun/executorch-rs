# runtime/core/event_tracer.h

> [spec:et:def:event-tracer.executorch.runtime.allocator-id]
> typedef uint32_t AllocatorID

> [spec:et:def:event-tracer.executorch.runtime.chain-id]
> typedef int32_t ChainID

> [spec:et:def:event-tracer.executorch.runtime.debug-handle]
> typedef uint32_t DebugHandle

> [spec:et:def:event-tracer.executorch.runtime.delegate-debug-id-type]
> enum class DelegateDebugIdType {
>   kNone;
>   kInt;
>   kStr;
> }

> [spec:et:def:event-tracer.executorch.runtime.delegate-debug-int-id]
> typedef int32_t DelegateDebugIntId

> [spec:et:def:event-tracer.executorch.runtime.event-tracer]
> class EventTracer {
>   ChainID chain_id_ = kUnsetChainId;
>   DebugHandle debug_handle_ = kUnsetDebugHandle;
>   bool event_tracer_enable_debugging_ = false;
>   bool log_intermediate_tensors_ = false;
>   int bundled_input_index_ = kUnsetBundledInputIndex;
>   EventTracerDebugLogLevel event_tracer_debug_level_ = EventTracerDebugLogLevel::kNoLogging;
>   EventTracerProfilingLevel event_tracer_profiling_level_ = EventTracerProfilingLevel::kProfileAllEvents;
> }

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-debug-log-level]
> enum class EventTracerDebugLogLevel {
>   kNoLogging;
>   kProgramOutputs;
>   kIntermediateOutputs;
> }

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]
> struct EventTracerEntry {
>   int64_t event_id;
>   ChainID chain_id;
>   DebugHandle debug_handle;
>   et_timestamp_t start_time;
>   DelegateDebugIdType delegate_event_id_type;
> }

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base]
> class EventTracerFilterBase

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base.event-tracer-filter-base-fn]
> virtual ~EventTracerFilterBase() = default

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.event-tracer-filter-base-fn]
> Virtual (`= default`) destructor of the abstract base class `EventTracerFilterBase`.
> It exists solely so that deleting a derived filter object through an
> `EventTracerFilterBase*` pointer runs the correct derived destructor. The base
> class holds no data members, so this destructor performs no cleanup work of its
> own beyond the normal destruction of any (absent) base-class members. In a Rust
> port there is no direct analog: a trait object dropped through `Box<dyn Trait>`
> already runs the concrete type's `Drop`; this rule only requires that the
> filter trait be usable as a drop-safe owned/borrowed abstraction, with no
> additional teardown behavior defined at the base level.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]
> virtual Result<bool> filter( const char* name, DelegateDebugIntId delegate_debug_index) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]
> Pure virtual method (`= 0`); the base class defines no behavior — this is the
> abstract contract every concrete filter must satisfy. Signature:
> `filter(const char* name, DelegateDebugIntId delegate_debug_index) -> Result<bool>`.
>
> Contract that all implementations must honor:
> - Exactly one of the two identifiers is meaningful per call: if `name` is a
>   non-null string, then `delegate_debug_index` must be `kUnsetDelegateDebugIntId`
>   (-1, see `[spec:et:def:event-tracer.executorch.runtime.delegate-debug-int-id]`);
>   if `delegate_debug_index` is a real id (not -1), then `name` must be `nullptr`.
>   The caller is responsible for upholding this "only one is set" invariant.
> - Returns a `Result<bool>` (see `[spec:et:sem:result.result.result-fn]`-style
>   Result semantics): the ok value is `true` when the event matches the filter
>   criteria, `false` when the event does not match or is unknown, and the Result
>   instead carries an `Error` code if an error occurred while evaluating the
>   filter.
> - No output/side-effect semantics are mandated by the interface; a filter is a
>   pure decision function over (name XOR delegate_debug_index).

> [spec:et:def:event-tracer.executorch.runtime.event-tracer-profiling-level]
> enum class EventTracerProfilingLevel {
>   kProfileMethodOnly;
>   kProfileAllEvents;
> }

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn]
> int bundled_input_index()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn]
> Non-virtual inline accessor. Returns the protected member `bundled_input_index_`
> by value (an `int`). No arguments, no validation, no side effects. The value is
> `kUnsetBundledInputIndex` (-1) until `set_bundled_input_index` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn]`)
> is called; otherwise it is whatever value was last set. Pure getter.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.create-event-block-fn]
> virtual void create_event_block(const char* name) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.create-event-block-fn]
> Pure virtual (`= 0`); the base class provides no body. Abstract contract:
> begins a new named "event block" — a logical grouping of the profiling and/or
> debugging events that follow (e.g. all events produced during one `execute()`
> model-inference call are grouped into one block).
>
> Contract that implementations must honor:
> - `name` is a human-readable identifier for the block. The pointer is only
>   valid for the duration of this call: the implementation MUST copy the string
>   into its own internal storage during the call and must not retain the caller's
>   pointer afterward.
> - Returns nothing. Side effect only: the tracer's internal state advances so
>   that subsequently logged events are associated with this newly created block.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]
> ChainID current_chain_id()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]
> Non-virtual inline accessor. Returns the protected member `chain_id_` by value
> (a `ChainID`, i.e. `int32_t`). No arguments, no validation, no side effects.
> Initial value is `kUnsetChainId` (-1); becomes whatever was last set via
> `set_chain_debug_handle` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]`).
> Pure getter.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.current-debug-handle-fn]
> DebugHandle current_debug_handle()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-debug-handle-fn]
> Non-virtual inline accessor. Returns the protected member `debug_handle_` by
> value (a `DebugHandle`, i.e. `uint32_t`). No arguments, no validation, no side
> effects. Initial value is `kUnsetDebugHandle` (0); becomes whatever was last set
> via `set_chain_debug_handle` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]`).
> Pure getter.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn]
> virtual void end_profiling_delegate( EventTracerEntry event_tracer_entry, const void* metadata = nullptr, size_t metadata_len = 0) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `end_profiling_delegate(EventTracerEntry event_tracer_entry, const void* metadata = nullptr, size_t metadata_len = 0) -> void`.
> Abstract contract: signals completion of the delegate profiling event that was
> begun by a prior `start_profiling_delegate` call (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn]`).
>
> Contract implementations must honor:
> - `event_tracer_entry` MUST be the exact `EventTracerEntry` value returned by
>   the matching `start_profiling_delegate`; it identifies which event is ending
>   (carries event_id, chain_id, debug_handle, start_time, delegate_event_id_type,
>   see `[spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]`).
> - `metadata` is optional free-form, opaque bytes (default `nullptr`) of length
>   `metadata_len` bytes (default 0). Its contents/format are transparent to the
>   tracer — it stores/pipes them through verbatim for the user's post-processing.
>   The `metadata` pointer need not remain valid after this call returns, so an
>   implementation that keeps it MUST copy the bytes during the call.
> - Returns nothing. Side effect: records the event's end (typically computing
>   elapsed time from `event_tracer_entry.start_time` to now) plus any metadata.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.end-profiling-fn]
> virtual void end_profiling(EventTracerEntry prof_entry) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `end_profiling(EventTracerEntry prof_entry) -> void`. Abstract contract: signals
> completion of a non-delegate profiling event begun by a prior `start_profiling`
> call (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]`).
>
> Contract implementations must honor:
> - `prof_entry` MUST be the exact `EventTracerEntry` value returned by the
>   matching `start_profiling`; it identifies the event and carries its
>   `start_time` (see
>   `[spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]`).
> - Returns nothing. Side effect: finalizes the event, typically recording the
>   end timestamp and the elapsed duration from `prof_entry.start_time`.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn]
> EventTracerDebugLogLevel event_tracer_debug_level()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn]
> Non-virtual inline accessor. Returns the protected member
> `event_tracer_debug_level_` by value (an `EventTracerDebugLogLevel`, see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-debug-log-level]`).
> No arguments, no validation, no side effects. Initial value is
> `EventTracerDebugLogLevel::kNoLogging`; becomes whatever was last set via
> `set_event_tracer_debug_level` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn]`).
> Pure getter.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-fn]
> virtual ~EventTracer()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-fn]
> Virtual destructor of the abstract base class `EventTracer`, defined with an
> empty body (`{}`). It performs no explicit cleanup; it exists so that deleting a
> concrete tracer through an `EventTracer*` (or owned base pointer) dispatches to
> the derived destructor. The base holds only trivially-destructible members
> (`chain_id_`, `debug_handle_`, two bools, `bundled_input_index_`, and two enum
> level fields), so no member teardown work is required. In a Rust port this maps
> to using the tracer as a trait object where dropping the owner runs the concrete
> type's `Drop`; the base level defines no teardown behavior of its own.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn]
> EventTracerProfilingLevel event_tracer_profiling_level()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn]
> Non-virtual inline accessor. Returns the protected member
> `event_tracer_profiling_level_` by value (an `EventTracerProfilingLevel`, see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-profiling-level]`).
> No arguments, no validation, no side effects. Initial value is
> `EventTracerProfilingLevel::kProfileAllEvents`; becomes whatever was last set via
> `set_event_tracer_profiling_level` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn]`).
> Pure getter.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.intermediate-outputs-logging-status-fn]
> bool intermediate_outputs_logging_status()

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.intermediate-outputs-logging-status-fn]
> Non-virtual inline accessor. Returns the protected `bool` member
> `log_intermediate_tensors_`. No arguments, no validation, no side effects.
> Initial value is `false`. Note: this class defines no public setter for
> `log_intermediate_tensors_`; it is a protected field that a derived tracer sets
> directly (typically according to its configured debug level). Pure getter that
> reports whether intermediate-tensor logging is currently enabled.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-evalue-fn]
> virtual Result<bool> log_evalue( const EValue& evalue, LoggedEValueType evalue_type) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-evalue-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `log_evalue(const EValue& evalue, LoggedEValueType evalue_type) -> Result<bool>`.
> Abstract contract: records the value `evalue` produced during model execution
> for debugging/output capture.
>
> Contract implementations must honor:
> - `evalue` is the value to log (see EValue in `[spec:et:def:evalue...]`); it is
>   passed by const reference and its lifetime is only guaranteed for the call, so
>   any retained representation must be copied out during the call.
> - `evalue_type` (see
>   `[spec:et:def:event-tracer.executorch.runtime.logged-e-value-type]`) tags the
>   value as either `kIntermediateOutput` (0, an operator's intermediate output) or
>   `kProgramOutput` (1, a model/program-level output). Program outputs are the
>   special case logged with the "output" flag set.
> - Callers use the tracer's current chain_id / debug_handle (see
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]`
>   and `...current-debug-handle-fn`) to give each logged evalue its op context.
> - Returns `Result<bool>`: ok `true` when the evalue was successfully logged, or
>   an `Error` code carried in the Result when logging failed.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]
> virtual Result<bool> log_intermediate_output_delegate( const char* name, DelegateDebugIntId delegate_debug_index, const executorch::aten::Tensor& output) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]
> Pure virtual (`= 0`); base class provides no body. This rule anchors an
> overload family of `log_intermediate_output_delegate`, all sharing the same
> `(const char* name, DelegateDebugIntId delegate_debug_index, <output>)` shape
> and all returning `Result<bool>`. The five output overloads are:
> - `const executorch::aten::Tensor& output` (the one named by the def rule),
> - `const ArrayRef<executorch::aten::Tensor> output` (a tensor array),
> - `const int& output`,
> - `const bool& output`,
> - `const double& output`.
>
> Abstract contract shared by all overloads: logs one intermediate output value
> emitted by a delegate/backend, associated with a delegate event identified by
> either `name` or `delegate_debug_index`.
> - Identifier convention (same as filter, see
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]`):
>   `name` must be the exact delegate-mapping name assigned during ahead-of-time
>   export; if the delegate identifies ops by integer index instead of name, pass
>   `name = nullptr` and supply `delegate_debug_index`, otherwise pass
>   `delegate_debug_index = kUnsetDelegateDebugIntId`. The `name` pointer is only
>   valid for the call and must be copied into internal storage if retained.
> - `output` is the value to log, of the overload's type (tensor, tensor array,
>   int, bool, or double). References/ArrayRef are valid only for the call.
> - Returns `Result<bool>`: ok `true` when the output was logged; ok `false` when
>   the output was filtered out (by the delegation intermediate-output filter set
>   via
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn]`)
>   and therefore not logged; or an `Error` code in the Result on failure.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-profiling-delegate-fn]
> virtual void log_profiling_delegate( const char* name, DelegateDebugIntId delegate_debug_index, et_timestamp_t start_time, et_timestamp_t end_time, const void* metadata = nullptr, size_t metadata_len = 0) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-profiling-delegate-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `log_profiling_delegate(const char* name, DelegateDebugIntId delegate_debug_index, et_timestamp_t start_time, et_timestamp_t end_time, const void* metadata = nullptr, size_t metadata_len = 0) -> void`.
> Abstract contract: logs a single, already-completed delegate profiling event in
> one shot (as opposed to the paired start/end calls). This exists for delegates
> that only learn timing after the whole graph has run; it may be called
> repeatedly (e.g. in a loop) to emit any number of such events.
>
> Contract implementations must honor:
> - `name` / `delegate_debug_index` follow the same "one-of" delegate identifier
>   convention as
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]`:
>   pass the AOT delegate-mapping `name` (pointer valid only for the call, copy if
>   retained) with `delegate_debug_index = kUnsetDelegateDebugIntId`, or pass
>   `name = nullptr` with an integer `delegate_debug_index`.
> - `start_time` and `end_time` are the caller-supplied begin/end timestamps of
>   the delegate event (`et_timestamp_t`); the implementation records this
>   interval directly rather than measuring it itself.
> - `metadata` (default `nullptr`, length `metadata_len` bytes, default 0) is
>   optional opaque free-form data, transparent to the tracer, piped through for
>   post-processing. The pointer need not remain valid after the call, so retained
>   metadata must be copied during the call.
> - Returns nothing; side effect only: one delegate profiling event is recorded.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn]
> void set_bundled_input_index(int bundled_input_index)

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn]
> Non-virtual inline mutator. Assigns its `int` argument to the protected member
> `bundled_input_index_` (`bundled_input_index_ = bundled_input_index;`). No
> validation, no return value. Used, when running inside a bundled program, to
> record which bundled input index is currently being tested on this method. To
> reset to "unset", the caller passes `kUnsetBundledInputIndex` (-1). The value is
> later read back via
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn]`.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]
> void set_chain_debug_handle(ChainID chain_id, DebugHandle debug_handle)

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]
> Non-virtual inline mutator. Stores both arguments into protected members:
> `chain_id_ = chain_id;` then `debug_handle_ = debug_handle;`. No validation, no
> return value. This lets a caller (e.g. method.cpp before entering the codegen
> layer) preset the chain id and debug handle so that a subsequent
> `start_profiling` call made without explicit ids (i.e. with `kUnsetChainId` /
> `kUnsetDebugHandle`) picks up these stored values — see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]`.
> Explicit non-default ids passed to `start_profiling` take precedence over these
> stored values. The stored values are read back via
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]`
> and `...current-debug-handle-fn`. Intended for RAII-scoped use so the ids are
> reset (back to the unset defaults) after the corresponding `end_profiling`.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn]
> virtual void set_delegation_intermediate_output_filter( EventTracerFilterBase* event_tracer_filter) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `set_delegation_intermediate_output_filter(EventTracerFilterBase* event_tracer_filter) -> void`.
> Abstract contract: installs a filter object that the tracer consults to decide
> which delegate intermediate outputs to log. The filter is an
> `EventTracerFilterBase*` (see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base]`); its
> `filter(name, delegate_debug_index)` method (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]`)
> returning `false` causes the corresponding
> `log_intermediate_output_delegate` call to report the output as filtered-out
> (ok `false`, not logged — see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]`).
> Returns nothing. Side effect: the tracer retains the supplied pointer as its
> active delegation intermediate-output filter (ownership/lifetime of the pointed
> filter object is the caller's responsibility).

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn]
> void set_event_tracer_debug_level(EventTracerDebugLogLevel log_level)

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn]
> Non-virtual inline mutator. Assigns its argument to the protected member:
> `event_tracer_debug_level_ = log_level;`. `log_level` is an
> `EventTracerDebugLogLevel` (see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-debug-log-level]`).
> No validation, no return value. The stored value is read back via
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn]`.
> Note: this setter updates only `event_tracer_debug_level_`; it does not itself
> modify `log_intermediate_tensors_`.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn]
> void set_event_tracer_profiling_level( EventTracerProfilingLevel profiling_level)

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn]
> Non-virtual inline mutator. Assigns its argument to the protected member:
> `event_tracer_profiling_level_ = profiling_level;`. `profiling_level` is an
> `EventTracerProfilingLevel` (see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-profiling-level]`).
> No validation, no return value. The stored value is read back via
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn]`.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn]
> virtual EventTracerEntry start_profiling_delegate( const char* name, DelegateDebugIntId delegate_debug_index) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `start_profiling_delegate(const char* name, DelegateDebugIntId delegate_debug_index) -> EventTracerEntry`.
> Abstract contract: begins profiling of a delegate event and returns an
> `EventTracerEntry` (see
> `[spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]`) that
> uniquely identifies it; this entry must later be passed to
> `end_profiling_delegate` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn]`)
> to close the event.
>
> Contract implementations must honor:
> - `name` / `delegate_debug_index` follow the "one-of" delegate identifier
>   convention: pass the AOT delegate-mapping `name` (pointer valid only for the
>   call; copy if retained) with `delegate_debug_index = kUnsetDelegateDebugIntId`,
>   or pass `name = nullptr` with an integer `delegate_debug_index`.
> - The returned `EventTracerEntry` captures the start timestamp and, for delegate
>   events, sets `delegate_event_id_type` to `kInt` or `kStr` (per
>   `[spec:et:def:event-tracer.executorch.runtime.delegate-debug-id-type]`)
>   with `event_id` holding the integer delegate id or the string-table index
>   accordingly.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]
> virtual EventTracerEntry start_profiling( const char* name, ChainID chain_id = kUnsetChainId, DebugHandle debug_handle = kUnsetDebugHandle) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `start_profiling(const char* name, ChainID chain_id = kUnsetChainId, DebugHandle debug_handle = kUnsetDebugHandle) -> EventTracerEntry`.
> Abstract contract: begins profiling of a non-delegate event identified by
> `name` plus a (chain_id, debug_handle) pair, and returns an `EventTracerEntry`
> (see `[spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]`) that
> must later be passed to `end_profiling` (see
> `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-fn]`).
>
> Contract implementations must honor:
> - `name` is a human-readable event name; the pointer is valid only for the call
>   and must be copied into internal storage if retained.
> - Id fallback rule: if both `chain_id == kUnsetChainId` (-1) and
>   `debug_handle == kUnsetDebugHandle` (0) are passed (the defaults), the
>   implementation uses the ids previously stored on the tracer via
>   `set_chain_debug_handle` (see
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]`,
>   readable via `current_chain_id` / `current_debug_handle`). Non-default ids
>   passed here always take precedence over the stored values.
> - The returned entry records the start timestamp and the resolved
>   chain_id/debug_handle; `delegate_event_id_type` is `kNone` for these
>   non-delegate events.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.track-allocation-fn]
> virtual void track_allocation(AllocatorID id, size_t size) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocation-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `track_allocation(AllocatorID id, size_t size) -> void`. Abstract contract:
> records that an allocation of `size` bytes occurred against the allocator
> previously registered under `id`.
>
> Contract implementations must honor:
> - `id` is an `AllocatorID` (`uint32_t`) that MUST have been produced by an
>   earlier `track_allocator` call (see
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocator-fn]`);
>   it selects which tracked allocator this allocation belongs to.
> - `size` is the allocation size in bytes.
> - Returns nothing; side effect only: the allocation is logged/accumulated
>   against that allocator for later reporting.

> [spec:et:def:event-tracer.executorch.runtime.event-tracer.track-allocator-fn]
> virtual AllocatorID track_allocator(const char* name) = 0

> [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocator-fn]
> Pure virtual (`= 0`); base class provides no body. Signature:
> `track_allocator(const char* name) -> AllocatorID`. Abstract contract:
> registers a memory allocator under a human-readable `name` and returns a fresh
> `AllocatorID` (`uint32_t`) that uniquely identifies it for the lifetime of the
> tracer.
>
> Contract implementations must honor:
> - `name` is a human-readable allocator name; the pointer is valid only for the
>   call and must be copied into internal storage if retained.
> - The returned `AllocatorID` is subsequently supplied to `track_allocation`
>   (see
>   `[spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocation-fn]`)
>   to attribute individual allocations to this allocator. Ids should be unique
>   per registered allocator within the tracer instance.

> [spec:et:def:event-tracer.executorch.runtime.logged-e-value-type]
> enum class LoggedEValueType {
>   kIntermediateOutput = 0;
>   kProgramOutput = 1;
> }

