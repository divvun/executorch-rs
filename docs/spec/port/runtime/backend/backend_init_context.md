# runtime/backend/backend_init_context.h

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context]
> class BackendInitContext final {
>   MemoryAllocator* runtime_allocator_ = nullptr;
>   EventTracer* event_tracer_ = nullptr;
>   const char* method_name_ = nullptr;
>   const NamedDataMap* named_data_map_ = nullptr;
>   Span<const BackendOption> runtime_specs_;
> }

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn]
> explicit BackendInitContext( MemoryAllocator* runtime_allocator, EventTracer* event_tracer = nullptr, const char* method_name = nullptr, const NamedDataMap* named_data_map = nullptr, Span<const BackendOption> runtime_specs = {}) : runtim...

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn]
> Explicit constructor. Parameters: `runtime_allocator` (required `MemoryAllocator*`), and four optional args each defaulting to empty/null: `event_tracer = nullptr`, `method_name = nullptr`, `named_data_map = nullptr`, `runtime_specs = {}` (empty `Span<const BackendOption>`). Stores each into the matching member with no validation or ownership transfer (all raw non-owning pointers / span).
>
> Event-tracer gating: the stored `event_tracer_` depends on a compile-time flag. When `ET_EVENT_TRACER_ENABLED` is defined, `event_tracer_` is set to the passed `event_tracer`; otherwise `event_tracer_` is forced to `nullptr` regardless of the argument. All other members (`runtime_allocator_`, `method_name_`, `named_data_map_`, `runtime_specs_`) are always stored as given.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.event-tracer-fn]
> EventTracer* event_tracer()

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.event-tracer-fn]
> Accessor. Returns the stored `event_tracer_` pointer verbatim. Note this is `nullptr` whenever event tracing was disabled at compile time (`ET_EVENT_TRACER_ENABLED` undefined), regardless of what was passed to the constructor. No side effects, no ownership transfer.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-method-name-fn]
> const char* get_method_name() const

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-method-name-fn]
> Const accessor. Returns the stored `method_name_` C-string pointer verbatim (may be `nullptr`). This is the name of the method being initialized/loaded (usually "forward", but e.g. "prefill" or "decode" for multi-method .pte files). No side effects, no copy.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-named-data-map-fn]
> const NamedDataMap* get_named_data_map() const

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-named-data-map-fn]
> Const accessor. Returns the stored `named_data_map_` pointer verbatim (may be `nullptr`). Backends use this map to retrieve data blobs by key during init. No side effects, no ownership transfer.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-allocator-fn]
> MemoryAllocator* get_runtime_allocator()

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-allocator-fn]
> Accessor. Returns the stored `runtime_allocator_` pointer verbatim. This is the same runtime allocator used by the standard executor runtime; memory from it lives as long as the model. No side effects, no ownership transfer.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
> Result<T> get_runtime_spec(const char* key) const

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
> Templated lookup of a single runtime-spec (load-time option) by `key`, returning `Result<T>`. `T` is restricted at compile time (static_assert) to exactly one of `bool`, `int`, or `const char*`; instantiating with any other type is a compile error.
>
> Behavior:
> 1. Iterate the stored `runtime_specs_` span in order, `i` from 0 to `size()-1`.
> 2. For each option `opt`, compare `opt.key` against `key` with `std::strcmp`. First exact (0-return) match wins.
> 3. On a key match, resolve by type:
>    - If `T == const char*`: succeed only if the option's `value` variant currently holds `std::array<char, kMaxOptionValueLength>` (a stored string); return a `const char*` pointing at that array's data. If the variant does not hold a string, fall through to return `Error::InvalidArgument`.
>    - Otherwise (`T` is `bool` or `int`): succeed only if the variant currently holds a `T`; return the held value by copy. If the variant holds a different alternative, return `Error::InvalidArgument`.
> 4. If no key matched after scanning the whole span, return `Error::NotFound`.
>
> So: key found + type matches → the value; key found + type mismatch → `Error::InvalidArgument`; key absent → `Error::NotFound`. An empty `runtime_specs_` span always yields `Error::NotFound`.

> [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn]
> Span<const BackendOption> runtime_specs() const

> [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn]
> Const accessor. Returns the stored `runtime_specs_` span (`Span<const BackendOption>`) verbatim — the per-delegate load-time options passed at `Module::load()` time. Returns an empty span if none were provided. No copy of the underlying option data; the span is non-owning.

