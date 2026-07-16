# runtime/backend/interface.cpp, runtime/backend/interface.h

> [spec:et:def:interface.executorch.et-runtime-namespace.backend]
> struct Backend {
>   const char* name;
>   BackendInterface* backend;
> }

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface]
> class BackendInterface {
>   ET_NODISCARD virtual Result<DelegateHandle*>;
> }

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.backend-interface-fn]
> BackendInterface::~BackendInterface()

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.backend-interface-fn]
> The out-of-line definition of `BackendInterface`'s pure-virtual
> destructor. It has an empty body and performs no work; it exists only
> because C++ requires an implementation for a pure-virtual destructor so
> that derived-class destruction can chain through the base. In the Rust
> port `BackendInterface` maps to a trait; this rule carries no behavior
> beyond "base cleanup is a no-op" and needs no direct analogue.

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.destroy-fn]
> virtual void destroy(ET_UNUSED DelegateHandle* handle) const

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.destroy-fn]
> Virtual hook, called by the runtime when a delegated method (execution
> plan) is being torn down, to release any resources associated with a
> `DelegateHandle` returned by `init()`. `handle` is that opaque
> backend-private pointer (may be null). The base-class implementation is
> a no-op (empty body, `handle` unused) — backends that allocate nothing
> needing explicit release can inherit it. Backends that must free
> resources override this. Returns nothing. The runtime is responsible
> for calling this exactly once per successfully-initialized handle at
> program teardown; the method itself performs no lifetime bookkeeping.

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.execute-fn]
> ET_NODISCARD virtual Error execute( BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const = 0

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.execute-fn]
> Pure-virtual (`= 0`) method every backend must implement. Runs the
> delegated method identified by `handle` (the opaque `DelegateHandle*`
> returned from `init()`), using `context` (a `BackendExecutionContext`
> providing runtime services such as temp allocation and event tracing)
> and `args` (a `Span<EValue*>` holding the method's inputs and outputs;
> the backend reads input EValues and writes results into the output
> EValues in place). Returns `Error::Ok` on success, or any other `Error`
> code on failure. No default implementation exists at this interface
> level — behavior is entirely backend-defined; this rule only fixes the
> contract (inputs/outputs passed via `args`, success signaled by
> `Error::Ok`).

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn]
> ET_NODISCARD virtual Error get_option( __ET_UNUSED BackendOptionContext& context, executorch::runtime::Span<BackendOption>& backend_options)

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn]
> Virtual method by which a backend fills caller-provided option slots
> with its current values. `context` is a `BackendOptionContext`
> (currently unused, marked `__ET_UNUSED`). `backend_options` is a
> mutable `Span<BackendOption>&`: the caller pre-populates each entry's
> `key`, and the backend is expected to write the corresponding current
> `value` into each entry. The base-class implementation is a no-op that
> ignores `backend_options` and returns `Error::Ok`; backends that expose
> readable options override it. Returns `Error::Ok` on success, otherwise
> an `Error` code. Distinct from the free function `get_option` per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.get-option-fn]`,
> which dispatches to this method after resolving the backend by name.

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.init-fn]
> init( BackendInitContext& context, FreeableBuffer* processed, ArrayRef<CompileSpec> compile_specs) const = 0

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.init-fn]
> Pure-virtual (`= 0`) method every backend must implement. Called once
> per program initialization to ready a delegated method for execution:
> the backend may compile/transform/optimize the preprocessed blob and
> perform device-dependent setup. `context` is a `BackendInitContext`
> (runtime services, e.g. allocators and the named-data map). `processed`
> is a `FreeableBuffer*` holding the opaque, backend-specific compiled
> unit produced ahead-of-time; the backend may call `processed->Free()`
> to reclaim its memory once it no longer needs the data after init.
> `compile_specs` is an `ArrayRef<CompileSpec>` — the exact compile
> specification used ahead-of-time to produce `processed`.
>
> On success returns a `Result` holding an opaque `DelegateHandle*` that
> represents the initialized method; this handle is later passed to
> `execute()` and `destroy()`, and the memory it points to is owned by
> the backend. On failure returns a `Result` holding an `Error` other
> than `Error::Ok`; specifically, if the preprocessed unit is
> incompatible with the current backend runtime the backend should return
> `Error::DelegateInvalidCompatibility`. Behavior is otherwise
> backend-defined; this rule fixes only the contract.

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.is-available-fn]
> ET_NODISCARD virtual bool is_available() const = 0

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.is-available-fn]
> Pure-virtual (`= 0`) predicate every backend must implement. Returns
> `true` if the backend is present and able to process delegation calls
> (e.g. required hardware/driver/library is available at runtime),
> `false` otherwise. Const, side-effect-free query. The runtime consults
> it before dispatching to a backend. No default implementation;
> behavior is backend-defined.

> [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn]
> ET_NODISCARD virtual Error set_option( __ET_UNUSED BackendOptionContext& context, const executorch::runtime::Span<BackendOption>& backend_options)

> [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn]
> Virtual method by which a backend applies caller-supplied options to
> its internal state. `context` is a `BackendOptionContext` (currently
> unused, marked `__ET_UNUSED`). `backend_options` is a `const
> Span<BackendOption>&` — the list of key/value options to apply. The
> base-class implementation is a no-op that ignores `backend_options` and
> returns `Error::Ok`; backends that accept configuration override it to
> parse the options and update their status. Returns `Error::Ok` on
> success, otherwise an `Error` code. Distinct from the free function
> `set_option` per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.set-option-fn]`,
> which dispatches to this method after resolving the backend by name.

> [spec:et:def:interface.executorch.et-runtime-namespace.compile-spec]
> struct CompileSpec {
>   const char* key;
>   SizedBuffer value;
> }

> [spec:et:def:interface.executorch.et-runtime-namespace.get-backend-class-fn]
> BackendInterface* get_backend_class(const char* name)

> [spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-class-fn]
> Resolves a registered backend by name. Iterates the global
> `registered_backends` table from index `0` to `num_registered_backends
> - 1` in ascending (registration) order. For each entry, compares its
> `name` field against the argument `name` using `strcmp` (C-string
> equality). On the first match returns that entry's `backend`
> (`BackendInterface*`). If no entry matches after scanning all
> registered backends, returns `nullptr`. Read-only; does not modify the
> table. The global table is a fixed array of capacity 16
> (`kMaxRegisteredBackends`) and `num_registered_backends` is a separate
> global counter — process-wide mutable state (see TODO(T128866626) noting
> this global registration should eventually be removed).

> [spec:et:def:interface.executorch.et-runtime-namespace.get-backend-name-fn]
> Result<const char*> get_backend_name(size_t index)

> [spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-name-fn]
> Returns the name of the backend registered at position `index`. If
> `index >= num_registered_backends`, returns a `Result` holding
> `Error::InvalidArgument`. Otherwise returns a `Result` holding
> `registered_backends[index].name` (a `const char*` into the global
> table). Indices follow registration order (index 0 = first registered).

> [spec:et:def:interface.executorch.et-runtime-namespace.get-num-registered-backends-fn]
> size_t get_num_registered_backends()

> [spec:et:sem:interface.executorch.et-runtime-namespace.get-num-registered-backends-fn]
> Returns the current value of the global counter
> `num_registered_backends` (a `size_t`), i.e. how many backends have
> been successfully registered so far. No side effects.

> [spec:et:def:interface.executorch.et-runtime-namespace.get-option-fn]
> Error get_option( const char* backend_name, executorch::runtime::Span<executorch::runtime::BackendOption> backend_options)

> [spec:et:sem:interface.executorch.et-runtime-namespace.get-option-fn]
> Free function that retrieves options from a named backend. Resolves the
> backend via `get_backend_class(backend_name)` per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-class-fn]`.
> If the result is null (no such backend), returns `Error::NotFound`.
> Otherwise constructs a default `BackendOptionContext`, forms a
> `Span<BackendOption>` over the caller's `backend_options` (same data
> pointer and size), and calls the backend's `get_option(context, span)`
> virtual method per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn]`.
> If that call returns anything other than `Error::Ok`, propagates that
> error unchanged; otherwise returns `Error::Ok`. The caller pre-fills
> keys in `backend_options`; on success the backend has written the
> corresponding values back into that same span.

> [spec:et:def:interface.executorch.et-runtime-namespace.register-backend-fn]
> Error register_backend(const Backend& backend)

> [spec:et:sem:interface.executorch.et-runtime-namespace.register-backend-fn]
> Registers a `Backend` (name + `BackendInterface*` pair) in the global
> table. Steps:
>
> Step 1 (capacity): if `num_registered_backends >= kMaxRegisteredBackends`
> (16), return `Error::Internal` without modifying the table.
>
> Step 2 (uniqueness): call `get_backend_class(backend.name)` per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-class-fn]`;
> if it returns non-null (a backend with that name already exists), return
> `Error::InvalidArgument` without modifying the table.
>
> Step 3 (insert): copy `backend` into
> `registered_backends[num_registered_backends]` and post-increment
> `num_registered_backends`. Return `Error::Ok`. The stored `Backend`
> copies the `name` pointer and `backend` pointer by value (the caller
> must keep both alive for the process lifetime). Not thread-safe:
> mutates process-global state without synchronization.

> [spec:et:def:interface.executorch.et-runtime-namespace.set-option-fn]
> Error set_option( const char* backend_name, const executorch::runtime::Span<executorch::runtime::BackendOption> backend_options)

> [spec:et:sem:interface.executorch.et-runtime-namespace.set-option-fn]
> Free function that applies options to a named backend. Resolves the
> backend via `get_backend_class(backend_name)` per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-class-fn]`.
> If the result is null (no such backend), returns `Error::NotFound`.
> Otherwise constructs a default `BackendOptionContext` and calls the
> backend's `set_option(context, backend_options)` virtual method per
> `[spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn]`,
> passing the caller's `Span<BackendOption>` through unchanged. If that
> call returns anything other than `Error::Ok`, propagates that error
> unchanged; otherwise returns `Error::Ok`.

> [spec:et:def:interface.executorch.et-runtime-namespace.sized-buffer]
> struct SizedBuffer {
>   void* buffer;
>   size_t nbytes;
> }

