# backends/xnnpack/runtime/XNNPACKBackend.cpp, backends/xnnpack/runtime/XNNPACKBackend.h

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack.workspace-sharing-mode]
> enum class WorkspaceSharingMode {
>   Disabled = 0;
>   PerModel = 1;
>   Global = 2;
>   Count;
> }

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend]
> class XnnpackBackend final {
>   mutable xnnpack::XnnpackBackendOptions options_;
> }

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn]
> void destroy(DelegateHandle* handle) const override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn]
> Tears down a delegate handle previously returned by `init`. The `handle`
> points to an `XNNExecutor` that was placement-constructed in the runtime
> allocator, so it must be destroyed manually here (its destructor is not
> trivial and the allocator will not call it). Returns void.
>
> Steps:
> 1. If `handle` is null, do nothing and return (no-op).
> 2. Cast `handle` to `XNNExecutor*` (`executor`).
> 3. Fetch `workspace = executor->get_workspace()` (a shared_ptr, per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn]`)
>    and `cache = executor->get_weights_cache()` (a possibly-empty shared_ptr,
>    per `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn]`).
>    Holding a local copy of `cache` keeps the weights-cache instance alive
>    through the `delete_packed_data` call below even if this executor was
>    its last owner.
> 4. If `cache` is non-empty, lock its mutex (`cache->mutex()`) for the
>    remainder of the function, serializing against sibling executors that
>    share the same file-backed cache. An empty `cache` means this PTE did
>    not opt into file-backed weight caching, and no lock is taken.
> 5. If compiled with `ENABLE_XNNPACK_PROFILING`, call
>    `executor->print_avg_op_timings()` (profiling-only; omit when the
>    profiling feature is disabled).
> 6. If `cache` is non-empty AND `executor->uses_weight_cache()` is true
>    (per `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn]`),
>    call `cache->delete_packed_data(executor->get_packed_data_names())` to
>    release this executor's packed weight entries from the shared cache.
> 7. Acquire the workspace lock via `workspace->acquire()` (destructuring
>    `[raii_lock, _]`) to serialize `xnn_delete_runtime`, which is not thread
>    safe, against concurrent `destroy`/`execute` on the same backend. The
>    workspace shared_ptr is held locally because the executor (which owns
>    the mutex the runtime deleter touches) is about to be destroyed.
> 8. Call `executor->~XNNExecutor()` explicitly (which, via the executor's
>    `runtime_` unique_ptr deleter, invokes `xnn_delete_runtime`). See
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn]`.
>    The allocator memory itself is not freed here; only the object is
>    destroyed. All held locks (cache mutex, workspace lock) release on
>    return.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.execute-fn]
> Error execute( BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.execute-fn]
> Runs one forward pass of a compiled XNNPACK delegate. `handle` is the
> `XNNExecutor*` from `init`; `args` is the span of `EValue*` for this
> CALL_DELEGATE (inputs followed by outputs, indexed by the external ids
> baked into the executor). Returns `Error::Ok` on success or the first
> failing `Error` encountered.
>
> Steps:
> 1. Cast `handle` to `XNNExecutor*` (`executor`).
> 2. `workspace = executor->get_workspace()`.
> 3. `cache = executor->get_weights_cache()`. If `cache` is non-empty, lock
>    its mutex for the duration of execute, serializing sibling executors
>    at the same cache path. Empty cache means the PTE did not opt into
>    file-backed weight caching (no lock taken).
> 4. Acquire the workspace lock via `workspace->acquire()` (destructuring
>    `[raii_lock, _]`); this enforces the workspace-sharing serialization
>    (only one delegate using a shared workspace runs at a time).
> 5. Call `err = executor->prepare_args(args)` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]`),
>    which binds each arg tensor's data pointer into the executor's external
>    values, reshapes inputs, propagates shapes through the runtime, and
>    resizes output tensors. If `err != Error::Ok`, return `err` immediately.
> 6. Call `err = executor->forward(context)` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn]`),
>    which sets up and invokes the XNNPACK runtime (with profiling around
>    the invocation). If `err != Error::Ok`, return `err` immediately.
> 7. Call `err = executor->convert_outputs(args)` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn]`),
>    which post-processes any Long output tensors from int32 to int64.
> 8. Return `err` (the result of step 7). All locks release on return.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn]
> Error get_option( BackendOptionContext& context, Span<BackendOption>& backend_options) override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn]
> Reads current backend option values into a caller-provided span of
> `BackendOption` entries. Each entry carries a key identifying which option
> to read; this fills in that entry's value in place.
>
> Steps:
> 1. Iterate `i` over `[0, backend_options.size())` in order.
> 2. For each entry, call `options_.get_option(backend_options[i])`, which
>    dispatches on the entry's key and writes the current value into it.
> 3. If any call returns `err != Error::Ok`, stop immediately and return
>    that `err` (fail-fast; later entries are not populated).
> 4. If all succeed, return `Error::Ok`.
>
> `context` (BackendOptionContext) is unused. The option-key dispatch lives
> in `XnnpackBackendOptions::get_option`.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn]
> Result<DelegateHandle*> init( BackendInitContext& context, FreeableBuffer* processed, ArrayRef<CompileSpec> compile_specs) const override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn]
> Compiles a serialized XNNPACK delegate blob (`processed`) into a live
> `XNNExecutor` and returns it as an opaque `DelegateHandle*`. `compile_specs`
> is accepted but not consumed here. Returns `Result<DelegateHandle*>`: the
> executor pointer on success, or an `Error` on failure. Marked `const`; the
> only mutable state is `options_` (declared `mutable`).
>
> Steps:
> 1. Allocate uninitialized storage for one `XNNExecutor` from the init
>    context's runtime allocator via
>    `context.get_runtime_allocator()->allocateInstance<XNNExecutor>()`. If
>    this returns null, return `Error::MemoryAllocationFailed`. The object is
>    NOT yet constructed (placement-new happens in step 6); the destructor
>    must be invoked manually in `destroy` /on the error paths below.
> 2. Get `named_data_map = context.get_named_data_map()`.
> 3. Compute `program_id = reinterpret_cast<uintptr_t>(
>    context.get_runtime_allocator())` — a stable per-program key derived from
>    the allocator pointer identity.
> 4. Resolve the workspace-sharing mode: `sharing_mode_result =
>    options_.resolve_sharing_mode(context)`. If not ok, return its error.
>    Then obtain the workspace:
>    `workspace_result = options_.workspace_manager().get_or_create_workspace(
>    program_id, sharing_mode)`. If not ok, return its error. `workspace =
>    workspace_result.get()`.
> 5. Weight-cache setup. `use_weight_cache = options_.resolve_weight_cache(
>    context)`. Declare an empty `weights_cache` shared_ptr and an unlocked
>    `lock_weights_cache`. If `use_weight_cache`:
>    - Determine the cache path: read the runtime spec keyed by
>      `packed_cache_path_option_key` via
>      `context.get_runtime_spec<const char*>(...)`; if that lookup is ok, use
>      its value as `cache_path`, else leave `cache_path` empty. Only the
>      per-PTE runtime_spec path is honored (never a backend-global path), so
>      a non-opt-in PTE cannot inherit another model's cache file.
>    - `wc_result = options_.get_or_create_weights_cache(cache_path)`; if not
>      ok, return its error. `weights_cache = wc_result.get()`.
>    - Lock `weights_cache->mutex()` into `lock_weights_cache` (held for the
>      rest of init).
>    - Call `weights_cache->initialize_for_runtime(
>      context.get_runtime_allocator(), named_data_map)`.
>    - Call `workspace->set_uses_weight_cache()`.
> 6. Acquire the workspace lock: `[workspace_lock, workspace_ptr] =
>    workspace->acquire()` (serializes access to the shared workspace during
>    compilation).
> 7. Placement-construct the executor in place: `new (executor)
>    XNNExecutor(workspace)`, initializing its `workspace_` member. See
>    `[spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor]`.
> 8. Compile: `err = XNNCompiler::compileModel(processed->data(),
>    processed->size(), executor, weights_cache.get(), workspace_ptr,
>    named_data_map, use_weight_cache)`. This populates the executor's runtime
>    and input/output ids.
> 9. Free the source blob unconditionally after compiling: `processed->Free()`
>    (the backend no longer needs it).
> 10. If `err != Error::Ok`: manually destroy the just-constructed executor
>    (`executor->~XNNExecutor()`) because `destroy` will not be called for a
>    handle that was never returned, log the failure, and return `err`.
> 11. On success, if `use_weight_cache`, hand the cache to the executor:
>    `executor->set_weights_cache(std::move(weights_cache))` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn]`),
>    so the shared_ptr outlives sibling executors sharing it.
> 12. Return `executor` as the `DelegateHandle*`. The weights-cache mutex and
>    workspace lock release on return.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn]
> bool is_available() const override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn]
> Reports whether the XNNPACK backend can run. Calls `xnn_initialize(
> /*allocator=*/nullptr)` (idempotent; safe to call repeatedly) and returns
> `true` iff it returns `xnn_status_success`, `false` otherwise. `const`, no
> arguments, no side effects beyond the (idempotent) XNNPACK initialization.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn]
> Error set_option( BackendOptionContext& context, const Span<BackendOption>& backend_options) override

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn]
> Applies a batch of backend option settings. Unlike `get_option`, this does
> NOT fail-fast: every option is applied even if an earlier one fails, so
> order-independent combinations (e.g. setting `packed_cache_path` and
> triggering `save_weight_cache_on_disk` in the same array) behave the same
> regardless of declaration order.
>
> Steps:
> 1. Initialize `first_err = Error::Ok`.
> 2. Iterate over each `option` in `backend_options` (in order).
> 3. For each, call `err = options_.set_option(option)`. All key dispatch —
>    including side effects like the one-shot on-disk weight-cache save — lives
>    inside `XnnpackBackendOptions::set_option`, which owns the weights-cache
>    instance and its mutex.
> 4. If `err != Error::Ok` and `first_err` is still `Error::Ok`, record
>    `first_err = err` (capture only the FIRST failure; do not overwrite it).
> 5. After processing all options, return `first_err` (`Error::Ok` if every
>    option succeeded).
>
> `context` (BackendOptionContext) is unused.

> [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.xnnpack-backend-fn]
> XnnpackBackend()

> [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.xnnpack-backend-fn]
> Constructs the `XnnpackBackend` singleton. Eagerly initializes the XNNPACK
> library by calling `xnn_initialize(/*allocator=*/nullptr)`.
>
> Steps:
> 1. Call `xnn_initialize(nullptr)`.
> 2. If the returned `xnn_status` is not `xnn_status_success`, log an error
>    (with the status value) and return early. There is no exception and no
>    stored failure flag: a failed init still yields a constructed backend
>    object; whether XNNPACK is actually usable is re-checked later via
>    `is_available` (per
>    `[spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn]`)
>    and by subsequent XNNPACK calls.
> 3. On success, no further action; construction completes.
>
> A single static instance is constructed at load time and registered as the
> backend under the key `xnnpack_backend_key` ("XnnpackBackend") via
> `register_backend`.

