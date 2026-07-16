# backends/xnnpack/runtime/XnnpackBackendOptions.cpp, backends/xnnpack/runtime/XnnpackBackendOptions.h

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
> T resolve_option( const ET_RUNTIME_NAMESPACE::BackendInitContext& context, const char* key, T global_default)

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
> File-local template helper (anonymous namespace). Given a `BackendInitContext`,
> an option `key`, and a `global_default` of type `T`, returns the per-delegate
> runtime spec override for `key` if one is present, otherwise the `global_default`.
>
> Steps:
> - Query the context for a runtime spec value under `key` typed as `T`:
>   `context.get_runtime_spec<T>(key)`, which returns a `Result<T>`.
>   (`get_runtime_spec` only supports `T` in {bool, int, const char*}; here it is
>   instantiated for bool and int.)
> - If that `Result` is ok (an override exists and has the requested type), return
>   the override value `spec.get()`.
> - Otherwise return `global_default` unchanged.
>
> No logging, no mutation of the options object; a pure lookup-with-fallback.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options]
> class XnnpackBackendOptions {
>   XNNWorkspaceManager& workspace_manager();
>   const XNNWorkspaceManager& workspace_manager() const;
>   XNNWeightsCacheManager& weights_cache_manager();
>   const XNNWeightsCacheManager& weights_cache_manager() const;
>   XNNWorkspaceManager workspace_manager_;
>   XNNWeightsCacheManager weights_cache_manager_;
>   mutable std::mutex path_mutex_;
>   std::string packed_cache_path_;
> }

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn]
> Error XnnpackBackendOptions::get_option(BackendOption& option) const

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn]
> Const. Reads a single option identified by `option.key` and writes the current
> value into `option.value` (a `std::variant<bool, int, array<char, 256>>`, where
> 256 is `runtime::kMaxOptionValueLength`). Returns `Error::Ok`.
>
> Dispatch by exact string comparison (`strcmp == 0`) of `option.key`:
> - `workspace_sharing_mode_option_key` ("workspace_sharing_mode"): set
>   `option.value` to `static_cast<int>(sharing_mode_.load())` — the current sharing
>   mode as an int (Disabled=0, PerModel=1, Global=2).
> - `weight_cache_option_key` ("weight_cache_enabled"): set `option.value` to the
>   bool `weight_cache_enabled_.load()`.
> - `packed_cache_path_option_key` ("packed_cache_path"): build a zero-initialized
>   `array<char, kMaxOptionValueLength>`; under `path_mutex_` (lock_guard) copy
>   `min(packed_cache_path_.size(), kMaxOptionValueLength - 1)` bytes of
>   `packed_cache_path_` into the array (so at least one byte remains for a null
>   terminator; the zero-init guarantees termination), then set `option.value` to
>   that array.
> - Any other key: no branch matches; `option.value` is left unchanged.
>
> The `option.key` field must be set by the caller. Always returns `Error::Ok`
> (unknown keys are not treated as an error here).

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn]
> runtime::Result<std::shared_ptr<delegate::XNNWeightsCache>>

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn]
> Returns a shared `XNNWeightsCache` for the given `cache_file_path` by forwarding
> to the backend-owned manager: `weights_cache_manager_.get_or_create(cache_file_path)`,
> returning its `Result<shared_ptr<delegate::XNNWeightsCache>>` unchanged.
>
> Per the manager's contract: the same non-empty path yields the same shared
> instance; different non-empty paths yield independent instances; an empty path
> yields one shared heap-only instance across all empty-path callers (so XNNPACK's
> in-memory name dedup still works). Never null on success. The caller must hold the
> returned instance's `XNNWeightsCache::mutex()` around every cache-method call,
> including the XNNPACK callbacks invoked during `xnn_create_runtime`.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn]
> std::string XnnpackBackendOptions::get_packed_cache_path() const

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn]
> Const. Acquires `path_mutex_` (lock_guard) and returns a copy of
> `packed_cache_path_`. The lock serializes against
> `[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn]`
> and the `packed_cache_path` branch of `set_option`, so the returned string is a
> consistent snapshot that cannot tear against a concurrent write. The caller
> receives an independent copy; later writes do not affect it. Empty string if never
> set.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-sharing-mode-fn]
> WorkspaceSharingMode XnnpackBackendOptions::get_sharing_mode() const

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-sharing-mode-fn]
> Const accessor. Atomically loads and returns the options object's own
> `sharing_mode_` (`atomic<WorkspaceSharingMode>`). This is the globally-configured
> default mode stored on the `XnnpackBackendOptions`; it is distinct from the
> `XNNWorkspaceManager`'s copy. No side effects.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-sharing-mode-fn]
> runtime::Result<WorkspaceSharingMode>

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-sharing-mode-fn]
> Const. Computes the effective workspace sharing mode for a single delegate init,
> applying any per-delegate runtime spec override. Returns
> `Result<WorkspaceSharingMode>`.
>
> Steps:
> - Load the global default `global_mode = sharing_mode_.load()`.
> - Resolve the raw int value: `resolve_option<int>(context,
>   workspace_sharing_mode_option_key, (int)global_mode)`
>   (`[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]`)
>   — i.e. the "workspace_sharing_mode" runtime spec override if present, else the
>   global default as an int.
> - Validate: require `0 <= raw_mode < (int)WorkspaceSharingMode::Count`. If out of
>   range, log at Error ("XNNPACK workspace sharing mode must be between 0 and %d,
>   inclusive, but was %d." with `Count - 1` and `raw_mode`) and return
>   `runtime::Error::InvalidArgument`.
> - Otherwise return `static_cast<WorkspaceSharingMode>(raw_mode)`.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-weight-cache-fn]
> bool XnnpackBackendOptions::resolve_weight_cache( const ET_RUNTIME_NAMESPACE::BackendInitContext& context) const

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-weight-cache-fn]
> Const. Computes the effective weight-cache-enabled flag for a single delegate init,
> applying any per-delegate runtime spec override. Returns bool.
>
> Delegates to `resolve_option<bool>(context, weight_cache_option_key,
> weight_cache_enabled_.load())`
> (`[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]`):
> returns the "weight_cache_enabled" runtime spec override if present, else the
> atomically-loaded global default `weight_cache_enabled_`. No validation, no
> mutation.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn]
> Error XnnpackBackendOptions::save_weights_cache_locked()

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn]
> Persists all live weight caches to disk by forwarding to the manager:
> `weights_cache_manager_.save_all()`, returning its `Error` unchanged.
>
> Per the manager's contract: walks every live cache instance the manager has handed
> out and invokes `save_packed_index()` on each under that instance's own mutex,
> opportunistically erasing expired weak entries; returns the first error encountered
> but continues attempting every instance. Returns `Error::Ok` if all succeed.
> Invoked from the `save_weight_cache_on_disk` branch of `set_option`.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn]
> Error XnnpackBackendOptions::set_option(const BackendOption& option)

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn]
> Sets a single option, validating its type and domain. Takes a const
> `BackendOption&` (whose `value` is a `variant<bool, int, array<char, 256>>`).
> Returns `Error`.
>
> Dispatch by exact string compare (`strcmp == 0`) of `option.key`:
> - `workspace_sharing_mode_option_key` ("workspace_sharing_mode"):
>   - Extract an int from the variant via `get_if<int>`. If the alternative is not
>     int, log at Error ("XNNPACK workspace sharing mode must be an integer.") and
>     return `Error::InvalidArgument`.
>   - Range-check: require `0 <= *val < (int)WorkspaceSharingMode::Count`; if out of
>     range, log at Error ("...must be between 0 and %d, inclusive, but was %d." with
>     `Count - 1` and `*val`) and return `Error::InvalidArgument`.
>   - Otherwise log at Debug ("Setting XNNPACK workspace sharing mode to %d.") and
>     atomically store `static_cast<WorkspaceSharingMode>(*val)` into `sharing_mode_`.
> - `weight_cache_option_key` ("weight_cache_enabled"):
>   - Extract a bool via `get_if<bool>`. If not a bool, log at Error ("XNNPACK weight
>     cache enabled must be a bool.") and return `Error::InvalidArgument`.
>   - Otherwise log at Debug and atomically store `*val` into `weight_cache_enabled_`.
> - `packed_cache_path_option_key` ("packed_cache_path"):
>   - Extract the `array<char, kMaxOptionValueLength>` alternative via `get_if`. If
>     absent, log at Error ("XNNPACK packed cache path must be a string.") and return
>     `Error::InvalidArgument`.
>   - Otherwise, under `path_mutex_` (lock_guard), set `packed_cache_path_ =
>     std::string(val->data())` — constructing from the char array's data as a
>     C-string, so the string ends at the first NUL (or at the array end if
>     unterminated). Log at Debug. The lock guards this write against concurrent
>     `get_packed_cache_path`/`get_option` reads.
> - `save_weight_cache_on_disk_option_key` ("save_weight_cache_on_disk"):
>   - Extract a bool via `get_if<bool>`. If not a bool, log at Error ("XNNPACK
>     save_weight_cache_on_disk must be a bool.") and return `Error::InvalidArgument`.
>   - If the bool is true, return the result of `save_weights_cache_locked()`
>     (`[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn]`)
>     — a one-shot side-effect that flushes all live caches to disk. If false, do
>     nothing (fall through).
> - Any other/unrecognized key: no branch matches; nothing is changed.
>
> If no early return occurred, return `Error::Ok`.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn]
> void XnnpackBackendOptions::set_packed_cache_path(const std::string& path)

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn]
> Acquires `path_mutex_` (lock_guard) and assigns `packed_cache_path_ = path`
> (copying the given string). No return value. The lock serializes this write against
> `[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn]`
> and `get_option`/`set_option` path reads so the stored string never tears.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.weights-cache-manager-fn]
> const XNNWeightsCacheManager& XnnpackBackendOptions::weights_cache_manager()

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.weights-cache-manager-fn]
> Accessor for the backend-owned `XNNWeightsCacheManager`. Returns a reference to the
> member `weights_cache_manager_`. Two overloads exist: the const overload returns a
> `const XNNWeightsCacheManager&`; the non-const overload returns a mutable
> `XNNWeightsCacheManager&`. Both simply return the member by reference (no copy, no
> locking). Useful for tests and the `save_weight_cache_on_disk` side-effect path;
> production callers should prefer
> `[spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn]`.

> [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.workspace-manager-fn]
> XNNWorkspaceManager& XnnpackBackendOptions::workspace_manager()

> [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.workspace-manager-fn]
> Accessor for the backend-owned `XNNWorkspaceManager`. Returns a reference to the
> member `workspace_manager_`. Two overloads: the non-const overload returns a
> mutable `XNNWorkspaceManager&`; the const overload returns a
> `const XNNWorkspaceManager&`. Both just return the member by reference (no copy, no
> locking).

