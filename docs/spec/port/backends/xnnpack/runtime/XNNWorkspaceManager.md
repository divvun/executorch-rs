# backends/xnnpack/runtime/XNNWorkspaceManager.cpp, backends/xnnpack/runtime/XNNWorkspaceManager.h

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager]
> class XNNWorkspaceManager {
>   std::atomic<WorkspaceSharingMode> sharing_mode_;
>   mutable std::mutex workspace_meta_mutex_;
>   mutable std::weak_ptr<XNNWorkspace> global_workspace_;
>   mutable std::unordered_map<uintptr_t, std::weak_ptr<XNNWorkspace>> model_workspaces_;
> }

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn]
> Result<std::shared_ptr<XNNWorkspace>>

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn]
> Private helper (const): returns the process-wide shared global workspace, lazily
> creating it if none is currently live.
>
> Steps:
> - Acquire `workspace_meta_mutex_` for the whole body (scoped lock), serializing
>   concurrent creation.
> - Initialize a local `shared_ptr<XNNWorkspace> workspace` to empty.
> - Attempt to lock the `weak_ptr` `global_workspace_`. If it yields a live
>   `shared_ptr` (a previously created global workspace still referenced somewhere),
>   set `workspace` to it.
> - If `workspace` is still empty (no live global workspace):
>   - Call `XNNWorkspace::create()`
>     (`[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]`).
>     If it fails, return that error immediately (propagating the `Result` error;
>     `global_workspace_` is left unchanged).
>   - On success, set `workspace` to the created `shared_ptr`, log at Debug
>     ("Created global workspace %p." with the raw workspace pointer from
>     `unsafe_get_workspace()`), and store `workspace` into the `weak_ptr`
>     `global_workspace_` (so future calls can reuse it while it stays alive).
> - Return `workspace` (never null on success). Because `global_workspace_` is a
>   weak pointer, the global workspace is destroyed once the last owning
>   `shared_ptr` is released, and a subsequent call recreates it.

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn]
> Result<std::shared_ptr<XNNWorkspace>>

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn]
> Private helper (const): returns the workspace shared by all delegate instances of
> the program identified by `program_id` (a `uintptr_t`), lazily creating it if none
> is currently live for that program.
>
> Steps:
> - Acquire `workspace_meta_mutex_` for the whole body (scoped lock).
> - Look up `program_id` in the `model_workspaces_` map (`unordered_map<uintptr_t,
>   weak_ptr<XNNWorkspace>>`). Initialize a local `shared_ptr workspace` to empty.
>   If an entry exists, attempt to lock its `weak_ptr`; if that yields a live
>   `shared_ptr`, set `workspace` to it.
> - If `workspace` is still empty (no entry, or the entry's weak_ptr expired):
>   - Call `XNNWorkspace::create()`
>     (`[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]`).
>     If it fails, return that error immediately (the map is left unchanged).
>   - On success, set `workspace` to the created `shared_ptr`, log at Debug
>     ("Created workspace %p for program %<PRIuPTR>." with the raw workspace pointer
>     and `program_id`), and insert/overwrite `model_workspaces_[program_id]` with
>     `workspace` (stored as a weak_ptr).
> - Return `workspace` (never null on success). Entries are weak pointers, so a
>   program's workspace is freed when its last owning `shared_ptr` is released; the
>   stale map entry (an expired weak_ptr) may remain until the next lookup replaces
>   it. Different `program_id`s get independent workspaces.

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn]
> Result<std::shared_ptr<XNNWorkspace>>

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn]
> Public (const). Returns a workspace for `program_id` according to a workspace
> sharing mode, creating one as needed.
>
> There are two overloads:
> - The single-argument overload `get_or_create_workspace(program_id)` reads the
>   current mode via `sharing_mode_.load()` and delegates to the two-argument form
>   with that mode.
> - The two-argument overload `get_or_create_workspace(program_id, mode)`
>   dispatches on `mode` (a `WorkspaceSharingMode`, see
>   `[spec:et:def:xnnpack-backend.executorch.backends.xnnpack.workspace-sharing-mode]`):
>   - `Disabled`: log at Debug ("Instantiating workspace."), then call
>     `XNNWorkspace::create()`
>     (`[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]`).
>     If creation fails, return its error. On success, unless the compile-time macro
>     `XNNPACK_WORKSPACE_ALWAYS_LOCK` is defined, call `disable_locking()`
>     (`[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]`)
>     on the new workspace — since a Disabled-mode workspace is not shared, its mutex
>     is unnecessary. Return the new `shared_ptr`. This path creates a fresh,
>     unshared workspace on every call (it is not registered in any map).
>   - `PerModel`: delegate to `get_or_create_model_workspace(program_id)`
>     (`[spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn]`).
>   - `Global`: delegate to `get_or_create_global_workspace()`
>     (`[spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn]`).
>   - Any other value (e.g. `Count` or out of range): log at Error ("Invalid
>     workspace sharing mode: %d.") and return `Error::Internal`.

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-sharing-mode-fn]
> WorkspaceSharingMode XNNWorkspaceManager::get_sharing_mode() const

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-sharing-mode-fn]
> Const accessor. Atomically loads and returns the current `sharing_mode_`
> (`atomic<WorkspaceSharingMode>`) with default (sequentially consistent) memory
> ordering. No side effects.

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn]
> runtime::Error XNNWorkspaceManager::set_sharing_mode( WorkspaceSharingMode mode)

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn]
> Sets the active workspace sharing mode. Returns `runtime::Error`.
>
> Steps:
> - Validate `mode`: cast to int and require `0 <= (int)mode < (int)Count` (the
>   sentinel `WorkspaceSharingMode::Count`, which equals 3 given Disabled=0,
>   PerModel=1, Global=2). If out of range, log at Error ("XNNPACK workspace sharing
>   mode must be between 0 and %d, inclusive, but was %d." with `Count - 1` and the
>   given mode value) and return `Error::InvalidArgument` without changing state.
> - Otherwise atomically store `mode` into `sharing_mode_` and return `Error::Ok`.
>
> Note: the change only affects workspaces created after this call; already-created
> shared workspaces are unaffected.

> [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.xnn-workspace-manager-fn]
> XNNWorkspaceManager::XNNWorkspaceManager()

> [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.xnn-workspace-manager-fn]
> Constructor. Initializes `sharing_mode_` based on a compile-time flag:
> - If the macro `ENABLE_XNNPACK_SHARED_WORKSPACE` is defined, `sharing_mode_` is set
>   to `WorkspaceSharingMode::Global`.
> - Otherwise it is set to `WorkspaceSharingMode::Disabled`.
>
> The other members are default-constructed: `workspace_meta_mutex_` unlocked,
> `global_workspace_` an empty `weak_ptr`, and `model_workspaces_` an empty map.
> The destructor is defaulted.

