# backends/xnnpack/runtime/XNNWorkspace.h

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace]
> class XNNWorkspace {
>   static inline std::atomic<uint64_t> next_id_{0};
>   std::mutex mutex_;
>   uint64_t id_;
>   bool lock_required_ = true;
>   std::atomic<bool> uses_weight_cache_{false};
>   WorkspacePtr workspace_;
> }

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]
> std::pair<std::unique_lock<std::mutex>, xnn_workspace_t> acquire()

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]
> Returns a pair `(lock, workspace_ptr)` where `lock` is an owning lock guard over
> the instance's `mutex_` and `workspace_ptr` is the raw `xnn_workspace_t` held by
> `workspace_` (the underlying pointer of the `WorkspacePtr` unique_ptr; may be null
> only if the object was constructed with a null workspace).
>
> Behavior:
> - If `lock_required_` is false (locking was disabled via
>   `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]`),
>   return an empty/unlocked lock object (owning no mutex) paired with the raw
>   workspace pointer. No synchronization is performed.
> - Otherwise, construct a lock that acquires `mutex_` (blocking until the mutex is
>   available), and return that held lock together with the raw workspace pointer.
>
> The returned lock keeps `mutex_` held for as long as the caller retains it; the
> caller is expected to hold the lock for the entire duration of any use of the
> returned workspace pointer, so that concurrent access to the same workspace is
> serialized. Releasing/dropping the lock releases `mutex_`. When locking is
> disabled the returned pointer is unsynchronized and the caller is responsible for
> ensuring it is not used concurrently.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]
> static runtime::Result<std::shared_ptr<XNNWorkspace>> create()

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]
> Static factory. Constructs a new `XNNWorkspace` wrapping a freshly created
> XNNPACK workspace, returned as a `shared_ptr` inside a `Result`.
>
> Steps:
> - Initialize a raw `xnn_workspace_t` local to null.
> - Call XNNPACK's `xnn_create_workspace(&workspace)` to allocate a new workspace.
> - If the returned XNNPACK status is not `xnn_status_success`: log at Error level
>   ("Failed to create XNN workspace, XNNPACK status: 0x%x" with the status value as
>   an unsigned int) and return `runtime::Error::Internal`. The partially created
>   local is not used.
> - On success, wrap the raw workspace in a `WorkspacePtr` — a `unique_ptr` whose
>   deleter is `xnn_release_workspace`, so releasing the wrapper frees the XNNPACK
>   workspace — and construct an `XNNWorkspace` in place from it (via
>   `make_shared`, because the class is non-movable), returning the resulting
>   `shared_ptr<XNNWorkspace>`.
>
> The new instance's construction assigns its unique id and default flags per
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]`.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]
> void disable_locking()

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]
> Sets the instance's `lock_required_` flag to false. After this call,
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]`
> stops taking `mutex_` and returns an unlocked lock, i.e. the workspace becomes
> unsynchronized.
>
> This is a plain (non-atomic) write to `lock_required_`. It is intended to be
> called before the workspace is shared across threads (e.g. right after creation
> for a non-shared/Disabled workspace), and is not itself guarded by any lock. No
> return value.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn]
> uint64_t id() const

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn]
> Returns the instance's `id_` (`uint64_t`), the unique identifier assigned at
> construction per
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]`.
> Const, side-effect free. The id lets callers distinguish two distinct workspace
> objects even when their raw workspace pointers happen to coincide due to memory
> reuse after one is freed.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn]
> void set_uses_weight_cache()

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn]
> Atomically stores `true` into the `uses_weight_cache_` atomic bool using
> release memory ordering (`memory_order_release`). This publishes the flag so a
> later acquire-load by
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn]`
> observes it. Idempotent, no return value. Marks that this workspace is being
> used together with a weight cache.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn]
> xnn_workspace_t unsafe_get_workspace()

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn]
> Returns the raw `xnn_workspace_t` held by `workspace_` (the unique_ptr's
> underlying pointer) WITHOUT acquiring `mutex_`. No synchronization is performed.
> Intended for cases where only the pointer identity/value is needed (e.g. logging)
> or where the caller has otherwise established exclusive access; using the returned
> pointer concurrently with another thread's use of the same workspace can cause
> crashes or data corruption. Prefer
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]`
> for actual workspace use.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn]
> bool uses_weight_cache() const

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn]
> Atomically loads and returns the `uses_weight_cache_` atomic bool using acquire
> memory ordering (`memory_order_acquire`). Const, no side effects. Defaults to
> false until
> `[spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn]`
> sets it true.

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]
> XNNWorkspace(WorkspacePtr workspace)

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]
> Constructor taking ownership of a `WorkspacePtr` (a `unique_ptr<xnn_workspace,
> &xnn_release_workspace>`).
>
> Initialization:
> - `id_` is set to the current value of the static atomic counter `next_id_`,
>   which is then post-incremented (`next_id_++`). `next_id_` starts at 0 and is
>   shared across all `XNNWorkspace` instances process-wide, so each instance gets a
>   distinct, monotonically increasing id (the first constructed instance gets id 0).
>   The increment is atomic, making concurrent construction safe.
> - `workspace_` is move-initialized from the passed `WorkspacePtr`, taking exclusive
>   ownership of the underlying XNNPACK workspace (freed via `xnn_release_workspace`
>   when this object is destroyed).
> - `mutex_` is default-constructed (unlocked).
> - `lock_required_` defaults to true.
> - `uses_weight_cache_` defaults to false.
>
> The class is neither copyable nor movable (deleted copy/move; the mutex makes it
> non-movable).

> [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.operator-fn]
> XNNWorkspace& operator=(const XNNWorkspace&) = delete

> [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.operator-fn]
> The copy-assignment operator is explicitly deleted; `XNNWorkspace` instances
> cannot be copy-assigned. (Copy construction, move construction, and move
> assignment are likewise deleted — the instance owns a mutex and a uniquely-owned
> workspace and is therefore neither copyable nor movable.) In a Rust port this
> corresponds to a type that is not `Clone` and is pinned/owned in place (e.g. held
> behind an `Arc`), matching the C++ usage where instances live only inside a
> `shared_ptr`.

