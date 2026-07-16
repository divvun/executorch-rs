//! Literal port of backends/xnnpack/runtime/XNNWorkspace.h.
//!
//! PORT-NOTE: The C++ `XNNWorkspace` wraps a raw `xnn_workspace_t` and its
//! synchronization. It depends directly on the XNNPACK C API
//! (`xnn_create_workspace` / `xnn_release_workspace`), so the whole
//! implementation is gated behind the `xnnpack` feature. XNNPACK is never
//! actually absent in shipping builds; a non-feature fallback is not provided
//! because every caller of this module is itself feature-gated.
#![cfg(feature = "xnnpack")]

use super::sys::{
    xnn_create_workspace, xnn_release_workspace, xnn_status_success, xnn_workspace_t,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// C++ `WorkspacePtr = std::unique_ptr<xnn_workspace, &xnn_release_workspace>`.
///
/// PORT-NOTE: Modeled as a newtype owning the raw handle whose `Drop` calls
/// `xnn_release_workspace`, exactly mirroring the unique_ptr deleter. A null
/// handle is not released (matches unique_ptr with a null pointer).
pub struct WorkspacePtr(xnn_workspace_t);

impl WorkspacePtr {
    /// C++ `WorkspacePtr(workspace, &xnn_release_workspace)`.
    pub fn new(workspace: xnn_workspace_t) -> Self {
        WorkspacePtr(workspace)
    }

    /// C++ `unique_ptr::get()`.
    pub fn get(&self) -> xnn_workspace_t {
        self.0
    }
}

impl Drop for WorkspacePtr {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                xnn_release_workspace(self.0);
            }
        }
    }
}

// PORT-NOTE: The raw XNNPACK workspace handle is used across threads (the class
// is shared via `Arc` and its access is serialized by `mutex_`), matching the
// C++ `std::shared_ptr<XNNWorkspace>` usage. Mark the pointer wrapper Send/Sync
// to reflect that the C++ relies on the same sharing.
unsafe impl Send for WorkspacePtr {}
unsafe impl Sync for WorkspacePtr {}

/// A lightweight wrapper around an underlying xnn_workspace_t instance, bundled
/// with appropriate synchronization.
// [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace]
//
// PORT-NOTE: `next_id_` is a process-wide static atomic. `mutex_` is a
// `std::sync::Mutex<()>` used purely as a lock (the workspace it guards is
// accessed through `workspace_`); this mirrors the C++ `std::mutex mutex_`
// which guards nothing structurally but serializes callers of `acquire()`. The
// deleted copy/move-assign is the Rust default (non-`Clone`, owned in place
// behind an `Arc`), so its markers collapse onto this struct.
// [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.operator-fn]
// [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.operator-fn]
pub struct XNNWorkspace {
    mutex_: Mutex<()>,
    id_: u64,
    lock_required_: bool,
    uses_weight_cache_: AtomicBool,
    workspace_: WorkspacePtr,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

impl XNNWorkspace {
    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn]
    pub fn new(workspace: WorkspacePtr) -> Self {
        XNNWorkspace {
            mutex_: Mutex::new(()),
            id_: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            lock_required_: true,
            uses_weight_cache_: AtomicBool::new(false),
            workspace_: workspace,
        }
    }

    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn]
    //
    // PORT-NOTE: Returns `(Option<MutexGuard<'_, ()>>, xnn_workspace_t)`. When
    // `lock_required_` is false the guard is `None` (the C++ empty
    // `std::unique_lock` owning no mutex); otherwise it is `Some(guard)` holding
    // `mutex_` for as long as the caller retains it. The raw workspace pointer
    // is returned alongside, matching the C++ `std::pair`.
    pub fn acquire(&self) -> (Option<MutexGuard<'_, ()>>, xnn_workspace_t) {
        if !self.lock_required_ {
            return (None, self.workspace_.get());
        }
        let lock = self.mutex_.lock().unwrap();
        (Some(lock), self.workspace_.get())
    }

    // Return the workspace pointer withot acquiring the lock. This should be
    // used carefully, as it can lead to crashes or data corruption if the
    // workspace is used concurrently.
    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn]
    pub fn unsafe_get_workspace(&self) -> xnn_workspace_t {
        self.workspace_.get()
    }

    // Returns a unique ID for this workspace instance.
    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn]
    pub fn id(&self) -> u64 {
        self.id_
    }

    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn]
    //
    // PORT-NOTE: The C++ writes the plain (non-atomic) `lock_required_` member
    // through `this`; it is called before the workspace is shared across
    // threads. That requires `&mut self` in Rust. Callers hold the workspace
    // via `Arc` while still uniquely owned at creation time (see
    // `XNNWorkspaceManager::get_or_create_workspace` Disabled path), so a
    // `&mut` reference is available via `Arc::get_mut`.
    pub fn disable_locking(&mut self) {
        self.lock_required_ = false;
    }

    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn]
    pub fn set_uses_weight_cache(&self) {
        self.uses_weight_cache_.store(true, Ordering::Release);
    }

    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn]
    pub fn uses_weight_cache(&self) -> bool {
        self.uses_weight_cache_.load(Ordering::Acquire)
    }

    // [spec:et:def:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn]
    pub fn create() -> Result<Arc<XNNWorkspace>> {
        // Because this class can't be moved, we need to construct it in-place.
        let mut workspace = xnn_workspace_t(core::ptr::null_mut());
        let status = unsafe { xnn_create_workspace(&mut workspace) };
        if status != xnn_status_success {
            crate::et_log!(
                Error,
                "Failed to create XNN workspace, XNNPACK status: 0x{:x}",
                status.0
            );
            return Err(Error::Internal);
        }

        Ok(Arc::new(XNNWorkspace::new(WorkspacePtr::new(workspace))))
    }
}

// PORT-NOTE: `create()` (and hence the whole `XNNWorkspaceManager` surface that
// drives it) needs a live XNNPACK workspace from `xnn_create_workspace`, which
// in turn requires `xnn_initialize` — those paths are covered by the
// XNNWorkspaceManager suite. The state-only members below, however, only touch
// Rust fields: a null `WorkspacePtr` constructs an `XNNWorkspace` without any
// XNNPACK C call (its `Drop` skips `xnn_release_workspace` for a null handle),
// so the plain accessors/mutators are unit-testable directly.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::xnn_workspace_t;

    fn null_workspace() -> XNNWorkspace {
        XNNWorkspace::new(WorkspacePtr::new(xnn_workspace_t(core::ptr::null_mut())))
    }

    // A freshly-constructed workspace has locking required and does not use the
    // weight cache; each instance gets a distinct monotonically increasing id.
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn/test]
    #[test]
    fn fresh_workspace_defaults() {
        let ws = null_workspace();
        assert!(!ws.uses_weight_cache());
        assert!(ws.lock_required_);

        // acquire() takes the lock while lock_required_ is true.
        let (lock, ptr) = ws.acquire();
        assert!(lock.is_some());
        assert!(ptr.0.is_null());
        drop(lock);

        let ws2 = null_workspace();
        assert_ne!(ws.id(), ws2.id());
    }

    // set_uses_weight_cache flips the flag observed by uses_weight_cache.
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.set-uses-weight-cache-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.uses-weight-cache-fn/test]
    #[test]
    fn set_uses_weight_cache_flips_flag() {
        let ws = null_workspace();
        assert!(!ws.uses_weight_cache());
        ws.set_uses_weight_cache();
        assert!(ws.uses_weight_cache());
    }

    // disable_locking makes acquire() return no lock (empty unique_lock).
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.disable-locking-fn/test]
    #[test]
    fn disable_locking_makes_acquire_lockless() {
        let mut ws = null_workspace();
        ws.disable_locking();
        let (lock, _ptr) = ws.acquire();
        assert!(lock.is_none());
    }

    // create() drives the real XNNPACK C API (`xnn_create_workspace`), which
    // requires `xnn_initialize` first. It returns an `Arc<XNNWorkspace>` wrapping
    // a live, non-null `xnn_workspace_t` with the same defaults as the ctor:
    // locking required, no weight cache, and a fresh monotonically increasing id.
    // The `WorkspacePtr` `Drop` releases the handle when the Arc is dropped.
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.create-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.xnn-workspace-fn/test]
    #[test]
    fn create_yields_live_workspace() {
        // `xnn_create_workspace` requires XNNPACK be initialized first, mirroring
        // the backend's own `xnn_initialize(nullptr)` in its ctor.
        let init_status =
            unsafe { crate::backends::xnnpack::runtime::sys::xnn_initialize(core::ptr::null()) };
        assert_eq!(init_status, xnn_status_success);

        let result = XNNWorkspace::create();
        assert!(result.is_ok());
        let ws = result.unwrap();

        // A real workspace has a non-null underlying handle.
        assert!(!ws.unsafe_get_workspace().0.is_null());
        // Defaults match the ctor: locking on, weight cache off.
        assert!(ws.lock_required_);
        assert!(!ws.uses_weight_cache());

        // acquire() takes the lock (lock_required_ is true) and hands back the
        // same non-null pointer.
        let (lock, ptr) = ws.acquire();
        assert!(lock.is_some());
        assert!(!ptr.0.is_null());
        assert_eq!(ptr.0, ws.unsafe_get_workspace().0);
        drop(lock);

        // Each created workspace gets a distinct id.
        let ws2 = XNNWorkspace::create().unwrap();
        assert_ne!(ws.id(), ws2.id());
    }

    // The deleted copy-assign (`operator=(const XNNWorkspace&) = delete`)
    // collapses onto the non-`Clone` struct in Rust: a workspace is never
    // duplicated, only shared behind an `Arc` (C++ `shared_ptr<XNNWorkspace>`).
    // The probe resolves the inherent `is_clone` (true) only when the type
    // implements `Clone`; otherwise the blanket trait method (false) is picked,
    // so this fails to hold if `Clone` is ever derived. The aliasing check then
    // shows handles share one instance rather than copying its state.
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.operator-fn/test]
    #[test]
    fn workspace_is_not_copyable() {
        struct Probe<T>(core::marker::PhantomData<T>);
        trait NotClone {
            fn is_clone(&self) -> bool {
                false
            }
        }
        impl<T> NotClone for Probe<T> {}
        impl<T: Clone> Probe<T> {
            #[allow(dead_code)]
            fn is_clone(&self) -> bool {
                true
            }
        }
        assert!(!Probe::<XNNWorkspace>(core::marker::PhantomData).is_clone());

        // Two Arc handles alias one instance: state set through one is observed
        // through the other, and both report the same id.
        let a = Arc::new(null_workspace());
        let b = Arc::clone(&a);
        a.set_uses_weight_cache();
        assert!(b.uses_weight_cache());
        assert_eq!(a.id(), b.id());
    }
}
