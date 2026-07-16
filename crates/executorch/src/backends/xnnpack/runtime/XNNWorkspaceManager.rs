//! Literal port of backends/xnnpack/runtime/XNNWorkspaceManager.cpp +
//! backends/xnnpack/runtime/XNNWorkspaceManager.h.
//!
//! PORT-NOTE: Depends on the XNNPACK C API (via `XNNWorkspace`), so the whole
//! module is gated behind the `xnnpack` feature.
#![cfg(feature = "xnnpack")]

use super::XNNWorkspace::XNNWorkspace;
use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, Weak};

// PORT-NOTE: `WorkspaceSharingMode` is defined by the XNNPACKBackend module
// ([spec:et:def:xnnpack-backend.executorch.backends.xnnpack.workspace-sharing-mode]),
// which is still a stub in this wave. Re-export the discriminants here as an
// `#[repr(i32)]` enum matching the C++ `enum class WorkspaceSharingMode`
// (Disabled=0, PerModel=1, Global=2, Count=3) so this module compiles; when the
// backend module lands, this should be replaced by a re-use of its definition.
// UNRESOLVED CROSS-MODULE REFERENCE: super::XNNPACKBackend::WorkspaceSharingMode.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkspaceSharingMode {
    Disabled = 0,
    PerModel = 1,
    Global = 2,
    Count = 3,
}

/// PORT-NOTE: `std::atomic<WorkspaceSharingMode>` — modeled as an `AtomicI32`
/// storing the enum's `i32` discriminant, with helpers converting to/from the
/// enum. The C++ default memory ordering (seq_cst) is preserved.
struct AtomicSharingMode(AtomicI32);

impl AtomicSharingMode {
    fn new(mode: WorkspaceSharingMode) -> Self {
        AtomicSharingMode(AtomicI32::new(mode as i32))
    }

    fn load(&self) -> WorkspaceSharingMode {
        match self.0.load(Ordering::SeqCst) {
            0 => WorkspaceSharingMode::Disabled,
            1 => WorkspaceSharingMode::PerModel,
            2 => WorkspaceSharingMode::Global,
            _ => WorkspaceSharingMode::Count,
        }
    }

    fn store(&self, mode: WorkspaceSharingMode) {
        self.0.store(mode as i32, Ordering::SeqCst);
    }
}

/// XNNWorkspaceManager manages XNNPACK workspaces based on the configured
/// workspace sharing mode.
// [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager]
//
// PORT-NOTE: The C++ `weak_ptr` members are ported as `std::sync::Weak`, and
// `shared_ptr` as `Arc`. `global_workspace_` and `model_workspaces_` are
// `mutable` in C++ (mutated from const methods under `workspace_meta_mutex_`);
// in Rust the whole guarded state lives inside a single `Mutex` so the const
// methods take `&self`.
pub struct XNNWorkspaceManager {
    sharing_mode_: AtomicSharingMode,
    // A mutex guarding global_workspace_ and model_workspaces_.
    meta_: Mutex<WorkspaceMeta>,
}

struct WorkspaceMeta {
    global_workspace_: Weak<XNNWorkspace>,
    model_workspaces_: HashMap<usize, Weak<XNNWorkspace>>,
}

impl XNNWorkspaceManager {
    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.xnn-workspace-manager-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.xnn-workspace-manager-fn]
    //
    // PORT-NOTE: `ENABLE_XNNPACK_SHARED_WORKSPACE` is a compile-time macro. It
    // is not defined in this build, so the constructor selects `Disabled`,
    // matching the default C++ configuration. A future feature flag can mirror
    // the macro.
    pub fn new() -> Self {
        // PORT-NOTE: `#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE` selects Global,
        // else Disabled. That macro is not defined in this build, so the
        // Disabled default is selected. When a corresponding cfg/feature is
        // introduced, restore the two-way selection.
        XNNWorkspaceManager {
            sharing_mode_: AtomicSharingMode::new(WorkspaceSharingMode::Disabled),
            meta_: Mutex::new(WorkspaceMeta {
                global_workspace_: Weak::new(),
                model_workspaces_: HashMap::new(),
            }),
        }
    }

    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn]
    pub fn set_sharing_mode(&self, mode: WorkspaceSharingMode) -> Error {
        // Validate that the mode is valid
        if (mode as i32) < 0 || (mode as i32) >= (WorkspaceSharingMode::Count as i32) {
            crate::et_log!(
                Error,
                "XNNPACK workspace sharing mode must be between 0 and {}, inclusive, but was {}.",
                (WorkspaceSharingMode::Count as i32) - 1,
                mode as i32
            );
            return Error::InvalidArgument;
        }

        self.sharing_mode_.store(mode);
        Error::Ok
    }

    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-sharing-mode-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-sharing-mode-fn]
    pub fn get_sharing_mode(&self) -> WorkspaceSharingMode {
        self.sharing_mode_.load()
    }

    pub fn get_or_create_workspace(&self, program_id: usize) -> Result<Arc<XNNWorkspace>> {
        self.get_or_create_workspace_with_mode(program_id, self.sharing_mode_.load())
    }

    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn]
    pub fn get_or_create_workspace_with_mode(
        &self,
        program_id: usize,
        mode: WorkspaceSharingMode,
    ) -> Result<Arc<XNNWorkspace>> {
        // Get or create the workspace according to the specified sharing mode.
        if mode == WorkspaceSharingMode::Disabled {
            crate::et_log!(Debug, "Instantiating workspace.");
            let create_result = XNNWorkspace::create();
            if !create_result.is_ok() {
                return Err(create_result.err().unwrap());
            }

            let mut workspace = create_result.unwrap();

            // PORT-NOTE: `#ifndef XNNPACK_WORKSPACE_ALWAYS_LOCK` — the macro is
            // not defined in this build, so locking is disabled for the freshly
            // created, unshared workspace. `disable_locking` needs `&mut`, and
            // `workspace` is still uniquely owned here so `Arc::get_mut`
            // succeeds. Restore a cfg guard here if the macro is introduced.
            Arc::get_mut(&mut workspace)
                .expect("freshly created workspace is uniquely owned")
                .disable_locking();
            return Ok(workspace);
        } else if mode == WorkspaceSharingMode::PerModel {
            return self.get_or_create_model_workspace(program_id);
        } else if mode == WorkspaceSharingMode::Global {
            return self.get_or_create_global_workspace();
        } else {
            crate::et_log!(Error, "Invalid workspace sharing mode: {}.", mode as i32);
            return Err(Error::Internal);
        }
    }

    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn]
    fn get_or_create_global_workspace(&self) -> Result<Arc<XNNWorkspace>> {
        let mut meta = self.meta_.lock().unwrap();

        // Check for an existing (live) global workspace.
        let mut workspace: Option<Arc<XNNWorkspace>> = None;
        if let Some(live_workspace) = meta.global_workspace_.upgrade() {
            workspace = Some(live_workspace);
        }

        // Allocate a new workspace if needed.
        if workspace.is_none() {
            let create_result = XNNWorkspace::create();
            if !create_result.is_ok() {
                return Err(create_result.err().unwrap());
            }
            let new_workspace = create_result.unwrap();
            crate::et_log!(
                Debug,
                "Created global workspace {:p}.",
                new_workspace.unsafe_get_workspace().0
            );
            meta.global_workspace_ = Arc::downgrade(&new_workspace);
            workspace = Some(new_workspace);
        }

        Ok(workspace.unwrap())
    }

    // [spec:et:def:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn]
    fn get_or_create_model_workspace(&self, program_id: usize) -> Result<Arc<XNNWorkspace>> {
        let mut meta = self.meta_.lock().unwrap();

        // Check for an existing (live) workspace for this program.
        let mut workspace: Option<Arc<XNNWorkspace>> = None;
        if let Some(entry) = meta.model_workspaces_.get(&program_id) {
            if let Some(live_workspace) = entry.upgrade() {
                workspace = Some(live_workspace);
            }
        }

        // Allocate a new workspace if needed.
        if workspace.is_none() {
            let create_result = XNNWorkspace::create();
            if !create_result.is_ok() {
                return Err(create_result.err().unwrap());
            }
            let new_workspace = create_result.unwrap();
            crate::et_log!(
                Debug,
                "Created workspace {:p} for program {}.",
                new_workspace.unsafe_get_workspace().0,
                program_id
            );
            meta.model_workspaces_
                .insert(program_id, Arc::downgrade(&new_workspace));
            workspace = Some(new_workspace);
        }

        Ok(workspace.unwrap())
    }
}

// Literal port of backends/xnnpack/test/runtime/test_workspace_manager.cpp.
//
// PORT-NOTE: LINK GAP. Every test here goes through `XNNWorkspace::create()`,
// which calls the XNNPACK C API (`xnn_create_workspace`). The `xnnpack` feature
// declares those `extern "C"` symbols but nothing links the XNNPACK C library
// yet (there is no build script for it). These tests therefore compile under
// `--features xnnpack` but cannot link/run until XNNPACK is wired into the
// build. They are left as plain `#[test]`s (not `#[ignore]`d) per the Wave-3
// xnnpack-gated convention; the default `cargo test` build does not enable the
// feature, so they are simply excluded there.
#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors `XNNWorkspaceManagerTest::SetUp()`: PAL init + a fresh manager.
    fn setup() -> XNNWorkspaceManager {
        crate::runtime::platform::runtime::runtime_init();
        XNNWorkspaceManager::new()
    }

    // setup() constructs a fresh manager via the ctor, whose default sharing
    // mode (Disabled) is asserted below.
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.xnn-workspace-manager-fn/test]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn/test]
    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-sharing-mode-fn/test]
    #[test]
    fn set_and_get_sharing_mode() {
        let wm = setup();
        assert_eq!(
            wm.set_sharing_mode(WorkspaceSharingMode::Disabled),
            Error::Ok
        );
        assert_eq!(wm.get_sharing_mode(), WorkspaceSharingMode::Disabled);

        assert_eq!(
            wm.set_sharing_mode(WorkspaceSharingMode::PerModel),
            Error::Ok
        );
        assert_eq!(wm.get_sharing_mode(), WorkspaceSharingMode::PerModel);

        assert_eq!(wm.set_sharing_mode(WorkspaceSharingMode::Global), Error::Ok);
        assert_eq!(wm.get_sharing_mode(), WorkspaceSharingMode::Global);
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.set-sharing-mode-fn/test]
    #[test]
    fn set_invalid_sharing_mode() {
        let wm = setup();
        // Start from a known state.
        assert_eq!(
            wm.set_sharing_mode(WorkspaceSharingMode::Disabled),
            Error::Ok
        );
        assert_eq!(wm.get_sharing_mode(), WorkspaceSharingMode::Disabled);

        // static_cast<WorkspaceSharingMode>(70): out of range → InvalidArgument.
        // PORT-NOTE: Rust cannot form an out-of-range enum value directly the way
        // the C++ `static_cast<WorkspaceSharingMode>(70)` does; transmute the raw
        // discriminant to mirror the C++ (which likewise produces an invalid enum
        // value passed into `set_sharing_mode`).
        let invalid_mode: WorkspaceSharingMode =
            unsafe { core::mem::transmute::<i32, WorkspaceSharingMode>(70) };
        assert_eq!(wm.set_sharing_mode(invalid_mode), Error::InvalidArgument);

        // Mode should not have changed.
        assert_eq!(wm.get_sharing_mode(), WorkspaceSharingMode::Disabled);
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn/test]
    #[test]
    fn disabled_mode() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::Disabled);

        let program_id: usize = 12345;
        let workspace1 = wm.get_or_create_workspace(program_id).unwrap();
        let workspace2 = wm.get_or_create_workspace(program_id).unwrap();
        let workspace3 = wm.get_or_create_workspace(program_id + 1).unwrap();

        assert!(!Arc::ptr_eq(&workspace1, &workspace2));
        assert!(!Arc::ptr_eq(&workspace1, &workspace3));
        assert!(!Arc::ptr_eq(&workspace2, &workspace3));
        assert_ne!(
            workspace1.unsafe_get_workspace().0,
            workspace2.unsafe_get_workspace().0
        );
        assert_ne!(
            workspace1.unsafe_get_workspace().0,
            workspace3.unsafe_get_workspace().0
        );
        assert_ne!(
            workspace2.unsafe_get_workspace().0,
            workspace3.unsafe_get_workspace().0
        );
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn/test]
    #[test]
    fn disabled_mode_acquire_does_not_lock() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::Disabled);

        let workspace = wm.get_or_create_workspace(12345).unwrap();
        let (lock, ptr) = workspace.acquire();
        assert!(!ptr.0.is_null());
        // PORT-NOTE: `XNNPACK_WORKSPACE_ALWAYS_LOCK` is not defined in this build
        // (see XNNWorkspaceManager Disabled path, which calls `disable_locking`),
        // so acquire returns no lock. `lock.owns_lock()` → `lock.is_some()`.
        assert!(lock.is_none());
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn/test]
    #[test]
    fn per_model_mode() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::PerModel);

        let program_id: usize = 12345;
        let workspace1 = wm.get_or_create_workspace(program_id).unwrap();
        let workspace2 = wm.get_or_create_workspace(program_id).unwrap();
        let workspace3 = wm.get_or_create_workspace(program_id + 1).unwrap();

        assert!(Arc::ptr_eq(&workspace1, &workspace2));
        assert_eq!(
            workspace1.unsafe_get_workspace().0,
            workspace2.unsafe_get_workspace().0
        );

        assert!(!Arc::ptr_eq(&workspace1, &workspace3));
        assert_ne!(
            workspace1.unsafe_get_workspace().0,
            workspace3.unsafe_get_workspace().0
        );
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.acquire-fn/test]
    #[test]
    fn per_model_acquire_still_locks() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::PerModel);

        let workspace = wm.get_or_create_workspace(12345).unwrap();
        let (lock, ptr) = workspace.acquire();
        assert!(!ptr.0.is_null());
        assert!(lock.is_some());
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.unsafe-get-workspace-fn/test]
    #[test]
    fn global_mode() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::Global);

        let program_id1: usize = 12345;
        let workspace1 = wm.get_or_create_workspace(program_id1).unwrap();

        let program_id2: usize = 67890;
        let workspace2 = wm.get_or_create_workspace(program_id2).unwrap();

        assert!(Arc::ptr_eq(&workspace1, &workspace2));
        assert_eq!(
            workspace1.unsafe_get_workspace().0,
            workspace2.unsafe_get_workspace().0
        );
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-model-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn/test]
    #[test]
    fn per_model_mode_cleanup() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::PerModel);

        let program_id: usize = 12345;
        let workspace1_id: u64;

        // Scope to control the lifetime of workspace1.
        {
            let workspace1 = wm.get_or_create_workspace(program_id).unwrap();
            workspace1_id = workspace1.id();
        }

        // The previous workspace was destroyed → a new one is created.
        let workspace2 = wm.get_or_create_workspace(program_id).unwrap();
        assert_ne!(workspace2.id(), workspace1_id);
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-global-workspace-fn/test]
    // [spec:et:sem:xnn-workspace.executorch.backends.xnnpack.xnn-workspace.id-fn/test]
    #[test]
    fn global_mode_cleanup() {
        let wm = setup();
        wm.set_sharing_mode(WorkspaceSharingMode::Global);

        let program_id: usize = 12345;
        let workspace1_id: u64;

        {
            let workspace1 = wm.get_or_create_workspace(program_id).unwrap();
            workspace1_id = workspace1.id();
        }

        let workspace2 = wm.get_or_create_workspace(program_id).unwrap();
        assert_ne!(workspace2.id(), workspace1_id);
    }

    // [spec:et:sem:xnn-workspace-manager.executorch.backends.xnnpack.xnn-workspace-manager.get-or-create-workspace-fn/test]
    #[test]
    fn switching_modes() {
        let wm = setup();

        // Start with Disabled mode.
        wm.set_sharing_mode(WorkspaceSharingMode::Disabled);

        let program_id: usize = 12345;
        let workspace1 = wm.get_or_create_workspace(program_id).unwrap();

        // Switch to PerModel mode.
        wm.set_sharing_mode(WorkspaceSharingMode::PerModel);
        let workspace2 = wm.get_or_create_workspace(program_id).unwrap();

        // Should be a different workspace.
        assert!(!Arc::ptr_eq(&workspace1, &workspace2));

        // Another with the same program ID in PerModel mode.
        let workspace3 = wm.get_or_create_workspace(program_id).unwrap();
        // Should be the same as workspace2.
        assert!(Arc::ptr_eq(&workspace2, &workspace3));

        // Switch to Global mode.
        wm.set_sharing_mode(WorkspaceSharingMode::Global);
        let workspace4 = wm.get_or_create_workspace(program_id).unwrap();
        // Different workspace since we switched modes.
        assert!(!Arc::ptr_eq(&workspace3, &workspace4));

        // Different program ID in Global mode.
        let different_program_id: usize = 67890;
        let workspace5 = wm.get_or_create_workspace(different_program_id).unwrap();
        // Same as workspace4.
        assert!(Arc::ptr_eq(&workspace4, &workspace5));
    }
}
