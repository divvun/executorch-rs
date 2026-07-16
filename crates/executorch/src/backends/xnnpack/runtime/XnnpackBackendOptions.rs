//! Literal port of backends/xnnpack/runtime/XnnpackBackendOptions.cpp +
//! backends/xnnpack/runtime/XnnpackBackendOptions.h.
//!
//! PORT-NOTE: Depends on the XNNPACK C API (via the workspace / weights-cache
//! managers), so the whole module is gated behind the `xnnpack` feature.
#![cfg(feature = "xnnpack")]
#![allow(non_snake_case)]

use super::XNNPACKBackend::{
    WorkspaceSharingMode, packed_cache_path_option_key, weight_cache_option_key,
    workspace_sharing_mode_option_key,
};
use super::XNNWeightsCache::XNNWeightsCache;
use super::XNNWeightsCacheManager::XNNWeightsCacheManager;
use super::XNNWorkspaceManager::XNNWorkspaceManager;

use crate::runtime::backend::backend_init_context::BackendInitContext;
use crate::runtime::backend::options::{BackendOption, K_MAX_OPTION_VALUE_LENGTH, OptionValue};
use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;
use crate::runtime::core::result::ResultExt;

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

// PORT-NOTE: `std::atomic<WorkspaceSharingMode>` modeled as an `AtomicI32`
// storing the enum's discriminant, mirroring the same wrapper used in
// XNNWorkspaceManager.
struct AtomicSharingMode(AtomicI32);

impl AtomicSharingMode {
    fn new(mode: WorkspaceSharingMode) -> Self {
        AtomicSharingMode(AtomicI32::new(mode as i32))
    }
    fn load(&self) -> WorkspaceSharingMode {
        WorkspaceSharingMode::from_i32(self.0.load(Ordering::SeqCst))
    }
    fn store(&self, mode: WorkspaceSharingMode) {
        self.0.store(mode as i32, Ordering::SeqCst);
    }
}

// PORT-NOTE: File-local template helper `resolve_option<T>` (anonymous
// namespace). The C++ instantiates it for `T ∈ {bool, int}`; those two
// monomorphizations become explicit `_bool` / `_int` helpers, each carrying the
// same lookup-with-fallback control flow. `get_runtime_spec<T>` becomes the
// per-type `get_runtime_spec_{bool,int}` methods on `BackendInitContext`.
// [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
// [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
fn resolve_option_bool(
    context: &BackendInitContext,
    key: *const core::ffi::c_char,
    global_default: bool,
) -> bool {
    let spec = context.get_runtime_spec_bool(key);
    if spec.is_ok() {
        return *spec.get();
    }
    global_default
}

// [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
// [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn]
fn resolve_option_int(
    context: &BackendInitContext,
    key: *const core::ffi::c_char,
    global_default: i32,
) -> i32 {
    let spec = context.get_runtime_spec_int(key);
    if spec.is_ok() {
        return *spec.get();
    }
    global_default
}

// [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options]
pub struct XnnpackBackendOptions {
    workspace_manager_: XNNWorkspaceManager,
    weights_cache_manager_: XNNWeightsCacheManager,

    // PORT-NOTE: `#ifdef ENABLE_XNNPACK_SHARED_WORKSPACE` selects Global else
    // Disabled; that macro is not defined in this build so the Disabled default
    // is selected (see `new`). Similarly `#ifdef ENABLE_XNNPACK_WEIGHTS_CACHE`
    // selects true, else false.
    sharing_mode_: AtomicSharingMode,
    weight_cache_enabled_: AtomicBool,

    // The most-recently-set packed cache path. `path_mutex_` serializes the
    // get/set pair so a concurrent set_option from another caller doesn't tear
    // the string mid-read. (`mutable std::mutex path_mutex_` in C++.)
    path_mutex_: Mutex<()>,
    packed_cache_path_: Mutex<String>,
}

impl XnnpackBackendOptions {
    pub fn new() -> Self {
        XnnpackBackendOptions {
            workspace_manager_: XNNWorkspaceManager::new(),
            weights_cache_manager_: XNNWeightsCacheManager::new(),
            sharing_mode_: AtomicSharingMode::new(WorkspaceSharingMode::Disabled),
            weight_cache_enabled_: AtomicBool::new(false),
            path_mutex_: Mutex::new(()),
            packed_cache_path_: Mutex::new(String::new()),
        }
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn]
    #[must_use]
    pub fn get_option(&self, option: &mut BackendOption) -> Error {
        if strcmp_key(&option.key, workspace_sharing_mode_option_key) {
            option.value = OptionValue::Int(self.sharing_mode_.load() as i32);
        } else if strcmp_key(&option.key, weight_cache_option_key) {
            option.value = OptionValue::Bool(self.weight_cache_enabled_.load(Ordering::SeqCst));
        } else if strcmp_key(&option.key, packed_cache_path_option_key) {
            let mut arr: [core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH] =
                [0; K_MAX_OPTION_VALUE_LENGTH];
            // path_mutex_ (lock_guard). The stored string itself lives behind a
            // separate mutex here; take path_mutex_ to serialize the same
            // way the C++ does.
            let _lock = self.path_mutex_.lock().unwrap();
            let path = self.packed_cache_path_.lock().unwrap();
            let len = core::cmp::min(path.len(), K_MAX_OPTION_VALUE_LENGTH - 1);
            let bytes = path.as_bytes();
            for i in 0..len {
                arr[i] = bytes[i] as core::ffi::c_char;
            }
            option.value = OptionValue::CharArray(arr);
        }
        Error::Ok
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn]
    #[must_use]
    pub fn set_option(&self, option: &BackendOption) -> Error {
        if strcmp_key(&option.key, workspace_sharing_mode_option_key) {
            let val = match &option.value {
                OptionValue::Int(v) => v,
                _ => {
                    crate::et_log!(Error, "XNNPACK workspace sharing mode must be an integer.");
                    return Error::InvalidArgument;
                }
            };
            if *val < 0 || *val >= WorkspaceSharingMode::Count as i32 {
                crate::et_log!(
                    Error,
                    "XNNPACK workspace sharing mode must be between 0 and {}, inclusive, but was {}.",
                    WorkspaceSharingMode::Count as i32 - 1,
                    *val
                );
                return Error::InvalidArgument;
            }
            crate::et_log!(Debug, "Setting XNNPACK workspace sharing mode to {}.", *val);
            self.sharing_mode_
                .store(WorkspaceSharingMode::from_i32(*val));
        } else if strcmp_key(&option.key, weight_cache_option_key) {
            let val = match &option.value {
                OptionValue::Bool(v) => v,
                _ => {
                    crate::et_log!(Error, "XNNPACK weight cache enabled must be a bool.");
                    return Error::InvalidArgument;
                }
            };
            crate::et_log!(
                Debug,
                "Setting XNNPACK weight cache enabled to {}.",
                *val as i32
            );
            self.weight_cache_enabled_.store(*val, Ordering::SeqCst);
        } else if strcmp_key(&option.key, packed_cache_path_option_key) {
            let val = match &option.value {
                OptionValue::CharArray(arr) => arr,
                _ => {
                    crate::et_log!(Error, "XNNPACK packed cache path must be a string.");
                    return Error::InvalidArgument;
                }
            };
            // path_mutex_ also guards get_packed_cache_path so the read in
            // XNNPACKBackend::init never tears against a concurrent write here.
            let _lock = self.path_mutex_.lock().unwrap();
            let s = cstr_from_char_array(val);
            *self.packed_cache_path_.lock().unwrap() = s.clone();
            crate::et_log!(Debug, "Setting XNNPACK packed cache path to {}.", s);
        } else if strcmp_key(&option.key, save_weight_cache_on_disk_option_key()) {
            let val = match &option.value {
                OptionValue::Bool(v) => v,
                _ => {
                    crate::et_log!(Error, "XNNPACK save_weight_cache_on_disk must be a bool.");
                    return Error::InvalidArgument;
                }
            };
            if *val {
                return self.save_weights_cache_locked();
            }
        }
        Error::Ok
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.weights-cache-manager-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.weights-cache-manager-fn]
    pub fn weights_cache_manager(&self) -> &XNNWeightsCacheManager {
        &self.weights_cache_manager_
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn]
    #[must_use]
    pub fn get_or_create_weights_cache(
        &self,
        cache_file_path: &str,
    ) -> Result<Arc<XNNWeightsCache>> {
        self.weights_cache_manager_.get_or_create(cache_file_path)
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn]
    #[must_use]
    pub fn save_weights_cache_locked(&self) -> Error {
        self.weights_cache_manager_.save_all()
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-weight-cache-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-weight-cache-fn]
    #[must_use]
    pub fn resolve_weight_cache(&self, context: &BackendInitContext) -> bool {
        resolve_option_bool(
            context,
            weight_cache_option_key.as_ptr() as *const core::ffi::c_char,
            self.weight_cache_enabled_.load(Ordering::SeqCst),
        )
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-sharing-mode-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-sharing-mode-fn]
    #[must_use]
    pub fn resolve_sharing_mode(
        &self,
        context: &BackendInitContext,
    ) -> Result<WorkspaceSharingMode> {
        let global_mode = self.sharing_mode_.load();
        let raw_mode = resolve_option_int(
            context,
            workspace_sharing_mode_option_key.as_ptr() as *const core::ffi::c_char,
            global_mode as i32,
        );
        if raw_mode < 0 || raw_mode >= WorkspaceSharingMode::Count as i32 {
            crate::et_log!(
                Error,
                "XNNPACK workspace sharing mode must be between 0 and {}, inclusive, but was {}.",
                WorkspaceSharingMode::Count as i32 - 1,
                raw_mode
            );
            return Err(Error::InvalidArgument);
        }
        Ok(WorkspaceSharingMode::from_i32(raw_mode))
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-sharing-mode-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-sharing-mode-fn]
    pub fn get_sharing_mode(&self) -> WorkspaceSharingMode {
        self.sharing_mode_.load()
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.workspace-manager-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.workspace-manager-fn]
    pub fn workspace_manager(&self) -> &XNNWorkspaceManager {
        &self.workspace_manager_
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn]
    pub fn get_packed_cache_path(&self) -> String {
        let _lock = self.path_mutex_.lock().unwrap();
        self.packed_cache_path_.lock().unwrap().clone()
    }

    // [spec:et:def:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn]
    pub fn set_packed_cache_path(&self, path: &str) {
        let _lock = self.path_mutex_.lock().unwrap();
        *self.packed_cache_path_.lock().unwrap() = path.to_string();
    }
}

// PORT-NOTE: The `save_weight_cache_on_disk_option_key` lives in
// XNNPACKBackend.h alongside the other keys; re-expose it here through a helper
// to avoid an import cycle at the const level.
fn save_weight_cache_on_disk_option_key() -> &'static [u8] {
    super::XNNPACKBackend::save_weight_cache_on_disk_option_key
}

// PORT-NOTE: `strcmp(option.key, key) == 0` over the fixed `key` char array. The
// key buffer is a NUL-terminated C string; compare it against the option-key
// byte-slice constant (which includes its own trailing NUL).
fn strcmp_key(key: &[core::ffi::c_char], expected: &[u8]) -> bool {
    // Walk the C string in `key` and compare to `expected` (up to and including
    // the shared NUL terminator).
    let mut i = 0usize;
    loop {
        let k = if i < key.len() { key[i] as u8 } else { 0 };
        let e = if i < expected.len() { expected[i] } else { 0 };
        if k != e {
            return false;
        }
        if k == 0 {
            return true;
        }
        i += 1;
    }
}

// PORT-NOTE: `std::string(val->data())` — build a `String` from the char array's
// data as a C string, ending at the first NUL (or the array end if
// unterminated).
fn cstr_from_char_array(arr: &[core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH]) -> String {
    let mut end = 0usize;
    while end < arr.len() && arr[end] != 0 {
        end += 1;
    }
    let bytes: Vec<u8> = arr[..end].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// PORT-NOTE: No C++ unit-test file exists for XnnpackBackendOptions; the
// get/set/resolve option surface is only exercised indirectly by the delegate
// integration tests (which need a live XNNPACK runtime). These focused unit
// tests pin the pure option-plumbing semantics directly against the C++ source.
// `XnnpackBackendOptions::new()` and every method exercised below manipulate only
// Rust-side state (atomics, mutexes, and the two sub-managers, whose
// constructors likewise touch no XNNPACK C symbol), so they run without a live
// runtime. The module is `#![cfg(feature = "xnnpack")]`, so these run under
// `--features xnnpack`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::backend_init_context::BackendInitContext;
    use crate::runtime::backend::options::{BackendOption, K_MAX_OPTION_KEY_LENGTH, OptionValue};
    use crate::runtime::core::event_tracer::EventTracer;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::named_data_map::NamedDataMap;
    use crate::runtime::core::span::Span;

    // Build a key array from a NUL-terminated byte literal (mirrors the option
    // key char arrays the get/set methods compare against).
    fn key_array(bytes: &[u8]) -> [core::ffi::c_char; K_MAX_OPTION_KEY_LENGTH] {
        let mut out = [0 as core::ffi::c_char; K_MAX_OPTION_KEY_LENGTH];
        for (i, b) in bytes.iter().enumerate() {
            out[i] = *b as core::ffi::c_char;
        }
        out
    }

    fn char_array(bytes: &[u8]) -> [core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH] {
        let mut out = [0 as core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH];
        for (i, b) in bytes.iter().enumerate() {
            out[i] = *b as core::ffi::c_char;
        }
        out
    }

    fn opt(bytes: &[u8], value: OptionValue) -> BackendOption {
        BackendOption {
            key: key_array(bytes),
            value,
        }
    }

    // set_option/get_option log via the PAL, so it must be initialized first
    // (mirrors the sibling xnnpack test modules' setup()).
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }
    fn null_event_tracer() -> *mut dyn EventTracer {
        core::ptr::null_mut::<UninitTracer>() as *mut dyn EventTracer
    }
    fn null_named_data_map() -> *const dyn NamedDataMap {
        core::ptr::null::<UninitNamedDataMap>() as *const dyn NamedDataMap
    }

    // Never-instantiated map used only to spell a null `*const dyn NamedDataMap`.
    struct UninitNamedDataMap;
    impl NamedDataMap for UninitNamedDataMap {
        fn get_tensor_layout(
            &self,
            _key: &str,
        ) -> Result<crate::runtime::core::tensor_layout::TensorLayout> {
            unreachable!()
        }
        fn get_data(
            &self,
            _key: &str,
        ) -> Result<crate::runtime::core::freeable_buffer::FreeableBuffer> {
            unreachable!()
        }
        fn load_data_into(
            &self,
            _key: &str,
            _buffer: *mut core::ffi::c_void,
            _size: usize,
        ) -> Error {
            unreachable!()
        }
        fn get_num_keys(&self) -> Result<u32> {
            unreachable!()
        }
        fn get_key(&self, _index: u32) -> Result<*const core::ffi::c_char> {
            unreachable!()
        }
    }

    fn context_with_specs(specs: &[BackendOption]) -> BackendInitContext {
        let span = Span::from_raw_parts(specs.as_ptr() as *mut BackendOption, specs.len());
        BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            span,
        )
    }

    // set_option(workspace_sharing_mode) accepts valid modes and get_option /
    // get_sharing_mode read them back; get_sharing_mode's default is Disabled.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-sharing-mode-fn/test]
    #[test]
    fn sharing_mode_option_roundtrip() {
        setup();
        let options = XnnpackBackendOptions::new();
        assert_eq!(options.get_sharing_mode(), WorkspaceSharingMode::Disabled);

        let set = opt(
            workspace_sharing_mode_option_key,
            OptionValue::Int(WorkspaceSharingMode::Global as i32),
        );
        assert_eq!(options.set_option(&set), Error::Ok);
        assert_eq!(options.get_sharing_mode(), WorkspaceSharingMode::Global);

        let mut get = opt(workspace_sharing_mode_option_key, OptionValue::Bool(false));
        assert_eq!(options.get_option(&mut get), Error::Ok);
        match get.value {
            OptionValue::Int(v) => assert_eq!(v, WorkspaceSharingMode::Global as i32),
            _ => panic!("expected Int"),
        }
    }

    // set_option rejects an out-of-range sharing mode and a wrong-typed value,
    // leaving the stored mode unchanged.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn/test]
    #[test]
    fn sharing_mode_option_rejects_invalid() {
        setup();
        let options = XnnpackBackendOptions::new();
        // Out of range (>= Count).
        let bad_range = opt(
            workspace_sharing_mode_option_key,
            OptionValue::Int(WorkspaceSharingMode::Count as i32),
        );
        assert_eq!(options.set_option(&bad_range), Error::InvalidArgument);
        // Negative.
        let neg = opt(workspace_sharing_mode_option_key, OptionValue::Int(-1));
        assert_eq!(options.set_option(&neg), Error::InvalidArgument);
        // Wrong type (bool instead of int).
        let wrong_type = opt(workspace_sharing_mode_option_key, OptionValue::Bool(true));
        assert_eq!(options.set_option(&wrong_type), Error::InvalidArgument);

        assert_eq!(options.get_sharing_mode(), WorkspaceSharingMode::Disabled);
    }

    // weight_cache_enabled get/set roundtrip and type checking.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn/test]
    #[test]
    fn weight_cache_option_roundtrip() {
        setup();
        let options = XnnpackBackendOptions::new();
        let set = opt(weight_cache_option_key, OptionValue::Bool(true));
        assert_eq!(options.set_option(&set), Error::Ok);

        let mut get = opt(weight_cache_option_key, OptionValue::Int(0));
        assert_eq!(options.get_option(&mut get), Error::Ok);
        match get.value {
            OptionValue::Bool(v) => assert!(v),
            _ => panic!("expected Bool"),
        }

        // Wrong type is rejected.
        let wrong = opt(weight_cache_option_key, OptionValue::Int(1));
        assert_eq!(options.set_option(&wrong), Error::InvalidArgument);
    }

    // packed_cache_path get/set through the option surface and the direct
    // accessors; get_option copies the stored path back into the char array.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-packed-cache-path-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-packed-cache-path-fn/test]
    #[test]
    fn packed_cache_path_roundtrip() {
        setup();
        let options = XnnpackBackendOptions::new();
        assert_eq!(options.get_packed_cache_path(), "");

        options.set_packed_cache_path("/tmp/weights.bin");
        assert_eq!(options.get_packed_cache_path(), "/tmp/weights.bin");

        // Through the option surface.
        let set = opt(
            packed_cache_path_option_key,
            OptionValue::CharArray(char_array(b"/var/cache.bin")),
        );
        assert_eq!(options.set_option(&set), Error::Ok);
        assert_eq!(options.get_packed_cache_path(), "/var/cache.bin");

        let mut get = opt(packed_cache_path_option_key, OptionValue::Bool(false));
        assert_eq!(options.get_option(&mut get), Error::Ok);
        match get.value {
            OptionValue::CharArray(arr) => {
                assert_eq!(cstr_from_char_array(&arr), "/var/cache.bin");
            }
            _ => panic!("expected CharArray"),
        }
    }

    // set_option / get_option ignore unknown keys and return Ok.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-option-fn/test]
    #[test]
    fn unknown_key_is_ignored() {
        setup();
        let options = XnnpackBackendOptions::new();
        let unknown = opt(b"totally_unknown_key\0", OptionValue::Int(7));
        assert_eq!(options.set_option(&unknown), Error::Ok);
        let mut get = opt(b"totally_unknown_key\0", OptionValue::Int(7));
        assert_eq!(options.get_option(&mut get), Error::Ok);
    }

    // The manager accessors hand back the owned sub-managers and the
    // get_or_create/save wrappers delegate to the weights-cache manager.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.workspace-manager-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.weights-cache-manager-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.get-or-create-weights-cache-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.save-weights-cache-locked-fn/test]
    #[test]
    fn manager_accessors_and_delegation() {
        setup();
        let options = XnnpackBackendOptions::new();

        // workspace_manager() default sharing mode mirrors the options default.
        assert_eq!(
            options.workspace_manager().get_sharing_mode(),
            super::super::XNNWorkspaceManager::WorkspaceSharingMode::Disabled
        );

        // weights_cache_manager() starts with no live caches.
        assert_eq!(options.weights_cache_manager().live_count(), 0);

        // get_or_create_weights_cache dedups by path via the manager.
        let a = options
            .get_or_create_weights_cache("/tmp/opt_cache.bin")
            .unwrap();
        let b = options
            .get_or_create_weights_cache("/tmp/opt_cache.bin")
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        assert_eq!(options.weights_cache_manager().live_count(), 1);

        // No cache has been through initialize_for_runtime, so save short-circuits.
        assert_eq!(options.save_weights_cache_locked(), Error::Ok);
    }

    // resolve_weight_cache prefers a runtime spec, else the stored global.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-weight-cache-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn/test]
    #[test]
    fn resolve_weight_cache_prefers_spec() {
        setup();
        let options = XnnpackBackendOptions::new();
        // Global default is false initially.
        let empty = context_with_specs(&[]);
        assert!(!options.resolve_weight_cache(&empty));

        // A runtime spec overrides the global default.
        let specs = [opt(weight_cache_option_key, OptionValue::Bool(true))];
        let ctx = context_with_specs(&specs);
        assert!(options.resolve_weight_cache(&ctx));

        // Set the global default true; empty specs now fall back to it.
        assert_eq!(
            options.set_option(&opt(weight_cache_option_key, OptionValue::Bool(true))),
            Error::Ok
        );
        assert!(options.resolve_weight_cache(&empty));
    }

    // resolve_sharing_mode prefers a runtime spec, validates range, else the
    // stored global.
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.xnnpack-backend-options.resolve-sharing-mode-fn/test]
    // [spec:et:sem:xnnpack-backend-options.executorch.backends.xnnpack.resolve-option-fn/test]
    #[test]
    fn resolve_sharing_mode_prefers_spec_and_validates() {
        setup();
        let options = XnnpackBackendOptions::new();
        let empty = context_with_specs(&[]);
        assert_eq!(
            options.resolve_sharing_mode(&empty).unwrap(),
            WorkspaceSharingMode::Disabled
        );

        let specs = [opt(
            workspace_sharing_mode_option_key,
            OptionValue::Int(WorkspaceSharingMode::PerModel as i32),
        )];
        let ctx = context_with_specs(&specs);
        assert_eq!(
            options.resolve_sharing_mode(&ctx).unwrap(),
            WorkspaceSharingMode::PerModel
        );

        // Out-of-range spec value is rejected.
        let bad = [opt(workspace_sharing_mode_option_key, OptionValue::Int(99))];
        let bad_ctx = context_with_specs(&bad);
        assert_eq!(
            options.resolve_sharing_mode(&bad_ctx).unwrap_err(),
            Error::InvalidArgument
        );
    }

    // Never-instantiated tracer used only to spell a null `*mut dyn EventTracer`.
    struct UninitTracer;
    impl EventTracer for UninitTracer {
        fn state(&self) -> &crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn state_mut(&mut self) -> &mut crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
            unreachable!()
        }
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: crate::runtime::core::event_tracer::ChainID,
            _debug_handle: crate::runtime::core::event_tracer::DebugHandle,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: crate::runtime::core::event_tracer::EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn log_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _start_time: crate::runtime::platform::types::et_timestamp_t,
            _end_time: crate::runtime::platform::types::et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn end_profiling(
            &mut self,
            _prof_entry: crate::runtime::core::event_tracer::EventTracerEntry,
        ) {
            unreachable!()
        }
        fn track_allocation(
            &mut self,
            _id: crate::runtime::core::event_tracer::AllocatorID,
            _size: usize,
        ) {
            unreachable!()
        }
        fn track_allocator(
            &mut self,
            _name: *const core::ffi::c_char,
        ) -> crate::runtime::core::event_tracer::AllocatorID {
            unreachable!()
        }
        fn log_evalue(
            &mut self,
            _evalue: &crate::runtime::core::evalue::EValue,
            _evalue_type: crate::runtime::core::event_tracer::LoggedEValueType,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &crate::runtime::core::portable_type::tensor::Tensor,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: crate::runtime::core::array_ref::ArrayRef<
                crate::runtime::core::portable_type::tensor::Tensor,
            >,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &i32,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &bool,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &f64,
        ) -> Result<bool> {
            unreachable!()
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn crate::runtime::core::event_tracer::EventTracerFilterBase,
        ) {
            unreachable!()
        }
    }
}
