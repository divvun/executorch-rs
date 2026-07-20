//! Literal port of backends/xnnpack/runtime/XNNPACKBackend.cpp +
//! backends/xnnpack/runtime/XNNPACKBackend.h.
//!
//! PORT-NOTE: The `XnnpackBackend` implements the tier-2 `BackendInterface`
//! trait. Because `init`/`execute`/`destroy` drive the XNNPACK C API (via
//! `XNNExecutor` / `XNNCompiler`), the backend body is gated behind the
//! `xnnpack` feature. The option-key constants and `WorkspaceSharingMode` enum
//! (originally in `XNNPACKBackend.h`, which has no XNNPACK dependency) are
//! defined unconditionally so other modules can name them regardless of the
//! feature.
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

// ------------------------------------------------------------------------
// XNNPACKBackend.h — option keys + WorkspaceSharingMode (no XNNPACK dep).
// ------------------------------------------------------------------------

// PORT-NOTE: The C++ header stores these keys as `const char name[] = "..."`.
// Ported as byte-string constants that include the trailing NUL, so `strcmp`
// ports (byte-by-byte compare through the NUL) and `.as_ptr()` yields a valid
// `*const c_char` for the runtime-spec lookups.

/// The key for the backend. Used to register the backend, check availability,
/// and get/set options.
pub const xnnpack_backend_key: &[u8] = b"XnnpackBackend\0";

/// The key for the workspace sharing option.
pub const workspace_sharing_mode_option_key: &[u8] = b"workspace_sharing_mode\0";

/// The key for the weight cache option.
pub const weight_cache_option_key: &[u8] = b"weight_cache_enabled\0";

/// Path for the packed weight file.
pub const packed_cache_path_option_key: &[u8] = b"packed_cache_path\0";

/// EXPERIMENTAL — triggers persisting the packed weight cache to disk.
pub const save_weight_cache_on_disk_option_key: &[u8] = b"save_weight_cache_on_disk\0";

/// Workspace sharing mode. Controls memory sharing between CALL_DELEGATE
/// instances.
// [spec:et:def:xnnpack-backend.executorch.backends.xnnpack.workspace-sharing-mode]
//
// PORT-NOTE: This is the canonical `WorkspaceSharingMode` the spec assigns to
// the XNNPACKBackend module. The sibling `XNNWorkspaceManager` currently defines
// its own copy (a wave placeholder); when it lands it should re-use this
// definition. UNRESOLVED CROSS-MODULE REFERENCE: the two `WorkspaceSharingMode`
// definitions are structurally identical (Disabled=0, PerModel=1, Global=2,
// Count=3) but distinct types; conversions go through the `i32` discriminant.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WorkspaceSharingMode {
    Disabled = 0,
    PerModel = 1,
    Global = 2,
    Count = 3,
}

impl WorkspaceSharingMode {
    /// PORT-NOTE: `static_cast<WorkspaceSharingMode>(raw)` in the C++, used only
    /// after the caller has range-checked `0 <= raw < Count`.
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => WorkspaceSharingMode::Disabled,
            1 => WorkspaceSharingMode::PerModel,
            2 => WorkspaceSharingMode::Global,
            _ => WorkspaceSharingMode::Count,
        }
    }
}

// ------------------------------------------------------------------------
// XNNPACKBackend.cpp — the backend implementation.
// ------------------------------------------------------------------------

#[cfg(feature = "xnnpack")]
mod backend {
    use super::*;

    use crate::backends::xnnpack::runtime::sys::{xnn_initialize, xnn_status_success};
    // UNRESOLVED CROSS-MODULE REFERENCE: `XNNCompiler::compileModel` — the
    // `XNNCompiler` module is still a wave stub; `init` calls
    // `XNNCompiler::compileModel(...)` below via its full path, which resolves
    // once the compiler module lands.
    use crate::backends::xnnpack::runtime::XNNCompiler::XNNCompiler;
    use crate::backends::xnnpack::runtime::XNNExecutor::XNNExecutor;
    use crate::backends::xnnpack::runtime::XnnpackBackendOptions::XnnpackBackendOptions;

    use crate::runtime::backend::backend_execution_context::BackendExecutionContext;
    use crate::runtime::backend::backend_init_context::BackendInitContext;
    use crate::runtime::backend::backend_option_context::BackendOptionContext;
    use crate::runtime::backend::interface::{
        Backend, BackendInterface, CompileSpec, DelegateHandle, get_backend_class, register_backend,
    };
    use crate::runtime::backend::options::BackendOption;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::freeable_buffer::FreeableBuffer;
    use crate::runtime::core::memory_allocator::MemoryAllocatorExt;
    use crate::runtime::core::result::{Result, ResultExt};
    use crate::runtime::core::span::Span;

    // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend]
    pub struct XnnpackBackend {
        // `mutable xnnpack::XnnpackBackendOptions options_;` — the only mutable
        // state; the C++ `init` is `const` but writes `options_` through its
        // `mutable` qualifier. In Rust the interior mutability lives inside
        // `XnnpackBackendOptions` (atomics + mutexes), so a shared `&self`
        // suffices and no outer cell is needed.
        options_: XnnpackBackendOptions,
    }

    impl XnnpackBackend {
        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.xnnpack-backend-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.xnnpack-backend-fn]
        pub fn new() -> Self {
            // Initialize XNNPACK
            let status = unsafe { xnn_initialize(core::ptr::null()) };
            if status != xnn_status_success {
                crate::et_log!(
                    Error,
                    "Failed to initialize, XNNPACK status: 0x{:x}",
                    status.0
                );
                // PORT-NOTE: `return;` from the ctor — a failed init still yields
                // a constructed object.
                return XnnpackBackend {
                    options_: XnnpackBackendOptions::new(),
                };
            }
            XnnpackBackend {
                options_: XnnpackBackendOptions::new(),
            }
        }
    }

    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface]
    impl BackendInterface for XnnpackBackend {
        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn]
        fn is_available(&self) -> bool {
            xnn_status_success == unsafe { xnn_initialize(core::ptr::null()) }
        }

        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn]
        fn init(
            &self,
            context: &mut BackendInitContext,
            processed: *mut FreeableBuffer,
            _compile_specs: ArrayRef<CompileSpec>,
        ) -> Result<*mut DelegateHandle> {
            // PORT-NOTE: `allocateInstance<XNNExecutor>()` — the C++ template
            // helper takes no explicit alignment (uses `alignof(T)`). The Rust
            // helper takes an explicit `alignment`; pass `align_of::<T>()`.
            let allocator = context.get_runtime_allocator();
            let executor: *mut XNNExecutor = unsafe {
                (*allocator).allocate_instance::<XNNExecutor>(core::mem::align_of::<XNNExecutor>())
            };
            if executor.is_null() {
                return Err(Error::MemoryAllocationFailed);
            }

            let named_data_map = context.get_named_data_map();

            let program_id = context.get_runtime_allocator() as *const () as usize;
            let sharing_mode_result = self.options_.resolve_sharing_mode(context);
            if !sharing_mode_result.is_ok() {
                return Err(sharing_mode_result.error());
            }
            // PORT-NOTE: C++ calls `workspace_manager().get_or_create_workspace(
            // program_id, sharing_mode)`. The sibling Rust manager exposes this
            // as `get_or_create_workspace_with_mode(program_id, mode)`; the mode
            // is the manager's own `WorkspaceSharingMode` type, so translate this
            // module's mode through its `i32` discriminant. UNRESOLVED
            // CROSS-MODULE REFERENCE: the two `WorkspaceSharingMode` types.
            let mgr_mode = {
                use crate::backends::xnnpack::runtime::XNNWorkspaceManager::WorkspaceSharingMode as MgrMode;
                match *sharing_mode_result.get() as i32 {
                    0 => MgrMode::Disabled,
                    1 => MgrMode::PerModel,
                    2 => MgrMode::Global,
                    _ => MgrMode::Count,
                }
            };
            let workspace_result = self
                .options_
                .workspace_manager()
                .get_or_create_workspace_with_mode(program_id, mgr_mode);
            if !workspace_result.is_ok() {
                return Err(workspace_result.error());
            }
            let workspace = workspace_result.get().clone();

            let use_weight_cache = self.options_.resolve_weight_cache(context);
            // Per-path weights cache: same-path PTEs share an instance and
            // serialize on its mutex; different paths run in parallel.
            let mut weights_cache: Option<
                std::sync::Arc<crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache>,
            > = None;
            // PORT-NOTE: `std::unique_lock<std::mutex> lock_weights_cache;`
            // (initially unlocked). Modeled as an `Option<MutexGuard>` held for
            // the rest of init. `'a` lifetime borrows the cache Arc below.
            let mut lock_weights_cache_holder: Option<std::sync::MutexGuard<'_, ()>> = None;
            if use_weight_cache {
                // Only honor a path coming through runtime_spec (per-PTE opt-in).
                let mut cache_path = String::new();
                let path_spec = context.get_runtime_spec_str(
                    packed_cache_path_option_key.as_ptr() as *const core::ffi::c_char
                );
                if path_spec.is_ok() {
                    cache_path = unsafe { cstr_to_string(*path_spec.get()) };
                }
                let wc_result = self.options_.get_or_create_weights_cache(&cache_path);
                if !wc_result.is_ok() {
                    return Err(wc_result.error());
                }
                let cache = wc_result.get().clone();
                // SAFETY: the guard borrows the cache's mutex; the cache Arc is
                // kept alive in `weights_cache` for the rest of init. The C++
                // holds `lock_weights_cache` over the same span.
                let guard = unsafe {
                    core::mem::transmute::<
                        std::sync::MutexGuard<'_, ()>,
                        std::sync::MutexGuard<'_, ()>,
                    >(cache.mutex().lock().unwrap())
                };
                lock_weights_cache_holder = Some(guard);

                // PORT-NOTE: `initialize_for_runtime` needs `&mut` on the cache;
                // the C++ mutates through the shared_ptr while holding the
                // instance mutex. Mirror that aliasing via a raw-pointer &mut,
                // exactly as XNNWeightsCacheManager::save_all does.
                let cache_mut = unsafe {
                    &mut *(std::sync::Arc::as_ptr(&cache)
                        as *mut crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache)
                };
                // PORT-NOTE / CROSS-MODULE MISMATCH: the sibling
                // `XNNWeightsCache::initialize_for_runtime` takes `*mut
                // MemoryAllocator` (the concrete struct), but
                // `BackendInitContext::get_runtime_allocator` returns `*mut dyn
                // MemoryAllocatorBase` (matching the C++ `MemoryAllocator*`
                // erased to a base pointer). The two should agree on one type;
                // here the trait-object's data pointer is cast to the concrete
                // allocator. Noted for the fixer — the cache API should accept
                // `*mut dyn MemoryAllocatorBase`.
                let alloc_concrete = context.get_runtime_allocator() as *mut ()
                    as *mut crate::runtime::core::memory_allocator::MemoryAllocator;
                cache_mut.initialize_for_runtime(alloc_concrete, named_data_map);
                workspace.set_uses_weight_cache();
                weights_cache = Some(cache);
            }

            let (_workspace_lock, workspace_ptr) = workspace.acquire();

            // Executor has been allocated but not constructed; construct it in
            // place. NOTE: since we "placement new" and the type is not trivially
            // destructible, the destructor must be called manually in destroy().
            unsafe {
                core::ptr::write(executor, XNNExecutor::new(workspace.clone()));
            }
            let wc_ptr = match &weights_cache {
                Some(c) => unsafe {
                    &mut *(std::sync::Arc::as_ptr(c)
                        as *mut crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache)
                        as *mut crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache
                },
                None => core::ptr::null_mut(),
            };
            let err = XNNCompiler::compileModel(
                unsafe { (*processed).data() },
                unsafe { (*processed).size() },
                executor,
                wc_ptr,
                workspace_ptr,
                named_data_map,
                use_weight_cache,
            );
            // This backend does not need its processed data after compiling.
            unsafe { (*processed).free() };

            if err != Error::Ok {
                // destroy() won't be called on this handle, so clean it up now.
                unsafe {
                    core::ptr::drop_in_place(executor);
                }
                crate::et_log!(
                    Error,
                    "XNNCompiler::compileModel failed: 0x{:x}",
                    err as u32
                );
                return Err(err);
            }

            // Hand the cache to the executor (held by Arc so it outlives any
            // sibling executors that share it).
            if use_weight_cache {
                unsafe {
                    (*executor).set_weights_cache(weights_cache);
                }
            }

            // Locks release on return.
            drop(lock_weights_cache_holder);
            Ok(executor as *mut DelegateHandle)
        }

        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.execute-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.execute-fn]
        fn execute(
            &self,
            context: &mut BackendExecutionContext,
            handle: *mut DelegateHandle,
            args: Span<*mut EValue>,
        ) -> Error {
            let executor = handle as *mut XNNExecutor;

            let workspace = unsafe { (*executor).get_workspace() };

            // Lock the cache shared with sibling executors at the same path.
            // Empty cache → PTE didn't opt into file-backed mode.
            let cache = unsafe { (*executor).get_weights_cache() };
            let mut _lock_weights_cache: Option<std::sync::MutexGuard<'_, ()>> = None;
            if let Some(c) = &cache {
                let guard = unsafe {
                    core::mem::transmute::<
                        std::sync::MutexGuard<'_, ()>,
                        std::sync::MutexGuard<'_, ()>,
                    >(c.mutex().lock().unwrap())
                };
                _lock_weights_cache = Some(guard);
            }

            let (_raii_lock, _) = workspace.acquire();

            // Prepare Inputs/Outputs and Propagate Input Shapes
            let mut err = unsafe { (*executor).prepare_args(args) };
            if err != Error::Ok {
                return err;
            }

            err = unsafe { (*executor).forward(context) };

            if err != Error::Ok {
                return err;
            }

            // Convert output data types if necessary (int32 -> int64 for Long)
            err = unsafe { (*executor).convert_outputs(args) };

            err
        }

        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn]
        fn destroy(&self, handle: *mut DelegateHandle) {
            if !handle.is_null() {
                let executor = handle as *mut XNNExecutor;
                let workspace = unsafe { (*executor).get_workspace() };
                let cache = unsafe { (*executor).get_weights_cache() };

                // Local Arc keeps the instance alive through delete_packed_data
                // even if the executor was the last holder.
                let mut _lock_weights_cache: Option<std::sync::MutexGuard<'_, ()>> = None;
                if let Some(c) = &cache {
                    let guard = unsafe {
                        core::mem::transmute::<
                            std::sync::MutexGuard<'_, ()>,
                            std::sync::MutexGuard<'_, ()>,
                        >(c.mutex().lock().unwrap())
                    };
                    _lock_weights_cache = Some(guard);
                }

                #[cfg(any(feature = "profiling-enabled", feature = "xnnpack-profiling"))]
                unsafe {
                    (*executor).print_avg_op_timings();
                }

                if let Some(c) = &cache {
                    if unsafe { (*executor).uses_weight_cache() } {
                        // PORT-NOTE: `delete_packed_data` needs `&mut` on the
                        // cache; mutate through the Arc under the held instance
                        // mutex, mirroring the C++ shared_ptr aliasing.
                        let cache_mut = unsafe {
                            &mut *(std::sync::Arc::as_ptr(c)
                                as *mut crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache)
                        };
                        let names = unsafe { (*executor).get_packed_data_names() };
                        let _ = cache_mut.delete_packed_data(&names);
                    }
                }

                // Serialize access to xnn_delete_runtime (not thread safe). Hold
                // onto the workspace Arc, as the executor (whose runtime pointer
                // deleter runs during drop) is about to be destroyed.
                let (_raii_lock, _) = workspace.acquire();

                // XNNExecutor is not trivially destructible; it was constructed
                // manually in init(), so destroy it manually here.
                unsafe {
                    core::ptr::drop_in_place(executor);
                }
            }
        }

        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn]
        fn get_option(
            &self,
            _context: &mut BackendOptionContext,
            backend_options: &mut Span<BackendOption>,
        ) -> Error {
            for i in 0..backend_options.size() {
                let opt = unsafe { backend_options.index(i) };
                let err = self.options_.get_option(opt);
                if err != Error::Ok {
                    return err;
                }
            }
            Error::Ok
        }

        // [spec:et:def:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn]
        // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn]
        fn set_option(
            &self,
            _context: &mut BackendOptionContext,
            backend_options: &Span<BackendOption>,
        ) -> Error {
            // Process every option even if one fails; capture the first error.
            let mut first_err = Error::Ok;
            for i in 0..backend_options.size() {
                let option = unsafe { backend_options.index(i) };
                let err = self.options_.set_option(option);
                if err != Error::Ok && first_err == Error::Ok {
                    first_err = err;
                }
            }
            first_err
        }
    }

    // PORT-NOTE: `std::string(const char*)` — build a `String` from a C string.
    unsafe fn cstr_to_string(p: *const core::ffi::c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { core::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    // ---- static registration --------------------------------------------
    //
    // PORT-NOTE: The C++ registers a single static instance at load time:
    //   `auto backend_instance = XnnpackBackend();`
    //   `Backend backend{xnnpack_backend_key, &backend_instance};`
    //   `static auto success = register_backend(backend);`
    // Rust has no C++-style static initializers that run code at load. The
    // `#[pragma clang diagnostic ignored "-Wglobal-constructors"]` global-ctor
    // machinery has no Rust analogue. The registration is exposed as an explicit
    // `register()` hook the runtime must call once during startup. The backend
    // instance is a leaked `Box` so its address is stable for the lifetime of
    // the process, matching the C++ file-scope static's lifetime.
    //
    // DEVIATION FROM C++ (intentional): Rust has no load-time static
    // initializer, so this is an explicit entry point a consumer calls once
    // before loading an XNNPACK-delegated `.pte`. This is the same seam the C++
    // hides behind `+whole-archive` linker tricks (see the divvun-speech-rs
    // wrapper's build.rs) — surfaced here as an ordinary, idempotent function.
    ///
    /// Registers the XNNPACK delegate in the global backend registry so
    /// `Method` loading dispatches `XnnpackBackend`-tagged delegate calls to it.
    /// Idempotent: calling it more than once is a no-op that returns `Ok`. Must
    /// be called before loading a `.pte` that uses the XNNPACK delegate.
    pub fn register() -> Error {
        // Idempotent: if a backend with this key is already registered, don't
        // leak a second instance or trip `register_backend`'s duplicate-name
        // guard.
        let key = xnnpack_backend_key.as_ptr() as *const core::ffi::c_char;
        if !get_backend_class(key).is_null() {
            return Error::Ok;
        }
        let backend_instance: &'static mut XnnpackBackend =
            Box::leak(Box::new(XnnpackBackend::new()));
        let backend = Backend {
            name: key,
            backend: backend_instance as *mut XnnpackBackend as *mut dyn BackendInterface,
        };
        register_backend(&backend)
    }
}

#[cfg(feature = "xnnpack")]
pub use backend::{XnnpackBackend, register};

// Literal port of backends/xnnpack/test/runtime/test_weight_cache.cpp and
// backends/xnnpack/test/runtime/test_workspace_sharing.cpp. Both suites drive
// the XNNPACK backend's runtime options (`weight_cache_enabled`,
// `workspace_sharing_mode`) through `set_option`/`get_option`, and some also
// load/run PTE fixtures via `Module`.
//
// PORT-NOTE: LINK GAP. Every test registers the XNNPACK backend
// (`XnnpackBackend::register()`) so the option get/set path resolves; the
// backend body references `extern "C"` XNNPACK symbols that nothing links yet,
// so these compile under `--features xnnpack` but cannot link/run. Gated by the
// feature per the Wave-3 xnnpack convention; the default `cargo test` build does
// not enable it.
//
// PORT-NOTE: FIXTURE + MODULE-API DIVERGENCE. The `RunWith*` / `Overrides*` /
// `NotSet*` cases load a `.pte` via `Module` (env vars
// `ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH` /
// `ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH`) and drive `Module::load(
// LoadBackendOptionsMap)` + `Module::forward({tensors})`. The ported Rust
// `Module` exposes `load(Verification)` and a different `forward` shape, so the
// backend-options `load` overload and the tensor-list `forward` used by these
// tests have no faithful Rust surface yet. They are `#[ignore]`d and skip early
// when the fixture env var is unset; port them alongside a `Module` port that
// grows the `load(LoadBackendOptionsMap)` / tensor-list `forward` API.
#[cfg(all(test, feature = "xnnpack"))]
mod tests {
    use super::*;
    use crate::runtime::backend::interface::{get_option, set_option};
    use crate::runtime::backend::options::{BackendOption, BackendOptions, OptionValue};
    use crate::runtime::core::error::Error;
    use crate::runtime::core::span::Span;

    // Turn a NUL-terminated byte literal into the `[c_char; N]` key array the
    // option API expects (mirrors the C++ `const char key[N]`).
    const fn key<const N: usize>(bytes: &[u8; N]) -> [core::ffi::c_char; N] {
        let mut out = [0 as core::ffi::c_char; N];
        let mut i = 0;
        while i < N {
            out[i] = bytes[i] as core::ffi::c_char;
            i += 1;
        }
        out
    }

    // Mirrors the various `runtime_init()` + backend-registration prologue. The
    // C++ backend self-registers at load; the Rust port registers explicitly.
    // Registration is idempotent-safe to attempt once per process here.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
        let _ = register();
    }

    fn xnnpack_key() -> *const core::ffi::c_char {
        xnnpack_backend_key.as_ptr() as *const core::ffi::c_char
    }

    // ---- test_weight_cache.cpp helpers ------------------------------------

    fn set_and_check_weight_cache_enabled(enabled: bool) {
        setup();

        let mut backend_options: BackendOptions<1> = BackendOptions::new();
        backend_options.set_option_bool(&key(b"weight_cache_enabled\0"), enabled);

        let status = set_option(xnnpack_key(), backend_options.view());
        assert_eq!(status, Error::Ok);

        let mut read_option = BackendOption::new();
        read_option.key[..weight_cache_option_key.len()].copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                weight_cache_option_key.as_ptr() as *const core::ffi::c_char,
                weight_cache_option_key.len(),
            )
        });
        read_option.value = OptionValue::Bool(!enabled);
        let mut arr = [read_option];
        let span: Span<BackendOption> = Span::from_raw_parts(arr.as_mut_ptr(), 1);
        let _ = get_option(xnnpack_key(), span);

        if let OptionValue::Bool(v) = arr[0].value {
            assert_eq!(v, enabled);
        } else {
            panic!("expected bool option value");
        }
    }

    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn/test]
    #[test]
    #[ignore]
    fn weight_cache_set_enabled() {
        set_and_check_weight_cache_enabled(true);
        set_and_check_weight_cache_enabled(false);
        set_and_check_weight_cache_enabled(true);
    }

    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    #[test]
    #[ignore]
    fn weight_cache_set_invalid_type() {
        setup();

        // Weight cache option expects a bool, not an int.
        let mut backend_options: BackendOptions<1> = BackendOptions::new();
        backend_options.set_option_int(&key(b"weight_cache_enabled\0"), 1);

        let status = set_option(xnnpack_key(), backend_options.view());
        assert_eq!(status, Error::InvalidArgument);
    }

    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn/test]
    #[test]
    #[ignore]
    fn weight_cache_set_multiple_options() {
        setup();

        // Set both options at once.
        let mut backend_options: BackendOptions<2> = BackendOptions::new();
        backend_options.set_option_int(
            &key(b"workspace_sharing_mode\0"),
            WorkspaceSharingMode::Global as i32,
        );
        backend_options.set_option_bool(&key(b"weight_cache_enabled\0"), false);

        let status = set_option(xnnpack_key(), backend_options.view());
        assert_eq!(status, Error::Ok);

        // Read both back.
        let mut read_workspace = BackendOption::new();
        read_workspace.key[..workspace_sharing_mode_option_key.len()].copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                workspace_sharing_mode_option_key.as_ptr() as *const core::ffi::c_char,
                workspace_sharing_mode_option_key.len(),
            )
        });
        read_workspace.value = OptionValue::Int(-1);
        let mut ws_arr = [read_workspace];
        let _ = get_option(xnnpack_key(), Span::from_raw_parts(ws_arr.as_mut_ptr(), 1));
        if let OptionValue::Int(v) = ws_arr[0].value {
            assert_eq!(v, WorkspaceSharingMode::Global as i32);
        } else {
            panic!("expected int option value");
        }

        let mut read_cache = BackendOption::new();
        read_cache.key[..weight_cache_option_key.len()].copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                weight_cache_option_key.as_ptr() as *const core::ffi::c_char,
                weight_cache_option_key.len(),
            )
        });
        read_cache.value = OptionValue::Bool(true);
        let mut c_arr = [read_cache];
        let _ = get_option(xnnpack_key(), Span::from_raw_parts(c_arr.as_mut_ptr(), 1));
        if let OptionValue::Bool(v) = c_arr[0].value {
            assert_eq!(v, false);
        } else {
            panic!("expected bool option value");
        }
    }

    // PORT-NOTE: FIXTURE + MODULE-API DIVERGENCE (see module-level note). The
    // C++ `RuntimeSpec.OverridesGlobalWeightCache` loads
    // `ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH` through `Module` with a
    // `LoadBackendOptionsMap` override. No faithful Rust surface yet.
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    #[test]
    #[ignore]
    fn runtime_spec_overrides_global_weight_cache() {
        setup();
        if std::env::var("ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH").is_err() {
            eprintln!(
                "skipping runtime_spec_overrides_global_weight_cache: \
                 ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH unset"
            );
            return;
        }
        // Global-vs-runtime-spec override + Module load/run: no faithful surface.
    }

    // ---- test_workspace_sharing.cpp helpers -------------------------------

    fn set_and_check_workspace_sharing_mode(mode: WorkspaceSharingMode) {
        setup();

        let mut backend_options: BackendOptions<1> = BackendOptions::new();
        backend_options.set_option_int(&key(b"workspace_sharing_mode\0"), mode as i32);

        let status = set_option(xnnpack_key(), backend_options.view());
        assert_eq!(status, Error::Ok);

        // Read back to sanity check.
        let mut read_option = BackendOption::new();
        read_option.key[..workspace_sharing_mode_option_key.len()].copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                workspace_sharing_mode_option_key.as_ptr() as *const core::ffi::c_char,
                workspace_sharing_mode_option_key.len(),
            )
        });
        read_option.value = OptionValue::Int(-1);
        let mut arr = [read_option];
        let _ = get_option(xnnpack_key(), Span::from_raw_parts(arr.as_mut_ptr(), 1));
        if let OptionValue::Int(v) = arr[0].value {
            assert_eq!(v, mode as i32);
        } else {
            panic!("expected int option value");
        }
    }

    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.get-option-fn/test]
    #[test]
    #[ignore]
    fn workspace_sharing_set_mode() {
        set_and_check_workspace_sharing_mode(WorkspaceSharingMode::Disabled);
        set_and_check_workspace_sharing_mode(WorkspaceSharingMode::PerModel);
        set_and_check_workspace_sharing_mode(WorkspaceSharingMode::Global);
    }

    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    #[test]
    #[ignore]
    fn workspace_sharing_set_invalid_mode() {
        // Set to an initial known value.
        set_and_check_workspace_sharing_mode(WorkspaceSharingMode::PerModel);

        // Set to a bad value.
        let mut backend_options: BackendOptions<1> = BackendOptions::new();
        backend_options.set_option_int(&key(b"workspace_sharing_mode\0"), 70);

        let status = set_option(xnnpack_key(), backend_options.view());
        assert_eq!(status, Error::InvalidArgument);

        // Make sure the option is still set to a valid value.
        let mut read_option = BackendOption::new();
        read_option.key[..workspace_sharing_mode_option_key.len()].copy_from_slice(unsafe {
            core::slice::from_raw_parts(
                workspace_sharing_mode_option_key.as_ptr() as *const core::ffi::c_char,
                workspace_sharing_mode_option_key.len(),
            )
        });
        read_option.value = OptionValue::Int(-1);
        let mut arr = [read_option];
        let _ = get_option(xnnpack_key(), Span::from_raw_parts(arr.as_mut_ptr(), 1));
        if let OptionValue::Int(v) = arr[0].value {
            assert_eq!(v, WorkspaceSharingMode::PerModel as i32);
        } else {
            panic!("expected int option value");
        }
    }

    // The `RunWith*` / `RunWithModeSwitch` / `RuntimeSpec.*` cases load and run
    // `.pte` fixtures through `Module`; see the module-level FIXTURE note. Each is
    // ported as its own fixture-skipping stub (one per C++ TEST) so no test
    // identity is dropped; all skip early when the fixtures are unset.
    fn require_add_sub_fixtures(name: &str) -> bool {
        if std::env::var("ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH").is_err()
            || std::env::var("ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH").is_err()
        {
            eprintln!(
                "skipping {}: ET_XNNPACK_GENERATED_{{ADD,SUB}}_LARGE_PTE_PATH unset",
                name
            );
            return false;
        }
        true
    }

    #[test]
    #[ignore]
    fn workspace_sharing_run_with_disabled_mode() {
        setup();
        if !require_add_sub_fixtures("workspace_sharing_run_with_disabled_mode") {
            return;
        }
        // run_and_validate_two_models(Disabled): Module load/run — no surface yet.
    }

    #[test]
    #[ignore]
    fn workspace_sharing_run_with_per_model_mode() {
        setup();
        if !require_add_sub_fixtures("workspace_sharing_run_with_per_model_mode") {
            return;
        }
    }

    #[test]
    #[ignore]
    fn workspace_sharing_run_with_global_mode() {
        setup();
        if !require_add_sub_fixtures("workspace_sharing_run_with_global_mode") {
            return;
        }
    }

    #[test]
    #[ignore]
    fn workspace_sharing_run_with_mode_switch() {
        setup();
        if !require_add_sub_fixtures("workspace_sharing_run_with_mode_switch") {
            return;
        }
    }

    // PORT-NOTE: FIXTURE + MODULE-API DIVERGENCE (see module-level note).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    #[test]
    #[ignore]
    fn runtime_spec_overrides_global_workspace_mode() {
        setup();
        if !require_add_sub_fixtures("runtime_spec_overrides_global_workspace_mode") {
            return;
        }
    }

    // PORT-NOTE: FIXTURE + MODULE-API DIVERGENCE (see module-level note).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.set-option-fn/test]
    #[test]
    #[ignore]
    fn runtime_spec_not_set_falls_back_to_global() {
        setup();
        if !require_add_sub_fixtures("runtime_spec_not_set_falls_back_to_global") {
            return;
        }
    }

    // ---- constructor + is_available -------------------------------------
    //
    // PORT-NOTE: The C++ backend has no dedicated ctor/is_available gtest — it is
    // a file-scope static exercised transitively by every suite. These two unit
    // tests genuinely drive the ctor (`xnn_initialize`) and `is_available`
    // (`xnn_initialize == success`) directly, which the option suites above do
    // not: they go through the already-registered singleton without constructing
    // a fresh instance or calling `is_available`. Neither path needs a delegated
    // `.pte`; they only touch `xnn_initialize`, which the XNNPACK C lib provides.

    // The ctor initializes XNNPACK and always yields a constructed object (a
    // failed init still `return`s a constructed backend in the C++).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.xnnpack-backend-fn/test]
    #[test]
    fn ctor_constructs_backend() {
        crate::runtime::platform::runtime::runtime_init();
        // Constructing must succeed and not panic; XNNPACK gets initialized.
        let _backend = XnnpackBackend::new();
    }

    // is_available() returns true once XNNPACK initializes successfully.
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.is-available-fn/test]
    #[test]
    fn is_available_reports_true() {
        use crate::runtime::backend::interface::BackendInterface;
        crate::runtime::platform::runtime::runtime_init();
        let backend = XnnpackBackend::new();
        assert!(backend.is_available());
    }

    // ---- init / execute / destroy ----------------------------------------
    //
    // PORT-NOTE: These have no gtest counterpart (the C++ exercises
    // init/execute/destroy only transitively through Method loading of a
    // delegated .pte). init()'s SUCCESS path needs a valid serialized XNNPACK
    // delegate payload (flatbuffer + XN01 header) and remains fixture-gapped;
    // its allocation-failure and corrupt-payload error paths, and the full
    // execute()/destroy() flow over a handle backed by a real XNNPACK runtime
    // (the same executor shape init() itself produces), are exercised directly
    // below against the linked XNNPACK C library.

    // Never-instantiated map used only to spell a null `*const dyn
    // NamedDataMap` for `BackendInitContext` (the corrupt-payload init paths
    // return before the map is ever dereferenced).
    struct NullMap;
    impl crate::runtime::core::named_data_map::NamedDataMap for NullMap {
        fn get_tensor_layout(
            &self,
            _key: &str,
        ) -> crate::runtime::core::result::Result<crate::runtime::core::tensor_layout::TensorLayout>
        {
            unreachable!()
        }
        fn get_data(
            &self,
            _key: &str,
        ) -> crate::runtime::core::result::Result<
            crate::runtime::core::freeable_buffer::FreeableBuffer,
        > {
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
        fn get_num_keys(&self) -> crate::runtime::core::result::Result<u32> {
            unreachable!()
        }
        fn get_key(
            &self,
            _index: u32,
        ) -> crate::runtime::core::result::Result<*const core::ffi::c_char> {
            unreachable!()
        }
    }

    fn null_named_data_map() -> *const dyn crate::runtime::core::named_data_map::NamedDataMap {
        core::ptr::null::<NullMap>()
            as *const dyn crate::runtime::core::named_data_map::NamedDataMap
    }

    fn null_event_tracer() -> *mut dyn crate::runtime::core::event_tracer::EventTracer {
        crate::extension::module::module::null_event_tracer()
    }

    // init() must fail with MemoryAllocationFailed when the runtime allocator
    // cannot allocate the XNNExecutor slot (the very first init step), before
    // touching the processed buffer.
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn/test]
    #[test]
    fn init_reports_memory_allocation_failure() {
        use crate::runtime::backend::backend_init_context::BackendInitContext;
        use crate::runtime::backend::interface::{BackendInterface, CompileSpec};
        use crate::runtime::core::array_ref::ArrayRef;
        use crate::runtime::core::freeable_buffer::FreeableBuffer;
        use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

        crate::runtime::platform::runtime::runtime_init();
        let backend = XnnpackBackend::new();

        let mut allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let mut context = BackendInitContext::new(
            &mut allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );
        let mut processed = FreeableBuffer::new();
        let compile_specs: ArrayRef<CompileSpec> = ArrayRef::new();

        let result = backend.init(&mut context, &mut processed, compile_specs);
        assert_eq!(result.err(), Some(Error::MemoryAllocationFailed));
    }

    // init() over a corrupt processed payload (no XN00 XNNHeader magic, no
    // XN00/XN01 flatbuffer identifier) must run the real flow — allocate the
    // executor, resolve sharing mode and workspace, dispatch
    // XNNCompiler::compileModel — then fail DelegateInvalidCompatibility and
    // still free the processed buffer (the C++ calls `processed->Free()`
    // unconditionally after compileModel).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.init-fn/test]
    #[test]
    fn init_with_corrupt_payload_fails_and_frees_processed() {
        use crate::runtime::backend::backend_init_context::BackendInitContext;
        use crate::runtime::backend::interface::{BackendInterface, CompileSpec};
        use crate::runtime::core::array_ref::ArrayRef;
        use crate::runtime::core::freeable_buffer::FreeableBuffer;
        use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
        use std::sync::atomic::{AtomicBool, Ordering};

        crate::runtime::platform::runtime::runtime_init();
        let backend = XnnpackBackend::new();

        let mut pool = vec![0u8; 8192];
        let mut allocator = MemoryAllocator::new(pool.len() as u32, pool.as_mut_ptr());
        let mut context = BackendInitContext::new(
            &mut allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );

        unsafe extern "C" fn mark_freed(
            context: *mut core::ffi::c_void,
            _data: *mut core::ffi::c_void,
            _size: usize,
        ) {
            unsafe { (*(context as *const AtomicBool)).store(true, Ordering::SeqCst) };
        }

        // Large enough to pass the XNNHeader minimum-size check (so the magic
        // mismatch reports NotFound and falls through to the flatbuffer
        // identifier check), but carrying no valid identifier.
        let payload = vec![0xABu8; 64];
        let freed = AtomicBool::new(false);
        let mut processed = FreeableBuffer::from_pointer(
            payload.as_ptr() as *const core::ffi::c_void,
            payload.len(),
            Some(mark_freed),
            &freed as *const AtomicBool as *mut core::ffi::c_void,
        );
        let compile_specs: ArrayRef<CompileSpec> = ArrayRef::new();

        let result = backend.init(&mut context, &mut processed, compile_specs);
        assert_eq!(result.err(), Some(Error::DelegateInvalidCompatibility));
        assert!(
            freed.load(Ordering::SeqCst),
            "init must free the processed buffer even when compileModel fails"
        );
    }

    // execute() + destroy() over a delegate handle backed by a real XNNPACK
    // runtime (single binary-add fp32 subgraph), constructed exactly the way
    // init() constructs one: placement-new into runtime-allocator memory,
    // wrapped around a live workspace. execute() locks the workspace, prepares
    // args, forwards, and converts outputs; destroy() re-acquires the workspace
    // and manually destructs the executor (deleting the runtime).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.execute-fn/test]
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn/test]
    #[test]
    fn execute_and_destroy_roundtrip_real_runtime() {
        use crate::backends::xnnpack::runtime::XNNExecutor::XNNExecutor;
        use crate::backends::xnnpack::runtime::XNNWorkspace::XNNWorkspace;
        use crate::backends::xnnpack::runtime::sys::{
            XNN_FLAG_BASIC_PROFILING, XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_EXTERNAL_INPUT,
            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, pthreadpool_t, xnn_binary_add, xnn_create_runtime_v4,
            xnn_create_subgraph, xnn_datatype_fp32, xnn_define_binary, xnn_define_tensor_value,
            xnn_delete_subgraph, xnn_initialize, xnn_runtime_t, xnn_status_success, xnn_subgraph_t,
            xnn_weights_cache_t,
        };
        use crate::runtime::backend::backend_execution_context::BackendExecutionContext;
        use crate::runtime::backend::interface::{BackendInterface, DelegateHandle};
        use crate::runtime::core::evalue::EValue;
        use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
        use crate::runtime::core::memory_allocator::{
            MemoryAllocator, MemoryAllocatorBase, MemoryAllocatorExt,
        };
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

        crate::runtime::platform::runtime::runtime_init();
        let backend = XnnpackBackend::new();

        // Build the runtime inside a live workspace, mirroring init().
        let workspace = XNNWorkspace::create().unwrap();
        let mut subgraph = xnn_subgraph_t(core::ptr::null_mut());
        let mut rt = xnn_runtime_t(core::ptr::null_mut());
        let dims: [usize; 4] = [1, 4, 4, 4];
        unsafe {
            assert_eq!(xnn_initialize(core::ptr::null()), xnn_status_success);
            assert_eq!(xnn_create_subgraph(3, 0, &mut subgraph), xnn_status_success);

            let mut in0_id: u32 = XNN_INVALID_VALUE_ID;
            let mut in1_id: u32 = XNN_INVALID_VALUE_ID;
            let mut out_id: u32 = XNN_INVALID_VALUE_ID;
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    0,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut in0_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    1,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut in1_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    2,
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    &mut out_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_binary(
                    subgraph,
                    xnn_binary_add,
                    core::ptr::null(),
                    in0_id,
                    in1_id,
                    out_id,
                    0,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_create_runtime_v4(
                    subgraph,
                    xnn_weights_cache_t(core::ptr::null_mut()),
                    workspace.unsafe_get_workspace(),
                    pthreadpool_t(core::ptr::null_mut()),
                    XNN_FLAG_BASIC_PROFILING,
                    &mut rt,
                ),
                xnn_status_success
            );
        }

        // Placement-construct the executor in runtime-allocator memory, exactly
        // as init() does; destroy() destructs it manually without deallocating.
        let mut pool = vec![0u8; 4096];
        let mut allocator = MemoryAllocator::new(pool.len() as u32, pool.as_mut_ptr());
        let executor: *mut XNNExecutor =
            allocator.allocate_instance::<XNNExecutor>(core::mem::align_of::<XNNExecutor>());
        assert!(!executor.is_null());
        unsafe {
            core::ptr::write(executor, XNNExecutor::new(workspace.clone()));
            assert_eq!(
                (*executor).initialize(rt, vec![0, 1], vec![2], vec![], vec![], false),
                Error::Ok
            );
        }
        let handle = executor as *mut DelegateHandle;

        let tf = TensorFactory::<f32>::new();
        let input0 = tf.make(
            vec![1, 4, 4, 4],
            vec![1.0f32; 64],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let input1 = tf.make(
            vec![1, 4, 4, 4],
            vec![2.0f32; 64],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let output = tf.make(
            vec![1, 4, 4, 4],
            vec![0.0f32; 64],
            Vec::new(),
            TensorShapeDynamism::STATIC,
        );
        let mut ev_in0 = EValue::from_tensor(input0);
        let mut ev_in1 = EValue::from_tensor(input1);
        let mut ev_out = EValue::from_tensor(output);
        let mut args: [*mut EValue; 3] = [&mut ev_in0, &mut ev_in1, &mut ev_out];
        let span: Span<*mut EValue> = Span::from_raw_parts(args.as_mut_ptr(), 3);

        let mut execute_context = BackendExecutionContext::new(
            null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
            core::ptr::null(),
        );
        assert_eq!(
            backend.execute(&mut execute_context, handle, span),
            Error::Ok
        );

        // 1 + 2 = 3 across the whole output.
        let result_tensor = unsafe { (*args[2]).to_tensor() };
        for i in 0..64 {
            assert_eq!(
                unsafe { *result_tensor.const_data_ptr::<f32>().add(i) },
                3.0
            );
        }

        // destroy() destructs the executor (and its runtime) in place.
        backend.destroy(handle);

        unsafe {
            assert_eq!(xnn_delete_subgraph(subgraph), xnn_status_success);
        }
    }

    // destroy() with a null handle is an explicit no-op (`if (handle !=
    // nullptr)` in the C++).
    // [spec:et:sem:xnnpack-backend.executorch.backends.xnnpack-backend.destroy-fn/test]
    #[test]
    fn destroy_null_handle_is_noop() {
        use crate::runtime::backend::interface::BackendInterface;
        crate::runtime::platform::runtime::runtime_init();
        let backend = XnnpackBackend::new();
        backend.destroy(core::ptr::null_mut());
    }
}
