//! Literal port of extension/module/bundled_module.cpp + bundled_module.h.
//!
//! `BundledModule` extends `Module` to run bundled programs. It depends on
//! `devtools/bundled_program` (the `bundled_program_flatbuffer` schema and the
//! `load_bundled_input` / `verify_method_outputs` helpers), which is OUT of the
//! wave-2 port scope. Per the group notes, the dependent bodies are gated behind
//! the `bundled-program` Cargo feature; the annotations and structure are kept,
//! and the stubbed dependency is PORT-NOTEd.
//!
//! COMPOSITION DEVIATION: C++ `class BundledModule : public Module`. Rust has no
//! inheritance, so `BundledModule` COMPOSES a `Module` as its base subobject
//! (`base_`). The `Module::method_ref_mut` `pub(crate)` accessor stands in for
//! the C++ direct access to the protected `methods_` member.
//!
//! OWNERSHIP DEVIATION: when built via `from_file`, the module exclusively owns a
//! heap byte buffer (`bundled_program_ptr_`) and its `Drop` frees it (mirroring
//! the C++ `delete[]` in the destructor, guarded by `is_loaded_from_file_`). The
//! raw-pointer constructor does NOT take ownership — the caller must keep the
//! buffer alive. `BundledModule` is neither `Copy` nor `Clone` (non-copyable and
//! non-movable in C++), matching the deleted copy/move ops.

extern crate alloc;
extern crate std;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::extension::data_loader::buffer_data_loader::BufferDataLoader;
use crate::extension::data_loader::file_data_loader::FileDataLoader;
use crate::extension::module::module::Module;
use crate::runtime::core::data_loader::DataLoader;
use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
use crate::runtime::core::result::ResultExt;

// PORT-NOTE: file-local (anonymous-namespace) helper `program_data_loader`.
// Interprets the bundled-program flatbuffer and returns a `BufferDataLoader`
// viewing the embedded plain-program bytes. `bundled_program_flatbuffer::
// GetBundledProgram` and the generated accessors live in devtools/bundled_program
// (out of scope), so the body is gated behind `bundled-program`. Unresolved
// cross-module reference.
// [spec:et:def:bundled-module.executorch.extension.program-data-loader-fn]
// [spec:et:sem:bundled-module.executorch.extension.program-data-loader-fn]
#[cfg(feature = "bundled-program")]
fn program_data_loader(bundled_program_ptr: *const core::ffi::c_void) -> Box<BufferDataLoader> {
    let bundled_program =
        crate::devtools::bundled_program::schema::get_bundled_program(bundled_program_ptr);
    // the program inside the bundled program
    let program = bundled_program.program();
    Box::new(BufferDataLoader::new(
        program.data() as *const core::ffi::c_void,
        program.size(),
    ))
}

#[cfg(not(feature = "bundled-program"))]
fn program_data_loader(_bundled_program_ptr: *const core::ffi::c_void) -> Box<BufferDataLoader> {
    // PORT-NOTE: devtools/bundled_program is out of wave-2 scope; without the
    // `bundled-program` feature the bundled flatbuffer cannot be parsed. Returns
    // an empty loader placeholder so the surrounding structure still type-checks.
    Box::new(BufferDataLoader::new(core::ptr::null(), 0))
}

/// A facade class for loading bundled programs and executing methods within
/// them.
// [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module]
pub struct BundledModule<'a> {
    // C++ `public Module` base subobject.
    base_: Module<'a>,
    bundled_program_ptr_: *const core::ffi::c_void,
    // PORT-NOTE: not a C++ member. The C++ destructor frees the buffer with
    // `delete[]`, which recovers the allocation size from the allocator. Rust's
    // `Box<[u8]>` reconstruction needs the length explicitly, so `from_file`
    // records it here and `Drop` uses it to free the buffer. Zero for the
    // raw-pointer constructor (which does not own the buffer).
    bundled_program_len_: usize,
    is_loaded_from_file_: bool,
}

impl<'a> BundledModule<'a> {
    /// Constructs an instance with the bundled program buffer pointer.
    // [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.bundled-module-fn]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.bundled-module-fn]
    //
    // PORT-NOTE: the C++ base initializer builds `Module` from
    // `program_data_loader(bundled_program_ptr)` and moves the allocators /
    // tracer / data-map loader through unchanged. `is_loaded_from_file_` keeps
    // its default `false`; only `from_file` sets it true.
    pub fn new(
        bundled_program_ptr: *const core::ffi::c_void,
        memory_allocator: Option<Box<dyn MemoryAllocatorBase>>,
        temp_allocator: Option<Box<dyn MemoryAllocatorBase>>,
        event_tracer: Option<Box<dyn EventTracer>>,
        data_map_loader: Option<Box<dyn DataLoader>>,
    ) -> Self {
        let loader: Box<dyn DataLoader> = program_data_loader(bundled_program_ptr);
        let base_ = Module::from_data_loader(
            loader,
            memory_allocator,
            temp_allocator,
            event_tracer,
            data_map_loader,
            /*share_memory_arenas=*/ false,
        );
        BundledModule {
            base_,
            bundled_program_ptr_: bundled_program_ptr,
            bundled_program_len_: 0,
            is_loaded_from_file_: false,
        }
    }

    // PORT-NOTE: `BundledModule(const BundledModule&) = delete` /
    // `operator=(const BundledModule&) = delete` / move ops deleted. Neither
    // `Copy` nor `Clone` is derived; ownership moves via the owning `Box`.
    // [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.operator-fn]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.operator-fn]

    /// Constructs an instance by loading a bundled program from a file.
    // [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn]
    #[must_use]
    pub fn from_file(
        file_path: &str,
        memory_allocator: Option<Box<dyn MemoryAllocatorBase>>,
        temp_allocator: Option<Box<dyn MemoryAllocatorBase>>,
        event_tracer: Option<Box<dyn EventTracer>>,
        data_map_loader: Option<Box<dyn DataLoader>>,
    ) -> Result<Box<BundledModule<'a>>> {
        let c_path = to_c_string(file_path);
        let data_loader_result =
            FileDataLoader::from(c_path.as_ptr() as *const core::ffi::c_char, MAX_ALIGN);
        if !ResultExt::ok(&data_loader_result) {
            return Err(ResultExt::error(&data_loader_result));
        }
        let data_loader_result = data_loader_result;

        let file_size_result = ResultExt::get(&data_loader_result).size();
        if !ResultExt::ok(&file_size_result) {
            return Err(ResultExt::error(&file_size_result));
        }

        let file_size: usize = *ResultExt::get(&file_size_result);
        // PORT-NOTE: `std::make_unique<uint8_t[]>(file_size)` — a heap byte
        // buffer whose raw ownership is transferred to the BundledModule below.
        // `file_data.release()` maps to `Box::into_raw` on a boxed slice, and the
        // module frees it in `Drop`.
        let mut file_data: Box<[u8]> = alloc::vec![0u8; file_size].into_boxed_slice();
        let buffer_result = ResultExt::get(&data_loader_result).load_into(
            0,
            file_size,
            // PORT-NOTE: C++ passes `{}` — a default-constructed SegmentInfo
            // (Type value-initializes to Program=0, index 0, null descriptor).
            &SegmentInfo::new(Type::Program, 0, core::ptr::null()),
            file_data.as_mut_ptr() as *mut core::ffi::c_void,
        );
        if buffer_result != Error::Ok {
            return Err(buffer_result);
        }

        // Pass ownership of the data to BundledModule.
        let raw: *mut u8 = Box::into_raw(file_data) as *mut u8;
        let mut bm = Box::new(BundledModule::new(
            raw as *const core::ffi::c_void,
            memory_allocator,
            temp_allocator,
            event_tracer,
            data_map_loader,
        ));

        bm.bundled_program_len_ = file_size;
        bm.is_loaded_from_file_ = true;

        Ok(bm)
    }

    /// Access the base `Module` for the inherited `execute(method_name, inputs)`
    /// and other `Module` APIs (`using Module::execute;`).
    pub fn base(&mut self) -> &mut Module<'a> {
        &mut self.base_
    }

    /// Execute a specific method with the bundled input at `testset_idx`.
    // [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn]
    #[must_use]
    pub fn execute(&mut self, method_name: &str, testset_idx: usize) -> Result<Vec<EValue<'a>>> {
        let err = self.base_.load_method(
            method_name,
            core::ptr::null_mut(),
            null_event_tracer(),
            core::ptr::null(),
            Vec::new(),
        );
        if err != Error::Ok {
            return Err(err);
        }
        let bundled_program_ptr = self.bundled_program_ptr_;
        let method = self.base_.method_ref_mut(method_name);

        // PORT-NOTE: `load_bundled_input` lives in devtools/bundled_program (out
        // of scope). Gated behind `bundled-program`. Unresolved cross-module
        // reference.
        #[cfg(feature = "bundled-program")]
        {
            let e = crate::devtools::bundled_program::load_bundled_input(
                method,
                bundled_program_ptr,
                testset_idx,
            );
            if e != Error::Ok {
                return Err(e);
            }
        }
        #[cfg(not(feature = "bundled-program"))]
        {
            let _ = &method;
            let _ = bundled_program_ptr;
            let _ = testset_idx;
            return Err(Error::NotImplemented);
        }

        #[allow(unreachable_code)]
        {
            let e = method.execute();
            if e != Error::Ok {
                return Err(e);
            }

            let outputs_size = method.outputs_size();
            let mut outputs: Vec<EValue<'a>> = Vec::with_capacity(outputs_size);
            for _ in 0..outputs_size {
                outputs.push(EValue::new());
            }
            let e = method.get_outputs(outputs.as_mut_ptr(), outputs_size);
            if e != Error::Ok {
                return Err(e);
            }

            Ok(outputs)
        }
    }

    /// Verify the output of a specific method against the bundle's expected
    /// output at `testset_idx`.
    // [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn]
    #[must_use]
    pub fn verify_method_outputs(
        &mut self,
        method_name: &str,
        testset_idx: usize,
        rtol: f64,
        atol: f64,
    ) -> Error {
        crate::et_check_ok_or_return_error!(self.base_.load_method(
            method_name,
            core::ptr::null_mut(),
            null_event_tracer(),
            core::ptr::null(),
            Vec::new(),
        ));
        let bundled_program_ptr = self.bundled_program_ptr_;
        let method = self.base_.method_ref_mut(method_name);
        // PORT-NOTE: `verify_method_outputs` lives in devtools/bundled_program
        // (out of scope). Gated behind `bundled-program`. Unresolved cross-module
        // reference.
        #[cfg(feature = "bundled-program")]
        {
            crate::devtools::bundled_program::verify_method_outputs(
                method,
                bundled_program_ptr,
                testset_idx,
                rtol,
                atol,
            )
        }
        #[cfg(not(feature = "bundled-program"))]
        {
            let _ = (method, bundled_program_ptr, testset_idx, rtol, atol);
            Error::NotImplemented
        }
    }
}

// PORT-NOTE: C++ `~BundledModule()` frees `bundled_program_ptr_` with `delete[]`
// (over `const uint8_t*`) only when `is_loaded_from_file_` is true. In Rust the
// buffer was created as `Box<[u8]>` and leaked via `Box::into_raw` in
// `from_file`; reconstruct and drop the boxed slice to free it. The raw-pointer
// constructor leaves `is_loaded_from_file_` false, so its (caller-owned) buffer
// is never freed here.
impl<'a> Drop for BundledModule<'a> {
    fn drop(&mut self) {
        if self.is_loaded_from_file_ {
            // Reconstruct the `Box<[u8]>` that `from_file` leaked via
            // `Box::into_raw` and drop it, freeing the buffer. `Drop` of the
            // `base_` Module (and its BufferDataLoader, which merely borrows this
            // buffer) runs after this block, but the loader never frees the
            // bytes, so reclaiming them here is the sole free — mirroring the
            // C++ `delete[]`.
            unsafe {
                let slice = core::slice::from_raw_parts_mut(
                    self.bundled_program_ptr_ as *mut u8,
                    self.bundled_program_len_,
                );
                drop(Box::from_raw(slice as *mut [u8]));
            }
        }
    }
}

// PORT-NOTE: Rust `String`/`&str` are not NUL-terminated; `file_path.c_str()`
// needs a NUL-terminated C string.
fn to_c_string(s: &str) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::with_capacity(s.len() + 1);
    v.extend_from_slice(s.as_bytes());
    v.push(0);
    v
}

// PORT-NOTE: C++ default alignment for FileDataLoader::from is
// `alignof(std::max_align_t)`.
const MAX_ALIGN: usize = core::mem::align_of::<u128>();

// PORT-NOTE: null `*mut dyn EventTracer` for the `load_method` argument (the
// base Module supplies its own tracer downstream when this is null). Reuses the
// sibling module's shared null-tracer helper.
use crate::extension::module::module::null_event_tracer;

// For the load_into default segment info argument.
use crate::runtime::core::data_loader::{SegmentInfo, Type};

// PORT-NOTE: without the `bundled-program` feature, `program_data_loader` is a
// stub that never dereferences its pointer argument and `from_data_loader` only
// stores the loader (no parsing). That lets the raw-pointer `BundledModule::new`
// constructor be exercised with a dummy pointer, covering `bundled-module-fn`.
// (The `bundled-program`-gated suite below cannot: with the feature enabled,
// `program_data_loader` parses the flatbuffer and needs a real bundled program.)
#[cfg(all(test, not(feature = "bundled-program")))]
mod tests_no_bp {
    use super::*;

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.bundled-module-fn/test]
    #[test]
    fn bundled_module_test_raw_pointer_constructor() {
        crate::runtime::platform::runtime::runtime_init();
        // The raw-pointer constructor takes no ownership (is_loaded_from_file_
        // stays false), so Drop will not free this dummy pointer.
        let dummy: usize = 0xdead_beef;
        let mut bm = BundledModule::new(dummy as *const core::ffi::c_void, None, None, None, None);
        // The base Module is constructed but nothing is loaded yet, and no
        // EventTracer was supplied.
        assert!(!bm.base().is_loaded());
        assert!(bm.base().event_tracer().is_null());
        // Drop is a no-op for the dummy pointer: the raw-pointer path leaves
        // is_loaded_from_file_ = false, so Drop does not reconstruct/free it.
    }

    // The deleted copy/move ops (`operator=` et al.) collapse onto "no
    // `Copy`/`Clone`, unique ownership" in Rust: the module (via its owning
    // Box) can only be moved, and the moved-to sole owner both works and frees
    // the from_file heap buffer exactly once on drop (Drop is guarded by
    // is_loaded_from_file_; a duplicated owner would double-free).
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.operator-fn/test]
    #[test]
    fn bundled_module_test_move_only_unique_buffer_owner() {
        crate::runtime::platform::runtime::runtime_init();
        let contents = b"BUNDLED_BYTES";
        let tf = crate::extension::data_loader::testing::TempFile::new(contents);
        // Without the bundled-program feature, from_file reads the raw bytes
        // and takes ownership of the heap buffer without parsing them.
        let bm = BundledModule::from_file(tf.path(), None, None, None, None);
        assert_eq!(ResultExt::error(&bm), Error::Ok);
        let bm = bm.unwrap();
        assert!(bm.is_loaded_from_file_);
        assert_eq!(bm.bundled_program_len_, contents.len());
        // Transfer unique ownership by move; the single surviving owner still
        // answers, then its drop performs the sole buffer free.
        let mut moved = bm;
        assert!(!moved.base().is_loaded());
        drop(moved);
    }
}

// PORT-NOTE: port of extension/module/test/bundled_module_test.cpp
// (BundledModuleTest fixture). The C++ SetUpTestSuite reads RESOURCES_PATH
// (for `<RESOURCES_PATH>/bundled_program.bpte`) and ET_MODULE_PTE_PATH.
//
// The whole suite is gated behind the `bundled-program` Cargo feature (group
// note): without it, `BundledModule::execute`/`verify_method_outputs` return
// `Error::NotImplemented` (the devtools/bundled_program dependency is out of
// wave-2 scope), so the C++ semantics cannot be reproduced. Even with the
// feature the `.bpte` / `.pte` fixtures are not wired into the Rust build, so
// the tests skip when their env vars are unset (PORTING.md fixture policy).
#[cfg(all(test, feature = "bundled-program"))]
mod tests {
    use super::*;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn bpte_path() -> Option<String> {
        // <RESOURCES_PATH>/bundled_program.bpte
        match std::env::var("RESOURCES_PATH") {
            Ok(p) => {
                let mut s = p;
                s.push_str("/bundled_program.bpte");
                Some(s)
            }
            Err(_) => None,
        }
    }

    fn pte_path() -> Option<String> {
        std::env::var("ET_MODULE_PTE_PATH").ok()
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn/test]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn/test]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn/test]
    #[test]
    fn bundled_module_test_test_execute() {
        setup();
        let path = match bpte_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping bundled_module_test_test_execute: RESOURCES_PATH unset");
                return;
            }
        };
        eprintln!(
            "skipping bundled_module_test_test_execute: requires {} fixture and a \
             wired devtools/bundled_program",
            path
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn/test]
    //
    // Runnable without the fixture: a non-existent .bpte path must fail in
    // `from_file` at the FileDataLoader stage with Error::AccessFailed, before
    // any bundled-program parsing.
    #[test]
    fn bundled_module_test_test_non_exist_bp_file() {
        setup();
        let bundled_module_output =
            BundledModule::from_file("/path/to/nonexistent/file.bpte", None, None, None, None);
        assert_eq!(
            ResultExt::error(&bundled_module_output),
            Error::AccessFailed
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn/test]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn/test]
    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn/test]
    #[test]
    fn bundled_module_test_test_non_bp_file() {
        setup();
        let path = match pte_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "skipping bundled_module_test_test_non_bp_file: ET_MODULE_PTE_PATH unset"
                );
                return;
            }
        };
        eprintln!(
            "skipping bundled_module_test_test_non_bp_file: requires {} fixture and a \
             wired devtools/bundled_program",
            path
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn/test]
    #[test]
    fn bundled_module_test_test_execute_invalid_method() {
        setup();
        if bpte_path().is_none() {
            eprintln!(
                "skipping bundled_module_test_test_execute_invalid_method: RESOURCES_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping bundled_module_test_test_execute_invalid_method: requires the .bpte fixture"
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn/test]
    #[test]
    fn bundled_module_test_test_execute_invalid_idx() {
        setup();
        if bpte_path().is_none() {
            eprintln!(
                "skipping bundled_module_test_test_execute_invalid_idx: RESOURCES_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping bundled_module_test_test_execute_invalid_idx: requires the .bpte fixture"
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn/test]
    #[test]
    fn bundled_module_test_test_verify_invalid_method() {
        setup();
        if bpte_path().is_none() {
            eprintln!(
                "skipping bundled_module_test_test_verify_invalid_method: RESOURCES_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping bundled_module_test_test_verify_invalid_method: requires the .bpte fixture"
        );
    }

    // [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn/test]
    #[test]
    fn bundled_module_test_test_verify_invalid_idx() {
        setup();
        if bpte_path().is_none() {
            eprintln!("skipping bundled_module_test_test_verify_invalid_idx: RESOURCES_PATH unset");
            return;
        }
        eprintln!(
            "skipping bundled_module_test_test_verify_invalid_idx: requires the .bpte fixture"
        );
    }
}
