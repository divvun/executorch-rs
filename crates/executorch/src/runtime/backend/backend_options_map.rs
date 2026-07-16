//! Literal port of runtime/backend/backend_options_map.h.

use crate::runtime::backend::options::BackendOption;
use crate::runtime::core::error::Error;
use crate::runtime::core::span::Span;

const K_MAX_BACKENDS: usize = 8;
const K_MAX_BACKEND_ID_LENGTH: usize = 64;

/// Non-owning view of a single (backend_id, options) entry, returned by
/// entry_at(). The pointer / span are valid until the map is mutated or
/// destroyed.
// [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry-view]
#[derive(Clone, Copy)]
pub struct EntryView {
    pub backend_id: *const core::ffi::c_char,
    pub options: Span<BackendOption>,
}

impl EntryView {
    pub fn new() -> Self {
        EntryView {
            backend_id: core::ptr::null(),
            options: Span::new(),
        }
    }
}

impl Default for EntryView {
    fn default() -> Self {
        EntryView::new()
    }
}

// [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry]
#[derive(Clone, Copy)]
struct Entry {
    backend_id: [core::ffi::c_char; K_MAX_BACKEND_ID_LENGTH],
    options: Span<BackendOption>,
}

impl Entry {
    fn new() -> Self {
        Entry {
            backend_id: [0; K_MAX_BACKEND_ID_LENGTH],
            options: Span::new(),
        }
    }
}

/// Maps backend IDs to their load-time options.
///
/// This class is used to provide per-delegate configuration at Module::load()
/// time. Users can set options for multiple backends, and the runtime will
/// route the appropriate options to each backend during initialization.
///
/// Note: This class does NOT take ownership of the option spans. The caller
/// must ensure that the BackendOptions objects outlive the LoadBackendOptionsMap
/// and any loaded models that use it.
// [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map]
pub struct LoadBackendOptionsMap {
    entries_: [Entry; K_MAX_BACKENDS],
    size_: usize,
}

impl LoadBackendOptionsMap {
    /// Default constructor - creates an empty map.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.load-backend-options-map-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.load-backend-options-map-fn]
    pub fn new() -> Self {
        let mut map = LoadBackendOptionsMap {
            entries_: [Entry::new(); K_MAX_BACKENDS],
            size_: 0,
        };
        for i in 0..K_MAX_BACKENDS {
            map.entries_[i].backend_id[0] = 0;
        }
        map
    }

    /// Sets options for a specific backend.
    ///
    /// If options for the given backend_id already exist, they will be replaced.
    ///
    /// `backend_id` The backend identifier (e.g., "CoreMLBackend",
    /// "XNNPACKBackend"). Must not be null or empty.
    /// `options` Span of BackendOption to associate with this backend. The
    /// span's underlying data must outlive this map and any models loaded with
    /// it.
    /// @return Error::Ok on success. Error::InvalidArgument if backend_id is
    /// null/empty or max backends exceeded.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn]
    pub fn set_options(
        &mut self,
        backend_id: *const core::ffi::c_char,
        options: Span<BackendOption>,
    ) -> Error {
        if backend_id.is_null() || unsafe { *backend_id } == 0 {
            return Error::InvalidArgument;
        }

        self.set_options_impl(backend_id, options)
    }

    // PORT-NOTE: the C++ templated convenience overload
    // `set_options(Builder& builder)` — forwarding `builder.backend_id()` and
    // `builder.view()` to `set_options_impl` and bypassing the null/empty
    // check — is not annotated with a spec rule and is omitted from this
    // literal port; call `set_options_impl` (via `set_options`) directly.

    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]
    fn set_options_impl(
        &mut self,
        backend_id: *const core::ffi::c_char,
        options: Span<BackendOption>,
    ) -> Error {
        // Check if backend already exists and update it
        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.entries_[i].backend_id.as_ptr(), backend_id) } == 0 {
                self.entries_[i].options = options;
                return Error::Ok;
            }
        }

        // Add new entry if space available
        if self.size_ >= K_MAX_BACKENDS {
            return Error::InvalidArgument;
        }

        let id_len: usize = unsafe { libc::strlen(backend_id) };
        if id_len >= K_MAX_BACKEND_ID_LENGTH {
            return Error::InvalidArgument;
        }
        unsafe {
            libc::memcpy(
                self.entries_[self.size_].backend_id.as_mut_ptr() as *mut core::ffi::c_void,
                backend_id as *const core::ffi::c_void,
                id_len,
            );
        }
        self.entries_[self.size_].backend_id[id_len] = 0;
        self.entries_[self.size_].options = options;
        self.size_ += 1;

        Error::Ok
    }

    /// Gets options for a specific backend.
    ///
    /// `backend_id` The backend identifier to look up.
    /// @return Span of options for this backend, or an empty span if the backend
    ///         has no options configured or backend_id is null.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn]
    pub fn get_options(&self, backend_id: *const core::ffi::c_char) -> Span<BackendOption> {
        if backend_id.is_null() {
            return Span::from_raw_parts(core::ptr::null_mut(), 0_usize);
        }

        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.entries_[i].backend_id.as_ptr(), backend_id) } == 0 {
                return Span::from_raw_parts(
                    self.entries_[i].options.data(),
                    self.entries_[i].options.size(),
                );
            }
        }

        Span::from_raw_parts(core::ptr::null_mut(), 0_usize)
    }

    /// Checks if options have been configured for a specific backend.
    ///
    /// `backend_id` The backend identifier to check.
    /// @return true if options are set for this backend, false otherwise.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn]
    pub fn has_options(&self, backend_id: *const core::ffi::c_char) -> bool {
        if backend_id.is_null() {
            return false;
        }

        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.entries_[i].backend_id.as_ptr(), backend_id) } == 0 {
                return true;
            }
        }

        false
    }

    /// Returns the number of backends with configured options.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.size-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn]
    pub fn size(&self) -> usize {
        self.size_
    }

    /// Returns the (backend_id, options) entry at the given index for
    /// enumeration over the map's contents.
    ///
    /// `index` The entry index. Must be < size(); behavior is undefined
    ///     otherwise. Use this together with size() to walk every entry.
    /// @return EntryView referencing the entry's backend_id and options. The
    ///     view is valid until the next mutation of, or destruction of, this
    ///     map.
    // [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry-at-fn]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.entry-at-fn]
    pub fn entry_at(&self, index: usize) -> EntryView {
        // PORT-NOTE: C++ `ET_DCHECK_MSG(index < size_, ...)` — debug-only,
        // elided in release, undefined index out of range. `debug_assert!` is
        // the equivalent.
        debug_assert!(
            index < self.size_,
            "entry_at index {} out of bounds (size={})",
            index,
            self.size_
        );
        EntryView {
            backend_id: self.entries_[index].backend_id.as_ptr(),
            options: Span::from_raw_parts(
                self.entries_[index].options.data(),
                self.entries_[index].options.size(),
            ),
        }
    }
}

impl Default for LoadBackendOptionsMap {
    fn default() -> Self {
        LoadBackendOptionsMap::new()
    }
}

// Literal port of runtime/backend/test/backend_options_map_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::options::{
        BackendOptions, K_MAX_OPTION_VALUE_LENGTH, OptionValue,
    };
    use crate::runtime::platform::runtime::runtime_init;

    // C++ passes string-literal keys/ids as `const char (&)[N]` (N includes the
    // trailing NUL). This helper turns a NUL-terminated byte literal into the
    // equivalent `[c_char; N]` array.
    const fn key<const N: usize>(bytes: &[u8; N]) -> [core::ffi::c_char; N] {
        let mut out = [0 as core::ffi::c_char; N];
        let mut i = 0;
        while i < N {
            out[i] = bytes[i] as core::ffi::c_char;
            i += 1;
        }
        out
    }

    fn setup() {
        runtime_init();
    }

    // Test default constructor creates empty map
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.load-backend-options-map-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    #[test]
    fn load_backend_options_map_test_default_constructor_creates_empty_map() {
        setup();
        let map = LoadBackendOptionsMap::new();
        assert_eq!(map.size(), 0);
        assert!(!map.has_options(c"CoreMLBackend".as_ptr()));
    }

    // Test set_options and get_options round trip
    // set_options forwards to set_options_impl, which stores the entry that
    // get_options reads back — so this exercises set_options_impl's add path.
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    #[test]
    fn load_backend_options_map_test_set_and_get_options_round_trip() {
        setup();
        let mut coreml_opts: BackendOptions<4> = BackendOptions::new();
        coreml_opts.set_option_str(&key(b"compute_unit\0"), c"cpu_and_gpu".as_ptr());
        coreml_opts.set_option_int(&key(b"num_threads\0"), 4);

        let mut map = LoadBackendOptionsMap::new();
        assert_eq!(
            map.set_options(c"CoreMLBackend".as_ptr(), coreml_opts.view()),
            Error::Ok
        );

        let retrieved = map.get_options(c"CoreMLBackend".as_ptr());
        assert_eq!(retrieved.size(), 2);

        // Verify we can read the options back
        let mut compute_unit: *const core::ffi::c_char = core::ptr::null();
        let mut num_threads = 0;
        for i in 0..retrieved.size() {
            let opt = unsafe { retrieved.index(i) };
            if unsafe { libc::strcmp(opt.key.as_ptr(), c"compute_unit".as_ptr()) } == 0 {
                if let OptionValue::CharArray(arr) = &opt.value {
                    compute_unit = arr.as_ptr();
                }
            } else if unsafe { libc::strcmp(opt.key.as_ptr(), c"num_threads".as_ptr()) } == 0 {
                if let OptionValue::Int(val) = &opt.value {
                    num_threads = *val;
                }
            }
        }
        assert_eq!(
            unsafe { libc::strcmp(compute_unit, c"cpu_and_gpu".as_ptr()) },
            0
        );
        assert_eq!(num_threads, 4);
    }

    // Test has_options returns correct values
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    #[test]
    fn load_backend_options_map_test_has_options_returns_correct_values() {
        setup();
        let mut opts: BackendOptions<2> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        let mut map = LoadBackendOptionsMap::new();
        assert!(!map.has_options(c"CoreMLBackend".as_ptr()));

        map.set_options(c"CoreMLBackend".as_ptr(), opts.view());
        assert!(map.has_options(c"CoreMLBackend".as_ptr()));
        assert!(!map.has_options(c"XNNPACKBackend".as_ptr()));
    }

    // Test get_options returns empty span for unknown backend
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    #[test]
    fn load_backend_options_map_test_get_options_returns_empty_for_unknown_backend() {
        setup();
        let map = LoadBackendOptionsMap::new();
        let opts = map.get_options(c"UnknownBackend".as_ptr());
        assert_eq!(opts.size(), 0);
    }

    // Test multiple backends
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    #[test]
    fn load_backend_options_map_test_multiple_backends() {
        setup();
        let mut coreml_opts: BackendOptions<2> = BackendOptions::new();
        coreml_opts.set_option_str(&key(b"compute_unit\0"), c"cpu_and_ne".as_ptr());

        let mut xnnpack_opts: BackendOptions<2> = BackendOptions::new();
        xnnpack_opts.set_option_int(&key(b"num_threads\0"), 8);

        let mut map = LoadBackendOptionsMap::new();
        assert_eq!(
            map.set_options(c"CoreMLBackend".as_ptr(), coreml_opts.view()),
            Error::Ok
        );
        assert_eq!(
            map.set_options(c"XNNPACKBackend".as_ptr(), xnnpack_opts.view()),
            Error::Ok
        );

        assert_eq!(map.size(), 2);
        assert!(map.has_options(c"CoreMLBackend".as_ptr()));
        assert!(map.has_options(c"XNNPACKBackend".as_ptr()));

        let coreml_retrieved = map.get_options(c"CoreMLBackend".as_ptr());
        let xnnpack_retrieved = map.get_options(c"XNNPACKBackend".as_ptr());

        assert_eq!(coreml_retrieved.size(), 1);
        assert_eq!(xnnpack_retrieved.size(), 1);
    }

    // Test updating existing backend options
    // Exercises set_options_impl's update-in-place branch (matching backend_id
    // replaces the span without growing size_).
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    #[test]
    fn load_backend_options_map_test_update_existing_backend_options() {
        setup();
        let mut opts_v1: BackendOptions<2> = BackendOptions::new();
        opts_v1.set_option_str(&key(b"compute_unit\0"), c"cpu_only".as_ptr());

        let mut opts_v2: BackendOptions<2> = BackendOptions::new();
        opts_v2.set_option_str(&key(b"compute_unit\0"), c"cpu_and_gpu".as_ptr());
        opts_v2.set_option_bool(&key(b"enable_profiling\0"), true);

        let mut map = LoadBackendOptionsMap::new();
        map.set_options(c"CoreMLBackend".as_ptr(), opts_v1.view());
        assert_eq!(map.get_options(c"CoreMLBackend".as_ptr()).size(), 1);

        // Update with new options
        map.set_options(c"CoreMLBackend".as_ptr(), opts_v2.view());
        assert_eq!(map.size(), 1); // Still only one backend
        assert_eq!(map.get_options(c"CoreMLBackend".as_ptr()).size(), 2); // But now 2 options
    }

    // Test max backends limit
    // Exercises set_options_impl's capacity-full branch (size_ >= kMaxBackends
    // returns InvalidArgument).
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    #[test]
    fn load_backend_options_map_test_max_backends_limit() {
        setup();
        let mut map = LoadBackendOptionsMap::new();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        // Add 8 backends (the limit)
        let backend_ids: [&core::ffi::CStr; 8] = [
            c"Backend1",
            c"Backend2",
            c"Backend3",
            c"Backend4",
            c"Backend5",
            c"Backend6",
            c"Backend7",
            c"Backend8",
        ];

        for id in backend_ids.iter().take(8) {
            assert_eq!(map.set_options(id.as_ptr(), opts.view()), Error::Ok);
        }

        assert_eq!(map.size(), 8);

        // Adding a 9th backend should fail
        assert_eq!(
            map.set_options(c"Backend9".as_ptr(), opts.view()),
            Error::InvalidArgument
        );
        assert_eq!(map.size(), 8);
    }

    // Test null backend_id handling
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    #[test]
    fn load_backend_options_map_test_null_backend_id_handling() {
        setup();
        let mut map = LoadBackendOptionsMap::new();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        // set_options with null should fail
        assert_eq!(
            map.set_options(core::ptr::null(), opts.view()),
            Error::InvalidArgument
        );

        // get_options with null should return empty span
        let result = map.get_options(core::ptr::null());
        assert_eq!(result.size(), 0);

        // has_options with null should return false
        assert!(!map.has_options(core::ptr::null()));
    }

    // Test empty backend_id handling
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    #[test]
    fn load_backend_options_map_test_empty_backend_id_handling() {
        setup();
        let mut map = LoadBackendOptionsMap::new();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        // set_options with empty string should fail
        assert_eq!(
            map.set_options(c"".as_ptr(), opts.view()),
            Error::InvalidArgument
        );
    }

    // Test empty options span
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    #[test]
    fn load_backend_options_map_test_empty_options_span() {
        setup();
        let mut map = LoadBackendOptionsMap::new();
        let mut empty_opts: BackendOptions<1> = BackendOptions::new();

        // Should be able to set empty options
        assert_eq!(
            map.set_options(c"CoreMLBackend".as_ptr(), empty_opts.view()),
            Error::Ok
        );
        assert!(map.has_options(c"CoreMLBackend".as_ptr()));

        let retrieved = map.get_options(c"CoreMLBackend".as_ptr());
        assert_eq!(retrieved.size(), 0);
    }

    // Test long backend_id is rejected
    // Exercises set_options_impl's id-length guard (id_len >= kMaxBackendIdLength
    // returns InvalidArgument) and its accepting boundary at 63 chars.
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    #[test]
    fn load_backend_options_map_test_long_backend_id_rejected() {
        setup();
        let mut map = LoadBackendOptionsMap::new();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        // Create a backend ID that is exactly at the limit (64 chars including null)
        // This should fail because id_len >= kMaxBackendIdLength
        let mut long_id = [b'A' as core::ffi::c_char; 65];
        long_id[64] = 0;

        assert_eq!(
            map.set_options(long_id.as_ptr(), opts.view()),
            Error::InvalidArgument
        );
        assert_eq!(map.size(), 0);

        // A backend ID of 63 chars (plus null = 64 total) should succeed
        let mut max_valid_id = [b'B' as core::ffi::c_char; 64];
        max_valid_id[63] = 0;

        assert_eq!(
            map.set_options(max_valid_id.as_ptr(), opts.view()),
            Error::Ok
        );
        assert_eq!(map.size(), 1);
        assert!(map.has_options(max_valid_id.as_ptr()));
    }

    // Example backend options builder demonstrating the recommended pattern.
    //
    // PORT-NOTE: mirrors the C++ `ExampleBackendOptions` test helper. The
    // fluent setters return `&mut self` for chaining. The C++ tests invoke the
    // templated `map.set_options(builder)` overload, which the module omits (see
    // its PORT-NOTE); the ported tests instead call `set_options(backend_id(),
    // view())` — exactly what that overload forwards to.
    enum Precision {
        Float32,
        Float16,
        Int8,
    }

    struct ExampleBackendOptions {
        options_: BackendOptions<8>,
    }

    impl ExampleBackendOptions {
        fn new() -> Self {
            ExampleBackendOptions {
                options_: BackendOptions::new(),
            }
        }

        fn set_precision(&mut self, p: Precision) -> &mut Self {
            let value: &core::ffi::CStr = match p {
                Precision::Float32 => c"float32",
                Precision::Float16 => c"float16",
                Precision::Int8 => c"int8",
            };
            self.options_
                .set_option_str(&key(b"precision\0"), value.as_ptr());
            self
        }

        fn set_num_threads(&mut self, num_threads: i32) -> &mut Self {
            self.options_
                .set_option_int(&key(b"num_threads\0"), num_threads);
            self
        }

        fn set_enable_optimization(&mut self, enable: bool) -> &mut Self {
            self.options_
                .set_option_bool(&key(b"enable_optimization\0"), enable);
            self
        }

        fn backend_id() -> *const core::ffi::c_char {
            c"ExampleBackend".as_ptr()
        }

        fn view(&mut self) -> Span<BackendOption> {
            self.options_.view()
        }
    }

    // Test template set_options with builder
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn/test]
    #[test]
    fn load_backend_options_map_test_set_options_with_builder() {
        setup();
        let mut map = LoadBackendOptionsMap::new();

        // Example of fluent builder API usage
        let mut builder = ExampleBackendOptions::new();
        builder
            .set_precision(Precision::Float16)
            .set_num_threads(4)
            .set_enable_optimization(true);

        assert_eq!(
            map.set_options(ExampleBackendOptions::backend_id(), builder.view()),
            Error::Ok
        );
        assert_eq!(map.size(), 1);
        assert!(map.has_options(c"ExampleBackend".as_ptr()));

        let retrieved = map.get_options(c"ExampleBackend".as_ptr());
        assert_eq!(retrieved.size(), 3);

        // Verify we can read the options back
        let mut precision_value: *const core::ffi::c_char = core::ptr::null();
        let mut num_threads_value = 0;
        let mut enable_optimization_value = false;
        for i in 0..retrieved.size() {
            let opt = unsafe { retrieved.index(i) };
            if unsafe { libc::strcmp(opt.key.as_ptr(), c"precision".as_ptr()) } == 0 {
                if let OptionValue::CharArray(arr) = &opt.value {
                    precision_value = arr.as_ptr();
                }
            } else if unsafe { libc::strcmp(opt.key.as_ptr(), c"num_threads".as_ptr()) } == 0 {
                if let OptionValue::Int(val) = &opt.value {
                    num_threads_value = *val;
                }
            } else if unsafe { libc::strcmp(opt.key.as_ptr(), c"enable_optimization".as_ptr()) }
                == 0
            {
                if let OptionValue::Bool(val) = &opt.value {
                    enable_optimization_value = *val;
                }
            }
        }
        assert_eq!(
            unsafe { libc::strcmp(precision_value, c"float16".as_ptr()) },
            0
        );
        assert_eq!(num_threads_value, 4);
        assert!(enable_optimization_value);
        let _ = K_MAX_OPTION_VALUE_LENGTH;
    }

    // Test template set_options updates existing backend
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    #[test]
    fn load_backend_options_map_test_set_options_with_builder_updates_existing() {
        setup();
        let mut map = LoadBackendOptionsMap::new();

        // First set options via builder
        let mut builder1 = ExampleBackendOptions::new();
        builder1.set_num_threads(4);
        assert_eq!(
            map.set_options(ExampleBackendOptions::backend_id(), builder1.view()),
            Error::Ok
        );
        assert_eq!(map.size(), 1);

        // Verify initial value
        let retrieved1 = map.get_options(c"ExampleBackend".as_ptr());
        assert_eq!(retrieved1.size(), 1);
        let mut num_threads1 = 0;
        if let OptionValue::Int(val) = &unsafe { retrieved1.index(0) }.value {
            num_threads1 = *val;
        }
        assert_eq!(num_threads1, 4);

        // Update via builder API with different value
        let mut builder2 = ExampleBackendOptions::new();
        builder2.set_num_threads(8);
        assert_eq!(
            map.set_options(ExampleBackendOptions::backend_id(), builder2.view()),
            Error::Ok
        );
        assert_eq!(map.size(), 1); // Still only one backend

        // Verify value was updated
        let retrieved2 = map.get_options(c"ExampleBackend".as_ptr());
        assert_eq!(retrieved2.size(), 1);
        let mut num_threads2 = 0;
        if let OptionValue::Int(val) = &unsafe { retrieved2.index(0) }.value {
            num_threads2 = *val;
        }
        assert_eq!(num_threads2, 8); // Should be updated value
    }

    // Test entry_at returns each (backend_id, options) pair in insertion order
    // and the spans reference the same data the corresponding get_options
    // calls return.
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.entry-at-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn/test]
    // [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn/test]
    #[test]
    fn load_backend_options_map_test_entry_at_enumerates_all_entries() {
        setup();
        let mut map = LoadBackendOptionsMap::new();

        let mut opts1: BackendOptions<2> = BackendOptions::new();
        opts1.set_option_int(&key(b"k1\0"), 1);
        assert_eq!(
            map.set_options(c"BackendA".as_ptr(), opts1.view()),
            Error::Ok
        );

        let mut opts2: BackendOptions<2> = BackendOptions::new();
        opts2.set_option_bool(&key(b"k2\0"), true);
        opts2.set_option_str(&key(b"k3\0"), c"v".as_ptr());
        assert_eq!(
            map.set_options(c"BackendB".as_ptr(), opts2.view()),
            Error::Ok
        );

        assert_eq!(map.size(), 2);

        let e0 = map.entry_at(0);
        assert_eq!(
            unsafe { libc::strcmp(e0.backend_id, c"BackendA".as_ptr()) },
            0
        );
        assert_eq!(e0.options.size(), 1);
        // Spans returned by entry_at point at the same storage as get_options.
        assert_eq!(
            e0.options.data(),
            map.get_options(c"BackendA".as_ptr()).data()
        );

        let e1 = map.entry_at(1);
        assert_eq!(
            unsafe { libc::strcmp(e1.backend_id, c"BackendB".as_ptr()) },
            0
        );
        assert_eq!(e1.options.size(), 2);
        assert_eq!(
            e1.options.data(),
            map.get_options(c"BackendB".as_ptr()).data()
        );
    }
}
