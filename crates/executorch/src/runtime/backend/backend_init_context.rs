//! Literal port of runtime/backend/backend_init_context.h.

use crate::runtime::backend::options::{BackendOption, OptionValue};
use crate::runtime::core::error::Error;
use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::Result;
use crate::runtime::core::span::Span;

/// BackendInitContext will be used to inject runtime info for to initialize
/// delegate.
// [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context]
pub struct BackendInitContext {
    runtime_allocator_: *mut dyn MemoryAllocatorBase,
    event_tracer_: *mut dyn EventTracer,
    method_name_: *const core::ffi::c_char,
    named_data_map_: *const dyn NamedDataMap,
    runtime_specs_: Span<BackendOption>,
}

impl BackendInitContext {
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn]
    //
    // PORT-NOTE: C++ default args (`event_tracer = nullptr, method_name =
    // nullptr, named_data_map = nullptr, runtime_specs = {}`). Rust has no
    // default args; callers pass all five explicitly. `runtime_specs` is
    // `Span<const BackendOption>` in C++; the crate's `Span<T>` is over a
    // `*mut T`, so the const-ness of the view is carried through the API's
    // read-only accessors rather than the element type.
    pub fn new(
        runtime_allocator: *mut dyn MemoryAllocatorBase,
        event_tracer: *mut dyn EventTracer,
        method_name: *const core::ffi::c_char,
        named_data_map: *const dyn NamedDataMap,
        runtime_specs: Span<BackendOption>,
    ) -> Self {
        BackendInitContext {
            runtime_allocator_: runtime_allocator,
            // Event-tracer gating: only stored when tracing is enabled at
            // compile time, otherwise forced null regardless of the argument.
            #[cfg(feature = "event-tracer")]
            event_tracer_: event_tracer,
            #[cfg(not(feature = "event-tracer"))]
            event_tracer_: {
                let _ = event_tracer;
                NULL_EVENT_TRACER
            },
            method_name_: method_name,
            named_data_map_: named_data_map,
            runtime_specs_: runtime_specs,
        }
    }

    /// Get the runtime allocator passed from Method. It's the same runtime
    /// executor used by the standard executor runtime and the life span is the
    /// same as the model.
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-allocator-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-allocator-fn]
    pub fn get_runtime_allocator(&mut self) -> *mut dyn MemoryAllocatorBase {
        self.runtime_allocator_
    }

    /// Returns a pointer (null if not installed) to an instance of EventTracer to
    /// do profiling/debugging logging inside the delegate backend. Users will need
    /// access to this pointer to use any of the event tracer APIs.
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.event-tracer-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.event-tracer-fn]
    pub fn event_tracer(&mut self) -> *mut dyn EventTracer {
        self.event_tracer_
    }

    /// Get the loaded method name from ExecuTorch runtime. Usually it's
    /// "forward", however, if there are multiple methods in the .pte file, it can
    /// be different. One example is that we may have prefill and decode methods in
    /// the same .pte file. In this case, when client loads "prefill" method, the
    /// `get_method_name` function will return "prefill", when client loads
    /// "decode" method, the `get_method_name` function will return "decode".
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-method-name-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-method-name-fn]
    pub fn get_method_name(&self) -> *const core::ffi::c_char {
        self.method_name_
    }

    /// Get the named data map from ExecuTorch runtime.
    /// This provides a way for backends to retrieve data blobs by key.
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-named-data-map-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-named-data-map-fn]
    pub fn get_named_data_map(&self) -> *const dyn NamedDataMap {
        self.named_data_map_
    }

    /// Get the runtime specs (load-time options) for this backend.
    /// These are per-delegate options passed at Module::load() time.
    ///
    /// @return Span of BackendOption containing the runtime specs, or empty span
    ///         if no runtime specs were provided.
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn]
    pub fn runtime_specs(&self) -> Span<BackendOption> {
        self.runtime_specs_
    }

    /// Get a runtime spec value by key and type.
    ///
    /// `key` The option key to look up.
    /// @return Result containing the value if found and type matches,
    ///         Error::NotFound if key doesn't exist,
    ///         Error::InvalidArgument if key exists but type doesn't match.
    //
    // PORT-NOTE: C++ `template <typename T> get_runtime_spec(const char* key)`
    // static_asserts `T` in {bool, int, const char*} and resolves via `if
    // constexpr`. In Rust the three instantiations become three explicit
    // methods, each carrying the same span-scan + variant-match control flow.
    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    pub fn get_runtime_spec_bool(&self, key: *const core::ffi::c_char) -> Result<bool> {
        for i in 0..self.runtime_specs_.size() {
            let opt = unsafe { self.runtime_specs_.index(i) };
            if unsafe { libc::strcmp(opt.key.as_ptr(), key) } == 0 {
                if let OptionValue::Bool(val) = &opt.value {
                    return Ok(*val);
                }
                return Err(Error::InvalidArgument);
            }
        }
        Err(Error::NotFound)
    }

    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    pub fn get_runtime_spec_int(&self, key: *const core::ffi::c_char) -> Result<i32> {
        for i in 0..self.runtime_specs_.size() {
            let opt = unsafe { self.runtime_specs_.index(i) };
            if unsafe { libc::strcmp(opt.key.as_ptr(), key) } == 0 {
                if let OptionValue::Int(val) = &opt.value {
                    return Ok(*val);
                }
                return Err(Error::InvalidArgument);
            }
        }
        Err(Error::NotFound)
    }

    // [spec:et:def:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn]
    pub fn get_runtime_spec_str(
        &self,
        key: *const core::ffi::c_char,
    ) -> Result<*const core::ffi::c_char> {
        for i in 0..self.runtime_specs_.size() {
            let opt = unsafe { self.runtime_specs_.index(i) };
            if unsafe { libc::strcmp(opt.key.as_ptr(), key) } == 0 {
                if let OptionValue::CharArray(arr) = &opt.value {
                    return Ok(arr.as_ptr());
                }
                return Err(Error::InvalidArgument);
            }
        }
        Err(Error::NotFound)
    }
}

// PORT-NOTE: The C++ constructor's `#ifdef ET_EVENT_TRACER_ENABLED` gate forces
// `event_tracer_` to `nullptr` when tracing is compiled out. A `*mut dyn
// EventTracer` is a fat pointer and cannot be built with
// `core::ptr::null_mut()`. Mirroring the `NullDeviceAllocator` idiom in
// device_allocator.rs, a null trait-object pointer is produced by coercing a
// null thin pointer of a never-instantiated concrete implementor. Only compiled
// in the tracing-disabled configuration where it is actually used.
#[cfg(not(feature = "event-tracer"))]
const NULL_EVENT_TRACER: *mut dyn EventTracer =
    core::ptr::null_mut::<null_tracer::NullEventTracer>() as *mut dyn EventTracer;

#[cfg(not(feature = "event-tracer"))]
mod null_tracer {
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::event_tracer::{
        AllocatorID, ChainID, DebugHandle, DelegateDebugIntId, EventTracer, EventTracerEntry,
        EventTracerFilterBase, EventTracerState, LoggedEValueType,
    };
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::core::result::Result;
    use crate::runtime::platform::types::et_timestamp_t;

    pub struct NullEventTracer;
    impl EventTracer for NullEventTracer {
        fn state(&self) -> &EventTracerState {
            unreachable!()
        }
        fn state_mut(&mut self) -> &mut EventTracerState {
            unreachable!()
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
            unreachable!()
        }
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: ChainID,
            _debug_handle: DebugHandle,
        ) -> EventTracerEntry {
            unreachable!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
        ) -> EventTracerEntry {
            unreachable!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn log_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _start_time: et_timestamp_t,
            _end_time: et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn end_profiling(&mut self, _prof_entry: EventTracerEntry) {
            unreachable!()
        }
        fn track_allocation(&mut self, _id: AllocatorID, _size: usize) {
            unreachable!()
        }
        fn track_allocator(&mut self, _name: *const core::ffi::c_char) -> AllocatorID {
            unreachable!()
        }
        fn log_evalue(&mut self, _evalue: &EValue, _evalue_type: LoggedEValueType) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &Tensor,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: ArrayRef<Tensor>,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &i32,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &bool,
        ) -> Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &f64,
        ) -> Result<bool> {
            unreachable!()
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn EventTracerFilterBase,
        ) {
            unreachable!()
        }
    }
}

// Literal port of runtime/backend/test/backend_init_context_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::options::BackendOptions;
    use crate::runtime::core::memory_allocator::MemoryAllocator;
    use crate::runtime::platform::runtime::runtime_init;

    // C++ passes string-literal keys as `const char (&key)[N]` (N includes the
    // trailing NUL). This helper turns a NUL-terminated byte literal into the
    // equivalent `[c_char; N]` key array the set methods expect.
    const fn key<const N: usize>(bytes: &[u8; N]) -> [core::ffi::c_char; N] {
        let mut out = [0 as core::ffi::c_char; N];
        let mut i = 0;
        while i < N {
            out[i] = bytes[i] as core::ffi::c_char;
            i += 1;
        }
        out
    }

    // Null fat-pointer helpers: the tests only pass and compare null pointers,
    // never dereference them. A `*mut dyn Trait` / `*const dyn Trait` is a fat
    // pointer, so the null slot value is produced by coercing a null thin
    // pointer of a never-instantiated concrete implementor.
    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }
    fn null_event_tracer() -> *mut dyn EventTracer {
        core::ptr::null_mut::<TestNullEventTracer>() as *mut dyn EventTracer
    }
    fn null_named_data_map() -> *const dyn NamedDataMap {
        core::ptr::null::<TestNullNamedDataMap>() as *const dyn NamedDataMap
    }

    fn setup() {
        runtime_init();
    }

    // Test default constructor without runtime specs
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn/test]
    #[test]
    fn backend_init_context_test_default_constructor_no_runtime_specs() {
        setup();
        // C++ `BackendInitContext context(nullptr)` uses default args for the
        // rest; the Rust ctor has no default args, so pass the equivalent nulls
        // and an empty span.
        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );

        let specs = context.runtime_specs();
        assert_eq!(specs.size(), 0);
    }

    // Test constructor with runtime specs
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.backend-init-context-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn/test]
    #[test]
    fn backend_init_context_test_constructor_with_runtime_specs() {
        setup();
        let mut opts: BackendOptions<4> = BackendOptions::new();
        opts.set_option_str(&key(b"compute_unit\0"), c"cpu_and_gpu".as_ptr());
        opts.set_option_int(&key(b"num_threads\0"), 4);
        opts.set_option_bool(&key(b"enable_profiling\0"), true);

        // Create a const span from the mutable view
        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),      // runtime_allocator
            null_event_tracer(),   // event_tracer
            c"forward".as_ptr(),   // method_name
            null_named_data_map(), // named_data_map
            const_span,            // runtime_specs
        );

        let specs = context.runtime_specs();
        assert_eq!(specs.size(), 3);
    }

    // Test get_runtime_spec<bool> with valid key
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_get_runtime_spec_bool_valid() {
        setup();
        let mut opts: BackendOptions<2> = BackendOptions::new();
        opts.set_option_bool(&key(b"enable_profiling\0"), true);
        opts.set_option_bool(&key(b"debug_mode\0"), false);

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            const_span,
        );

        let result1 = context.get_runtime_spec_bool(c"enable_profiling".as_ptr());
        assert!(result1.is_ok());
        assert!(result1.unwrap());

        let result2 = context.get_runtime_spec_bool(c"debug_mode".as_ptr());
        assert!(result2.is_ok());
        assert!(!result2.unwrap());
    }

    // Test get_runtime_spec<int> with valid key
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_get_runtime_spec_int_valid() {
        setup();
        let mut opts: BackendOptions<2> = BackendOptions::new();
        opts.set_option_int(&key(b"num_threads\0"), 8);
        opts.set_option_int(&key(b"batch_size\0"), 32);

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            const_span,
        );

        let result1 = context.get_runtime_spec_int(c"num_threads".as_ptr());
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), 8);

        let result2 = context.get_runtime_spec_int(c"batch_size".as_ptr());
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), 32);
    }

    // Test get_runtime_spec<const char*> with valid key
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_get_runtime_spec_string_valid() {
        setup();
        let mut opts: BackendOptions<2> = BackendOptions::new();
        opts.set_option_str(&key(b"compute_unit\0"), c"cpu_and_gpu".as_ptr());
        opts.set_option_str(&key(b"cache_dir\0"), c"/tmp/cache".as_ptr());

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            const_span,
        );

        let result1 = context.get_runtime_spec_str(c"compute_unit".as_ptr());
        assert!(result1.is_ok());
        assert_eq!(
            unsafe { libc::strcmp(result1.unwrap(), c"cpu_and_gpu".as_ptr()) },
            0
        );

        let result2 = context.get_runtime_spec_str(c"cache_dir".as_ptr());
        assert!(result2.is_ok());
        assert_eq!(
            unsafe { libc::strcmp(result2.unwrap(), c"/tmp/cache".as_ptr()) },
            0
        );
    }

    // Test get_runtime_spec<T> with non-existent key returns NotFound
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_get_runtime_spec_not_found() {
        setup();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            const_span,
        );

        let bool_result = context.get_runtime_spec_bool(c"nonexistent".as_ptr());
        assert!(bool_result.is_err());
        assert_eq!(bool_result.unwrap_err(), Error::NotFound);

        let int_result = context.get_runtime_spec_int(c"nonexistent".as_ptr());
        assert!(int_result.is_err());
        assert_eq!(int_result.unwrap_err(), Error::NotFound);

        let string_result = context.get_runtime_spec_str(c"nonexistent".as_ptr());
        assert!(string_result.is_err());
        assert_eq!(string_result.unwrap_err(), Error::NotFound);
    }

    // Test get_runtime_spec<T> with wrong type returns InvalidArgument
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_get_runtime_spec_type_mismatch() {
        setup();
        let mut opts: BackendOptions<3> = BackendOptions::new();
        opts.set_option_bool(&key(b"bool_opt\0"), true);
        opts.set_option_int(&key(b"int_opt\0"), 42);
        opts.set_option_str(&key(b"string_opt\0"), c"hello".as_ptr());

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            const_span,
        );

        // Try to get bool as int
        let result1 = context.get_runtime_spec_int(c"bool_opt".as_ptr());
        assert!(result1.is_err());
        assert_eq!(result1.unwrap_err(), Error::InvalidArgument);

        // Try to get int as string
        let result2 = context.get_runtime_spec_str(c"int_opt".as_ptr());
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), Error::InvalidArgument);

        // Try to get string as bool
        let result3 = context.get_runtime_spec_bool(c"string_opt".as_ptr());
        assert!(result3.is_err());
        assert_eq!(result3.unwrap_err(), Error::InvalidArgument);
    }

    // Test empty runtime specs
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.runtime-specs-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-spec-fn/test]
    #[test]
    fn backend_init_context_test_empty_runtime_specs() {
        setup();
        let empty_span: Span<BackendOption> = Span::new();
        let context = BackendInitContext::new(
            null_allocator(),
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            empty_span,
        );

        assert_eq!(context.runtime_specs().size(), 0);

        // All lookups should return NotFound
        let bool_result = context.get_runtime_spec_bool(c"any_key".as_ptr());
        assert!(bool_result.is_err());
        assert_eq!(bool_result.unwrap_err(), Error::NotFound);
    }

    // Test that other context fields still work
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-runtime-allocator-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.event-tracer-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-method-name-fn/test]
    // [spec:et:sem:backend-init-context.executorch.et-runtime-namespace.backend-init-context.get-named-data-map-fn/test]
    #[test]
    fn backend_init_context_test_other_fields_still_work() {
        setup();
        let mut opts: BackendOptions<1> = BackendOptions::new();
        opts.set_option_str(&key(b"key\0"), c"value".as_ptr());

        let view = opts.view();
        let const_span = Span::from_raw_parts(view.data(), view.size());

        let mut context = BackendInitContext::new(
            null_allocator(),      // runtime_allocator
            null_event_tracer(),   // event_tracer
            c"forward".as_ptr(),   // method_name
            null_named_data_map(), // named_data_map
            const_span,            // runtime_specs
        );

        assert!(context.get_runtime_allocator().is_null());
        assert!(context.event_tracer().is_null());
        assert_eq!(
            unsafe { libc::strcmp(context.get_method_name(), c"forward".as_ptr()) },
            0
        );
        assert!(context.get_named_data_map().is_null());
    }

    // Zero-sized types used only to synthesize null trait-object pointers; never
    // instantiated or dereferenced.
    struct TestNullNamedDataMap;
    impl NamedDataMap for TestNullNamedDataMap {
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

    struct TestNullEventTracer;
    impl EventTracer for TestNullEventTracer {
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
