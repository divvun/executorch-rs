//! Literal port of runtime/backend/interface.cpp + runtime/backend/interface.h.

use crate::runtime::backend::backend_execution_context::BackendExecutionContext;
use crate::runtime::backend::backend_init_context::BackendInitContext;
use crate::runtime::backend::backend_option_context::BackendOptionContext;
use crate::runtime::backend::options::BackendOption;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;
use crate::runtime::core::span::Span;

// [spec:et:def:interface.executorch.et-runtime-namespace.sized-buffer]
#[derive(Clone, Copy)]
pub struct SizedBuffer {
    pub buffer: *mut core::ffi::c_void,
    pub nbytes: usize, // number of bytes of buffer
}

// [spec:et:def:interface.executorch.et-runtime-namespace.compile-spec]
#[derive(Clone, Copy)]
pub struct CompileSpec {
    pub key: *const core::ffi::c_char, // spec key
    pub value: SizedBuffer,            // spec value
}

/// An opaque handle managed by a backend. Typically points to a backend-private
/// class/struct.
//
// PORT-NOTE: C++ `using DelegateHandle = void;`, so `DelegateHandle*` is
// `void*`. Mapped to `core::ffi::c_void`, and handles are `*mut DelegateHandle`.
pub type DelegateHandle = core::ffi::c_void;

// [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface]
//
// PORT-NOTE: C++ abstract `class BackendInterface` maps to a trait. The
// out-of-line pure-virtual destructor carries no behavior beyond "base cleanup
// is a no-op" and needs no analogue in Rust; `Drop` runs on the concrete
// implementor behind a `dyn` reference, so its markers collapse onto the trait.
// [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.backend-interface-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.backend-interface-fn]
pub trait BackendInterface {
    /// Returns true if the backend is available to process delegation calls.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.is-available-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.is-available-fn]
    #[must_use]
    fn is_available(&self) -> bool;

    /// Responsible to further process (compile/transform/optimize) the compiled
    /// unit that was produced, ahead-of-time, as well as perform any backend
    /// initialization to ready it for execution.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.init-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.init-fn]
    #[must_use]
    fn init(
        &self,
        context: &mut BackendInitContext,
        processed: *mut FreeableBuffer,
        compile_specs: ArrayRef<CompileSpec>,
    ) -> Result<*mut DelegateHandle>;

    /// Responsible for executing the given method's handle, as it was produced
    /// by compile.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.execute-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.execute-fn]
    #[must_use]
    fn execute(
        &self,
        context: &mut BackendExecutionContext,
        handle: *mut DelegateHandle,
        args: Span<*mut EValue>,
    ) -> Error;

    /// Responsible update the backend status, if any. The backend options are
    /// passed in by users, and the backend can update its internal status based
    /// on the options.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn]
    #[must_use]
    fn set_option(
        &self,
        context: &mut BackendOptionContext,
        backend_options: &Span<BackendOption>,
    ) -> Error {
        let _ = context;
        let _ = backend_options;
        Error::Ok
    }

    /// Responsible update the backend status, if any. The backend options are
    /// passed in by users, and the backend can update its internal status based
    /// on the options.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn]
    #[must_use]
    fn get_option(
        &self,
        context: &mut BackendOptionContext,
        backend_options: &mut Span<BackendOption>,
    ) -> Error {
        let _ = context;
        let _ = backend_options;
        Error::Ok
    }

    /// Responsible for destroying a handle, if it's required for some backend.
    // [spec:et:def:interface.executorch.et-runtime-namespace.backend-interface.destroy-fn]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.destroy-fn]
    fn destroy(&self, handle: *mut DelegateHandle) {
        let _ = handle;
    }
}

/// Returns the corresponding object pointer for a given string name.
/// The mapping is populated using register_backend method.
///
/// `name` Name of the user-defined backend delegate.
/// @retval Pointer to the appropriate object that implements BackendInterface.
///         Nullptr if it can't find anything with the given name.
// (declaration; definition below)

/// A named instance of a backend.
// [spec:et:def:interface.executorch.et-runtime-namespace.backend]
#[derive(Clone, Copy)]
pub struct Backend {
    /// The name of the backend. Must match the string used in the PTE file.
    pub name: *const core::ffi::c_char,
    /// The instance of the backend to use when loading and executing programs.
    pub backend: *mut dyn BackendInterface,
}

// The max number of backends that can be registered globally.
const K_MAX_REGISTERED_BACKENDS: usize = 16;

// TODO(T128866626): Remove global static variables. We want to be able to run
// multiple Executor instances and having a global registration isn't a viable
// solution in the long term.
//
// PORT-NOTE: the C++ global `Backend registered_backends[16]` is default-
// initialized (name/backend pointers zeroed). A `*mut dyn BackendInterface` is
// a fat pointer that cannot be `core::ptr::null_mut()`; the null slot value is
// produced by coercing a null thin pointer of a never-instantiated concrete
// implementor, mirroring the `NullDeviceAllocator` idiom.
const NULL_BACKEND_INTERFACE: *mut dyn BackendInterface =
    core::ptr::null_mut::<NullBackendInterface>() as *mut dyn BackendInterface;

const NULL_BACKEND: Backend = Backend {
    name: core::ptr::null(),
    backend: NULL_BACKEND_INTERFACE,
};

/// Global table of registered backends.
//
// PORT-NOTE: the C++ globals live in an anonymous namespace (file-local mutable
// state). Mapped to `static mut` accessed through `&raw` — matching the C++
// "register during single-threaded static init, read concurrently" contract.
// Not thread-safe, exactly as the original.
static mut REGISTERED_BACKENDS: [Backend; K_MAX_REGISTERED_BACKENDS] =
    [NULL_BACKEND; K_MAX_REGISTERED_BACKENDS];

/// The number of backends registered in the table.
static mut NUM_REGISTERED_BACKENDS: usize = 0;

// [spec:et:def:interface.executorch.et-runtime-namespace.get-backend-class-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-class-fn]
pub fn get_backend_class(name: *const core::ffi::c_char) -> *mut dyn BackendInterface {
    let num = unsafe { NUM_REGISTERED_BACKENDS };
    for i in 0..num {
        let backend: Backend = unsafe { (*(&raw const REGISTERED_BACKENDS))[i] };
        if unsafe { libc::strcmp(backend.name, name) } == 0 {
            return backend.backend;
        }
    }
    NULL_BACKEND_INTERFACE
}

// [spec:et:def:interface.executorch.et-runtime-namespace.register-backend-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.register-backend-fn]
#[must_use]
pub fn register_backend(backend: &Backend) -> Error {
    if unsafe { NUM_REGISTERED_BACKENDS } >= K_MAX_REGISTERED_BACKENDS {
        return Error::Internal;
    }

    // Check if the name already exists in the table
    if !get_backend_class(backend.name).is_null() {
        return Error::InvalidArgument;
    }

    unsafe {
        let idx = NUM_REGISTERED_BACKENDS;
        (*(&raw mut REGISTERED_BACKENDS))[idx] = *backend;
        NUM_REGISTERED_BACKENDS += 1;
    }
    Error::Ok
}

// [spec:et:def:interface.executorch.et-runtime-namespace.get-num-registered-backends-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.get-num-registered-backends-fn]
pub fn get_num_registered_backends() -> usize {
    unsafe { NUM_REGISTERED_BACKENDS }
}

// [spec:et:def:interface.executorch.et-runtime-namespace.get-backend-name-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-name-fn]
#[must_use]
pub fn get_backend_name(index: usize) -> Result<*const core::ffi::c_char> {
    if index >= unsafe { NUM_REGISTERED_BACKENDS } {
        return Err(Error::InvalidArgument);
    }
    Ok(unsafe { (*(&raw const REGISTERED_BACKENDS))[index].name })
}

// [spec:et:def:interface.executorch.et-runtime-namespace.set-option-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.set-option-fn]
#[must_use]
pub fn set_option(
    backend_name: *const core::ffi::c_char,
    backend_options: Span<BackendOption>,
) -> Error {
    let backend_class = get_backend_class(backend_name);
    if backend_class.is_null() {
        return Error::NotFound;
    }

    let mut backend_option_context = BackendOptionContext::new();
    let result =
        unsafe { (*backend_class).set_option(&mut backend_option_context, &backend_options) };
    if result != Error::Ok {
        return result;
    }
    Error::Ok
}

// [spec:et:def:interface.executorch.et-runtime-namespace.get-option-fn]
// [spec:et:sem:interface.executorch.et-runtime-namespace.get-option-fn]
#[must_use]
pub fn get_option(
    backend_name: *const core::ffi::c_char,
    backend_options: Span<BackendOption>,
) -> Error {
    let backend_class = get_backend_class(backend_name);
    if backend_class.is_null() {
        return Error::NotFound;
    }
    let mut backend_option_context = BackendOptionContext::new();
    let mut backend_options_ref: Span<BackendOption> =
        Span::from_raw_parts(backend_options.data(), backend_options.size());
    let result = unsafe {
        (*backend_class).get_option(&mut backend_option_context, &mut backend_options_ref)
    };
    if result != Error::Ok {
        return result;
    }
    Error::Ok
}

// PORT-NOTE: never-instantiated concrete implementor used only to spell the
// null `*mut dyn BackendInterface` slot value for the fixed registry array.
struct NullBackendInterface;
impl BackendInterface for NullBackendInterface {
    fn is_available(&self) -> bool {
        unreachable!()
    }
    fn init(
        &self,
        _context: &mut BackendInitContext,
        _processed: *mut FreeableBuffer,
        _compile_specs: ArrayRef<CompileSpec>,
    ) -> Result<*mut DelegateHandle> {
        unreachable!()
    }
    fn execute(
        &self,
        _context: &mut BackendExecutionContext,
        _handle: *mut DelegateHandle,
        _args: Span<*mut EValue>,
    ) -> Error {
        unreachable!()
    }
}

// Literal port of runtime/backend/test/backend_interface_update_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::backend::options::{BackendOption, BackendOptions, OptionValue};
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::platform::runtime::runtime_init;
    use core::cell::Cell;
    use core::cell::RefCell;

    // C++ passes string-literal keys as `const char (&key)[N]` (N includes the
    // trailing NUL). This helper turns a NUL-terminated byte literal into the
    // equivalent `[c_char; N]` key array.
    const fn key<const N: usize>(bytes: &[u8; N]) -> [core::ffi::c_char; N] {
        let mut out = [0 as core::ffi::c_char; N];
        let mut i = 0;
        while i < N {
            out[i] = bytes[i] as core::ffi::c_char;
            i += 1;
        }
        out
    }

    // Builds a full-width `BackendOption::key` array (kMaxOptionKeyLength wide),
    // NUL-padded, for constructing `BackendOption` struct literals directly.
    const fn key_field<const N: usize>(
        bytes: &[u8; N],
    ) -> [core::ffi::c_char; crate::runtime::backend::options::K_MAX_OPTION_KEY_LENGTH] {
        let mut out =
            [0 as core::ffi::c_char; crate::runtime::backend::options::K_MAX_OPTION_KEY_LENGTH];
        let mut i = 0;
        while i < N {
            out[i] = bytes[i] as core::ffi::c_char;
            i += 1;
        }
        out
    }

    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }
    // Reuses the crate-internal null event-tracer helper (a null fat pointer over
    // a never-instantiated concrete implementor).
    fn null_event_tracer() -> *mut dyn crate::runtime::core::event_tracer::EventTracer {
        crate::extension::module::module::null_event_tracer()
    }
    fn null_named_data_map() -> *const dyn crate::runtime::core::named_data_map::NamedDataMap {
        core::ptr::null::<TestNullNamedDataMap>()
            as *const dyn crate::runtime::core::named_data_map::NamedDataMap
    }

    // Zero-sized type used only to synthesize a null NamedDataMap pointer; never
    // instantiated or dereferenced.
    struct TestNullNamedDataMap;
    impl crate::runtime::core::named_data_map::NamedDataMap for TestNullNamedDataMap {
        fn get_tensor_layout(
            &self,
            _key: &str,
        ) -> Result<crate::runtime::core::tensor_layout::TensorLayout> {
            unreachable!()
        }
        fn get_data(&self, _key: &str) -> Result<FreeableBuffer> {
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

    // Reads the NUL-terminated C string stored in a `[c_char; N]` array into a
    // Rust String — mirrors C++ `std::string(arr.data())`.
    fn c_array_to_string<const N: usize>(arr: &[core::ffi::c_char; N]) -> String {
        let mut s = String::new();
        for &c in arr.iter() {
            if c == 0 {
                break;
            }
            s.push(c as u8 as char);
        }
        s
    }

    // PORT-NOTE: C++ `MockBackend` mutates members in `const` methods via
    // `mutable`. Rust models that with interior mutability (`Cell`/`RefCell`)
    // behind the `&self` trait methods.
    struct MockBackend {
        target_backend: RefCell<Option<String>>,
        num_threads: Cell<i32>,
        debug: Cell<bool>,
        init_called: Cell<bool>,
        execute_count: Cell<i32>,
        set_option_count: Cell<i32>,
    }

    impl MockBackend {
        fn new() -> Self {
            MockBackend {
                target_backend: RefCell::new(None),
                num_threads: Cell::new(0),
                debug: Cell::new(false),
                init_called: Cell::new(false),
                execute_count: Cell::new(0),
                set_option_count: Cell::new(0),
            }
        }
    }

    impl BackendInterface for MockBackend {
        fn is_available(&self) -> bool {
            true
        }

        fn init(
            &self,
            _context: &mut BackendInitContext,
            _processed: *mut FreeableBuffer,
            _compile_specs: ArrayRef<CompileSpec>,
        ) -> Result<*mut DelegateHandle> {
            self.init_called.set(true);
            Ok(core::ptr::null_mut())
        }

        fn execute(
            &self,
            _context: &mut BackendExecutionContext,
            _handle: *mut DelegateHandle,
            _args: Span<*mut EValue>,
        ) -> Error {
            self.execute_count.set(self.execute_count.get() + 1);
            Error::Ok
        }

        fn set_option(
            &self,
            _context: &mut BackendOptionContext,
            backend_options: &Span<BackendOption>,
        ) -> Error {
            self.set_option_count.set(self.set_option_count.get() + 1);
            let mut success_update = 0;
            for i in 0..backend_options.size() {
                let backend_option = unsafe { backend_options.index(i) };
                if unsafe { libc::strcmp(backend_option.key.as_ptr(), c"Backend".as_ptr()) } == 0 {
                    if let OptionValue::CharArray(arr) = &backend_option.value {
                        // Store the value in our member variable
                        *self.target_backend.borrow_mut() = Some(c_array_to_string(arr));
                        success_update += 1;
                    }
                } else if unsafe {
                    libc::strcmp(backend_option.key.as_ptr(), c"NumberOfThreads".as_ptr())
                } == 0
                {
                    if let OptionValue::Int(val) = &backend_option.value {
                        self.num_threads.set(*val);
                        success_update += 1;
                    }
                } else if unsafe { libc::strcmp(backend_option.key.as_ptr(), c"Debug".as_ptr()) }
                    == 0
                {
                    if let OptionValue::Bool(val) = &backend_option.value {
                        self.debug.set(*val);
                        success_update += 1;
                    }
                }
            }
            if success_update == backend_options.size() {
                return Error::Ok;
            }
            Error::InvalidArgument
        }
    }

    // Fixture: BackendInterfaceUpdateTest.
    fn mock_setup() -> MockBackend {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        runtime_init();
        MockBackend::new()
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    #[test]
    fn backend_interface_update_test_handles_invalid_option() {
        let mock_backend = mock_setup();
        let mut context = BackendOptionContext::new();

        // Test invalid key case
        // std::array<char, 256> value_array{"None"};
        let mut value_array: [core::ffi::c_char;
            crate::runtime::backend::options::K_MAX_OPTION_VALUE_LENGTH] =
            [0; crate::runtime::backend::options::K_MAX_OPTION_VALUE_LENGTH];
        value_array[0] = b'N' as core::ffi::c_char;
        value_array[1] = b'o' as core::ffi::c_char;
        value_array[2] = b'n' as core::ffi::c_char;
        value_array[3] = b'e' as core::ffi::c_char;
        let mut invalid_option = BackendOption {
            key: key_field(b"InvalidKey\0"),
            value: OptionValue::CharArray(value_array),
        };

        // C++ passes a single BackendOption where Span is expected (implicit
        // single-element Span ctor); mirror with a length-1 span over it.
        let span = Span::from_raw_parts(&mut invalid_option as *mut BackendOption, 1);
        let err = mock_backend.set_option(&mut context, &span);
        assert_eq!(err, Error::InvalidArgument);
    }

    // Also constructs a BackendOptionContext and passes it through set_option,
    // exercising the empty-struct constructor as a usable placeholder handle.
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    // [spec:et:sem:backend-option-context.executorch.et-runtime-namespace.backend-option-context.backend-option-context-fn/test]
    #[test]
    fn backend_interface_update_test_handles_string_option() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        let mut context = BackendOptionContext::new();
        options.set_option_str(&key(b"Backend\0"), c"GPU".as_ptr());

        assert_eq!(*mock_backend.target_backend.borrow(), None);

        // Test successful update
        let err = mock_backend.set_option(&mut context, &options.view());
        assert_eq!(err, Error::Ok);

        assert_eq!(mock_backend.target_backend.borrow().as_deref(), Some("GPU"));
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    #[test]
    fn backend_interface_update_test_handles_int_option() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        // Check the default num_threads value is 0
        assert!(!mock_backend.debug.get());
        let mut context = BackendOptionContext::new();

        let expected_num_threads = 4;

        options.set_option_int(&key(b"NumberOfThreads\0"), expected_num_threads);

        // Test successful update
        let err = mock_backend.set_option(&mut context, &options.view());
        assert_eq!(err, Error::Ok);
        assert_eq!(mock_backend.num_threads.get(), expected_num_threads);
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    #[test]
    fn backend_interface_update_test_handles_bool_option() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        assert!(!mock_backend.debug.get());
        let mut context = BackendOptionContext::new();

        options.set_option_bool(&key(b"Debug\0"), true);

        // Test successful update
        let err = mock_backend.set_option(&mut context, &options.view());
        assert_eq!(err, Error::Ok);

        assert!(mock_backend.debug.get());
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    #[test]
    fn backend_interface_update_test_handles_multiple_options() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        assert!(!mock_backend.debug.get());
        let mut context = BackendOptionContext::new();

        options.set_option_bool(&key(b"Debug\0"), true);
        options.set_option_int(&key(b"NumberOfThreads\0"), 4);
        options.set_option_str(&key(b"Backend\0"), c"GPU".as_ptr());

        // Test successful update
        let err = mock_backend.set_option(&mut context, &options.view());
        assert_eq!(err, Error::Ok);

        assert!(mock_backend.debug.get());
        assert_eq!(mock_backend.num_threads.get(), 4);
        assert_eq!(mock_backend.target_backend.borrow().as_deref(), Some("GPU"));
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.init-fn/test]
    #[test]
    fn backend_interface_update_test_update_before_init() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        let mut option_context = BackendOptionContext::new();
        let mut memory_allocator = MemoryAllocator::new(0, core::ptr::null_mut());

        let mut init_context = BackendInitContext::new(
            &mut memory_allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );

        // Create backend option
        options.set_option_str(&key(b"Backend\0"), c"GPU".as_ptr());

        // Update before init
        let err = mock_backend.set_option(&mut option_context, &options.view());
        assert_eq!(err, Error::Ok);

        // Now call init
        let processed: *mut FreeableBuffer = core::ptr::null_mut(); // Not used in mock
        let compile_specs: ArrayRef<CompileSpec> = ArrayRef::new(); // Empty
        let handle_or_error = mock_backend.init(&mut init_context, processed, compile_specs);
        // C++ checks handle_or_error.error() == Error::Ok, i.e. the Result is Ok.
        assert!(handle_or_error.is_ok());

        // Verify state
        assert!(mock_backend.init_called.get());
        assert_eq!(mock_backend.set_option_count.get(), 1);
        assert_eq!(mock_backend.execute_count.get(), 0);
        assert!(mock_backend.target_backend.borrow().is_some());
        assert_eq!(mock_backend.target_backend.borrow().as_deref(), Some("GPU"));
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.init-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.execute-fn/test]
    #[test]
    fn backend_interface_update_test_update_after_init_before_execute() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        let mut option_context = BackendOptionContext::new();
        let mut init_memory_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let mut init_context = BackendInitContext::new(
            &mut init_memory_allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );
        let mut execute_context =
            BackendExecutionContext::new(null_event_tracer(), null_allocator(), core::ptr::null());

        // First call init
        let processed: *mut FreeableBuffer = core::ptr::null_mut();
        let compile_specs: ArrayRef<CompileSpec> = ArrayRef::new();
        let handle_or_error = mock_backend.init(&mut init_context, processed, compile_specs);
        assert!(handle_or_error.is_ok());

        // Verify init called but execute not called
        assert!(mock_backend.init_called.get());
        assert_eq!(mock_backend.execute_count.get(), 0);

        // Now update
        options.set_option_str(&key(b"Backend\0"), c"CPU".as_ptr());
        let err = mock_backend.set_option(&mut option_context, &options.view());
        assert_eq!(err, Error::Ok);

        // Now execute
        let handle: *mut DelegateHandle = handle_or_error.unwrap();
        let args: Span<*mut EValue> = Span::from_raw_parts(core::ptr::null_mut(), 0); // Not used in mock
        let err = mock_backend.execute(&mut execute_context, handle, args);
        assert_eq!(err, Error::Ok);

        // Verify state
        assert_eq!(mock_backend.set_option_count.get(), 1);
        assert_eq!(mock_backend.execute_count.get(), 1);
        assert!(mock_backend.target_backend.borrow().is_some());
        assert_eq!(mock_backend.target_backend.borrow().as_deref(), Some("CPU"));
    }

    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.init-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.execute-fn/test]
    #[test]
    fn backend_interface_update_test_update_between_executes() {
        let mock_backend = mock_setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        let mut option_context = BackendOptionContext::new();
        let mut init_memory_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let mut init_context = BackendInitContext::new(
            &mut init_memory_allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
            null_event_tracer(),
            core::ptr::null(),
            null_named_data_map(),
            Span::new(),
        );
        let mut execute_context =
            BackendExecutionContext::new(null_event_tracer(), null_allocator(), core::ptr::null());

        // Initialize
        let processed: *mut FreeableBuffer = core::ptr::null_mut();
        let compile_specs: ArrayRef<CompileSpec> = ArrayRef::new();
        let handle_or_error = mock_backend.init(&mut init_context, processed, compile_specs);
        assert!(handle_or_error.is_ok());
        let handle: *mut DelegateHandle = handle_or_error.unwrap();

        // First execute
        let args: Span<*mut EValue> = Span::from_raw_parts(core::ptr::null_mut(), 0); // Not used in mock
        let err = mock_backend.execute(&mut execute_context, handle, args);
        assert_eq!(err, Error::Ok);

        // Update between executes
        options.set_option_str(&key(b"Backend\0"), c"NPU".as_ptr());
        let err = mock_backend.set_option(&mut option_context, &options.view());
        assert_eq!(err, Error::Ok);

        // Second execute
        let err = mock_backend.execute(&mut execute_context, handle, args);
        assert_eq!(err, Error::Ok);

        // Verify state
        assert_eq!(mock_backend.set_option_count.get(), 1);
        assert_eq!(mock_backend.execute_count.get(), 2);
        assert!(mock_backend.target_backend.borrow().is_some());
        assert_eq!(mock_backend.target_backend.borrow().as_deref(), Some("NPU"));
    }

    // Mock backend for testing (StubBackend).
    struct StubBackend {
        last_options_size: Cell<usize>,
        last_num_threads: Cell<i32>,
        get_option_called: Cell<bool>,
        get_option_call_count: Cell<i32>,
        last_get_option_size: Cell<usize>,
        found_expected_key: Cell<bool>,
    }

    impl StubBackend {
        fn new() -> Self {
            StubBackend {
                last_options_size: Cell::new(0),
                last_num_threads: Cell::new(0),
                get_option_called: Cell::new(false),
                get_option_call_count: Cell::new(0),
                last_get_option_size: Cell::new(0),
                found_expected_key: Cell::new(false),
            }
        }
    }

    impl BackendInterface for StubBackend {
        fn is_available(&self) -> bool {
            true
        }

        fn init(
            &self,
            _context: &mut BackendInitContext,
            _processed: *mut FreeableBuffer,
            _compile_specs: ArrayRef<CompileSpec>,
        ) -> Result<*mut DelegateHandle> {
            Ok(core::ptr::null_mut())
        }

        fn execute(
            &self,
            _context: &mut BackendExecutionContext,
            _handle: *mut DelegateHandle,
            _args: Span<*mut EValue>,
        ) -> Error {
            Error::Ok
        }

        fn get_option(
            &self,
            _context: &mut BackendOptionContext,
            backend_options: &mut Span<BackendOption>,
        ) -> Error {
            // For testing purposes, just record that get_option was called
            // and verify the input parameters
            self.get_option_called.set(true);
            self.get_option_call_count
                .set(self.get_option_call_count.get() + 1);
            self.last_get_option_size.set(backend_options.size());

            // Verify that the expected option key is present and modify the value
            for i in 0..backend_options.size() {
                let opt = unsafe { backend_options.index(i) };
                if unsafe { libc::strcmp(opt.key.as_ptr(), c"NumberOfThreads".as_ptr()) } == 0 {
                    // Set the value to what was stored by set_option
                    opt.value = OptionValue::Int(self.last_num_threads.get());
                    self.found_expected_key.set(true);
                    break;
                }
            }

            Error::Ok
        }

        fn set_option(
            &self,
            _context: &mut BackendOptionContext,
            backend_options: &Span<BackendOption>,
        ) -> Error {
            // Store the options for verification
            self.last_options_size.set(backend_options.size());
            if backend_options.size() > 0 {
                for i in 0..backend_options.size() {
                    let option = unsafe { backend_options.index(i) };
                    if unsafe { libc::strcmp(option.key.as_ptr(), c"NumberOfThreads".as_ptr()) }
                        == 0
                    {
                        if let OptionValue::Int(val) = &option.value {
                            self.last_num_threads.set(*val);
                        }
                    }
                }
            }
            Error::Ok
        }
    }

    // Test basic string functionality
    //
    // PORT-NOTE: C++ registers `stub_backend.get()` (a raw pointer that outlives
    // the test) into the global registry. The Rust registry stores a
    // `*mut dyn BackendInterface`; the stub is boxed and leaked so the raw
    // pointer stays valid, mirroring the C++ registration contract (the global
    // table is never cleared between tests either).
    // Also folds in the registry-enumeration surface (get_num_registered_backends,
    // get_backend_name) and the stub's is_available / destroy, since this is the
    // only test in this module that mutates the process-global registry — keeping
    // all registry assertions in one test avoids racing a second registration.
    // [spec:et:sem:interface.executorch.et-runtime-namespace.register-backend-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.set-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.get-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.get-num-registered-backends-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.get-backend-name-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.set-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.get-option-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.is-available-fn/test]
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.destroy-fn/test]
    #[test]
    fn backend_update_test_test_set_get_option() {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        runtime_init();

        // is_available on the stub is a pure predicate returning true.
        let stub_probe = StubBackend::new();
        assert!(stub_probe.is_available());
        // destroy default hook is a no-op; calling it (incl. with a null handle)
        // must not panic or touch state.
        stub_probe.destroy(core::ptr::null_mut());

        // Register the stub backend.
        let stub_backend: &'static StubBackend = Box::leak(Box::new(StubBackend::new()));
        let backend_config = Backend {
            name: c"StubBackend".as_ptr(),
            backend: stub_backend as *const StubBackend as *mut StubBackend
                as *mut dyn BackendInterface,
        };
        let register_result = register_backend(&backend_config);
        assert_eq!(register_result, Error::Ok);

        // Registration bumped the global counter; our name is now enumerable.
        // Race-tolerant: even if a parallel test registers concurrently, our
        // entry is present within [0, count) and count itself is out of bounds.
        let count = get_num_registered_backends();
        assert!(count >= 1);
        let mut found = false;
        for i in 0..count {
            let name = get_backend_name(i).expect("index < count is in bounds");
            if unsafe { libc::strcmp(name, c"StubBackend".as_ptr()) } == 0 {
                found = true;
            }
        }
        assert!(found);
        // One past the last registered index is rejected.
        assert_eq!(get_backend_name(count), Err(Error::InvalidArgument));

        let mut backend_options: BackendOptions<1> = BackendOptions::new();
        let new_num_threads = 4;
        backend_options.set_option_int(&key(b"NumberOfThreads\0"), new_num_threads);

        let status = set_option(c"StubBackend".as_ptr(), backend_options.view());
        assert_eq!(status, Error::Ok);

        // Set up the default option, which will be populated by the get_option call
        let mut ref_backend_option = BackendOption {
            key: key_field(b"NumberOfThreads\0"),
            value: OptionValue::Int(0),
        };
        // C++ passes a single BackendOption where Span is expected (implicit
        // single-element Span ctor); mirror with a length-1 span over it.
        let ref_span = Span::from_raw_parts(&mut ref_backend_option as *mut BackendOption, 1);
        let status = get_option(c"StubBackend".as_ptr(), ref_span);
        assert_eq!(status, Error::Ok);

        // Verify that the backend actually received the options
        assert!(matches!(ref_backend_option.value, OptionValue::Int(v) if v == new_num_threads));

        // Verify that the backend actually update the options
        assert_eq!(stub_backend.last_options_size.get(), 1);
        assert_eq!(stub_backend.last_num_threads.get(), new_num_threads);
    }

    // The out-of-line pure-virtual `~BackendInterface()` exists so that
    // destroying a concrete backend through the base runs the derived
    // destructor (the base cleanup itself is a no-op). The Rust collapse:
    // dropping a `Box<dyn BackendInterface>` runs the implementor's Drop.
    // [spec:et:sem:interface.executorch.et-runtime-namespace.backend-interface.backend-interface-fn/test]
    #[test]
    fn backend_interface_dyn_drop_runs_concrete_destructor() {
        runtime_init();

        struct DroppableBackend<'a> {
            dropped: &'a Cell<bool>,
        }
        impl Drop for DroppableBackend<'_> {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }
        impl BackendInterface for DroppableBackend<'_> {
            fn is_available(&self) -> bool {
                true
            }
            fn init(
                &self,
                _context: &mut BackendInitContext,
                _processed: *mut FreeableBuffer,
                _compile_specs: ArrayRef<CompileSpec>,
            ) -> Result<*mut DelegateHandle> {
                Ok(core::ptr::null_mut())
            }
            fn execute(
                &self,
                _context: &mut BackendExecutionContext,
                _handle: *mut DelegateHandle,
                _args: Span<*mut EValue>,
            ) -> Error {
                Error::Ok
            }
        }

        let dropped = Cell::new(false);
        {
            let backend: Box<dyn BackendInterface + '_> =
                Box::new(DroppableBackend { dropped: &dropped });
            assert!(backend.is_available());
            assert!(!dropped.get());
        }
        assert!(dropped.get());
    }
}
