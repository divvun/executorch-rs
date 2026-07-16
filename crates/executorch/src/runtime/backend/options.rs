//! Literal port of runtime/backend/options.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::span::Span;

pub const K_MAX_OPTION_KEY_LENGTH: usize = 64;
pub const K_MAX_OPTION_VALUE_LENGTH: usize = 256;

// All string keys and values must have static storage duration (string
// literals, static const char arrays, or global constants). The BackendOptions
// class does NOT take ownership of strings.
//
// PORT-NOTE: C++ `OptionValue = std::variant<bool, int, std::array<char,
// kMaxOptionValueLength>>`. Modeled as a Rust enum with the same three
// alternatives, in the same order. `int` maps to `i32`.
#[derive(Clone, Copy)]
pub enum OptionValue {
    Bool(bool),
    Int(i32),
    CharArray([core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH]),
}

// [spec:et:def:options.executorch.runtime.backend-option]
#[derive(Clone, Copy)]
pub struct BackendOption {
    // key is the name of the backend option, like num_threads, enable_profiling,
    // etc
    pub key: [core::ffi::c_char; K_MAX_OPTION_KEY_LENGTH],
    // value is the value of the backend option, like 4, true, etc
    pub value: OptionValue,
}

impl BackendOption {
    // Default-initialized BackendOption: `key` zeroed (`{}` in C++), `value`
    // default-constructed to the variant's first alternative (`bool` == false).
    pub fn new() -> Self {
        BackendOption {
            key: [0; K_MAX_OPTION_KEY_LENGTH],
            value: OptionValue::Bool(false),
        }
    }
}

impl Default for BackendOption {
    fn default() -> Self {
        BackendOption::new()
    }
}

/// A template class for storing and managing backend-specific configuration
/// options.
///
/// This class provides a type-safe way to store key-value pairs for backend
/// configuration, with compile-time capacity limits and runtime type checking.
/// It supports bool, int, and const char* value types.
///
/// `MaxCapacity` The maximum number of options that can be stored.
// [spec:et:def:options.executorch.runtime.backend-options]
pub struct BackendOptions<const MAX_CAPACITY: usize> {
    /// Storage for backend options
    options_: [BackendOption; MAX_CAPACITY],
    /// Current number of options
    size_: usize,
}

impl<const MAX_CAPACITY: usize> Clone for BackendOptions<MAX_CAPACITY> {
    /// Copy constructor
    // [spec:et:def:options.executorch.runtime.backend-options.backend-options-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.backend-options-fn]
    fn clone(&self) -> Self {
        let mut new = BackendOptions {
            options_: [BackendOption::new(); MAX_CAPACITY],
            size_: self.size_,
        };
        for i in 0..new.size_ {
            new.options_[i] = self.options_[i];
        }
        new
    }

    /// Copy assignment operator
    // [spec:et:def:options.executorch.runtime.backend-options.operator-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.operator-fn]
    fn clone_from(&mut self, other: &Self) {
        if !core::ptr::eq(self, other) {
            self.size_ = other.size_;
            for i in 0..self.size_ {
                self.options_[i] = other.options_[i];
            }
        }
    }
}

impl<const MAX_CAPACITY: usize> BackendOptions<MAX_CAPACITY> {
    /// Default constructor - initializes with zero options.
    pub fn new() -> Self {
        BackendOptions {
            options_: [BackendOption::new(); MAX_CAPACITY],
            size_: 0,
        }
    }

    /// Returns a mutable view of all stored options as a Span.
    ///
    /// @return A mutable Span containing all BackendOption entries
    // [spec:et:def:options.executorch.runtime.backend-options.view-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.view-fn]
    pub fn view(&mut self) -> Span<BackendOption> {
        Span::from_raw_parts(self.options_.as_mut_ptr(), self.size_)
    }

    /// Sets a boolean option value for the given key.
    /// If the key already exists, updates its value. Otherwise, adds a new option.
    ///
    /// `key` The option key (must be a string literal or array).
    /// `value` The boolean value to set.
    /// @return Error::Ok on success, Error::InvalidArgument if storage is full
    //
    // PORT-NOTE: C++ `template <size_t N> set_option(const char (&key)[N],
    // bool value)` with `static_assert(N <= kMaxOptionKeyLength)`. `key` is a
    // fixed-size char array reference; in Rust the array length is passed as a
    // const generic `N` and the static_assert is a const-evaluated assertion.
    pub fn set_option_bool<const N: usize>(
        &mut self,
        key: &[core::ffi::c_char; N],
        value: bool,
    ) -> Error {
        const { assert!(N <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        self.set_option_impl(key.as_ptr(), OptionValue::Bool(value))
    }

    /// Sets an integer option value for the given key.
    /// If the key already exists, updates its value. Otherwise, adds a new option.
    ///
    /// `key` The option key (must be a string literal or array).
    /// `value` The integer value to set.
    /// @return Error::Ok on success, Error::InvalidArgument if storage is full
    pub fn set_option_int<const N: usize>(
        &mut self,
        key: &[core::ffi::c_char; N],
        value: i32,
    ) -> Error {
        const { assert!(N <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        self.set_option_impl(key.as_ptr(), OptionValue::Int(value))
    }

    /// Sets a string option value for the given key.
    /// If the key already exists, updates its value. Otherwise, adds a new option.
    ///
    /// Note: The string value must have static storage duration. This class does
    /// NOT take ownership of the string - it only stores the pointer.
    ///
    /// `key` The option key (must be a string literal or array).
    /// `value` The string value to set (must have static storage duration).
    /// @return Error::Ok on success, Error::InvalidArgument if storage is full
    // [spec:et:def:options.executorch.runtime.backend-options.set-option-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn]
    pub fn set_option_str<const N: usize>(
        &mut self,
        key: &[core::ffi::c_char; N],
        value: *const core::ffi::c_char,
    ) -> Error {
        const { assert!(N <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        // Create a fixed-size array and copy the string
        let mut arr: [core::ffi::c_char; K_MAX_OPTION_VALUE_LENGTH] =
            [0; K_MAX_OPTION_VALUE_LENGTH];
        unsafe {
            libc::strncpy(arr.as_mut_ptr(), value, K_MAX_OPTION_VALUE_LENGTH - 1);
        }
        arr[K_MAX_OPTION_VALUE_LENGTH - 1] = 0; // Ensure null termination
        self.set_option_impl(key.as_ptr(), OptionValue::CharArray(arr))
    }

    /// Retrieves an option value by key and type.
    ///
    /// `key` The option key to look up.
    /// `out` Reference to store the retrieved value.
    /// @return Error::Ok if found and type matches, Error::NotFound if key
    /// doesn't exist, Error::InvalidArgument if type doesn't match.
    //
    // PORT-NOTE: C++ `template <typename T, size_t KeyLen> get_option(const
    // char (&key)[KeyLen], T& out)`. `T` (bool/int/const char*) is resolved at
    // compile time via `if constexpr`; in Rust the three instantiations become
    // three explicit methods (`get_option_bool`/`get_option_int`/
    // `get_option_str`) that each carry the same key-scan + variant-match
    // control flow.
    // [spec:et:def:options.executorch.runtime.backend-options.get-option-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn]
    pub fn get_option_bool<const KEY_LEN: usize>(
        &self,
        key: &[core::ffi::c_char; KEY_LEN],
        out: &mut bool,
    ) -> Error {
        const { assert!(KEY_LEN <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.options_[i].key.as_ptr(), key.as_ptr()) } == 0 {
                // Default handling for bool/int
                if let OptionValue::Bool(val) = &self.options_[i].value {
                    *out = *val;
                    return Error::Ok;
                }
                return Error::InvalidArgument;
            }
        }
        Error::NotFound
    }

    // [spec:et:def:options.executorch.runtime.backend-options.get-option-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn]
    pub fn get_option_int<const KEY_LEN: usize>(
        &self,
        key: &[core::ffi::c_char; KEY_LEN],
        out: &mut i32,
    ) -> Error {
        const { assert!(KEY_LEN <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.options_[i].key.as_ptr(), key.as_ptr()) } == 0 {
                if let OptionValue::Int(val) = &self.options_[i].value {
                    *out = *val;
                    return Error::Ok;
                }
                return Error::InvalidArgument;
            }
        }
        Error::NotFound
    }

    // [spec:et:def:options.executorch.runtime.backend-options.get-option-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn]
    pub fn get_option_str<const KEY_LEN: usize>(
        &self,
        key: &[core::ffi::c_char; KEY_LEN],
        out: &mut *const core::ffi::c_char,
    ) -> Error {
        const { assert!(KEY_LEN <= K_MAX_OPTION_KEY_LENGTH, "Option key is too long") };
        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.options_[i].key.as_ptr(), key.as_ptr()) } == 0 {
                // Special handling for string (convert array to const char*)
                if let OptionValue::CharArray(arr) = &self.options_[i].value {
                    *out = arr.as_ptr(); // Return pointer to stored array
                    return Error::Ok;
                }
                return Error::InvalidArgument;
            }
        }
        Error::NotFound
    }

    /// Internal implementation for setting option values.
    /// Handles both updating existing options and adding new ones.
    ///
    /// `key` The option key.
    /// `value` The value to set.
    /// @return Error::Ok on success, Error::InvalidArgument if storage is full
    // [spec:et:def:options.executorch.runtime.backend-options.set-option-impl-fn]
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-impl-fn]
    fn set_option_impl(&mut self, key: *const core::ffi::c_char, value: OptionValue) -> Error {
        // Update existing if found
        for i in 0..self.size_ {
            if unsafe { libc::strcmp(self.options_[i].key.as_ptr(), key) } == 0 {
                self.options_[i].value = value;
                return Error::Ok;
            }
        }
        if self.size_ < MAX_CAPACITY {
            let mut new_option = BackendOption::new();
            let key_len: usize = unsafe { libc::strlen(key) };
            let copy_len: usize = core::cmp::min(key_len, K_MAX_OPTION_KEY_LENGTH - 1);
            unsafe {
                libc::memcpy(
                    new_option.key.as_mut_ptr() as *mut core::ffi::c_void,
                    key as *const core::ffi::c_void,
                    copy_len,
                );
            }
            new_option.key[copy_len] = 0;
            new_option.value = value; // Restored value assignment
            self.options_[self.size_] = new_option; // Store option and increment size
            self.size_ += 1;
            return Error::Ok;
        }
        Error::InvalidArgument
    }
}

impl<const MAX_CAPACITY: usize> Default for BackendOptions<MAX_CAPACITY> {
    fn default() -> Self {
        BackendOptions::new()
    }
}

// Literal port of runtime/backend/test/backend_options_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::platform::runtime::runtime_init;

    // C++ passes string-literal keys as `const char (&key)[N]` (N includes the
    // trailing NUL). This helper turns a NUL-terminated byte literal into the
    // equivalent `[c_char; N]` key array the set/get methods expect.
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

    // Test basic string functionality
    // set_option_str forwards to set_option_impl; the first set exercises its
    // add branch and the second (same key) its update-existing branch, both
    // observed through get_option_str.
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-impl-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_handles_string_options() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        // Set and retrieve valid string
        options.set_option_str(&key(b"backend_type\0"), c"GPU".as_ptr());
        let mut result: *const core::ffi::c_char = core::ptr::null();
        assert_eq!(
            options.get_option_str(&key(b"backend_type\0"), &mut result),
            Error::Ok
        );
        assert_eq!(unsafe { libc::strcmp(result, c"GPU".as_ptr()) }, 0);

        // Update existing key
        options.set_option_str(&key(b"backend_type\0"), c"CPU".as_ptr());
        assert_eq!(
            options.get_option_str(&key(b"backend_type\0"), &mut result),
            Error::Ok
        );
        assert_eq!(unsafe { libc::strcmp(result, c"CPU".as_ptr()) }, 0);
    }

    // Test boolean options
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_handles_bool_options() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        options.set_option_bool(&key(b"debug\0"), true);
        let mut debug = false;
        assert_eq!(
            options.get_option_bool(&key(b"debug\0"), &mut debug),
            Error::Ok
        );
        assert!(debug);

        // Test false value
        options.set_option_bool(&key(b"verbose\0"), false);
        assert_eq!(
            options.get_option_bool(&key(b"verbose\0"), &mut debug),
            Error::Ok
        );
        assert!(!debug);
    }

    // Test integer options
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_handles_int_options() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        options.set_option_int(&key(b"num_threads\0"), 256);
        let mut num_threads = 0;
        assert_eq!(
            options.get_option_int(&key(b"num_threads\0"), &mut num_threads),
            Error::Ok
        );
        assert_eq!(num_threads, 256);
    }

    // Test error conditions
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_handles_errors() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        // Non-existent key
        let mut dummy_bool = false;
        assert_eq!(
            options.get_option_bool(&key(b"missing\0"), &mut dummy_bool),
            Error::NotFound
        );

        // Type mismatch
        options.set_option_int(&key(b"threshold\0"), 100);
        let mut dummy_str: *const core::ffi::c_char = core::ptr::null();
        assert_eq!(
            options.get_option_str(&key(b"threshold\0"), &mut dummy_str),
            Error::InvalidArgument
        );

        // Null value handling, should expect failure.
        //
        // PORT-NOTE: C++ `ET_EXPECT_DEATH(options.set_option("nullable",
        // static_cast<const char*>(nullptr)), "")`. The death is a null-pointer
        // dereference inside `strncpy(arr.data(), value, ...)`. The literal Rust
        // port calls `libc::strncpy(arr, value, ...)` with a null `value`, which
        // is a genuine segfault, not a Rust `panic!` — `#[should_panic]` cannot
        // catch it and running it would abort the test process. Ported but
        // ignored; the module reproduces the C++ crash faithfully.
    }

    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    #[test]
    #[ignore = "null-deref segfault in strncpy, not a catchable panic (see PORT-NOTE)"]
    fn backend_options_test_handles_errors_null_value_death() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();
        options.set_option_str(&key(b"nullable\0"), core::ptr::null());
    }

    // Test type-specific keys
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_enforces_key_types() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        // Same key name - later set operations overwrite earlier ones
        options.set_option_bool(&key(b"flag\0"), true);
        options.set_option_int(&key(b"flag\0"), 123); // Overwrites the boolean entry

        let mut bval = false;
        let mut ival = 0;

        // Boolean get should fail - type was overwritten to INT
        assert_eq!(
            options.get_option_bool(&key(b"flag\0"), &mut bval),
            Error::InvalidArgument
        );

        // Integer get should succeed with correct value
        assert_eq!(
            options.get_option_int(&key(b"flag\0"), &mut ival),
            Error::Ok
        );
        assert_eq!(ival, 123);
    }

    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.view-fn/test]
    #[test]
    fn backend_options_test_mutable_option() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        let mut ival = 0;
        options.set_option_int(&key(b"flag\0"), 0);
        // Integer get should succeed with correct value
        assert_eq!(
            options.get_option_int(&key(b"flag\0"), &mut ival),
            Error::Ok
        );
        assert_eq!(ival, 0);

        // options.view()[0].value = 123; // Overwrites the entry
        let view = options.view();
        unsafe { view.index(0) }.value = OptionValue::Int(123);

        // Integer get should succeed with the updated value
        assert_eq!(
            options.get_option_int(&key(b"flag\0"), &mut ival),
            Error::Ok
        );
        assert_eq!(ival, 123);
    }

    // Test copy constructor
    // [spec:et:sem:options.executorch.runtime.backend-options.backend-options-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_copy_constructor() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        // Set up original option
        options.set_option_bool(&key(b"debug\0"), true);

        // Create copy using copy constructor
        let mut copied_options: BackendOptions<5> = options.clone();

        // Verify option was copied correctly
        let mut debug_val = false;
        assert_eq!(
            copied_options.get_option_bool(&key(b"debug\0"), &mut debug_val),
            Error::Ok
        );
        assert!(debug_val);

        // Verify independence - modifying original doesn't affect copy
        options.set_option_bool(&key(b"debug\0"), false);
        assert_eq!(
            copied_options.get_option_bool(&key(b"debug\0"), &mut debug_val),
            Error::Ok
        );
        assert!(debug_val); // Should still be true in copy

        // Verify independence - modifying copy doesn't affect original
        copied_options.set_option_bool(&key(b"debug\0"), false);
        assert_eq!(
            options.get_option_bool(&key(b"debug\0"), &mut debug_val),
            Error::Ok
        );
        assert!(!debug_val); // Should be false in original
    }

    // Test copy assignment operator
    // [spec:et:sem:options.executorch.runtime.backend-options.operator-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.set-option-fn/test]
    // [spec:et:sem:options.executorch.runtime.backend-options.get-option-fn/test]
    #[test]
    fn backend_options_test_copy_assignment_operator() {
        setup();
        let mut options: BackendOptions<5> = BackendOptions::new();

        // Set up original option
        options.set_option_bool(&key(b"enable_profiling\0"), true);

        // Create another options object and assign to it
        let mut assigned_options: BackendOptions<5> = BackendOptions::new();
        assigned_options.set_option_bool(&key(b"temp_option\0"), false); // Add something first

        assigned_options.clone_from(&options);

        // Verify option was copied correctly
        let mut profiling_val = false;
        assert_eq!(
            assigned_options.get_option_bool(&key(b"enable_profiling\0"), &mut profiling_val),
            Error::Ok
        );
        assert!(profiling_val);

        // Verify the temp_option was overwritten (not present in assigned object)
        let mut temp_val = false;
        assert_eq!(
            assigned_options.get_option_bool(&key(b"temp_option\0"), &mut temp_val),
            Error::NotFound
        );

        // Verify independence - modifying original doesn't affect assigned copy
        options.set_option_bool(&key(b"enable_profiling\0"), false);
        assert_eq!(
            assigned_options.get_option_bool(&key(b"enable_profiling\0"), &mut profiling_val),
            Error::Ok
        );
        assert!(profiling_val); // Should still be true in assigned copy
    }
}
