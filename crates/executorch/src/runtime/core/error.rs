//! Literal port of runtime/core/error.h.

// Alias error code integral type to minimal platform width (32-bits for now).
// [spec:et:def:error.executorch.runtime.error-code-t]
pub type ErrorCodeT = u32;

/// ExecuTorch Error type.
// [spec:et:def:error.executorch.runtime.error]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /*
     * System errors.
     */
    /// Status indicating a successful operation.
    Ok = 0x00,

    /// An internal error occurred.
    Internal = 0x01,

    /// Status indicating the executor is in an invalid state for a target
    /// operation
    InvalidState = 0x2,

    /// Status indicating there are no more steps of execution to run
    EndOfMethod = 0x03,

    /// Status indicating a resource has already been loaded.
    AlreadyLoaded = 0x04,

    /*
     * Logical errors.
     */
    /// Operation is not supported in the current context.
    NotSupported = 0x10,

    /// Operation is not yet implemented.
    NotImplemented = 0x11,

    /// User provided an invalid argument.
    InvalidArgument = 0x12,

    /// Object is an invalid type for the operation.
    InvalidType = 0x13,

    /// Operator(s) missing in the operator registry.
    OperatorMissing = 0x14,

    /// Registration error: Exceeding the maximum number of kernels.
    RegistrationExceedingMaxKernels = 0x15,

    /// Registration error: The kernel is already registered.
    RegistrationAlreadyRegistered = 0x16,

    /*
     * Resource errors.
     */
    /// Requested resource could not be found.
    NotFound = 0x20,

    /// Could not allocate the requested memory.
    MemoryAllocationFailed = 0x21,

    /// Could not access a resource.
    AccessFailed = 0x22,

    /// Error caused by the contents of a program.
    InvalidProgram = 0x23,

    /// Error caused by the contents of external data.
    InvalidExternalData = 0x24,

    /// Does not have enough resources to perform the requested operation.
    OutOfResources = 0x25,

    /*
     * Delegate errors.
     */
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = 0x30,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = 0x31,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = 0x32,
}

// Stringify the Error enum.
// [spec:et:def:error.executorch.runtime.to-string-fn]
// [spec:et:sem:error.executorch.runtime.to-string-fn]
pub const fn to_string(error: Error) -> &'static str {
    match error {
        Error::Ok => "Error::Ok",
        Error::Internal => "Error::Internal",
        Error::InvalidState => "Error::InvalidState",
        Error::EndOfMethod => "Error::EndOfMethod",
        Error::AlreadyLoaded => "Error::AlreadyLoaded",
        Error::NotSupported => "Error::NotSupported",
        Error::NotImplemented => "Error::NotImplemented",
        Error::InvalidArgument => "Error::InvalidArgument",
        Error::InvalidType => "Error::InvalidType",
        Error::OperatorMissing => "Error::OperatorMissing",
        Error::NotFound => "Error::NotFound",
        Error::MemoryAllocationFailed => "Error::MemoryAllocationFailed",
        Error::AccessFailed => "Error::AccessFailed",
        Error::InvalidProgram => "Error::InvalidProgram",
        Error::InvalidExternalData => "Error::InvalidExternalData",
        Error::OutOfResources => "Error::OutOfResources",
        Error::DelegateInvalidCompatibility => "Error::DelegateInvalidCompatibility",
        Error::DelegateMemoryAllocationFailed => "Error::DelegateMemoryAllocationFailed",
        Error::DelegateInvalidHandle => "Error::DelegateInvalidHandle",
        Error::RegistrationExceedingMaxKernels => "Error::RegistrationExceedingMaxKernels",
        Error::RegistrationAlreadyRegistered => "Error::RegistrationAlreadyRegistered",
        // PORT-NOTE: The C++ `switch` has a `default` arm returning
        // "Error::Unknown" for any bit pattern that is not a declared
        // enumerator. A Rust `#[repr(u8)]` enum can only hold declared
        // discriminants, so the match is exhaustive and the "Error::Unknown"
        // fallthrough is unreachable and thus omitted.
    }
}

/// The `Result<T>` alias lives in `crate::runtime::core::result`; re-exported
/// here so call sites can name it via the error module as the C++ `error.h`
/// consumers do through `result.h`.
pub use crate::runtime::core::result::Result;

/// Models the C++ implicit conversion path taken when
/// `ET_CHECK_OR_RETURN_ERROR` `return`s an `Error` from a function. If the
/// function returns `Error`, the value is returned as-is; if it returns
/// `Result<T>`, the compiler invokes the implicit `Result(Error)` ctor (which
/// maps `Error::Ok` to `Error::Internal`). This trait lets the single check
/// macro serve both return types, as the C++ overload set does.
pub trait ReturnableFromError {
    fn returnable_from_error(error: Error) -> Self;
}

impl ReturnableFromError for Error {
    fn returnable_from_error(error: Error) -> Self {
        error
    }
}

impl<T> ReturnableFromError for Result<T> {
    fn returnable_from_error(error: Error) -> Self {
        crate::runtime::core::result::result_from_error(error)
    }
}

/// If `cond__` is false, log the specified message and return the specified
/// Error from the current function, whose return type must be either
/// `crate::runtime::core::error::Error` or a `Result<T>` over it (matching the
/// C++ overloads reachable through the implicit `Result(Error)` ctor).
///
/// `error__` is the Error enum value to return without the `Error::` prefix,
/// like `InvalidArgument`.
// PORT-NOTE: references `crate::et_log!`, owned by the platform-core group
// (runtime/platform/log.rs), which is still a stub at time of writing.
#[macro_export]
macro_rules! et_check_or_return_error {
    ($cond:expr, $error:ident, $($message:tt)*) => {{
        if !($cond) {
            $crate::et_log!(Error, $($message)*);
            return $crate::runtime::core::error::ReturnableFromError::returnable_from_error(
                $crate::runtime::core::error::Error::$error,
            );
        }
    }};
}

/// A convenience macro to be used in utility functions that check whether input
/// tensor(s) are valid, which are expected to return a boolean. Checks whether
/// `cond` is true; if not, log the failed check with `message` and return false.
#[macro_export]
macro_rules! et_check_or_return_false {
    ($cond:expr, $($message:tt)*) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $crate::__et_first_fmt!($($message)*)),
                ::core::stringify!($cond),
            );
            return false;
        }
    }};
}

/// Internal helper: extracts the leading format-string literal from the
/// message token stream so `et_check_or_return_false!` can prepend the failed
/// condition, mirroring the C++ `"Check failed (%s): " message__` concatenation.
#[doc(hidden)]
#[macro_export]
macro_rules! __et_first_fmt {
    ($fmt:literal $(, $($rest:tt)*)?) => {
        $fmt
    };
}

/// If `error__` is not `Error::Ok`, optionally log a message and return the
/// error from the current function, which must be of return type
/// `crate::runtime::core::error::Error`.
#[macro_export]
macro_rules! et_check_ok_or_return_error {
    ($error:expr $(,)?) => {{
        let et_error__ = ($error);
        if et_error__ != $crate::runtime::core::error::Error::Ok {
            return et_error__;
        }
    }};
    ($error:expr, $($message:tt)*) => {{
        let et_error__ = ($error);
        if et_error__ != $crate::runtime::core::error::Error::Ok {
            $crate::et_log!(Error, $($message)*);
            return et_error__;
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::result::{Result, ResultExt, result_from_error};

    // static void* test_ptr = (void*)0xDEADBEEF;
    fn test_ptr() -> *const core::ffi::c_void {
        0xDEADBEEFusize as *const core::ffi::c_void
    }

    fn get_abs(num: i64) -> Result<u64> {
        if num >= 0 {
            Ok(num as u64)
        } else {
            result_from_error(Error::InvalidArgument)
        }
    }

    // Helpers that use ET_UNWRAP in expression position (result.h). No
    // `et_unwrap!` macro is ported; the `?` operator reproduces ET_UNWRAP's
    // early-return-on-error expression contract, so these helpers stand in for
    // the C++ `double_abs_unwrap` / `_with_message`.
    fn double_abs_unwrap(num: i64) -> Result<u64> {
        let value: u64 = get_abs(num)?;
        Ok(value * 2)
    }

    fn double_abs_unwrap_with_message(num: i64) -> Result<u64> {
        let value: u64 = get_abs(num)?;
        Ok(value * 2)
    }

    fn get_op_name(op: i64) -> Result<String> {
        let abs_result = get_abs(op);
        if !ResultExt::ok(&abs_result) {
            return Err(abs_result.error());
        }
        let unsigned_op = *abs_result.get();

        match unsigned_op {
            0 => Ok(String::from("Zero")),
            1 => Ok(String::from("One")),
            _ => result_from_error(Error::Internal),
        }
    }

    fn get_ptr(value: i64) -> Result<*const core::ffi::c_void> {
        match value {
            0 => Ok(core::ptr::null()),
            1 => Ok(test_ptr()),
            _ => result_from_error(Error::InvalidArgument),
        }
    }

    // class Uncopiable { explicit Uncopiable(uint32_t value); ... }
    struct Uncopiable {
        value_: u32,
    }
    impl Uncopiable {
        fn new(value: u32) -> Self {
            Uncopiable { value_: value }
        }
        fn get_value(&self) -> u32 {
            self.value_
        }
    }

    fn get_no_copy(value: u32) -> Result<Uncopiable> {
        Ok(Uncopiable::new(value))
    }

    // A non-trivially-movable type. In Rust, moving a Box transfers ownership;
    // the double-free the C++ guards against cannot occur, but the move-out
    // semantics are preserved (source no longer points at the buffer).
    struct Movable {
        buffer_: *mut core::ffi::c_void,
    }
    impl Movable {
        fn new(nbytes: usize) -> Self {
            let buf = unsafe {
                let layout = core::alloc::Layout::from_size_align(nbytes, 1).unwrap();
                std::alloc::alloc(layout) as *mut core::ffi::c_void
            };
            Movable { buffer_: buf }
        }
        fn buffer(&self) -> *const core::ffi::c_void {
            self.buffer_
        }
        fn take_buffer(&mut self) -> *mut core::ffi::c_void {
            let b = self.buffer_;
            self.buffer_ = core::ptr::null_mut();
            b
        }
    }
    impl Drop for Movable {
        fn drop(&mut self) {
            if !self.buffer_.is_null() {
                // Size is irrelevant to dealloc here; the original test frees a
                // small allocation. Use a matching layout of size 2.
                unsafe {
                    let layout = core::alloc::Layout::from_size_align(2, 1).unwrap();
                    std::alloc::dealloc(self.buffer_ as *mut u8, layout);
                }
            }
        }
    }

    // [spec:et:sem:error.executorch.runtime.to-string-fn/test]
    #[test]
    fn error_handling_test_to_string_known_error() {
        assert_eq!(to_string(Error::Ok), "Error::Ok");
        assert_eq!(to_string(Error::Internal), "Error::Internal");
        assert_eq!(to_string(Error::InvalidState), "Error::InvalidState");
        assert_eq!(to_string(Error::EndOfMethod), "Error::EndOfMethod");
        assert_eq!(to_string(Error::AlreadyLoaded), "Error::AlreadyLoaded");
        assert_eq!(to_string(Error::NotSupported), "Error::NotSupported");
        assert_eq!(to_string(Error::NotImplemented), "Error::NotImplemented");
        assert_eq!(to_string(Error::InvalidArgument), "Error::InvalidArgument");
        assert_eq!(to_string(Error::InvalidType), "Error::InvalidType");
        assert_eq!(to_string(Error::OperatorMissing), "Error::OperatorMissing");
        assert_eq!(
            to_string(Error::RegistrationExceedingMaxKernels),
            "Error::RegistrationExceedingMaxKernels"
        );
        assert_eq!(
            to_string(Error::RegistrationAlreadyRegistered),
            "Error::RegistrationAlreadyRegistered"
        );
        assert_eq!(to_string(Error::NotFound), "Error::NotFound");
        assert_eq!(
            to_string(Error::MemoryAllocationFailed),
            "Error::MemoryAllocationFailed"
        );
        assert_eq!(to_string(Error::AccessFailed), "Error::AccessFailed");
        assert_eq!(to_string(Error::InvalidProgram), "Error::InvalidProgram");
        assert_eq!(
            to_string(Error::InvalidExternalData),
            "Error::InvalidExternalData"
        );
        assert_eq!(to_string(Error::OutOfResources), "Error::OutOfResources");
        assert_eq!(
            to_string(Error::DelegateInvalidCompatibility),
            "Error::DelegateInvalidCompatibility"
        );
        assert_eq!(
            to_string(Error::DelegateMemoryAllocationFailed),
            "Error::DelegateMemoryAllocationFailed"
        );
        assert_eq!(
            to_string(Error::DelegateInvalidHandle),
            "Error::DelegateInvalidHandle"
        );
    }

    // [spec:et:sem:error.executorch.runtime.to-string-fn/test]
    //
    // PORT-NOTE: the C++ builds `static_cast<Error>(0xFF)` — an out-of-range bit
    // pattern — and expects the `default` "Error::Unknown" arm. A Rust
    // `#[repr(u8)]` enum cannot legally hold `0xFF`; constructing it is UB and
    // `to_string`'s match is exhaustive with no Unknown arm (see the PORT-NOTE in
    // `to_string`). This test therefore has no safe Rust surface and is ignored.
    #[test]
    #[ignore]
    fn error_handling_test_to_string_unknown_error() {
        let unknown = unsafe { core::mem::transmute::<u8, Error>(0xFF) };
        let result = to_string(unknown);
        assert_eq!(result, "Error::Unknown");
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.error-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.get-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn/test]
    #[test]
    fn error_handling_test_result_basic() {
        let r: Result<u32> = Ok(1);
        assert!(ResultExt::ok(&r));
        assert_eq!(r.error(), Error::Ok);
        assert_eq!(*r.get(), 1);
        assert_eq!(*r.deref(), 1);
    }

    // [spec:et:sem:result.executorch.runtime.result.result-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.error-fn/test]
    #[test]
    fn error_handling_test_ok_error_not_possible() {
        crate::runtime::platform::runtime::runtime_init();
        let r: Result<u32> = result_from_error(Error::Ok);
        assert!(!ResultExt::ok(&r));
        assert_ne!(r.error(), Error::Ok);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.get-fn/test]
    #[test]
    fn error_handling_test_result_with_primitive() {
        let res = get_abs(100);
        assert!(ResultExt::ok(&res));
        assert_eq!(res.error(), Error::Ok);

        let mut unsigned_result = *res.get();
        assert_eq!(unsigned_result, 100);
        unsigned_result = *res.deref();
        assert_eq!(unsigned_result, 100);

        let res2 = get_abs(-3);
        assert!(!ResultExt::ok(&res2));
        assert_eq!(res2.error(), Error::InvalidArgument);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.get-fn/test]
    #[test]
    fn error_handling_test_result_with_compound() {
        let res = get_op_name(0);
        assert!(ResultExt::ok(&res));
        assert_eq!(res.error(), Error::Ok);
        assert_eq!(res.get(), "Zero");
        assert_eq!(res.deref(), "Zero");

        let res2 = get_op_name(1);
        assert!(ResultExt::ok(&res2));
        assert_eq!(res2.error(), Error::Ok);
        assert_eq!(res2.get(), "One");
        assert_eq!(res2.deref(), "One");

        let res3 = get_op_name(2);
        assert!(!ResultExt::ok(&res3));
        assert_eq!(res3.error(), Error::Internal);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.get-fn/test]
    #[test]
    fn error_handling_test_result_with_pointer() {
        let res = get_ptr(0);
        assert!(ResultExt::ok(&res));
        assert_eq!(res.error(), Error::Ok);
        assert_eq!(*res.get(), core::ptr::null());
        assert_eq!(*res.deref(), core::ptr::null());

        let res2 = get_ptr(1);
        assert!(ResultExt::ok(&res2));
        assert_eq!(res2.error(), Error::Ok);
        assert_eq!(*res2.get(), test_ptr());
        assert_eq!(*res2.deref(), test_ptr());

        let res3 = get_ptr(2);
        assert!(!ResultExt::ok(&res3));
        assert_eq!(res3.error(), Error::InvalidArgument);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.error-fn/test]
    #[test]
    fn error_handling_test_result_unwrap() {
        let res = get_op_name(-1);
        assert!(!ResultExt::ok(&res));
        assert_eq!(res.error(), Error::InvalidArgument);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn/test]
    //
    // PORT-NOTE: pins the ET_UNWRAP expression-form contract. The `et_unwrap!`
    // macro is not ported in the Rust runtime; the `?` operator in
    // `double_abs_unwrap` reproduces its early-return-on-error semantics.
    #[test]
    fn error_handling_test_et_unwrap_expression_form() {
        let ok = double_abs_unwrap(5);
        assert!(ResultExt::ok(&ok));
        assert_eq!(*ok.deref(), 10u64);

        let err = double_abs_unwrap(-1);
        assert!(!ResultExt::ok(&err));
        assert_eq!(err.error(), Error::InvalidArgument);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn/test]
    #[test]
    fn error_handling_test_et_unwrap_expression_form_with_message() {
        crate::runtime::platform::runtime::runtime_init();

        let ok = double_abs_unwrap_with_message(4);
        assert!(ResultExt::ok(&ok));
        assert_eq!(*ok.deref(), 8u64);

        let err = double_abs_unwrap_with_message(-2);
        assert!(!ResultExt::ok(&err));
        assert_eq!(err.error(), Error::InvalidArgument);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.get-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn/test]
    #[test]
    fn error_handling_test_result_no_copy() {
        let mut res = get_no_copy(2);
        assert!(ResultExt::ok(&res));
        assert_eq!(res.error(), Error::Ok);
        assert_eq!(res.get().get_value(), 2);
        assert_eq!(res.deref().get_value(), 2);

        let mut res2 = res;
        assert!(ResultExt::ok(&res2));
        assert_eq!(res2.error(), Error::Ok);
        assert_eq!(res2.get().get_value(), 2);
        assert_eq!(res2.deref().get_value(), 2);

        let uc: &Uncopiable = res2.deref();
        assert_eq!(uc.get_value(), 2);
    }

    // [spec:et:sem:result.executorch.runtime.result.ok-fn/test]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn/test]
    #[test]
    fn error_handling_test_result_move() {
        crate::runtime::platform::runtime::runtime_init();

        let mut res: Result<Movable> = Ok(Movable::new(2));
        assert!(ResultExt::ok(&res));
        assert_eq!(res.error(), Error::Ok);
        assert_ne!(res.get().buffer(), core::ptr::null());
        assert_ne!(res.deref().buffer(), core::ptr::null());

        let buffer = res.deref().buffer();

        // Move the value out of the buffer field.
        let taken = res.deref_mut().take_buffer();
        // The target should point to the same buffer as the source originally did.
        assert_eq!(taken as *const core::ffi::c_void, buffer);
        // The source inside the Result should no longer point to the buffer.
        assert_eq!(res.deref().buffer(), core::ptr::null());

        // Free the taken buffer to match the C++ Movable destructor.
        unsafe {
            let layout = core::alloc::Layout::from_size_align(2, 1).unwrap();
            std::alloc::dealloc(taken as *mut u8, layout);
        }
    }
}
