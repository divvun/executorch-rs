//! Literal port of runtime/core/result.h.
//!
//! Result type to be used in conjunction with ExecuTorch Error type.
//!
//! The C++ `Result<T>` is a tagged union of a value or a non-Ok `Error`. Per
//! the wave-2 type mapping it is modeled as `core::result::Result<T, Error>`;
//! the C++ member functions that call sites use (`ok()`, `error()`, `get()`,
//! `operator*`, `operator->`) are ported as an extension trait plus a
//! construction helper mirroring the `Result(Error)` ctor invariant.

use crate::runtime::core::error::Error;

/// Result type wrapping either a value of type T or an error.
///
/// Example use case:
/// ```ignore
///   fn get_op(opcode: i32) -> Result<OpFn> {
///     if is_valid_op_code(opcode) {
///       return Ok(op_fns[opcode]);
///     }
///     result_from_error(Error::NotFound)
///   }
///
///   fn use_op(opcode: i32) -> Error {
///     let op = get_op(opcode);
///     if !op.ok() {
///       return op.error();
///     }
///     print(op.get().to_string());
///     execute(op.get());
///     Error::Ok
///   }
/// ```
// [spec:et:def:result.executorch.runtime.result]
pub type Result<T> = core::result::Result<T, Error>;

/// Creates a Result object from an Error.
///
/// To preserve the invariant that `(result.error() == Error::Ok) ==
/// result.ok()`, an `error` parameter value of `Error::Ok` will be converted to
/// a non-Ok value.
// [spec:et:def:result.executorch.runtime.result.result-fn]
// [spec:et:sem:result.executorch.runtime.result.result-fn]
pub fn result_from_error<T>(error: Error) -> Result<T> {
    if error == Error::Ok {
        // PORT-NOTE: The C++ ctor emits `ET_LOG(Debug, ...)` here; `et_log!`
        // is owned by the platform-core group and is still a stub, so the log
        // is elided until it lands.
        Err(Error::Internal)
    } else {
        Err(error)
    }
}

/// Ported member functions of the C++ `Result<T>` class as an extension trait
/// over `core::result::Result<T, Error>`.
pub trait ResultExt<T> {
    /// `value_type` member for generic programming.
    // [spec:et:def:result.executorch.runtime.result.value-type]
    type ValueType;

    /// Returns true if this Result has a value.
    ///
    /// If true, it is guaranteed that `error()` will return `Error::Ok`.
    /// If false, it is guaranteed that `error()` will not return `Error::Ok`.
    // [spec:et:def:result.executorch.runtime.result.ok-fn]
    // [spec:et:sem:result.executorch.runtime.result.ok-fn]
    #[must_use]
    fn ok(&self) -> bool;

    /// Returns the error code of this Result.
    ///
    /// If this returns `Error::Ok`, it is guaranteed that `ok()` will return
    /// true. If this does not return `Error::Ok`, it is guaranteed that `ok()`
    /// will return false.
    // [spec:et:def:result.executorch.runtime.result.error-fn]
    // [spec:et:sem:result.executorch.runtime.result.error-fn]
    #[must_use]
    fn error(&self) -> Error;

    /// Returns a reference to the Result's value; longhand for `operator*()`.
    ///
    /// Only legal to call if `ok()` returns true.
    // [spec:et:def:result.executorch.runtime.result.get-fn]
    // [spec:et:sem:result.executorch.runtime.result.get-fn]
    fn get(&self) -> &T;

    /// Mutable longhand accessor for the stored value.
    ///
    /// Only legal to call if `ok()` returns true.
    fn get_mut(&mut self) -> &mut T;

    /// Returns a reference to the Result's value; shorthand for `get()`.
    ///
    /// Only legal to call if `ok()` returns true.
    // [spec:et:def:result.executorch.runtime.result-t.operator-fn]
    // [spec:et:sem:result.executorch.runtime.result-t.operator-fn]
    // [spec:et:def:result.executorch.runtime.result.operator-fn]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn]
    fn deref(&self) -> &T;

    /// Mutable shorthand accessor.
    ///
    /// Only legal to call if `ok()` returns true.
    fn deref_mut(&mut self) -> &mut T;
}

impl<T> ResultExt<T> for Result<T> {
    type ValueType = T;

    // [spec:et:def:result.executorch.runtime.result.ok-fn]
    // [spec:et:sem:result.executorch.runtime.result.ok-fn]
    fn ok(&self) -> bool {
        self.is_ok()
    }

    // [spec:et:def:result.executorch.runtime.result.error-fn]
    // [spec:et:sem:result.executorch.runtime.result.error-fn]
    fn error(&self) -> Error {
        match self {
            core::result::Result::Ok(_) => Error::Ok,
            core::result::Result::Err(error_) => *error_,
        }
    }

    // [spec:et:def:result.executorch.runtime.result.get-fn]
    // [spec:et:sem:result.executorch.runtime.result.get-fn]
    fn get(&self) -> &T {
        check_ok(self);
        // Only reached when `hasValue_` (Ok); mirrors returning `&value_`.
        match self {
            core::result::Result::Ok(value_) => value_,
            core::result::Result::Err(_) => unreachable!(),
        }
    }

    fn get_mut(&mut self) -> &mut T {
        check_ok(self);
        match self {
            core::result::Result::Ok(value_) => value_,
            core::result::Result::Err(_) => unreachable!(),
        }
    }

    // [spec:et:def:result.executorch.runtime.result.operator-fn]
    // [spec:et:sem:result.executorch.runtime.result.operator-fn]
    fn deref(&self) -> &T {
        self.get()
    }

    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

/// Panics if `ok()` would return false.
// [spec:et:def:result.executorch.runtime.result.check-ok-fn]
// [spec:et:sem:result.executorch.runtime.result.check-ok-fn]
fn check_ok<T>(result: &Result<T>) {
    // PORT-NOTE: C++ evaluates `ET_CHECK(hasValue_)`, which logs and aborts on
    // failure. `ET_CHECK` is owned by the platform assert group (still a stub);
    // `assert!` is used here as the closest fatal, non-recoverable stand-in
    // until it lands.
    assert!(result.is_ok());
}

// PORT-NOTE: result.h has no dedicated *_test.cpp; the class is exercised all
// over the runtime tests (see error.rs's ported error-handling tests). These
// focused tests pin the out-of-class `Result<T>::operator*` / `operator->`
// definitions (ported as `deref` / `deref_mut` shorthands over `get`).
#[cfg(test)]
mod tests {
    use super::*;

    // The out-of-line `operator*()` / `operator->()` bodies: CheckOk() then
    // return the stored value (`&value_`), in both const and mutable flavors —
    // writes through the mutable accessor are visible through the const one.
    // [spec:et:sem:result.executorch.runtime.result-t.operator-fn/test]
    #[test]
    fn result_deref_returns_stored_value() {
        let mut res: Result<i32> = Ok(42);
        assert!(ResultExt::ok(&res));
        assert_eq!(*ResultExt::deref(&res), 42);
        *ResultExt::deref_mut(&mut res) = 7;
        assert_eq!(*ResultExt::deref(&res), 7);
        // Shorthand and longhand return the same reference.
        assert!(core::ptr::eq(ResultExt::deref(&res), ResultExt::get(&res)));
    }

    // The `CheckOk()` arm of `operator*`: dereferencing a non-Ok Result trips
    // `ET_CHECK(hasValue_)` (a fatal check; `assert!` in the Rust port).
    // [spec:et:sem:result.executorch.runtime.result-t.operator-fn/test]
    #[test]
    #[should_panic]
    fn result_deref_on_error_check_fails() {
        let res: Result<i32> = result_from_error(Error::NotFound);
        let _ = ResultExt::deref(&res);
    }
}
