# runtime/core/result.h

> [spec:et:def:result.executorch.runtime.result]
> class Result final {
>   const T& operator*() const&;
>   T& operator*() &;
>   union { T value_; // Used if hasValue_ is true. Error error_; // Used if hasValue_ is false. };
>   const bool hasValue_;
> }

> [spec:et:def:result.executorch.runtime.result-t.operator-fn]
> const T* Result<T>::operator->() const

> [spec:et:sem:result.executorch.runtime.result-t.operator-fn]
> Out-of-line definition of the const `operator->()` for `Result<T>`.
> Calls `CheckOk()` (see `[spec:et:sem:result.executorch.runtime.result.check-ok-fn]`),
> which panics/aborts if the Result does not hold a value (`hasValue_ == false`).
> If the check passes, returns `&value_`: a `const T*` pointing at the stored
> value inside the union. Only legal to call when `ok()` is true; otherwise the
> program aborts before dereference. The mutable overload `T* operator->()`
> behaves identically but returns a non-const pointer.

> [spec:et:def:result.executorch.runtime.result.check-ok-fn]
> void CheckOk() const

> [spec:et:sem:result.executorch.runtime.result.check-ok-fn]
> Private helper that enforces the value invariant before the value is accessed.
> Evaluates `ET_CHECK(hasValue_)`: if `hasValue_` is false (the Result holds an
> Error, not a value), the ET_CHECK fires, which logs and aborts the program
> (a fatal, non-recoverable panic — it does not return or throw). If `hasValue_`
> is true, the function returns normally with no side effects. Used by `get()`,
> `operator*()`, and `operator->()` to guarantee the union's `value_` member is
> the active member before it is read.

> [spec:et:def:result.executorch.runtime.result.error-fn]
> ET_NODISCARD Error error() const

> [spec:et:sem:result.executorch.runtime.result.error-fn]
> Returns the error code of this Result. If `hasValue_` is true (the Result
> holds a value), returns `Error::Ok`. Otherwise returns the stored `error_`
> member. By construction `error_` is never `Error::Ok` (see
> `[spec:et:sem:result.executorch.runtime.result.result-fn]`), so this upholds
> the invariant: `error()` returns `Error::Ok` iff `ok()` returns true. Has no
> side effects; marked ET_NODISCARD so the caller should not ignore the return.

> [spec:et:def:result.executorch.runtime.result.ok-fn]
> ET_NODISCARD bool ok() const

> [spec:et:sem:result.executorch.runtime.result.ok-fn]
> Returns the boolean `hasValue_` member unchanged: true if the Result holds a
> value of type T, false if it holds an Error. No side effects. When true it is
> guaranteed that `error()` returns `Error::Ok`; when false, `error()` returns a
> non-Ok error (see `[spec:et:sem:result.executorch.runtime.result.error-fn]`).
> Marked ET_NODISCARD.

> [spec:et:def:result.executorch.runtime.result.operator-fn]
> const T* operator->() const

> [spec:et:sem:result.executorch.runtime.result.operator-fn]
> Declaration of the pointer-access `operator->()`. The const overload returns a
> `const T*` and the non-const overload returns a `T*`. Both are defined
> out-of-line and behave identically: they call `CheckOk()` (aborting if the
> Result holds no value, see
> `[spec:et:sem:result.executorch.runtime.result.check-ok-fn]`) and then return
> the address of the stored `value_`. See
> `[spec:et:sem:result.executorch.runtime.result-t.operator-fn]` for the const
> definition. Only legal to call when `ok()` is true.

> [spec:et:def:result.executorch.runtime.result.result-fn]
> Result(Error error)

> [spec:et:sem:result.executorch.runtime.result.result-fn]
> Implicit constructor that builds an error-holding Result from an `Error` value.
> Sets `hasValue_` to false. Initializes the union's `error_` member: if the
> passed `error` equals `Error::Ok`, stores `Error::Internal` instead;
> otherwise stores `error` as-is. This preserves the invariant that an
> error-holding Result never carries `Error::Ok`, so that
> `(result.error() == Error::Ok) == result.ok()` always holds. When the input
> was `Error::Ok`, additionally emits a Debug-level log noting that the
> conversion to `Error::Internal` occurred. The `value_` union member is left
> uninitialized (not the active member). Related value-holding constructors
> (from `const T&`, `T&&`, or another `Result&&`) instead set `hasValue_` true
> and construct `value_`.

> [spec:et:def:result.executorch.runtime.result.value-type]
> typedef T value_type

> [spec:et:def:result.executorch.runtime.result.get-fn]
> T& get()

> [spec:et:sem:result.executorch.runtime.result.get-fn]
> Longhand accessor for the stored value; equivalent to `operator*()`. Calls
> `CheckOk()` (aborting if the Result holds no value, see
> `[spec:et:sem:result.executorch.runtime.result.check-ok-fn]`) and then returns
> a reference to the union's `value_` member. The non-const overload returns
> `T&`; a parallel const overload returns `const T&`. Only legal to call when
> `ok()` is true; otherwise the program aborts.

