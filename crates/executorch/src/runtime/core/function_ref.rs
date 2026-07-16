//! Literal port of runtime/core/function_ref.h.
//!
//! An efficient, type-erasing, non-owning reference to a callable. This is
//! intended for use as the type of a function parameter that is not used after
//! the function in question returns.
//!
//! This class does not own the callable, so it is not in general safe to store
//! a FunctionRef.
//!
//! torch::executor: modified from llvm::function_ref
//! - renamed to FunctionRef
//! - removed LLVM_GSL_POINTER and LLVM_LIFETIME_BOUND macro uses
//! - use namespaced internal::remove_cvref_t

// Features from C++20

pub mod internal {
    // [spec:et:def:function-ref.executorch.runtime.internal.remove-cvref]
    // PORT-NOTE: `remove_cvref` strips const/volatile/reference qualifiers from
    // a C++ type at compile time. Rust generics carry no such qualifiers on the
    // erased `Callable` type, so there is no runtime or type-level analogue to
    // port; the SFINAE that used `remove_cvref_t` becomes ordinary trait
    // bounds on the constructor below.
}

/// `FunctionRef<Ret(Params...)>` — modeled for the single-parameter shape that
/// the C++ variadic template expands to. The erased callable is stored as a raw
/// pointer plus a monomorphized trampoline that casts it back to the concrete
/// type and calls it.
// [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params]
pub struct FunctionRef<P, Ret> {
    // [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.callback-fn]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn]
    //
    // The type-erased trampoline stored alongside the erased `callable`
    // pointer; `None` (nullptr) for a default- or nullptr-constructed ref.
    callback: Option<fn(callable: isize, params: P) -> Ret>,
    callable: isize,
}

impl<P, Ret> FunctionRef<P, Ret> {
    // [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]
    //
    // The static trampoline instantiated per concrete `Callable` type. It
    // reinterprets the erased `intptr_t callable` back into a `Callable*`,
    // dereferences it, and invokes it with the forwarded parameters.
    fn callback_fn<Callable: Fn(P) -> Ret>(callable: isize, params: P) -> Ret {
        (unsafe { &*(callable as *const Callable) })(params)
    }

    /// `FunctionRef() = default` — leaves `callback == nullptr`.
    pub const fn new() -> Self {
        FunctionRef {
            callback: None,
            callable: 0,
        }
    }

    /// `FunctionRef(std::nullptr_t)` — produces an empty ref.
    pub const fn from_null() -> Self {
        FunctionRef {
            callback: None,
            callable: 0,
        }
    }

    /// Templated converting constructor that binds the FunctionRef to any
    /// callable without owning it.
    ///
    /// Stores the ADDRESS of the referenced callable object, not a copy.
    /// Because it is non-owning, the referenced object must outlive every use
    /// of this FunctionRef.
    // [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn]
    //
    // PORT-NOTE: the C++ SFINAE constraint (1) — the callable's decayed type is
    // not `FunctionRef` itself, preventing hijack of the copy/move ctors — has
    // no analogue in Rust (this is a distinct `fn`, not an overload set), so it
    // is dropped. Constraint (2) (invocation result convertible to `Ret`) is
    // encoded as the `Fn(P) -> Ret` bound.
    pub fn from_callable<Callable: Fn(P) -> Ret>(callable: &Callable) -> Self {
        FunctionRef {
            callback: Some(Self::callback_fn::<Callable>),
            callable: callable as *const Callable as isize,
        }
    }

    /// Call operator (`const`). Invokes the referenced callable by forwarding
    /// through the stored trampoline.
    ///
    /// Precondition: the FunctionRef must be non-empty (`callback != nullptr`);
    /// calling an empty ref dereferences a null function pointer (UB in C++).
    // [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.operator-fn]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.operator-fn]
    pub fn call(&self, params: P) -> Ret {
        (self.callback.unwrap())(self.callable, params)
    }

    /// `explicit operator bool()` — whether `callback` is non-null, i.e.
    /// whether the FunctionRef refers to a callable.
    pub const fn is_some(&self) -> bool {
        self.callback.is_some()
    }
}

impl<P, Ret> PartialEq for FunctionRef<P, Ret> {
    fn eq(&self, other: &Self) -> bool {
        self.callable == other.callable
    }
}

impl<P, Ret> Default for FunctionRef<P, Ret> {
    fn default() -> Self {
        FunctionRef::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // namespace { void one(int32_t& i) { i = 1; } }
    fn one(i: &mut i32) {
        *i = 1;
    }

    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn/test]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.operator-fn/test]
    // also verifies callback_fn: `.call()` dispatches through the per-Callable
    // trampoline, which reinterprets the erased pointer and invokes the closure.
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn/test]
    #[test]
    fn function_ref_test_capturing_lambda() {
        let one_val = 1;
        let f = |i: &mut i32| *i = one_val;
        let mut val: i32 = 0;
        // FunctionRef<void(int32_t&)>{f}(val);
        FunctionRef::<&mut i32, ()>::from_callable(&f).call(&mut val);
        assert_eq!(val, 1);
    }

    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn/test]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.operator-fn/test]
    #[test]
    fn function_ref_test_non_capturing_lambda() {
        let mut val: i32 = 0;
        let lam0 = |i: &mut i32| *i = 1;
        let r#ref = FunctionRef::<&mut i32, ()>::from_callable(&lam0);
        r#ref.call(&mut val);
        assert_eq!(val, 1);

        val = 0;
        let lambda = |i: &mut i32| *i = 1;
        let ref1 = FunctionRef::<&mut i32, ()>::from_callable(&lambda);
        ref1.call(&mut val);
        assert_eq!(val, 1);
    }

    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn/test]
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.operator-fn/test]
    #[test]
    fn function_ref_test_function_pointer() {
        let mut val: i32 = 0;
        let r#ref = FunctionRef::<&mut i32, ()>::from_callable(&one);
        r#ref.call(&mut val);
        assert_eq!(val, 1);

        val = 0;
        let ref2 = FunctionRef::<&mut i32, ()>::from_callable(&one);
        ref2.call(&mut val);
        assert_eq!(val, 1);
    }

    // PORT-NOTE: no direct C++ counterpart. Pins the `callback` member's
    // contract: `= nullptr` (None) for default-/nullptr-constructed refs —
    // read back through `explicit operator bool` — set by the converting
    // constructor, and the pointer that `operator()` dispatches through.
    // [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn/test]
    #[test]
    fn function_ref_test_callback_null_states() {
        let empty = FunctionRef::<i32, i32>::new();
        assert!(!empty.is_some());

        let null = FunctionRef::<i32, i32>::from_null();
        assert!(!null.is_some());

        let f = |x: i32| x + 1;
        let bound = FunctionRef::<i32, i32>::from_callable(&f);
        assert!(bound.is_some());
        assert_eq!(bound.call(1), 2);
    }
}
