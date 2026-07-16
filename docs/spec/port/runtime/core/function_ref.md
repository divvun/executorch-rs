# runtime/core/function_ref.h

> [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params]
> class FunctionRef<Ret(Params...)> {
>   intptr_t callable;
> }

> [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.callback-fn]
> Ret (*callback)(intptr_t callable, Params... params) = nullptr

> [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn]
> A member function pointer `Ret (*callback)(intptr_t callable, Params...
> params)`, default-initialized to `nullptr`. It is the type-erased trampoline
> the FunctionRef stores alongside the erased `callable` pointer: given the
> stored callable (as an `intptr_t`) and the call arguments, it dispatches to
> the concrete callable. It is set to `&callback_fn<Callable>`
> (`[spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]`)
> by the templated constructor
> (`[spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn]`),
> and remains `nullptr` for a default- or `nullptr`-constructed FunctionRef.
> `explicit operator bool()` returns whether `callback` is non-null, i.e.
> whether the FunctionRef refers to a callable.

> [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]
> static Ret callback_fn(intptr_t callable, Params... params)

> [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]
> The static trampoline instantiated per concrete `Callable` type. It
> reinterprets the erased `intptr_t callable` back into a `Callable*`
> (`reinterpret_cast<Callable*>(callable)`), dereferences it, and invokes it
> with the perfectly-forwarded parameters:
> `(*reinterpret_cast<Callable*>(callable))(std::forward<Params>(params)...)`,
> returning the result as `Ret` (implicitly convertible per the ctor's
> constraint). The `intptr_t` must be the address of a live `Callable` object;
> the reinterpret cast is the inverse of the `reinterpret_cast<intptr_t>` done
> in the constructor. FunctionRef is non-owning, so the referenced callable
> must still be alive at call time. A Rust port erases the callable as a raw
> pointer plus a monomorphized function pointer that casts it back to the
> concrete type and calls it.

> [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn]
> FunctionRef( Callable&& callable, // This is not the copy-constructor. std::enable_if_t<!std::is_same< internal::remove_cvref_t<Callable>, FunctionRef>::value>* = nullptr, // Functor must be callable and return a suitable type. std::enab...

> [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.function-ref-fn]
> Templated converting constructor that binds the FunctionRef to any callable
> `callable` (function pointer, lambda, or functor) without owning it.
>
> SFINAE constraints (both must hold or the overload is removed): (1) the
> callable's decayed type (`remove_cvref_t<Callable>`) is NOT `FunctionRef`
> itself — this prevents the template from hijacking the copy/move
> constructors; (2) either `Ret` is `void`, or the result of invoking the
> callable with `Params...` is convertible to `Ret`.
>
> Initialization: `callback = &callback_fn<std::remove_reference_t<Callable>>`
> (`[spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn-fn]`)
> and `callable = reinterpret_cast<intptr_t>(&callable)` — it stores the
> ADDRESS of the referenced callable object, not a copy. Because it is
> non-owning, the referenced object must outlive every use of this
> FunctionRef; storing a FunctionRef beyond the callable's lifetime is unsafe.
> There are also `FunctionRef() = default` (leaves `callback == nullptr`) and
> `FunctionRef(std::nullptr_t)` producing an empty ref.

> [spec:et:def:function-ref.executorch.runtime.function-ref-ret-params.operator-fn]
> Ret operator()(Params... params) const

> [spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.operator-fn]
> Call operator (`const`). Invokes the referenced callable by forwarding
> through the stored trampoline: `return callback(callable,
> std::forward<Params>(params)...)`. The stored `callback`
> (`[spec:et:sem:function-ref.executorch.runtime.function-ref-ret-params.callback-fn]`)
> receives the erased `callable` pointer and the perfectly-forwarded
> arguments, casts the pointer back to the concrete callable type, calls it,
> and returns `Ret`. Precondition: the FunctionRef must be non-empty
> (`callback != nullptr`) — calling an empty/default-constructed FunctionRef
> dereferences a null function pointer (undefined behavior); use `operator
> bool()` to test first. No arguments are copied beyond what the callable's
> signature requires.

> [spec:et:def:function-ref.executorch.runtime.internal.remove-cvref]
> struct remove_cvref

