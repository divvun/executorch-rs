# runtime/core/portable_type/scalar.h

> [spec:et:def:scalar.executorch.runtime.etensor.scalar]
> class Scalar {
>   Tag tag;
>   union v_t { double as_double; int64_t as_int; bool as_bool; v_t() {} // default constructor } v;
> }

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]
> bool isBoolean() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]
> Returns `true` iff the scalar's discriminant `tag` equals `Tag::Bool`,
> otherwise `false`. Pure `const` predicate over `tag` alone; does not read the
> union `v`, no mutation, cannot fail. In a Rust port: matches the enum variant
> carrying the boolean payload.

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-floating-point-fn]
> bool isFloatingPoint() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-floating-point-fn]
> Returns `true` iff `tag == Tag::Double`, otherwise `false`. Note this is the
> only floating-point tag: `BFloat16` and `Half` scalars are stored as
> `Tag::Double` (their constructors convert through `(double)(float)val`), so
> they also report `true`. Pure `const` predicate over `tag` alone; no
> mutation, cannot fail.

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.is-integral-fn]
> bool isIntegral(bool includeBool) const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-integral-fn]
> Returns `true` iff `tag == Tag::Int`, OR (`includeBool` is `true` AND the
> scalar is boolean per `[spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]`,
> i.e. `tag == Tag::Bool`). Concretely: `Tag::Int == tag || (includeBool &&
> isBoolean())`. So with `includeBool == false` only `Tag::Int` qualifies; with
> `includeBool == true` both `Tag::Int` and `Tag::Bool` qualify. `Tag::Double`
> never qualifies. Pure `const` predicate over `tag`; no mutation, cannot fail.

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.scalar-fn]
> Scalar(T val) : tag(Tag::Int)

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.scalar-fn]
> Templated implicit constructor enabled (via SFINAE on
> `std::is_integral<T>`) only for integral C types `T` — this overload excludes
> `bool`, `double`, `BFloat16`, and `Half`, which are handled by the sibling
> non-templated constructors. Sets the discriminant `tag = Tag::Int`, then
> stores the value into the union as `v.as_int = static_cast<int64_t>(val)`.
> The cast widens/sign-extends any signed or unsigned integer to `int64_t`
> following C++ integral-conversion rules (unsigned 64-bit values above
> `INT64_MAX` wrap modulo 2^64 into the signed representation). No validation,
> no side effects, cannot fail. The other union members `as_double`/`as_bool`
> are left inactive. Sibling constructors (not this rule) instead set:
> `Tag::Bool` with `v.as_bool = val` for `bool`; `Tag::Double` with
> `v.as_double = val` for `double`; and for `BFloat16`/`Half`, delegate to the
> `double` constructor via `(double)(float)val`.

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-bool-fn]
> bool toBool() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-bool-fn]
> Private accessor returning the boolean payload. First asserts the scalar is
> boolean via `ET_CHECK_MSG(isBoolean(), "Scalar is not a Boolean.")`: if `tag
> != Tag::Bool` this aborts execution (the ExecuTorch fatal-check path, which
> terminates the program) with that message. Otherwise returns `v.as_bool`
> unchanged. Pure read; no conversion. In a Rust port the failed check maps to
> a panic (or an error return if the API is fallibilized).

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-double-fn]
> double toDouble() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-double-fn]
> Private accessor returning the floating-point payload. Asserts the scalar is
> floating point via `ET_CHECK_MSG(isFloatingPoint(), "Scalar is not a
> Double.")`: if `tag != Tag::Double` this aborts execution (fatal check,
> terminates the program) with that message. Otherwise returns `v.as_double`
> unchanged. Behaviorally identical to `toFloatingPoint()`
> (`[spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn]`);
> both exist for source compatibility. In a Rust port the failed check maps to
> a panic (or an error return if fallibilized).

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn]
> double toFloatingPoint() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-floating-point-fn]
> Private accessor returning the floating-point payload. Asserts the scalar is
> floating point via `ET_CHECK_MSG(isFloatingPoint(), "Scalar is not a
> Double.")`: if `tag != Tag::Double` this aborts execution (fatal check,
> terminates the program) with that message. Otherwise returns `v.as_double`
> unchanged. Identical in behavior to `toDouble()`
> (`[spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-double-fn]`). In a
> Rust port the failed check maps to a panic (or an error return if
> fallibilized).

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-fn]
> T to() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-fn]
> Public templated value extractor `T to() const`, declared generically but
> only defined (via the `ET_DEFINE_SCALAR_TO_METHOD` macro) for exactly three
> `T`: `double`, `int64_t`, and `bool`. Each explicit specialization forwards
> to the matching private accessor:
> - `to<double>()` -> `toDouble()`
>   (`[spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-double-fn]`)
> - `to<int64_t>()` -> `toInt()`
>   (`[spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn]`)
> - `to<bool>()` -> `toBool()`
>   (`[spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-bool-fn]`)
>
> Instantiating `to<T>()` for any other `T` is a compile-time error (undefined
> reference / no matching specialization), so the set of retrievable types is
> fixed. Each specialization inherits its underlying accessor's fatal-check
> behavior on a tag mismatch (e.g. `to<double>()` aborts if the scalar is not
> `Tag::Double`). In a Rust port model this as three concrete conversion
> methods (or a small enum of supported target types) rather than an open
> generic.

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.to-int-fn]
> int64_t toInt() const

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.to-int-fn]
> Private accessor returning the integer payload as `int64_t`. Branches on the
> discriminant:
> - If `isIntegral(/*includeBool=*/false)` is true (i.e. `tag == Tag::Int`,
>   per `[spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-integral-fn]`),
>   returns `v.as_int` unchanged.
> - Else if `isBoolean()` (`tag == Tag::Bool`,
>   per `[spec:et:sem:scalar.executorch.runtime.etensor.scalar.is-boolean-fn]`),
>   returns `static_cast<int64_t>(v.as_bool)` — i.e. `1` for `true`, `0` for
>   `false`. Note this differs from the other `to*` accessors: a boolean scalar
>   is accepted here and coerced to 0/1.
> - Else (a `Tag::Double` scalar) calls `ET_CHECK_MSG(false, "Scalar is not an
>   int nor a Boolean.")`, which unconditionally aborts execution (fatal check,
>   terminates the program) with that message.
>
> No mutation. In a Rust port the fatal branch maps to a panic (or an error
> return if fallibilized).

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.v-t]
> union v_t {
>   double as_double;
>   int64_t as_int;
>   bool as_bool;
> }

> [spec:et:def:scalar.executorch.runtime.etensor.scalar.v-t.v-t-fn]
> v_t()

> [spec:et:sem:scalar.executorch.runtime.etensor.scalar.v-t.v-t-fn]
> User-provided default constructor for the anonymous-payload union `v_t` with
> an empty body `{}`. It exists only to make the union default-constructible in
> C++ (a union with members of non-trivially-constructible or ambiguous types
> otherwise has a deleted default constructor). It activates none of the
> members `as_double`/`as_int`/`as_bool` and initializes no storage — the
> union's bytes are left indeterminate until the enclosing `Scalar`
> constructor assigns one member and sets `tag`. No side effects, cannot fail.
> In a Rust port there is no direct analog: the payload is modeled as a
> tagged enum (or the value is always written together with its tag), so no
> standalone empty union constructor is needed.

