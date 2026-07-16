# kernels/portable/cpu/scalar_utils.h

> [spec:et:def:scalar-utils.torch.executor.native.utils.extract-scalar-fn]
> bool extract_scalar(Scalar scalar, FLOAT_T* out_val)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.extract-scalar-fn]
> This `def` line names the floating-point overload of `extract_scalar`, but
> `extract_scalar` is a set of three enable_if-selected template overloads keyed
> on the destination type `T`; describe all three since a Rust port implements
> one function. Returns `true` on success (writing `*out_val`) and `false` on
> failure (leaving `*out_val` unspecified), never aborts.
>
> Integer overload (`T` integral and not bool): if `!scalar.isIntegral(
> includeBool=false)` return false. Otherwise read `val = scalar.to<int64_t>()`;
> if `val < numeric_limits<T>::lowest()` or `val > numeric_limits<T>::max()`
> return false (out-of-range fails, matching PyTorch clamp which raises); else
> `*out_val = static_cast<T>(val)`, return true.
>
> Floating overload (`T` floating point): if `scalar.isFloatingPoint()`, read
> `val = scalar.to<double>()`; if `val` is finite AND (`val <
> (double)numeric_limits<T>::lowest()` or `val > (double)numeric_limits<T>::max()`)
> return false — but infinite and NaN values are allowed through (only finite
> out-of-range values fail). Else if `scalar.isIntegral(includeBool=false)`, set
> `val = (double)scalar.to<int64_t>()`. Else (not numeric) return false. Then
> `*out_val = static_cast<T>(val)`, return true.
>
> Bool overload (`T == bool`): if `scalar.isIntegral(false)` set `*out_val =
> (bool)scalar.to<int64_t>()` (nonzero → true) and return true; else if
> `scalar.isBoolean()` set `*out_val = scalar.to<bool>()` and return true; else
> return false.

> [spec:et:def:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn]
> inline ScalarType get_scalar_dtype(Scalar scalar)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn]
> Returns the ScalarType category of a `Scalar`, checked in this exact order:
> if `scalar.isBoolean()` return `ScalarType::Bool`; else if
> `scalar.isIntegral(false)` (integral, excluding bool) return `ScalarType::Long`;
> else if `scalar.isFloatingPoint()` return `ScalarType::Double`; else
> ET_CHECK_MSG(false, ...) which aborts with "Scalar must be Boolean, Integral or
> Floating." (unreachable for a well-formed Scalar). Note the ordering: a boolean
> Scalar is classified before the integral check.

> [spec:et:def:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn]
> std::optional<To> check_overflow_cast(From in)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn]
> Attempts to cast `in` (type `From`) to type `To`, returning `std::optional<To>`
> = the value on success or `std::nullopt` if the value cannot be represented in
> `To`. Logic: if `To` is NOT `bool` AND `c10::overflows<To, From>(in)` is true,
> return `std::nullopt`; otherwise return `static_cast<To>(in)`. Casting to bool
> is explicitly exempted from the overflow check (any value maps: nonzero→true,
> zero→false). `c10::overflows` implements PyTorch's range/representability check
> (integer range bounds; float-to-int finiteness/range; complex handling), so a
> Rust port should reject conversions where the source value is outside the
> destination's representable range, consistent with c10 overflow semantics.

> [spec:et:def:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]
> std::optional<To> check_overflow_scalar_cast(const Scalar& in)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]
> Casts a `Scalar` to type `To` with overflow checking, returning
> `std::optional<To>`. Dispatches on the Scalar's stored category: if
> `in.isBoolean()` call `check_overflow_cast<To>(in.to<bool>())`; else if
> `in.isFloatingPoint()` call `check_overflow_cast<To>(in.to<double>())`; else
> (integral) call `check_overflow_cast<To>(in.to<int64_t>())`. Delegates the
> range check to `check_overflow_cast` per
> `[spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-cast-fn]`,
> so returns `std::nullopt` if the underlying value overflows `To` (except when
> `To` is bool, which never overflows).

> [spec:et:def:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]
> inline ScalarType promote_type_with_scalar( ScalarType t, Scalar scalar, bool half_to_float = false)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-fn]
> Computes the promoted ScalarType of a tensor dtype `t` combined with a `Scalar`,
> with optional `half_to_float` (default false). If the Scalar's value category
> matches the tensor's category, the tensor dtype is preserved; otherwise it
> promotes to the Scalar's category dtype. Steps:
>
> 1. If `half_to_float` is true and `t == Half`, set `t = Float` first.
> 2. ET_CHECK `!isQIntType(t)` and ET_CHECK `!isBitsType(t)` (abort if `t` is a
>    quantized or bits type — unsupported).
> 3. If `isComplexType(t)`: return `t` unchanged (complex is always preserved).
> 4. If `scalar.isFloatingPoint()`: return `t` if `isFloatingType(t)`, else return
>    `ScalarType::Float` (ATen promotes to Float, not Double).
> 5. Else if `scalar.isIntegral(false)`: return `t` if `isFloatingType(t)` OR
>    `isIntegralType(t, false)`, else return `ScalarType::Long` (i.e. a Bool
>    tensor with an integer scalar promotes to Long).
> 6. Else if `scalar.isBoolean()`: return `t` unchanged.
> 7. Otherwise ET_CHECK_MSG(false, "Scalar must be Boolean, Integral or
>    Floating.") (abort; unreachable for a well-formed Scalar).

> [spec:et:def:scalar-utils.torch.executor.native.utils.promote-type-with-scalar-type]
> struct promote_type_with_scalar_type

> [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-double-fn]
> inline double scalar_to<double>(const Scalar& s)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-double-fn]
> Full specialization of `scalar_to<double>`: converts a `Scalar` to `double`. If
> `s.isFloatingPoint()` return `s.to<double>()`; otherwise (integral or boolean)
> return `static_cast<double>(s.to<int64_t>())`. Note it does not special-case
> boolean separately: a boolean Scalar is read via `to<int64_t>()` (false→0.0,
> true→1.0). Never fails or aborts.

> [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-fn]
> T scalar_to(const Scalar& s)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-fn]
> Primary template `scalar_to<T>`: converts a `Scalar` to C++ type `T` by
> dispatching on the Scalar's stored category and static_cast'ing to `T`. If
> `s.isBoolean()` return `static_cast<T>(s.to<bool>())`; else if
> `s.isFloatingPoint()` return `static_cast<T>(s.to<double>())`; else (integral)
> return `static_cast<T>(s.to<int64_t>())`. There are full specializations for
> `T == double` (see
> `[spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-double-fn]`)
> and `T == int64_t` (see
> `[spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn]`)
> that read the underlying value without the boolean branch. No range/overflow
> checking is performed — the cast follows C++ conversion rules; never aborts.

> [spec:et:def:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn]
> inline int64_t scalar_to<int64_t>(const Scalar& s)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.scalar-to-int64-t-fn]
> Full specialization of `scalar_to<int64_t>`: converts a `Scalar` to `int64_t`.
> If `s.isFloatingPoint()` return `static_cast<int64_t>(s.to<double>())`
> (truncation toward zero per C++ float-to-int conversion); otherwise (integral
> or boolean) return `s.to<int64_t>()` (false→0, true→1). Never fails or aborts.

> [spec:et:def:scalar-utils.torch.executor.native.utils.scalars-have-same-dtype-fn]
> inline bool scalars_have_same_dtype(Scalar a, Scalar b)

> [spec:et:sem:scalar-utils.torch.executor.native.utils.scalars-have-same-dtype-fn]
> Returns whether two Scalars share the same category dtype. Computes
> `a_dtype = get_scalar_dtype(a)` and `b_dtype = get_scalar_dtype(b)` per
> `[spec:et:sem:scalar-utils.torch.executor.native.utils.get-scalar-dtype-fn]`
> (each is one of Bool/Long/Double by category). If `a_dtype == b_dtype` return
> true. Otherwise log at Error level "Expected scalars to have the same dtype,
> but found <a> and <b>" and return false. Does not abort.

