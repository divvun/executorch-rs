# runtime/core/exec_aten/util/scalar_type_util.h

> [spec:et:def:scalar-type-util.executorch.runtime.can-cast]
> struct can_cast : std::integral_constant<

> [spec:et:def:scalar-type-util.executorch.runtime.can-cast-fn]
> inline constexpr bool canCast( const ::executorch::aten::ScalarType from, const ::executorch::aten::ScalarType to)

> [spec:et:sem:scalar-type-util.executorch.runtime.can-cast-fn]
> Returns whether a value of ScalarType `from` may be cast to ScalarType
> `to` under ATen's type-casting rules. Pure function; no side effects.
> Returns `true` unless one of these three prohibitions applies:
> - complex -> non-complex: `from` is a complex type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-complex-type-fn]`)
>   and `to` is not a complex type. Casting complex to real loses the
>   imaginary part and is disallowed.
> - floating -> integral: `from` is a floating type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-floating-type-fn]`,
>   i.e. Double/Float/Half/BFloat16) and `to` is an integral type with
>   `includeBool=false` (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-integral-type-fn]`,
>   i.e. Byte/Char/Short/Int/Long but NOT Bool).
> - non-bool -> Bool: `from` is not Bool and `to` is Bool.
>
> If none of these three conditions hold, the cast is allowed. Note the
> checks are purely on the enum categories; there is no numeric-range
> reasoning. Casting Bool -> Bool, integral -> integral, integral ->
> floating, floating -> floating, real -> complex, and complex ->
> complex are all permitted. QInt/Bits/Float8/barebones-unsigned types
> are not special-cased here and fall through to the default "allowed"
> path unless they trip one of the three category rules.

> [spec:et:def:scalar-type-util.executorch.runtime.convert-fn]
> To convert(From val)

> [spec:et:sem:scalar-type-util.executorch.runtime.convert-fn]
> A compile-time-dispatched value conversion `convert<To, From>(val)`
> that returns `val` converted to type `To`. There are two overloads
> selected by SFINAE on the C++ types `From` and `To`:
> - When `From` is a floating-point type AND `To` is an integral type:
>   return `static_cast<To>(static_cast<int64_t>(val))`. That is, first
>   truncate the floating value toward zero into an `int64_t`, then
>   narrow that `int64_t` to `To`. The intermediate `int64_t` step
>   exists to avoid undefined-behavior / sanitizer traps that a direct
>   float->small-int cast can produce when the float is outside `To`'s
>   representable range; the observable result is that out-of-range
>   floats are first reduced modulo/wrapped through `int64_t` semantics
>   rather than triggering a sanitizer error. NaN/inf conversion to
>   int64_t is itself C++ undefined behavior and is not further defined
>   here.
> - Otherwise (any case that is not float->integral, e.g.
>   integral->anything, float->float, anything->float, complex->complex):
>   return plain `static_cast<To>(val)`.
> Pure function; no context, no error reporting. `To` and `From` are C++
> types, not ScalarType enum values.

> [spec:et:def:scalar-type-util.executorch.runtime.element-size-fn]
> inline size_t elementSize(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.element-size-fn]
> Returns the size in bytes of the C++ element type associated with the
> given ScalarType `t`. Implemented as a switch over every ScalarType
> covered by `ET_FORALL_SCALAR_TYPES` (the full set of core scalar
> types: Byte/Char/Short/Int/Long, Float/Double, Half, BFloat16, Bool,
> the complex types, the quantized QInt types, etc.), each case
> returning `sizeof(ctype)` for that type's mapped C++ type. Concretely:
> Byte=1, Char=1, Short=2, Int=4, Long=8, Half=2, Float=4, Double=8,
> ComplexHalf=4, ComplexFloat=8, ComplexDouble=16, Bool=1, BFloat16=2,
> QInt8=1, QUInt8=1, QInt32=4, QUInt4x2=1, QUInt2x4=1, and any other type
> in the FORALL set sized by its underlying C++ type. For the `default`
> case (a value not covered by the FORALL set, e.g.
> ScalarType::Undefined or an out-of-range enum): calls
> `ET_CHECK_MSG(false, "Unknown ScalarType %d", (int8_t)t)`, which
> aborts the program (fatal check failure); it does not return. Pure/no
> error-context variant — failure is a hard abort, not an Error return.

> [spec:et:def:scalar-type-util.executorch.runtime.fail-fn]
> __VA_ARGS__ \

> [spec:et:sem:scalar-type-util.executorch.runtime.fail-fn]
> This rule annotates the `default:` (fall-through) branch of the
> `ET_INTERNAL_SWITCH` macro, which is the common dispatch scaffold used
> by all the type-dispatch switch macros (e.g. the ET_SWITCH_* family).
> The macro builds a lambda that switches on the runtime ScalarType
> `TYPE`; the `__VA_ARGS__` expands to the caller-supplied `case` labels
> (one per accepted dtype, each binding a `CTYPE_ALIAS` and invoking the
> caller's body). When the runtime type matches none of the supplied
> cases, control reaches this `default` branch, whose behavior is:
> - Call `CONTEXT.fail(torch::executor::Error::InvalidArgument)` on the
>   provided KernelRuntimeContext, recording InvalidArgument as the
>   context's error state (this is how a kernel signals an unhandled
>   dtype to its caller rather than aborting).
> - Emit a log at Error level with the message
>   "Unhandled dtype %s for %s", where the first argument is
>   `toString(TYPE)` (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.to-string-fn]`) and
>   the second is the switch's NAME string (an operator/op-name label
>   passed by the caller for diagnostics).
> The lambda then falls out of the switch and returns the default-
> constructed value of the lambda's return type (control simply reaches
> the end of the lambda without an explicit `return` in this branch).
> No abort occurs; the failure is propagated via the context's Error.

> [spec:et:def:scalar-type-util.executorch.runtime.internal.calculate-dtype2index-fn]
> constexpr std::array<

> [spec:et:sem:scalar-type-util.executorch.runtime.internal.calculate-dtype2index-fn]
> A `constexpr` builder that computes the inverse mapping of the
> `index2dtype` table used by type promotion. `index2dtype` is a fixed
> 13-element array listing the promotable ScalarTypes in promotion-table
> order: [Byte(u1), Char(i1), Short(i2), Int(i4), Long(i8), Half(f2),
> Float(f4), Double(f8), ComplexHalf(c2), ComplexFloat(c4),
> ComplexDouble(c8), Bool(b1), BFloat16(bf)].
> Steps:
> - Allocate an array `inverse` of length `NumOptions` (the total number
>   of ScalarType enumerators), value-initialized.
> - First loop: set every entry `inverse[i] = -1` for i in
>   [0, NumOptions), marking all dtypes as "not promotable / no index".
> - Second loop: for each position `i` in [0, 13), set
>   `inverse[(int)index2dtype[i]] = i`. I.e. for each promotable dtype,
>   store its row/column index in the promotion lookup table, keyed by
>   the dtype's enum integer value.
> - Return `inverse`.
> Result: `dtype2index[(int)dtype]` yields the promotion-table index
> (0..12) for a promotable dtype, or -1 for any dtype that is not part of
> the promotion table. Used by
> `[spec:et:sem:scalar-type-util.executorch.runtime.promote-types-fn]`.

> [spec:et:def:scalar-type-util.executorch.runtime.is-barebones-unsigned-type]
> struct is_barebones_unsigned_type

> [spec:et:def:scalar-type-util.executorch.runtime.is-barebones-unsigned-type-fn]
> constexpr bool isBarebonesUnsignedType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-barebones-unsigned-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the "barebones"
> wide unsigned integer types: UInt16, UInt32, or UInt64. Returns `false`
> for every other ScalarType (including Byte, which is the 8-bit unsigned
> type but is treated as a normal integral type, not a barebones
> unsigned). Pure; no side effects. These types are excluded from
> promotion (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.promote-types-fn]`).

> [spec:et:def:scalar-type-util.executorch.runtime.is-bits-type]
> struct is_bits_type

> [spec:et:def:scalar-type-util.executorch.runtime.is-bits-type-fn]
> constexpr bool isBitsType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-bits-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the opaque
> "bits" types with no arithmetic semantics: Bits1x8, Bits2x4, Bits4x2,
> Bits8, or Bits16. Returns `false` for every other ScalarType. Pure; no
> side effects. Bits types are excluded from promotion (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.promote-types-fn]`).

> [spec:et:def:scalar-type-util.executorch.runtime.is-complex-type]
> struct is_complex_type : std::integral_constant<

> [spec:et:def:scalar-type-util.executorch.runtime.is-complex-type-fn]
> inline constexpr bool isComplexType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-complex-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the complex
> ScalarTypes: ComplexHalf, ComplexFloat, or ComplexDouble. Returns
> `false` for every other ScalarType. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-float8-type]
> struct is_float8_type

> [spec:et:def:scalar-type-util.executorch.runtime.is-float8-type-fn]
> constexpr bool isFloat8Type(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-float8-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the 8-bit
> floating types: Float8_e5m2, Float8_e4m3fn, Float8_e5m2fnuz, or
> Float8_e4m3fnuz. Returns `false` for every other ScalarType. Pure; no
> side effects. Float8 types are excluded from promotion (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.promote-types-fn]`).

> [spec:et:def:scalar-type-util.executorch.runtime.is-floating-point]
> struct is_floating_point

> [spec:et:def:scalar-type-util.executorch.runtime.is-floating-type-fn]
> inline constexpr bool isFloatingType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-floating-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the real
> floating-point ScalarTypes: Double, Float, Half, or BFloat16. Returns
> `false` for every other ScalarType (including the complex types and
> Float8 types, which are NOT counted as floating here). Pure; no side
> effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-integral-type]
> struct is_integral_type

> [spec:et:def:scalar-type-util.executorch.runtime.is-integral-type-fn]
> inline constexpr bool isIntegralType( ::executorch::aten::ScalarType t, bool includeBool)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-integral-type-fn]
> `constexpr` predicate `isIntegralType(t, includeBool)`. Returns `true`
> iff either:
> - `includeBool` is true and `t == Bool`, OR
> - `t` is one of the signed/unsigned integer ScalarTypes: Byte, Char,
>   Int, Long, or Short.
> Returns `false` otherwise. When `includeBool` is false, Bool is NOT
> considered integral. Note this covers only the core integer types;
> the barebones unsigned types (UInt16/32/64) and quantized QInt types
> are NOT counted as integral here. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-q-int-type-fn]
> constexpr bool isQIntType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-q-int-type-fn]
> `constexpr` predicate. Returns `true` iff `t` is one of the quantized
> integer ScalarTypes: QInt8, QUInt8, QInt32, QUInt4x2, or QUInt2x4.
> Returns `false` for every other ScalarType. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-qint-type]
> struct is_qint_type

> [spec:et:def:scalar-type-util.executorch.runtime.is-real-h-type-fn]
> inline bool isRealHType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-real-h-type-fn]
> Predicate. Returns `true` iff `t` is a "real" type plus Half — that is,
> one of: Byte, Char, Short, Int, Long, Float, Double, or Half. Returns
> `false` for every other ScalarType (Bool, BFloat16, complex, quantized,
> etc.). This is the base set that the RealHB / RealHBF16 / RealHBBF16
> predicates extend. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-real-hb-type-fn]
> inline bool isRealHBType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-real-hb-type-fn]
> Predicate. Returns `true` iff `t` satisfies
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-h-type-fn]`
> (Byte/Char/Short/Int/Long/Float/Double/Half) OR `t == Bool`. That is,
> the RealH set with Bool added. Returns `false` for BFloat16, complex,
> quantized, etc. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-real-hbbf16-type-fn]
> inline bool isRealHBBF16Type(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-real-hbbf16-type-fn]
> Predicate. Returns `true` iff `t` satisfies
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-hb-type-fn]`
> (RealH set plus Bool) OR `t == BFloat16`. Equivalently, `true` for any
> of: Byte, Char, Short, Int, Long, Float, Double, Half, Bool, BFloat16.
> Returns `false` for complex, quantized, Float8, and barebones-unsigned
> types. This is the "REALHBBF16" dtype set referenced by many portable
> kernels. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-real-hbf16-type-fn]
> inline bool isRealHBF16Type(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-real-hbf16-type-fn]
> Predicate. Returns `true` iff `t` satisfies
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-h-type-fn]`
> (Byte/Char/Short/Int/Long/Float/Double/Half) OR `t == BFloat16`. That
> is, the RealH set with BFloat16 added but NOT Bool. Returns `false`
> otherwise. This is the "REALHBF16" dtype set. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-real-type-fn]
> inline bool isRealType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-real-type-fn]
> Predicate. Returns `true` iff `t` is one of the "real" (integer/float)
> ScalarTypes: Byte, Char, Short, Int, Long, Float, or Double. Returns
> `false` for every other ScalarType, including Bool, Half, BFloat16,
> complex, and quantized types (Half and BFloat16 are deliberately
> excluded from the plain "REAL" set). Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-reduced-floating-point]
> struct is_reduced_floating_point

> [spec:et:def:scalar-type-util.executorch.runtime.is-signed-type-fn]
> inline bool isSignedType(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-signed-type-fn]
> Returns whether values of ScalarType `t` are signed.
> - Precondition check: if `t` is a quantized type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-q-int-type-fn]`),
>   call `ET_CHECK_MSG(false, "isSignedType not supported for quantized
>   types ...")`, which fatally aborts. isSignedType is not defined for
>   quantized dtypes.
> - The three complex types (ComplexHalf, ComplexFloat, ComplexDouble)
>   return `true`.
> - For each real type plus Half, Bool, and BFloat16 (the
>   ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16) set: Byte, Char,
>   Short, Int, Long, Float, Double, Half, Bool, BFloat16), return
>   `std::numeric_limits<ctype>::is_signed` for that type's C++ type.
>   Concretely: Byte (uint8) -> false, Bool -> false; Char, Short, Int,
>   Long, Float, Double, Half, BFloat16 -> true.
> - For any `t` not covered above (e.g. Undefined, Float8, barebones
>   unsigned, bits types): `ET_CHECK_MSG(false, "Unknown ScalarType ...")`
>   fatally aborts.
> Failure is a hard abort, not an Error return.

> [spec:et:def:scalar-type-util.executorch.runtime.is-underlying-fn]
> inline bool isUnderlying( ::executorch::aten::ScalarType type, ::executorch::aten::ScalarType qtype)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-underlying-fn]
> Predicate `isUnderlying(type, qtype)`. Returns `true` iff `type` equals
> the underlying (non-quantized) storage ScalarType of the quantized type
> `qtype`, i.e. `type == toUnderlying(qtype)` per
> `[spec:et:sem:scalar-type-util.executorch.runtime.to-underlying-fn]`.
> For a non-quantized `qtype`, `toUnderlying` returns `qtype` unchanged,
> so this reduces to `type == qtype`. Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.is-valid-fn]
> inline bool isValid(::executorch::aten::ScalarType type)

> [spec:et:sem:scalar-type-util.executorch.runtime.is-valid-fn]
> Predicate. Returns `true` iff `type` is a real, usable ScalarType
> covered by ET_FORALL_SCALAR_TYPES. Computed as the conjunction of three
> conditions on the enum's underlying `int8_t` value:
> - `(int8_t)type >= 0` (not a negative sentinel),
> - `type < ScalarType::NumOptions` (below the one-past-the-end marker),
> - `type != ScalarType::Undefined`.
> Returns `false` if any condition fails (negative, at/beyond NumOptions,
> or Undefined). Pure; no side effects.

> [spec:et:def:scalar-type-util.executorch.runtime.promote-types]
> struct promote_types

> [spec:et:def:scalar-type-util.executorch.runtime.promote-types-fn]
> inline constexpr ::executorch::aten::ScalarType promoteTypes( ::executorch::aten::ScalarType a, ::executorch::aten::ScalarType b, bool half_to_float = false)

> [spec:et:sem:scalar-type-util.executorch.runtime.promote-types-fn]
> `constexpr promoteTypes(a, b, half_to_float=false)`. Computes the ATen/
> NumPy-consistent result ScalarType of combining dtypes `a` and `b`.
> Steps, in order:
> - QInt handling: if `a` is a quantized type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-q-int-type-fn]`)
>   and `a == b`, return `a`. Otherwise, if either `a` or `b` is
>   quantized, `ET_CHECK_MSG(false, "promoteTypes not valid for quantized
>   dtypes")` (fatal abort).
> - Bits handling: if `a` is a bits type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-bits-type-fn]`)
>   and `a == b`, return `a`. Otherwise if either is a bits type, fatal
>   abort with "promoteTypes not valid for bits dtypes".
> - Float8 handling: if `a` is a float8 type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-float8-type-fn]`)
>   and `a == b`, return `a`. Otherwise if either is float8, fatal abort
>   with "promoteTypes not valid for float8 dtypes".
> - Barebones-unsigned handling: if `a` is a barebones unsigned type (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.is-barebones-unsigned-type-fn]`)
>   and `a == b`, return `a`. Otherwise if either is barebones unsigned,
>   fatal abort with "promoteTypes not valid for barebone unsigned
>   dtypes".
> - Look up the table index for each: `ix_a = dtype2index[(int)a]`,
>   `ix_b = dtype2index[(int)b]` (per
>   `[spec:et:sem:scalar-type-util.executorch.runtime.internal.calculate-dtype2index-fn]`).
>   `ET_CHECK(ix_a != -1)` and `ET_CHECK(ix_b != -1)` (fatal abort if
>   either dtype is not in the promotion table).
> - `promoted_type = promoteTypesLookup[ix_a][ix_b]`, a fixed 13x13 table
>   whose rows/cols are ordered [u1,i1,i2,i4,i8,f2,f4,f8,c2,c4,c8,b1,bf]
>   (Byte, Char, Short, Int, Long, Half, Float, Double, ComplexHalf,
>   ComplexFloat, ComplexDouble, Bool, BFloat16) and which matches
>   PyTorch core's `_promoteTypesLookup`. The table is symmetric and
>   encodes NumPy promote_types rules (e.g. Byte x Char -> Short; any int
>   x Half -> Half; Half x BFloat16 -> Float; Bool acts as identity for
>   promotion, promoting to the other operand; BFloat16 x Float -> Float).
> - half_to_float adjustment: if `half_to_float` is true and
>   `promoted_type` is Half or BFloat16, override `promoted_type` to
>   Float.
> - Return `promoted_type`.
> Pure aside from the fatal ET_CHECK aborts on invalid dtype categories.

> [spec:et:def:scalar-type-util.executorch.runtime.to-complex-type-fn]
> inline constexpr ::executorch::aten::ScalarType toComplexType( ::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.to-complex-type-fn]
> `constexpr` map from a floating-or-complex ScalarType to the complex
> ScalarType with the corresponding value precision:
> - BFloat16 -> ComplexFloat (BFloat16 has Float-equivalent range, so it
>   maps to ComplexFloat rather than ComplexHalf).
> - Half -> ComplexHalf.
> - Float -> ComplexFloat.
> - Double -> ComplexDouble.
> - ComplexHalf -> ComplexHalf; ComplexFloat -> ComplexFloat;
>   ComplexDouble -> ComplexDouble (already complex, returned unchanged).
> - Any other `t` (integers, Bool, quantized, etc.): `ET_CHECK_MSG(false,
>   "Unknown Complex ScalarType for %d", (int8_t)t)` (fatal abort). This
>   function does not return for non-floating/non-complex inputs.

> [spec:et:def:scalar-type-util.executorch.runtime.to-q-int-type-fn]
> inline ::executorch::aten::ScalarType toQIntType( ::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.to-q-int-type-fn]
> Maps a plain integer ScalarType to its corresponding quantized type:
> - Byte -> QUInt8
> - Char -> QInt8
> - Int -> QInt32
> - Any other `t` (default): return `t` unchanged.
> Pure; no side effects, no error path.

> [spec:et:def:scalar-type-util.executorch.runtime.to-real-value-type-fn]
> inline constexpr ::executorch::aten::ScalarType toRealValueType( ::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.to-real-value-type-fn]
> `constexpr` map from a complex ScalarType to the real ScalarType of its
> component values:
> - ComplexHalf -> Half
> - ComplexFloat -> Float
> - ComplexDouble -> Double
> - Any other `t` (default, i.e. already-real or non-complex types):
>   return `t` unchanged.
> Pure; no side effects, no error path.

> [spec:et:def:scalar-type-util.executorch.runtime.to-string-fn]
> inline const char* toString(::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.to-string-fn]
> Returns a static C string with the name of ScalarType `t`. Implemented
> as a switch: for every type in ET_FORALL_SCALAR_TYPES, returns the
> stringized enumerator name (e.g. Byte -> "Byte", Float -> "Float",
> ComplexFloat -> "ComplexFloat", QInt8 -> "QInt8"). The
> `ScalarType::Undefined` case returns "Undefined". The `default` case
> (any value not covered above, including out-of-range enum values)
> returns "UNKNOWN_SCALAR". Never aborts; always returns a valid
> non-owning string literal.

> [spec:et:def:scalar-type-util.executorch.runtime.to-underlying-fn]
> inline ::executorch::aten::ScalarType toUnderlying( ::executorch::aten::ScalarType t)

> [spec:et:sem:scalar-type-util.executorch.runtime.to-underlying-fn]
> Maps a quantized ScalarType to the plain integer ScalarType of its
> underlying storage:
> - QUInt8 -> Byte
> - QInt8 -> Char
> - QInt32 -> Int
> - QUInt4x2 -> Byte
> - QUInt2x4 -> Byte
> - Any other `t` (default, including non-quantized types): return `t`
>   unchanged.
> Pure; no side effects, no error path. Inverse (for the three scalar
> quantized types) of
> `[spec:et:sem:scalar-type-util.executorch.runtime.to-q-int-type-fn]`.

