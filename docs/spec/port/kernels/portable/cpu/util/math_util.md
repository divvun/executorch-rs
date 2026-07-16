# kernels/portable/cpu/util/math_util.h

> [spec:et:def:math-util.torch.executor.native.utils.floor-divide-fn]
> FLOAT_T floor_divide(FLOAT_T a, FLOAT_T b)

> [spec:et:sem:math-util.torch.executor.native.utils.floor-divide-fn]
> Floating-point floor-division of `a` by `b` matching Python's `__floordiv__`
> semantics (this template overload is selected when `FLOAT_T` is a
> floating-point type; a separate integral overload exists but is not this rule).
> Computes `a // b` such that the identity `a == (a // b) * b + remainder(a, b)`
> holds as closely as floating-point rounding allows, rather than naively
> computing `floor(a / b)`.
>
> Steps:
> 1. If `b == 0`: return signed infinity — `-INFINITY` if `a` has its sign bit
>    set (per `std::signbit(a)`, which is true for negative values and for
>    negative zero), otherwise `+INFINITY`. (Note: when `a` is NaN this still
>    returns `+INFINITY` because `signbit(NaN)` is typically false; no explicit
>    NaN guard is present.)
> 2. Otherwise compute `mod = std::fmod(a, b)` (C remainder: same sign as `a`,
>    magnitude less than `|b|`).
> 3. Compute `div = (a - mod) / b`.
> 4. If `mod != 0` and the sign bit of `b` differs from the sign bit of `mod`
>    (i.e. `std::signbit(b) != std::signbit(mod)`), return `div - 1`.
> 5. Otherwise return `div`.
>
> All arithmetic is performed in `FLOAT_T`. This is a pure elementwise scalar
> helper with no tensor/error-context interaction.

> [spec:et:def:math-util.torch.executor.native.utils.isnan-override-fn]
> bool isnan_override(T a)

> [spec:et:sem:math-util.torch.executor.native.utils.isnan-override-fn]
> NaN test that is well-defined for every scalar type `T`, wrapping
> `std::isnan` (which is provided to sidestep ambiguous overload resolution for
> integral inputs on MSVC).
>
> Behavior:
> 1. If `T` is a floating-point (non-integral) type: return `std::isnan(a)`
>    (true iff `a` is a NaN value).
> 2. If `T` is an integral type: return `false` unconditionally (integers can
>    never be NaN).
>
> Pure scalar helper; no side effects.

> [spec:et:def:math-util.torch.executor.native.utils.max-override-fn]
> T max_override(T a, T b)

> [spec:et:sem:math-util.torch.executor.native.utils.max-override-fn]
> Binary maximum with PyTorch NaN-propagation semantics. This rule covers the
> `T max_override(T a, T b)` overload selected when `T` is `Half` or
> `BFloat16`; sibling overloads exist for floating-point and integral types
> with equivalent semantics (see below).
>
> Half / BFloat16 overload steps:
> 1. Widen `a` to `float` as `float_a`. If `std::isnan(float_a)`, return the
>    original `a` (NaN propagates).
> 2. Widen `b` to `float` as `float_b`. If `std::isnan(float_b)`, return the
>    original `b`.
> 3. If `float_a > float_b`, return `a`; otherwise return `b` (so ties return
>    `b`).
>
> Sibling overloads (for re-implementation completeness):
> - Floating-point `FLOAT_T`: if `std::isnan(a)` return `a`; else if
>   `std::isnan(b)` return `b`; else return `std::max(a, b)`.
> - Integral `INT_T`: return `std::max(a, b)` (no NaN handling; integers are
>   never NaN).
>
> Pure scalar helper; no side effects. (Vectorized `at::vec` overloads exist
> only when built with PyTorch headers and delegate to `at::vec::maximum`; they
> are outside this port's scope.)

> [spec:et:def:math-util.torch.executor.native.utils.min-override-fn]
> T min_override(T a, T b)

> [spec:et:sem:math-util.torch.executor.native.utils.min-override-fn]
> Binary minimum with PyTorch NaN-propagation semantics. This rule covers the
> `T min_override(T a, T b)` overload selected when `T` is `Half` or
> `BFloat16`; sibling overloads exist for floating-point and integral types
> with equivalent semantics (see below).
>
> Half / BFloat16 overload steps:
> 1. Widen `a` to `float` as `float_a`. If `std::isnan(float_a)`, return the
>    original `a` (NaN propagates).
> 2. Widen `b` to `float` as `float_b`. If `std::isnan(float_b)`, return the
>    original `b`.
> 3. If `float_a < float_b`, return `a`; otherwise return `b` (so ties return
>    `b`).
>
> Sibling overloads (for re-implementation completeness):
> - Floating-point `FLOAT_T`: if `std::isnan(a)` return `a`; else if
>   `std::isnan(b)` return `b`; else return `std::min(a, b)`.
> - Integral `INT_T`: return `std::min(a, b)` (no NaN handling; integers are
>   never NaN).
>
> Pure scalar helper; no side effects. (Vectorized `at::vec` overloads exist
> only when built with PyTorch headers and delegate to `at::vec::minimum`; they
> are outside this port's scope.)

> [spec:et:def:math-util.torch.executor.native.utils.remainder-override-fn]
> CTYPE remainder_override(CTYPE a, CTYPE b)

> [spec:et:sem:math-util.torch.executor.native.utils.remainder-override-fn]
> Remainder with ATen (PyTorch) sign convention, where the result always takes
> the sign of the divisor `b` (unlike C's `std::fmod`, whose result takes the
> sign of the dividend `a`). This rule covers the floating-point overload
> (`CTYPE` is a floating-point type); an integral overload also exists.
>
> Floating-point overload steps:
> 1. Compute `rem = std::fmod(a, b)`. (Note: the intermediate `rem` is declared
>    as `float` in the source regardless of `CTYPE`, so for `double` inputs
>    there is a narrowing to `float` before the final return converts back to
>    `CTYPE`; a faithful port must reproduce this `float` intermediate to match
>    bit-for-bit.)
> 2. If exactly one of `a`, `b` is negative (`(a < 0) ^ (b < 0)`) and
>    `rem != 0`, add `b` to `rem` so the sign matches `b`.
> 3. Return `rem` (converted to `CTYPE`).
>
> Integral overload (for completeness): return `a % b` directly.
>
> Pure scalar helper; no side effects.

