# kernels/portable/cpu/vec_ops.h

> [spec:et:def:vec-ops.torch.executor.bounds-min-fn]
> static inline size_t bounds_min(size_t a, size_t b)

> [spec:et:sem:vec-ops.torch.executor.bounds-min-fn]
> Returns the smaller of the two `size_t` arguments `a` and `b`.
> Evaluates `(a < b) ? a : b`: if `a` is strictly less than `b` it
> returns `a`, otherwise it returns `b` (so when `a == b` it returns
> `b`, which is numerically identical). No overflow or wraparound
> concerns because it only compares, never combines. This is a private
> helper used by `vec_quantized_matmul_transb_int8` to clamp the upper
> bound of the last (possibly short) quantization group to the true
> column count `n`.

> [spec:et:def:vec-ops.torch.executor.dequantize-i8-f32-fn]
> inline void dequantize_i8_f32( float* ET_RESTRICT y, const int8_t* ET_RESTRICT x, float scale, int32_t zero_point, size_t size)

> [spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn]
> Dequantizes `size` int8 elements from `x` into `size` float elements
> in `y`, using an affine transform with scalar `scale` (float) and
> `zero_point` (int32). Pointers `x` and `y` are restrict-qualified so
> they must not alias. For each index `i` in `[0, size)` in ascending
> order:
> - Compute `y[i] = scale * (x[i] - zero_point)`. The subtraction
>   `x[i] - zero_point` is performed in `int32_t` arithmetic (the int8
>   `x[i]` promotes to int, subtracting the int32 `zero_point`), then
>   that int32 result is converted to float and multiplied by `scale`
>   in float.
> No clamping, rounding, or bounds checks are applied. If `size == 0`
> nothing is written. This is the exact inverse of
> `[spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn]`.

> [spec:et:def:vec-ops.torch.executor.quantize-i8-f32-fn]
> inline void quantize_i8_f32( int8_t* ET_RESTRICT y, const float* ET_RESTRICT x, float scale, int32_t zero_point, size_t size)

> [spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn]
> Quantizes `size` float elements from `x` into `size` int8 elements in
> `y`, using scalar `scale` (float) and `zero_point` (int32). Pointers
> `x` and `y` are restrict-qualified so they must not alias. For each
> index `i` in `[0, size)` in ascending order:
> - Compute `tmp = std::round(x[i] * scale + zero_point)` entirely in
>   float. Note the affine form here MULTIPLIES by `scale` (i.e. the
>   caller passes the reciprocal of the dequant scale). `std::round`
>   rounds half away from zero (ties go to the larger-magnitude
>   integer, e.g. 0.5 -> 1.0, -0.5 -> -1.0), returning a float.
> - Clamp `tmp` to the closed range `[-128.0f, 127.0f]` via
>   `[spec:et:sem:vec-ops.torch.executor.internal.clamp-fn]` (lo =
>   -128.0f, hi = 127.0f), then narrow the clamped float to `int8_t`
>   and store in `y[i]`. Because of the prior clamp the float is always
>   in signed-8-bit range, so the float->int8 conversion (truncation
>   toward zero of an already-integral value) is exact.
> No NaN handling is special-cased: if `x[i]` produces NaN then
> `std::round` yields NaN and the clamp comparisons `v < lo` / `hi < v`
> are both false, so NaN passes through the clamp and the subsequent
> narrowing to int8 is implementation-defined. If `size == 0` nothing
> is written. Inverse of
> `[spec:et:sem:vec-ops.torch.executor.dequantize-i8-f32-fn]`.

> [spec:et:def:vec-ops.torch.executor.reduce-add-fn]
> inline float reduce_add(const T* x, size_t size)

> [spec:et:sem:vec-ops.torch.executor.reduce-add-fn]
> Returns the sum of the `size` elements at `x`, computed as a `float`.
> Template parameter `T` is the element type of `x` (any type
> convertible to/addable with float). Implemented via
> `std::accumulate(x, x + size, 0.f)`: the accumulator starts at
> `0.0f` (float) and is updated left-to-right as
> `acc = acc + x[i]` for `i` in `[0, size)` in ascending order. Because
> the accumulator is float, each `x[i]` is promoted to float (or the
> addition is done in float) before adding; the running sum is kept in
> single precision, so ordering-dependent float rounding applies. For
> `size == 0` it returns `0.0f`. Does not mutate `x`.

> [spec:et:def:vec-ops.torch.executor.vec-addf-fn]
> inline void vec_addf( float* ET_RESTRICT z, const float* ET_RESTRICT x, const float* ET_RESTRICT y, size_t size)

> [spec:et:sem:vec-ops.torch.executor.vec-addf-fn]
> Elementwise adds two float arrays `x` and `y`, each of `size`
> elements, writing the result into `z`. All three pointers are
> restrict-qualified: `z`, `x`, and `y` must not alias one another. For
> each index `i` in `[0, size)` in ascending order, computes
> `z[i] = x[i] + y[i]` in float. No broadcasting, no bounds checks; all
> arrays must have exactly `size` elements. For `size == 0` nothing is
> written. Standard IEEE-754 float addition semantics (NaN/inf
> propagate).

> [spec:et:def:vec-ops.torch.executor.vec-maxf-fn]
> inline float vec_maxf(const float* x, size_t size)

> [spec:et:sem:vec-ops.torch.executor.vec-maxf-fn]
> Returns the maximum element of the float array `x`, which must have
> `size` elements. Implemented as `*std::max_element(x, x + size)`:
> scans `x[0..size)` and returns the largest element under the default
> `operator<` comparison. `std::max_element` returns the FIRST maximal
> element on ties (though for equal values the returned value is
> identical). PRECONDITION: `size >= 1`; if `size == 0` the range is
> empty and `std::max_element` returns `x + size == x`, which is then
> dereferenced — undefined behavior, so callers must guarantee at least
> one element. NaN handling follows `std::max_element`/`operator<`: NaN
> comparisons are false, so a NaN does not displace the current best and
> the result is comparison-order-dependent (not a documented "max
> ignores NaN"). Does not mutate `x`.

> [spec:et:def:vec-ops.torch.executor.vec-minf-fn]
> inline float vec_minf(const float* x, size_t size)

> [spec:et:sem:vec-ops.torch.executor.vec-minf-fn]
> Returns the minimum element of the float array `x`, which must have
> `size` elements. Implemented as `*std::min_element(x, x + size)`:
> scans `x[0..size)` and returns the smallest element under the default
> `operator<` comparison. `std::min_element` returns the FIRST minimal
> element on ties. PRECONDITION: `size >= 1`; if `size == 0` the range
> is empty and `std::min_element` returns `x + size == x`, which is
> then dereferenced — undefined behavior, so callers must guarantee at
> least one element. NaN handling follows `std::min_element`/
> `operator<`: NaN comparisons are false, so a NaN does not displace
> the current best and the result is comparison-order-dependent. Does
> not mutate `x`.

> [spec:et:def:vec-ops.torch.executor.vec-powerf-fn]
> inline float vec_powerf(const T* x, size_t size)

> [spec:et:sem:vec-ops.torch.executor.vec-powerf-fn]
> Returns the sum of squares of the `size` elements at `x`, computed as
> a `float`. Template parameter `T` is the element type. Initializes a
> float accumulator `sum = 0`, then for each index `i` in `[0, size)`
> in ascending order adds `static_cast<float>(x[i]) * x[i]` to `sum`.
> Note only the LEFT operand is explicitly cast to float; the right
> `x[i]` is the raw `T`, but the multiplication promotes it to float,
> so each term is the float square of `x[i]`. The running sum is kept
> in single precision, giving order-dependent float rounding. Returns
> `0.0f` for `size == 0`. Does not mutate `x`.

> [spec:et:def:vec-ops.torch.executor.vec-scalef-fn]
> inline void vec_scalef( float* ET_RESTRICT y, const float* ET_RESTRICT x, float scale, size_t size)

> [spec:et:sem:vec-ops.torch.executor.vec-scalef-fn]
> Multiplies every element of float array `x` by the float scalar
> `scale`, writing the result into float array `y`. Both `x` and `y`
> have `size` elements and are restrict-qualified (must not alias). For
> each index `i` in `[0, size)` in ascending order, computes
> `y[i] = x[i] * scale` in float. No bounds checks; for `size == 0`
> nothing is written. Standard IEEE-754 float multiply semantics
> (NaN/inf propagate).

> [spec:et:def:vec-ops.torch.executor.internal.clamp-fn]
> constexpr const T& clamp(const T& v, const T& lo, const T& hi)

> [spec:et:sem:vec-ops.torch.executor.internal.clamp-fn]
> Clamps value `v` to the closed interval `[lo, hi]` and returns a
> const reference to the selected input. Generic over type `T` ordered
> by `operator<`. Two implementations selected at compile time by the
> `__cpp_lib_clamp` feature macro, but they are behaviorally equivalent:
> - If `std::clamp` is available: returns `std::clamp(v, lo, hi)`.
> - Otherwise: returns `v < lo ? lo : (hi < v ? hi : v)` — i.e. if `v`
>   is less than `lo` return `lo`; else if `hi` is less than `v` return
>   `hi`; else return `v` unchanged.
> PRECONDITION: `lo <= hi` (otherwise the result is unspecified, as in
> std::clamp). If `v` is NaN under a float `T`, both `v < lo` and
> `hi < v` are false, so `v` (the NaN) is returned unchanged. This is
> used by `[spec:et:sem:vec-ops.torch.executor.quantize-i8-f32-fn]` to
> clamp to `[-128.0f, 127.0f]`.

