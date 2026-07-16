# kernels/portable/cpu/util/vectorized_math.h

> [spec:et:def:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn]
> auto convert_to_vectorized_n_of_float(at::vec::Vectorized<T> vec)

> [spec:et:sem:vectorized-math.executorch.math.internal.convert-to-vectorized-n-of-float-fn]
> Only compiled when `ET_USE_PYTORCH_HEADERS` is set (uses ATen `at::vec`). Widens
> a SIMD vector of a non-float element type `T` into an equivalent group of
> `float` SIMD vectors, so float-only math (e.g. transcendental functions) can be
> applied to reduced-precision inputs like Half/BFloat16.
>
> 1. `float_vec_size = at::vec::Vectorized<float>::size()` (float lanes per vector).
> 2. `t_vec_size = at::vec::Vectorized<T>::size()` (T lanes per vector).
> 3. `result_size = (t_vec_size < float_vec_size) ? 1 : t_vec_size / float_vec_size`
>    — the number of float vectors needed to hold all `t_vec_size` lanes (at least
>    1; `static_assert(result_size >= 1)`).
> 4. Returns `at::vec::convert<float, result_size, T, 1, keep=true>(at::vec::VectorizedN<T, 1>(vec))`
>    — the input single `T` vector converted to a `VectorizedN<float, result_size>`.
> A Rust port targets the same numeric conversion of each `T` lane to `f32`.

> [spec:et:def:vectorized-math.executorch.math.rsqrt-fn]
> T rsqrt(T x)

> [spec:et:sem:vectorized-math.executorch.math.rsqrt-fn]
> Scalar reciprocal square root for a floating-point type `T` (the template is
> enabled only for floating-point `T`). Returns `T(1) / std::sqrt(x)`, i.e.
> `1/sqrt(x)` computed in `T` precision. Follows `std::sqrt` semantics for special
> inputs: `sqrt` of a negative `x` is NaN so the result is NaN; `x == 0` yields
> `+inf` (or `-inf` for negative zero via IEEE division); `x == +inf` yields `+0`;
> NaN input yields NaN. (A separate vectorized `rsqrt` overload exists for SIMD
> vectors under `ET_USE_PYTORCH_HEADERS`, delegating to the vector `rsqrt` after
> converting non-float lanes to float; this scalar rule covers the plain `T` path.)

