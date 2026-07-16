# runtime/core/exec_aten/util/tensor_util.h

> [spec:et:def:tensor-util.executorch.calculate-linear-index-fn]
> inline size_t calculate_linear_index( const executorch::aten::SizesType* coordinate, const executorch::aten::StridesType* strides, const size_t ndim)

> [spec:et:sem:tensor-util.executorch.calculate-linear-index-fn]
> Computes a flat linear buffer index from an n-dimensional coordinate and an
> explicit strides array. Inputs: `coordinate` (pointer to `ndim` SizesType
> values), `strides` (pointer to `ndim` StridesType values), and `ndim`.
> Initialize `index = 0`; for `i` in `0..ndim` (ascending), add
> `coordinate[i] * strides[i]` to `index` (accumulated as `size_t`). Return
> `index`. No bounds/validation is performed. For `ndim == 0` returns 0. Unlike
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`,
> this uses caller-supplied strides rather than deriving trailing-dim products,
> so it honors arbitrary (e.g. non-contiguous or channels-last) layouts.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]
> inline size_t coordinateToIndex( const executorch::aten::Tensor& tensor, const size_t* const coordinate)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]
> Computes a flat linear buffer index for an element at the given n-dimensional
> `coordinate` within `tensor`, assuming a default (contiguous, row-major)
> layout derived from the tensor's sizes. `coordinate` is a pointer to at least
> `tensor.dim()` `size_t` values (the API contract assumes the array has
> `kTensorDimensionLimit` elements). Initialize `index = 0`; for each dimension
> `d` in `0..tensor.dim()` (ascending), add
> `coordinate[d] * getTrailingDims(tensor, d)` to `index`, where
> `getTrailingDims(tensor, d)` is the product of the sizes of all dimensions
> after `d` per
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]`.
> Return `index` (`size_t`). For a 0-dim tensor the loop does not execute and 0
> is returned. This ignores the tensor's actual strides; it always uses the
> contiguous stride implied by sizes.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-with-trailing-dims-memo-fn]
> inline size_t coordinateToIndexWithTrailingDimsMemo( const executorch::aten::Tensor& tensor, const size_t* const coordinate, const size_t trailing_dims_memo[kTensorDimensionLimit])

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-with-trailing-dims-memo-fn]
> Like
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`
> but uses a precomputed table of trailing-dim products instead of recomputing
> them per dimension, so it is faster for repeated calls on the same tensor.
> Inputs: `tensor`, `coordinate` (pointer to at least `tensor.dim()` `size_t`
> values), and `trailing_dims_memo` (a `size_t[kTensorDimensionLimit]` array
> that must have been filled by
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn]`
> for this same tensor). Initialize `index = 0`; for each dimension `d` in
> `0..tensor.dim()` (ascending), add `coordinate[d] * trailing_dims_memo[d]` to
> `index`. Return `index`. For a 0-dim tensor returns 0. Produces identical
> results to `coordinateToIndex` for the same coordinate.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.dim-is-valid-fn]
> inline bool dim_is_valid(int64_t dim, int64_t upper_bound)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.dim-is-valid-fn]
> Returns whether `dim` is a valid (possibly negative) dimension index against
> `upper_bound`. Evaluates the condition `dim >= -upper_bound && dim <
> upper_bound` (i.e. `dim` lies in the closed range `[-upper_bound, upper_bound
> - 1]`). Wrapped in `ET_CHECK_OR_RETURN_FALSE`: if the condition is false, logs
> an error message naming `dim`, `-upper_bound`, and `upper_bound - 1`, and
> returns `false`. Otherwise returns `true`. No context/side effects beyond
> logging.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn]
> bool extract_scalar_tensor(executorch::aten::Tensor tensor, INT_T* out_val)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn]
> Integer overload (SFINAE-selected when `INT_T` is an integral type other than
> `bool`). Extracts the single scalar element of `tensor` into `*out_val` as an
> `INT_T`, with range checking.
>
> Steps:
> 1. If `tensor.numel() != 1`, return `false` (do not write `*out_val`).
> 2. Switch on `tensor.scalar_type()`. Only the integer scalar types enumerated
>    by `ET_FORALL_INT_TYPES` are handled: Byte (uint8), Char (int8), Short
>    (int16), Int (int32), Long (int64). For any other dtype (including Bool,
>    floating, complex, bits) fall through to `default` and return `false`.
> 3. For the matched integer type with C type `TENSOR_CTYPE`, read the element
>    at buffer offset 0 via `tensor.const_data_ptr<TENSOR_CTYPE>()[0]` (element
>    0 of the data buffer; assumes contiguous scalar).
> 4. Range-check: if `val < std::numeric_limits<INT_T>::lowest()` or `val >
>    std::numeric_limits<INT_T>::max()`, return `false` without writing.
> 5. Otherwise set `*out_val = static_cast<INT_T>(val)` and return `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn]
> inline size_t getLeadingDims( const executorch::aten::Tensor& tensor, int64_t dim)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-leading-dims-fn]
> Returns the product of the tensor's sizes over dimensions `[0, dim)` (i.e. all
> dimensions strictly before `dim`, not including `dim`).
>
> Steps:
> 1. Validate via `ET_CHECK_MSG` that `dim >= 0 && dim <= tensor.dim()`; note
>    `dim` may equal `tensor.dim()` (yielding the product of all dims, i.e.
>    numel). On failure this is a fatal `ET_CHECK` (aborts), not a recoverable
>    error.
> 2. Initialize `dims = 1` (`size_t`).
> 3. For `i` in `0..dim` (ascending), multiply `dims` by
>    `static_cast<size_t>(tensor.size(i))`, checking for `size_t` multiplication
>    overflow with `c10::mul_overflows`; on overflow, fatal `ET_CHECK_MSG`.
> 4. Return `dims`. For `dim == 0` returns 1.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]
> inline size_t getTrailingDims( const executorch::aten::Tensor& tensor, int64_t dim)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]
> Returns the product of the tensor's sizes over dimensions `(dim, tensor.dim())`
> (i.e. all dimensions strictly after `dim`), i.e. the number of contiguous
> elements spanned by a single step along `dim`.
>
> Steps:
> 1. Validate via `ET_CHECK_MSG` that `dim >= -1 && dim < tensor.dim()`; `dim ==
>    -1` yields the product of all dims (numel). On failure this is a fatal
>    `ET_CHECK` (aborts).
> 2. Initialize `dims = 1` (`size_t`).
> 3. For `i` from `dim + 1` up to (exclusive) `tensor.dim()` (ascending),
>    multiply `dims` by `static_cast<size_t>(tensor.size(i))`, checking for
>    `size_t` multiplication overflow with `c10::mul_overflows`; on overflow,
>    fatal `ET_CHECK_MSG`.
> 4. Return `dims`. For `dim == tensor.dim() - 1` returns 1.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]
> inline void indexToCoordinate( const executorch::aten::Tensor& tensor, size_t index, size_t* coordinate)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.index-to-coordinate-fn]
> Inverse of
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.coordinate-to-index-fn]`:
> given a flat `index` (assuming default contiguous row-major layout), writes
> the corresponding n-dimensional coordinate into `coordinate` (pointer to at
> least `tensor.dim()` `size_t` values).
>
> Steps:
> 1. `ET_CHECK` (fatal) that `index < static_cast<size_t>(tensor.numel())`.
> 2. Iterate dimensions from last to first: for `i` in `0..tensor.dim()`, let
>    `dim = tensor.dim() - 1 - i`, `dim_size = tensor.size(dim)`; set
>    `coordinate[dim] = index % dim_size`, then `index /= dim_size`.
> 3. Returns void; result written in-place. For a 0-dim tensor the loop does not
>    execute (index must be 0 to pass the check).

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn]
> inline void memoizeTrailingDims( const executorch::aten::Tensor& tensor, size_t trailing_dims_memo[kTensorDimensionLimit])

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.memoize-trailing-dims-fn]
> Fills `trailing_dims_memo` (a `size_t[kTensorDimensionLimit]` array) so that
> `trailing_dims_memo[d]` equals the product of `tensor.size(k)` for all `k > d`
> (the same value as
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.get-trailing-dims-fn]`
> for that `d`), enabling fast repeated coordinate-to-index conversion.
>
> Steps:
> 1. Let `tensorDim = tensor.dim()`; initialize `dims = 1` (`size_t`).
> 2. For `ii` from `tensorDim - 1` down to `0` (descending): set
>    `trailing_dims_memo[ii] = dims`, then `dims *=
>    static_cast<size_t>(tensor.size(ii))`.
> 3. Returns void. No overflow checking is performed here (unlike
>    `getTrailingDims`). For a 0-dim tensor no entries are written. Only entries
>    `[0, tensorDim)` are written; higher indices are left untouched.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn]
> inline ssize_t nonempty_size( const executorch::aten::Tensor& tensor, ssize_t dim)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn]
> Returns the size of `tensor` along dimension `dim`, except that a 0-dim
> (scalar) tensor is treated as a 1D tensor with a single element. Concretely:
> if `tensor.dim() == 0` return `1`; otherwise return `tensor.size(dim)`
> (`ssize_t`). No bounds validation on `dim` beyond what `tensor.size` itself
> does.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.nonzero-dim-fn]
> inline ssize_t nonzero_dim(const executorch::aten::Tensor& tensor)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonzero-dim-fn]
> Returns the tensor's number of dimensions, except that a 0-dim (scalar) tensor
> is treated as 1D. Concretely: if `tensor.dim() == 0` return `1`; otherwise
> return `tensor.dim()` (`ssize_t`). Companion to
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.nonempty-size-fn]`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-can-cast-to-fn]
> inline bool tensor_can_cast_to( executorch::aten::Tensor a, executorch::aten::ScalarType dtype)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-can-cast-to-fn]
> Returns whether tensor `a`'s dtype can be cast to `dtype` per the runtime's
> type-promotion cast rules. Evaluates `canCast(a.scalar_type(), dtype)` (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.can-cast-fn]`), which
> broadly permits casts except: complex→non-complex, and floating→integral (a
> narrowing that would lose the fractional/complex part). Wrapped in
> `ET_CHECK_OR_RETURN_FALSE`: if the cast is not allowed, logs an error naming
> both dtype strings and returns `false`; otherwise returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-dim-has-index-fn]
> inline bool

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-dim-has-index-fn]
> Returns whether index `ix` is in range for dimension `d` of tensor `t`,
> supporting negative `d` and negative `ix`.
>
> Steps:
> 1. `ET_CHECK` (fatal) that `t.dim() != 0` (indexing ops do not support 0-dim
>    tensors).
> 2. If `d < 0`, set `d += t.dim()` (normalize to non-negative).
> 3. `ET_CHECK` (fatal) that `d >= 0 && d < t.dim()` — the dimension must have
>    already been validated (e.g. by
>    `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]`).
> 4. `ET_CHECK_OR_RETURN_FALSE` that `ix >= -t.size(d) && ix < t.size(d)` (i.e.
>    `ix` in `[-size, size)`); on failure log and return `false`.
> 5. Return `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]
> inline bool tensor_has_dim(executorch::aten::Tensor t, int64_t d)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]
> Returns whether `d` is a valid dimension index for tensor `t` (supporting
> negative indexing), logging and returning `false` when not.
>
> Steps:
> 1. If `t.dim() == 0`: `ET_CHECK_OR_RETURN_FALSE` that `d == 0 || d == -1`
>    (only these are valid for a 0-dim tensor); else log and return `false`.
> 2. Else (`t.dim() > 0`): `ET_CHECK_OR_RETURN_FALSE` on the condition
>    `d > 0 ? d < t.dim() : t.dim() + d >= 0`. That is, for positive `d` require
>    `d < t.dim()`; for `d <= 0` require `t.dim() + d >= 0`, which accepts
>    `d == 0` and negatives down to `-t.dim()`. On failure log and return
>    `false`.
> 3. Return `true`.
> Note: for positive `d`, `d == 0` is not covered by the `d > 0` branch and
> falls to the else branch (`t.dim() + 0 >= 0`, always true), so `d == 0` is
> always accepted for a nonzero-rank tensor.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-expected-size-fn]
> inline bool tensor_has_expected_size( executorch::aten::Tensor a, executorch::aten::ArrayRef<executorch::aten::SizesType> expected_sizes)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-expected-size-fn]
> Returns whether `a.sizes()` exactly equals `expected_sizes` (element-wise and
> same length). Compares `a.sizes() == expected_sizes` (ArrayRef equality). If
> unequal: logs an error with `a.dim()` and `expected_sizes.size()`, then logs
> the mismatching `size(d)` pairs for `d` in `0..min(a.dim(),
> expected_sizes.size())`, and returns `false`. If equal, returns `true`. Note
> this is a strict shape equality check, not broadcast-compatibility.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-non-empty-dim-fn]
> inline bool tensor_has_non_empty_dim(executorch::aten::Tensor t, int64_t d)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-non-empty-dim-fn]
> Returns whether dimension `d` exists in `t` and has non-zero size.
>
> Steps:
> 1. Normalize the index: `udim = d < 0 ? d + t.dim() : d` (via
>    `ET_NORMALIZE_IX`), computed before validation, so a wildly out-of-range
>    `d` may yield an out-of-bounds `udim` — callers rely on the subsequent
>    checks.
> 2. `ET_LOG_AND_RETURN_IF_FALSE(tensor_has_dim(t, d))`: return `false` if `d`
>    is not a valid dim per
>    `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-dim-fn]`.
> 3. `ET_LOG_AND_RETURN_IF_FALSE(t.size(udim) != 0)`: return `false` if that
>    dimension's size is 0.
> 4. Return `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-greater-or-equal-to-fn]
> inline bool tensor_has_rank_greater_or_equal_to( executorch::aten::Tensor t, size_t rank)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-greater-or-equal-to-fn]
> Returns whether `t.dim() >= rank`. `ET_CHECK_OR_RETURN_FALSE` on
> `static_cast<size_t>(t.dim()) >= rank`; on failure logs the expected and
> actual rank and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-smaller-or-equal-to-fn]
> inline bool tensor_has_rank_smaller_or_equal_to( executorch::aten::Tensor t, size_t rank)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-has-rank-smaller-or-equal-to-fn]
> Returns whether `t.dim() <= rank`. `ET_CHECK_OR_RETURN_FALSE` on
> `static_cast<size_t>(t.dim()) <= rank`; on failure logs the expected and
> actual rank and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-bits-type-fn]
> inline bool tensor_is_bits_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bits-type-fn]
> Returns whether `t`'s dtype is a "bits" type. `ET_CHECK_OR_RETURN_FALSE` on
> `isBitsType(t.scalar_type())`, which is true iff the dtype is one of Bits1x8,
> Bits2x4, Bits4x2, Bits8, or Bits16 (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-bits-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-bool-type-fn]
> inline bool tensor_is_bool_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-bool-type-fn]
> Returns whether `t`'s dtype is exactly `ScalarType::Bool`.
> `ET_CHECK_OR_RETURN_FALSE` on `t.scalar_type() == ScalarType::Bool`; on
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-complex-type-fn]
> inline bool tensor_is_complex_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-complex-type-fn]
> Returns whether `t`'s dtype is a complex type. `ET_CHECK_OR_RETURN_FALSE` on
> `isComplexType(t.scalar_type())`, true iff the dtype is one of ComplexHalf,
> ComplexFloat, or ComplexDouble (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-complex-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-contiguous-fn]
> inline bool tensor_is_contiguous(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-contiguous-fn]
> Returns whether `t` has fully contiguous (row-major/default) strides.
>
> Steps:
> 1. Let `strides = t.strides()`, `sizes = t.sizes()`.
> 2. If `strides.size() == 0` (0-dim/scalar tensor), return `true` (scalars are
>    contiguous).
> 3. `ET_CHECK_OR_RETURN_FALSE` that the last stride equals 1:
>    `strides[strides.size() - 1] == 1`; on failure log and return `false`.
> 4. For `i` from `strides.size() - 1` down to `1` (descending),
>    `ET_CHECK_OR_RETURN_FALSE` that `strides[i - 1] == strides[i] * sizes[i]`;
>    on any failure log and return `false`.
> 5. Return `true`. This is the exact contiguity invariant: each dimension's
>    stride is the product of the size and stride of the next inner dimension.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-floating-type-fn]
> inline bool tensor_is_floating_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-floating-type-fn]
> Returns whether `t`'s dtype is a floating type. `ET_CHECK_OR_RETURN_FALSE` on
> `isFloatingType(t.scalar_type())`, true iff the dtype is one of Double, Float,
> Half, or BFloat16 (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-floating-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-integral-type-fn]
> inline bool tensor_is_integral_type( executorch::aten::Tensor t, bool includeBool = false)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-integral-type-fn]
> Returns whether `t`'s dtype is an integral type. Takes an optional
> `includeBool` flag defaulting to `false`. `ET_CHECK_OR_RETURN_FALSE` on
> `isIntegralType(t.scalar_type(), includeBool)`, true iff the dtype is one of
> Byte, Char, Short, Int, Long — plus Bool when `includeBool` is true (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-integral-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-rank-fn]
> inline bool tensor_is_rank(executorch::aten::Tensor t, size_t rank)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-rank-fn]
> Returns whether `t.dim()` exactly equals `rank`. `ET_CHECK_OR_RETURN_FALSE` on
> `static_cast<size_t>(t.dim()) == rank`; on failure logs the expected and
> actual rank and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-real-type-fn]
> inline bool tensor_is_real_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-real-type-fn]
> Returns whether `t`'s dtype is a "real" type. `ET_CHECK_OR_RETURN_FALSE` on
> `isRealType(t.scalar_type())`, true iff the dtype is one of Byte, Char, Short,
> Int, Long, Float, or Double — i.e. the integer and standard floating types,
> excluding Half, BFloat16, Bool, complex, and bits (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realh-type-fn]
> inline bool tensor_is_realh_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realh-type-fn]
> Returns whether `t`'s dtype is a real-or-Half type ("realh").
> `ET_CHECK_OR_RETURN_FALSE` on `isRealHType(t.scalar_type())`, true iff the
> dtype is one of Byte, Char, Short, Int, Long, Float, Double, or Half — i.e.
> the real set plus Half (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-h-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhb-type-fn]
> inline bool tensor_is_realhb_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhb-type-fn]
> Returns whether `t`'s dtype is a real-or-Half-or-Bool type ("realhb").
> `ET_CHECK_OR_RETURN_FALSE` on `isRealHBType(t.scalar_type())`, true iff
> `isRealHType(t)` is true OR the dtype is Bool — i.e. Byte, Char, Short, Int,
> Long, Float, Double, Half, or Bool (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-hb-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbbf16-type-fn]
> inline bool tensor_is_realhbbf16_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbbf16-type-fn]
> Returns whether `t`'s dtype is a real-or-Half-or-Bool-or-BFloat16 type
> ("realhbbf16"). `ET_CHECK_OR_RETURN_FALSE` on
> `isRealHBBF16Type(t.scalar_type())`, true iff `isRealHBType(t)` is true OR the
> dtype is BFloat16 — i.e. Byte, Char, Short, Int, Long, Float, Double, Half,
> Bool, or BFloat16 (see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-hbbf16-type-fn]`).
> On failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbf16-type-fn]
> inline bool tensor_is_realhbf16_type(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-realhbf16-type-fn]
> Returns whether `t`'s dtype is a real-or-Half-or-BFloat16 type ("realhbf16").
> `ET_CHECK_OR_RETURN_FALSE` on `isRealHBF16Type(t.scalar_type())`, true iff
> `isRealHType(t)` is true OR the dtype is BFloat16 — i.e. Byte, Char, Short,
> Int, Long, Float, Double, Half, or BFloat16 (this set excludes Bool; see
> `[spec:et:sem:scalar-type-util.executorch.runtime.is-real-hbf16-type-fn]`). On
> failure logs the actual dtype and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-scalar-fn]
> inline bool tensor_is_scalar(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-scalar-fn]
> Returns `true` iff `t` is a scalar tensor: `t.dim() == 0 && t.numel() == 1`.
> Pure predicate, no logging or side effects.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensor-is-type-fn]
> inline bool tensor_is_type( executorch::aten::Tensor t, executorch::aten::ScalarType dtype, executorch::aten::ScalarType dtype2, executorch::aten::ScalarType dtype3)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensor-is-type-fn]
> Returns whether `t`'s dtype matches any one of three candidate dtypes. This is
> the three-dtype overload of `tensor_is_type` (there are also one- and
> two-dtype overloads with analogous behavior). `ET_CHECK_OR_RETURN_FALSE` (via
> `ET_LOG_MSG_AND_RETURN_IF_FALSE`) on `t.scalar_type() == dtype ||
> t.scalar_type() == dtype2 || t.scalar_type() == dtype3`; on failure logs the
> three expected dtype strings and the actual dtype, and returns `false`, else
> returns `true`. The one-dtype overload checks equality with a single `dtype`;
> the two-dtype overload checks against `dtype` or `dtype2`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn]
> inline bool tensors_have_same_dtype( executorch::aten::Tensor a, executorch::aten::Tensor b, executorch::aten::Tensor c)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn]
> Three-tensor overload. Returns whether tensors `a`, `b`, `c` all share the
> same dtype. `ET_CHECK_OR_RETURN_FALSE` on `a.scalar_type() == b.scalar_type()
> && b.scalar_type() == c.scalar_type()`; on failure logs the three dtype
> strings and returns `false`, else returns `true`. (The two-tensor overload
> checks `a.scalar_type() == b.scalar_type()`.)

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-rank-fn]
> inline bool tensors_have_same_rank( executorch::aten::Tensor a, executorch::aten::Tensor b)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-rank-fn]
> Returns whether tensors `a` and `b` have the same number of dimensions.
> `ET_CHECK_OR_RETURN_FALSE` on `a.dim() == b.dim()`; on failure logs both ranks
> and returns `false`, else returns `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-and-dtype-fn]
> inline bool tensors_have_same_shape_and_dtype( executorch::aten::Tensor a, executorch::aten::Tensor b, executorch::aten::Tensor c)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-and-dtype-fn]
> Three-tensor overload. Returns `tensors_have_same_shape(a, b, c) &&
> tensors_have_same_dtype(a, b, c)` — i.e. all three tensors have the same shape
> per
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn]`
> AND the same dtype per
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-dtype-fn]`.
> Short-circuits: if shapes differ, dtype is not checked. Each sub-call logs its
> own diagnostics on failure. (The two-tensor overload is the analogous `&&` of
> the two-tensor shape and dtype checks.)

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn]
> inline bool tensors_have_same_shape( executorch::aten::Tensor a, executorch::aten::Tensor b, executorch::aten::Tensor c)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-shape-fn]
> Three-tensor overload. Returns whether `a`, `b`, `c` have the same shape,
> treating all scalar tensors as equal-shaped.
>
> Steps:
> 1. If `a.numel() == 1 && b.numel() == 1 && c.numel() == 1`, return `true` —
>    PyTorch treats all single-element (scalar) tensors as the same shape even
>    with differing dims.
> 2. Compute `cond1 = (a.sizes() == b.sizes()) && (a.numel() == b.numel())` and
>    `cond2 = (b.sizes() == c.sizes()) && (b.numel() == c.numel())` (ArrayRef
>    element-wise equality plus numel equality).
> 3. If `!(cond1 && cond2)`: log the numels/dims and the per-dimension size
>    triples for `d` in `0..min3(a.dim(), b.dim(), c.dim())`, then return
>    `false`.
> 4. Otherwise return `true`.
> (The two-tensor overload is analogous: scalar shortcut on both numel==1, else
> requires `a.sizes() == b.sizes() && a.numel() == b.numel()`.)

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-size-at-dims-fn]
> inline bool tensors_have_same_size_at_dims( executorch::aten::Tensor a, size_t dim_a, executorch::aten::Tensor b, size_t dim_b)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-size-at-dims-fn]
> Returns whether tensor `a` at dimension `dim_a` and tensor `b` at dimension
> `dim_b` have equal size. Inputs: `a`, `dim_a` (`size_t`), `b`, `dim_b`
> (`size_t`). `dim_a` and `dim_b` are non-negative (`size_t`); no negative-index
> normalization is done.
>
> Steps:
> 1. `ET_CHECK_OR_RETURN_FALSE` that `dim_a < static_cast<size_t>(a.dim())`; on
>    failure log the requested dim and `a.dim()`, and return `false`.
> 2. `ET_CHECK_OR_RETURN_FALSE` that `dim_b < static_cast<size_t>(b.dim())`; on
>    failure log the requested dim and `b.dim()`, and return `false`.
> 3. `ET_CHECK_OR_RETURN_FALSE` that `a.size(dim_a) == b.size(dim_b)`; on failure
>    log both dims and both sizes, and return `false`.
> 4. Return `true`.

> [spec:et:def:tensor-util.executorch.et-runtime-namespace.tensors-have-same-strides-fn]
> inline bool tensors_have_same_strides( executorch::aten::Tensor a, executorch::aten::Tensor b, executorch::aten::Tensor c)

> [spec:et:sem:tensor-util.executorch.et-runtime-namespace.tensors-have-same-strides-fn]
> Three-tensor overload. Returns whether tensors `a`, `b`, `c` have identical
> strides (element-wise and same length).
>
> Steps:
> 1. Evaluate `a.strides() == b.strides() && b.strides() == c.strides()`
>    (ArrayRef equality: same length and equal elements). This equality is a
>    diagnostic-only check that does NOT verify contiguity.
> 2. If the condition is false: log an error with `a.dim()`, `b.dim()`, `c.dim()`,
>    then for `d` in `0..min3(a.dim(), b.dim(), c.dim())` log the per-dimension
>    stride triple `(a.strides()[d], b.strides()[d], c.strides()[d])`, and return
>    `false`.
> 3. Otherwise return `true`. (The two-tensor overload is analogous, comparing
>    only `a.strides() == b.strides()` and logging stride pairs; note it uses
>    `!=` on ArrayRefs to trigger the failure branch, which is equivalent.)

> [spec:et:def:tensor-util.executorch.extract-scalar-tensor-fn]
> bool extract_scalar_tensor(executorch::aten::Tensor tensor, BOOL_T* out_val)

> [spec:et:sem:tensor-util.executorch.extract-scalar-tensor-fn]
> Boolean overload (SFINAE-selected when `BOOL_T` is exactly `bool`). Extracts
> the single boolean scalar element of `tensor` into `*out_val`.
>
> Steps:
> 1. If `tensor.scalar_type() != ScalarType::Bool`, return `false` (do not write
>    `*out_val`). Note the dtype check comes first, before the numel check.
> 2. If `tensor.numel() != 1`, return `false` (do not write `*out_val`).
> 3. Read `val = tensor.const_data_ptr<bool>()[0]` (element 0 of the data
>    buffer).
> 4. Set `*out_val = static_cast<BOOL_T>(val)` (i.e. `val` itself) and return
>    `true`. No range checking is applicable.
>
> This is one of three `extract_scalar_tensor` overloads; the integer overload is
> `[spec:et:sem:tensor-util.executorch.et-runtime-namespace.extract-scalar-tensor-fn]`.
> There is also a floating-point overload (SFINAE-selected when `FLOAT_T` is a
> C++ floating type, `BFloat16`, or `Half`) which: returns `false` if
> `tensor.numel() != 1`; switches on `scalar_type()` over the `realhbf16` type
> set (`ET_FORALL_REALHBF16_TYPES`: Byte, Char, Short, Int, Long, Float, Double,
> Half, BFloat16 — real types plus Half and BFloat16, excluding Bool and complex
> and bits; any other dtype falls to `default` and returns `false`); reads
> element 0 and widens it to `double val`; if `std::isfinite(val)` is true AND
> `val` is outside `[lowest, max]` of `FLOAT_T`, returns `false` (non-finite
> values such as NaN/inf skip the range check and are passed through); otherwise
> sets `*out_val = static_cast<FLOAT_T>(val)` and returns `true`.

> [spec:et:def:tensor-util.executorch.get-dim-order-fn]
> ET_NODISCARD Error get_dim_order( const executorch::aten::Tensor& tensor, executorch::aten::DimOrderType* out_dim_order, size_t out_dim_order_size)

> [spec:et:sem:tensor-util.executorch.get-dim-order-fn]
> Writes `tensor`'s dim order into the caller-provided `out_dim_order` buffer.
> Declared here; defined out-of-line with two build-mode variants (portable/lean
> vs. ATen). Returns `ET_NODISCARD Error`.
>
> Portable (lean, non-ATen) behavior:
> 1. `ET_CHECK_OR_RETURN_ERROR` that `out_dim_order_size == tensor.dim_order().size()`;
>    on mismatch set `Error::InvalidArgument` (log both sizes) and return it.
> 2. `std::memcpy` `tensor.dim_order().size() * sizeof(DimOrderType)` bytes from
>    `tensor.dim_order().data()` into `out_dim_order` (the lean Tensor stores an
>    explicit dim_order array).
> 3. Return `Error::Ok`.
>
> ATen-mode behavior (equivalent contract, different derivation, since at::Tensor
> has no stored dim_order):
> 1. `ET_CHECK_OR_RETURN_ERROR` that `out_dim_order_size == tensor.dim()`; on
>    mismatch set `Error::InvalidArgument` and return it.
> 2. Derive the dim order from the tensor's strides via `stride_to_dim_order`
>    (see `[spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn]`),
>    writing `tensor.dim()` entries into `out_dim_order`, and return that call's
>    `Error`.

> [spec:et:def:tensor-util.executorch.internal.copy-tensor-data-fn]
> ET_NODISCARD Error copy_tensor_data( const executorch::aten::Tensor& t_dst, const executorch::aten::Tensor& t_src)

> [spec:et:sem:tensor-util.executorch.internal.copy-tensor-data-fn]
> Copies the raw bytes of `t_src`'s data buffer into `t_dst`'s preallocated data
> buffer. Internal API (namespace `internal`); returns `ET_NODISCARD Error`.
> Defined out-of-line with portable/ATen variants.
>
> Portable (lean) behavior:
> 1. `ET_CHECK_OR_RETURN_ERROR` that `t_dst.const_data_ptr() != nullptr || (t_dst.nbytes() == 0 && t_src.nbytes() == 0)`
>    — destination must be preallocated unless both tensors are empty; on failure
>    return `Error::InvalidArgument`.
> 2. If `t_src.const_data_ptr() != nullptr` (source may be null for a size-0
>    dimension):
>    a. `ET_CHECK_OR_RETURN_ERROR` that `t_dst.nbytes() == t_src.nbytes()`;
>       on mismatch return `Error::InvalidArgument`.
>    b. `std::memcpy(t_dst.mutable_data_ptr(), t_src.const_data_ptr(), t_src.nbytes())`.
> 3. If the source pointer is null, no copy occurs. Return `Error::Ok`.
>
> ATen-mode behavior is equivalent: it obtains the destination pointer from the
> tensor's storage impl (`data_ptr().get()`), requires it be non-null
> (`Error::InvalidArgument` otherwise), and performs the same nbytes check and
> `std::memcpy` when the source pointer is non-null.

> [spec:et:def:tensor-util.executorch.internal.reset-data-ptr-fn]
> void reset_data_ptr(const executorch::aten::Tensor& tensor)

> [spec:et:sem:tensor-util.executorch.internal.reset-data-ptr-fn]
> Clears `tensor`'s data pointer. Internal API; returns void. Defined out-of-line
> with portable/ATen variants.
>
> Portable (lean) behavior:
> - Calls `tensor.unsafeGetTensorImpl()->set_data(nullptr)`, setting the impl's
>   data pointer to null. Lean mode does not deallocate the buffer (the allocator
>   owns it); only the pointer is cleared.
>
> ATen-mode behavior:
> - Retrieves the impl, calls `set_sizes_contiguous(0)` (shrinking the tensor to
>   0 elements), then resets the underlying storage impl via
>   `unsafe_storage().unsafeGetStorageImpl()->reset()`, clearing all storage.

> [spec:et:def:tensor-util.executorch.internal.resize-tensor-impl-fn]
> ET_NODISCARD Error resize_tensor_impl( executorch::aten::TensorImpl* impl, executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes)

> [spec:et:sem:tensor-util.executorch.internal.resize-tensor-impl-fn]
> Resizes the tensor backing `impl` to `new_sizes`. Internal API; returns
> `ET_NODISCARD Error`. Defined out-of-line with portable/ATen variants. The rank
> must not change; the buffer is not grown beyond current capacity.
>
> Portable (lean) behavior:
> - Delegates (via a friend class) to `impl->internal_resize_contiguous(new_sizes)`
>   and returns its `Error`. That call enforces the runtime's resize constraints
>   (rank unchanged, capacity not exceeded) and computes contiguous strides for
>   the new sizes; on violation it returns a non-Ok `Error` (fails an internal
>   check for over-capacity resizes).
>
> ATen-mode behavior:
> 1. If `impl->dim() != new_sizes.size()` (rank change): log an error and return
>    `Error::NotSupported` (at::Tensor could resize but the runtime forbids rank
>    mutation).
> 2. Otherwise call `impl->set_sizes_contiguous(new_sizes)` (panics on failure)
>    and return `Error::Ok`.

> [spec:et:def:tensor-util.executorch.internal.set-tensor-data-fn]
> ET_NODISCARD Error set_tensor_data( const executorch::aten::Tensor& t, void* buffer, size_t buffer_size)

> [spec:et:sem:tensor-util.executorch.internal.set-tensor-data-fn]
> Points `t`'s data at the caller-supplied `buffer`. Internal API; returns
> `ET_NODISCARD Error`. Inputs: `t`, `buffer` (void*), `buffer_size` (size_t).
> Defined out-of-line with portable/ATen variants.
>
> Behavior (both variants equivalent):
> 1. `ET_CHECK_OR_RETURN_ERROR` that `buffer_size >= t.nbytes()`; on failure
>    return `Error::InvalidArgument` (log both sizes). The buffer must be at least
>    large enough for the tensor.
> 2. Set the tensor's data pointer to `buffer`:
>    - Portable: `t.unsafeGetTensorImpl()->set_data(buffer)`.
>    - ATen: `t.unsafeGetTensorImpl()->unsafe_storage().set_data_ptr(at::DataPtr(buffer, at::DeviceType::CPU))`.
> 3. Return `Error::Ok`. Ownership of `buffer` is not transferred; no copy occurs.

> [spec:et:def:tensor-util.executorch.internal.share-tensor-data-fn]
> ET_NODISCARD Error share_tensor_data( const executorch::aten::Tensor& t_dst, const executorch::aten::Tensor& t_src)

> [spec:et:sem:tensor-util.executorch.internal.share-tensor-data-fn]
> Makes `t_dst` share `t_src`'s data buffer (no copy). Internal API; returns
> `ET_NODISCARD Error`. Defined out-of-line with portable/ATen variants.
>
> Portable (lean) behavior:
> 1. `ET_CHECK_OR_RETURN_ERROR` that `t_dst.nbytes() == t_src.nbytes()`; on
>    mismatch return `Error::InvalidArgument`.
> 2. `ET_CHECK_OR_RETURN_ERROR` that `t_src.mutable_data_ptr() != nullptr || t_src.nbytes() == 0`
>    (source is either non-empty with data or empty); on failure return
>    `Error::InvalidArgument`.
> 3. Compute the pointer to share: `t_src_data_ptr = (t_src.numel() == 0) ? nullptr : t_src.mutable_data_ptr()`
>    (an empty source shares a null pointer explicitly).
> 4. Assign it via `t_dst.unsafeGetTensorImpl()->set_data(t_src_data_ptr)`, so
>    `t_dst` now aliases `t_src`'s buffer.
> 5. Return `Error::Ok`.
>
> ATen-mode behavior: performs the same `nbytes` equality check
> (`Error::InvalidArgument` on mismatch) and requires `t_src.mutable_data_ptr() != nullptr`
> (unconditionally, `Error::InvalidArgument` otherwise); then sets the
> destination storage's data pointer to a CPU `at::DataPtr` wrapping
> `t_src.mutable_data_ptr()` and updates the storage nbytes to `t_src.nbytes()`.
> Returns `Error::Ok`.

> [spec:et:def:tensor-util.executorch.resize-fn]
> ET_DEPRECATED inline void resize( executorch::aten::Tensor t, executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes)

> [spec:et:sem:tensor-util.executorch.resize-fn]
> DEPRECATED fatal-on-failure resize. Inline; returns void. Takes `t` and
> `new_sizes` (an `ArrayRef<SizesType>`).
>
> Steps:
> 1. Call `resize_tensor(t, new_sizes)` per
>    `[spec:et:sem:tensor-util.executorch.resize-tensor-fn]` (the same-type
>    overload, which forwards to
>    `[spec:et:sem:tensor-util.executorch.internal.resize-tensor-impl-fn]`),
>    capturing the returned `Error`.
> 2. `ET_CHECK_MSG` that the returned `err == Error::Ok`; if not, this is a fatal
>    abort (the deprecated API cannot fail non-fatally).
> 3. Return void. Prefer `resize_tensor`, which surfaces failures as a returnable
>    `Error`.

> [spec:et:def:tensor-util.executorch.resize-tensor-fn]
> ET_NODISCARD inline Error resize_tensor( executorch::aten::Tensor t, executorch::aten::ArrayRef<T> new_sizes)

> [spec:et:sem:tensor-util.executorch.resize-tensor-fn]
> Resizes tensor `t` to `new_sizes`, where `new_sizes` is an `ArrayRef<T>` whose
> element type `T` is NOT `SizesType` (this templated overload is SFINAE-selected
> for foreign integer types, e.g. `int64_t`; a separate non-templated overload
> handles `ArrayRef<SizesType>` by forwarding directly). Inline; returns
> `ET_NODISCARD Error`. Rank must stay the same; does not grow beyond current
> capacity.
>
> Steps:
> 1. Allocate `new_sizes_casted`, a stack `std::array<SizesType, kTensorDimensionLimit>`
>    (zero-initialized).
> 2. Let `new_sizes_ndim = new_sizes.size()`.
> 3. `ET_CHECK_OR_RETURN_ERROR` that `new_sizes_ndim <= kTensorDimensionLimit`;
>    on failure return `Error::InvalidArgument` (log both).
> 4. For `i` in `0..new_sizes_ndim`, set
>    `new_sizes_casted[i] = static_cast<SizesType>(new_sizes[i])` (narrowing cast
>    per element).
> 5. Return `internal::resize_tensor_impl(t.unsafeGetTensorImpl(), {new_sizes_casted.data(), new_sizes_ndim})`
>    per
>    `[spec:et:sem:tensor-util.executorch.internal.resize-tensor-impl-fn]`,
>    propagating its `Error`.
>
> The non-templated `ArrayRef<SizesType>` overload skips the cast/copy and calls
> `internal::resize_tensor_impl` directly with `new_sizes`.

> [spec:et:def:tensor-util.executorch.tensor-has-valid-dim-order-fn]
> bool tensor_has_valid_dim_order(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.tensor-has-valid-dim-order-fn]
> Returns whether `t`'s dim order is a valid permutation. Declared here; defined
> out-of-line with portable/ATen variants. Returns `false` (with logging) if the
> dim order is invalid or cannot be determined.
>
> Portable (lean) behavior:
> 1. Call `validate_dim_order(t.dim_order().data(), t.dim_order().size())` per
>    `[spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn]`.
> 2. If it returns false: log "Tensor dim order is not valid:" and, for each
>    `d` in `0..t.dim()`, log `dim_order(d)`; return `false`.
> 3. Otherwise return `true`.
>
> ATen-mode behavior: first derives the dim order into a local
> `DimOrderType[kTensorDimensionLimit]` via `get_dim_order(t, dim_order, t.dim())`
> (see `[spec:et:sem:tensor-util.executorch.get-dim-order-fn]`);
> `ET_CHECK_OR_RETURN_FALSE` that this returned `Error::Ok` (log and return
> `false` on failure). Then validates that derived array with
> `validate_dim_order` and returns `false` (logging each entry) if invalid, else
> `true`.

> [spec:et:def:tensor-util.executorch.tensor-is-channels-last-dim-order-fn]
> bool tensor_is_channels_last_dim_order(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.tensor-is-channels-last-dim-order-fn]
> Returns whether `t` has channels-last dim order (NHWC-style). Declared here;
> defined out-of-line in the portable/lean build only. Logs on failure.
>
> Behavior:
> 1. Compute `ret_val = is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size())`
>    per `[spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]`.
> 2. If `ret_val` is false: log "Expected tensor to have channels last dim order,
>    but got" and, for each `d` in `0..t.dim()`, log `dim_order(d)`.
> 3. Return `ret_val`.

> [spec:et:def:tensor-util.executorch.tensor-is-default-dim-order-fn]
> bool tensor_is_default_dim_order(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.tensor-is-default-dim-order-fn]
> Returns whether `t` has the default (contiguous / row-major) dim order.
> Declared here; defined out-of-line in the portable/lean build only. Logs on
> failure.
>
> Behavior:
> 1. Compute `ret_val = is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size())`
>    per `[spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]`.
> 2. If `ret_val` is false: log "Expected tensor to have default dim order, but
>    got" and, for each `d` in `0..t.dim()`, log `dim_order(d)`.
> 3. Return `ret_val`.

> [spec:et:def:tensor-util.executorch.tensor-is-default-or-channels-last-dim-order-fn]
> bool tensor_is_default_or_channels_last_dim_order(executorch::aten::Tensor t)

> [spec:et:sem:tensor-util.executorch.tensor-is-default-or-channels-last-dim-order-fn]
> Returns whether `t` has either the default (contiguous) OR the channels-last
> dim order. Declared here; defined out-of-line (portable/ATen). Logs on failure.
>
> Portable (lean) behavior:
> 1. Compute `ret_val = is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size()) || is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size())`
>    (see `[spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]`
>    and `[spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]`).
> 2. If `ret_val` is false: log "Expected tensor to have default or channels last
>    dim order, but got" and, for each `d` in `0..t.dim()`, log `dim_order(d)`.
> 3. Return `ret_val`.
>
> ATen-mode behavior first derives the dim order into a local
> `DimOrderType[kTensorDimensionLimit]` via `get_dim_order(t, dim_order, t.dim())`
> (`ET_CHECK_OR_RETURN_FALSE` on non-Ok, per
> `[spec:et:sem:tensor-util.executorch.get-dim-order-fn]`), then applies the same
> `is_contiguous_dim_order || is_channels_last_dim_order` disjunction to that
> derived array with identical logging.

> [spec:et:def:tensor-util.executorch.tensors-have-same-dim-order-fn]
> inline bool tensors_have_same_dim_order( const executorch::aten::Tensor& a, const executorch::aten::Tensor& b, const executorch::aten::Tensor& c, const executorch::aten::Tensor& d)

> [spec:et:sem:tensor-util.executorch.tensors-have-same-dim-order-fn]
> Four-tensor inline overload. Returns whether `a`, `b`, `c`, `d` all share a
> compatible dim order. Packs `{a, b, c, d}` into a local `Tensor[4]` and
> forwards to the `ArrayRef<Tensor>` overload `tensors_have_same_dim_order`
> (there are also 2- and 3-tensor inline overloads that forward `Tensor[2]` /
> `Tensor[3]` the same way). The ArrayRef overload is defined out-of-line
> (portable/ATen) and implements the actual check.
>
> ArrayRef overload behavior — the tensors are compatible iff they are ALL
> contiguous (default) dim order OR ALL channels-last dim order (a mix, or any
> other/unrecognized dim order, fails):
> 1. If `tensor_list.size() < 2`, return `true` (0 or 1 tensor trivially agrees).
> 2. Initialize `all_contiguous = true`, `all_channels_last = true`.
> 3. Portable: for each tensor `i` in the list, AND `all_contiguous` with
>    `is_contiguous_dim_order(tensor_list[i].dim_order().data(), ...size())` and
>    AND `all_channels_last` with `is_channels_last_dim_order(... )` (see
>    `[spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]`
>    and `[spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]`).
>    ATen: derives each tensor's dim order from strides via `get_dim_order`
>    (`ET_CHECK_OR_RETURN_FALSE` on non-Ok for each, per
>    `[spec:et:sem:tensor-util.executorch.get-dim-order-fn]`) into a local buffer
>    before applying the same two predicates; it seeds `all_contiguous` /
>    `all_channels_last` from tensor 0, then folds tensors `1..size`.
> 4. `ET_CHECK_OR_RETURN_FALSE` that `all_contiguous || all_channels_last`; on
>    failure log "%zd input tensors have different dim orders" and return `false`.
> 5. Return `true`.

