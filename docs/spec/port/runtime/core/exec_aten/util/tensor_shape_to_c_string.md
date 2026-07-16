# runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp

> [spec:et:def:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
> std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string( executorch::runtime::Span<const std::int32_t> shape)

> [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
> Public entry point. There are two overloads, one taking
> `Span<const int32_t>` and one taking `Span<const int64_t>` (to support
> both ExecuTorch and ATen tensor sizes types without a circular header
> dependency). Each simply forwards its `shape` span to the templated
> `tensor_shape_to_c_string_impl` (per
> `[spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn]`)
> and returns its result: a fixed-size `std::array<char,
> kTensorShapeStringSizeLimit>` holding a NUL-terminated, human-readable
> rendering of the shape. No validation or transformation is done at this
> layer beyond overload/type selection.

> [spec:et:def:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn]
> std::array<char, kTensorShapeStringSizeLimit> tensor_shape_to_c_string_impl( executorch::runtime::Span<SizesType> shape)

> [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn]
> Templated (over `SizesType`, either int32_t or int64_t) core routine
> that renders a shape span into a NUL-terminated string of the form
> `"(d0, d1, ..., dN)"`. Returns a `std::array<char,
> kTensorShapeStringSizeLimit>` by value. The array is sized so that up
> to `kTensorDimensionLimit` (16) elements, each printed as up to 10
> decimal digits plus ", " separators plus the parentheses and NUL, fit
> without overflow.
> Algorithm, writing through a cursor `p` starting at `out.data()`:
> - Dimension-limit guard: if `shape.size() > kTensorDimensionLimit`
>   (i.e. more than 16 dims), copy the literal string
>   "(ERR: tensor ndim exceeds limit)" into `out` and return immediately.
>   No per-element formatting is done.
> - Write an opening '(' at `p`, then advance `p`.
> - For each element `elem` of `shape`, in order:
>   - If `elem < 0` OR `(size_t)elem > kMaximumPrintableTensorShapeElement`
>     (where `kMaximumPrintableTensorShapeElement == INT32_MAX ==
>     2147483647`): write the literal "ERR, " at `p` and advance `p` by 5.
>     Negative dims and dims exceeding INT32_MAX are rendered as ERR.
>   - Otherwise: format the element with `snprintf(p, remaining, "%u, ",
>     (uint32_t)elem)` — i.e. the unsigned decimal value followed by a
>     comma and a space — and advance `p` by snprintf's return value (the
>     number of characters written, excluding the NUL). `remaining` is the
>     bytes left in the buffer, `kTensorShapeStringSizeLimit - (p -
>     out.data())`.
> - After the loop, overwrite the trailing ", " of the last element:
>   set `*(p-2) = ')'` and `*(p-1) = '\0'`. This converts the final
>   separator into the closing paren + NUL, yielding e.g. "(2, 3, 4)".
> - Return `out`.
> Edge cases:
> - Empty shape (size 0): the loop writes nothing; `p` still points just
>   past the '(' (at out.data()+1). The final `*(p-2)`/`*(p-1)` writes
>   then target out.data()-1 (one byte BEFORE the buffer) and out.data().
>   The C++ source performs these writes unconditionally, so an empty
>   shape is a degenerate/unsupported input that writes out of bounds;
>   the string content of `out[0]` is left as the NUL from the `*(p-1)`
>   write. A Rust reimplementation should treat an empty shape explicitly
>   (e.g. produce "()") rather than reproduce this out-of-bounds write.
> - The output is always NUL-terminated (for the normal and limit-exceeded
>   paths).

