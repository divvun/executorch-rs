# runtime/core/tensor_layout.cpp, runtime/core/tensor_layout.h

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn]
> Result<size_t> calculate_nbytes( const Span<const int32_t>& sizes, const executorch::aten::ScalarType& scalar_type)

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn]
> File-local (anonymous-namespace) helper. Computes the total byte size of a
> contiguous tensor with the given `sizes` and `scalar_type`, returning
> `Result<size_t>`.
>
> Steps:
> - Initialize an accumulator `n = 1` (type `size_t`).
> - Iterate `i` from 0 to `sizes.size() - 1` in order:
>   - If `sizes[i] < 0`, return `Error::InvalidArgument`.
>   - Compute `next = n * static_cast<size_t>(sizes[i])` using a checked
>     multiply (`c10::mul_overflows`); if the multiplication overflows `size_t`,
>     return `Error::InvalidArgument`. Otherwise set `n = next`.
>   - Note: a zero size produces `n = 0`, and an empty `sizes` span leaves
>     `n = 1` (scalar/0-dim tensor has one element).
> - Compute the per-element byte size `elem_size` by calling
>   `executorch::runtime::elementSize(scalar_type)` (see
>   `[spec:et:sem:scalar-type-util.executorch.runtime.element-size-fn]`) and
>   casting to `size_t`.
> - Compute `total = n * elem_size` using the checked multiply; if it overflows
>   `size_t`, return `Error::InvalidArgument`.
> - Return `total` as a successful `Result<size_t>`.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout]
> class TensorLayout final {
>   const Span<const int32_t> sizes_;
>   const Span<const uint8_t> dim_order_;
>   const executorch::aten::ScalarType scalar_type_;
>   const size_t nbytes_;
> }

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn]
> Result<const TensorLayout> TensorLayout::create( Span<const int32_t> sizes, Span<const uint8_t> dim_order, executorch::aten::ScalarType scalar_type)

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn]
> Static factory validating the parameters and building a `TensorLayout`,
> returning `Result<const TensorLayout>`.
>
> Steps, in order:
> - Compute `nbytes` by calling `calculate_nbytes(sizes, scalar_type)` (see
>   `[spec:et:sem:tensor-layout.executorch.et-runtime-namespace.calculate-nbytes-fn]`).
>   If it is not ok (any negative size or multiply overflow), return its error
>   (`Error::InvalidArgument`) immediately.
> - If `dim_order.size() != sizes.size()`, return `Error::InvalidArgument`.
> - Iterate `i` over `dim_order`: if any `dim_order[i] >= sizes.size()`, return
>   `Error::InvalidArgument`. (Each dim-order entry must be a valid dimension
>   index; note this does not check that dim_order is a permutation without
>   duplicates — only that each entry is in range `[0, sizes.size())`.)
> - On success, construct and return `TensorLayout(sizes, dim_order,
>   scalar_type, nbytes.get())` (see
>   `[spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn]`).
>   The `sizes` and `dim_order` spans are stored by reference-view; the caller
>   must ensure the underlying buffers outlive the TensorLayout and all copies.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.dim-order-fn]
> Span<const uint8_t> dim_order() const

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.dim-order-fn]
> Returns the stored `dim_order_` member: a `Span<const uint8_t>` viewing the
> tensor's dim order. No copy of the underlying buffer is made; the returned
> span aliases the buffer originally passed to `create`, which must outlive the
> TensorLayout. No side effects.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.nbytes-fn]
> size_t nbytes() const

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.nbytes-fn]
> Returns the stored `nbytes_` member: the tensor's total size in bytes as a
> `size_t`, precomputed at construction by `calculate_nbytes` (product of all
> sizes times the scalar type's element size). No side effects.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.scalar-type-fn]
> executorch::aten::ScalarType scalar_type() const

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.scalar-type-fn]
> Returns the stored `scalar_type_` member (an `executorch::aten::ScalarType`)
> by value. No side effects.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.sizes-fn]
> Span<const int32_t> sizes() const

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.sizes-fn]
> Returns the stored `sizes_` member: a `Span<const int32_t>` viewing the
> tensor's sizes. No copy of the underlying buffer is made; the returned span
> aliases the buffer originally passed to `create`, which must outlive the
> TensorLayout. No side effects.

> [spec:et:def:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn]
> TensorLayout( Span<const int32_t> sizes, Span<const uint8_t> dim_order, executorch::aten::ScalarType scalar_type, size_t nbytes) : sizes_(sizes), dim_order_(dim_order), scalar_type_(scalar_type), nbytes_(nbytes)

> [spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.tensor-layout-fn]
> Private constructor (the only constructor; the default constructor is deleted).
> Initializes the four const members directly from the arguments with no
> validation or copying of buffers: `sizes_ = sizes`, `dim_order_ = dim_order`,
> `scalar_type_ = scalar_type`, `nbytes_ = nbytes`. The `sizes` and `dim_order`
> spans are stored as views (aliasing the caller's buffers, which must outlive
> the object). Only reachable via `create` (see
> `[spec:et:sem:tensor-layout.executorch.et-runtime-namespace.tensor-layout.create-fn]`),
> which performs all validation and supplies the precomputed `nbytes`.

