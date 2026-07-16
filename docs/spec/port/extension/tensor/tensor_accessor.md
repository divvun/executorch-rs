# extension/tensor/tensor_accessor.h

> [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base]
> class TensorAccessorBase {
>   T* data_;
>   const executorch::aten::SizesType* sizes_;
>   const executorch::aten::StridesType* strides_;
>   ssize_t dim_;
> }

> [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.size-fn]
> executorch::aten::SizesType size(ssize_t i) const

> [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.size-fn]
> Returns the size of dimension `i` of the accessed tensor.
>
> Steps:
> 1. Bounds-check `i` against the stored rank `dim_`: assert `i >= 0 && i < dim_`
>    (ET_CHECK_MSG). If the check fails the process aborts (fatal check, message
>    "Dimension outside of [0, dim_-1], got i"); this is a hard abort, not a
>    recoverable error.
> 2. Return `sizes_[i]` (an `executorch::aten::SizesType`), i.e. the i-th entry of
>    the size array the accessor was constructed with.

> [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.stride-fn]
> executorch::aten::StridesType stride(ssize_t i) const

> [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.stride-fn]
> Returns the stride of dimension `i` of the accessed tensor.
>
> Steps:
> 1. Bounds-check `i` against the stored rank `dim_`: assert `i >= 0 && i < dim_`
>    (ET_CHECK_MSG). On failure the process aborts (fatal check, message
>    "Dimension outside of [0, dim_-1], got i"); not a recoverable error.
> 2. Return `strides_[i]` (an `executorch::aten::StridesType`), i.e. the i-th entry
>    of the stride array the accessor was constructed with. Strides are expressed
>    in elements (not bytes).

> [spec:et:def:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]
> TensorAccessorBase( T* data, const executorch::aten::SizesType* sizes, const executorch::aten::StridesType* strides, ssize_t dim) : data_(data), sizes_(sizes), strides_(strides), dim_(dim)

> [spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]
> Protected constructor of the base accessor. Stores, by direct member init and
> without copying or validation:
> - `data_ = data` (raw pointer to the first element of the sub-view this accessor
>   addresses; `T*`, `const`-qualified when `T` is const),
> - `sizes_ = sizes` (pointer to the size array for this accessor's dimensions),
> - `strides_ = strides` (pointer to the stride array, in elements),
> - `dim_ = dim` (the rank this accessor exposes).
>
> The accessor is a non-owning view: it does not copy or take ownership of the
> pointed-to data, size, or stride arrays. The caller must ensure those outlive
> the accessor (in practice they live in the origin tensor's TensorImpl). No
> bounds or consistency checks are performed here. Being protected, it is only
> reachable via the derived `TensorAccessor` constructors and `make_tensor_accessor`.

> [spec:et:def:tensor-accessor.executorch.extension.make-tensor-accessor-fn]
> executorch::runtime::Result<TensorAccessor<T, N>> make_tensor_accessor( const executorch::aten::Tensor& tensor)

> [spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn]
> Builds a `TensorAccessor<T, N>` viewing the given `executorch::aten::Tensor`.
> `T` is the element C++ type and `N` the compile-time rank. Returns a
> `Result<TensorAccessor<T, N>>`: an error code on validation failure, else the
> constructed accessor.
>
> Steps:
> 1. Compile-time: `static_assert(N > 0, ...)` — rank 0 (scalar) is rejected at
>    compile time; use `*_data_ptr<T>()` for scalars.
> 2. Rank check: if `N != tensor.dim()`, log an Error ("Expecting N dimensions but
>    tensor has D.") and return `Error::InvalidArgument`.
> 3. Element-size check: if `sizeof(T) != tensor.element_size()`, log an Error and
>    return `Error::InvalidArgument`. Only the byte size is compared, not the
>    scalar-type identity, so any `T` whose size matches the tensor's element size
>    is accepted (e.g. reinterpreting an int32 tensor through a same-size float T).
> 4. Dim-order check (ExecuTorch/portable build only, i.e. when not USE_ATEN_LIB):
>    iterate `dim_order` and require it to be the trivial identity order
>    `dim_order[i] == i` for every `i`. If any entry differs, log an Error
>    ("Non-trival dim_order not supported.") and return `Error::NotSupported`.
>    Non-contiguous / permuted layouts are unsupported. In USE_ATEN_LIB builds this
>    check is skipped entirely.
> 5. Obtain the data pointer: if `T` is const-qualified, use
>    `tensor.const_data_ptr<T>()`; otherwise `tensor.mutable_data_ptr<T>()`.
> 6. Construct and return `TensorAccessor<T, N>(ptr, tensor.sizes().data(),
>    tensor.strides().data(), N)`. The accessor aliases the tensor's own sizes and
>    strides arrays and data buffer (no copy); the tensor's TensorImpl must outlive
>    the returned accessor.
>
> The three rule ids `make-tensor-accessor-fn`,
> `tensor-accessor.make-tensor-accessor-fn`, and
> `tensor-accessor-t-1.make-tensor-accessor-fn` all refer to this single free
> function; the latter two are `friend` declarations of it inside the
> `TensorAccessor<T,N>` and `TensorAccessor<T,1>` classes that grant it access to
> the private constructors.

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor]
> class TensorAccessor : public internal::TensorAccessorBase<T, N>

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1]
> class TensorAccessor<T, 1> : public internal::TensorAccessorBase<T, 1>

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.make-tensor-accessor-fn]
> executorch::runtime::Result<TensorAccessor<T2, N2>>

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.make-tensor-accessor-fn]
> Friend declaration inside the `TensorAccessor<T, 1>` specialization of the free
> function `make_tensor_accessor` (see
> `[spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn]`).
> It has no body of its own; it merely grants `make_tensor_accessor` access to the
> private rank-1 constructor so the factory can build a rank-1 accessor. All
> behavior is that of the free function.

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.tensor-accessor-fn]
> TensorAccessor( T* data, const executorch::aten::SizesType* sizes, const executorch::aten::StridesType* strides, ssize_t dim) : internal::TensorAccessorBase<T, 1>(data, sizes, strides, dim)

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.tensor-accessor-fn]
> Private constructor of the rank-1 accessor specialization `TensorAccessor<T, 1>`.
> Forwards its four arguments unchanged to the base constructor
> `internal::TensorAccessorBase<T, 1>(data, sizes, strides, dim)` (see
> `[spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]`),
> which stores them as `data_`, `sizes_`, `strides_`, `dim_`. `dim` is expected to
> be 1. Non-owning; performs no validation. Being private, it is only reachable
> from `make_tensor_accessor` (a friend) and from the rank-2 accessor's
> `operator[]` when it decays to rank 1.

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.make-tensor-accessor-fn]
> executorch::runtime::Result<TensorAccessor<T2, N2>>

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.make-tensor-accessor-fn]
> Friend declaration inside the general `TensorAccessor<T, N>` template of the free
> function `make_tensor_accessor` (see
> `[spec:et:sem:tensor-accessor.executorch.extension.make-tensor-accessor-fn]`).
> No body of its own; it only grants `make_tensor_accessor` access to the private
> rank-N constructor. All behavior is that of the free function.

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.operator-fn]
> TensorAccessor<T, N - 1> operator[](ssize_t i)

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.operator-fn]
> Indexes into the outermost dimension of a rank-N accessor (`N > 1`), producing a
> rank-(N-1) accessor addressing the i-th slice.
>
> Steps:
> 1. Compute the sub-view data pointer: `this->data_ + this->strides_[0] * i`.
>    `strides_[0]` is the outermost stride in elements, so the pointer advances
>    `strides_[0] * i` elements past the current base. No bounds check on `i` is
>    performed (out-of-range `i` yields an out-of-bounds pointer, undefined
>    behavior on deref).
> 2. Advance the metadata pointers by one dimension: `sizes_ + 1` and
>    `strides_ + 1`, dropping the outermost dimension.
> 3. Construct and return `TensorAccessor<T, N-1>(data_ + strides_[0]*i, sizes_+1,
>    strides_+1, N-1)` via that specialization's private constructor.
>
> There are two overloads: the non-const one returns `TensorAccessor<T, N-1>` and
> the const one returns `const TensorAccessor<T, N-1>`; both compute identical
> pointer arithmetic and both yield a non-owning sub-view aliasing the same buffer
> and metadata arrays. When `N-1 == 1` the result is the rank-1 specialization
> whose `operator[]` returns a scalar reference (see
> `[spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn]`).

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor.tensor-accessor-fn]
> TensorAccessor( T* data, const executorch::aten::SizesType* sizes, const executorch::aten::StridesType* strides, ssize_t dim) : internal::TensorAccessorBase<T, N>(data, sizes, strides, dim)

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor.tensor-accessor-fn]
> Private constructor of the general rank-N accessor `TensorAccessor<T, N>`.
> Forwards its four arguments unchanged to the base constructor
> `internal::TensorAccessorBase<T, N>(data, sizes, strides, dim)` (see
> `[spec:et:sem:tensor-accessor.executorch.extension.internal.tensor-accessor-base.tensor-accessor-base-fn]`),
> which stores them as `data_`, `sizes_`, `strides_`, `dim_`. Non-owning; performs
> no validation. Being private, it is reachable only from `make_tensor_accessor`
> (a friend) and from a rank-(N+1) accessor's `operator[]`.

> [spec:et:def:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn]
> T& operator[](ssize_t i)

> [spec:et:sem:tensor-accessor.executorch.extension.tensor-accessor-t-1.operator-fn]
> Indexes into the single dimension of a rank-1 accessor, returning a reference to
> the addressed scalar (not a sub-accessor).
>
> Steps:
> 1. Compute the element offset `this->strides_[0] * i` (in elements), where
>    `strides_[0]` is the stride of the one remaining dimension.
> 2. Return `this->data_[strides_[0] * i]`, i.e. a reference to that element.
>
> Two overloads: non-const returns `T&` (writable), const returns `const T&`. No
> bounds check on `i` is performed; an out-of-range index gives out-of-bounds
> access (undefined behavior). Because it returns a reference into the aliased
> buffer, writes through the non-const overload mutate the origin tensor's data
> in place.

