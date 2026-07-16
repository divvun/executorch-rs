# runtime/core/portable_type/tensor.h

> [spec:et:def:tensor.executorch.runtime.etensor.tensor]
> class Tensor {
>   TensorImpl* impl_ = nullptr;
> }

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn]
> inline const T* const_data_ptr() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.const-data-ptr-fn]
> Templated accessor `const_data_ptr<T>()`. Returns a read-only pointer of
> element type `T` to the tensor's underlying data blob by delegating to the
> owned `TensorImpl`'s templated `data<T>()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.data-fn]`),
> which returns the raw data pointer reinterpreted (static_cast) as `const T*`.
> There is also a non-templated overload `const_data_ptr()` returning `const
> void*` that returns the raw data pointer unchanged (may be null if no data
> has been set). No bounds/type checking is performed: the caller is
> responsible for ensuring `T` matches the tensor's `scalar_type()`; a
> mismatch reinterprets the bytes. The returned pointer aliases the impl's
> data and is only valid while the impl and its data blob remain alive.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.data-ptr-fn]
> ET_DEPRECATED inline T* data_ptr() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.data-ptr-fn]
> DEPRECATED (marked `ET_DEPRECATED`). Templated accessor `data_ptr<T>()`.
> Returns a mutable pointer of element type `T` to the tensor's underlying
> data blob by delegating to the owned `TensorImpl`'s templated
> `mutable_data<T>()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.mutable-data-fn]`),
> which returns the raw data pointer reinterpreted (static_cast) as `T*`.
> There is also a non-templated `ET_DEPRECATED` overload `data_ptr()`
> returning `void*` that returns the raw data pointer unchanged (may be null).
> Behaves identically to `mutable_data_ptr`
> (`[spec:et:sem:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn]`);
> it is retained only for source compatibility. No bounds/type checking; the
> caller must ensure `T` matches `scalar_type()`. Prefer `const_data_ptr` or
> `mutable_data_ptr` instead. In a Rust port this may be omitted or aliased to
> the mutable accessor.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-fn]
> Device device() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-fn]
> Returns the `Device` where the tensor's data resides by delegating to the
> owned `TensorImpl`'s `device()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-fn]`),
> which returns the impl's stored `Device` value by copy. A `Device`
> bundles a `DeviceType` (e.g. CPU) and a `DeviceIndex`. No computation or
> validation is performed. Const method; does not mutate the tensor.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-index-fn]
> DeviceIndex device_index() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-index-fn]
> Returns the `DeviceIndex` of the tensor's device by delegating to the owned
> `TensorImpl`'s `device_index()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-index-fn]`),
> which returns `device_.index()` — the index field of the impl's stored
> `Device`. Value is 0 when the device is default/unspecified. No computation
> or validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.device-type-fn]
> DeviceType device_type() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.device-type-fn]
> Returns the `DeviceType` (e.g. CPU) of the tensor's device by delegating to
> the owned `TensorImpl`'s `device_type()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-type-fn]`),
> which returns `device_.type()` — the type field of the impl's stored
> `Device`. No computation or validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.dim-fn]
> ssize_t dim() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-fn]
> Returns the tensor's number of dimensions (rank) as an `ssize_t` by
> delegating to the owned `TensorImpl`'s `dim()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-fn]`),
> which returns the impl's stored `dim_` field verbatim. For a scalar
> (zero-dimensional) tensor this is 0. No computation or validation. Const
> method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.dim-order-fn]
> const ArrayRef<DimOrderType> dim_order() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dim-order-fn]
> Returns the tensor's dimension order — the order in which dimensions are
> laid out in memory — as a read-only `ArrayRef<DimOrderType>` (each element a
> `uint8_t`) by delegating to the owned `TensorImpl`'s `dim_order()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-order-fn]`),
> which constructs the ArrayRef as `{dim_order_, dim_}` — i.e. it views `dim()`
> elements starting at the impl's stored `dim_order_` pointer without copying.
> The returned view aliases the impl's dim_order array and is valid only while
> the impl (and that array) remain alive; its length equals `dim()`. Const
> method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.dtype-fn]
> inline ScalarType dtype() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.dtype-fn]
> Returns the tensor's element `ScalarType`. Implemented purely as a call to
> `scalar_type()`
> (`[spec:et:sem:tensor.executorch.runtime.etensor.tensor.scalar-type-fn]`);
> `dtype()` and `scalar_type()` are exact synonyms and always return the same
> value. Provided for source compatibility with `at::Tensor`. No computation
> or validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.element-size-fn]
> ssize_t element_size() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.element-size-fn]
> Returns the size in bytes of a single element of the tensor as an `ssize_t`
> by delegating to the owned `TensorImpl`'s `element_size()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.element-size-fn]`),
> which returns the byte width of the tensor's `scalar_type()` (e.g. 4 for
> Float/Int, 1 for Byte/Char/Bool, 2 for Half/BFloat16/Short, 8 for
> Long/Double). No validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn]
> inline T* mutable_data_ptr() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.mutable-data-ptr-fn]
> Templated accessor `mutable_data_ptr<T>()`. Returns a mutable pointer of
> element type `T` to the tensor's underlying data blob by delegating to the
> owned `TensorImpl`'s templated `mutable_data<T>()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.mutable-data-fn]`),
> which returns the raw data pointer reinterpreted (static_cast) as `T*`.
> There is also a non-templated overload `mutable_data_ptr()` returning
> `void*` that returns the raw data pointer unchanged (may be null if no data
> has been set). No bounds/type checking: the caller must ensure `T` matches
> the tensor's `scalar_type()`. The returned pointer aliases the impl's data
> and is valid only while the impl and its data blob remain alive. Note the
> method is declared `const` on the Tensor but yields a writable pointer,
> since the Tensor does not own the data.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.nbytes-fn]
> size_t nbytes() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.nbytes-fn]
> Returns the size in bytes of the tensor's live data as a `size_t` by
> delegating to the owned `TensorImpl`'s `nbytes()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.nbytes-fn]`),
> which returns `numel() * element_size()` for the tensor's current shape.
> This is the alive/used size for the current shape, NOT the capacity of the
> underlying buffer (which may be larger for dynamically-shaped tensors). For
> an empty tensor (numel 0) this is 0. No validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.numel-fn]
> ssize_t numel() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.numel-fn]
> Returns the total number of elements in the tensor as an `ssize_t` by
> delegating to the owned `TensorImpl`'s `numel()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.numel-fn]`),
> which returns the impl's precomputed/cached `numel_` field (the product of
> all dimension sizes, or 1 for a zero-dim scalar tensor, or 0 if any size is
> 0). It is not recomputed on each call. No validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.scalar-type-fn]
> ScalarType scalar_type() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.scalar-type-fn]
> Returns the `ScalarType` of the tensor's elements (e.g. Float, Int, Bool) by
> delegating to the owned `TensorImpl`'s `scalar_type()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.scalar-type-fn]`),
> which returns the impl's stored `type_` field verbatim. This is the
> canonical dtype accessor; `dtype()` is a synonym
> (`[spec:et:sem:tensor.executorch.runtime.etensor.tensor.dtype-fn]`). No
> computation or validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.set-data-fn]
> ET_DEPRECATED void set_data(void* ptr) const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.set-data-fn]
> DEPRECATED (marked `ET_DEPRECATED`). Changes which data buffer the tensor
> aliases by delegating to the owned `TensorImpl`'s `set_data(ptr)`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.set-data-fn]`),
> which simply overwrites the impl's `data_` pointer with `ptr`. It does NOT
> free the previously pointed-to memory and does NOT take ownership of the new
> pointer (no allocation, no copy, no size validation — the caller guarantees
> the new buffer is large enough for the current shape and lives long enough).
> `ptr` may be null. Declared `const` on the Tensor because the Tensor does
> not own the impl. Returns void. This API has no `at::Tensor` equivalent and
> kernel developers should avoid it; a Rust port may omit it or gate it behind
> an unsafe/deprecated interface.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.shape-dynamism-fn]
> TensorShapeDynamism shape_dynamism() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.shape-dynamism-fn]
> Returns the tensor's `TensorShapeDynamism` — the mutability of its shape
> (e.g. STATIC, DYNAMIC_BOUND, DYNAMIC_UNBOUND) — by delegating to the owned
> `TensorImpl`'s `shape_dynamism()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.shape-dynamism-fn]`),
> which returns the impl's stored `shape_dynamism_` field verbatim. No
> computation or validation. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.size-fn]
> ssize_t size(ssize_t dim) const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.size-fn]
> Returns the size of the tensor along dimension `dim` as an `ssize_t` by
> delegating to the owned `TensorImpl`'s `size(dim)`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.size-fn]`).
> The impl first bounds-checks with `ET_CHECK_MSG(dim < dim_ && dim >= 0,
> ...)`: `dim` must satisfy `0 <= dim < dim()`; an out-of-range `dim` aborts
> (fatal check, not a recoverable Error) with a message naming the valid range
> `[0, dim()-1]`. Negative indices are NOT supported (unlike PyTorch). On a
> valid index it returns `sizes_[dim]`, the size of that dimension. Return
> type is intentionally `ssize_t` (not `SizesType`) for `at::Tensor`
> compatibility. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.sizes-fn]
> const ArrayRef<SizesType> sizes() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.sizes-fn]
> Returns the tensor's shape as a read-only `ArrayRef<SizesType>` (each element
> an `int32_t`) by delegating to the owned `TensorImpl`'s `sizes()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.sizes-fn]`),
> which constructs the ArrayRef as `{sizes_, dim_}` — a non-owning view of
> `dim()` elements starting at the impl's stored `sizes_` pointer, no copy.
> The view aliases the impl's sizes array and is valid only while the impl
> (and that array) remain alive; its length equals `dim()` (empty for a
> zero-dim tensor). Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.strides-fn]
> const ArrayRef<StridesType> strides() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.strides-fn]
> Returns the tensor's per-dimension strides (in elements) as a read-only
> `ArrayRef<StridesType>` (each element an `int32_t`) by delegating to the
> owned `TensorImpl`'s `strides()`
> (`[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.strides-fn]`),
> which constructs the ArrayRef as `{strides_, dim_}` — a non-owning view of
> `dim()` elements starting at the impl's stored `strides_` pointer, no copy.
> The view aliases the impl's strides array and is valid only while the impl
> (and that array) remain alive; its length equals `dim()`. Const method.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.tensor-fn]
> Tensor() = delete

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.tensor-fn]
> Construction of a `Tensor`. The default constructor `Tensor()` is explicitly
> deleted: a Tensor cannot be created without an impl. The usable constructor
> is `explicit constexpr Tensor(TensorImpl* impl)`, which stores the given
> `TensorImpl*` in the private `impl_` member and performs no other work — no
> allocation, no copy, no validation, no null check. The Tensor is a thin
> non-owning handle: it does NOT own the pointed-to `TensorImpl`, so the
> caller must guarantee the impl (and everything it references — sizes,
> strides, dim_order, data) outlives every Tensor pointing to it. In a Rust
> port this maps to a wrapper holding a borrowed/reference to a TensorImpl with
> an explicit lifetime; there is no default/empty construction.

> [spec:et:def:tensor.executorch.runtime.etensor.tensor.unsafe-get-tensor-impl-fn]
> TensorImpl* unsafeGetTensorImpl() const

> [spec:et:sem:tensor.executorch.runtime.etensor.tensor.unsafe-get-tensor-impl-fn]
> Returns the raw `TensorImpl*` stored in the Tensor's private `impl_` member,
> verbatim, with no copy, no null check, and no validation. This exposes the
> underlying impl so callers can operate on it directly; it is deliberately
> "unsafe" because bypassing the Tensor API can easily break invariants.
> Ownership is unchanged — the returned pointer still aliases the impl the
> Tensor does not own, valid only while that impl is alive. Const method (the
> Tensor is not mutated) even though the returned pointer is non-const. In a
> Rust port this maps to exposing the borrowed impl reference and should be
> considered an escape hatch.

