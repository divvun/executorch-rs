# runtime/core/portable_type/tensor_impl.cpp, runtime/core/portable_type/tensor_impl.h

> [spec:et:def:tensor-impl.executorch.runtime.etensor.compute-numel-fn]
> ssize_t compute_numel(const TensorImpl::SizesType* sizes, ssize_t dim)

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.compute-numel-fn]
> Computes the number of elements (product of sizes) for a tensor with `dim`
> dimensions whose per-dimension sizes are the first `dim` entries of the
> `sizes` array.
>
> Steps:
> 1. Precondition check (fatal): assert that `dim == 0 || sizes != nullptr`. If
>    `dim != 0` and `sizes` is null, this is a fatal `ET_CHECK_MSG` failure
>    ("Sizes must be provided for non-scalar tensors") that aborts; it does not
>    return an error. (A null `sizes` is permitted only for scalar/0-dim
>    tensors.)
> 2. Initialize an `ssize_t` accumulator `numel = 1`. A zero-dimensional tensor
>    (scalar, `dim == 0`) therefore has `numel == 1`; the loop below does not
>    execute.
> 3. For each `i` in `[0, dim)` in increasing order:
>    - Fatal check (`ET_CHECK_MSG`): assert `sizes[i] >= 0`; a negative size
>      aborts with "Size must be non-negative, got <size> at dimension <i>".
>    - Multiply: `numel *= sizes[i]` (accumulation in `ssize_t`). No overflow
>      check is performed here (unlike `[spec:et:sem:tensor-impl.executorch.runtime.etensor.safe-numel-fn]`);
>      overflow wraps per `ssize_t` multiplication.
> 4. Return `numel`.
>
> `sizes[i]` has element type `SizesType` (`int32_t`); it is used directly in
> the `ssize_t` product. This function never returns an error — invalid inputs
> abort the process.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.safe-numel-fn]
> ::executorch::runtime::Result<ssize_t> safe_numel( const TensorImpl::SizesType* sizes, ssize_t dim)

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.safe-numel-fn]
> Overflow-checked variant of
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.compute-numel-fn]`.
> Computes the product of the first `dim` entries of `sizes` and returns it as a
> `Result<ssize_t>`, propagating an `Error` instead of aborting on invalid
> input or overflow.
>
> Steps:
> 1. Argument check (`ET_CHECK_OR_RETURN_ERROR`): if NOT (`dim == 0 || sizes !=
>    nullptr`), return `Error::InvalidArgument` ("Sizes must be provided for
>    non-scalar tensors"). A null `sizes` is allowed only for `dim == 0`.
> 2. Initialize `ssize_t numel = 1`. For `dim == 0` the loop is skipped and the
>    function returns `1`.
> 3. For each `i` in `[0, dim)` in increasing order:
>    - Non-negative check (`ET_CHECK_OR_RETURN_ERROR`): if `sizes[i] < 0`,
>      return `Error::InvalidArgument` ("Size must be non-negative, got <size>
>      at dimension <i>").
>    - Overflow-checked multiply: compute `next_numel = numel * sizes[i]` using
>      a checked multiplication (`c10::mul_overflows`) over `ssize_t`. If the
>      multiplication would overflow `ssize_t`, return `Error::InvalidArgument`
>      ("Overflow computing numel at dimension <i>"). Otherwise set `numel =
>      next_numel`.
> 4. Return `Result<ssize_t>` holding `numel` on success.
>
> Preferred over `compute_numel` on any path that can propagate an `Error`
> upward (e.g. dynamic resize in
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`).

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl]
> class TensorImpl {
>   SizesType* sizes_;
>   DimOrderType* dim_order_;
>   StridesType* strides_;
>   void* data_;
>   const ssize_t dim_;
>   ssize_t numel_;
>   size_t numel_bound_;
>   const ScalarType type_;
>   const TensorShapeDynamism shape_dynamism_;
>   Device device_;
> }

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.data-fn]
> inline const T* data() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.data-fn]
> Templated const accessor for the underlying data blob, reinterpreting it as
> the caller-supplied element type `T`.
>
> Returns the raw `data_` pointer (via the non-templated `data()` which returns
> `const void*`) cast to `const T*` (a `static_cast` from `const void*`). No
> dtype validation is performed: `T` is trusted to match the tensor's element
> layout; the caller is responsible for consistency with `scalar_type()`.
> Returns null if `data_` is null (data pointers may be null). Does not
> dereference or bounds-check.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.device-fn]
> Device device() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-fn]
> Returns the tensor's `Device` by value: returns the `device_` member
> unchanged. `device_` bundles the device type and device index and was set at
> construction from the `device_type`/`device_index` constructor arguments.
> Pure accessor; no side effects.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.device-index-fn]
> DeviceIndex device_index() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-index-fn]
> Returns the device index of the tensor's storage: returns `device_.index()`,
> i.e. the `DeviceIndex` component of the stored `Device`. This is the
> `device_index` passed to the constructor (default `0` if
> default/unspecified). Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.device-type-fn]
> DeviceType device_type() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.device-type-fn]
> Returns the `DeviceType` of the tensor's storage: returns `device_.type()`,
> i.e. the type component of the stored `Device` (e.g. `DeviceType::CPU`). This
> is the `device_type` passed to the constructor (default `DeviceType::CPU`).
> Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-fn]
> ssize_t dim() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-fn]
> Returns the tensor's number of dimensions (rank): returns the `dim_` member
> as an `ssize_t`. `dim_` is `const`, set once at construction and never
> changed (rank is immutable, even across resizes). A scalar has `dim() == 0`.
> Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-order-fn]
> const ArrayRef<DimOrderType> dim_order() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dim-order-fn]
> Returns the dimension order (the order in which dimensions are laid out in
> memory) as a read-only view.
>
> Constructs and returns an `ArrayRef<DimOrderType>` over the `dim_order_`
> pointer with length `dim_` (cast to `size_t`). Element type `DimOrderType` is
> `uint8_t`. The returned view aliases the tensor's `dim_order_` storage (no
> copy); the tensor does not own that storage and the caller must not outlive
> it. If `dim_order_` is null and `dim_ == 0`, this is an empty ArrayRef. Pure
> accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.dtype-fn]
> inline ScalarType dtype() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dtype-fn]
> Alias for `scalar_type()`: returns the tensor's element `ScalarType` by
> delegating to
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.scalar-type-fn]`
> (i.e. returns the `type_` member). Provided for at::Tensor API compatibility.
> Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.element-size-fn]
> ssize_t TensorImpl::element_size() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.element-size-fn]
> Returns the size in bytes of a single element of the tensor's dtype: returns
> `elementSize(type_)`, i.e. the byte width of the `ScalarType` stored in
> `type_` (e.g. 1 for Bool/Byte/Char, 2 for Half/BFloat16/Short, 4 for
> Int/Float, 8 for Long/Double, and the appropriate size for complex/qint
> types) as an `ssize_t`. Independent of tensor shape/numel. Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]
> Error TensorImpl::internal_resize_contiguous(ArrayRef<SizesType> new_sizes)

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]
> Resizes the tensor in place to `new_sizes`, updating `sizes_`, `strides_`, and
> `numel_` while keeping rank and data pointer fixed. Contiguous-strides
> semantics (same as `at::TensorImpl::set_sizes_contiguous`) but returns an
> `Error` instead of aborting. `ET_NODISCARD`. Only callable via the
> `TensorResizerFriend` friend (i.e. `resize_tensor`/`resize_tensor_impl`).
>
> Steps:
> 1. Rank check (`ET_CHECK_OR_RETURN_ERROR`): if `new_sizes.size() != dim_`,
>    return `Error::NotSupported` ("Attempted to change the tensor rank which is
>    immutable: old=<dim_>, new=<new_sizes.size()>"). Rank is immutable.
> 2. Zero-rank fast path: if `dim_ == 0`, return `Error::Ok` immediately. A
>    0-dim out tensor already has the correct (empty) shape; nothing to change.
> 3. Dispatch on `shape_dynamism_`:
>    - `STATIC`: compare the existing `dim_` sizes (`sizes_[0..dim_)`) element-
>      wise with `new_sizes`. If they are equal, do nothing (break). If they
>      differ, log an Error-level message ("Attempted to resize a static tensor.
>      Expected shape <old>, but received <new>." — only when logging is
>      enabled) and return `Error::NotSupported`. A static tensor may not change
>      shape.
>    - `DYNAMIC_BOUND` and `DYNAMIC_UNBOUND` (shared handling; unbounded is
>      currently treated as upper-bounded by `numel_bound_`):
>      a. Compute `new_numel = safe_numel(new_sizes.data(), dim_)` per
>         `[spec:et:sem:tensor-impl.executorch.runtime.etensor.safe-numel-fn]`.
>         If it returns an error (negative size or overflow), propagate that
>         `Error` unchanged.
>      b. Capacity check (`ET_CHECK_OR_RETURN_ERROR`): if `new_numel >
>         numel_bound_` (compared as `size_t`), return `Error::NotSupported`
>         ("Attempted to resize a bounded tensor with a maximum capacity of
>         <numel_bound_> elements to <new_numel> elements."). `numel_bound_` is
>         the original numel captured at construction and is the hard upper
>         bound on element count; the data buffer is never grown.
>      c. Recompute strides: if both `strides_` and `dim_order_` are non-null,
>         call `dim_order_to_stride(new_sizes.data(), dim_order_, dim_,
>         strides_)` to overwrite `strides_` with contiguous strides consistent
>         with `dim_order_` and `new_sizes` (see the dim-order utility). If it
>         returns a non-Ok `Error`, propagate it. If either pointer is null,
>         strides are left unchanged.
>      d. Commit: set `numel_ = new_numel` and copy `new_sizes` into `sizes_`
>         (overwriting the first `dim_` entries).
> 4. Return `Error::Ok`.
>
> Note: only `numel_` and the sizes/strides arrays are mutated; `data_`,
> `dim_`, `type_`, `numel_bound_`, and `shape_dynamism_` are unchanged, and the
> underlying data buffer is neither reallocated nor moved.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.mutable-data-fn]
> inline T* mutable_data() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.mutable-data-fn]
> Templated mutable accessor for the underlying data blob, reinterpreting it as
> the caller-supplied element type `T`.
>
> Returns the raw `data_` pointer (via the non-templated `mutable_data()` which
> returns `void*`) cast to `T*` (a `static_cast` from `void*`). Like
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.data-fn]` but
> yields a non-const pointer for writing. No dtype validation; `T` is trusted to
> match the element layout. Returns null if `data_` is null. Note the method is
> `const` on the TensorImpl but returns a mutable element pointer — it does not
> mutate the TensorImpl itself.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.nbytes-fn]
> size_t TensorImpl::nbytes() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.nbytes-fn]
> Returns the number of bytes occupied by the tensor's data for its *current*
> shape (not the buffer capacity `numel_bound_`).
>
> Computes `numel_ * elementSize(type_)`: the current element count times the
> byte width of the dtype. Returns a `size_t`. For a scalar (`numel_ == 1`)
> this is one element's size; for an empty tensor (`numel_ == 0`) this is `0`.
> No overflow check.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.numel-fn]
> ssize_t numel() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.numel-fn]
> Returns the current number of elements in the tensor: returns the cached
> `numel_` member as an `ssize_t`. `numel_` is set at construction from
> `compute_numel(sizes, dim)` and updated by
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`
> on dynamic resize. A scalar has `numel() == 1`. Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.scalar-type-fn]
> ScalarType scalar_type() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.scalar-type-fn]
> Returns the tensor's element type: returns the `type_` member (a
> `ScalarType`). `type_` is `const`, set once at construction and validated
> there (see
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.tensor-impl-fn]`).
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.dtype-fn]` is
> an alias of this method. Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.set-data-fn]
> void set_data(void* ptr)

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.set-data-fn]
> Repoints the tensor at a new data blob: sets `data_ = ptr`, overwriting the
> previous data pointer. Accepts any `void*` (including null). Does not free or
> take ownership of either the old or new buffer — the tensor never owns its
> data storage; the caller manages lifetime. Does not touch sizes, strides,
> numel, or dtype; the new buffer is assumed to match the existing shape/dtype.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.set-sizes-contiguous-fn]
> ET_DEPRECATED

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.set-sizes-contiguous-fn]
> DEPRECATED (`ET_DEPRECATED`) public wrapper around
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`
> with panic-on-failure semantics (matches `at::TensorImpl::
> set_sizes_contiguous`). Callers should use `torch::executor::resize_tensor()`
> / `resize_tensor_impl()` instead.
>
> Steps:
> 1. Call `internal_resize_contiguous(new_sizes)` and capture the returned
>    `Error err`.
> 2. Fatal check (`ET_CHECK_MSG`): assert `err == Error::Ok`; on any non-Ok
>    error this aborts ("Could not resize Tensor; see logs for details"). Unlike
>    `internal_resize_contiguous`, this function does not return an error — it
>    panics.
> Returns void.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.shape-dynamism-fn]
> TensorShapeDynamism shape_dynamism() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.shape-dynamism-fn]
> Returns the mutability policy of the tensor's shape: returns the
> `shape_dynamism_` member (a `TensorShapeDynamism`, one of `STATIC`,
> `DYNAMIC_BOUND`, `DYNAMIC_UNBOUND`). `shape_dynamism_` is `const`, set at
> construction. This value governs whether/how
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`
> permits resizing. Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.size-fn]
> ssize_t size(ssize_t dim) const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.size-fn]
> Returns the size of the tensor along dimension `dim`.
>
> Steps:
> 1. Bounds check (`ET_CHECK_MSG`, fatal): assert `dim < dim_ && dim >= 0`. An
>    out-of-range `dim` (negative — note negative indices are NOT supported — or
>    `>= dim_`) aborts with "Dimension out of range (expected to be in range of
>    [0, <dim_-1>], but got <dim>". No error return.
> 2. Return `sizes_[dim]` as an `ssize_t`. The return type is `ssize_t` (not
>    `SizesType`) for at::Tensor API compatibility, even though it holds one
>    `int32_t` size element.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.sizes-fn]
> const ArrayRef<SizesType> sizes() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.sizes-fn]
> Returns the per-dimension sizes as a read-only view.
>
> Constructs and returns an `ArrayRef<SizesType>` over the `sizes_` pointer with
> length `dim_` (cast to `size_t`). Element type `SizesType` is `int32_t`. The
> view aliases the tensor's `sizes_` storage (no copy); the tensor does not own
> that storage. For a scalar (`dim_ == 0`) this is an empty ArrayRef. Pure
> accessor. Unlike
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.size-fn]`,
> no bounds check is involved.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.strides-fn]
> const ArrayRef<StridesType> strides() const

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.strides-fn]
> Returns the per-dimension strides as a read-only view.
>
> Constructs and returns an `ArrayRef<StridesType>` over the `strides_` pointer
> with length `dim_` (cast to `size_t`). Element type `StridesType` is
> `int32_t`. The view aliases the tensor's `strides_` storage (no copy); the
> tensor does not own that storage. For a scalar (`dim_ == 0`) this is an empty
> ArrayRef. Strides are updated on dynamic resize by
> `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`.
> Pure accessor.

> [spec:et:def:tensor-impl.executorch.runtime.etensor.tensor-impl.tensor-impl-fn]
> TensorImpl::TensorImpl( ScalarType type, ssize_t dim, SizesType* sizes, void* data, DimOrderType* dim_order, StridesType* strides, TensorShapeDynamism dynamism, DeviceType device_type, DeviceIndex device_index) : sizes_(sizes), dim_order...

> [spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.tensor-impl-fn]
> Constructs a TensorImpl from externally-owned sizes/strides/dim_order/data
> arrays. The TensorImpl does NOT own any of these arrays; the caller must keep
> them alive longer than the TensorImpl.
>
> Parameters (with defaults): `type` (ScalarType), `dim` (rank), `sizes`
> (SizesType* of length `dim`), `data` (void*, default null), `dim_order`
> (DimOrderType* of length `dim`, default null), `strides` (StridesType* of
> length `dim`, default null), `dynamism` (TensorShapeDynamism, default
> `STATIC`), `device_type` (default `DeviceType::CPU`), `device_index` (default
> `0`).
>
> Member initialization (in declaration order):
> - `sizes_ = sizes`, `dim_order_ = dim_order`, `strides_ = strides`,
>   `data_ = data` (pointers stored directly, no copy).
> - `dim_ = dim` (const).
> - `numel_ = compute_numel(sizes, dim)` per
>   `[spec:et:sem:tensor-impl.executorch.runtime.etensor.compute-numel-fn]`
>   (this aborts if a size is negative, or if `sizes` is null while `dim != 0`).
> - `numel_bound_ = numel_` — captures the initial element count as the fixed
>   upper capacity used by dynamic resizing in
>   `[spec:et:sem:tensor-impl.executorch.runtime.etensor.tensor-impl.internal-resize-contiguous-fn]`.
> - `type_ = type` (const), `shape_dynamism_ = dynamism` (const),
>   `device_ = Device(device_type, device_index)`.
>
> Body validation (both fatal `ET_CHECK_MSG`, in order):
> 1. `isValid(type_)` must be true; otherwise abort ("Invalid type <type>").
> 2. `dim_ >= 0`; otherwise abort ("Dimension must be non-negative, got
>    <dim>"). (Note: because `numel_` is computed via `compute_numel` in the
>    initializer list before the body runs, a negative-size or null-sizes abort
>    from `compute_numel` fires before these body checks.)
>
> The default constructor is deleted; a TensorImpl always requires at least
> `type`, `dim`, and `sizes`.

