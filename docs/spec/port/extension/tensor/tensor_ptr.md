# extension/tensor/tensor_ptr.cpp, extension/tensor/tensor_ptr.h

> [spec:et:def:tensor-ptr.executorch.extension.casted-data-fn]
> std::vector<uint8_t> casted_data(casted_bytes)

> [spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn]
> A local `std::vector<uint8_t> casted_data(casted_bytes)` allocated inside the
> vector-data `make_tensor_ptr<T>` template overload on the type-mismatch path
> (`type != deduced_type`). It is the raw byte buffer that receives the element-by-
> element cast of the input `std::vector<T>` into the requested scalar `type`.
>
> Context: it is created only after (a) `data.size()` was verified to equal the
> tensor's `numel`, (b) `runtime::canCast(deduced_type, type)` was verified true,
> and (c) `casted_bytes = data.size() * elementSize(type)` was computed with an
> overflow guard (`c10::mul_overflows`, fatal ET_CHECK_MSG on overflow). The vector
> is zero-initialized to exactly `casted_bytes` bytes.
>
> It is then filled by an `ET_SWITCH_REALHBBF16_AND_UINT_TYPES` dispatch on `type`
> that, for the selected C++ target type `CTYPE`, runs
> `std::transform` over `data`, writing `static_cast<CTYPE>(val)` for each element
> into `reinterpret_cast<CTYPE*>(casted_data.data())`. See the cast-set semantics
> in `[spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]`; the
> accepted `type` set is REALHBBF16 plus the unsigned integer types.
>
> After filling, `casted_data` is moved into a `shared_ptr<vector<uint8_t>>` and its
> raw pointer is passed to the primary `make_tensor_ptr(void* data, ...)` factory
> with a deleter that captures (and thus keeps alive) the shared buffer, so the
> Tensor's data lifetime is tied to this cast buffer.

> [spec:et:def:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]
> TensorPtr clone_tensor_ptr( const executorch::aten::Tensor& tensor, executorch::aten::ScalarType type)

> [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]
> Creates a new owning `TensorPtr` whose Tensor has the same shape/layout as
> `tensor` but with a freshly allocated, owned copy of the data, cast to scalar
> type `type`.
>
> Steps:
> 1. CPU check (fatal ET_CHECK_MSG, aborts on failure): in the portable build
>    (not USE_ATEN_LIB) require `tensor.device_type() == DeviceType::CPU`; in the
>    ATen build require `tensor.is_cpu()`. Non-CPU tensors must first be moved to
>    CPU (via `[spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn]`).
> 2. Copy metadata into owning vectors: `sizes` from `tensor.sizes()`; `dim_order`
>    from `tensor.dim_order()` in the portable build (left empty in USE_ATEN_LIB,
>    where dim_order is not available); `strides` from `tensor.strides()`.
> 3. Choose dynamism: `DYNAMIC_BOUND` by default; in the portable build override
>    with `tensor.shape_dynamism()`.
> 4. Null-data case: if `tensor.const_data_ptr()` is null, return
>    `make_tensor_ptr(sizes, /*data=*/nullptr, dim_order, strides, type, dynamism)`
>    — a same-shape tensor with null data, no allocation or copy.
> 5. Same-type fast path: if `tensor.scalar_type() == type`, allocate a
>    `std::vector<uint8_t>` copy of the raw bytes `[data, data + tensor.nbytes())`
>    and pass it to the raw-buffer `make_tensor_ptr(sizes, bytes, dim_order,
>    strides, tensor_type, dynamism)`. This is a byte-for-byte copy, no conversion.
> 6. Cast path (types differ): require `runtime::canCast(tensor_type, type)`
>    (fatal ET_CHECK_MSG otherwise). Compute `clone_nbytes = numel *
>    elementSize(type)` with an overflow guard (`c10::mul_overflows`, fatal on
>    overflow) and allocate a zeroed `std::vector<uint8_t> data(clone_nbytes)`.
>    Then a nested double `ET_SWITCH_REALHBBF16_AND_UINT_TYPES` dispatch runs: the
>    outer switch on the source `tensor_type` selects `CTYPE_FROM`, the inner switch
>    on the destination `type` selects `CTYPE_TO`; `std::transform` over the
>    `tensor_numel` source elements writes `static_cast<CTYPE_TO>(val)` for each.
>    Conversion is a plain C++ `static_cast` per element (float<->int truncation,
>    integer wrap/narrowing, etc. as the underlying static_cast defines). On an
>    unsupported dtype the local `ctx.fail(...)` aborts (see
>    `[spec:et:sem:tensor-ptr.executorch.extension.fail-fn]`). REALHBBF16_AND_UINT
>    denotes the real dtypes plus Half, Bool, BFloat16 and the unsigned integer
>    types; both `tensor_type` and `type` must fall in this set.
> 7. Return `make_tensor_ptr(sizes, cast-bytes, dim_order, strides, type, dynamism)`.
>    In all non-null cases the returned TensorPtr owns its data buffer (via the
>    raw-buffer factory whose deleter holds the vector).

> [spec:et:def:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn]
> TensorPtr clone_tensor_ptr_to( const TensorPtr& tensor, executorch::aten::Device target)

> [spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-to-fn]
> Copies a tensor's data across the CPU/device boundary onto `target`, returning a
> new `TensorPtr` backed by `target` memory. Portable build only (compiled out
> under USE_ATEN_LIB, as it relies on the ExecuTorch DeviceAllocator).
>
> Steps:
> 1. Determine `source = tensor->device()`.
> 2. Reject unsupported directions (fatal ET_CHECK_MSG, aborts):
>    - if both `source` and `target` are CPU: abort — CPU-to-CPU must use
>      `[spec:et:sem:tensor-ptr.executorch.extension.clone-tensor-ptr-fn]`.
>    - if neither `source` nor `target` is CPU (device-to-device): abort — route
>      through CPU.
>    So exactly one of source/target is CPU.
> 3. Read `nbytes = tensor->nbytes()` and `src_data = tensor->const_data_ptr()`;
>    require `src_data != nullptr` (fatal check).
> 4. Pick the allocator from whichever side is NOT CPU:
>    `device = target.is_cpu() ? source : target`, then
>    `allocator = runtime::get_device_allocator(device.type())`; require it
>    non-null (fatal check, message names the device type).
> 5. Copy metadata into owning vectors: `sizes`, `dim_order`, `strides` from the
>    source tensor's respective accessors.
> 6. Direction dispatch:
>    - Device-to-host (`target.is_cpu()`): allocate `std::vector<uint8_t>
>      cpu_data(nbytes)`, call `allocator->copy_device_to_host(cpu_data.data(),
>      src_data, nbytes, source.index())`, require `Error::Ok` (fatal check), then
>      return `make_tensor_ptr(sizes, cpu_data, dim_order, strides,
>      tensor->scalar_type(), tensor->shape_dynamism())` — an owning CPU tensor.
>    - Host-to-device (else): `allocator->allocate(nbytes, target.index())`,
>      require ok (fatal check), obtain `device_data`; call
>      `allocator->copy_host_to_device(device_data, src_data, nbytes,
>      target.index())`, require `Error::Ok` (fatal check); then return
>      `make_tensor_ptr(sizes, device_data, dim_order, strides,
>      tensor->scalar_type(), tensor->shape_dynamism(), deleter, target)` where the
>      deleter captures `allocator` and `target` and calls
>      `allocator->deallocate(ptr, target.index())`, so the returned device tensor
>      owns and frees its device memory on destruction.
> 7. Scalar type and shape dynamism are always inherited from the source tensor;
>    no dtype conversion is performed (raw byte copy of `nbytes`).

> [spec:et:def:tensor-ptr.executorch.extension.fail-fn]
> [[noreturn]] void fail(torch::executor::Error /* error */)

> [spec:et:sem:tensor-ptr.executorch.extension.fail-fn]
> `[[noreturn]] void fail(torch::executor::Error)` is a method on a minimal local
> context struct passed to `ET_SWITCH_REALHBBF16_AND_UINT_TYPES` in the vector-data
> `make_tensor_ptr<T>` overload. The `ET_SWITCH` machinery invokes `ctx.fail(...)`
> when the runtime scalar type does not match any case in the type set. The body
> unconditionally calls `ET_CHECK_MSG(false, "Unsupported dtype in
> make_tensor_ptr")`, which aborts the process. The `Error` argument is ignored.
> It never returns. (The clone path in tensor_ptr.cpp defines an analogous
> `fail` with the message "Unsupported dtype in clone_tensor_ptr".) In a Rust port
> this corresponds to the unreachable/panic default arm of the dtype dispatch.

> [spec:et:def:tensor-ptr.executorch.extension.make-tensor-ptr-fn]
> TensorPtr make_tensor_ptr( std::vector<executorch::aten::SizesType> sizes, void* data, std::vector<executorch::aten::DimOrderType> dim_order, std::vector<executorch::aten::StridesType> strides, executorch::aten::ScalarType type, executor...

> [spec:et:sem:tensor-ptr.executorch.extension.make-tensor-ptr-fn]
> Primary factory: constructs a `TensorPtr` (shared_ptr to Tensor) over a caller-
> provided `void* data` buffer, with the given sizes/dim_order/strides/type/
> dynamism/deleter/device. The Tensor does not own `data` unless the supplied
> `deleter` frees it; the factory only manages the metadata's lifetime. This is
> the overload every other `make_tensor_ptr` variant ultimately calls.
>
> Steps:
> 1. `dim = sizes.size()`.
> 2. Validate optional metadata sizes (fatal ET_CHECK_MSG, aborts on failure):
>    `dim_order` must be empty or of length `dim`; `strides` must be empty or of
>    length `dim`.
> 3. Derive dim_order if absent: if `dim_order` is empty, resize it to `dim` and
>    fill it with `0..dim-1` (`std::iota`). Then, if `strides` was provided
>    (non-empty), stable-sort the dim_order indices so that dimensions with larger
>    stride come first (`sort` with comparator `strides[a] > strides[b]`), i.e.
>    reconstruct dim_order from the given strides (descending stride order).
> 4. Compute canonical strides from `(sizes, dim_order)`:
>    `computed_strides` of length `dim` via `runtime::dim_order_to_stride(...)`;
>    require it returns `Error::Ok` (fatal check "Failed to compute strides.").
> 5. If `strides` was provided, validate each dimension: for every `i`,
>    `strides[i] == computed_strides[i]` OR `sizes[i] == 1` (a size-1 dim may carry
>    any stride). Any mismatch on a non-size-1 dim is a fatal ET_CHECK_MSG abort.
> 6. Replace `strides` with `computed_strides` (the canonical strides are always
>    used for the actual TensorImpl).
> 7. Portable build (not USE_ATEN_LIB): construct an `aten::TensorImpl(type, dim,
>    sizes.data(), data, dim_order.data(), strides.data(), shapeDynamism, device
>    type, device index)` where the shape dynamism passed is `dynamism` when
>    `dim > 0` but forced to `STATIC` when `dim == 0` (scalar). Move the TensorImpl
>    plus the owning `sizes`/`dim_order`/`strides` vectors and the `deleter` into a
>    heap-allocated `Storage` (see
>    `[spec:et:sem:tensor-ptr.executorch.extension.storage.storage-fn]`), held by a
>    `shared_ptr<Storage>`. Return a `shared_ptr<Tensor>` that aliases
>    `&storage->tensor` while sharing ownership with the Storage (aliasing-
>    constructor), so the Storage (and thus metadata + deleter) lives exactly as
>    long as the returned TensorPtr.
> 8. ATen build (USE_ATEN_LIB): build `c10::TensorOptions` from `type` and `device`,
>    create a `c10::Storage` sized by `at::detail::computeStorageNbytes(sizes,
>    strides, itemsize)` wrapping `data` with an `InefficientStdFunctionContext`
>    data-ptr that carries `deleter`; make an intrusive `TensorImpl`, call
>    `set_sizes_and_strides(sizes, strides)`, and return a `shared_ptr<Tensor>`
>    wrapping it. (dim_order is unused in this build.)
>
> The other overloads: the `std::vector<uint8_t>` raw-buffer overload first
> validates `data.size()` equals `numel * elementSize(type)` (with `safe_numel`
> and `mul_overflows` overflow guards, fatal checks) and then calls this factory
> with a deleter capturing (owning) the byte vector. The templated
> `std::vector<T>` / initializer-list / scalar overloads deduce the scalar type
> from `T`, validate `data.size() == numel`, optionally cast when `type` differs
> from the deduced type (see
> `[spec:et:sem:tensor-ptr.executorch.extension.casted-data-fn]`), and forward a
> deleter that owns the data vector.

> [spec:et:def:tensor-ptr.executorch.extension.resize-tensor-ptr-fn]
> runtime::Error resize_tensor_ptr( TensorPtr& tensor, const std::vector<executorch::aten::SizesType>& sizes)

> [spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn]
> Definition (in tensor_ptr.cpp) of `resize_tensor_ptr`. Resizes the Tensor managed
> by `tensor` to the new `sizes` and returns a `runtime::Error`.
>
> Steps:
> 1. Wrap the caller's `sizes` vector in an `ArrayRef<SizesType>(sizes.data(),
>    sizes.size())` (no copy).
> 2. Delegate to `ET_RUNTIME_NAMESPACE::resize_tensor(*tensor, that ArrayRef)` and
>    return its `Error` result directly. See
>    `[spec:et:sem:tensor-util.executorch.resize-tensor-fn]` for the resize
>    semantics (bounds/dynamism validation against the tensor's capacity, in-place
>    metadata update, no reallocation of the data buffer). No CPU/device or dtype
>    checks are added here; behavior and error codes are exactly those of the
>    underlying `resize_tensor` util.

> [spec:et:def:tensor-ptr.executorch.extension.runtime.error-resize-tensor-ptr-fn]
> ET_NODISCARD

> [spec:et:sem:tensor-ptr.executorch.extension.runtime.error-resize-tensor-ptr-fn]
> Declaration (in tensor_ptr.h) of the same `resize_tensor_ptr` function whose body
> is specified in
> `[spec:et:sem:tensor-ptr.executorch.extension.resize-tensor-ptr-fn]`. It is marked
> `ET_NODISCARD` (the returned `runtime::Error` must not be ignored by the caller)
> and takes `TensorPtr& tensor` plus `const std::vector<SizesType>& sizes`. No
> behavior beyond the definition; the annotation captures the discard-warning
> contract on the return value.

> [spec:et:def:tensor-ptr.executorch.extension.storage]
> struct Storage final {
>   executorch::aten::TensorImpl tensor_impl;
>   executorch::aten::Tensor tensor;
>   std::vector<executorch::aten::SizesType> sizes;
>   std::vector<executorch::aten::DimOrderType> dim_order;
>   std::vector<executorch::aten::StridesType> strides;
>   std::function<void(void*)> deleter;
> }

> [spec:et:def:tensor-ptr.executorch.extension.storage.storage-fn]
> Storage( executorch::aten::TensorImpl&& tensor_impl, std::vector<executorch::aten::SizesType>&& sizes, std::vector<executorch::aten::DimOrderType>&& dim_order, std::vector<executorch::aten::StridesType>&& strides, std::function<void(void...

> [spec:et:sem:tensor-ptr.executorch.extension.storage.storage-fn]
> Constructor of the portable-build (`#ifndef USE_ATEN_LIB`) internal `Storage`
> struct, which co-locates a Tensor with its owned metadata so all share one
> lifetime. Takes rvalue references and moves each into the corresponding member:
> - `tensor_impl(std::move(tensor_impl))` — the moved-in TensorImpl,
> - `tensor(&this->tensor_impl)` — the Tensor is constructed to point at THIS
>   struct's `tensor_impl` member (crucially, at the moved-into copy, not the
>   argument), so the Tensor references stable in-struct storage,
> - `sizes(std::move(sizes))`, `dim_order(std::move(dim_order))`,
>   `strides(std::move(strides))` — the owning metadata vectors whose buffers the
>   TensorImpl points into,
> - `deleter(std::move(deleter))` — optional data-buffer deleter.
>
> Member initialization order matters: `tensor_impl` is initialized before
> `tensor`, so taking `&this->tensor_impl` is valid. On destruction `~Storage`
> invokes `deleter(tensor_impl.mutable_data())` if the deleter is set, freeing the
> managed data buffer. A Rust port must ensure the metadata vectors and TensorImpl
> are pinned/stable in memory (the TensorImpl holds raw pointers into them) and are
> dropped together.

> [spec:et:def:tensor-ptr.executorch.extension.storage.operator-fn]
> Storage& operator=(const Storage&) = delete

> [spec:et:sem:tensor-ptr.executorch.extension.storage.operator-fn]
> `Storage& operator=(const Storage&) = delete` — the copy-assignment operator is
> explicitly deleted, so `Storage` is non-copy-assignable. (Alongside it the copy
> constructor, move constructor, and move-assignment are also deleted, making
> `Storage` non-copyable and non-movable entirely.) This is required because
> `tensor` holds a raw pointer to this object's own `tensor_impl` member and the
> TensorImpl holds raw pointers into the owned metadata vectors; copying or moving
> the struct would leave those self-references dangling. `Storage` is therefore
> only ever heap-allocated and handled through a `shared_ptr`. There is no runtime
> behavior; the equivalent in a Rust port is a pinned, non-`Clone`/non-movable
> owner type.

