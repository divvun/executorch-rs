# runtime/core/device_memory_buffer.cpp, runtime/core/device_memory_buffer.h

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer]
> class DeviceMemoryBuffer final {
>   void* ptr_ = nullptr;
>   size_t size_ = 0;
>   DeviceAllocator* allocator_ = nullptr;
>   etensor::DeviceIndex device_index_ = 0;
> }

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn]
> Span<uint8_t> as_span() const

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn]
> `const` accessor. Returns a `Span<uint8_t>` constructed from the buffer's
> two fields: `{ static_cast<uint8_t*>(ptr_), size_ }`, i.e. the device
> pointer reinterpreted as a `uint8_t*` together with the byte length. No
> allocation, no copy, no dereference of the pointer occurs.
>
> If the buffer is empty or moved-from (`ptr_ == nullptr`, `size_ == 0`), the
> returned span has a null data pointer and length 0.
>
> Important: the returned span may wrap a device (e.g. CUDA) pointer, which is
> not necessarily dereferenceable from the host. It is intended only for
> pointer arithmetic (e.g. by `HierarchicalAllocator`), which computes offsets
> into the span without ever reading or writing the underlying bytes from the
> CPU side. The span does not own the memory; its validity is tied to the
> lifetime of this `DeviceMemoryBuffer`.

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn]
> Result<DeviceMemoryBuffer> DeviceMemoryBuffer::create( size_t size, etensor::DeviceType type, etensor::DeviceIndex index, size_t alignment)

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn]
> Static factory that allocates device memory and wraps it in an owning
> `DeviceMemoryBuffer`. Parameters: `size` (bytes to allocate),
> `type` (`etensor::DeviceType`), `index` (`etensor::DeviceIndex`, default 0),
> `alignment` (bytes, default `DeviceAllocator::kDefaultAlignment`).
>
> Steps:
> 1. Look up the allocator: call `get_device_allocator(type)`
>    (`[spec:et:sem:device-allocator.executorch.runtime.get-device-allocator-fn]`),
>    which returns the registered `DeviceAllocator*` for `type` or `nullptr`.
> 2. If the returned allocator is `nullptr`, log at Error level "No device
>    allocator registered for device type %d" (with `type` cast to `int`) and
>    return `Error::NotFound`.
> 3. Otherwise call `allocator->allocate(size, index, alignment)`
>    (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]`).
> 4. If that `Result<void*>` is not ok (`!result.ok()`), return its error code
>    unchanged (`result.error()`); no partial buffer is constructed.
> 5. On success, construct and return a `DeviceMemoryBuffer` via its private
>    4-argument constructor `(ptr = result.get(), size, allocator, index)`,
>    taking ownership so the pointer is freed via the same allocator on
>    destruction.
>
> Return type is `Result<DeviceMemoryBuffer>`: either the owning buffer or an
> `Error`. The allocated memory is not zeroed or otherwise initialized here.

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn]
> void* data() const

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn]
> `const` accessor that returns the raw device pointer `ptr_` as `void*`. No
> side effects. Returns `nullptr` when the buffer is default-constructed,
> empty, or moved-from. The pointer refers to device memory and is not
> guaranteed to be dereferenceable from the host. Ownership is not transferred;
> the memory remains owned by this `DeviceMemoryBuffer` and is freed on
> destruction.

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.device-memory-buffer-fn]
> DeviceMemoryBuffer(DeviceMemoryBuffer&& other) noexcept

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.device-memory-buffer-fn]
> Move constructor (`noexcept`). Transfers ownership of the allocation from
> `other` to the new object, leaving `other` in a valid empty state that will
> not free anything on destruction.
>
> Steps:
> 1. Copy all four fields from `other` into the new object: `ptr_ =
>    other.ptr_`, `size_ = other.size_`, `allocator_ = other.allocator_`,
>    `device_index_ = other.device_index_`.
> 2. Null out `other`'s ownership fields so its destructor is a no-op:
>    `other.ptr_ = nullptr`, `other.size_ = 0`, `other.allocator_ = nullptr`.
>    (Note: `other.device_index_` is intentionally left unchanged; it is
>    harmless because the destructor only frees when both `ptr_` and
>    `allocator_` are non-null.)
>
> The destination is assumed to be freshly constructed (this is a constructor,
> so no prior allocation is released). `DeviceMemoryBuffer` is move-only; the
> copy constructor and copy assignment are deleted.

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn]
> size_t size() const

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn]
> `const` accessor that returns the field `size_`: the number of bytes
> requested when the allocation was created (via
> `[spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn]`).
> No side effects. Returns 0 for a default-constructed, empty, or moved-from
> buffer. Note this is the requested logical size, not necessarily the
> physically allocated size the underlying device API may have rounded up to.

> [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.operator-fn]
> DeviceMemoryBuffer& operator=(DeviceMemoryBuffer&& other) noexcept

> [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.operator-fn]
> Move assignment operator (`noexcept`). Releases any allocation this object
> currently owns, then takes over `other`'s allocation, leaving `other` empty.
>
> Steps:
> 1. Self-assignment guard: if `this == &other`, do nothing and return
>    `*this`.
> 2. Otherwise, free the currently-owned memory if any: if `ptr_ != nullptr`
>    && `allocator_ != nullptr`, call `allocator_->deallocate(ptr_,
>    device_index_)`
>    (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator.deallocate-fn]`).
> 3. Copy all four fields from `other`: `ptr_`, `size_`, `allocator_`,
>    `device_index_`.
> 4. Null out `other`'s ownership fields: `other.ptr_ = nullptr`,
>    `other.size_ = 0`, `other.allocator_ = nullptr` (its `device_index_` is
>    left unchanged, which is harmless since deallocation only runs when both
>    `ptr_` and `allocator_` are non-null).
> 5. Return `*this`.
>
> Copy assignment is deleted; this type is move-only.

