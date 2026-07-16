# runtime/core/hierarchical_allocator.h

> [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator]
> class HierarchicalAllocator final {
>   ET_NODISCARD Result<void*>;
>   static constexpr size_t kSpanArraySize = 16;
>   Span<uint8_t> span_array_[kSpanArraySize];
>   Span<Span<uint8_t>> buffers_;
>   Span<const etensor::Device> planned_buffer_devices_;
> }

> [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn]
> get_offset_address( uint32_t memory_id, size_t offset_bytes, size_t size_bytes)

> [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn]
> Resolves `(memory_id, offset_bytes, size_bytes)` to an absolute address
> inside the selected buffer, returning `Result<void*>`. The memory ID is the
> index into `buffers_`. Steps, in order (each ET_CHECK_OR_RETURN_ERROR
> returns early with the named Error on failure without mutating state):
>
> 1. Compute `end_bytes = offset_bytes + size_bytes` using a checked add. If it
>    overflows size_t, return Error::InvalidArgument ("Integer overflow in
>    offset_bytes + size_bytes").
> 2. If `memory_id >= buffers_.size()`, return Error::InvalidArgument ("id ...
>    >= ...").
> 3. Select `buffer = buffers_[memory_id]` (a `Span<uint8_t>`).
> 4. If `end_bytes > buffer.size()` (i.e. the requested range
>    `[offset_bytes, offset_bytes + size_bytes)` does not fit within the
>    buffer), return Error::MemoryAllocationFailed.
> 5. On success, return `buffer.data() + offset_bytes` as a `void*`. This does
>    NOT advance any cursor — the allocator is purely an address-resolution
>    map over pre-planned buffers; repeated calls with the same arguments
>    return the same address.
>
> Note `end_bytes <= buffer.size()` is inclusive: `offset_bytes == buffer.size()`
> with `size_bytes == 0` succeeds and returns a one-past-the-end pointer.
> Buffers may hold device (non-CPU) pointers; only pointer arithmetic is
> performed, never a dereference.

> [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn]
> HierarchicalAllocator( Span<Span<uint8_t>> buffers, Span<const etensor::Device> planned_buffer_devices) : buffers_(buffers), planned_buffer_devices_(planned_buffer_devices)

> [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn]
> Two-argument constructor that stores per-buffer device metadata alongside the
> buffers. Sets `buffers_ = buffers` and
> `planned_buffer_devices_ = planned_buffer_devices` (both stored as spans by
> reference; the underlying storage must outlive the allocator).
>
> Precondition check: `planned_buffer_devices.size() == buffers.size()` — the
> device metadata must have exactly one entry per buffer. Enforced via
> ET_CHECK_MSG, a fatal abort (not a recoverable Error) on mismatch.
>
> Memory IDs equal indices: `buffers[N]` has memory ID `N`, and
> `planned_buffer_devices[N]` describes buffer `N`'s `Device` (type + index),
> letting callers distinguish e.g. `cuda:0` from `cuda:1`.
>
> There is also a single-argument constructor (`def`-only, not a sem rule)
> that sets `buffers_` and leaves `planned_buffer_devices_` empty, for CPU-only
> programs; and a deprecated `(uint32_t n_allocators, MemoryAllocator*
> allocators)` constructor that builds `buffers_` via
> `[spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn]`.

> [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn]
> Span<const etensor::Device> planned_buffer_devices() const

> [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn]
> Returns `planned_buffer_devices_`, a `Span<const etensor::Device>` with one
> entry per buffer (same count as the `buffers` passed to the constructor).
> Each entry carries a device type and index. Returns an empty span when no
> device metadata was supplied (i.e. when the single-argument, CPU-only
> constructor was used). No side effects.

> [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn]
> Span<Span<uint8_t>> to_spans( uint32_t n_allocators, MemoryAllocator* allocators)

> [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn]
> Private legacy helper (used only by the deprecated `MemoryAllocator*`
> constructor) that converts an array of `n_allocators` `MemoryAllocator`
> objects into a `Span<Span<uint8_t>>` backed by the fixed member array
> `span_array_` (capacity `kSpanArraySize` = 16).
>
> 1. Check `n_allocators <= kSpanArraySize`; on failure ET_CHECK_MSG aborts
>    fatally ("n_allocators ... > ...").
> 2. For each `i` in `[0, n_allocators)`: set `span_array_[i] =
>    Span<uint8_t>(allocators[i].base_address(), allocators[i].size())`, i.e. a
>    span over each allocator's base address for its full size (see
>    `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]`
>    and
>    `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.size-fn]`).
> 3. Return `{span_array_, n_allocators}` — a span covering the first
>    `n_allocators` entries of `span_array_`.
>
> Note `span_array_` is declared before `buffers_` so that its initialization
> is not clobbered when `buffers_` is set from this return value in the
> constructor initializer list. This whole path is deprecated; the span-based
> constructors are preferred.

