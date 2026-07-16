# runtime/executor/memory_manager.h

> [spec:et:def:memory-manager.executorch.runtime.memory-manager]
> class MemoryManager final {
>   MemoryAllocator* method_allocator_;
>   HierarchicalAllocator* planned_memory_;
>   MemoryAllocator* temp_allocator_;
> }

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn]
> bool has_device_memory() const

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn]
> Const observer that reports whether any planned buffer carries device
> metadata (i.e. the memory setup is not CPU-only).
>
> Computes `planned_buffer_devices()` (see
> `[spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn]`)
> and returns `true` iff the resulting span is non-empty (its `.size() > 0`).
> Returns `false` when `planned_memory_` is null, when the underlying
> HierarchicalAllocator has no per-buffer device metadata (CPU-only program),
> or when there are zero planned buffers. No allocation, no mutation, no
> failure path.

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.memory-manager-fn]
> explicit MemoryManager( MemoryAllocator* method_allocator, HierarchicalAllocator* planned_memory = nullptr, MemoryAllocator* temp_allocator = nullptr) : method_allocator_(method_allocator), planned_memory_(planned_memory), temp_allocator...

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn]
> Explicit constructor that stores three borrowed allocator pointers without
> taking ownership; each pointer must outlive the MemoryManager and the Method
> that uses it.
>
> Parameters (last two default to `nullptr`):
> - `method_allocator`: allocator used while loading a Method and allocating its
>   internal structures. Required (callers pass a valid, non-null allocator);
>   the constructor itself does not null-check it.
> - `planned_memory` (default `nullptr`): the HierarchicalAllocator holding the
>   memory-planned buffers for mutable tensor data during execution, plus any
>   per-buffer device metadata. May be null when the Method uses no
>   memory-planned tensor data. Buffer count/sizes must match the Program's
>   `MethodMeta::num_memory_planned_buffers()` /
>   `memory_planned_buffer_size(N)` (not validated here).
> - `temp_allocator` (default `nullptr`): allocator for temporary data during
>   kernel/delegate execution, reset after every kernel/delegate call. May be
>   null when no kernel/delegate allocates temporary data.
>
> Steps:
> 1. Initialize the fields `method_allocator_ = method_allocator`,
>    `planned_memory_ = planned_memory`, `temp_allocator_ = temp_allocator`
>    (pointers stored verbatim; no copying of the pointed-to objects).
> 2. Assert `method_allocator != temp_allocator` via `ET_CHECK_MSG` with message
>    "method allocator cannot be the same as temp allocator". This is a fatal
>    check (aborts/panics on failure in default configs) rather than a
>    recoverable Error; the same allocator instance must not serve both the
>    method and temp roles. Note that when both default to `nullptr` they compare
>    equal and this check would fail, so a null `method_allocator` combined with a
>    null `temp_allocator` is disallowed.
>
> A deprecated four-argument overload
> `MemoryManager(constant_allocator, non_constant_allocator, runtime_allocator,
> temporary_allocator)` exists as a compatibility shim: it delegates to this
> constructor with `method_allocator = runtime_allocator`,
> `planned_memory = non_constant_allocator`,
> `temp_allocator = temporary_allocator`, and ignores `constant_allocator`
> entirely. New Rust code need not reproduce this deprecated overload.

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.method-allocator-fn]
> MemoryAllocator* method_allocator() const

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn]
> Const accessor. Returns the stored `method_allocator_` pointer unchanged (the
> same value passed at construction). No null-check, no mutation, no allocation.
> This is the allocator the runtime uses for internal structures while loading a
> Method; callers must not use it after the associated Method has been loaded.

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn]
> Span<const etensor::Device> planned_buffer_devices() const

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn]
> Const accessor returning a `Span<const etensor::Device>` (a borrowed,
> non-owning view: pointer + length) of per-buffer device metadata, one entry
> per planned memory buffer.
>
> Steps:
> 1. If `planned_memory_ == nullptr`, return an empty span `{}` (null pointer,
>    length 0).
> 2. Otherwise delegate to `planned_memory_->planned_buffer_devices()` and return
>    its result verbatim. That underlying span has the same element count as the
>    planned buffers, or is empty for a CPU-only program that supplied no device
>    metadata.
>
> This is a thin, non-owning wrapper over
> `HierarchicalAllocator::planned_buffer_devices()`. The returned span borrows
> from the HierarchicalAllocator and is valid only while it outlives the caller.
> No allocation, no mutation, no failure path.

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.planned-memory-fn]
> HierarchicalAllocator* planned_memory() const

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn]
> Const accessor. Returns the stored `planned_memory_` pointer unchanged (the
> same HierarchicalAllocator* passed at construction, which may be `nullptr` when
> the Method uses no memory-planned tensor data). No null-check, no mutation, no
> allocation. This provides the memory-planned buffers used for mutable tensor
> data during execution.

> [spec:et:def:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn]
> MemoryAllocator* temp_allocator() const

> [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn]
> Const accessor. Returns the stored `temp_allocator_` pointer unchanged (the
> same value passed at construction, which may be `nullptr` when no kernel or
> delegate allocates temporary data). No null-check, no mutation, no allocation.
> This allocator is used for temporary data during kernel/delegate execution and
> is expected to be reset after every kernel or delegate call.

