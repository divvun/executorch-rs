# runtime/core/memory_allocator.h

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator]
> class MemoryAllocator {
>   static constexpr size_t kDefaultAlignment = alignof(void*);
>   uint8_t* const begin_;
>   uint8_t* const end_;
>   uint8_t* cur_;
>   uint32_t const size_;
>   int32_t prof_id_ = -1;
> }

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]
> static inline uint8_t* alignPointer(void* ptr, size_t alignment)

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]
> Rounds a pointer up to the next multiple of `alignment`. `alignment` is
> assumed to be a power of 2 (callers validate this via
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]`
> before calling; behavior is undefined otherwise).
>
> Steps:
> 1. Reinterpret `ptr` as an unsigned integer address `address` of pointer
>    width (uintptr_t).
> 2. Compute `mask = alignment - 1` (also uintptr_t width).
> 3. Compute `address = (address + mask) & ~mask`. Adding `mask` then masking
>    off the low bits rounds the address up to the nearest multiple of
>    `alignment`; if `address` is already aligned it is left unchanged.
> 4. Reinterpret the resulting integer back as a `uint8_t*` and return it.
>
> Note: the `address + mask` addition can wrap around the address space if
> `ptr` is very near UINTPTR_MAX; the caller
> (`[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`)
> guards against out-of-range results by comparing the returned pointer's
> offset against the available space rather than relying on this function to
> detect overflow.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
> virtual void* allocate(size_t size, size_t alignment = kDefaultAlignment)

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
> Bump-pointer allocation of `size` bytes with the given `alignment` (default
> `kDefaultAlignment` = `alignof(void*)`). This is `virtual`; subclasses may
> override. The default implementation:
>
> 1. If the allocator has zero capacity (`begin_` is null OR `end_` is null —
>    the latter occurs when the constructor detected address-space overflow),
>    log an Error ("allocate() on zero-capacity allocator") and return
>    `nullptr`.
> 2. If `alignment` is not a power of 2 (per
>    `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]`;
>    note 0 is not a power of 2), log an Error and return `nullptr`.
> 3. Compute `start = alignPointer(cur_, alignment)` (see
>    `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]`),
>    the next position at/after `cur_` that is a multiple of `alignment`.
> 4. Compute `padding = start - cur_` (bytes skipped for alignment) and
>    `available = end_ - cur_` (bytes remaining from the current cursor to the
>    end of the buffer), both as size_t.
> 5. If `padding > available` OR `size > available - padding` (i.e. the
>    aligned allocation does not fit in the remaining space), log an Error
>    reporting `padding + size` requested vs `available` and return `nullptr`.
>    The `padding > available` check is evaluated first so that
>    `available - padding` never underflows.
> 6. Compute `end = start + size`.
> 7. Record the number of bytes consumed for profiling via
>    EXECUTORCH_TRACK_ALLOCATION(prof_id_, end - cur_). The consumed count is
>    `end - cur_` (padding + size), not `end - start`, so alignment padding is
>    counted as used.
> 8. Advance `cur_ = end` and return `start` as a `void*`.
>
> On success the returned pointer is aligned to `alignment` and points to at
> least `size` uninitialized bytes. `size == 0` succeeds and returns the
> aligned cursor without advancing beyond the padding.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn]
> T* allocateInstance(size_t alignment = alignof(T))

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn]
> Templated convenience wrapper that allocates uninitialized storage for a
> single instance of type `T`. Calls
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`
> with `size = sizeof(T)` and the given `alignment` (default `alignof(T)`),
> then reinterprets the resulting `void*` as `T*` and returns it. Returns
> `nullptr` on allocation failure (propagated from `allocate`). The memory is
> NOT initialized/constructed — the caller is responsible for placement-new or
> writing valid bytes before use.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn]
> T* allocateList(size_t size, size_t alignment = alignof(T))

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn]
> Templated convenience wrapper that allocates uninitialized storage for a
> contiguous array of `size` elements of type `T`.
>
> 1. Compute the total byte count `bytes_size = size * sizeof(T)` using a
>    checked multiply. If the multiplication overflows size_t, log an Error
>    ("Failed to allocate list: size(...) * sizeof(T)(...) overflowed") and
>    return `nullptr`.
> 2. Otherwise call
>    `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`
>    with `bytes_size` and the given `alignment` (default `alignof(T)`),
>    reinterpret the returned `void*` as `T*`, and return it.
>
> Returns `nullptr` on overflow or on allocation failure. The memory is NOT
> initialized. `size == 0` yields `bytes_size == 0` and defers to `allocate`,
> which succeeds.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
> virtual uint8_t* base_address() const

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
> Returns `begin_`, the base address of the allocator's memory buffer (the
> `base_address` passed to the constructor). `virtual`; subclasses may
> override. No side effects.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]
> void enable_profiling(ET_UNUSED const char* name)

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]
> Registers this allocator with the profiler under the given `name` and stores
> the returned profiler id into `prof_id_`. Implemented as
> `prof_id_ = EXECUTORCH_TRACK_ALLOCATOR(name)`. In builds where profiling is
> compiled out, the `name` argument is unused and the macro is a no-op that
> yields the sentinel/invalid id. After this call, subsequent successful
> allocations attribute their consumed bytes to `prof_id_` (see
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`).

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
> virtual size_t free_size() const

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
> Returns the number of bytes still available for allocation from the current
> cursor: `end_ - cur_`, cast to size_t. This does NOT account for any
> alignment padding a future allocation may consume, so an allocation of
> `free_size()` bytes can still fail if it requires alignment padding.
> `virtual`; the default reflects the bump cursor. Subclasses using a
> different accounting scheme should override to stay consistent with
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]`.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]
> static constexpr bool isPowerOf2(size_t value)

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]
> Returns true iff `value` is a positive integer power of 2. Computed as
> `value != 0 && (value & (value - 1)) == 0`: a power of 2 has exactly one bit
> set, so `value & (value - 1)` clears that bit and yields 0. `value == 0`
> returns false. `constexpr` and pure (no side effects).

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.memory-allocator-fn]
> MemoryAllocator(uint32_t size, uint8_t* base_address)

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.memory-allocator-fn]
> Constructs a bump allocator over the buffer `[base_address, base_address +
> size)`. Does not take ownership of the buffer; it must outlive the
> allocator. Initializes fields (in declaration order begin_, end_, cur_,
> size_):
>
> 1. `begin_ = base_address`.
> 2. `end_` is computed to guard against address-space overflow:
>    - If `base_address` is null, `end_ = nullptr`.
>    - Else if adding `size` to `base_address` would overflow the address
>      space (i.e. `UINTPTR_MAX - (uintptr_t)base_address < size`),
>      `end_ = nullptr`.
>    - Otherwise `end_ = base_address + size`.
> 3. `cur_ = base_address` (cursor starts at the base).
> 4. `size_ = size`.
>
> Then two invariant checks (ET_CHECK_MSG — fatal abort on failure, not a
> recoverable error):
> - `base_address != null || size == 0`: a null base is only permitted for a
>   zero-size allocator.
> - `base_address == null || size == 0 || (UINTPTR_MAX -
>   (uintptr_t)base_address >= size)`: a non-null, non-empty allocator must not
>   overflow the address space.
>
> Note: `prof_id_` is not set here; it retains its default of -1 until
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]`
> is called. A zero-capacity allocator (null begin_ or null end_) causes
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`
> to always return nullptr.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.prof-id-fn]
> int32_t prof_id() const

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.prof-id-fn]
> Protected accessor returning `prof_id_`, the profiler id assigned by
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]`,
> or -1 if profiling was never enabled. No side effects.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
> virtual void reset()

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
> Resets the bump cursor to the base of the buffer: `cur_ = begin_`. This makes
> all previously allocated space available again but does NOT modify or zero
> the buffer contents; previously returned pointers become dangling/reusable.
> `virtual`; subclasses may override. No return value.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.size-fn]
> virtual uint32_t size() const

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.size-fn]
> Returns `size_`, the total capacity in bytes of the allocator's buffer (the
> `size` passed to the constructor), as uint32_t. `virtual`; subclasses may
> override. No side effects.

> [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
> virtual size_t used_size() const

> [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
> Returns the number of bytes currently consumed from the buffer, measured as
> the bump cursor's offset from the base: `cur_ - begin_`, cast to size_t.
> Immediately after construction or a `reset()` this is 0; each successful
> allocation increases it by the allocation's size plus any alignment padding
> (see
> `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]`).
> `virtual`; subclasses backed by a different allocation scheme should override
> to match their own accounting.

