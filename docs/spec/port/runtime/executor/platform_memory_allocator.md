# runtime/executor/platform_memory_allocator.h

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator]
> class PlatformMemoryAllocator final : public MemoryAllocator {
>   struct AllocationNode { void* data; AllocationNode* next; };
>   AllocationNode* head_ = nullptr;
> }

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn]
> void* allocate(size_t size, size_t alignment = kDefaultAlignment) override

> [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn]
> Overrides `MemoryAllocator::allocate`. Unlike the arena-based base class, this
> allocator ignores the base `(0, nullptr)` arena entirely and instead requests
> memory from the platform abstraction layer (PAL) per call, prepending an
> intrusive linked-list node so every allocation can be freed later. Returns a
> `size`-byte region aligned to `alignment`, or `nullptr` on any failure.
> `alignment` defaults to `kDefaultAlignment` = `alignof(void*)`.
>
> Steps:
> - Validate `alignment` is a power of two via `isPowerOf2(alignment)` (i.e.
>   `alignment != 0 && (alignment & (alignment - 1)) == 0`, see
>   `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]`).
>   On failure: log Error ("Alignment <alignment> is not a power of 2") and
>   return `nullptr`.
> - Compute the total bytes to request from the PAL, with overflow checks:
>   - `alloc_size = sizeof(AllocationNode) + size` via a checked add
>     (`c10::add_overflows`), then `alloc_size = alloc_size + (alignment - 1)`
>     via another checked add. `AllocationNode` is `{ void* data; AllocationNode*
>     next; }`, so `sizeof(AllocationNode)` is two pointers (8 bytes on 32-bit,
>     16 on 64-bit). The `alignment - 1` slack guarantees room to bump the data
>     pointer up to alignment.
>   - If either add overflows `size_t`: log Error ("Allocation size overflow:
>     size <size>, alignment <alignment>") and return `nullptr`.
> - Call `node_memory = runtime::pal_allocate(alloc_size)` (the PAL fallback
>   allocator `et_pal_allocate`). If it returns `nullptr`: log Error ("Failed to
>   allocate <alloc_size> bytes") and return `nullptr`.
> - Reserve the front of the block for the node: `data_ptr = (uint8_t*)node_memory
>   + sizeof(AllocationNode)`.
> - Align the data pointer upward: `aligned_data_ptr = alignPointer(data_ptr,
>   alignment)` = round `data_ptr` up to the next multiple of `alignment`
>   (`(addr + (alignment-1)) & ~(alignment-1)`, see
>   `[spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]`).
> - A debug-only assertion (`ET_DCHECK_MSG`, compiled out in release) checks that
>   `aligned_data_ptr + size` does not exceed `node_memory + alloc_size`; it never
>   affects release behavior.
> - Construct the node in-place at the block start: treat `node_memory` as an
>   `AllocationNode*`, set `new_node->data = aligned_data_ptr`, set
>   `new_node->next = head_`, then push it: `head_ = new_node`. This threads the
>   new allocation onto the front of the singly-linked list.
> - Return `head_->data` (the aligned data pointer). Ownership of the underlying
>   PAL block stays with the allocator; the caller must not free it directly.
>
> Note: the returned data region and the node share one PAL allocation; the node
> lives in the `[node_memory, node_memory + sizeof(AllocationNode))` prefix and
> the caller's data lives at `aligned_data_ptr`. Every successful allocation is
> reclaimed by `reset()` / the destructor, never per-call.

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocation-node]
> struct AllocationNode {
>   void* data;
>   AllocationNode* next;
> }

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.platform-memory-allocator-fn]
> ~PlatformMemoryAllocator() override

> [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.platform-memory-allocator-fn]
> Destructor (`override`). Frees every outstanding PAL allocation by calling
> `reset()` (see
> `[spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn]`),
> which walks the `head_` linked list, `pal_free`s each node block, and sets
> `head_ = nullptr`. After this all memory handed out by `allocate` is invalid.
> In the Rust port this maps to a `Drop` impl that releases the tracked list.

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn]
> void reset() override

> [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn]
> Overrides `MemoryAllocator::reset`. Frees all memory currently tracked by this
> allocator and empties the list, leaving the allocator reusable for fresh
> allocations.
>
> - Set `current = head_`.
> - While `current != nullptr`:
>   - Save `next = current->next` before freeing (the node lives inside the block
>     being freed, so read `next` first).
>   - Call `runtime::pal_free(current)` — free the whole PAL block starting at the
>     node pointer (the same pointer originally returned by `pal_allocate`, since
>     the node sits at the block start). This invalidates both the node and its
>     associated data region.
>   - Advance `current = next`.
> - Set `head_ = nullptr`.
>
> After `reset()`, all pointers previously returned by `allocate` are dangling.
> The order of frees is LIFO (most-recently allocated first) because the list is
> a stack, though free order is not semantically significant.

> [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.operator-fn]
> PlatformMemoryAllocator& operator=(PlatformMemoryAllocator&&) noexcept =

> [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.operator-fn]
> Move-assignment operator, explicitly deleted (`= delete`). Together with the
> deleted copy constructor, copy-assignment, and move constructor, this makes
> `PlatformMemoryAllocator` non-copyable and non-movable: it owns raw PAL
> allocations via `head_`, and copying/moving the head pointer would risk double
> frees. Any attempt to move-assign is a compile-time error. In the Rust port,
> the equivalent is a type that is neither `Clone` nor allowed to be moved out of
> in a way that duplicates ownership of the list (e.g. pin it or hand out only
> references); its sole owner reclaims memory on `Drop`.

