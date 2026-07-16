//! Literal port of runtime/executor/platform_memory_allocator.h.

use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
use crate::runtime::platform::platform::{pal_allocate, pal_free};

// PORT-NOTE: `EXECUTORCH_TRACK_ALLOCATION` is not exercised by this override —
// the base-class arena is unused. `AllocationNode` is a two-pointer node; the
// C++ code relies on `sizeof(AllocationNode)`, mirrored via
// `size_of::<AllocationNode>()`.

/// PlatformMemoryAllocator is a memory allocator that uses a linked list to
/// manage allocated nodes. It overrides the allocate method of MemoryAllocator
/// using the PAL fallback allocator method `et_pal_allocate`.
// [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator]
// PORT-NOTE: C++ derives from `MemoryAllocator` and constructs the base with
// `(0, nullptr)`. Rust has no inheritance, so the base is held by composition in
// `base_` and its non-overridden methods (isPowerOf2/alignPointer/base_address/
// size/used_size/free_size/enable_profiling) are reached through it. `allocate`
// and `reset` are overridden here; the destructor maps to `Drop`.
pub struct PlatformMemoryAllocator {
    base_: MemoryAllocator,
    // We allocate a little more than requested and use that memory as a node in
    // a linked list, pushing the allocated buffers onto a list that's iterated
    // and freed when the KernelRuntimeContext is destroyed.
    head_: *mut AllocationNode,
}

// [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocation-node]
#[repr(C)]
struct AllocationNode {
    data: *mut core::ffi::c_void,
    next: *mut AllocationNode,
}

impl PlatformMemoryAllocator {
    pub fn new() -> Self {
        PlatformMemoryAllocator {
            base_: MemoryAllocator::new(0, core::ptr::null_mut()),
            head_: core::ptr::null_mut(),
        }
    }

    // [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn]
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn]
    pub fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
        if !MemoryAllocator::is_power_of2(alignment) {
            crate::et_log!(Error, "Alignment {} is not a power of 2", alignment);
            return core::ptr::null_mut();
        }

        // Check for overflow before computing total allocation size.
        // Allocate enough for the node, data, and alignment bump (at most
        // alignment - 1 extra bytes to align the data pointer).
        let mut alloc_size: usize = 0;
        let overflow = match core::mem::size_of::<AllocationNode>().checked_add(size) {
            Some(sum1) => match sum1.checked_add(alignment - 1) {
                Some(sum2) => {
                    alloc_size = sum2;
                    false
                }
                None => true,
            },
            None => true,
        };
        if overflow {
            crate::et_log!(
                Error,
                "Allocation size overflow: size {}, alignment {}",
                size,
                alignment
            );
            return core::ptr::null_mut();
        }

        let node_memory: *mut core::ffi::c_void = pal_allocate(alloc_size);

        // If allocation failed, log message and return nullptr.
        if node_memory.is_null() {
            crate::et_log!(Error, "Failed to allocate {} bytes", alloc_size);
            return core::ptr::null_mut();
        }

        // Compute data pointer.
        let data_ptr: *mut u8 =
            unsafe { (node_memory as *mut u8).add(core::mem::size_of::<AllocationNode>()) };

        // Align the data pointer.
        let aligned_data_ptr: *mut core::ffi::c_void =
            MemoryAllocator::align_pointer(data_ptr as *mut core::ffi::c_void, alignment)
                as *mut core::ffi::c_void;

        // Assert that the alignment didn't overflow the allocated memory.
        debug_assert!(
            (aligned_data_ptr as usize) + size <= (node_memory as usize) + alloc_size,
            "aligned_data_ptr {:p} + size {} > node_memory {:p} + alloc_size {}",
            aligned_data_ptr,
            size,
            node_memory,
            alloc_size
        );

        // Construct the node.
        let new_node: *mut AllocationNode = node_memory as *mut AllocationNode;
        unsafe {
            (*new_node).data = aligned_data_ptr;
            (*new_node).next = self.head_;
        }
        self.head_ = new_node;

        // Return the aligned data pointer.
        unsafe { (*self.head_).data }
    }

    // [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn]
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn]
    pub fn reset(&mut self) {
        let mut current: *mut AllocationNode = self.head_;
        while !current.is_null() {
            let next: *mut AllocationNode = unsafe { (*current).next };
            pal_free(current as *mut core::ffi::c_void);
            current = next;
        }
        self.head_ = core::ptr::null_mut();
    }
}

// PORT-NOTE: C++ `PlatformMemoryAllocator` IS-A `MemoryAllocator` (public
// inheritance) and overrides the virtual `allocate`/`reset`; the other virtuals
// (`base_address`/`size`/`used_size`/`free_size`) are inherited unchanged. Rust
// has no inheritance, so the trait impl routes `allocate`/`reset` to this
// struct's overrides and forwards the rest to the composed `base_`. This makes a
// `PlatformMemoryAllocator` usable wherever the C++ used a `MemoryAllocator*`
// base pointer (e.g. `Method::load` installs it as the temp allocator).
impl MemoryAllocatorBase for PlatformMemoryAllocator {
    fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
        PlatformMemoryAllocator::allocate(self, size, alignment)
    }

    fn base_address(&self) -> *mut u8 {
        self.base_.base_address()
    }

    fn size(&self) -> u32 {
        self.base_.size()
    }

    fn used_size(&self) -> usize {
        self.base_.used_size()
    }

    fn free_size(&self) -> usize {
        self.base_.free_size()
    }

    fn reset(&mut self) {
        PlatformMemoryAllocator::reset(self)
    }
}

// [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.platform-memory-allocator-fn]
// [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.platform-memory-allocator-fn]
impl Drop for PlatformMemoryAllocator {
    fn drop(&mut self) {
        self.reset();
    }
}

// [spec:et:def:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.operator-fn]
// [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.operator-fn]
// PORT-NOTE: copy/move ctors and copy/move assignment are `= delete` in C++
// because the allocator owns raw PAL allocations via `head_`. Not deriving
// `Clone`/`Copy` and owning the list through `Drop` reproduces that: duplicating
// ownership of the list is impossible.

// PORT-NOTE: there is no `platform_memory_allocator_test.cpp` in the C++ suite
// (the allocator is exercised transitively by executor end-to-end tests, which
// need fixtures/kernels the port doesn't have yet). These focused tests pin the
// pure/deterministic `allocate`/`reset` behavior directly against the sem rules
// in docs/spec/port/runtime/executor/platform_memory_allocator.md using the PAL
// fallback allocator (available after `runtime_init`).
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // allocate returns a region aligned to `alignment` and writable for `size`
    // bytes, and threads each block onto the list so reset can reclaim it.
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn/test]
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.reset-fn/test]
    #[test]
    fn platform_memory_allocator_allocate_and_reset() {
        setup();
        let mut allocator = PlatformMemoryAllocator::new();
        assert!(allocator.head_.is_null());

        // A basic allocation returns non-null, aligned memory.
        for &alignment in &[1usize, 4, 16, 64] {
            let p = allocator.allocate(32, alignment);
            assert!(!p.is_null());
            assert_eq!((p as usize) % alignment, 0);
            // The returned region is writable for the requested size.
            unsafe {
                core::ptr::write_bytes(p as *mut u8, 0xAB, 32);
                assert_eq!(*(p as *const u8), 0xAB);
                assert_eq!(*(p as *const u8).add(31), 0xAB);
            }
        }

        // Each successful allocation was pushed onto the list.
        assert!(!allocator.head_.is_null());

        // reset frees everything and empties the list.
        allocator.reset();
        assert!(allocator.head_.is_null());

        // The allocator is reusable after reset.
        let p = allocator.allocate(8, 16);
        assert!(!p.is_null());
        assert_eq!((p as usize) % 16, 0);
    }

    // A non-power-of-two alignment fails the isPowerOf2 check and returns null,
    // without pushing a node.
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn/test]
    #[test]
    fn platform_memory_allocator_rejects_non_power_of_two_alignment() {
        setup();
        let mut allocator = PlatformMemoryAllocator::new();
        assert!(allocator.allocate(32, 3).is_null());
        assert!(allocator.allocate(32, 0).is_null());
        assert!(allocator.head_.is_null());
    }

    // The overflow check on sizeof(AllocationNode) + size + (alignment - 1)
    // returns null rather than requesting a wrapped allocation size.
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.allocate-fn/test]
    #[test]
    fn platform_memory_allocator_rejects_overflowing_size() {
        setup();
        let mut allocator = PlatformMemoryAllocator::new();
        assert!(allocator.allocate(usize::MAX, 16).is_null());
        assert!(allocator.head_.is_null());
    }

    // ---- Destructor / move-only tests, observing pal_free via the PAL table ----

    // Pointers passed to pal_free while `record_frees` is active. Like the
    // shared `test_spy` stub, `recording_free` only records — it does not
    // release the memory — so the recorded addresses can never be recycled and
    // recounted; the few node-sized blocks intentionally leak for the (tiny)
    // duration of the override.
    static FREED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    extern "C" fn recording_free(ptr: *mut core::ffi::c_void) {
        FREED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(ptr as usize);
    }

    // Walks the private linked list and returns the raw node addresses (the
    // exact pointers `reset`/the destructor must hand to pal_free).
    fn node_addresses(allocator: &PlatformMemoryAllocator) -> Vec<usize> {
        let mut nodes = Vec::new();
        let mut cur = allocator.head_;
        while !cur.is_null() {
            nodes.push(cur as usize);
            cur = unsafe { (*cur).next };
        }
        nodes
    }

    // Runs `f` with pal_free swapped for `recording_free` (all other PAL
    // entries keep their current implementations), restores the previous table,
    // and returns the pointers freed while the stub was installed. The caller
    // must hold PAL_TEST_LOCK.
    fn record_frees<F: FnOnce()>(f: F) -> Vec<usize> {
        use crate::runtime::platform::platform::{PalImpl, get_pal_impl, register_pal};
        FREED.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let original = unsafe { *get_pal_impl() };
        register_pal(PalImpl::create(
            None, // init
            None, // abort
            None, // current_ticks
            None, // ticks_to_ns_multiplier
            None, // emit_log_message
            None, // allocate
            Some(recording_free),
            c"platform_memory_allocator.rs".as_ptr(),
        ));
        f();
        register_pal(original);
        FREED.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    // The destructor (C++ `~PlatformMemoryAllocator`, Rust `Drop`) runs `reset`:
    // every node still on the list when the allocator is destroyed is handed to
    // pal_free exactly once.
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.platform-memory-allocator-fn/test]
    #[test]
    fn platform_memory_allocator_drop_frees_every_node() {
        setup();
        let _lock = crate::runtime::platform::platform::test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let mut allocator = PlatformMemoryAllocator::new();
        for _ in 0..3 {
            assert!(!allocator.allocate(16, 8).is_null());
        }
        let nodes = node_addresses(&allocator);
        assert_eq!(nodes.len(), 3);

        let freed = record_frees(move || drop(allocator));
        for node in &nodes {
            assert_eq!(
                freed.iter().filter(|&&p| p == *node).count(),
                1,
                "node {node:#x} must be freed exactly once by the destructor"
            );
        }
    }

    // C++ deletes copy ctor, move ctor, and both assignment operators: the
    // allocator's raw PAL list must have exactly one owner. The Rust analog is
    // a move-only value with no Clone: moving it transfers the intact list to
    // the new binding (the moved-from binding is statically unusable), and the
    // single final drop frees each node exactly once — never twice through a
    // duplicated head_.
    // [spec:et:sem:platform-memory-allocator.executorch.et-runtime-namespace.internal.platform-memory-allocator.operator-fn/test]
    #[test]
    fn platform_memory_allocator_move_only_single_release() {
        setup();
        let _lock = crate::runtime::platform::platform::test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let mut allocator = PlatformMemoryAllocator::new();
        assert!(!allocator.allocate(32, 16).is_null());
        assert!(!allocator.allocate(8, 4).is_null());
        let nodes = node_addresses(&allocator);
        assert_eq!(nodes.len(), 2);

        // Move; `allocator` is now statically unusable (the deleted
        // copy/assignment operators' contract) and the list travels intact.
        let moved = allocator;
        assert_eq!(node_addresses(&moved), nodes);

        let freed = record_frees(move || drop(moved));
        for node in &nodes {
            assert_eq!(
                freed.iter().filter(|&&p| p == *node).count(),
                1,
                "node {node:#x} must be freed exactly once by the sole owner"
            );
        }
    }
}
