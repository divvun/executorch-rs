//! Literal port of extension/memory_allocator/malloc_memory_allocator.h.
//!
//! C++ `MallocMemoryAllocator : public MemoryAllocator` overrides `allocate()`
//! (malloc-backed) and `reset()` (frees the tracked pointers) via virtual
//! dispatch, constructing the base with `MemoryAllocator(0, nullptr)`.
//!
//! PORT-NOTE: the virtual surface of `MemoryAllocator` is ported as the
//! `MemoryAllocatorBase` trait (runtime/core/memory_allocator.rs). This type is
//! a distinct struct implementing that trait — mirroring the C++ subclass — so
//! callers holding a `Box<dyn MemoryAllocatorBase>` / `*mut dyn
//! MemoryAllocatorBase` base pointer dispatch to these malloc-backed
//! `allocate`/`reset` overrides, exactly as the C++ base-pointer callers do.
//! The non-overridden virtual accessors (`base_address`/`size`/`used_size`/
//! `free_size`) inherit the C++ base behavior over a zero-capacity base, so they
//! are delegated to an embedded `MemoryAllocator(0, nullptr)`.

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

// PORT-NOTE: `executorch::extension::utils::get_aligned_size`
// (extension/memory_allocator/memory_allocator_utils.h) is not yet ported to
// its own module. Ported literally here where the single caller lives; move to
// a `memory_allocator_utils.rs` once that file is assigned. Unresolved
// cross-module reference.
// Util to get alighment adjusted allocation size
fn get_aligned_size(mut size: usize, alignment: usize) -> Result<usize> {
    // The minimum alignment that malloc() is guaranteed to provide.
    const K_MALLOC_ALIGNMENT: usize = core::mem::align_of::<MaxAlign>();
    if alignment > K_MALLOC_ALIGNMENT {
        // To get higher alignments, allocate extra and then align the returned
        // pointer. This will waste an extra `alignment - 1` bytes every time, but
        // this is the only portable way to get aligned memory from the heap.
        let extra = alignment - 1;
        if extra >= usize::MAX - size {
            crate::et_log!(
                Error,
                "Malloc size overflow: size={} + extra={}",
                size,
                extra
            );
            return Err(Error::InvalidArgument);
        }
        size += extra;
    }
    Ok(size)
}

// Mirrors `alignof(std::max_align_t)`: the strictest alignment any scalar may
// require, which is what `malloc` guarantees.
#[repr(C)]
struct MaxAlign {
    _a: u128,
    _b: f64,
}

/// Dynamically allocates memory using malloc() and frees all pointers at
/// destruction time.
///
/// For systems with malloc(), this can be easier than using a fixed-sized
/// MemoryAllocator.
pub struct MallocMemoryAllocator {
    // Base class subobject. C++ constructs it as `MemoryAllocator(0, nullptr)`;
    // it backs the non-overridden virtual accessors and `prof_id()`.
    base_: MemoryAllocator,
    mem_ptrs_: Vec<*mut core::ffi::c_void>,
}

impl MallocMemoryAllocator {
    /// Construct a new Malloc memory allocator.
    #[must_use]
    pub fn new() -> Self {
        MallocMemoryAllocator {
            base_: MemoryAllocator::new(0, core::ptr::null_mut()),
            mem_ptrs_: Vec::new(),
        }
    }
}

impl MemoryAllocatorBase for MallocMemoryAllocator {
    /// Allocates 'size' bytes of memory, returning a pointer to the allocated
    /// region, or nullptr upon failure. The size will be rounded up based on the
    /// memory alignment size.
    fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
        if !MemoryAllocator::is_power_of2(alignment) {
            crate::et_log!(Error, "Alignment {} is not a power of 2", alignment);
            return core::ptr::null_mut();
        }

        let adjusted_size_value = get_aligned_size(size, alignment);
        let size = match adjusted_size_value {
            Ok(v) => v,
            Err(_) => return core::ptr::null_mut(),
        };
        let mem_ptr: *mut core::ffi::c_void = unsafe { libc::malloc(size) };
        if mem_ptr.is_null() {
            crate::et_log!(Error, "Malloc failed to allocate {} bytes", size);
            return core::ptr::null_mut();
        }
        self.mem_ptrs_.push(mem_ptr);
        crate::executorch_track_allocation!(self.base_.prof_id(), size);
        MemoryAllocator::align_pointer(*self.mem_ptrs_.last().unwrap(), alignment)
            as *mut core::ffi::c_void
    }

    // Free up each hosted memory pointer. The memory was created via malloc.
    fn reset(&mut self) {
        for &mem_ptr in &self.mem_ptrs_ {
            unsafe { libc::free(mem_ptr) };
        }
        self.mem_ptrs_.clear();
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
}

// `~MallocMemoryAllocator() override { reset(); }`
impl Drop for MallocMemoryAllocator {
    fn drop(&mut self) {
        self.reset();
    }
}
