//! Literal port of runtime/core/memory_allocator.h.

// PORT-NOTE: `ET_CHECK_MSG` (runtime/platform/assert.h) has no ported shared
// macro yet; this local macro mirrors its semantics (log the message, then
// abort via the PAL abort path) so this file compiles. It should be replaced by
// the shared `et_check_msg!` once the assert module is ported. Unresolved
// cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::et_log!(Fatal, $($arg)*);
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `EXECUTORCH_TRACK_ALLOCATOR` / `EXECUTORCH_TRACK_ALLOCATION`
// (runtime/platform/profiler.h). When profiling is compiled out (the default),
// these are no-ops: the tracker yields the sentinel id -1 and the allocation
// tracker discards its arguments. The profiling-enabled expansions forward to
// the profiler module.
#[cfg(feature = "profiling-enabled")]
macro_rules! executorch_track_allocator {
    ($name:expr) => {
        $crate::runtime::platform::profiler::track_allocator($name) as i32
    };
}
#[cfg(not(feature = "profiling-enabled"))]
macro_rules! executorch_track_allocator {
    ($name:expr) => {{
        let _ = $name;
        -1
    }};
}

// Exported (unlike the sibling macros above) because
// `MallocMemoryAllocator::allocate` also expands `EXECUTORCH_TRACK_ALLOCATION`.
#[cfg(feature = "profiling-enabled")]
#[macro_export]
macro_rules! executorch_track_allocation {
    ($id:expr, $size:expr) => {
        $crate::runtime::platform::profiler::track_allocation($id, $size as u32)
    };
}
#[cfg(not(feature = "profiling-enabled"))]
#[macro_export]
macro_rules! executorch_track_allocation {
    ($id:expr, $size:expr) => {{
        let _ = $id;
        let _ = $size;
    }};
}

/// The virtual surface of `MemoryAllocator`. C++ declares `allocate`,
/// `base_address`, `size`, `used_size`, `free_size`, and `reset` as `virtual`
/// so that subclasses (`MallocMemoryAllocator`, `PlatformMemoryAllocator`) can
/// override the malloc-backed / linked-list behavior and callers that hold a
/// `MemoryAllocator*` base pointer dispatch to the override. This trait mirrors
/// that vtable; the concrete `MemoryAllocator` struct implements it with the
/// bump-allocator base behavior, and the `dyn` objects appear wherever C++ held
/// a `MemoryAllocator*` base pointer (per PORTING.md's virtual-interface rule).
///
/// The non-virtual template helpers `allocate_instance<T>`/`allocate_list<T>`
/// (which call the virtual `allocate`) are not part of the vtable; they live in
/// `MemoryAllocatorExt` so they remain callable through a `dyn` base pointer,
/// mirroring the C++ base-class methods that any subclass inherits.
pub trait MemoryAllocatorBase {
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
    fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void;

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
    fn base_address(&self) -> *mut u8;

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.size-fn]
    fn size(&self) -> u32;

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
    fn used_size(&self) -> usize;

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
    fn free_size(&self) -> usize;

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
    fn reset(&mut self);
}

/// Non-virtual base-class helpers that build on the virtual `allocate`. In C++
/// these are ordinary (non-`virtual`) `MemoryAllocator` methods that every
/// subclass inherits and that internally call the virtual `allocate`. Exposed as
/// a blanket-impl extension trait so they stay callable on any `dyn
/// MemoryAllocatorBase` base pointer.
pub trait MemoryAllocatorExt: MemoryAllocatorBase {
    /// Allocates a buffer large enough for an instance of type T. Note that the
    /// memory will not be initialized.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn]
    fn allocate_instance<T>(&mut self, alignment: usize) -> *mut T {
        self.allocate(core::mem::size_of::<T>(), alignment) as *mut T
    }

    /// Allocates `size` number of chunks of type T, where each chunk is of size
    /// equal to sizeof(T) bytes.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn]
    fn allocate_list<T>(&mut self, size: usize, alignment: usize) -> *mut T {
        // Some users of this method allocate lists of pointers, causing the next
        // line to expand to `sizeof(type *)`, which triggers a clang-tidy warning.
        let bytes_size: usize;
        let overflow: bool;
        match size.checked_mul(core::mem::size_of::<T>()) {
            Some(product) => {
                bytes_size = product;
                overflow = false;
            }
            None => {
                bytes_size = 0;
                overflow = true;
            }
        }
        if overflow {
            crate::et_log!(
                Error,
                "Failed to allocate list: size({}) * sizeof(T)({}) overflowed",
                size,
                core::mem::size_of::<T>()
            );
            return core::ptr::null_mut();
        }
        self.allocate(bytes_size, alignment) as *mut T
    }
}

impl<A: MemoryAllocatorBase + ?Sized> MemoryAllocatorExt for A {}

/// A class that does simple allocation based on a size and returns the pointer
/// to the memory address. It bookmarks a buffer with certain size. The
/// allocation is simply checking space and growing the cur_ pointer with each
/// allocation request.
///
/// Simple example:
///
///   // User allocates a 100 byte long memory in the heap.
///   uint8_t* memory_pool = malloc(100 * sizeof(uint8_t));
///   MemoryAllocator allocator(100, memory_pool)
///   // Pass allocator object in the Executor
///
///   Underneath the hood, ExecuTorch will call
///   allocator.allocate() to keep iterating cur_ pointer
// [spec:et:def:memory-allocator.executorch.runtime.memory-allocator]
pub struct MemoryAllocator {
    begin_: *mut u8,
    end_: *mut u8,
    cur_: *mut u8,
    size_: u32,
    prof_id_: i32,
}

impl MemoryAllocator {
    /// Default alignment of memory returned by this class. Ensures that pointer
    /// fields of structs will be aligned. Larger types like `long double` may not
    /// be, however, depending on the toolchain and architecture.
    pub const K_DEFAULT_ALIGNMENT: usize = core::mem::align_of::<*const core::ffi::c_void>();

    /// Constructs a new memory allocator of a given `size`, starting at the
    /// provided `base_address`.
    ///
    /// @param[in] size The size in bytes of the buffer at `base_address`.
    /// @param[in] base_address The buffer to allocate from. Does not take
    ///     ownership of this buffer, so it must be valid for the lifetime of of
    ///     the MemoryAllocator.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.memory-allocator-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.memory-allocator-fn]
    pub fn new(size: u32, base_address: *mut u8) -> Self {
        let begin_ = base_address;
        let end_ = if !base_address.is_null() {
            if usize::MAX - (base_address as usize) >= size as usize {
                unsafe { base_address.add(size as usize) }
            } else {
                core::ptr::null_mut()
            }
        } else {
            core::ptr::null_mut()
        };
        let cur_ = base_address;
        let size_ = size;
        et_check_msg!(
            !base_address.is_null() || size == 0,
            "Base address is null but size={}",
            size
        );
        et_check_msg!(
            base_address.is_null()
                || size == 0
                || (usize::MAX - (base_address as usize) >= size as usize),
            "Address space overflow in allocator"
        );
        MemoryAllocator {
            begin_,
            end_,
            cur_,
            size_,
            prof_id_: -1,
        }
    }

    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn]
    pub fn enable_profiling(&mut self, name: *const core::ffi::c_char) {
        self.prof_id_ = executorch_track_allocator!(name);
    }

    /// Returns the profiler ID for this allocator.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.prof-id-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.prof-id-fn]
    pub(crate) fn prof_id(&self) -> i32 {
        self.prof_id_
    }

    /// Returns true if the value is an integer power of 2.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn]
    pub const fn is_power_of2(value: usize) -> bool {
        value != 0 && (value & (value - 1)) == 0
    }

    /// Returns the next alignment for a given pointer.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn]
    pub fn align_pointer(ptr: *mut core::ffi::c_void, alignment: usize) -> *mut u8 {
        let mut address: usize = ptr as usize;
        let mask: usize = alignment - 1;
        address = address.wrapping_add(mask) & !mask;
        address as *mut u8
    }
}

// The base-class virtual methods. C++ marks these `virtual`; the concrete
// `MemoryAllocator` provides the default bump-allocator behavior that subclasses
// (`MallocMemoryAllocator`, `PlatformMemoryAllocator`) may override.
impl MemoryAllocatorBase for MemoryAllocator {
    /// Allocates `size` bytes of memory.
    ///
    /// @param[in] size Number of bytes to allocate.
    /// @param[in] alignment Minimum alignment for the returned pointer. Must be a
    ///     power of 2.
    ///
    /// @returns Aligned pointer to the allocated memory on success.
    /// @retval nullptr Not enough memory, or `alignment` was not a power of 2.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn]
    fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
        if self.begin_.is_null() || self.end_.is_null() {
            crate::et_log!(Error, "allocate() on zero-capacity allocator");
            return core::ptr::null_mut();
        }
        if !MemoryAllocator::is_power_of2(alignment) {
            crate::et_log!(Error, "Alignment {} is not a power of 2", alignment);
            return core::ptr::null_mut();
        }

        // The allocation will occupy [start, end), where the start is the next
        // position that's a multiple of alignment.
        let start: *mut u8 =
            MemoryAllocator::align_pointer(self.cur_ as *mut core::ffi::c_void, alignment);
        let padding: usize = (start as usize) - (self.cur_ as usize);
        let available: usize = (self.end_ as usize) - (self.cur_ as usize);
        if padding > available || size > available - padding {
            crate::et_log!(
                Error,
                "Memory allocation failed: {}B requested (adjusted for alignment), {}B available",
                padding + size,
                available
            );
            return core::ptr::null_mut();
        }
        let end: *mut u8 = unsafe { start.add(size) };

        // Otherwise, record how many bytes were used, advance cur_ to the new end,
        // and then return start. Note that the number of bytes used is (end - cur_)
        // instead of (end - start) because start > cur_ if there is a misalignment
        executorch_track_allocation!(self.prof_id_, (end as usize) - (self.cur_ as usize));
        self.cur_ = end;
        start as *mut core::ffi::c_void
    }

    /// Returns the allocator memory's base address.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.base-address-fn]
    fn base_address(&self) -> *mut u8 {
        self.begin_
    }

    /// Returns the total size of the allocator's memory buffer.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.size-fn]
    fn size(&self) -> u32 {
        self.size_
    }

    /// Returns the number of bytes currently allocated from this allocator. The
    /// default implementation reports the bump cursor's offset from the base
    /// (cur_ - begin_); subclasses backed by a different allocator should override
    /// this to match their own accounting.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn]
    fn used_size(&self) -> usize {
        (self.cur_ as usize) - (self.begin_ as usize)
    }

    /// Returns the number of bytes still available for allocation, not accounting
    /// for any alignment padding a future allocation may require. The default
    /// implementation reports end_ - cur_; subclasses should override to stay
    /// consistent with used_size().
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn]
    fn free_size(&self) -> usize {
        (self.end_ as usize) - (self.cur_ as usize)
    }

    /// Resets the current pointer to the base address. It does nothing to
    /// the contents.
    // [spec:et:def:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.reset-fn]
    fn reset(&mut self) {
        self.cur_ = self.begin_;
    }
}

// PORT-NOTE: `~MemoryAllocator() = default`; no owned resources, so no `Drop`.

// Literal port of runtime/core/test/memory_allocator_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;

    // TestType8 / TestType1024: sizeof == 8 / 1024. Used to force allocation
    // failure without depending on host `sizeof(void*)`.
    #[repr(C)]
    struct TestType8 {
        _data: [u8; 8],
    }
    #[repr(C)]
    struct TestType1024 {
        _data: [u8; 1024],
    }

    // Mirrors `is_aligned()` from test/utils/alignment.h.
    fn is_aligned(ptr: *const core::ffi::c_void, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    // Mirrors the C++ fixture `SetUp()`: the PAL must be initialized before code
    // paths that call `ET_LOG`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.memory-allocator-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.reset-fn/test]
    #[test]
    fn memory_allocator_test_memory_allocator() {
        setup();
        const MEM_SIZE: usize = 16;
        let mut mem_pool = [0u8; MEM_SIZE];
        let mut allocator = MemoryAllocator::new(MEM_SIZE as u32, mem_pool.as_mut_ptr());
        assert_ne!(
            allocator.allocate(7, MemoryAllocator::K_DEFAULT_ALIGNMENT),
            core::ptr::null_mut()
        );
        assert_ne!(
            allocator.allocate(6, MemoryAllocator::K_DEFAULT_ALIGNMENT),
            core::ptr::null_mut()
        );
        assert_eq!(
            allocator.allocate(3, MemoryAllocator::K_DEFAULT_ALIGNMENT),
            core::ptr::null_mut()
        );

        allocator.reset();
        assert_eq!(
            allocator.allocate(0, MemoryAllocator::K_DEFAULT_ALIGNMENT),
            mem_pool.as_mut_ptr() as *mut core::ffi::c_void
        );
        assert_ne!(
            allocator.allocate(16, MemoryAllocator::K_DEFAULT_ALIGNMENT),
            core::ptr::null_mut()
        );
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.base-address-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.reset-fn/test]
    #[test]
    fn memory_allocator_test_used_and_free_size() {
        setup();
        const MEM_SIZE: usize = 64;
        let mut mem_pool = [0u8; MEM_SIZE];
        let mut allocator = MemoryAllocator::new(MEM_SIZE as u32, mem_pool.as_mut_ptr());

        assert_eq!(allocator.used_size(), 0);
        assert_eq!(allocator.free_size(), MEM_SIZE);

        let p1 = allocator.allocate(8, 8);
        assert_ne!(p1, core::ptr::null_mut());
        let expected_used1 = (p1 as *mut u8 as usize) + 8 - (allocator.base_address() as usize);
        assert_eq!(allocator.used_size(), expected_used1);
        assert_eq!(allocator.free_size(), MEM_SIZE - expected_used1);

        let p2 = allocator.allocate(8, 8);
        assert_ne!(p2, core::ptr::null_mut());
        let expected_used2 = (p2 as *mut u8 as usize) + 8 - (allocator.base_address() as usize);
        assert!(expected_used2 > expected_used1);
        assert_eq!(allocator.used_size(), expected_used2);
        assert_eq!(allocator.free_size(), MEM_SIZE - expected_used2);

        allocator.reset();
        assert_eq!(allocator.used_size(), 0);
        assert_eq!(allocator.free_size(), MEM_SIZE);
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn/test]
    #[test]
    fn memory_allocator_test_used_and_free_size_zero_capacity() {
        setup();
        let allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        assert_eq!(allocator.used_size(), 0);
        assert_eq!(allocator.free_size(), 0);
    }

    // Overrides the accessors with sentinel values to prove base-reference calls
    // dispatch virtually to the override. Mirrors the C++
    // `SentinelAccessorAllocator` subclass; in Rust the "base pointer" is a
    // `&mut dyn MemoryAllocatorBase`.
    struct SentinelAccessorAllocator {
        inner: MemoryAllocator,
    }
    impl MemoryAllocatorBase for SentinelAccessorAllocator {
        fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
            self.inner.allocate(size, alignment)
        }
        fn base_address(&self) -> *mut u8 {
            self.inner.base_address()
        }
        fn size(&self) -> u32 {
            self.inner.size()
        }
        fn used_size(&self) -> usize {
            111
        }
        fn free_size(&self) -> usize {
            222
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn/test]
    #[test]
    fn memory_allocator_test_used_and_free_size_dispatch_virtually() {
        setup();
        let mut mem_pool = [0u8; 16];
        let mut derived = SentinelAccessorAllocator {
            inner: MemoryAllocator::new(16, mem_pool.as_mut_ptr()),
        };
        let base: &mut dyn MemoryAllocatorBase = &mut derived;
        assert_eq!(base.used_size(), 111);
        assert_eq!(base.free_size(), 222);
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.used-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.free-size-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    // also verifies size(): the final assertion pins free_size to
    // `allocator.size() - expected_used`, so a wrong size() reading fails here.
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.size-fn/test]
    #[test]
    fn memory_allocator_test_used_and_free_size_across_alignment_padding() {
        setup();
        const MEM_SIZE: usize = 128;
        let mut mem_pool = [0u8; MEM_SIZE];
        let mut allocator = MemoryAllocator::new(MEM_SIZE as u32, mem_pool.as_mut_ptr());

        // 1-byte block, then a 16-aligned block: 1 + 15 padding + 16 = 32 used.
        assert_ne!(allocator.allocate(1, 1), core::ptr::null_mut());
        let p2 = allocator.allocate(16, 16);
        assert_ne!(p2, core::ptr::null_mut());

        let expected_used = (p2 as *mut u8 as usize) + 16 - (allocator.base_address() as usize);
        assert_eq!(allocator.used_size(), expected_used);
        assert_eq!(
            allocator.free_size(),
            allocator.size() as usize - expected_used
        );
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.align-pointer-fn/test]
    #[test]
    fn memory_allocator_test_memory_allocator_alignment() {
        setup();
        const ARR_SIZE: usize = 6;
        let allocation: [usize; ARR_SIZE] = [7, 6, 3, 76, 4, 1];
        let alignment: [usize; ARR_SIZE] = [
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
            4,
            32,
            128,
            2,
        ];

        for i in 0..ARR_SIZE {
            let align_size = alignment[i];
            const MEM_SIZE: usize = 1000;
            let mut mem_pool = [0u8; MEM_SIZE];
            let mut allocator = MemoryAllocator::new(MEM_SIZE as u32, mem_pool.as_mut_ptr());
            for j in 0..ARR_SIZE {
                let size = allocation[j];
                let start = allocator.allocate(size, align_size);
                assert!(is_aligned(start, align_size));
            }
        }
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.is-power-of2-fn/test]
    #[test]
    fn memory_allocator_test_memory_allocator_non_power_of_two_alignment() {
        setup();
        const MEM_SIZE: usize = 128;
        let mut mem_pool = [0u8; MEM_SIZE];
        let mut allocator = MemoryAllocator::new(MEM_SIZE as u32, mem_pool.as_mut_ptr());

        let alignment: [usize; 5] = [0, 5, 6, 12, 34];
        for i in 0..5 {
            assert_eq!(allocator.allocate(8, alignment[i]), core::ptr::null_mut());
        }
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    #[test]
    fn memory_allocator_test_memory_allocator_too_large_fail_but_succeed_afterwards() {
        setup();
        const K_POOL_SIZE: usize = 10;
        let mut mem_pool = [0u8; K_POOL_SIZE];
        let mut allocator = MemoryAllocator::new(K_POOL_SIZE as u32, mem_pool.as_mut_ptr());
        // Align to 1 byte so the entire pool is used.
        assert_eq!(
            allocator.allocate(K_POOL_SIZE + 2, 1),
            core::ptr::null_mut()
        );
        assert_ne!(
            allocator.allocate(K_POOL_SIZE - 1, 1),
            core::ptr::null_mut()
        );
    }

    // Mirrors the C++ `test_allocate_instance<T>` template helper.
    fn test_allocate_instance<T>() {
        let mut buffer = [0u8; 256];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        // Default alignment
        let p = allocator.allocate_instance::<T>(MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_ne!(p, core::ptr::null_mut());
        assert!(is_aligned(
            p as *const core::ffi::c_void,
            core::mem::align_of::<T>()
        ));
        unsafe { core::ptr::write_bytes(p as *mut u8, 0x55, core::mem::size_of::<T>()) };

        // Override alignment
        const K_HIGHER_ALIGNMENT: usize = 64;
        assert!(K_HIGHER_ALIGNMENT > core::mem::align_of::<T>());
        let p = allocator.allocate_instance::<T>(K_HIGHER_ALIGNMENT);
        assert_ne!(p, core::ptr::null_mut());
        assert!(is_aligned(
            p as *const core::ffi::c_void,
            K_HIGHER_ALIGNMENT
        ));
        unsafe { core::ptr::write_bytes(p as *mut u8, 0x55, core::mem::size_of::<T>()) };
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn/test]
    #[test]
    fn memory_allocator_test_allocate_instance() {
        setup();
        test_allocate_instance::<u8>();
        test_allocate_instance::<u16>();
        test_allocate_instance::<u32>();
        test_allocate_instance::<u64>();
        test_allocate_instance::<*mut core::ffi::c_void>();

        #[repr(C)]
        struct StructWithPointer {
            _p: *mut core::ffi::c_void,
            _i: i32,
        }
        test_allocate_instance::<StructWithPointer>();

        // std::max_align_t maps to the platform max-alignment type; u128 matches
        // its alignment on the supported targets.
        #[repr(C)]
        struct StructWithLargestType {
            _max: u128,
            _i: i32,
        }
        test_allocate_instance::<StructWithLargestType>();
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn/test]
    #[test]
    fn memory_allocator_test_allocate_instance_failure() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        // Allocate more memory than the allocator provides, which should fail.
        let p = allocator.allocate_instance::<TestType1024>(MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(p, core::ptr::null_mut());
    }

    // Mirrors the C++ `test_allocate_list<T>` template helper.
    fn test_allocate_list<T>() {
        let mut buffer = [0u8; 256];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        // Default alignment
        const K_NUM_ELEM: usize = 5;
        let p = allocator.allocate_list::<T>(K_NUM_ELEM, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_ne!(p, core::ptr::null_mut());
        assert!(is_aligned(
            p as *const core::ffi::c_void,
            core::mem::align_of::<T>()
        ));
        unsafe {
            core::ptr::write_bytes(p as *mut u8, 0x55, K_NUM_ELEM * core::mem::size_of::<T>())
        };

        // Override alignment
        const K_HIGHER_ALIGNMENT: usize = 64;
        assert!(K_HIGHER_ALIGNMENT > core::mem::align_of::<T>());
        let p = allocator.allocate_list::<T>(K_NUM_ELEM, K_HIGHER_ALIGNMENT);
        assert_ne!(p, core::ptr::null_mut());
        assert!(is_aligned(
            p as *const core::ffi::c_void,
            K_HIGHER_ALIGNMENT
        ));
        unsafe {
            core::ptr::write_bytes(p as *mut u8, 0x55, K_NUM_ELEM * core::mem::size_of::<T>())
        };
    }

    // PORT-NOTE: the C++ `AllocateList` test's body calls `test_allocate_instance`
    // (not `test_allocate_list`) for the two struct types — an apparent copy-paste
    // in the C++ source. Reproduced bug-for-bug.
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn/test]
    #[test]
    fn memory_allocator_test_allocate_list() {
        setup();
        test_allocate_list::<u8>();
        test_allocate_list::<u16>();
        test_allocate_list::<u32>();
        test_allocate_list::<u64>();
        test_allocate_list::<*mut core::ffi::c_void>();

        #[repr(C)]
        struct StructWithPointer {
            _p: *mut core::ffi::c_void,
            _c: u8,
        }
        test_allocate_instance::<StructWithPointer>();

        #[repr(C)]
        struct StructWithLargestType {
            _max: u128,
            _i: i32,
        }
        test_allocate_instance::<StructWithLargestType>();
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn/test]
    #[test]
    fn memory_allocator_test_allocate_list_failure() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        // Allocate more memory than the allocator provides, which should fail.
        let p = allocator.allocate_list::<TestType8>(10, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(p, core::ptr::null_mut());
    }

    // ---- HelperMacrosTest fixture ----

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_success() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate(
            allocator.size() as usize / 2,
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
        );
        assert_ne!(p, core::ptr::null_mut());
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_failure() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate(
            allocator.size() as usize * 2,
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
        );
        assert_eq!(p, core::ptr::null_mut());
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_instance_success() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate_instance::<TestType8>(MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_ne!(p, core::ptr::null_mut());
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-instance-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_instance_failure() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate_instance::<TestType1024>(MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(p, core::ptr::null_mut());
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_list_success() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate_list::<u8>(
            allocator.size() as usize / 2,
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
        );
        assert_ne!(p, core::ptr::null_mut());
    }

    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-list-fn/test]
    #[test]
    fn helper_macros_test_try_allocate_list_failure() {
        setup();
        let mut buffer = [0u8; 16];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());

        let p = allocator.allocate_list::<u8>(
            allocator.size() as usize * 2,
            MemoryAllocator::K_DEFAULT_ALIGNMENT,
        );
        assert_eq!(p, core::ptr::null_mut());
    }

    // No corresponding C++ test: profiling is compiled out by default, so
    // `enable_profiling` forwards the name to the no-op `EXECUTORCH_TRACK_ALLOCATOR`
    // expansion (sentinel id -1) and `prof_id` reports it. A fresh allocator also
    // starts with prof_id_ == -1.
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.enable-profiling-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.prof-id-fn/test]
    #[cfg(not(feature = "profiling-enabled"))]
    #[test]
    fn memory_allocator_test_enable_profiling_default_sentinel() {
        setup();
        let mut mem_pool = [0u8; 16];
        let mut allocator = MemoryAllocator::new(mem_pool.len() as u32, mem_pool.as_mut_ptr());
        assert_eq!(allocator.prof_id(), -1);
        let name = c"my_allocator";
        allocator.enable_profiling(name.as_ptr());
        assert_eq!(allocator.prof_id(), -1);
    }
}
