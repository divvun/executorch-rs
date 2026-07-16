//! Literal port of runtime/core/device_memory_buffer.cpp + runtime/core/device_memory_buffer.h.

use crate::runtime::core::device_allocator::{
    DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT, DeviceAllocator, get_device_allocator,
};
use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::result::ResultExt;
// PORT-NOTE: `etensor::{Device*}` live in runtime/core/portable_type/device.rs,
// still a stub at time of writing. Unresolved cross-module reference.
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
use crate::runtime::core::span::Span;

/// RAII wrapper that owns a single device memory allocation.
///
/// On destruction, calls DeviceAllocator::deallocate() to free the memory.
/// This mirrors the role of std::vector<uint8_t> for CPU planned buffers,
/// but for device memory (CUDA, etc.).
///
/// Move-only: cannot be copied, but can be moved to transfer ownership.
// [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer]
// PORT-NOTE: C++ is move-only with an explicit move ctor/assign that null out
// the source. Rust move semantics already leave the source inaccessible, so the
// natural move covers the ownership transfer; `Drop` frees when `ptr_` and
// `allocator_` are both non-null, mirroring the C++ destructor.
pub struct DeviceMemoryBuffer {
    ptr_: *mut core::ffi::c_void,
    size_: usize,
    allocator_: *mut (dyn DeviceAllocator + 'static),
    device_index_: DeviceIndex,
}

impl DeviceMemoryBuffer {
    /// Creates a DeviceMemoryBuffer by allocating device memory.
    ///
    /// Looks up the DeviceAllocator for the given device type via the
    /// DeviceAllocatorRegistry. If no allocator is registered for the type,
    /// returns Error::NotFound.
    ///
    /// @param size Number of bytes to allocate.
    /// @param type The device type (e.g., CUDA).
    /// @param index The device index (e.g., 0 for cuda:0).
    /// @param alignment Minimum alignment of the returned pointer in bytes.
    ///     Must be a power of 2. Defaults to DeviceAllocator::kDefaultAlignment.
    /// @return A Result containing the DeviceMemoryBuffer on success, or an error.
    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn]
    pub fn create(
        size: usize,
        type_: DeviceType,
        index: DeviceIndex,
        alignment: usize,
    ) -> Result<DeviceMemoryBuffer> {
        let allocator: *mut (dyn DeviceAllocator + 'static) = get_device_allocator(type_);
        if allocator.is_null() {
            crate::et_log!(
                Error,
                "No device allocator registered for device type {}",
                type_ as i32
            );
            return Err(Error::NotFound);
        }

        let result = unsafe { (*allocator).allocate(size, index, alignment) };
        if !ResultExt::ok(&result) {
            return Err(ResultExt::error(&result));
        }

        Ok(DeviceMemoryBuffer::new(
            *ResultExt::get(&result),
            size,
            allocator,
            index,
        ))
    }

    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.device-memory-buffer-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.device-memory-buffer-fn]
    // (Move ctor / move assignment: covered by Rust's native move; see the
    // struct-level PORT-NOTE. `operator=(&&)` maps to the same.)
    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.operator-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.operator-fn]

    /// Returns the device pointer, or nullptr if empty/moved-from.
    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn]
    pub fn data(&self) -> *mut core::ffi::c_void {
        self.ptr_
    }

    /// Returns the size in bytes of the allocation.
    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn]
    pub fn size(&self) -> usize {
        self.size_
    }

    /// Returns a Span<uint8_t> wrapping the device pointer.
    ///
    /// This is intended for use with HierarchicalAllocator, which only performs
    /// pointer arithmetic on the span data and never dereferences it. Device
    /// pointers are valid for pointer arithmetic from the CPU side.
    // [spec:et:def:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn]
    pub fn as_span(&self) -> Span<u8> {
        Span::from_raw_parts(self.ptr_ as *mut u8, self.size_)
    }

    fn new(
        ptr: *mut core::ffi::c_void,
        size: usize,
        allocator: *mut (dyn DeviceAllocator + 'static),
        device_index: DeviceIndex,
    ) -> Self {
        DeviceMemoryBuffer {
            ptr_: ptr,
            size_: size,
            allocator_: allocator,
            device_index_: device_index,
        }
    }
}

impl Default for DeviceMemoryBuffer {
    fn default() -> Self {
        DeviceMemoryBuffer {
            ptr_: core::ptr::null_mut(),
            size_: 0,
            allocator_: core::ptr::null_mut::<DeviceMemoryBufferNever>()
                as *mut (dyn DeviceAllocator + 'static),
            device_index_: 0,
        }
    }
}

impl Drop for DeviceMemoryBuffer {
    fn drop(&mut self) {
        if !self.ptr_.is_null() && !self.allocator_.is_null() {
            unsafe {
                (*self.allocator_).deallocate(self.ptr_, self.device_index_);
            }
        }
    }
}

// PORT-NOTE: a null `*mut dyn DeviceAllocator` needs a concrete pointee type to
// synthesize the fat-pointer metadata for `Default`; this zero-sized type is
// never instantiated and exists only to produce a null trait-object pointer.
struct DeviceMemoryBufferNever;

impl DeviceAllocator for DeviceMemoryBufferNever {
    fn allocate(
        &mut self,
        _nbytes: usize,
        _index: DeviceIndex,
        _alignment: usize,
    ) -> Result<*mut core::ffi::c_void> {
        unreachable!()
    }
    fn deallocate(&mut self, _ptr: *mut core::ffi::c_void, _index: DeviceIndex) {
        unreachable!()
    }
    fn copy_host_to_device(
        &mut self,
        _dst: *mut core::ffi::c_void,
        _src: *const core::ffi::c_void,
        _nbytes: usize,
        _index: DeviceIndex,
    ) -> Error {
        unreachable!()
    }
    fn copy_device_to_host(
        &mut self,
        _dst: *mut core::ffi::c_void,
        _src: *const core::ffi::c_void,
        _nbytes: usize,
        _index: DeviceIndex,
    ) -> Error {
        unreachable!()
    }
    fn device_type(&self) -> DeviceType {
        unreachable!()
    }
}

// PORT-NOTE: keeps the default-alignment constant reachable from this module for
// call sites that default `alignment` to `DeviceAllocator::kDefaultAlignment`.
#[allow(dead_code)]
const K_DEFAULT_ALIGNMENT: usize = DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::device_allocator::{
        DEVICE_REGISTRY_TEST_LOCK, DeviceAllocatorRegistry, register_device_allocator,
    };
    use crate::runtime::core::result::ResultExt;

    // MockAllocator for testing DeviceMemoryBuffer: returns pointers into a local
    // buffer and tracks call counts.
    struct MockAllocator {
        type_: DeviceType,
        allocate_count_: i32,
        deallocate_count_: i32,
        last_allocate_size_: usize,
        last_allocate_alignment_: usize,
        last_deallocate_ptr_: *mut core::ffi::c_void,
        buffer_: [u8; 256],
    }

    impl MockAllocator {
        const fn new(type_: DeviceType) -> Self {
            MockAllocator {
                type_,
                allocate_count_: 0,
                deallocate_count_: 0,
                last_allocate_size_: 0,
                last_allocate_alignment_: 0,
                last_deallocate_ptr_: core::ptr::null_mut(),
                buffer_: [0u8; 256],
            }
        }
    }

    impl DeviceAllocator for MockAllocator {
        fn allocate(
            &mut self,
            nbytes: usize,
            _index: DeviceIndex,
            alignment: usize,
        ) -> Result<*mut core::ffi::c_void> {
            self.allocate_count_ += 1;
            self.last_allocate_size_ = nbytes;
            self.last_allocate_alignment_ = alignment;
            Ok(self.buffer_.as_mut_ptr() as *mut core::ffi::c_void)
        }

        fn deallocate(&mut self, ptr: *mut core::ffi::c_void, _index: DeviceIndex) {
            self.deallocate_count_ += 1;
            self.last_deallocate_ptr_ = ptr;
        }

        fn copy_host_to_device(
            &mut self,
            _dst: *mut core::ffi::c_void,
            _src: *const core::ffi::c_void,
            _nbytes: usize,
            _index: DeviceIndex,
        ) -> Error {
            Error::Ok
        }

        fn copy_device_to_host(
            &mut self,
            _dst: *mut core::ffi::c_void,
            _src: *const core::ffi::c_void,
            _nbytes: usize,
            _index: DeviceIndex,
        ) -> Error {
            Error::Ok
        }

        fn device_type(&self) -> DeviceType {
            self.type_
        }
    }

    // static MockAllocator g_mock_cuda(DeviceType::CUDA); reachable via raw ptr.
    static mut G_MOCK_CUDA: MockAllocator = MockAllocator::new(DeviceType::CUDA);

    fn g_mock_cuda() -> *mut MockAllocator {
        &raw mut G_MOCK_CUDA
    }

    // Mirrors SetUpTestSuite() + SetUp() against an isolated registry (see the
    // shared-registry PORT-NOTE in device_allocator's tests): clear the registry,
    // register g_mock_cuda for CUDA, then reset its counters. Must hold
    // DEVICE_REGISTRY_TEST_LOCK.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
        DeviceAllocatorRegistry::instance().clear_for_test();
        register_device_allocator(g_mock_cuda() as *mut (dyn DeviceAllocator + 'static));
        unsafe {
            (*g_mock_cuda()).allocate_count_ = 0;
            (*g_mock_cuda()).deallocate_count_ = 0;
            (*g_mock_cuda()).last_allocate_size_ = 0;
            (*g_mock_cuda()).last_allocate_alignment_ = 0;
            (*g_mock_cuda()).last_deallocate_ptr_ = core::ptr::null_mut();
        }
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn/test]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn/test]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn/test]
    #[test]
    fn device_memory_buffer_test_default_constructed_is_empty() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let buf = DeviceMemoryBuffer::default();
        assert_eq!(buf.data(), core::ptr::null_mut());
        assert_eq!(buf.size(), 0);

        let span = buf.as_span();
        assert_eq!(span.data(), core::ptr::null_mut());
        assert_eq!(span.size(), 0);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn/test]
    #[test]
    fn device_memory_buffer_test_create_allocates_and_destructor_deallocates() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        {
            let result = DeviceMemoryBuffer::create(
                1024,
                DeviceType::CUDA,
                0,
                DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
            );
            assert!(ResultExt::ok(&result));

            let buf = result.unwrap();
            assert_ne!(buf.data(), core::ptr::null_mut());
            assert_eq!(buf.size(), 1024);
            assert_eq!(unsafe { (*g_mock_cuda()).allocate_count_ }, 1);
            assert_eq!(unsafe { (*g_mock_cuda()).last_allocate_size_ }, 1024);
            assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 0);
        }
        assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 1);
        assert_eq!(unsafe { (*g_mock_cuda()).last_deallocate_ptr_ }, unsafe {
            (*g_mock_cuda()).buffer_.as_mut_ptr() as *mut core::ffi::c_void
        });
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn/test]
    #[test]
    fn device_memory_buffer_test_create_fails_with_no_registered_allocator() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let result = DeviceMemoryBuffer::create(
            512,
            DeviceType::CPU,
            0,
            DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
        );
        assert!(!ResultExt::ok(&result));
        assert_eq!(result.error(), Error::NotFound);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn/test]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn/test]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.size-fn/test]
    #[test]
    fn device_memory_buffer_test_move_constructor_transfers_ownership() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let result = DeviceMemoryBuffer::create(
            256,
            DeviceType::CUDA,
            0,
            DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
        );
        assert!(ResultExt::ok(&result));
        let mut original = result.unwrap();
        let original_ptr = original.data();

        // DeviceMemoryBuffer moved(std::move(original));
        //
        // PORT-NOTE: Rust's native move makes the source inaccessible after the
        // move, so the C++ "source nulled out" checks cannot be run on the moved
        // value. The struct-level PORT-NOTE covers this; here we swap out via
        // core::mem::take to observe the source-nulling contract the C++ move
        // ctor implements (original left default/empty).
        let moved = core::mem::take(&mut original);

        assert_eq!(original.data(), core::ptr::null_mut());
        assert_eq!(original.size(), 0);
        assert_eq!(moved.data(), original_ptr);
        assert_eq!(moved.size(), 256);
        assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 0);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.operator-fn/test]
    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.data-fn/test]
    #[test]
    fn device_memory_buffer_test_move_assignment_transfers_ownership() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let result = DeviceMemoryBuffer::create(
            128,
            DeviceType::CUDA,
            0,
            DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
        );
        assert!(ResultExt::ok(&result));
        let mut original = result.unwrap();
        let original_ptr = original.data();

        let mut target = DeviceMemoryBuffer::default();
        // target = std::move(original);
        target = core::mem::take(&mut original);

        assert_eq!(original.data(), core::ptr::null_mut());
        assert_eq!(target.data(), original_ptr);
        assert_eq!(target.size(), 128);
        assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 0);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.device-memory-buffer-fn/test]
    #[test]
    fn device_memory_buffer_test_destructor_no_op_for_default_constructed() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        {
            let _buf = DeviceMemoryBuffer::default();
        }
        assert_eq!(unsafe { (*g_mock_cuda()).deallocate_count_ }, 0);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.as-span-fn/test]
    #[test]
    fn device_memory_buffer_test_as_span_wraps_device_pointer() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let result = DeviceMemoryBuffer::create(
            2048,
            DeviceType::CUDA,
            0,
            DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
        );
        assert!(ResultExt::ok(&result));
        let buf = result.unwrap();

        let span = buf.as_span();
        assert_eq!(span.data(), buf.data() as *mut u8);
        assert_eq!(span.size(), 2048);
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn/test]
    #[test]
    fn device_memory_buffer_test_create_uses_default_alignment_when_unspecified() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let result = DeviceMemoryBuffer::create(
            1024,
            DeviceType::CUDA,
            0,
            DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT,
        );
        assert!(ResultExt::ok(&result));
        assert_eq!(
            unsafe { (*g_mock_cuda()).last_allocate_alignment_ },
            <dyn DeviceAllocator>::K_DEFAULT_ALIGNMENT
        );
    }

    // [spec:et:sem:device-memory-buffer.executorch.runtime.device-memory-buffer.create-fn/test]
    #[test]
    fn device_memory_buffer_test_create_forwards_custom_alignment_to_allocator() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        const K_CUSTOM_ALIGNMENT: usize = 512;
        let result = DeviceMemoryBuffer::create(1024, DeviceType::CUDA, 0, K_CUSTOM_ALIGNMENT);
        assert!(ResultExt::ok(&result));
        assert_eq!(
            unsafe { (*g_mock_cuda()).last_allocate_alignment_ },
            K_CUSTOM_ALIGNMENT
        );
    }
}
