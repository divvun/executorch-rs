//! Literal port of runtime/core/device_allocator.cpp + runtime/core/device_allocator.h.

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::memory_allocator::MemoryAllocator;
// PORT-NOTE: `etensor::{Device*}` live in runtime/core/portable_type/device.rs,
// still a stub at time of writing. Unresolved cross-module reference.
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType, K_NUM_DEVICE_TYPES};

// PORT-NOTE: `ET_CHECK_MSG` (runtime/platform/assert.h) has no ported shared
// macro yet; this local macro mirrors its semantics (log the message, then
// abort via the PAL abort path). Should be replaced by the shared
// `et_check_msg!` once the assert module is ported. Unresolved cross-module
// reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::et_log!(Fatal, $($arg)*);
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
}

/// Abstract interface for device-specific memory allocation.
///
/// Each device type (CUDA, etc.) provides a concrete implementation
/// that handles memory allocation on that device. Implementations are
/// expected to be singletons with static lifetime, registered via
/// DeviceAllocatorRegistry.
// [spec:et:def:device-allocator.executorch.runtime.device-allocator]
// PORT-NOTE: the abstract class with pure-virtual methods maps to a trait; the
// virtual destructor is subsumed by Rust's `Drop`. `kDefaultAlignment` is an
// associated const on the trait, mirroring the static constexpr member.
pub trait DeviceAllocator {
    /// Default alignment of memory returned by allocate(). Reuses
    /// MemoryAllocator::kDefaultAlignment so host- and device-side allocations
    /// share the same baseline contract.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.device-allocator-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-allocator-fn]
    // (virtual destructor: modeled by Drop on concrete implementors.)

    /// Allocate device memory.
    ///
    /// @param nbytes Number of bytes to allocate.
    /// @param index The device index.
    /// @param alignment Minimum alignment of the returned pointer in bytes.
    ///     Must be a power of 2. Defaults to kDefaultAlignment.
    /// @return A Result containing the device pointer on success, or an error.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.allocate-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]
    fn allocate(
        &mut self,
        nbytes: usize,
        index: DeviceIndex,
        alignment: usize,
    ) -> Result<*mut core::ffi::c_void>;

    /// Deallocate device memory previously allocated via allocate().
    ///
    /// @param ptr Pointer to the memory to deallocate.
    /// @param index The device index.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.deallocate-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.deallocate-fn]
    fn deallocate(&mut self, ptr: *mut core::ffi::c_void, index: DeviceIndex);

    /// Copy data from host memory to device memory.
    ///
    /// @param dst Destination pointer (device memory).
    /// @param src Source pointer (host memory).
    /// @param nbytes Number of bytes to copy.
    /// @param index The device index.
    /// @return Error::Ok on success, or an appropriate error code on failure.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn]
    fn copy_host_to_device(
        &mut self,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        nbytes: usize,
        index: DeviceIndex,
    ) -> Error;

    /// Copy data from device memory to host memory.
    ///
    /// @param dst Destination pointer (host memory).
    /// @param src Source pointer (device memory).
    /// @param nbytes Number of bytes to copy.
    /// @param index The device index.
    /// @return Error::Ok on success, or an appropriate error code on failure.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn]
    fn copy_device_to_host(
        &mut self,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        nbytes: usize,
        index: DeviceIndex,
    ) -> Error;

    /// Returns the device type this allocator handles.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator.device-type-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-type-fn]
    fn device_type(&self) -> DeviceType;
}

impl dyn DeviceAllocator {
    /// Default alignment of memory returned by allocate(). Reuses
    /// MemoryAllocator::kDefaultAlignment so host- and device-side allocations
    /// share the same baseline contract.
    pub const K_DEFAULT_ALIGNMENT: usize = MemoryAllocator::K_DEFAULT_ALIGNMENT;
}

/// Standalone mirror of `DeviceAllocator::kDefaultAlignment` for use at call
/// sites that reference it as a plain constant (e.g. default arguments).
pub const DEVICE_ALLOCATOR_K_DEFAULT_ALIGNMENT: usize = MemoryAllocator::K_DEFAULT_ALIGNMENT;

/// Registry for device allocators.
///
/// Provides a global mapping from DeviceType to DeviceAllocator instances.
/// Device allocators register themselves at static initialization time,
/// and the runtime queries the registry to find the appropriate allocator
/// for a given device type.
// [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry]
pub struct DeviceAllocatorRegistry {
    // Fixed-size array indexed by device type. This avoids dynamic allocation
    // and is suitable for embedded environments.
    allocators_: [*mut (dyn DeviceAllocator + 'static); K_NUM_DEVICE_TYPES],
}

// PORT-NOTE: a `*mut dyn DeviceAllocator` is a fat pointer and cannot be built
// with `core::ptr::null_mut()` (which requires a thin pointee). This helper
// produces a null trait-object pointer by coercing a null thin pointer of a
// never-instantiated concrete implementor, giving the C++ `nullptr` slot value.
struct NullDeviceAllocator;
impl DeviceAllocator for NullDeviceAllocator {
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

const NULL_DEVICE_ALLOCATOR: *mut (dyn DeviceAllocator + 'static) =
    core::ptr::null_mut::<NullDeviceAllocator>() as *mut (dyn DeviceAllocator + 'static);

// PORT-NOTE: the singleton holds raw `*mut dyn DeviceAllocator` pointers to
// static-lifetime allocators; the registry does not own them. The C++ singleton
// is a function-local static with thread-safe lazy init (`instance()`); mapped
// to a process-wide `static mut` guarded by first-call initialization, matching
// the C++ "register during single-threaded static init, read concurrently"
// contract. The registry is neither cloned nor moved.
static mut REGISTRY: DeviceAllocatorRegistry = DeviceAllocatorRegistry {
    allocators_: [NULL_DEVICE_ALLOCATOR; K_NUM_DEVICE_TYPES],
};

impl DeviceAllocatorRegistry {
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn]
    // (Private default ctor: the all-null array is the static initializer above;
    // the singleton is created only by `instance()`.)

    // [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.operator-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.operator-fn]
    // (Deleted copy/move: non-concern in Rust — the singleton is never cloned or
    // moved out of its global.)

    /// Returns the singleton instance of the registry.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]
    pub fn instance() -> &'static mut DeviceAllocatorRegistry {
        unsafe { &mut *(&raw mut REGISTRY) }
    }

    /// Register an allocator. The device type is taken from
    /// alloc->device_type(). Each device type may only be registered once;
    /// attempting to register a second allocator for the same device type
    /// will abort.
    ///
    /// Not thread-safe. Expected to be called during static initialization.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]
    pub fn register_allocator(&mut self, alloc: *mut (dyn DeviceAllocator + 'static)) {
        et_check_msg!(!alloc.is_null(), "Cannot register a null allocator");
        let type_ = unsafe { (*alloc).device_type() };
        let index = type_ as usize;
        et_check_msg!(
            index < K_NUM_DEVICE_TYPES,
            "Invalid device type: {}",
            type_ as i32
        );
        et_check_msg!(
            self.allocators_[index].is_null(),
            "Allocator already registered for device type: {}",
            type_ as i32
        );
        self.allocators_[index] = alloc;
    }

    /// Test-only: clears every registered slot so an in-process test suite can
    /// re-register a fresh allocator. The C++ has no such API because each gtest
    /// binary starts with a fresh static registry; the Rust test binary shares
    /// one process-wide `REGISTRY` across all suites, so registry-dependent
    /// suites reset it (under `DEVICE_REGISTRY_TEST_LOCK`) to stay isolated.
    #[cfg(test)]
    pub(crate) fn clear_for_test(&mut self) {
        self.allocators_ = [NULL_DEVICE_ALLOCATOR; K_NUM_DEVICE_TYPES];
    }

    /// Test-only: atomically take `DEVICE_REGISTRY_TEST_LOCK`, clear every
    /// slot, and register `alloc`, returning the guard. Making the lock
    /// inseparable from the mutation is what keeps suites race-free — a bare
    /// `setup()` can no longer clear+register unguarded (the cause of the
    /// "Allocator already registered" SIGABRT flake under parallel tests).
    /// Poison-tolerant so one failed test doesn't cascade. Test suites MUST
    /// install their mock through this and hold the guard for the test body.
    #[cfg(test)]
    pub(crate) fn install_for_test(
        alloc: *mut (dyn DeviceAllocator + 'static),
    ) -> std::sync::MutexGuard<'static, ()> {
        let guard = DEVICE_REGISTRY_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let registry = Self::instance();
        registry.clear_for_test();
        registry.register_allocator(alloc);
        guard
    }

    /// Get the allocator for a specific device type.
    ///
    /// Safe to call concurrently with other get_allocator() calls.
    ///
    /// @param type The device type.
    /// @return Pointer to the allocator, or nullptr if not registered.
    // [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn]
    pub fn get_allocator(&mut self, type_: DeviceType) -> *mut (dyn DeviceAllocator + 'static) {
        let index = type_ as usize;
        if index >= K_NUM_DEVICE_TYPES {
            return NULL_DEVICE_ALLOCATOR;
        }
        self.allocators_[index]
    }
}

// Convenience free functions

/// Register a device allocator. The device type is taken from
/// alloc->device_type(). See DeviceAllocatorRegistry::register_allocator()
/// for the threading contract.
// [spec:et:def:device-allocator.executorch.runtime.register-device-allocator-fn]
// [spec:et:sem:device-allocator.executorch.runtime.register-device-allocator-fn]
pub fn register_device_allocator(alloc: *mut (dyn DeviceAllocator + 'static)) {
    DeviceAllocatorRegistry::instance().register_allocator(alloc);
}

/// Get the device allocator for a specific device type.
///
/// @param type The device type.
/// @return Pointer to the allocator, or nullptr if not registered.
// [spec:et:def:device-allocator.executorch.runtime.get-device-allocator-fn]
// [spec:et:sem:device-allocator.executorch.runtime.get-device-allocator-fn]
pub fn get_device_allocator(type_: DeviceType) -> *mut (dyn DeviceAllocator + 'static) {
    DeviceAllocatorRegistry::instance().get_allocator(type_)
}

/// Test-only serialization lock shared by every registry-dependent test suite
/// (device_allocator + device_memory_buffer). The process-wide `REGISTRY` is a
/// shared mutable global with no reset in the C++; holding this lock lets a suite
/// clear + re-register the CUDA slot and read mock counters without racing other
/// suites that also touch the registry.
#[cfg(test)]
pub(crate) static DEVICE_REGISTRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::result::ResultExt;

    // PORT-NOTE: the C++ `DeviceAllocatorTest` fixture registers a single static
    // MockDeviceAllocator for the whole suite and relies on gtest running the
    // suite serially, resetting counters in SetUp(). Rust's test harness runs
    // tests in parallel and shares the process (and the process-wide registry),
    // so the counter-based assertions would race — and the sibling
    // device_memory_buffer suite also registers a CUDA allocator into the same
    // slot. `DEVICE_REGISTRY_TEST_LOCK` serializes all registry-dependent tests,
    // and `setup()` clears the registry then registers this suite's mock afresh,
    // mirroring SetUpTestSuite()/SetUp() against an isolated registry.

    // MockDeviceAllocator: tracks calls to verify the registry dispatches
    // correctly. Counters are mutated through the `&mut self` trait methods; the
    // single instance lives in a static reachable via a raw pointer, matching the
    // C++ static-lifetime allocator.
    struct MockDeviceAllocator {
        type_: DeviceType,
        dummy_buffer_: [u8; 64],

        last_allocate_size_: usize,
        last_allocate_index_: DeviceIndex,
        last_allocate_alignment_: usize,
        allocate_call_count_: i32,

        last_deallocate_ptr_: *mut core::ffi::c_void,
        last_deallocate_index_: DeviceIndex,
        deallocate_call_count_: i32,

        last_h2d_dst_: *mut core::ffi::c_void,
        last_h2d_src_: *const core::ffi::c_void,
        last_h2d_size_: usize,
        last_h2d_index_: DeviceIndex,
        copy_h2d_call_count_: i32,

        last_d2h_dst_: *mut core::ffi::c_void,
        last_d2h_src_: *const core::ffi::c_void,
        last_d2h_size_: usize,
        last_d2h_index_: DeviceIndex,
        copy_d2h_call_count_: i32,
    }

    impl MockDeviceAllocator {
        const fn new(type_: DeviceType) -> Self {
            MockDeviceAllocator {
                type_,
                dummy_buffer_: [0u8; 64],
                last_allocate_size_: 0,
                last_allocate_index_: -1,
                last_allocate_alignment_: 0,
                allocate_call_count_: 0,
                last_deallocate_ptr_: core::ptr::null_mut(),
                last_deallocate_index_: -1,
                deallocate_call_count_: 0,
                last_h2d_dst_: core::ptr::null_mut(),
                last_h2d_src_: core::ptr::null(),
                last_h2d_size_: 0,
                last_h2d_index_: -1,
                copy_h2d_call_count_: 0,
                last_d2h_dst_: core::ptr::null_mut(),
                last_d2h_src_: core::ptr::null(),
                last_d2h_size_: 0,
                last_d2h_index_: -1,
                copy_d2h_call_count_: 0,
            }
        }

        fn reset_counters(&mut self) {
            self.last_allocate_size_ = 0;
            self.last_allocate_index_ = -1;
            self.last_allocate_alignment_ = 0;
            self.allocate_call_count_ = 0;

            self.last_deallocate_ptr_ = core::ptr::null_mut();
            self.last_deallocate_index_ = -1;
            self.deallocate_call_count_ = 0;

            self.last_h2d_dst_ = core::ptr::null_mut();
            self.last_h2d_src_ = core::ptr::null();
            self.last_h2d_size_ = 0;
            self.last_h2d_index_ = -1;
            self.copy_h2d_call_count_ = 0;

            self.last_d2h_dst_ = core::ptr::null_mut();
            self.last_d2h_src_ = core::ptr::null();
            self.last_d2h_size_ = 0;
            self.last_d2h_index_ = -1;
            self.copy_d2h_call_count_ = 0;
        }
    }

    impl DeviceAllocator for MockDeviceAllocator {
        fn allocate(
            &mut self,
            nbytes: usize,
            index: DeviceIndex,
            alignment: usize,
        ) -> Result<*mut core::ffi::c_void> {
            self.last_allocate_size_ = nbytes;
            self.last_allocate_index_ = index;
            self.last_allocate_alignment_ = alignment;
            self.allocate_call_count_ += 1;
            Ok(self.dummy_buffer_.as_mut_ptr() as *mut core::ffi::c_void)
        }

        fn deallocate(&mut self, ptr: *mut core::ffi::c_void, index: DeviceIndex) {
            self.last_deallocate_ptr_ = ptr;
            self.last_deallocate_index_ = index;
            self.deallocate_call_count_ += 1;
        }

        fn copy_host_to_device(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            index: DeviceIndex,
        ) -> Error {
            self.last_h2d_dst_ = dst;
            self.last_h2d_src_ = src;
            self.last_h2d_size_ = nbytes;
            self.last_h2d_index_ = index;
            self.copy_h2d_call_count_ += 1;
            Error::Ok
        }

        fn copy_device_to_host(
            &mut self,
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            nbytes: usize,
            index: DeviceIndex,
        ) -> Error {
            self.last_d2h_dst_ = dst;
            self.last_d2h_src_ = src;
            self.last_d2h_size_ = nbytes;
            self.last_d2h_index_ = index;
            self.copy_d2h_call_count_ += 1;
            Error::Ok
        }

        fn device_type(&self) -> DeviceType {
            self.type_
        }
    }

    // static MockDeviceAllocator allocator(DeviceType::CUDA); (fixture's
    // cuda_allocator()). Reachable via a raw pointer as the registry stores one.
    static mut CUDA_ALLOCATOR: MockDeviceAllocator = MockDeviceAllocator::new(DeviceType::CUDA);

    fn cuda_allocator() -> *mut MockDeviceAllocator {
        &raw mut CUDA_ALLOCATOR
    }

    // Mirrors SetUpTestSuite() + SetUp() against an isolated registry: clear the
    // shared registry, register this suite's mock, then reset counters. Must be
    // called while holding DEVICE_REGISTRY_TEST_LOCK.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
        DeviceAllocatorRegistry::instance().clear_for_test();
        register_device_allocator(cuda_allocator() as *mut (dyn DeviceAllocator + 'static));
        unsafe { (*cuda_allocator()).reset_counters() };
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn/test]
    // [spec:et:sem:device-allocator.executorch.runtime.get-device-allocator-fn/test]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-type-fn/test]
    // also verifies register_device_allocator (setup() registers via the free
    // fn; the assertions below fail if registration did not store the allocator)
    // [spec:et:sem:device-allocator.executorch.runtime.register-device-allocator-fn/test]
    #[test]
    fn device_allocator_test_registered_allocator_reports_correct_device_type() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let alloc = get_device_allocator(DeviceType::CUDA);
        assert!(!alloc.is_null());
        assert_eq!(
            alloc as *const () as *const u8,
            cuda_allocator() as *const u8
        );
        assert_eq!(unsafe { (*alloc).device_type() }, DeviceType::CUDA);
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn/test]
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.deallocate-fn/test]
    #[test]
    fn device_allocator_test_allocate_and_deallocate() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let alloc = get_device_allocator(DeviceType::CUDA);
        assert!(!alloc.is_null());

        let result =
            unsafe { (*alloc).allocate(512, 0, <dyn DeviceAllocator>::K_DEFAULT_ALIGNMENT) };
        assert!(ResultExt::ok(&result));
        let ptr = *ResultExt::get(&result);
        assert!(!ptr.is_null());
        assert_eq!(unsafe { (*cuda_allocator()).allocate_call_count_ }, 1);
        assert_eq!(unsafe { (*cuda_allocator()).last_allocate_size_ }, 512);
        assert_eq!(unsafe { (*cuda_allocator()).last_allocate_index_ }, 0);

        unsafe { (*alloc).deallocate(ptr, 0) };
        assert_eq!(unsafe { (*cuda_allocator()).deallocate_call_count_ }, 1);
        assert_eq!(unsafe { (*cuda_allocator()).last_deallocate_ptr_ }, ptr);
        assert_eq!(unsafe { (*cuda_allocator()).last_deallocate_index_ }, 0);
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn/test]
    #[test]
    fn device_allocator_test_copy_host_to_device() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let alloc = get_device_allocator(DeviceType::CUDA);
        assert!(!alloc.is_null());

        let mut host_data: [u8; 64] = [0; 64];
        host_data[0] = 1;
        host_data[1] = 2;
        host_data[2] = 3;
        host_data[3] = 4;
        let mut device_data: [u8; 64] = [0; 64];

        let err = unsafe {
            (*alloc).copy_host_to_device(
                device_data.as_mut_ptr() as *mut core::ffi::c_void,
                host_data.as_ptr() as *const core::ffi::c_void,
                core::mem::size_of_val(&host_data),
                0,
            )
        };

        assert_eq!(err, Error::Ok);
        assert_eq!(unsafe { (*cuda_allocator()).copy_h2d_call_count_ }, 1);
        assert_eq!(
            unsafe { (*cuda_allocator()).last_h2d_dst_ },
            device_data.as_mut_ptr() as *mut core::ffi::c_void
        );
        assert_eq!(
            unsafe { (*cuda_allocator()).last_h2d_src_ },
            host_data.as_ptr() as *const core::ffi::c_void
        );
        assert_eq!(
            unsafe { (*cuda_allocator()).last_h2d_size_ },
            core::mem::size_of_val(&host_data)
        );
        assert_eq!(unsafe { (*cuda_allocator()).last_h2d_index_ }, 0);
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn/test]
    #[test]
    fn device_allocator_test_copy_device_to_host() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let alloc = get_device_allocator(DeviceType::CUDA);
        assert!(!alloc.is_null());

        let mut device_data: [u8; 64] = [0; 64];
        device_data[0] = 5;
        device_data[1] = 6;
        device_data[2] = 7;
        device_data[3] = 8;
        let mut host_data: [u8; 64] = [0; 64];

        let err = unsafe {
            (*alloc).copy_device_to_host(
                host_data.as_mut_ptr() as *mut core::ffi::c_void,
                device_data.as_ptr() as *const core::ffi::c_void,
                core::mem::size_of_val(&device_data),
                1,
            )
        };

        assert_eq!(err, Error::Ok);
        assert_eq!(unsafe { (*cuda_allocator()).copy_d2h_call_count_ }, 1);
        assert_eq!(
            unsafe { (*cuda_allocator()).last_d2h_dst_ },
            host_data.as_mut_ptr() as *mut core::ffi::c_void
        );
        assert_eq!(
            unsafe { (*cuda_allocator()).last_d2h_src_ },
            device_data.as_ptr() as *const core::ffi::c_void
        );
        assert_eq!(
            unsafe { (*cuda_allocator()).last_d2h_size_ },
            core::mem::size_of_val(&device_data)
        );
        assert_eq!(unsafe { (*cuda_allocator()).last_d2h_index_ }, 1);
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn/test]
    // also verifies the default-constructed registry state: the private ctor
    // (all-null allocators_ array) leaves every unregistered slot null, so an
    // unregistered CPU lookup returns nullptr.
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn/test]
    #[test]
    fn device_allocator_test_registry_get_unregistered_returns_nullptr() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        // The fixture only registers a CUDA allocator, so CPU must remain unset.
        let alloc = get_device_allocator(DeviceType::CPU);
        assert!(alloc.is_null());
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn/test]
    #[test]
    fn device_allocator_test_registry_singleton_instance() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        let instance1 = DeviceAllocatorRegistry::instance() as *const DeviceAllocatorRegistry;
        let instance2 = DeviceAllocatorRegistry::instance() as *const DeviceAllocatorRegistry;
        assert_eq!(instance1, instance2);
    }

    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn/test]
    //
    // The C++ `RegisteringSameDeviceTypeTwiceAborts` is an `EXPECT_DEATH` guarded
    // by `GTEST_HAS_DEATH_TEST`. Registering a second CUDA allocator aborts via
    // `runtime_abort()` (`libc::abort()`, not an unwind), so like the other death
    // tests it is `#[should_panic]`+`#[ignore]`.
    #[test]
    #[should_panic]
    #[ignore]
    fn device_allocator_test_registering_same_device_type_twice_aborts() {
        let _guard = DEVICE_REGISTRY_TEST_LOCK.lock().unwrap();
        setup();
        static mut ANOTHER: MockDeviceAllocator = MockDeviceAllocator::new(DeviceType::CUDA);
        register_device_allocator((&raw mut ANOTHER) as *mut (dyn DeviceAllocator + 'static));
    }

    // `virtual ~DeviceAllocator() = default` maps to Drop on the concrete
    // implementor behind a `Box<dyn DeviceAllocator>`: the trait is
    // object-safe, usable through the base pointer, and dropping through the
    // base runs the concrete type's cleanup exactly once. Does not touch the
    // shared registry, so no DEVICE_REGISTRY_TEST_LOCK is needed.
    // [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-allocator-fn/test]
    #[test]
    fn device_allocator_test_drop_through_dyn_runs_concrete_drop() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct TrackedAllocator;
        impl Drop for TrackedAllocator {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        impl DeviceAllocator for TrackedAllocator {
            fn allocate(
                &mut self,
                _nbytes: usize,
                _index: DeviceIndex,
                _alignment: usize,
            ) -> Result<*mut core::ffi::c_void> {
                Err(Error::NotSupported)
            }
            fn deallocate(&mut self, _ptr: *mut core::ffi::c_void, _index: DeviceIndex) {}
            fn copy_host_to_device(
                &mut self,
                _dst: *mut core::ffi::c_void,
                _src: *const core::ffi::c_void,
                _nbytes: usize,
                _index: DeviceIndex,
            ) -> Error {
                Error::NotSupported
            }
            fn copy_device_to_host(
                &mut self,
                _dst: *mut core::ffi::c_void,
                _src: *const core::ffi::c_void,
                _nbytes: usize,
                _index: DeviceIndex,
            ) -> Error {
                Error::NotSupported
            }
            fn device_type(&self) -> DeviceType {
                DeviceType::CPU
            }
        }

        let alloc: Box<dyn DeviceAllocator> = Box::new(TrackedAllocator);
        assert_eq!(alloc.device_type(), DeviceType::CPU);
        drop(alloc);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    }
}
