//! Literal port of runtime/executor/memory_manager.h.

use crate::runtime::core::hierarchical_allocator::HierarchicalAllocator;
use crate::runtime::core::memory_allocator::MemoryAllocatorBase;
use crate::runtime::core::portable_type::device::Device;
use crate::runtime::core::span::Span;

// PORT-NOTE: `ET_CHECK_MSG` (runtime/platform/assert.h) has no ported shared
// macro yet; this local macro mirrors its semantics (log the message, then
// abort via the PAL abort path), matching the pattern already used in
// runtime/core/memory_allocator.rs and hierarchical_allocator.rs. It should be
// replaced by the shared `et_check_msg!` once the assert module is ported.
// Unresolved cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            $crate::et_log!(Fatal, $($arg)*);
            $crate::runtime::platform::abort::runtime_abort();
        }
    };
}

/// A container class for allocators used during Method load and execution.
///
/// This class consolidates all dynamic memory needs for Method load and
/// execution. This can allow for heap-based as well as heap-less execution
/// (relevant to some embedded scenarios), and overall provides more control over
/// memory use.
///
/// This class, however, cannot ensure all allocation is accounted for since
/// kernel and backend implementations are free to use a separate way to allocate
/// memory (e.g., for things like scratch space). But we do suggest that backends
/// and kernels use these provided allocators whenever possible.
// [spec:et:def:memory-manager.executorch.runtime.memory-manager]
//
// PORT-NOTE: the three members are borrowed, non-owning C++ pointers
// (`MemoryAllocator*`, `HierarchicalAllocator*`, `MemoryAllocator*`) that may be
// null. They are stored as raw pointers to preserve pointer identity and the
// nullable semantics (`planned_memory` and `temp_allocator` default to null).
// The two `MemoryAllocator*` base pointers are `*mut dyn MemoryAllocatorBase`
// so they dispatch to the concrete allocator's overrides (e.g. the
// malloc-backed `MallocMemoryAllocator`), exactly as the C++ base pointers do.
pub struct MemoryManager {
    method_allocator_: *mut dyn MemoryAllocatorBase,
    planned_memory_: *mut HierarchicalAllocator,
    temp_allocator_: *mut dyn MemoryAllocatorBase,
}

impl MemoryManager {
    /// Constructs a new MemoryManager.
    ///
    /// @param[in] method_allocator The allocator to use when loading a Method and
    ///     allocating its internal structures. Must outlive the Method that uses
    ///     it.
    /// @param[in] planned_memory The memory-planned buffers to use for mutable
    ///     tensor data when executing a Method. Must outlive the Method that uses
    ///     it. May be null if the Method does not use any memory-planned
    ///     tensor data. The sizes of the buffers in this HierarchicalAllocator
    ///     must agree with the corresponding
    ///     `MethodMeta::num_memory_planned_buffers()` and
    ///     `MethodMeta::memory_planned_buffer_size(N)` values, which are embedded
    ///     in the Program. For device-aware programs, the per-buffer device
    ///     metadata is owned by the HierarchicalAllocator as well.
    /// @param[in] temp_allocator The allocator to use when allocating temporary
    ///     data during kernel or delegate execution. Must outlive the Method that
    ///     uses it. May be null if the Method does not use kernels or
    ///     delegates that allocate temporary data. This allocator will be reset
    ///     after every kernel or delegate call during execution.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.memory-manager-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn]
    //
    // PORT-NOTE: the C++ constructor gives `planned_memory` and `temp_allocator`
    // default `nullptr` arguments. Rust has no default args, so callers pass
    // `core::ptr::null_mut()` explicitly to match. The deprecated four-argument
    // overload (`constant_allocator, non_constant_allocator, runtime_allocator,
    // temporary_allocator`) is a compatibility shim and, per the spec, is not
    // reproduced.
    pub fn new(
        method_allocator: *mut dyn MemoryAllocatorBase,
        planned_memory: *mut HierarchicalAllocator,
        temp_allocator: *mut dyn MemoryAllocatorBase,
    ) -> Self {
        let this = MemoryManager {
            method_allocator_: method_allocator,
            planned_memory_: planned_memory,
            temp_allocator_: temp_allocator,
        };
        // C++ compares the two base pointers by address; `addr_eq` mirrors that
        // by ignoring the fat-pointer vtable half.
        et_check_msg!(
            !core::ptr::addr_eq(method_allocator, temp_allocator),
            "method allocator cannot be the same as temp allocator"
        );
        this
    }

    /// Returns the allocator that the runtime will use to allocate internal
    /// structures while loading a Method. Must not be used after its associated
    /// Method has been loaded.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.method-allocator-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn]
    pub fn method_allocator(&self) -> *mut dyn MemoryAllocatorBase {
        self.method_allocator_
    }

    /// Returns the memory-planned buffers to use for mutable tensor data.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.planned-memory-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn]
    pub fn planned_memory(&self) -> *mut HierarchicalAllocator {
        self.planned_memory_
    }

    /// Returns the allocator to use for allocating temporary data during kernel or
    /// delegate execution.
    ///
    /// This allocator will be reset after every kernel or delegate call during
    /// execution.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn]
    pub fn temp_allocator(&self) -> *mut dyn MemoryAllocatorBase {
        self.temp_allocator_
    }

    /// Returns per-buffer device metadata. One entry per planned memory buffer,
    /// same count as planned_memory buffers. Empty if no device metadata was
    /// provided (CPU-only program) or if `planned_memory` is null.
    ///
    /// This is a thin wrapper around
    /// `HierarchicalAllocator::planned_buffer_devices()`.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn]
    pub fn planned_buffer_devices(&self) -> Span<Device> {
        if self.planned_memory_.is_null() {
            return Span::new();
        }
        unsafe { (*self.planned_memory_).planned_buffer_devices() }
    }

    /// Returns true if any planned buffer has device metadata attached.
    /// When false, the memory setup is CPU-only.
    // [spec:et:def:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn]
    pub fn has_device_memory(&self) -> bool {
        self.planned_buffer_devices().size() > 0
    }
}

// Literal port of runtime/executor/test/memory_manager_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::memory_allocator::MemoryAllocator;
    use crate::runtime::core::portable_type::device::DeviceType;

    // Mirrors the C++ fixture: the PAL must be initialized before code paths that
    // call `ET_LOG` / abort.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // A null `*mut dyn MemoryAllocatorBase`: a fat pointer needs a concrete null
    // to cast from (mirrors the pattern in runtime/backend/interface.rs). Stands
    // in for the C++ constructor's default `nullptr` temp-allocator argument.
    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn/test]
    #[test]
    fn memory_manager_test_minimal_ctor() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;

        // C++ `MemoryManager mm(&method_allocator)` uses the null defaults.
        let mm = MemoryManager::new(
            method_allocator_ptr,
            core::ptr::null_mut(),
            null_allocator(),
        );

        assert!(core::ptr::addr_eq(
            mm.method_allocator(),
            method_allocator_ptr
        ));
        assert!(mm.planned_memory().is_null());
        assert!(mm.temp_allocator().is_null());
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn/test]
    #[test]
    fn memory_manager_test_ctor_with_planned_memory() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut planned_memory = HierarchicalAllocator::new(Span::new());
        let planned_memory_ptr: *mut HierarchicalAllocator = &mut planned_memory;

        let mm = MemoryManager::new(method_allocator_ptr, planned_memory_ptr, null_allocator());

        assert!(core::ptr::addr_eq(
            mm.method_allocator(),
            method_allocator_ptr
        ));
        assert_eq!(mm.planned_memory(), planned_memory_ptr);
        assert!(mm.temp_allocator().is_null());
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn/test]
    #[test]
    fn memory_manager_test_ctor_with_planned_memory_and_temp() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut planned_memory = HierarchicalAllocator::new(Span::new());
        let planned_memory_ptr: *mut HierarchicalAllocator = &mut planned_memory;
        let mut temp_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let temp_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut temp_allocator;

        let mm = MemoryManager::new(method_allocator_ptr, planned_memory_ptr, temp_allocator_ptr);

        assert!(core::ptr::addr_eq(
            mm.method_allocator(),
            method_allocator_ptr
        ));
        assert_eq!(mm.planned_memory(), planned_memory_ptr);
        assert!(core::ptr::addr_eq(mm.temp_allocator(), temp_allocator_ptr));
    }

    // PORT-NOTE: the C++ `DEPRECATEDCtor` test exercises the deprecated four-arg
    // overload (`constant_allocator, non_constant_allocator, runtime_allocator,
    // temporary_allocator`). Per memory_manager.rs's PORT-NOTE, that compatibility
    // shim is intentionally not reproduced in the Rust port, so this test cannot
    // be expressed. Ported then `#[ignore]`d per Wave-3 convention.
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    #[test]
    #[ignore]
    fn memory_manager_test_deprecated_ctor() {
        // The four-arg deprecated MemoryManager ctor is not ported.
    }

    // PORT-NOTE: the C++ `DeprecatedCtorWithSameAllocator` death test also uses
    // the deprecated four-arg overload, which is not ported. `#[ignore]`d.
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    #[test]
    #[ignore]
    fn memory_manager_test_deprecated_ctor_with_same_allocator() {
        // The four-arg deprecated MemoryManager ctor is not ported.
    }

    // Death test: constructing with method_allocator == temp_allocator aborts.
    // `runtime_abort` (via the local `et_check_msg!`) terminates the process
    // rather than unwinding, so this cannot run in-process; `#[should_panic]
    // #[ignore]` per Wave-3 convention (mirrors hierarchical_allocator.rs).
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.memory-manager-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn memory_manager_test_ctor_with_same_allocator() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut planned_memory = HierarchicalAllocator::new(Span::new());
        let planned_memory_ptr: *mut HierarchicalAllocator = &mut planned_memory;

        // runtime_allocator == temp_allocator — should abort "cannot be the same".
        MemoryManager::new(
            method_allocator_ptr,
            planned_memory_ptr,
            method_allocator_ptr,
        );
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn/test]
    #[test]
    fn memory_manager_test_three_arg_ctor_has_no_device_memory() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut planned_memory = HierarchicalAllocator::new(Span::new());
        let planned_memory_ptr: *mut HierarchicalAllocator = &mut planned_memory;
        let mut temp_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let temp_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut temp_allocator;

        let mm = MemoryManager::new(method_allocator_ptr, planned_memory_ptr, temp_allocator_ptr);

        assert!(!mm.has_device_memory());
        assert_eq!(mm.planned_buffer_devices().size(), 0);
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.method-allocator-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.temp-allocator-fn/test]
    #[test]
    fn memory_manager_test_delegates_device_metadata_to_hierarchical_allocator() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;
        let mut temp_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let temp_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut temp_allocator;

        // 4 buffers: cpu:0, cpu:0, cuda:0, cuda:1.
        const N_BUFFERS: usize = 4;
        let mut mem0 = [0u8; 4];
        let mut mem1 = [0u8; 4];
        let mut mem2 = [0u8; 4];
        let mut mem3 = [0u8; 4];
        let mut buffers: [Span<u8>; N_BUFFERS] = [
            Span::from_raw_parts(mem0.as_mut_ptr(), mem0.len()),
            Span::from_raw_parts(mem1.as_mut_ptr(), mem1.len()),
            Span::from_raw_parts(mem2.as_mut_ptr(), mem2.len()),
            Span::from_raw_parts(mem3.as_mut_ptr(), mem3.len()),
        ];
        let mut devices = [
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CUDA, 0),
            Device::new(DeviceType::CUDA, 1),
        ];
        let device_span = Span::from_raw_parts(devices.as_mut_ptr(), N_BUFFERS);

        let mut planned_memory = HierarchicalAllocator::with_devices(
            Span::from_raw_parts(buffers.as_mut_ptr(), N_BUFFERS),
            device_span,
        );
        let planned_memory_ptr: *mut HierarchicalAllocator = &mut planned_memory;
        let mm = MemoryManager::new(method_allocator_ptr, planned_memory_ptr, temp_allocator_ptr);

        assert!(core::ptr::addr_eq(
            mm.method_allocator(),
            method_allocator_ptr
        ));
        assert_eq!(mm.planned_memory(), planned_memory_ptr);
        assert!(core::ptr::addr_eq(mm.temp_allocator(), temp_allocator_ptr));
        assert!(mm.has_device_memory());
        assert_eq!(mm.planned_buffer_devices().size(), N_BUFFERS);
        // `Device` implements `PartialEq` but comparisons are via `==`.
        assert!(
            unsafe { *mm.planned_buffer_devices().index(0) } == Device::new(DeviceType::CPU, 0)
        );
        assert!(
            unsafe { *mm.planned_buffer_devices().index(1) } == Device::new(DeviceType::CPU, 0)
        );
        assert!(
            unsafe { *mm.planned_buffer_devices().index(2) } == Device::new(DeviceType::CUDA, 0)
        );
        assert!(
            unsafe { *mm.planned_buffer_devices().index(3) } == Device::new(DeviceType::CUDA, 1)
        );
    }

    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.has-device-memory-fn/test]
    // [spec:et:sem:memory-manager.executorch.runtime.memory-manager.planned-buffer-devices-fn/test]
    #[test]
    fn memory_manager_test_minimal_ctor_has_no_device_memory() {
        setup();
        let mut method_allocator = MemoryAllocator::new(0, core::ptr::null_mut());
        let method_allocator_ptr: *mut dyn MemoryAllocatorBase = &mut method_allocator;

        let mm = MemoryManager::new(
            method_allocator_ptr,
            core::ptr::null_mut(),
            null_allocator(),
        );

        assert!(!mm.has_device_memory());
        assert_eq!(mm.planned_buffer_devices().size(), 0);
    }
}
