//! Literal port of runtime/core/hierarchical_allocator.h.

use crate::runtime::core::error::Result;
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
// PORT-NOTE: `etensor::Device` lives in runtime/core/portable_type/device.rs,
// which is still a stub at time of writing. Unresolved cross-module reference.
use crate::runtime::core::portable_type::device::Device;
use crate::runtime::core::span::Span;

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

const K_SPAN_ARRAY_SIZE: usize = 16;

/// A group of buffers that can be used to represent a device's memory hierarchy.
// [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator]
pub struct HierarchicalAllocator {
    // TODO(T162089316): Remove the span array and to_spans once all users move to
    // spans. This array is necessary to hold the pointers and sizes that were
    // originally provided as MemoryAllocator instances.
    // NOTE: span_array_ must be declared before buffers_ so that it isn't
    // re-initialized to zeros after initializing buffers_.
    span_array_: [Span<u8>; K_SPAN_ARRAY_SIZE],
    /// The underlying buffers.
    buffers_: Span<Span<u8>>,
    /// Per-buffer device metadata. Empty when no device info was provided
    /// (CPU-only program).
    planned_buffer_devices_: Span<Device>,
}

impl HierarchicalAllocator {
    /// Constructs a new hierarchical allocator with the given array of buffers.
    ///
    /// - Memory IDs are based on the index into `buffers`: `buffers[N]` will have
    ///   a memory ID of `N`.
    /// - `buffers.size()` must be >= `MethodMeta::num_non_const_buffers()`.
    /// - `buffers[N].size()` must be >= `MethodMeta::non_const_buffer_size(N)`.
    pub fn new(buffers: Span<Span<u8>>) -> Self {
        HierarchicalAllocator {
            span_array_: [Span::new(); K_SPAN_ARRAY_SIZE],
            buffers_: buffers,
            planned_buffer_devices_: Span::new(),
        }
    }

    /// Constructs a new hierarchical allocator with per-buffer device metadata.
    ///
    /// @param[in] buffers Same as above. May contain a mix of CPU and device
    ///     pointers — HierarchicalAllocator only does pointer arithmetic, so
    ///     device pointers are valid.
    /// @param[in] planned_buffer_devices One entry per buffer (same count as
    ///     `buffers`), indicating the `Device` (type + index) for each buffer.
    ///     Different buffers can target the same device type but different
    ///     indices (e.g., `cuda:0` vs `cuda:1`). For CPU-only programs, use the
    ///     single-arg constructor instead.
    // [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn]
    pub fn with_devices(buffers: Span<Span<u8>>, planned_buffer_devices: Span<Device>) -> Self {
        let this = HierarchicalAllocator {
            span_array_: [Span::new(); K_SPAN_ARRAY_SIZE],
            buffers_: buffers,
            planned_buffer_devices_: planned_buffer_devices,
        };
        et_check_msg!(
            planned_buffer_devices.size() == buffers.size(),
            "planned_buffer_devices size ({}) must match buffers size ({})",
            planned_buffer_devices.size(),
            buffers.size()
        );
        this
    }

    /// DEPRECATED: Use spans instead.
    // PORT-NOTE: the C++ ctor sets `buffers_` from `to_spans(...)` in its
    // initializer list, which mutates `span_array_`. Here we build the object
    // (with a zeroed span_array_) then call `to_spans`, which writes into
    // `span_array_` and returns a span pointing back into it; the borrow of the
    // just-constructed value mirrors the C++ member-array aliasing.
    pub fn from_allocators(n_allocators: u32, allocators: *mut MemoryAllocator) -> Self {
        let mut this = HierarchicalAllocator {
            span_array_: [Span::new(); K_SPAN_ARRAY_SIZE],
            buffers_: Span::new(),
            planned_buffer_devices_: Span::new(),
        };
        this.buffers_ = this.to_spans(n_allocators, allocators);
        this
    }

    /// Returns the address at the byte offset `offset_bytes` from the given
    /// buffer's base address, which points to at least `size_bytes` of memory.
    ///
    /// @param[in] memory_id The ID of the buffer in the hierarchy.
    /// @param[in] offset_bytes The offset in bytes into the specified buffer.
    /// @param[in] size_bytes The amount of memory that should be available at
    ///     the offset.
    ///
    /// @returns On success, the address of the requested byte offset into the
    ///     specified buffer. On failure, a non-Ok Error.
    // [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn]
    #[must_use]
    pub fn get_offset_address(
        &mut self,
        memory_id: u32,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> Result<*mut core::ffi::c_void> {
        // Check for integer overflow in offset_bytes + size_bytes.
        let end_bytes: usize;
        match offset_bytes.checked_add(size_bytes) {
            Some(sum) => end_bytes = sum,
            None => {
                crate::et_check_or_return_error!(
                    false,
                    InvalidArgument,
                    "Integer overflow in offset_bytes ({}) + size_bytes ({})",
                    offset_bytes,
                    size_bytes
                );
                unreachable!()
            }
        }
        crate::et_check_or_return_error!(
            (memory_id as usize) < self.buffers_.size(),
            InvalidArgument,
            "id {} >= {}",
            memory_id,
            self.buffers_.size()
        );
        let buffer: Span<u8> = unsafe { *self.buffers_.index(memory_id as usize) };
        crate::et_check_or_return_error!(
            end_bytes <= buffer.size(),
            MemoryAllocationFailed,
            "offset_bytes ({}) + size_bytes ({}) >= allocator size ({}) for memory_id {}",
            offset_bytes,
            size_bytes,
            buffer.size(),
            memory_id
        );
        Ok(unsafe { buffer.data().add(offset_bytes) } as *mut core::ffi::c_void)
    }

    /// Returns per-buffer device metadata. One entry per buffer, same count as
    /// the `buffers` passed to the constructor. Each entry is a `Device`
    /// carrying both type and index, so callers can distinguish e.g. `cuda:0`
    /// from `cuda:1`. Empty if no device metadata was provided (CPU-only
    /// program).
    // [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn]
    pub fn planned_buffer_devices(&self) -> Span<Device> {
        self.planned_buffer_devices_
    }

    // [spec:et:def:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn]
    fn to_spans(&mut self, n_allocators: u32, allocators: *mut MemoryAllocator) -> Span<Span<u8>> {
        et_check_msg!(
            (n_allocators as usize) <= K_SPAN_ARRAY_SIZE,
            "n_allocators {} > {}",
            n_allocators,
            K_SPAN_ARRAY_SIZE
        );
        let mut i: u32 = 0;
        while i < n_allocators {
            let alloc: &MemoryAllocator = unsafe { &*allocators.add(i as usize) };
            self.span_array_[i as usize] =
                Span::from_raw_parts(alloc.base_address(), alloc.size() as usize);
            i += 1;
        }
        Span::from_raw_parts(self.span_array_.as_mut_ptr(), n_allocators as usize)
    }
}

// Literal port of runtime/core/test/hierarchical_allocator_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::portable_type::device::DeviceType;
    use crate::runtime::core::result::ResultExt;

    // Mirrors the C++ fixture `SetUp()`: the PAL must be initialized before code
    // paths that call `ET_LOG`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn/test]
    #[test]
    fn hierarchical_allocator_test_smoke() {
        setup();
        const N_BUFFERS: usize = 2;
        const SIZE0: usize = 4;
        const SIZE1: usize = 8;
        let mut mem0 = [0u8; SIZE0];
        let mut mem1 = [0u8; SIZE1];
        let mut buffers: [Span<u8>; N_BUFFERS] = [
            Span::from_raw_parts(mem0.as_mut_ptr(), SIZE0),
            Span::from_raw_parts(mem1.as_mut_ptr(), SIZE1),
        ];

        let mut allocator =
            HierarchicalAllocator::new(Span::from_raw_parts(buffers.as_mut_ptr(), N_BUFFERS));

        // get_offset_address() success cases
        {
            // Total size is 4, so off=0 + size=2 fits.
            let address = allocator.get_offset_address(0, 0, 2);
            assert_eq!(ResultExt::error(&address), Error::Ok);
            assert_ne!(*ResultExt::get(&address), core::ptr::null_mut());
            assert_eq!(
                *ResultExt::get(&address),
                mem0.as_mut_ptr() as *mut core::ffi::c_void
            );
        }
        {
            // Total size is 8, so off=4 + size=4 fits exactly.
            let address = allocator.get_offset_address(1, 4, 4);
            assert_eq!(ResultExt::error(&address), Error::Ok);
            assert_ne!(*ResultExt::get(&address), core::ptr::null_mut());
            assert_eq!(
                *ResultExt::get(&address),
                unsafe { mem1.as_mut_ptr().add(4) } as *mut core::ffi::c_void
            );
        }

        // get_offset_address() failure cases
        {
            // Total size is 4, so off=0 + size=5 is too large.
            let address = allocator.get_offset_address(0, 4, 5);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
        {
            // Total size is 4, so off=8 + size=0 is off the end.
            let address = allocator.get_offset_address(0, 8, 0);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
        {
            // ID too large; only two zero-indexed entries in the allocator.
            let address = allocator.get_offset_address(2, 0, 2);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
    }

    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn/test]
    #[test]
    fn hierarchical_allocator_test_no_device_metadata_by_default() {
        setup();
        let empty_buffers: Span<Span<u8>> = Span::new();
        let allocator = HierarchicalAllocator::new(empty_buffers);

        assert_eq!(allocator.planned_buffer_devices().size(), 0);
    }

    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn/test]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.planned-buffer-devices-fn/test]
    #[test]
    fn hierarchical_allocator_test_exposes_device_metadata_when_provided() {
        setup();
        // Use 4 buffers so the device span size matches.
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

        // CPU buffers come first because the runtime always sets up host-side
        // planned memory before any device buffers. The two CUDA entries use
        // distinct device indices to verify per-buffer index tracking.
        let mut devices = [
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CUDA, 0),
            Device::new(DeviceType::CUDA, 1),
        ];
        let device_span = Span::from_raw_parts(devices.as_mut_ptr(), N_BUFFERS);

        let allocator = HierarchicalAllocator::with_devices(
            Span::from_raw_parts(buffers.as_mut_ptr(), N_BUFFERS),
            device_span,
        );

        let planned = allocator.planned_buffer_devices();
        assert_eq!(planned.size(), N_BUFFERS);
        // `Device` implements `PartialEq` (its `operator==`) but not `Debug`, so
        // compare via `==` rather than `assert_eq!`.
        assert!(unsafe { *planned.index(0) } == Device::new(DeviceType::CPU, 0));
        assert!(unsafe { *planned.index(1) } == Device::new(DeviceType::CPU, 0));
        assert!(unsafe { *planned.index(2) } == Device::new(DeviceType::CUDA, 0));
        assert!(unsafe { *planned.index(3) } == Device::new(DeviceType::CUDA, 1));
    }

    // Death test: 3 device entries vs 2 buffers should abort. `runtime_abort`
    // terminates the process (via PAL abort) rather than unwinding, so this
    // cannot run in-process; `#[should_panic] #[ignore]` per Wave-3 convention.
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.hierarchical-allocator-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn hierarchical_allocator_test_mismatched_device_count_aborts() {
        setup();
        const N_BUFFERS: usize = 2;
        let mut mem0 = [0u8; 4];
        let mut mem1 = [0u8; 4];
        let mut buffers: [Span<u8>; N_BUFFERS] = [
            Span::from_raw_parts(mem0.as_mut_ptr(), mem0.len()),
            Span::from_raw_parts(mem1.as_mut_ptr(), mem1.len()),
        ];

        // 3 device entries vs 2 buffers — should abort.
        let mut devices = [
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CUDA, 0),
        ];
        let device_span = Span::from_raw_parts(devices.as_mut_ptr(), 3);

        HierarchicalAllocator::with_devices(
            Span::from_raw_parts(buffers.as_mut_ptr(), N_BUFFERS),
            device_span,
        );
    }

    // TODO(T162089316): Tests the deprecated API.
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.get-offset-address-fn/test]
    // [spec:et:sem:hierarchical-allocator.executorch.runtime.hierarchical-allocator.to-spans-fn/test]
    #[test]
    fn hierarchical_allocator_test_deprecated_smoke() {
        setup();
        const N_ALLOCATORS: usize = 2;
        const SIZE0: usize = 4;
        const SIZE1: usize = 8;
        let mut mem0 = [0u8; SIZE0];
        let mut mem1 = [0u8; SIZE1];
        let mut allocators = [
            MemoryAllocator::new(SIZE0 as u32, mem0.as_mut_ptr()),
            MemoryAllocator::new(SIZE1 as u32, mem1.as_mut_ptr()),
        ];

        let mut allocator =
            HierarchicalAllocator::from_allocators(N_ALLOCATORS as u32, allocators.as_mut_ptr());

        // get_offset_address() success cases
        {
            let address = allocator.get_offset_address(0, 0, 2);
            assert_eq!(ResultExt::error(&address), Error::Ok);
            assert_ne!(*ResultExt::get(&address), core::ptr::null_mut());
            assert_eq!(
                *ResultExt::get(&address),
                mem0.as_mut_ptr() as *mut core::ffi::c_void
            );
        }
        {
            let address = allocator.get_offset_address(1, 4, 4);
            assert_eq!(ResultExt::error(&address), Error::Ok);
            assert_ne!(*ResultExt::get(&address), core::ptr::null_mut());
            assert_eq!(
                *ResultExt::get(&address),
                unsafe { mem1.as_mut_ptr().add(4) } as *mut core::ffi::c_void
            );
        }

        // get_offset_address() failure cases
        {
            let address = allocator.get_offset_address(0, 4, 5);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
        {
            let address = allocator.get_offset_address(0, 8, 0);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
        {
            let address = allocator.get_offset_address(2, 0, 2);
            assert!(!ResultExt::ok(&address));
            assert_ne!(ResultExt::error(&address), Error::Ok);
        }
    }
}
