//! Literal port of runtime/core/portable_type/device.h.

/// Represents the type of compute device.
/// Note: ExecuTorch Device is distinct from PyTorch Device.
// [spec:et:def:device.executorch.runtime.etensor.device-type]
#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeviceType {
    CPU = 0,
    CUDA = 1,
}

/// Total number of device types, used for fixed-size registry arrays.
pub const K_NUM_DEVICE_TYPES: usize = 2;

/// An index representing a specific device; e.g. GPU 0 vs GPU 1.
pub type DeviceIndex = i8;

/// An abstraction for the compute device on which a tensor is located.
///
/// Tensors carry a Device to express where their underlying data resides
/// (e.g. CPU host memory vs CUDA device memory). The runtime uses this to
/// dispatch memory allocation to the appropriate device allocator.
// [spec:et:def:device.executorch.runtime.etensor.device]
// PORT-NOTE: `Debug` is derived so tests can use `assert_eq!` on `Device`
// (gtest's `EXPECT_EQ` prints operands via `operator<<`; Rust's `assert_eq!`
// requires `Debug`). Non-behavioral; mirrors the C++ struct being comparable.
#[derive(Clone, Copy, Debug)]
pub struct Device {
    type_: DeviceType,
    index_: DeviceIndex,
}

/// `using Type = DeviceType;`
pub type DeviceTypeAlias = DeviceType;

impl Device {
    /// Constructs a new `Device` from a `DeviceType` and an optional device
    /// index.
    // [spec:et:def:device.executorch.runtime.etensor.device.device-fn]
    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn]
    // PORT-NOTE: C++ default arg `index = 0` on an implicit constructor. Rust
    // has no default args; `new` takes the explicit index and `from_type`
    // provides the index-0 convenience conversion.
    pub fn new(type_: DeviceType, index: DeviceIndex) -> Self {
        Device {
            type_,
            index_: index,
        }
    }

    pub fn from_type(type_: DeviceType) -> Self {
        Device::new(type_, 0)
    }

    /// Returns the type of device the tensor data resides on.
    // [spec:et:def:device.executorch.runtime.etensor.device.type-fn]
    // [spec:et:sem:device.executorch.runtime.etensor.device.type-fn]
    pub fn type_(&self) -> DeviceType {
        self.type_
    }

    /// Returns true if the device is of CPU type.
    // [spec:et:def:device.executorch.runtime.etensor.device.is-cpu-fn]
    // [spec:et:sem:device.executorch.runtime.etensor.device.is-cpu-fn]
    pub fn is_cpu(&self) -> bool {
        self.type_ == DeviceType::CPU
    }

    /// Returns the device index.
    // [spec:et:def:device.executorch.runtime.etensor.device.index-fn]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn]
    pub fn index(&self) -> DeviceIndex {
        self.index_
    }
}

// [spec:et:def:device.executorch.runtime.etensor.device.operator-fn]
// [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn]
impl PartialEq for Device {
    fn eq(&self, other: &Device) -> bool {
        self.type_ == other.type_ && self.index_ == other.index_
    }
}

impl Eq for Device {}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DeviceType enum ---

    // [spec:et:sem:device.executorch.runtime.etensor.device-type/test]
    #[test]
    fn device_type_test_enum_values() {
        assert_eq!(DeviceType::CPU as i8, 0);
        assert_eq!(DeviceType::CUDA as i8, 1);
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device-type/test]
    #[test]
    fn device_type_test_num_device_types_covers_all_enums() {
        // kNumDeviceTypes must be large enough to index all defined device types.
        assert!(K_NUM_DEVICE_TYPES > DeviceType::CPU as usize);
        assert!(K_NUM_DEVICE_TYPES > DeviceType::CUDA as usize);
    }

    // --- Device: CPU ---

    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.is-cpu-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.type-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn/test]
    // PORT-NOTE: C++ `Device d(DeviceType::CPU)` uses the implicit index=0 ctor;
    // mapped to `Device::from_type` per the module's default-arg PORT-NOTE.
    #[test]
    fn device_test_cpu_default_index() {
        let d = Device::from_type(DeviceType::CPU);
        assert!(d.is_cpu());
        assert_eq!(d.type_(), DeviceType::CPU);
        assert_eq!(d.index(), 0);
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.is-cpu-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn/test]
    #[test]
    fn device_test_cpu_explicit_index() {
        let d = Device::new(DeviceType::CPU, 0);
        assert!(d.is_cpu());
        assert_eq!(d.index(), 0);
    }

    // --- Device: CUDA ---

    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.is-cpu-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.type-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn/test]
    #[test]
    fn device_test_cuda_default_index() {
        let d = Device::from_type(DeviceType::CUDA);
        assert!(!d.is_cpu());
        assert_eq!(d.type_(), DeviceType::CUDA);
        assert_eq!(d.index(), 0);
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn/test]
    #[test]
    fn device_test_cuda_explicit_index() {
        let d = Device::new(DeviceType::CUDA, 0);
        assert_eq!(d.index(), 0);
    }

    // --- Device: equality ---

    // [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn/test]
    #[test]
    fn device_test_equality_same_type_and_index() {
        assert_eq!(
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CPU, 0)
        );
        assert_eq!(
            Device::new(DeviceType::CUDA, 1),
            Device::new(DeviceType::CUDA, 1)
        );
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn/test]
    #[test]
    fn device_test_inequality_different_type() {
        assert_ne!(
            Device::new(DeviceType::CPU, 0),
            Device::new(DeviceType::CUDA, 0)
        );
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn/test]
    #[test]
    fn device_test_inequality_different_index() {
        assert_ne!(
            Device::new(DeviceType::CUDA, 0),
            Device::new(DeviceType::CUDA, 1)
        );
    }

    // [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn/test]
    #[test]
    fn device_test_equality_default_indices() {
        assert_eq!(
            Device::from_type(DeviceType::CPU),
            Device::from_type(DeviceType::CPU)
        );
        assert_eq!(
            Device::from_type(DeviceType::CUDA),
            Device::from_type(DeviceType::CUDA)
        );
        assert_ne!(
            Device::from_type(DeviceType::CPU),
            Device::from_type(DeviceType::CUDA)
        );
    }

    // --- Device: implicit construction ---

    // [spec:et:sem:device.executorch.runtime.etensor.device.device-fn/test]
    // [spec:et:sem:device.executorch.runtime.etensor.device.index-fn/test]
    // PORT-NOTE: C++ implicit `Device d = DeviceType::CUDA;` conversion maps to
    // the explicit `Device::from_type` per the module's default-arg PORT-NOTE
    // (Rust has no implicit conversion constructor).
    #[test]
    fn device_test_implicit_construction_from_device_type() {
        let d = Device::from_type(DeviceType::CUDA);
        assert_eq!(d.index(), 0);
    }

    // --- Deprecated namespace aliases ---

    // PORT-NOTE: the C++ `DeprecatedNamespaceAliases` test verifies the
    // `torch::executor::{Device, DeviceType}` aliases resolve to the etensor
    // types. The Rust port has a single `Device`/`DeviceType` (no separate
    // deprecated namespace alias), so the test reduces to the same construction
    // against the canonical types.
    #[test]
    fn device_test_deprecated_namespace_aliases() {
        let d = Device::new(DeviceType::CUDA, 0);
        assert_eq!(d.index(), 0);
    }
}
