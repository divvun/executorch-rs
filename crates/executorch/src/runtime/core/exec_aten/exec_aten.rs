//! Literal port of runtime/core/exec_aten/exec_aten.h.
//!
//! This port targets the non-ATen (`torch::executor`) build. The `USE_ATEN_LIB`
//! branch (mapping to `at::`/`c10::` types) is not part of this port set; the
//! executor-type branch is the single implementation.

use crate::runtime::core::result::Result;

// `using TensorShapeDynamism = executorch::runtime::TensorShapeDynamism;`
pub use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

// executor-type aliases (the `#else // Use executor types` branch).
//
// PORT-NOTE: `Tensor`/`TensorImpl`/`ArrayRef<Tensor>`/`Scalar`/`MemoryFormat`/
// `Device`/`DeviceType`/`Layout` come from `torch::executor::` modules that are
// still stubs (portable_type/tensor.rs etc.) at time of writing; only the
// aliases whose backing modules exist are re-exported. Unresolved cross-module
// references for the rest.
pub use crate::runtime::core::portable_type::scalar_type::ScalarType;

// SizesType/DimOrderType/StridesType are `torch::executor::Tensor::SizesType`
// etc. Their concrete widths (int32_t / uint8_t / int32_t) match the portable
// TensorImpl typedefs.
pub type SizesType = i32;
pub type DimOrderType = u8;
pub type StridesType = i32;

pub use crate::runtime::core::portable_type::BFloat16;
pub use crate::runtime::core::portable_type::Half;
pub use crate::runtime::core::portable_type::qint_types::{
    qint8, qint32, quint2x4, quint4x2, quint8,
};

// [spec:et:def:exec-aten.executorch.aten.compute-numel-fn]
// [spec:et:sem:exec-aten.executorch.aten.compute-numel-fn]
//
// Unchecked variant: forms the array view `sizes[0..dim)` and returns the
// product of all entries as `ssize_t` (`isize`). `dim == 0` yields the empty
// product `1`; multiplication wraps per two's-complement, matching
// `c10::multiply_integers`.
//
// # Safety
// `sizes` must point to at least `dim` valid `SizesType` elements (or be null
// when `dim == 0`, since it is not dereferenced).
pub unsafe fn compute_numel(sizes: *const SizesType, dim: isize) -> isize {
    let mut numel: isize = 1;
    let mut i: isize = 0;
    while i < dim {
        numel = numel.wrapping_mul(unsafe { *sizes.offset(i) } as isize);
        i += 1;
    }
    numel
}

// [spec:et:def:exec-aten.executorch.aten.safe-numel-fn]
// [spec:et:sem:exec-aten.executorch.aten.safe-numel-fn]
//
// # Safety
// `sizes` must point to at least `dim` valid `SizesType` elements (or be null
// when `dim == 0`).
pub unsafe fn safe_numel(sizes: *const SizesType, dim: isize) -> Result<isize> {
    crate::et_check_or_return_error!(
        dim == 0 || !sizes.is_null(),
        InvalidArgument,
        "Sizes must be provided for non-scalar tensors"
    );
    let mut numel: isize = 1;
    let mut i: isize = 0;
    while i < dim {
        let size_i = unsafe { *sizes.offset(i) };
        crate::et_check_or_return_error!(
            size_i >= 0,
            InvalidArgument,
            "Size must be non-negative, got {} at dimension {}",
            size_i as isize,
            i
        );
        // PORT-NOTE: C++ uses `c10::mul_overflows(numel, x, &next_numel)`,
        // returning true on overflow. `checked_mul` returns None on overflow,
        // mirroring the wraparound-detected branch; `next_numel` holds the
        // product when no overflow occurred.
        let next_numel = numel.checked_mul(size_i as isize);
        crate::et_check_or_return_error!(
            next_numel.is_some(),
            InvalidArgument,
            "Overflow computing numel at dimension {}",
            i
        );
        numel = next_numel.unwrap();
        i += 1;
    }
    Ok(numel)
}

// PORT-NOTE: exec_aten.h is header-only with no `exec_aten_test.cpp`; these are
// focused unit tests pinning `compute_numel` / `safe_numel` against the sem
// rules (docs/spec/port/runtime/core/exec_aten/exec_aten.md).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::result::ResultExt;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // [spec:et:sem:exec-aten.executorch.aten.compute-numel-fn/test]
    #[test]
    fn exec_aten_test_compute_numel() {
        setup();
        // dim == 0: empty product is 1; sizes may be null (not dereferenced).
        assert_eq!(unsafe { compute_numel(core::ptr::null(), 0) }, 1);

        let sizes: [SizesType; 4] = [2, 3, 4, 5];
        assert_eq!(unsafe { compute_numel(sizes.as_ptr(), 1) }, 2);
        assert_eq!(unsafe { compute_numel(sizes.as_ptr(), 2) }, 6);
        assert_eq!(unsafe { compute_numel(sizes.as_ptr(), 4) }, 120);

        // Any zero size makes the product 0.
        let with_zero: [SizesType; 3] = [2, 0, 5];
        assert_eq!(unsafe { compute_numel(with_zero.as_ptr(), 3) }, 0);
    }

    // [spec:et:sem:exec-aten.executorch.aten.safe-numel-fn/test]
    #[test]
    fn exec_aten_test_safe_numel() {
        setup();
        // dim == 0: result is 1; sizes may be null.
        let r = unsafe { safe_numel(core::ptr::null(), 0) };
        assert!(ResultExt::ok(&r));
        assert_eq!(*ResultExt::get(&r), 1);

        let sizes: [SizesType; 3] = [2, 3, 4];
        let r = unsafe { safe_numel(sizes.as_ptr(), 3) };
        assert!(ResultExt::ok(&r));
        assert_eq!(*ResultExt::get(&r), 24);

        // Zero size -> 0, no overflow.
        let with_zero: [SizesType; 3] = [2, 0, 4];
        let r = unsafe { safe_numel(with_zero.as_ptr(), 3) };
        assert!(ResultExt::ok(&r));
        assert_eq!(*ResultExt::get(&r), 0);

        // Non-scalar tensor with null sizes -> InvalidArgument.
        let r = unsafe { safe_numel(core::ptr::null(), 2) };
        assert!(!ResultExt::ok(&r));
        assert_eq!(ResultExt::error(&r), Error::InvalidArgument);

        // Negative size -> InvalidArgument.
        let neg: [SizesType; 2] = [3, -1];
        let r = unsafe { safe_numel(neg.as_ptr(), 2) };
        assert!(!ResultExt::ok(&r));
        assert_eq!(ResultExt::error(&r), Error::InvalidArgument);

        // Overflow of the running product -> InvalidArgument. On a 64-bit
        // isize, i32::MAX repeated overflows in three multiplies.
        let big: [SizesType; 3] = [SizesType::MAX, SizesType::MAX, SizesType::MAX];
        let r = unsafe { safe_numel(big.as_ptr(), 3) };
        assert!(!ResultExt::ok(&r));
        assert_eq!(ResultExt::error(&r), Error::InvalidArgument);
    }
}
