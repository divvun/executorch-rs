//! Literal port of runtime/core/exec_aten/util/dim_order_util.h.

use crate::runtime::core::error::Error;

// PORT-NOTE: `kTensorDimensionLimit` is defined in
// runtime/core/exec_aten/util/tensor_dimension_limit.h, which has no ported
// `tensor_dimension_limit.rs` target yet. The constant is inlined here with the
// same value (16) until that module lands. Unresolved cross-module reference.
const K_TENSOR_DIMENSION_LIMIT: usize = 16;

// PORT-NOTE: The C++ functions here are templated over `SizesType`,
// `DimOrderType`, and `StridesType` purely to break a header cycle (see the
// source comment referencing kernel_types.h / TODO(T148342910)); there is a
// single instantiation set in practice. This port fixes the concrete portable
// widths — `SizesType = i32`, `DimOrderType = u8`, `StridesType = i32` — from
// `torch::executor::Tensor`, matching the committed `Span<i32>`/`Span<u8>` shape
// used elsewhere in the runtime port. Construct deviation from the C++ templates.
type SizesType = i32;
type DimOrderType = u8;
type StridesType = i32;

// [spec:et:def:dim-order-util.executorch.runtime.validate-dim-order-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn]
//
// # Safety
// `dim_order` must point to at least `dims` valid `DimOrderType` elements.
// PORT-NOTE (wave-2 tensor_util group): promoted to `pub` so the ported
// `tensor_util_portable.rs` / `tensor_util_aten.rs` can call it, matching the
// C++ header contract (tensor_util calls `validate_dim_order`). Its sibling
// predicates (`is_contiguous_dim_order`, `is_channels_last_dim_order`,
// `stride_to_dim_order`) are already `pub`; this closes that inconsistency.
pub unsafe fn validate_dim_order(dim_order: *const DimOrderType, dims: usize) -> bool {
    // static_assert(kTensorDimensionLimit <= 16, "Bitmask-based validation
    // requires kTensorDimensionLimit <= 16");
    const _: () = assert!(K_TENSOR_DIMENSION_LIMIT <= 16);
    if dims > K_TENSOR_DIMENSION_LIMIT {
        return false;
    }
    let mut seen: u16 = 0;
    for i in 0..dims {
        if unsafe { *dim_order.add(i) } as usize >= dims {
            return false;
        }
        let mask: u16 = 1u16 << unsafe { *dim_order.add(i) };
        if seen & mask != 0 {
            return false;
        }
        seen |= mask;
    }
    true
}

/// Check if a given dim_order array is equivalent to the contiguous dim order of
/// {0, 1, 2, 3, ...}
///
/// # Safety
/// `dim_order` must point to at least `dims` valid `DimOrderType` elements.
// [spec:et:def:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]
pub unsafe fn is_contiguous_dim_order(dim_order: *const DimOrderType, dims: usize) -> bool {
    for i in 0..dims {
        if unsafe { *dim_order.add(i) } != i as DimOrderType {
            return false;
        }
    }
    true
}

/// Check if a given dim_order array is equivalent to a channels last dim order.
/// Channels last dim order is only valid for 4-dim and 5-dim tensors.
///
/// # Safety
/// `dim_order` must point to at least `dims` valid `DimOrderType` elements.
// [spec:et:def:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]
pub unsafe fn is_channels_last_dim_order(dim_order: *const DimOrderType, dims: usize) -> bool {
    if dims != 4 && dims != 5 {
        return false;
    }
    // 4-dim tensor is interpreted as NCHW, 5-dim tensor is interpreted as NCHWD
    let channels_dim: DimOrderType = 1;
    // Last value in the dim order should be the channels dim
    if unsafe { *dim_order.add(dims - 1) } != channels_dim {
        return false;
    }

    if unsafe { *dim_order.add(0) } != 0 {
        return false;
    }
    let mut d: DimOrderType = 1;
    while d < dims as DimOrderType - 1 {
        if unsafe { *dim_order.add(d as usize) } != d + 1 {
            return false;
        }
        d += 1;
    }
    true
}

/// This utility translated sizes to strides by using dimension order
/// information.
///
/// Note that this function does not check that the provided dim order is valid.
/// This function should only be used when the validity of the dim order has been
/// checked beforehand. A safer version of this function is provided below as
/// `dim_order_to_stride` which will check that the dim order is valid.
///
/// # Safety
/// `sizes`, `dim_order`, and `strides` must each point to at least `dims` valid
/// elements of their respective types.
// [spec:et:def:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn]
pub unsafe fn dim_order_to_stride_nocheck(
    sizes: *const SizesType,
    dim_order: *const DimOrderType,
    dims: usize,
    strides: *mut StridesType,
) {
    // For 0 dim tensors, just return ok.
    if dims == 0 {
        return;
    }
    // Fastest moving dim has stride of 1.
    // For example:
    // Size = [2, 3, 4, 5] dim_names = [N, C, H, W]
    // dim_order = [0, 2, 3, 1]
    // strides = [60, 1, 15, 3]
    unsafe {
        *strides.add(*dim_order.add(dims - 1) as usize) = 1;
    }
    let mut i: i32 = dims as i32 - 2;
    while i >= 0 {
        let nxt = unsafe { *dim_order.add((i + 1) as usize) } as usize;
        let cur = unsafe { *dim_order.add(i as usize) } as usize;
        if unsafe { *sizes.add(nxt) } == 0 {
            unsafe {
                *strides.add(cur) = *strides.add(nxt);
            }
        } else {
            unsafe {
                *strides.add(cur) = *strides.add(nxt) * *sizes.add(nxt);
            }
        }
        i -= 1;
    }
}

/// # Safety
/// `sizes`, `dim_order`, and `strides` must each point to at least `dims` valid
/// elements of their respective types.
// [spec:et:def:dim-order-util.executorch.runtime.dim-order-to-stride-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn]
#[must_use]
pub unsafe fn dim_order_to_stride(
    sizes: *const SizesType,
    dim_order: *const DimOrderType,
    dims: usize,
    strides: *mut StridesType,
) -> Error {
    // For 0 dim tensors, just return ok.
    if dims == 0 {
        return Error::Ok;
    }
    crate::et_check_or_return_error!(
        unsafe { validate_dim_order(dim_order, dims) },
        InvalidArgument,
        "Invalid dim order: values must be a permutation of [0, {})",
        dims
    );

    unsafe { dim_order_to_stride_nocheck(sizes, dim_order, dims, strides) };
    Error::Ok
}

pub mod internal {
    use super::{DimOrderType, StridesType};

    // [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order]
    #[derive(Clone, Copy)]
    pub struct StrideDimOrder {
        pub stride: StridesType,
        pub dim_order: DimOrderType,
    }

    impl StrideDimOrder {
        // [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn]
        // [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn]
        pub fn new(stride_: StridesType, dim_order_: DimOrderType) -> Self {
            StrideDimOrder {
                stride: stride_,
                dim_order: dim_order_,
            }
        }

        // `StrideDimOrder() = default;` — a defaulted zero-argument constructor.
        // PORT-NOTE: C++ `= default` leaves both fields uninitialized. This
        // default zero-initializes them, which is only used to fill the fixed
        // stack array before every live entry is overwritten in
        // stride_to_dim_order.
        pub fn default_value() -> Self {
            StrideDimOrder {
                stride: 0,
                dim_order: 0,
            }
        }

        // [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]
        // [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]
        pub fn gt(&self, other: &StrideDimOrder) -> bool {
            // descending order
            self.stride < other.stride
        }
    }

    // [spec:et:def:dim-order-util.executorch.runtime.internal.sorter]
    pub struct Sorter;

    impl Sorter {
        // [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn]
        // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn]
        pub fn quick_sort(&self, arr: &mut [StrideDimOrder], low: i32, high: i32) {
            if low < high {
                let pivot = arr[high as usize];
                let pos = self.partition(arr, low, high, pivot);

                self.quick_sort(arr, low, pos - 1);
                self.quick_sort(arr, pos + 1, high);
            }
        }

        // [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.swap-fn]
        // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.swap-fn]
        fn swap(&self, arr: &mut [StrideDimOrder], pos1: i32, pos2: i32) {
            let temp = arr[pos1 as usize];
            arr[pos1 as usize] = arr[pos2 as usize];
            arr[pos2 as usize] = temp;
        }

        // [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.partition-fn]
        // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.partition-fn]
        fn partition(
            &self,
            arr: &mut [StrideDimOrder],
            low: i32,
            high: i32,
            pivot: StrideDimOrder,
        ) -> i32 {
            let mut i: i32 = low;
            let mut j: i32 = low;
            while i <= high {
                if arr[i as usize].gt(&pivot) {
                    i += 1;
                } else {
                    self.swap(arr, i, j);
                    i += 1;
                    j += 1;
                }
            }
            j - 1
        }
    }
}

/// This utility translated strides to dimension order information.
///
/// # Safety
/// `strides` must point to at least `dims` valid `StridesType` elements, and
/// `dim_order` (when non-null) to at least `dims` valid `DimOrderType` elements.
// [spec:et:def:dim-order-util.executorch.runtime.stride-to-dim-order-fn]
// [spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn]
#[must_use]
pub unsafe fn stride_to_dim_order(
    strides: *const StridesType,
    dims: usize,
    dim_order: *mut DimOrderType,
) -> Error {
    const K_MAX_NUM_OF_DIMENSIONS: usize = 16;
    crate::et_check_or_return_error!(
        !dim_order.is_null(),
        MemoryAllocationFailed,
        "Need memory to get dim_order."
    );
    crate::et_check_or_return_error!(
        dims <= K_MAX_NUM_OF_DIMENSIONS,
        NotSupported,
        "dims {} exceeds maximum allowed {}",
        dims,
        K_MAX_NUM_OF_DIMENSIONS
    );
    let mut array: [internal::StrideDimOrder; K_MAX_NUM_OF_DIMENSIONS] =
        [internal::StrideDimOrder::default_value(); K_MAX_NUM_OF_DIMENSIONS];
    let mut i: DimOrderType = 0;
    while (i as usize) < dims {
        array[i as usize].dim_order = i;
        array[i as usize].stride = unsafe { *strides.add(i as usize) };
        i += 1;
    }

    let sorter = internal::Sorter;

    sorter.quick_sort(&mut array, 0, dims as i32 - 1);

    for i in 0..dims {
        unsafe {
            *dim_order.add(i) = array[i].dim_order;
        }
    }
    Error::Ok
}

// Literal port of runtime/core/exec_aten/util/test/dim_order_util_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors `DimOrderUtilTest::SetUp()`'s `runtime_init()`; the PAL must be
    // initialized before code paths that call `ET_LOG`.
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // Mirrors the anonymous-namespace `check_strides_eq`.
    fn check_strides_eq(strides_a: &[StridesType], strides_b: &[StridesType]) {
        for i in 0..strides_a.len() {
            assert_eq!(strides_a[i], strides_b[i]);
        }
    }

    // Mirrors the anonymous-namespace `check_dim_order_eq`.
    fn check_dim_order_eq(dim_order_a: &[DimOrderType], dim_order_b: &[DimOrderType]) {
        for i in 0..dim_order_a.len() {
            assert_eq!(dim_order_a[i], dim_order_b[i]);
        }
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // dim_order_to_stride delegates all stride math to dim_order_to_stride_nocheck
    // after validation; every expected-stride assertion below fails if the
    // nocheck body is wrong.
    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn/test]
    #[test]
    fn dim_order_util_test_dim_order_to_stride() {
        setup();
        let sizes_1: [SizesType; 1] = [5];
        let dim_order_1: [DimOrderType; 1] = [0];
        let mut strides_1: [StridesType; 1] = [0];
        let expected_strides_1: [StridesType; 1] = [1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_1.as_ptr(),
                dim_order_1.as_ptr(),
                1,
                strides_1.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_1, &expected_strides_1);

        let sizes_2: [SizesType; 2] = [2, 5];
        let mut dim_order_2: [DimOrderType; 2] = [0, 1];
        let mut strides_2: [StridesType; 2] = [0, 0];
        let mut expected_strides_2: [StridesType; 2] = [5, 1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_2.as_ptr(),
                dim_order_2.as_ptr(),
                2,
                strides_2.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_2, &expected_strides_2);

        dim_order_2[0] = 1;
        dim_order_2[1] = 0;
        expected_strides_2[0] = 1;
        expected_strides_2[1] = 2;
        let error = unsafe {
            dim_order_to_stride(
                sizes_2.as_ptr(),
                dim_order_2.as_ptr(),
                2,
                strides_2.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_2, &expected_strides_2);

        let sizes_3: [SizesType; 3] = [2, 5, 7];
        let mut dim_order_3: [DimOrderType; 3] = [0, 1, 2];
        let mut strides_3: [StridesType; 3] = [0, 0, 0];
        let mut expected_strides_3: [StridesType; 3] = [35, 7, 1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_3.as_ptr(),
                dim_order_3.as_ptr(),
                3,
                strides_3.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3, &expected_strides_3);

        // {0, 2, 1}
        dim_order_3[0] = 0;
        dim_order_3[1] = 2;
        dim_order_3[2] = 1;
        // Expected stride {35, 1, 5}
        expected_strides_3[0] = 35;
        expected_strides_3[1] = 1;
        expected_strides_3[2] = 5;
        let error = unsafe {
            dim_order_to_stride(
                sizes_3.as_ptr(),
                dim_order_3.as_ptr(),
                3,
                strides_3.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3, &expected_strides_3);

        // {2, 5, 7}
        // {1, 2, 0}
        dim_order_3[0] = 1;
        dim_order_3[1] = 2;
        dim_order_3[2] = 0;
        // Expected stride {35, 1, 5}
        expected_strides_3[0] = 1;
        expected_strides_3[1] = 14;
        expected_strides_3[2] = 2;
        let error = unsafe {
            dim_order_to_stride(
                sizes_3.as_ptr(),
                dim_order_3.as_ptr(),
                3,
                strides_3.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3, &expected_strides_3);

        let sizes_4: [SizesType; 4] = [2, 5, 7, 8];
        let mut dim_order_4: [DimOrderType; 4] = [0, 1, 2, 3];
        let mut strides_4: [StridesType; 4] = [0, 0, 0, 0];
        let mut expected_strides_4: [StridesType; 4] = [280, 56, 8, 1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_4.as_ptr(),
                dim_order_4.as_ptr(),
                4,
                strides_4.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_4, &expected_strides_4);

        // {2, 5, 7, 8}
        // {0, 2, 3, 1}
        dim_order_4[0] = 0;
        dim_order_4[1] = 2;
        dim_order_4[2] = 3;
        dim_order_4[3] = 1;
        // Expected stride {280, 1, 40, 5}
        expected_strides_4[0] = 280;
        expected_strides_4[1] = 1;
        expected_strides_4[2] = 40;
        expected_strides_4[3] = 5;
        let error = unsafe {
            dim_order_to_stride(
                sizes_4.as_ptr(),
                dim_order_4.as_ptr(),
                4,
                strides_4.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_4, &expected_strides_4);

        // {2, 5, 7, 8}
        // {3, 1, 2, 0}
        dim_order_4[0] = 3;
        dim_order_4[1] = 1;
        dim_order_4[2] = 2;
        dim_order_4[3] = 0;
        // Expected stride {1, 14, 2, 70}
        expected_strides_4[0] = 1;
        expected_strides_4[1] = 14;
        expected_strides_4[2] = 2;
        expected_strides_4[3] = 70;
        let error = unsafe {
            dim_order_to_stride(
                sizes_4.as_ptr(),
                dim_order_4.as_ptr(),
                4,
                strides_4.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_4, &expected_strides_4);

        let sizes_5: [SizesType; 5] = [2, 5, 7, 8, 9];
        let mut dim_order_5: [DimOrderType; 5] = [0, 1, 2, 3, 4];
        let mut strides_5: [StridesType; 5] = [0, 0, 0, 0, 0];
        let mut expected_strides_5: [StridesType; 5] = [2520, 504, 72, 9, 1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_5.as_ptr(),
                dim_order_5.as_ptr(),
                5,
                strides_5.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_5, &expected_strides_5);

        // {2, 5, 7, 8, 9}
        // {0, 2, 3, 4, 1}
        dim_order_5[0] = 0;
        dim_order_5[1] = 2;
        dim_order_5[2] = 3;
        dim_order_5[3] = 4;
        dim_order_5[4] = 1;
        // Expected stride {2520, 1, 360, 45, 5}
        expected_strides_5[0] = 2520;
        expected_strides_5[1] = 1;
        expected_strides_5[2] = 360;
        expected_strides_5[3] = 45;
        expected_strides_5[4] = 5;
        let error = unsafe {
            dim_order_to_stride(
                sizes_5.as_ptr(),
                dim_order_5.as_ptr(),
                5,
                strides_5.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_5, &expected_strides_5);

        // {2, 5, 7, 8, 9}
        // {4, 2, 0, 3, 1}
        dim_order_5[0] = 4;
        dim_order_5[1] = 2;
        dim_order_5[2] = 0;
        dim_order_5[3] = 3;
        dim_order_5[4] = 1;
        // Expected stride {40, 1, 80, 5, 560}
        expected_strides_5[0] = 40;
        expected_strides_5[1] = 1;
        expected_strides_5[2] = 80;
        expected_strides_5[3] = 5;
        expected_strides_5[4] = 560;
        let error = unsafe {
            dim_order_to_stride(
                sizes_5.as_ptr(),
                dim_order_5.as_ptr(),
                5,
                strides_5.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_5, &expected_strides_5);

        // Check 0 sized dims
        let sizes_3_zero: [SizesType; 3] = [2, 5, 0];
        let mut dim_order_3_zero: [DimOrderType; 3] = [0, 1, 2];
        let mut strides_3_zero: [StridesType; 3] = [0, 0, 0];
        let mut expected_strides_3_zero: [StridesType; 3] = [5, 1, 1];
        let error = unsafe {
            dim_order_to_stride(
                sizes_3_zero.as_ptr(),
                dim_order_3_zero.as_ptr(),
                3,
                strides_3_zero.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3_zero, &expected_strides_3_zero);

        // {0, 2, 1}
        // {2, 0, 5}
        dim_order_3_zero[0] = 0;
        dim_order_3_zero[1] = 2;
        dim_order_3_zero[2] = 1;
        // Expected stride {5, 5, 1}
        expected_strides_3_zero[0] = 5;
        expected_strides_3_zero[1] = 1;
        expected_strides_3_zero[2] = 5;
        let error = unsafe {
            dim_order_to_stride(
                sizes_3_zero.as_ptr(),
                dim_order_3_zero.as_ptr(),
                3,
                strides_3_zero.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3_zero, &expected_strides_3_zero);

        // {2, 0, 1}
        // {0, 2, 5}
        dim_order_3_zero[0] = 2;
        dim_order_3_zero[1] = 0;
        dim_order_3_zero[2] = 1;
        // Expected stride {10, 5, 1}
        expected_strides_3_zero[0] = 5;
        expected_strides_3_zero[1] = 1;
        expected_strides_3_zero[2] = 10;
        let error = unsafe {
            dim_order_to_stride(
                sizes_3_zero.as_ptr(),
                dim_order_3_zero.as_ptr(),
                3,
                strides_3_zero.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::Ok);
        check_strides_eq(&strides_3_zero, &expected_strides_3_zero);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn/test]
    // stride_to_dim_order sorts the StrideDimOrder records by descending stride
    // via Sorter::quick_sort -> partition -> swap; the reordered [2,0,1] result
    // (from strides [5,1,15]) fails if any of those three is wrong.
    // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.partition-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.swap-fn/test]
    #[test]
    fn dim_order_util_test_stride_to_dim_order() {
        setup();
        let strides: [StridesType; 3] = [5, 1, 15];
        let mut dim_order: [DimOrderType; 3] = [0, 0, 0];

        let error = unsafe { stride_to_dim_order(strides.as_ptr(), 3, dim_order.as_mut_ptr()) };
        assert_eq!(error, Error::Ok);

        let expected_dim_order: [DimOrderType; 3] = [2, 0, 1];
        check_dim_order_eq(&dim_order, &expected_dim_order);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_stride_to_dim_order_same_strides() {
        setup();
        let strides: [StridesType; 4] = [4, 3, 1, 1];
        let mut dim_order: [DimOrderType; 4] = [0, 0, 0, 0];

        let error = unsafe { stride_to_dim_order(strides.as_ptr(), 4, dim_order.as_mut_ptr()) };
        assert_eq!(error, Error::Ok);

        let expected_dim_order: [DimOrderType; 4] = [0, 1, 2, 3];
        check_dim_order_eq(&dim_order, &expected_dim_order);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_is_default_dim_order_test() {
        setup();
        for i in 1..7 {
            let dim_order: Vec<DimOrderType> = (0..i as DimOrderType).collect();

            assert!(unsafe { is_contiguous_dim_order(dim_order.as_ptr(), dim_order.len()) });

            // As a bonus, check that is_channels_last returns false
            assert!(!unsafe { is_channels_last_dim_order(dim_order.as_ptr(), dim_order.len()) });
        }
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_is_default_dim_order_fail_cases_test() {
        setup();
        // Dims is default order but have two elements swapped
        for i in 3..8 {
            let mut dim_order: Vec<DimOrderType> = (0..i as DimOrderType).collect();
            dim_order.swap(0, 1);

            assert!(!unsafe { is_contiguous_dim_order(dim_order.as_ptr(), dim_order.len()) });
        }

        // Dims is default order but shifted by 1
        for i in 3..8 {
            let mut dim_order: Vec<DimOrderType> = vec![0; i as usize];
            for d in 0..i {
                dim_order[d as usize] = ((d + 1) % i) as DimOrderType;
            }

            assert!(!unsafe { is_contiguous_dim_order(dim_order.as_ptr(), dim_order.len()) });
        }
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_is_channels_last_dim_order_test() {
        setup();
        let dim_order_4d: [DimOrderType; 4] = [0, 2, 3, 1];
        let dim_order_5d: [DimOrderType; 5] = [0, 2, 3, 4, 1];

        assert!(unsafe { is_channels_last_dim_order(dim_order_4d.as_ptr(), 4) });
        assert!(unsafe { is_channels_last_dim_order(dim_order_5d.as_ptr(), 5) });

        // As a bonus, check that is_default returns false
        assert!(!unsafe { is_contiguous_dim_order(dim_order_4d.as_ptr(), 4) });
        assert!(!unsafe { is_contiguous_dim_order(dim_order_5d.as_ptr(), 5) });
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_is_channels_last_dim_order_fail_cases_test() {
        setup();
        // Non 4D and 5D dim order returns false
        let dim_order_3d: [DimOrderType; 4] = [1, 2, 0, 0];
        let dim_order_6d: [DimOrderType; 6] = [0, 2, 3, 4, 5, 1];

        assert!(!unsafe { is_channels_last_dim_order(dim_order_3d.as_ptr(), 3) });
        assert!(!unsafe { is_channels_last_dim_order(dim_order_6d.as_ptr(), 6) });

        let dim_order_4d: [DimOrderType; 4] = [0, 3, 2, 1];
        let dim_order_5d: [DimOrderType; 5] = [4, 3, 2, 0, 1];

        assert!(!unsafe { is_channels_last_dim_order(dim_order_4d.as_ptr(), 4) });
        assert!(!unsafe { is_channels_last_dim_order(dim_order_5d.as_ptr(), 5) });
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_dim_order_with_all_duplicates_returns_error() {
        setup();
        let sizes: [SizesType; 3] = [2, 3, 4];
        let dim_order: [DimOrderType; 3] = [0, 0, 0];
        let mut strides: [StridesType; 3] = [0, 0, 0];

        let error = unsafe {
            dim_order_to_stride(sizes.as_ptr(), dim_order.as_ptr(), 3, strides.as_mut_ptr())
        };
        assert_eq!(error, Error::InvalidArgument);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_dim_order_with_partial_duplicate_returns_error() {
        setup();
        let sizes: [SizesType; 3] = [2, 3, 4];
        let dim_order: [DimOrderType; 3] = [0, 1, 1];
        let mut strides: [StridesType; 3] = [0, 0, 0];

        let error = unsafe {
            dim_order_to_stride(sizes.as_ptr(), dim_order.as_ptr(), 3, strides.as_mut_ptr())
        };
        assert_eq!(error, Error::InvalidArgument);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_dim_order_with_missing_value_returns_error() {
        setup();
        let sizes: [SizesType; 3] = [2, 3, 4];
        let dim_order: [DimOrderType; 3] = [1, 2, 2];
        let mut strides: [StridesType; 3] = [0, 0, 0];

        let error = unsafe {
            dim_order_to_stride(sizes.as_ptr(), dim_order.as_ptr(), 3, strides.as_mut_ptr())
        };
        assert_eq!(error, Error::InvalidArgument);
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_dim_order_with_out_of_bounds_value_returns_error() {
        setup();
        let sizes: [SizesType; 3] = [2, 3, 4];
        let dim_order: [DimOrderType; 3] = [0, 1, 5];
        let mut strides: [StridesType; 3] = [0, 0, 0];

        let error = unsafe {
            dim_order_to_stride(sizes.as_ptr(), dim_order.as_ptr(), 3, strides.as_mut_ptr())
        };
        assert_eq!(error, Error::InvalidArgument);
    }

    // PORT-NOTE: no C++ counterpart. stride_to_dim_order builds its records via
    // default_value() + field assignment, so the two-argument constructor has no
    // live caller; this focused test pins it directly against the sem rule (stores
    // stride_ into `stride` and dim_order_ into `dim_order`, no transformation).
    // [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_stride_dim_order_constructor() {
        setup();
        let entry = internal::StrideDimOrder::new(15, 2);
        assert_eq!(entry.stride, 15);
        assert_eq!(entry.dim_order, 2);
    }

    // PORT-NOTE: no direct C++ counterpart. `operator>` (ported as `gt`) is the
    // inverted comparison that makes Sorter::quick_sort order records by
    // DESCENDING stride: "greater" means a strictly SMALLER stride, and equal
    // strides compare not-greater (the same-stride stability exercised by
    // stride_to_dim_order_same_strides above).
    // [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn/test]
    #[test]
    fn dim_order_util_test_stride_dim_order_gt_orders_descending() {
        setup();
        let big = internal::StrideDimOrder::new(15, 0);
        let small = internal::StrideDimOrder::new(1, 1);
        let big_other_dim = internal::StrideDimOrder::new(15, 2);

        // Smaller stride is "greater", so it sorts after larger strides.
        assert!(small.gt(&big));
        assert!(!big.gt(&small));

        // Strict comparison: equal strides are not "greater" of each other,
        // regardless of dim_order.
        assert!(!big.gt(&big_other_dim));
        assert!(!big_other_dim.gt(&big));
    }

    // [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn/test]
    // [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn/test]
    #[test]
    fn dim_order_util_test_too_many_dims_returns_error() {
        setup();
        const K_TOO_MANY_DIMS: usize = K_TENSOR_DIMENSION_LIMIT + 1;
        let sizes: Vec<SizesType> = vec![1; K_TOO_MANY_DIMS];
        let dim_order: Vec<DimOrderType> = (0..K_TOO_MANY_DIMS as DimOrderType).collect();
        let mut strides: Vec<StridesType> = vec![0; K_TOO_MANY_DIMS];

        let error = unsafe {
            dim_order_to_stride(
                sizes.as_ptr(),
                dim_order.as_ptr(),
                K_TOO_MANY_DIMS,
                strides.as_mut_ptr(),
            )
        };
        assert_eq!(error, Error::InvalidArgument);
    }
}
