//! Literal port of runtime/core/exec_aten/util/tensor_util_aten.cpp.
//!
//! Implementation for ATen tensor util, should only be included in a
//! `<target>_aten` target and only be used in ATen mode. In C++ these functions
//! explicitly take `at::Tensor` (instead of `executorch::aten::Tensor`) so the
//! build fails at compile time if built incorrectly.
//!
//! PORT-NOTE: ATen mode requires libtorch's `at::Tensor` / `c10::TensorImpl` /
//! `at::StorageImpl`, which are NOT available in the Rust port — there is no
//! libtorch dependency. This whole module is therefore a *stub target*: the
//! entire body is gated behind `#[cfg(feature = "aten")]` and the logic is
//! ported against the same `Tensor` trait surface used elsewhere in the runtime
//! port (the portable `Tensor`), so the annotations, control flow, and symbol
//! shapes stay faithful. The at::-specific storage plumbing
//! (`unsafe_storage()`, `set_data_ptr(at::DataPtr(...))`, `StorageImpl::reset`,
//! `set_sizes_contiguous`) is represented by the nearest portable `TensorImpl`
//! operations; it will need real libtorch bindings if ATen mode is ever enabled.

#![cfg(feature = "aten")]

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order, stride_to_dim_order, validate_dim_order,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{DimOrderType, SizesType, TensorImpl};

const K_TENSOR_DIMENSION_LIMIT: usize =
    crate::runtime::core::exec_aten::util::tensor_util::K_TENSOR_DIMENSION_LIMIT;

// PORT-NOTE: local re-implementation of `ET_CHECK_OR_RETURN_FALSE` that forwards
// the caller's format args. See the note in tensor_util.rs: the crate-level
// `et_check_or_return_false!` drops all args after the leading literal.
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

/// Get dim_order of a Tensor and write it to out_dim_order.
///
/// # Safety
/// `out_dim_order` must point to at least `out_dim_order_size` valid
/// `DimOrderType` elements.
// [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]
// [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]
#[must_use]
pub unsafe fn get_dim_order(
    tensor: &Tensor,
    out_dim_order: *mut DimOrderType,
    out_dim_order_size: usize,
) -> Error {
    crate::et_check_or_return_error!(
        out_dim_order_size == tensor.dim() as usize,
        InvalidArgument,
        "out_dim_order_size needs to be equal to the number of dimensions of the tensor. out_dim_order_size {}, tensor.dim() {}",
        out_dim_order_size,
        tensor.dim()
    );
    unsafe {
        stride_to_dim_order(
            tensor.strides().data(),
            tensor.dim() as usize,
            out_dim_order,
        )
    }
}

// [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensor-has-valid-dim-order-fn]
// [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensor-has-valid-dim-order-fn]
pub fn tensor_has_valid_dim_order(t: &Tensor) -> bool {
    let mut dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    et_check_or_return_false!(
        unsafe { get_dim_order(t, dim_order.as_mut_ptr(), t.dim() as usize) } == Error::Ok,
        "Failed to retrieve dim order from tensor!"
    );

    if !unsafe { validate_dim_order(dim_order.as_ptr(), t.dim() as usize) } {
        crate::et_log!(Error, "Tensor dim order is not valid:");
        for d in 0..t.dim() {
            crate::et_log!(
                Error,
                "    dim_order({}): {}",
                d as usize,
                dim_order[d as usize] as usize
            );
        }
        return false;
    }
    true
}

// [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensor-is-default-or-channels-last-dim-order-fn]
// [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensor-is-default-or-channels-last-dim-order-fn]
pub fn tensor_is_default_or_channels_last_dim_order(t: &Tensor) -> bool {
    let mut dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    et_check_or_return_false!(
        unsafe { get_dim_order(t, dim_order.as_mut_ptr(), t.dim() as usize) } == Error::Ok,
        "Failed to retrieve dim order from tensor!"
    );

    let ret_val: bool = unsafe { is_contiguous_dim_order(dim_order.as_ptr(), t.dim() as usize) }
        || unsafe { is_channels_last_dim_order(dim_order.as_ptr(), t.dim() as usize) };

    if !ret_val {
        crate::et_log!(
            Error,
            "Expected tensor to have default or channels last dim order, but got"
        );
        for d in 0..t.dim() {
            crate::et_log!(
                Error,
                "    dim_order({}): {}",
                d as usize,
                dim_order[d as usize] as usize
            );
        }
    }
    ret_val
}

// [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensors-have-same-dim-order-fn]
// [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensors-have-same-dim-order-fn]
pub fn tensors_have_same_dim_order(tensor_list: ArrayRef<Tensor>) -> bool {
    if tensor_list.size() < 2 {
        return true;
    }

    let mut first_dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut other_dim_order: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];

    et_check_or_return_false!(
        unsafe {
            get_dim_order(
                tensor_list.at(0),
                first_dim_order.as_mut_ptr(),
                tensor_list.at(0).dim() as usize,
            )
        } == Error::Ok,
        "Failed to retrieve dim order from 1st input tensor!"
    );

    let mut all_contiguous: bool = unsafe {
        is_contiguous_dim_order(first_dim_order.as_ptr(), tensor_list.at(0).dim() as usize)
    };
    let mut all_channels_last: bool = unsafe {
        is_channels_last_dim_order(first_dim_order.as_ptr(), tensor_list.at(0).dim() as usize)
    };

    for i in 1..tensor_list.size() {
        et_check_or_return_false!(
            unsafe {
                get_dim_order(
                    tensor_list.at(i),
                    other_dim_order.as_mut_ptr(),
                    tensor_list.at(i).dim() as usize,
                )
            } == Error::Ok,
            "Failed to retrieve dim order from {}-th input tensor!",
            i
        );

        all_contiguous = all_contiguous
            && unsafe {
                is_contiguous_dim_order(other_dim_order.as_ptr(), tensor_list.at(i).dim() as usize)
            };
        all_channels_last = all_channels_last
            && unsafe {
                is_channels_last_dim_order(
                    other_dim_order.as_ptr(),
                    tensor_list.at(i).dim() as usize,
                )
            };
    }

    et_check_or_return_false!(
        all_contiguous || all_channels_last,
        "{} input tensors have different dim orders",
        tensor_list.size()
    );

    all_contiguous || all_channels_last
}

pub mod internal {
    use super::*;

    // [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.share-tensor-data-fn]
    // [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.share-tensor-data-fn]
    //
    // PORT-NOTE: C++ reaches through `t_dst.unsafeGetTensorImpl()->unsafe_storage()
    // .unsafeGetStorageImpl()` to `set_data_ptr(at::DataPtr(..., CPU))` +
    // `set_nbytes(...)`. No libtorch StorageImpl in the Rust port; the nearest
    // portable operation is `TensorImpl::set_data`. Storage nbytes is not tracked
    // separately in the portable impl (nbytes is derived), so `set_nbytes` has no
    // analog. Stub target; requires real libtorch bindings for ATen mode.
    pub fn share_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        crate::et_check_or_return_error!(
            t_dst.nbytes() == t_src.nbytes(),
            InvalidArgument,
            "t_dst.nbytes() {} != t_src.nbytes(). {}",
            t_dst.nbytes(),
            t_src.nbytes()
        );

        crate::et_check_or_return_error!(
            !t_src.mutable_data_ptr_typed().is_null(),
            InvalidArgument,
            "Source tensor should have data_ptr not being nullptr."
        );
        // Assign the dataptr as the input tensor dataptr
        unsafe { (*t_dst.unsafe_get_tensor_impl()).set_data(t_src.mutable_data_ptr_typed()) };

        Error::Ok
    }

    // [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.copy-tensor-data-fn]
    // [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.copy-tensor-data-fn]
    //
    // PORT-NOTE: C++ obtains the destination pointer through the StorageImpl
    // (`data_ptr().get()`). Modeled here via `mutable_data_ptr_typed()`, the
    // portable equivalent. Unconditional non-null destination check preserved.
    pub fn copy_tensor_data(t_dst: &Tensor, t_src: &Tensor) -> Error {
        let dst_data_ptr: *mut core::ffi::c_void = t_dst.mutable_data_ptr_typed();

        // Currently even 0 sized tensors receive a dataptr in pre_allocated
        // memory planning so we can do this check.
        crate::et_check_or_return_error!(
            !dst_data_ptr.is_null(),
            InvalidArgument,
            "Destination tensor data pointer must not be null."
        );

        // Sources with a size 0 dimension can be nullptr
        if !t_src.const_data_ptr_typed().is_null() {
            crate::et_check_or_return_error!(
                t_dst.nbytes() == t_src.nbytes(),
                InvalidArgument,
                "t_dst.nbytes() {} != t_src.nbytes(). {}",
                t_dst.nbytes(),
                t_src.nbytes()
            );
            // Copy the source data to the preallocated memory of the destination,
            // which must be the same size as the source.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    t_src.const_data_ptr_typed() as *const u8,
                    dst_data_ptr as *mut u8,
                    t_src.nbytes(),
                );
            }
        }

        Error::Ok
    }

    // [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.set-tensor-data-fn]
    // [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.set-tensor-data-fn]
    //
    // PORT-NOTE: C++ installs the buffer as a CPU `at::DataPtr` on the storage.
    // Modeled via portable `TensorImpl::set_data`. Stub target.
    #[must_use]
    pub fn set_tensor_data(
        t: &Tensor,
        buffer: *mut core::ffi::c_void,
        buffer_size: usize,
    ) -> Error {
        crate::et_check_or_return_error!(
            buffer_size >= t.nbytes(),
            InvalidArgument,
            "buffer_size {} is smaller than smaller than tensor nbytes {}",
            buffer_size,
            t.nbytes()
        );
        unsafe { (*t.unsafe_get_tensor_impl()).set_data(buffer) };
        Error::Ok
    }

    // [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.reset-data-ptr-fn]
    // [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.reset-data-ptr-fn]
    //
    // PORT-NOTE: C++ calls `impl->set_sizes_contiguous(0)` then resets the
    // StorageImpl. Modeled here via portable `set_sizes_contiguous` (empty sizes)
    // and `set_data(nullptr)`; the portable impl has no separate StorageImpl to
    // `reset()`. Stub target.
    pub fn reset_data_ptr(tensor: &Tensor) {
        let impl_: *mut TensorImpl = tensor.unsafe_get_tensor_impl();
        unsafe {
            (*impl_).set_sizes_contiguous(ArrayRef::from_raw_parts(core::ptr::null(), 0));
            (*impl_).set_data(core::ptr::null_mut());
        }
    }

    /// Most callers should use resize_tensor() instead.
    // [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.resize-tensor-impl-fn]
    // [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.resize-tensor-impl-fn]
    #[must_use]
    pub fn resize_tensor_impl(impl_: *mut TensorImpl, new_sizes: ArrayRef<SizesType>) -> Error {
        // The lean-mode Tensor will perform this check, but at::Tensor won't.
        // Although at::Tensor can be resized in this case, it's not allowed by the
        // higher-level constraints of the runtime.
        if unsafe { (*impl_).dim() } as usize != new_sizes.size() {
            crate::et_log!(
                Error,
                "Tensor rank is not mutable: old dim: {} new dim: {}",
                unsafe { (*impl_).dim() },
                new_sizes.size()
            );
            return Error::NotSupported;
        }
        // Will panic on failure.
        unsafe { (*impl_).set_sizes_contiguous(new_sizes) };
        Error::Ok
    }
}
