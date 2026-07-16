//! Literal port of kernels/portable/cpu/util/repeat_util.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. This
// local override mirrors the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully.
// Unresolved cross-module reference (matches tensor_util.rs).
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

// PORT-NOTE: `ET_LOG_AND_RETURN_IF_FALSE(cond)` expands to
// `ET_CHECK_OR_RETURN_FALSE(cond, "")`.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// [spec:et:def:repeat-util.torch.executor.check-repeat-args-fn]
// [spec:et:sem:repeat-util.torch.executor.check-repeat-args-fn]
//
// PORT-NOTE: C++ takes `Tensor self` by value; the ported `Tensor` is a cheap
// non-owning handle, so passed by shared reference here.
fn check_repeat_args(self_: &Tensor, repeats: ArrayRef<i64>, out: &Tensor) -> bool {
    // Ensure the self tensors list is non-empty.
    et_check_or_return_false!(
        repeats.size() as ssize_t >= self_.dim(),
        "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor; repeats.size() = {}, self.dim() = {}",
        repeats.size(),
        self_.dim()
    );

    // Repeat arrayref shall not contain negative element.
    let mut all_non_negative: bool = true;
    for i in 0..repeats.size() {
        all_non_negative = all_non_negative && (*repeats.at(i) >= 0);
    }
    et_check_or_return_false!(
        all_non_negative,
        "Trying to create tensor with negative dimension"
    );

    // Check if out.size() is legal.
    et_check_or_return_false!(
        out.dim() as usize == repeats.size(),
        "The dimension of out shall equal size of repeats, but now is {} and {}",
        out.dim(),
        repeats.size()
    );

    // Right now we only support the tensors whose dimension is no greater than
    // kTensorDimensionLimit. Only check out tensor because the number of
    // dimension of out tensor shall have more than or equal to self tensor
    et_check_or_return_false!(
        out.dim() as usize <= K_TENSOR_DIMENSION_LIMIT,
        "The dimension of input and output should not be larger than {}",
        K_TENSOR_DIMENSION_LIMIT
    );

    et_log_and_return_if_false!(tensors_have_same_dtype2(out, self_));

    // We pad one to the beginning of self.size() to make its length equal
    // repeats, and called it reformat_self_size. We then make point-to-point mul
    // of reformat_self_size and repeats. The result should equal out.size().
    let mut reformat_self_size: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut i: ssize_t = 0;
    while i < out.dim() - self_.dim() {
        reformat_self_size[i as usize] = 1;
        i += 1;
    }

    for i in 0..self_.dim() {
        reformat_self_size[(out.dim() - 1 - i) as usize] = self_.size(self_.dim() - 1 - i) as usize;
    }
    for i in 0..repeats.size() {
        et_check_or_return_false!(
            reformat_self_size[i] * (*repeats.at(i) as usize)
                == out.size(i as ssize_t) as u64 as usize,
            "Expect out size at dimension {} is {}, but now is {}",
            i,
            reformat_self_size[i] * *repeats.at(i) as usize,
            out.size(i as ssize_t)
        );
    }

    true
}

// Given the indices to a point in an n-D tensor, and the stride (in bytes)
// along each dimension, return the offset from origin to that point.
// [spec:et:def:repeat-util.torch.executor.compute-access-offset-fn]
// [spec:et:sem:repeat-util.torch.executor.compute-access-offset-fn]
//
// # Safety
// `indices` and `strides` must each point to at least `num_entries` elements.
unsafe fn compute_access_offset(
    indices: *const usize,
    strides: *const usize,
    num_entries: usize,
) -> usize {
    let mut offset: usize = 0;
    let mut i: isize = num_entries as isize - 1;
    while i >= 0 {
        // indices and strides share same length.
        offset += unsafe { *indices.add(i as usize) * *strides.add(i as usize) };
        i -= 1;
    }
    offset
}

// Copy an self array to multiple coordinates of the out tensor.
// [spec:et:def:repeat-util.torch.executor.repeat-internal-fn]
// [spec:et:sem:repeat-util.torch.executor.repeat-internal-fn]
//
// # Safety
// `strides` must point to at least `max(self.dim(), 1)` elements.
unsafe fn repeat_internal(
    self_: &Tensor,
    out: &Tensor,
    in_offset: usize,
    out_offset: usize,
    strides: *const usize,
) {
    let src: *const u8 = unsafe { self_.const_data_ptr::<u8>().add(in_offset) };
    let dest: *mut u8 = unsafe { out.mutable_data_ptr::<u8>().add(out_offset) };

    // Treats zero-dim self as one-dim tensor with size {1}.
    let self_dim: ssize_t = if self_.dim() != 0 { self_.dim() } else { 1 };
    let one: SizesType = 1;
    let self_size: ArrayRef<SizesType> = if self_.dim() != 0 {
        self_.sizes()
    } else {
        ArrayRef::from_raw_parts(&one, 1)
    };

    // Get the size of the array in bytes.
    let num_bytes: usize =
        *self_size.at((self_dim - 1) as usize) as usize * out.element_size() as usize;
    if num_bytes == 0 {
        return;
    }

    // Visualize the out tensor as a set of 1D arrays.
    let mut slots: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for s in slots.iter_mut().take(self_dim as usize) {
        *s = 0;
    }

    // The increment along index of slot array to reach the next possible valid
    // value.
    let mut incr: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..self_dim as usize {
        incr[i] = *self_size.at(i) as i64;
    }

    // And now copy the self data to possibly multiple points in the out tensor.
    let mut index: usize = (self_dim - 1) as usize;
    let start: usize = (out.dim() - self_dim) as usize;
    while slots[0] != out.size(start as ssize_t) as usize {
        // Compute the offset (from origin) in the out tensor where this self
        // data will be copied to.
        let offset: usize =
            unsafe { compute_access_offset(slots.as_ptr(), strides, self_dim as usize) };
        unsafe {
            core::ptr::copy_nonoverlapping(src, dest.add(offset), num_bytes);
        }

        // Find the next valid value of slot array.
        slots[index] += incr[index] as usize;
        // If we have reached the limit in the innermost dimension, successively
        // increment the slot index of outer dimensions.
        while slots[index] == out.size((start + index) as ssize_t) as usize {
            if index == 0 {
                break;
            }
            slots[index] = 0;
            index -= 1;
            slots[index] += incr[index] as usize;
        }
        index = (self_dim - 1) as usize;
    }
}

// TODO(gasoonjia): dynamic allocate array to support tensor dimension larger
// than kTensorDimensionLimit.
// [spec:et:def:repeat-util.torch.executor.repeat-tensor-fn]
// [spec:et:sem:repeat-util.torch.executor.repeat-tensor-fn]
pub fn repeat_tensor(self_: &Tensor, repeats: ArrayRef<i64>, out: &Tensor) -> Error {
    // Verify that the args are valid.
    crate::et_check_or_return_error!(
        check_repeat_args(self_, repeats, out),
        InvalidArgument,
        "Repeat arguments are invalid."
    );

    // Returns out if out.numel == 0, nothing needs to be repeated.
    if out.numel() == 0 {
        return Error::Ok;
    }

    let element_size: ssize_t = out.element_size();

    // The underlying data of tensor out shall equal tensor self.
    // Treats it specially to circumvent zero-dim tensor issue.
    if out.numel() == 1 {
        let src: *const u8 = self_.const_data_ptr::<u8>();
        let dest: *mut u8 = out.mutable_data_ptr::<u8>();
        unsafe {
            core::ptr::copy_nonoverlapping(src, dest, element_size as usize);
        }
        return Error::Ok;
    }

    // Treats zero-dim self as one-dim tensor with size {1}.
    let self_dim: ssize_t = if self_.dim() != 0 { self_.dim() } else { 1 };
    let one: SizesType = 1;
    let self_size: ArrayRef<SizesType> = if self_.sizes().empty() {
        ArrayRef::from_raw_parts(&one, 1)
    } else {
        self_.sizes()
    };

    // Compute the stride (in bytes) along each out tensor dimension.
    let mut strides: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for s in strides.iter_mut().take(self_dim as usize) {
        *s = 0;
    }
    let start: usize = (out.dim() - self_dim) as usize;
    let mut accum_offset: usize = element_size as usize;

    let mut i: ssize_t = self_dim - 1;
    while i >= 0 {
        strides[i as usize] = accum_offset;
        accum_offset *= out.size((start + i as usize) as ssize_t) as usize;
        i -= 1;
    }

    // Given an n-dimensional self X[d0, d1, ..., d{N-2}, d{N-1}], iterate over
    // all the points in X'[d0, ..., d{N-2}, 1].
    let mut slots: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for s in slots.iter_mut().take(self_dim as usize) {
        *s = 0;
    }

    // 'limits' indicates the upper bound on each index in slot.
    let mut limits: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..self_dim as usize {
        limits[i] = *self_size.at(i) as i64;
    }

    // Here limits is guaranteed a non-empty array.
    let in_incr: usize = limits[(self_dim - 1) as usize] as usize * element_size as usize;
    limits[(self_dim - 1) as usize] = 1;
    // 'in_offset' indicates the offset (in bytes) from the origin to an self data.
    let mut in_offset: usize = 0;
    let mut index: usize = (self_dim - 1) as usize;

    // Copy the entire self tensor into the out tensor (at origin), one array at a
    // time.
    while slots[0] as i64 != limits[0] {
        // Compute the offset (from origin) in the out tensor where the self array
        // will be copied.
        let out_offset: usize =
            unsafe { compute_access_offset(slots.as_ptr(), strides.as_ptr(), self_dim as usize) };
        // Now repeatedly copy the array to multiple coordinates in the out tensor.
        unsafe {
            repeat_internal(self_, out, in_offset, out_offset, strides.as_ptr());
        }

        // Find the next valid value of slot array
        slots[index] += 1;
        // If we have reached the limit in the innermost dimension, successively
        // increment the slot index of outer dimensions.
        while slots[index] as i64 == limits[index] {
            if index == 0 {
                break;
            }
            slots[index] = 0;
            index -= 1;
            slots[index] += 1;
        }
        index = (self_dim - 1) as usize;
        in_offset += in_incr;
    }

    // And now if an n-D self was meant to be replicated to m dimensions where
    // m>n, we can just do simple memcpy for (m-n) dimensions.
    let src: *const u8 = out.const_data_ptr::<u8>();
    let mut dest: *mut u8 = unsafe { out.mutable_data_ptr::<u8>().add(accum_offset) };
    let mut i: isize = start as isize - 1;
    while i >= 0 {
        for _j in 0..(*repeats.at(i as usize) - 1) {
            unsafe {
                core::ptr::copy_nonoverlapping(src, dest, accum_offset);
                dest = dest.add(accum_offset);
            }
        }
        accum_offset *= out.size(i as ssize_t) as usize;
        i -= 1;
    }

    Error::Ok
}
