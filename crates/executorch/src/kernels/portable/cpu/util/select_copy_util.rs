//! Literal port of kernels/portable/cpu/util/select_copy_util.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, getTrailingDims, nonzero_dim,
    resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_select_copy_out_args, get_select_copy_out_target_size,
};

// [spec:et:def:select-copy-util.torch.executor.select-copy-util-fn]
// [spec:et:sem:select-copy-util.torch.executor.select-copy-util-fn]
pub fn select_copy_util(in_: &Tensor, mut dim: i64, mut index: i64, out: &Tensor) -> Error {
    if !check_select_copy_out_args(in_, dim, index, out) {
        return Error::InvalidArgument;
    }

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    let mut target_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_select_copy_out_target_size(in_, dim, target_sizes.as_mut_ptr(), &mut target_ndim);
    }

    if !(resize_tensor_same_type(
        out,
        ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim),
    ) == Error::Ok)
    {
        return Error::InvalidArgument;
    }

    if !tensors_have_same_dim_order2(in_, out) {
        return Error::InvalidArgument;
    }

    // If the input is a empty tensor, no other operation could be done. We just
    // return the output.
    if in_.numel() == 0 {
        return Error::Ok;
    }
    // The code past this point assumes that the tensors are non-empty.

    // Support python-style negative indexing
    if index < 0 {
        index += in_.size(dim as ssize_t) as i64;
    }

    let leading_dims: usize = getLeadingDims(in_, dim);
    let trailing_dims: usize = getTrailingDims(in_, dim);
    let dim_length: usize = in_.size(dim as ssize_t) as usize;

    // Number of bytes to copy in the each memcpy operation
    let copy_size_per_op: usize = trailing_dims * out.element_size() as usize;

    // Step between the src locations of two adjcant memcpy operations
    let src_step_per_op: usize = dim_length * trailing_dims * in_.element_size() as usize;

    // the start point of data need to be copied is the start point of overall
    // data chunk plus the offset between the overall start point and the first
    // data to be copied.
    let input_data: *mut u8 = in_.mutable_data_ptr::<u8>();

    let start_offset: usize = index as usize * trailing_dims * in_.element_size() as usize;
    let mut src: *const u8 = unsafe { input_data.add(start_offset) };

    let mut dest: *mut u8 = out.mutable_data_ptr::<u8>();

    for _j in 0..leading_dims {
        unsafe {
            core::ptr::copy_nonoverlapping(src, dest, copy_size_per_op);
            src = src.add(src_step_per_op);
            dest = dest.add(copy_size_per_op);
        }
    }

    Error::Ok
}
