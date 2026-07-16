//! Literal port of kernels/portable/cpu/util/transpose_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, tensor_has_dim, tensor_has_rank_smaller_or_equal_to,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, StridesType, ssize_t};

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. These
// checks pass no arguments (`ET_LOG_AND_RETURN_IF_FALSE`), so the local
// `et_log_and_return_if_false!` mirroring the C++ is sufficient.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

/// Increments an N dimensional index like x[0,0,0] to x[0, 0, 1] to x[0, 0, 2]
/// to x[0, 1, 0] to x[0, 1, 1] etc...
// [spec:et:def:transpose-util.torch.executor.increment-index-and-offset-fn]
// [spec:et:sem:transpose-util.torch.executor.increment-index-and-offset-fn]
//
// # Safety
// `index`, `new_sizes`, and `new_strides` must each point to enough elements to
// cover every dimension listed in `non_one_indices`.
unsafe fn increment_index_and_offset(
    index: *mut usize,
    new_sizes: *const SizesType,
    new_strides: *const StridesType,
    non_one_indices: ArrayRef<usize>,
    offset: &mut usize,
) {
    let mut j: usize = non_one_indices.size();
    while j > 0 {
        let i: usize = *non_one_indices.at(j - 1);

        unsafe {
            *index.add(i) += 1;
            // Impossible to happen at i = 0 due to precondition check before this
            // function is called
            *offset += *new_strides.add(i) as usize;
            if *index.add(i) as SizesType == *new_sizes.add(i) {
                *offset -= (*new_sizes.add(i) * *new_strides.add(i)) as usize;
                *index.add(i) = 0;
            } else {
                return;
            }
        }
        j -= 1;
    }
}

/// Returns a tensor that is a transposed version of input in out.
/// The given dimensions dim0 and dim1 are swapped.
// [spec:et:def:transpose-util.torch.executor.transpose-tensors-fn]
// [spec:et:sem:transpose-util.torch.executor.transpose-tensors-fn]
pub fn transpose_tensors<T: Copy>(a: &Tensor, dim0: i64, dim1: i64, out: &Tensor) {
    let dim: ssize_t = a.dim();
    let data_a: *const T = a.const_data_ptr::<T>();
    let data_out: *mut T = out.mutable_data_ptr::<T>();

    let mut out_index: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    let mut new_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut new_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    if dim != 0 {
        let a_strides = a.strides();
        for i in 0..dim as usize {
            new_strides[i] = *a_strides.at(i);
        }

        let a_sizes = a.sizes();
        for i in 0..dim as usize {
            new_sizes[i] = *a_sizes.at(i);
        }

        new_sizes.swap(dim0 as usize, dim1 as usize);
        new_strides.swap(dim1 as usize, dim0 as usize);
    }

    // non_1_dim_indices stores the indices of the dimensions that have a value
    // greater than 1, in increasing output-dimension order.
    let mut non_1_dim_indices: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut num_non_1_dim_indices: usize = 0;
    for cur_dim in 0..dim as usize {
        if new_sizes[cur_dim] != 1 {
            non_1_dim_indices[num_non_1_dim_indices] = cur_dim;
            num_non_1_dim_indices += 1;
        }
    }

    let indices: ArrayRef<usize> =
        ArrayRef::from_raw_parts(non_1_dim_indices.as_ptr(), num_non_1_dim_indices);

    // Loop over and copy input elements into output
    let mut a_offset: usize = 0;
    for out_offset in 0..a.numel() {
        unsafe {
            *data_out.add(out_offset as usize) = *data_a.add(a_offset);
            increment_index_and_offset(
                out_index.as_mut_ptr(),
                new_sizes.as_ptr(),
                new_strides.as_ptr(),
                indices,
                &mut a_offset,
            );
        }
    }
}

// [spec:et:def:transpose-util.torch.executor.check-t-copy-args-fn]
// [spec:et:sem:transpose-util.torch.executor.check-t-copy-args-fn]
pub fn check_t_copy_args(in_: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_rank_smaller_or_equal_to(in_, 2));
    true
}

// [spec:et:def:transpose-util.torch.executor.check-transpose-copy-args-fn]
// [spec:et:sem:transpose-util.torch.executor.check-transpose-copy-args-fn]
pub fn check_transpose_copy_args(in_: &Tensor, dim0: i64, dim1: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim0));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim1));
    true
}

// [spec:et:def:transpose-util.torch.executor.get-transpose-out-target-size-fn]
// [spec:et:sem:transpose-util.torch.executor.get-transpose-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_transpose_out_target_size(
    in_: &Tensor,
    dim0: SizesType,
    dim1: SizesType,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = in_.dim() as usize;

    if in_.dim() == 0 {
        return;
    }

    for i in 0..in_.dim() {
        unsafe { *out_sizes.add(i as usize) = in_.size(i) as SizesType };
    }
    unsafe {
        *out_sizes.add(dim0 as usize) = in_.size(dim1 as ssize_t) as SizesType;
        *out_sizes.add(dim1 as usize) = in_.size(dim0 as ssize_t) as SizesType;
    }
}
