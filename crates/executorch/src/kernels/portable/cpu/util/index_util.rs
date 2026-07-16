//! Literal port of kernels/portable/cpu/util/index_util.cpp.

use crate::runtime::core::exec_aten::util::scalar_type_util::to_string;
use crate::runtime::core::exec_aten::util::tensor_util::{
    dim_is_valid, nonempty_size, nonzero_dim, tensor_has_dim, tensor_has_rank_smaller_or_equal_to,
    tensors_have_same_dtype2, tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::core::portable_type::tensor_impl::ssize_t;

// PORT-NOTE: the crate-level `et_check_or_return_false!` drops all
// caller-supplied format arguments after the leading literal (see the analogous
// note in tensor_util.rs). This local macro mirrors the C++
// `ET_CHECK_OR_RETURN_FALSE` faithfully so this module's ports stay literal.
// Unresolved cross-module reference.
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

// [spec:et:def:index-util.torch.executor.check-gather-args-fn]
// [spec:et:sem:index-util.torch.executor.check-gather-args-fn]
pub fn check_gather_args(
    in_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    _sparse_grad: bool,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    et_check_or_return_false!(
        index.scalar_type() == ScalarType::Long,
        "Expected dtype int64 for index; index.scalar_type() = {}",
        to_string(index.scalar_type())
    );
    if index.numel() != 0 {
        et_check_or_return_false!(
            nonzero_dim(in_) == nonzero_dim(index),
            "self and index should have the same dimensionality when index is not empty except for the case when one has dimension 0 and the other has dimension 1; nonzero_dim(in) = {}, nonzero_dim(index) = {}",
            nonzero_dim(in_),
            nonzero_dim(index)
        );
    }

    // Normalize dim to non-negative value
    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    for d in 0..nonzero_dim(in_) {
        if d != dim as ssize_t {
            et_check_or_return_false!(
                nonempty_size(index, d) <= nonempty_size(in_, d),
                "size of dimension {} of index should be smaller than the size of that dimension of input if dimension {} != dim {}",
                d,
                d,
                dim as usize
            );
        }
    }
    let index_data: *const i64 = index.const_data_ptr::<i64>();
    for i in 0..index.numel() {
        et_check_or_return_false!(
            unsafe { *index_data.add(i as usize) } >= 0
                && unsafe { *index_data.add(i as usize) }
                    < nonempty_size(in_, dim as ssize_t) as i64,
            "Index is out of bounds for dimension {} with size {}",
            dim as usize,
            nonempty_size(index, dim as ssize_t)
        );
    }

    true
}

// [spec:et:def:index-util.torch.executor.check-index-select-args-fn]
// [spec:et:sem:index-util.torch.executor.check-index-select-args-fn]
pub fn check_index_select_args(in_: &Tensor, mut dim: i64, index: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    dim = if dim < 0 {
        dim + nonzero_dim(in_) as i64
    } else {
        dim
    };
    et_check_or_return_false!(
        nonempty_size(in_, dim as ssize_t) > 0,
        "index_select: Indexing axis dim should be positive; nonempty_size(in, dim) = {}",
        nonempty_size(in_, dim as ssize_t)
    );

    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_check_or_return_false!(
        index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int,
        "Expected index to have type of Long or Int, but found {}",
        to_string(index.scalar_type())
    );

    et_log_and_return_if_false!(tensor_has_rank_smaller_or_equal_to(index, 1));
    if index.dim() > 0 && in_.dim() == 0 {
        et_check_or_return_false!(
            index.numel() == 1,
            "index_select: Index to scalar must have exactly 1 value; index.numel() = {}",
            index.numel()
        );
    }

    if index.scalar_type() == ScalarType::Long {
        let index_ptr: *const i64 = index.const_data_ptr::<i64>();
        for i in 0..index.numel() {
            et_check_or_return_false!(
                unsafe { *index_ptr.add(i as usize) } >= 0
                    && unsafe { *index_ptr.add(i as usize) }
                        < nonempty_size(in_, dim as ssize_t) as i64,
                "index[{}] = {} is out of range [0, {})",
                i as usize,
                unsafe { *index_ptr.add(i as usize) },
                nonempty_size(in_, dim as ssize_t) as usize
            );
        }
    } else {
        let index_ptr: *const i32 = index.const_data_ptr::<i32>();
        for i in 0..index.numel() {
            et_check_or_return_false!(
                unsafe { *index_ptr.add(i as usize) } as i64 >= 0
                    && (unsafe { *index_ptr.add(i as usize) } as i64)
                        < nonempty_size(in_, dim as ssize_t) as i64,
                "index[{}] = {} is out of range [0, {})",
                i as usize,
                unsafe { *index_ptr.add(i as usize) },
                nonempty_size(in_, dim as ssize_t) as usize
            );
        }
    }

    true
}

// [spec:et:def:index-util.torch.executor.get-index-select-out-target-size-fn]
// [spec:et:sem:index-util.torch.executor.get-index-select-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in.dim()` valid `TensorSizesType`
/// elements and `out_ndim` to a valid `usize`.
pub unsafe fn get_index_select_out_target_size(
    in_: &Tensor,
    dim: i64,
    index: &Tensor,
    out_sizes: *mut TensorSizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;
    }
    for i in 0..in_.dim() {
        if i == dim as ssize_t {
            unsafe {
                *out_sizes.add(i as usize) = index.numel() as TensorSizesType;
            }
        } else {
            unsafe {
                *out_sizes.add(i as usize) = in_.size(i) as TensorSizesType;
            }
        }
    }
}

// [spec:et:def:index-util.torch.executor.check-nonzero-args-fn]
// [spec:et:sem:index-util.torch.executor.check-nonzero-args-fn]
pub fn check_nonzero_args(in_: &Tensor, out: &Tensor) -> bool {
    let _ = in_;

    et_check_or_return_false!(
        out.scalar_type() == ScalarType::Long,
        "Expected out to be a Long tensor but received {}",
        to_string(out.scalar_type())
    );

    et_check_or_return_false!(
        out.dim() == 2,
        "Expected out to be a 2d tensor received {}",
        out.dim() as ssize_t
    );

    true
}

// [spec:et:def:index-util.torch.executor.check-scatter-add-args-fn]
// [spec:et:sem:index-util.torch.executor.check-scatter-add-args-fn]
pub fn check_scatter_add_args(
    self_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    src: &Tensor,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(self_, out));
    et_log_and_return_if_false!(tensors_have_same_dtype2(self_, src));
    et_check_or_return_false!(
        index.scalar_type() == ScalarType::Long,
        "Expected dtype int64 for index; index.scalar_type() = {}",
        to_string(index.scalar_type())
    );
    et_log_and_return_if_false!(tensor_has_dim(self_, dim));

    if index.numel() == 0 {
        return true;
    }

    et_check_or_return_false!(
        nonzero_dim(self_) == nonzero_dim(src) && nonzero_dim(self_) == nonzero_dim(index),
        "self, index and src should have same number of dimensions; nonzero_dim(self) = {}, nonzero_dim(src) = {}, nonzero_dim(index) = {}",
        nonzero_dim(self_),
        nonzero_dim(src),
        nonzero_dim(index)
    );

    // Normalize dim to non-negative value
    if dim < 0 {
        dim += nonzero_dim(self_) as i64;
    }

    for d in 0..nonzero_dim(self_) {
        et_check_or_return_false!(
            nonempty_size(index, d) <= nonempty_size(src, d),
            "size of dimension {} of index should be smaller than the size of that dimension of src",
            d
        );
        if d != dim as ssize_t {
            et_check_or_return_false!(
                nonempty_size(index, d) <= nonempty_size(self_, d),
                "size of dimension {} of index should be smaller than the size of that dimension of self if dimension {} != dim {}",
                d,
                d,
                dim as usize
            );
        }
    }
    let index_data: *const i64 = index.const_data_ptr::<i64>();
    for i in 0..index.numel() {
        et_check_or_return_false!(
            unsafe { *index_data.add(i as usize) } >= 0
                && unsafe { *index_data.add(i as usize) }
                    < nonempty_size(self_, dim as ssize_t) as i64,
            "Index is out of bounds for dimension {} with size {}",
            dim as usize,
            nonempty_size(self_, dim as ssize_t)
        );
    }
    true
}

// [spec:et:def:index-util.torch.executor.check-scatter-src-args-fn]
// [spec:et:sem:index-util.torch.executor.check-scatter-src-args-fn]
pub fn check_scatter_src_args(
    self_: &Tensor,
    dim: i64,
    index: &Tensor,
    src: &Tensor,
    out: &Tensor,
) -> bool {
    check_scatter_add_args(self_, dim, index, src, out)
}

// [spec:et:def:index-util.torch.executor.check-scatter-value-args-fn]
// [spec:et:sem:index-util.torch.executor.check-scatter-value-args-fn]
pub fn check_scatter_value_args(
    self_: &Tensor,
    dim: i64,
    index: &Tensor,
    _value: &Scalar,
    out: &Tensor,
) -> bool {
    check_gather_args(self_, dim, index, false, out)
}

// [spec:et:def:index-util.torch.executor.check-select-scatter-args-fn]
// [spec:et:sem:index-util.torch.executor.check-select-scatter-args-fn]
pub fn check_select_scatter_args(
    in_: &Tensor,
    src: &Tensor,
    dim: i64,
    index: i64,
    output: &Tensor,
) -> bool {
    // Assumptions for inputs:
    // 1. output size is the same as input size
    // 2. src size is the same as the selected slice from the input
    // 3. dim and index values are valid given the input tensor

    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, output));

    // The dim planed to be selected on shall exist in input
    et_log_and_return_if_false!(dim_is_valid(dim, in_.dim() as i64));

    // The index shall be valid in the given dimenson
    et_check_or_return_false!(
        index >= 0 && index < in_.size(dim as ssize_t) as i64,
        "index {} out of range [-{},{}) at in.size( {})",
        index,
        in_.size(dim as ssize_t),
        in_.size(dim as ssize_t),
        dim
    );

    // The src.dim() shall be one lower than in.dim() since src needs to fit
    // into the selected data on one dim of input
    // https://pytorch.org/docs/stable/generated/torch.select_scatter.html
    et_check_or_return_false!(
        in_.dim() == src.dim() + 1,
        "in.dim() {} != src.dim() + 1 {}",
        in_.dim(),
        src.dim() + 1
    );

    // The size of src tensor should follow these rules:
    // - src.size(i) shall equal to in.size(i) if i < dim,
    // - src.size(i) shall equal to in.size(i+1) if i >= dim

    let mut d: ssize_t = 0;
    while d < in_.dim() - 1 {
        if d < dim as ssize_t {
            et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                in_, d as usize, src, d as usize
            ));
        } else {
            et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                in_,
                (d + 1) as usize,
                src,
                d as usize
            ));
        }
        d += 1;
    }

    true
}
