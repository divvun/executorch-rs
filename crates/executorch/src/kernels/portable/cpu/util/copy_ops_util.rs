//! Literal port of kernels/portable/cpu/util/copy_ops_util.cpp + kernels/portable/cpu/util/copy_ops_util.h.

use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order,
};
use crate::runtime::core::exec_aten::util::scalar_type_util::can_cast;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, dim_is_valid, getTrailingDims, nonzero_dim, tensor_dim_has_index,
    tensor_has_dim, tensor_has_rank_greater_or_equal_to, tensor_is_rank,
    tensors_have_same_dim_order2, tensors_have_same_dtype2, tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{DimOrderType, SizesType, ssize_t};
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;

// PORT-NOTE: `TensorList` (kernel_includes.h) is `executorch::aten::ArrayRef<Tensor>`.
type TensorList<'a> = ArrayRef<Tensor<'a>>;

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. This
// local override mirrors the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully
// (prepend "Check failed (cond): " then forward the full message + args).
// Unresolved cross-module reference (matches tensor_util.rs / slice_util.rs).
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
// `ET_CHECK_OR_RETURN_FALSE(cond, "")` in the C++ header.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {
        et_check_or_return_false!($cond, "")
    };
}

// [spec:et:def:copy-ops-util.torch.executor.as-strided-copy-fn]
// [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-fn]
//
// PORT-NOTE: header template `_as_strided_copy<CTYPE>`. `CTYPE` is generic; the
// stride-1 contiguous case uses `copy_nonoverlapping` mirroring the C++ memcpy.
fn _as_strided_copy<CTYPE: Copy>(
    mut input_data: *mut CTYPE,
    mut output_data: *mut CTYPE,
    out: &Tensor,
    size: ArrayRef<i64>,
    stride: ArrayRef<i64>,
    dim: i64,
) {
    // the last dimension, copy data
    let stride_dim: i64 = *stride.at(dim as usize);
    if dim == size.size() as i64 - 1 {
        let num_elements: usize = *size.at(dim as usize) as usize;
        // use memcpy for contiguous memory
        if stride_dim == 1 {
            unsafe {
                core::ptr::copy_nonoverlapping(input_data, output_data, num_elements);
            }
        } else {
            for i in 0..num_elements {
                unsafe {
                    *output_data.add(i) = *input_data;
                    input_data = input_data.offset(stride_dim as isize);
                }
            }
        }
        return;
    }
    let trailing_dims: usize = getTrailingDims(out, dim);
    // recursively set data for the next dimension
    for _i in 0..*size.at(dim as usize) {
        _as_strided_copy::<CTYPE>(input_data, output_data, out, size, stride, dim + 1);
        unsafe {
            input_data = input_data.offset(stride_dim as isize);
            output_data = output_data.add(trailing_dims);
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn]
// [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn]
fn as_strided_copy_compute_storage_nbytes(
    sizes: IntArrayRef,
    strides: IntArrayRef,
    itemsize_bytes: usize,
) -> usize {
    // size of the underlying storage is 1 bigger than the offset
    // of the last element according to stride
    let mut size: usize = 1;
    for i in 0..sizes.size() {
        if *sizes.at(i) == 0 {
            return 0;
        }
        size += (*strides.at(i) * (*sizes.at(i) - 1)) as usize;
    }
    size * itemsize_bytes
}

// [spec:et:def:copy-ops-util.torch.executor.check-as-strided-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-as-strided-copy-args-fn]
pub fn check_as_strided_copy_args(
    in_: &Tensor,
    size: ArrayRef<i64>,
    stride: ArrayRef<i64>,
    storage_offset: Option<i64>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_check_or_return_false!(
        size.size() == stride.size(),
        "mismatch in length of strides and shape; size.size() = {}, stride.size() = {}",
        size.size(),
        stride.size()
    );
    for i in 0..stride.size() {
        let val: i64 = *stride.at(i);
        et_check_or_return_false!(
            val >= 0,
            "as_strided: Negative strides are not supported at the moment"
        );
    }

    let offset: i64 = if storage_offset.is_some() {
        storage_offset.unwrap()
    } else {
        0
    };
    et_check_or_return_false!(offset >= 0, "Negative storage offset");

    // Check that the requested storage is within bounds of input storage
    let storage_size_bytes: usize =
        as_strided_copy_compute_storage_nbytes(size, stride, in_.element_size() as usize);
    let storage_offset_bytes: usize = offset as usize * in_.element_size() as usize;
    if storage_size_bytes == 0 {
        return true;
    }
    let new_storage_size_bytes: usize = in_.nbytes();
    et_check_or_return_false!(
        storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
        "Requiring a storage size of {} are out of bounds for storage of size {}",
        storage_size_bytes + storage_offset_bytes,
        new_storage_size_bytes
    );
    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-cat-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-cat-args-fn]
pub fn check_cat_args(tensors: ArrayRef<Tensor>, dim: i64, out: &Tensor) -> bool {
    // Ensure the input tensors list is non-empty
    et_log_and_return_if_false!(tensors.size() > 0);

    // Find the first non-empty tensor in the list to use as a reference
    let mut ref_i: usize = 0;
    for i in 0..tensors.size() {
        if tensors.at(i).numel() > 0 {
            ref_i = i;
            break;
        }
    }

    // "All tensors must either have the same shape (except in the concatenating
    // dimension) or be empty."
    for i in 0..tensors.size() {
        // All input dtypes must be castable to the output dtype.
        et_log_and_return_if_false!(can_cast(tensors.at(i).scalar_type(), out.scalar_type()));

        et_log_and_return_if_false!(tensors_have_same_dim_order2(tensors.at(i), out));

        // Empty tensors have no shape constraints.
        if tensors.at(i).numel() == 0 {
            continue;
        }

        // All input tensors must have the same number of dimensions.
        et_log_and_return_if_false!(tensor_is_rank(
            tensors.at(ref_i),
            tensors.at(i).dim() as usize
        ));

        for d in 0..tensors.at(i).dim() {
            if d as i64 != dim {
                et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                    tensors.at(i),
                    d as usize,
                    tensors.at(ref_i),
                    d as usize
                ));
            }
        }
    }

    // Ensure dim is in range.
    et_log_and_return_if_false!(
        tensors.at(ref_i).numel() == 0 || tensors.at(ref_i).dim() as i64 > dim
    );
    et_log_and_return_if_false!(dim >= 0);

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-cat-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-cat-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `tensors[ref_i].dim()` writable elements.
pub unsafe fn get_cat_out_target_size(
    tensors: ArrayRef<Tensor>,
    dim: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    // Find the first non-1D-or-empty tensor in the list to use as a reference
    // because an 1D empty tensor is a wildcard and should be ignored when we
    // calculate out dim
    let mut ref_i: usize = 0;
    let mut cat_dim_size: usize = 0;
    for i in 0..tensors.size() {
        if tensors.at(i).numel() > 0 {
            cat_dim_size += tensors.at(i).size(dim as ssize_t) as usize;
        }
        if tensors.at(i).dim() != 1 || tensors.at(i).numel() != 0 {
            ref_i = i;
        }
    }

    *out_ndim = tensors.at(ref_i).dim() as usize;

    for d in 0..*out_ndim {
        if d as i64 != dim {
            unsafe { *out_sizes.add(d) = tensors.at(ref_i).size(d as ssize_t) as SizesType };
        } else {
            unsafe { *out_sizes.add(d) = cat_dim_size as SizesType };
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-expand-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-expand-copy-args-fn]
pub fn check_expand_copy_args(
    input: &Tensor,
    expand_sizes: ArrayRef<i64>,
    implicit: bool,
    out: &Tensor,
) -> bool {
    let _ = out;

    et_check_or_return_false!(
        implicit == false,
        "This operator is not implemented for when implicit == true."
    );

    et_check_or_return_false!(
        expand_sizes.size() >= input.sizes().size(),
        "The number of sizes provided ({}) must at least be equal to the number of dimensions in the tensor ({})",
        expand_sizes.size(),
        input.sizes().size()
    );

    et_check_or_return_false!(
        expand_sizes.size() <= K_TENSOR_DIMENSION_LIMIT,
        "The number of expanded dims ({}) exceeds the configured maximum ({}). Increase this limit.",
        expand_sizes.size(),
        K_TENSOR_DIMENSION_LIMIT
    );

    et_log_and_return_if_false!(tensors_have_same_dtype2(input, out));

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-expand-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-expand-copy-out-target-size-fn]
//
// # Safety
// `output_sizes` must point to at least `expand_sizes.size()` writable elements.
pub unsafe fn get_expand_copy_out_target_size(
    self_sizes: ArrayRef<SizesType>,
    expand_sizes: ArrayRef<i64>,
    output_sizes: *mut SizesType,
    output_rank: &mut usize,
) -> bool {
    let mut j: usize = expand_sizes.size();
    *output_rank = 0;

    let mut i: usize = self_sizes.size();
    while i > 0 && j > 0 {
        i -= 1;
        j -= 1;

        unsafe { *output_sizes.add(j) = *expand_sizes.at(j) as SizesType };

        if *expand_sizes.at(j) == -1 {
            // -1 can use for replacing any corresponding dimension
            unsafe { *output_sizes.add(j) = *self_sizes.at(i) };
        } else if *self_sizes.at(i) != 1 {
            et_check_or_return_false!(
                *expand_sizes.at(j) == *self_sizes.at(i) as i64,
                "The expanded size of the tensor ({}) must match the existing size ({}) at non-singleton dimension {}.",
                *expand_sizes.at(j) as usize,
                *self_sizes.at(i) as usize,
                i
            );
        }
    }

    // The leading expand_sizes cannot be negative
    while j > 0 {
        j -= 1;
        unsafe { *output_sizes.add(j) = *expand_sizes.at(j) as SizesType };
        et_check_or_return_false!(
            *expand_sizes.at(j) >= 0,
            "The expanded size of the tensor ({}) isn't allowed in a leading, non-existing dimension {}",
            *expand_sizes.at(j) as usize,
            j
        );
    }

    *output_rank = expand_sizes.size();
    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-permute-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-permute-copy-args-fn]
pub fn check_permute_copy_args(in_: &Tensor, dims: IntArrayRef, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_is_rank(in_, dims.size()));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    // Make sure no dimensions are duplicated and all in the range [-in.dim(),
    // in.dim() - 1].
    let mut dim_exist: [bool; K_TENSOR_DIMENSION_LIMIT] = [false; K_TENSOR_DIMENSION_LIMIT];

    for i in 0..dims.size() {
        et_log_and_return_if_false!(tensor_has_dim(in_, *dims.at(i)));
        // Convert dimension to a non-negative number in the range
        // [0 .. in.dim() - 1].
        let dim: usize = if *dims.at(i) >= 0 {
            *dims.at(i) as usize
        } else {
            (in_.dim() as i64 + *dims.at(i)) as usize
        };

        // Internal check, since we have already validated this
        et_log_and_return_if_false!(dim < K_TENSOR_DIMENSION_LIMIT);

        // Check that the dimension hasn't been seen previously.
        et_check_or_return_false!(
            dim_exist[dim] == false,
            "duplicate dims are not allowed; dim = {}",
            dim
        );

        dim_exist[dim] = true;
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-unbind-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-unbind-copy-args-fn]
pub fn check_unbind_copy_args(in_: &Tensor, dim: i64, out: TensorList) -> bool {
    et_check_or_return_false!(
        in_.dim() > 0,
        "in must have at least one dimension; saw {}",
        in_.dim()
    );

    et_log_and_return_if_false!(dim_is_valid(dim, in_.dim() as i64));

    let dim_size: ssize_t = in_.size(dim as ssize_t);
    et_check_or_return_false!(
        dim_size == out.size() as ssize_t,
        "out tensorlist's length {} must equal unbind dim {} size = {}.",
        out.size(),
        dim,
        dim_size
    );

    // Validate each output.
    for i in 0..out.size() {
        // All output dtypes must be the same.
        et_check_or_return_false!(
            out.at(i).scalar_type() == out.at(0).scalar_type(),
            "out[{}] dtype {} != out[0] dtype {}",
            i,
            out.at(i).scalar_type() as i8,
            out.at(0).scalar_type() as i8
        );

        // output tensor must have # of dims = in.dim() -1
        et_check_or_return_false!(
            out.at(i).dim() == in_.dim() - 1,
            "out[{}] dim {} != in dim {}",
            i,
            out.at(i).dim(),
            in_.dim() - 1
        );

        // Check the shape of the output.
        let mut out_d: ssize_t = 0;
        for d in 0..in_.dim() {
            if d as i64 != dim {
                et_check_or_return_false!(
                    out.at(i).size(out_d) == in_.size(d),
                    "out[{}].size({}) {} != in.size({}) {}",
                    i,
                    d,
                    out.at(i).size(out_d),
                    d,
                    in_.size(d)
                );
                out_d += 1;
            }
        }
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_permute_copy_out_target_size(
    in_: &Tensor,
    dims: IntArrayRef,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = in_.dim() as usize;

    for i in 0..in_.dim() {
        let src_dim: i64 = if *dims.at(i as usize) >= 0 {
            *dims.at(i as usize)
        } else {
            *dims.at(i as usize) + in_.dim() as i64
        };
        unsafe { *out_sizes.add(i as usize) = in_.size(src_dim as ssize_t) as SizesType };
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-pixel-shuffle-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-pixel-shuffle-args-fn]
pub fn check_pixel_shuffle_args(in_: &Tensor, upscale_factor: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 3));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(out, 3));
    et_log_and_return_if_false!(upscale_factor > 0);
    et_log_and_return_if_false!(
        in_.size((in_.dim() - 3) as ssize_t) as i64 % (upscale_factor * upscale_factor) == 0
    );
    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn]
pub fn check_pixel_unshuffle_args(in_: &Tensor, downscale_factor: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 3));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(out, 3));
    et_log_and_return_if_false!(downscale_factor > 0);
    et_log_and_return_if_false!(
        in_.size((in_.dim() - 1) as ssize_t) as i64 % downscale_factor == 0
    );
    et_log_and_return_if_false!(
        in_.size((in_.dim() - 2) as ssize_t) as i64 % downscale_factor == 0
    );
    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-pixel-shuffle-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-pixel-shuffle-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_pixel_shuffle_out_target_size(
    in_: &Tensor,
    upscale_factor: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) -> bool {
    // Prevent signed integer overflow when computing upscale_factor ^ 2.
    et_check_or_return_false!(
        upscale_factor < 32768,
        "Upscale factor must be less than 32768."
    );

    *out_ndim = in_.dim() as usize;
    let casted_upscale_factor: SizesType = upscale_factor as SizesType;

    let mut i: ssize_t = 0;
    while i < in_.dim() - 3 {
        // Copy all leading dimensions in.
        unsafe { *out_sizes.add(i as usize) = in_.size(i) as SizesType };
        i += 1;
    }
    // The last 3 dimensions are (channel, height, width). Divide by the upscale
    // factor squared and multiply the height and width by that factor.
    unsafe {
        *out_sizes.add(i as usize) =
            (in_.size(i) / (casted_upscale_factor * casted_upscale_factor) as ssize_t) as SizesType
    };
    i += 1;
    unsafe {
        *out_sizes.add(i as usize) = (in_.size(i) * casted_upscale_factor as ssize_t) as SizesType
    };
    i += 1;
    unsafe {
        *out_sizes.add(i as usize) = (in_.size(i) * casted_upscale_factor as ssize_t) as SizesType
    };

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_pixel_unshuffle_out_target_size(
    in_: &Tensor,
    downscale_factor: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = in_.dim() as usize;
    let casted_factor: SizesType = downscale_factor as SizesType;

    let mut i: ssize_t = 0;
    while i < in_.dim() - 3 {
        // Copy all leading dimensions in.
        unsafe { *out_sizes.add(i as usize) = in_.size(i) as SizesType };
        i += 1;
    }
    // The last 3 dimensions are (channel, height, width). Multiply channel by
    // the downscale factor squared and divide the height and width by that
    // factor.
    unsafe {
        *out_sizes.add(i as usize) =
            (in_.size(i) * (casted_factor * casted_factor) as ssize_t) as SizesType
    };
    i += 1;
    unsafe { *out_sizes.add(i as usize) = (in_.size(i) / casted_factor as ssize_t) as SizesType };
    i += 1;
    unsafe { *out_sizes.add(i as usize) = (in_.size(i) / casted_factor as ssize_t) as SizesType };
}

// [spec:et:def:copy-ops-util.torch.executor.check-select-copy-out-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-select-copy-out-args-fn]
pub fn check_select_copy_out_args(in_: &Tensor, dim: i64, index: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 1));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    et_log_and_return_if_false!(tensor_dim_has_index(in_, dim, index));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim() - 1` writable elements.
pub unsafe fn get_select_copy_out_target_size(
    in_: &Tensor,
    dim: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = (in_.dim() - 1) as usize;

    for d in 0..(in_.dim() - 1) {
        if (d as i64) < dim {
            unsafe { *out_sizes.add(d as usize) = in_.size(d) as SizesType };
        } else {
            unsafe { *out_sizes.add(d as usize) = in_.size(d + 1) as SizesType };
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-split-with-sizes-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-split-with-sizes-copy-args-fn]
pub fn check_split_with_sizes_copy_args(
    in_: &Tensor,
    split_sizes: ArrayRef<i64>,
    dim: i64,
    out: TensorList,
) -> bool {
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 1));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));

    et_check_or_return_false!(
        split_sizes.size() == out.size(),
        "Number of split sizes must match the number of output tensors; split_sizes.size() = {}, out.size() = {}",
        split_sizes.size(),
        out.size()
    );

    let mut sum: i64 = 0;
    for i in 0..split_sizes.size() {
        et_check_or_return_false!(
            *split_sizes.at(i) >= 0,
            "All split sizes must be non negative; split_sizes[{}] = {}",
            i,
            *split_sizes.at(i)
        );
        sum += *split_sizes.at(i);
    }

    let dim_size: ssize_t = in_.size(dim as ssize_t);
    et_check_or_return_false!(
        sum == dim_size as i64,
        "Sum of split sizes does not match input size at given dim; sum = {}, dim_size = {}",
        sum,
        dim_size
    );

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-split-with-sizes-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-split-with-sizes-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_split_with_sizes_copy_out_target_size(
    in_: &Tensor,
    split_size: i64,
    dim: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = in_.dim() as usize;

    for d in 0..in_.dim() {
        unsafe { *out_sizes.add(d as usize) = in_.size(d) as SizesType };
    }
    unsafe { *out_sizes.add(dim as usize) = split_size as SizesType };
}

// [spec:et:def:copy-ops-util.torch.executor.check-squeeze-copy-dim-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dim-args-fn]
pub fn check_squeeze_copy_dim_args(in_: &Tensor, dim: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-squeeze-copy-dim-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dim-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_squeeze_copy_dim_out_target_size(
    in_: &Tensor,
    dim: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    // For 0 dim tensors, the output should also be 0 dim.
    if in_.dim() == 0 {
        *out_ndim = 0;
        return;
    }

    // Specified dim is only removed if the size at the given dim is 1.
    if in_.size(dim as ssize_t) == 1 {
        *out_ndim = (in_.dim() - 1) as usize;
    } else {
        *out_ndim = in_.dim() as usize;
    }

    let mut out_d: usize = 0;
    for in_d in 0..in_.dim() {
        if in_d as i64 != dim || in_.size(in_d) != 1 {
            unsafe { *out_sizes.add(out_d) = in_.size(in_d) as SizesType };
            out_d += 1;
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-squeeze-copy-dims-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dims-args-fn]
pub fn check_squeeze_copy_dims_args(in_: &Tensor, dims: ArrayRef<i64>, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    for i in 0..dims.size() {
        let dim: i64 = if *dims.at(i) < 0 {
            *dims.at(i) + nonzero_dim(in_) as i64
        } else {
            *dims.at(i)
        };
        et_log_and_return_if_false!(tensor_has_dim(in_, dim));

        // Check that a dim does not appear twice in dims
        for j in 0..dims.size() {
            if i != j {
                let dim_temp: i64 = if *dims.at(j) < 0 {
                    *dims.at(j) + nonzero_dim(in_) as i64
                } else {
                    *dims.at(j)
                };
                et_check_or_return_false!(
                    dim != dim_temp,
                    "dim {} appears multiple times in dims!",
                    dim
                );
            }
        }
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-squeeze-copy-dims-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dims-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_squeeze_copy_dims_out_target_size(
    in_: &Tensor,
    dims: ArrayRef<i64>,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    // For 0 dim tensors, the output should also be 0 dim.
    if in_.dim() == 0 {
        *out_ndim = 0;
        return;
    }

    // A dim is only removed if the size at the given dim is 1.
    let mut dims_to_remove: SizesType = 0;
    for i in 0..dims.size() {
        let dim: i64 = if *dims.at(i) < 0 {
            *dims.at(i) + nonzero_dim(in_) as i64
        } else {
            *dims.at(i)
        };
        if in_.size(dim as ssize_t) == 1 {
            dims_to_remove += 1;
        }
    }
    *out_ndim = (in_.dim() - dims_to_remove as ssize_t) as usize;

    let mut out_d: usize = 0;
    for in_d in 0..in_.dim() {
        let mut in_d_in_dims: bool = false;
        for i in 0..dims.size() {
            let dim: i64 = if *dims.at(i) < 0 {
                *dims.at(i) + nonzero_dim(in_) as i64
            } else {
                *dims.at(i)
            };
            if in_d as i64 == dim {
                in_d_in_dims = true;
                break;
            }
        }
        if !in_d_in_dims || in_.size(in_d) != 1 {
            unsafe { *out_sizes.add(out_d) = in_.size(in_d) as SizesType };
            out_d += 1;
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-stack-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-stack-args-fn]
pub fn check_stack_args(tensors: ArrayRef<Tensor>, dim: i64, out: &Tensor) -> bool {
    // Ensure the input tensors list is non-empty
    et_log_and_return_if_false!(tensors.size() > 0);

    // All input tensors need to be of the same size
    for i in 0..tensors.size() {
        // All input dtypes must be castable to the output dtype.
        et_log_and_return_if_false!(can_cast(tensors.at(i).scalar_type(), out.scalar_type()));

        et_log_and_return_if_false!(tensor_is_rank(tensors.at(i), tensors.at(0).dim() as usize));
        for d in 0..tensors.at(i).dim() {
            et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                tensors.at(i),
                d as usize,
                tensors.at(0),
                d as usize
            ));
        }
    }

    // The output tensor will have a dimension inserted, so dim should be between
    // 0 and ndim_of_inputs + 1
    et_log_and_return_if_false!(dim >= 0 && dim < tensors.at(0).dim() as i64 + 1);

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-stack-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-stack-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `tensors[0].dim() + 1` writable elements.
pub unsafe fn get_stack_out_target_size(
    tensors: ArrayRef<Tensor>,
    dim: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = (tensors.at(0).dim() + 1) as usize;

    for d in 0..*out_ndim {
        let d_: i64 = d as i64;
        if d_ < dim {
            unsafe { *out_sizes.add(d_ as usize) = tensors.at(0).size(d_ as ssize_t) as SizesType };
        } else if d_ == dim {
            unsafe { *out_sizes.add(d_ as usize) = tensors.size() as SizesType };
        } else {
            unsafe {
                *out_sizes.add(d_ as usize) = tensors.at(0).size((d_ - 1) as ssize_t) as SizesType
            };
        }
    }
}

// [spec:et:def:copy-ops-util.torch.executor.check-tril-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-tril-args-fn]
pub fn check_tril_args(in_: &Tensor, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 2));
    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-split-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-split-copy-args-fn]
pub fn check_split_copy_args(input: &Tensor, split_size: i64, dim: i64, out: TensorList) -> bool {
    et_check_or_return_false!(
        input.dim() > 0,
        "input must have at least one dimension; saw {}",
        input.dim()
    );
    et_check_or_return_false!(
        dim >= 0 && dim < input.dim() as i64,
        "dim {} out of range [0,{})",
        dim,
        input.dim()
    );

    let dim_size: ssize_t = input.size(dim as ssize_t);
    et_check_or_return_false!(
        split_size >= 0,
        "split_size {} must be non-negative",
        split_size
    );
    et_check_or_return_false!(
        split_size > 0 || dim_size == 0,
        "split_size is zero but input.size({}) {} is non-zero",
        dim,
        dim_size
    );

    // Check the number of outputs.
    //
    // The specified dimension will be split into split_size-sized chunks, with
    // the final chunk possibly being smaller. So, the expected output length is
    // ceil(dim_size / split_size).
    let remainder: i64; // The size of the split dimension of the final out tensor.
    if split_size >= dim_size as i64 {
        // Note that this also handles the case where split_size == 0, avoiding a
        // division by zero in the other branch. When dim_size == 0 && split_size ==
        // 0, core PyTorch expects 1 output element.
        et_check_or_return_false!(
            out.size() == 1,
            "Unexpected out.size() {}: should be 1 because split_size {} >= input.size({}) {}",
            out.size(),
            split_size,
            dim,
            dim_size
        );
        remainder = dim_size as i64;
    } else {
        let expected_out_len: i64 = (dim_size as i64 + split_size - 1) / split_size;
        et_check_or_return_false!(
            out.size() as i64 == expected_out_len,
            "Unexpected out.size() {}: ceil(input.size({})={} / split_size={}) is {}",
            out.size(),
            dim,
            dim_size,
            split_size,
            expected_out_len
        );
        let mut remainder = dim_size as i64 % split_size;
        if remainder == 0 {
            remainder = split_size;
        }
        return validate_split_copy_outputs(input, split_size, dim, out, remainder);
    }

    validate_split_copy_outputs(input, split_size, dim, out, remainder)
}

// PORT-NOTE: the C++ `check_split_copy_args` performs the per-output validation
// loop inline after computing `remainder`. Rust's shadowing of `remainder` in
// the `else` branch (mirroring the C++ reassignment `remainder = split_size`)
// makes the loop run within that scope; extracted verbatim into this helper so
// both branches share one loop body without changing control flow.
fn validate_split_copy_outputs(
    input: &Tensor,
    split_size: i64,
    dim: i64,
    out: TensorList,
    remainder: i64,
) -> bool {
    for i in 0..out.size() {
        // All output dtypes must be the same.
        et_check_or_return_false!(
            out.at(i).scalar_type() == out.at(0).scalar_type(),
            "out[{}] dtype {} != out[0] dtype {}",
            i,
            out.at(i).scalar_type() as i8,
            out.at(0).scalar_type() as i8
        );

        // All outputs must have the same number of dimensions as the input.
        et_check_or_return_false!(
            out.at(i).dim() == input.dim(),
            "out[{}] dim {} != input dim {}",
            i,
            out.at(i).dim(),
            input.dim()
        );

        // Check the shape of the output.
        for d in 0..out.at(i).dim() {
            if d as i64 == dim {
                // This is the split dimension, which may be different.
                if i < out.size() - 1 {
                    // All outputs except the final one: split dimension should be
                    // split_size.
                    et_check_or_return_false!(
                        out.at(i).size(d) as i64 == split_size,
                        "out[{}].size({}) {} != split_size {}",
                        i,
                        d,
                        out.at(i).size(d),
                        split_size
                    );
                } else {
                    // The final output: split dimension should be the remainder of
                    // split_size.
                    et_check_or_return_false!(
                        out.at(i).size(d) as i64 == remainder,
                        "out[{}].size({}) {} != remainder {}",
                        i,
                        d,
                        out.at(i).size(d),
                        remainder
                    );
                }
            } else {
                // Non-split output dimensions must be the same as the input dimension.
                et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                    out.at(i),
                    d as usize,
                    input,
                    d as usize
                ));
            }
        }
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-to-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-to-copy-args-fn]
pub fn check_to_copy_args(
    input: &Tensor,
    non_blocking: bool,
    memory_format: Option<MemoryFormat>,
    out: &Tensor,
) -> bool {
    let _ = input;
    let _ = out;

    // Right now we only support blocking data transfer
    et_log_and_return_if_false!(non_blocking == false);

    // Right now we only focus on contiguous memory, memory_format shall be
    // exec::aten::MemoryFormat::Contiguous or none.
    et_log_and_return_if_false!(
        memory_format.is_none() || memory_format.unwrap() == MemoryFormat::Contiguous
    );

    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-to-dim-order-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-to-dim-order-copy-args-fn]
//
// PORT-NOTE: C++ `dim_order` is `executorch::aten::OptionalArrayRef<int64_t>`,
// which has no ported target yet. Modeled as `Option<ArrayRef<i64>>` (the
// non-owning optional-view the C++ type provides). Unresolved cross-module
// reference. Additionally the C++ calls `is_channels_last_dim_order` /
// `is_contiguous_dim_order` on the raw `int64_t*` dim-order data (an overload in
// dim_order_util.h). The ported `dim_order_util.rs` only exposes the
// `DimOrderType` (u8) pointer variant, so the int64 dim order is copied
// per-element into a `DimOrderType` stack buffer to reach the ported checker.
// Unresolved cross-module reference for the int64 overload.
pub fn check__to_dim_order_copy_args(
    input: &Tensor,
    non_blocking: bool,
    dim_order: Option<ArrayRef<i64>>,
    out: &Tensor,
) -> bool {
    // Right now we only support blocking data transfer
    et_log_and_return_if_false!(non_blocking == false);

    if dim_order.is_some() {
        let dim_order_ref: ArrayRef<i64> = dim_order.unwrap();

        // dim order size shall equal to input dim
        et_log_and_return_if_false!(dim_order_ref.size() as ssize_t == input.dim());

        let mut dim_order_bytes: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
            [0; K_TENSOR_DIMENSION_LIMIT];
        for i in 0..dim_order_ref.size() {
            dim_order_bytes[i] = *dim_order_ref.at(i) as DimOrderType;
        }
        et_log_and_return_if_false!(
            unsafe { is_channels_last_dim_order(dim_order_bytes.as_ptr(), dim_order_ref.size()) }
                || unsafe {
                    is_contiguous_dim_order(dim_order_bytes.as_ptr(), dim_order_ref.size())
                }
        );

        // Out tensor shall have same dim order as dim_order
        let out_dim_order = out.dim_order();
        et_log_and_return_if_false!(out_dim_order.size() == dim_order_ref.size());
        for i in 0..dim_order_ref.size() {
            et_log_and_return_if_false!(*out_dim_order.at(i) as i64 == *dim_order_ref.at(i));
        }
    } else {
        // dim_order is not set, preserve the dim order of input

        // Out tensor shall have same dim order as input dim_order
        let out_dim_order = out.dim_order();
        let input_dim_order = input.dim_order();
        et_log_and_return_if_false!(out_dim_order.size() == input_dim_order.size());
        for i in 0..input_dim_order.size() {
            et_log_and_return_if_false!(*out_dim_order.at(i) == *input_dim_order.at(i));
        }
    }
    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-unsqueeze-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-unsqueeze-copy-args-fn]
pub fn check_unsqueeze_copy_args(input: &Tensor, dim: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(dim >= 0);

    // The input and out shall share same dtype
    et_log_and_return_if_false!(tensors_have_same_dtype2(input, out));

    et_log_and_return_if_false!(tensor_has_dim(out, dim));

    // The shape of input and out shall obey the relationship:
    // 1. input.dim() == out.dim()-1
    // 2. input.size(i) == out.size(i) for all i < dim
    // 3. input.size(i-1) == out.size(i) for all i >= dim
    // 4. out.size(dim) == 1
    et_log_and_return_if_false!(input.dim() == out.dim() - 1);

    for d in 0..out.dim() {
        let mut dim_normalized: i64 = dim;
        if dim_normalized < 0 {
            dim_normalized += out.dim() as i64;
        }

        if (d as i64) < dim_normalized {
            et_check_or_return_false!(
                input.size(d) == out.size(d),
                "input.size({}) {} != out.size({}) {} | dim = {}",
                d,
                input.size(d),
                d,
                out.size(d),
                dim
            );
        } else if d as i64 > dim_normalized {
            et_check_or_return_false!(
                input.size(d - 1) == out.size(d),
                "input.size({}) {} != out.size({}) {} | dim = {}",
                d - 1,
                input.size(d),
                d,
                out.size(d),
                dim
            );
        } else {
            // d == dim
            et_check_or_return_false!(
                out.size(d) == 1,
                "out.size({}) {} shall equal 1 | dim = {}",
                d,
                out.size(d),
                dim
            );
        }
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-view-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-view-copy-args-fn]
pub fn check_view_copy_args(self_: &Tensor, size_int64_t: ArrayRef<i64>, out: &Tensor) -> bool {
    et_log_and_return_if_false!(size_int64_t.size() == out.sizes().size());

    // The input and out shall share same dtype and numel
    et_check_or_return_false!(
        self_.numel() == out.numel(),
        "self.numel() {} != out.numel() {}",
        self_.numel(),
        out.numel()
    );
    et_log_and_return_if_false!(tensors_have_same_dtype2(self_, out));

    // The size of out should equal target size.
    let mut size_inferred: bool = false;
    for i in 0..size_int64_t.size() {
        // If this value is -1 it implies that this dimension is inferred.
        if *size_int64_t.at(i) == -1 {
            et_check_or_return_false!(!size_inferred, "Multiple dimensions cannot be inferred.");
            size_inferred = true;
        }
        et_log_and_return_if_false!(
            (*out.sizes().at(i) as i64 == *size_int64_t.at(i)) || (*size_int64_t.at(i) == -1)
        );
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-view-copy-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-view-copy-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `dim` writable elements.
pub unsafe fn get_view_copy_target_size(
    input: &Tensor,
    size_int64_t: ArrayRef<i64>,
    dim: i64,
    out_sizes: *mut SizesType,
) -> bool {
    let mut out_numels_without_minus_1: usize = 1;
    let mut minus_1_dim: i32 = -1;

    et_log_and_return_if_false!(size_int64_t.size() as i64 == dim);

    for i in 0..dim {
        if *size_int64_t.at(i as usize) != -1 {
            unsafe { *out_sizes.add(i as usize) = *size_int64_t.at(i as usize) as SizesType };
            out_numels_without_minus_1 =
                out_numels_without_minus_1 * *size_int64_t.at(i as usize) as usize;
        } else {
            // TODO(kimishpatel): Add test to hit this line
            et_check_or_return_false!(minus_1_dim == -1, "At most one view copy dim can be -1.");
            minus_1_dim = i as i32;
        }
    }
    if minus_1_dim >= 0 {
        unsafe {
            *out_sizes.add(minus_1_dim as usize) =
                (input.numel() as usize / out_numels_without_minus_1) as SizesType
        };
    }

    true
}

// [spec:et:def:copy-ops-util.torch.executor.check-diagonal-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-diagonal-copy-args-fn]
pub fn check_diagonal_copy_args(in_: &Tensor, mut dim1: i64, mut dim2: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_rank_greater_or_equal_to(in_, 2));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim1));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim2));
    if dim1 < 0 {
        dim1 += nonzero_dim(in_) as i64;
    }
    if dim2 < 0 {
        dim2 += nonzero_dim(in_) as i64;
    }
    et_log_and_return_if_false!(dim1 != dim2);
    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-diagonal-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-diagonal-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim() - 1` writable elements.
pub unsafe fn get_diagonal_copy_out_target_size(
    in_: &Tensor,
    offset: i64,
    mut dim1: i64,
    mut dim2: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = (in_.dim() - 1) as usize;

    if dim1 < 0 {
        dim1 += nonzero_dim(in_) as i64;
    }
    if dim2 < 0 {
        dim2 += nonzero_dim(in_) as i64;
    }

    // PORT-NOTE: `diagonal_size` is initialized to 0 then overwritten in every
    // branch, exactly like the C++ `size_t diagonal_size = 0;`. Keep the dead
    // initializer for literal correspondence.
    #[allow(unused_assignments)]
    let mut diagonal_size: usize = 0;
    if offset >= 0 {
        if (in_.size(dim2 as ssize_t) as i64) <= offset {
            diagonal_size = 0;
        } else {
            diagonal_size = core::cmp::min(
                in_.size(dim1 as ssize_t) as usize,
                (in_.size(dim2 as ssize_t) as i64 - offset) as usize,
            );
        }
    } else {
        if (in_.size(dim1 as ssize_t) as i64) <= -offset {
            diagonal_size = 0;
        } else {
            diagonal_size = core::cmp::min(
                (in_.size(dim1 as ssize_t) as i64 + offset) as usize,
                in_.size(dim2 as ssize_t) as usize,
            );
        }
    }

    let mut shift: usize = 0;
    for d in 0..in_.dim() {
        if d as i64 == dim1 || d as i64 == dim2 {
            shift += 1;
        } else {
            unsafe { *out_sizes.add(d as usize - shift) = in_.size(d) as SizesType };
        }
    }
    unsafe { *out_sizes.add((in_.dim() - 2) as usize) = diagonal_size as SizesType };
}

// [spec:et:def:copy-ops-util.torch.executor.check-unfold-copy-args-fn]
// [spec:et:sem:copy-ops-util.torch.executor.check-unfold-copy-args-fn]
pub fn check_unfold_copy_args(self_: &Tensor, mut dim: i64, size: i64, step: i64) -> bool {
    if dim < 0 {
        dim += nonzero_dim(self_) as i64;
    }
    et_log_and_return_if_false!(tensor_has_dim(self_, dim));
    et_check_or_return_false!(size >= 0, "size is {} but must be >= 0", size);
    et_check_or_return_false!(
        size <= self_.size(dim as ssize_t) as i64,
        "maximum size for tensor at dimension {} is {} but size is {}",
        dim,
        self_.size(dim as ssize_t),
        size
    );
    et_check_or_return_false!(step > 0, "step is {} but must be > 0", step);
    true
}

// [spec:et:def:copy-ops-util.torch.executor.get-unfold-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-unfold-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `self.dim() + 1` writable elements.
pub unsafe fn get_unfold_copy_out_target_size(
    self_: &Tensor,
    dim: i64,
    size: i64,
    step: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    for i in 0..self_.dim() {
        unsafe { *out_sizes.add(i as usize) = self_.size(i) as SizesType };
    }
    // At `dim` dimension, we split the tensor into `size` chunks with `step`
    // stride.
    unsafe {
        *out_sizes.add(dim as usize) =
            ((self_.size(dim as ssize_t) as i64 - size + step) / step) as SizesType
    };

    unsafe { *out_sizes.add(self_.dim() as usize) = size as SizesType };
    *out_ndim = (self_.dim() + 1) as usize;
}

// [spec:et:def:copy-ops-util.torch.executor.get-view-as-real-copy-out-target-size-fn]
// [spec:et:sem:copy-ops-util.torch.executor.get-view-as-real-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `self.dim() + 1` writable elements.
pub unsafe fn get_view_as_real_copy_out_target_size(self_: &Tensor, out_sizes: *mut SizesType) {
    for i in 0..self_.dim() {
        unsafe { *out_sizes.add(i as usize) = self_.size(i) as SizesType };
    }
    unsafe { *out_sizes.add(self_.dim() as usize) = 2 };
}

// PORT-NOTE: header template `as_strided_copy<CTYPE>`; `CTYPE` is generic. The
// pointer offset mirrors the C++ `in.mutable_data_ptr<CTYPE>() + offset`.
pub fn as_strided_copy<CTYPE: Copy>(
    in_: &Tensor,
    size: ArrayRef<i64>,
    stride: ArrayRef<i64>,
    offset: i64,
    out: &Tensor,
) {
    let in_data: *mut CTYPE = unsafe { in_.mutable_data_ptr::<CTYPE>().offset(offset as isize) };
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    if size.empty() {
        unsafe { *out_data = *in_data };
    } else {
        _as_strided_copy::<CTYPE>(in_data, out_data, out, size, stride, 0);
    }
}

// [spec:et:def:copy-ops-util.torch.executor.to-dim-order-copy-impl-fn]
// [spec:et:sem:copy-ops-util.torch.executor.to-dim-order-copy-impl-fn]
//
// PORT-NOTE: drives `BroadcastIndexesRange` with
// `support_noncontiguous_input_tensors=true`. The C++ comment writes
// `BroadcastIndexesRange<2, ...>::new(self_, {self_, out})`, but that passes two
// inputs, so the ported (non-variadic) range is instantiated with NT=3 (dummy
// output + two inputs), matching the complex-conversion call sites in
// op__to_dim_order_copy.rs; the yielded index tuple is `[unused, self_index,
// out_index]`.
#[allow(non_camel_case_types)]
pub fn _to_dim_order_copy_impl<'a, SELF_CTYPE: Copy, OUT_CTYPE: Copy>(
    self_: &'a Tensor<'a>,
    out: &'a Tensor<'a>,
) where
    SELF_CTYPE: crate::extension::tensor::tensor_ptr::NumericCast<OUT_CTYPE>,
{
    let self_data = self_.mutable_data_ptr::<SELF_CTYPE>();
    let out_data = out.mutable_data_ptr::<OUT_CTYPE>();

    // Here we make a slightly off-label use of BroadcastIndexesRange. It always
    // assumes it doesn't have to care about different dim_order between input and
    // output, but we can just force it to respect strides (and thus dim_order)
    // for its inputs using support_noncontiguous_input_tensors=true, and then
    // pretend the output is just another input.
    for indexes in crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesRange::<3>::new_with_support(
        self_,
        &[self_, out],
        true,
    ) {
        let self_data_index = indexes[1];
        let out_data_index = indexes[2];
        unsafe {
            *out_data.offset(out_data_index) =
                <SELF_CTYPE as crate::extension::tensor::tensor_ptr::NumericCast<OUT_CTYPE>>::numeric_cast(
                    *self_data.offset(self_data_index),
                );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    // PORT-NOTE: `get_split_with_sizes_copy_out_target_size` has no call site in
    // the ported runtime (op_split_with_sizes_copy computes the per-chunk output
    // shape inline, mirroring the C++ op body), so no op test transitively covers
    // it. It is a pure size helper; this focused test pins its C++ semantics: copy
    // every input size into out_sizes, then overwrite the split dim with
    // `split_size` and set out_ndim = in.dim().
    // [spec:et:sem:copy-ops-util.torch.executor.get-split-with-sizes-copy-out-target-size-fn/test]
    #[test]
    fn copy_ops_util_test_get_split_with_sizes_copy_out_target_size() {
        let tf = TensorFactory::<i32>::new();
        let in_ = tf.zeros_default(vec![4, 6, 8]);

        let mut out_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let mut out_ndim: usize = 0;

        unsafe {
            get_split_with_sizes_copy_out_target_size(
                &in_,
                3,
                1,
                out_sizes.as_mut_ptr(),
                &mut out_ndim,
            );
        }

        assert_eq!(out_ndim, 3);
        assert_eq!(&out_sizes[0..3], &[4, 3, 8]);
    }
}
