//! Literal port of kernels/portable/cpu/util/advanced_index_util.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::linearize_access_indexes_tensor;
use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index_tensor;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::core::portable_type::tensor_impl::ssize_t;

// PORT-NOTE: `using TensorOptList = ArrayRef<std::optional<Tensor>>`. The inner
// `Tensor` carries a lifetime; the list is an `ArrayRef` of `Option<Tensor<'a>>`.
pub type TensorOptList<'a> = ArrayRef<Option<Tensor<'a>>>;

// PORT-NOTE: the crate-level `et_check_or_return_false!` drops all
// caller-supplied format arguments after the leading literal (see the analogous
// note in tensor_util.rs). This local macro mirrors the C++
// `ET_CHECK_OR_RETURN_FALSE` faithfully. Unresolved cross-module reference.
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

// PORT-NOTE: mirrors the file-local `c_string_data` helper used elsewhere
// (tensor_impl.rs, activation_ops_util.rs) to render a NUL-terminated buffer
// for logging.
fn c_string_data<const N: usize>(buf: &[core::ffi::c_char; N]) -> &str {
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    core::str::from_utf8(&bytes[..end]).unwrap_or("")
}

// [spec:et:def:advanced-index-util.torch.executor.check-indices-dtypes-fn]
// [spec:et:sem:advanced-index-util.torch.executor.check-indices-dtypes-fn]
fn check_indices_dtypes(indices: TensorOptList) -> bool {
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            let ix_type: ScalarType = index.scalar_type();
            et_check_or_return_false!(
                ix_type == ScalarType::Long
                    || ix_type == ScalarType::Int
                    || ix_type == ScalarType::Byte
                    || ix_type == ScalarType::Bool,
                "Index tensors should be Long, Int, Byte or Bool; got {}",
                ix_type as i32
            );
        }
    }
    true
}

// [spec:et:def:advanced-index-util.torch.executor.is-mask-index-fn]
// [spec:et:sem:advanced-index-util.torch.executor.is-mask-index-fn]
fn is_mask_index(index: &Tensor) -> bool {
    if index.scalar_type() == ScalarType::Bool || index.scalar_type() == ScalarType::Byte {
        return true;
    }
    false
}

// [spec:et:def:advanced-index-util.torch.executor.check-mask-indices-fn]
// [spec:et:sem:advanced-index-util.torch.executor.check-mask-indices-fn]
fn check_mask_indices(in_: &Tensor, indices: TensorOptList) -> bool {
    let mut in_i: usize = 0;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            if is_mask_index(index) {
                et_check_or_return_false!(
                    index.dim() > 0,
                    "Zero-dimensional mask index not allowed"
                );
                for j in 0..index.dim() {
                    if index.size(j) != in_.size(in_i as ssize_t + j) {
                        if crate::runtime::platform::log::ET_LOG_ENABLED {
                            let mask_shape =
                                crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                                    crate::runtime::core::span::Span::<TensorSizesType>::from_raw_parts(
                                        index.sizes().data() as *mut TensorSizesType,
                                        index.sizes().size(),
                                    ),
                                );
                            let input_shape =
                                crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                                    crate::runtime::core::span::Span::<TensorSizesType>::from_raw_parts(
                                        unsafe {
                                            in_.sizes().data().add(in_i) as *mut TensorSizesType
                                        },
                                        index.sizes().size(),
                                    ),
                                );
                            crate::et_log!(
                                Error,
                                "The shape of mask index {} must match the sizes of the corresponding input dimensions {}.",
                                c_string_data(&mask_shape),
                                c_string_data(&input_shape)
                            );
                        }
                        return false;
                    }
                }
                in_i += index.dim() as usize;
            } else {
                in_i += 1;
            }
        } else {
            in_i += 1;
        }
    }
    true
}

// PORT-NOTE: the C++ `_count_trues_in_mask_index<CTYPE_IX>` template is over the
// mask element type (`bool` / `uint8_t`), which share the same 1-byte layout;
// both are read as raw bytes with a nonzero test, matching the C++ truthiness
// check. Ported as a single monomorphic helper over `u8`.
// [spec:et:def:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]
// [spec:et:sem:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]
fn count_trues_in_mask_index_impl(index: &Tensor) -> usize {
    let index_ptr: *const u8 = index.const_data_ptr::<u8>();
    let mut sum: usize = 0;
    for i in 0..index.numel() {
        if unsafe { *index_ptr.add(i as usize) } != 0 {
            sum += 1;
        }
    }
    sum
}

pub fn count_trues_in_mask_index(index: &Tensor) -> usize {
    if index.scalar_type() == ScalarType::Bool {
        count_trues_in_mask_index_impl(index)
    } else {
        count_trues_in_mask_index_impl(index)
    }
}

// PORT-NOTE: `_query_mask_index<CTYPE_IX>` over `bool` / `uint8_t`; the two
// instantiations share byte layout and a nonzero truthiness test, so ported as
// a single monomorphic helper over `u8`.
// [spec:et:def:advanced-index-util.torch.executor.query-mask-index-fn]
// [spec:et:sem:advanced-index-util.torch.executor.query-mask-index-fn]
///
/// # Safety
/// `res` must point to at least `kTensorDimensionLimit` valid `usize` elements.
unsafe fn query_mask_index_impl(index: &Tensor, mut query_idx: usize, res: *mut usize) {
    let index_ptr: *const u8 = index.const_data_ptr::<u8>();
    // Broadcasting for mask index tensors
    let num_true: usize = count_trues_in_mask_index_impl(index);
    if num_true == 1 {
        query_idx = 0;
    }
    // Extract the index value by finding the idx-th element that is set to
    // true.
    let mut count: usize = 0;
    let mut flat_ix: usize = 0;
    for i in 0..index.numel() {
        if unsafe { *index_ptr.add(i as usize) } != 0 {
            if count == query_idx {
                flat_ix = i as usize;
                break;
            } else {
                count += 1;
            }
        }
    }
    delinearize_index_tensor(flat_ix, index, res, K_TENSOR_DIMENSION_LIMIT);
}

/// # Safety
/// `res` must point to at least `kTensorDimensionLimit` valid `usize` elements.
pub unsafe fn query_mask_index(index: &Tensor, query_idx: usize, res: *mut usize) {
    if index.scalar_type() == ScalarType::Bool {
        unsafe { query_mask_index_impl(index, query_idx, res) };
    } else {
        unsafe { query_mask_index_impl(index, query_idx, res) };
    }
}

// [spec:et:def:advanced-index-util.torch.executor.query-integral-index-fn]
// [spec:et:sem:advanced-index-util.torch.executor.query-integral-index-fn]
///
/// # Safety
/// `ix_coord` must point to at least `broadcast_ndim` valid `usize` elements.
pub unsafe fn query_integral_index(
    index: &Tensor,
    ix_coord: *mut usize,
    broadcast_ndim: usize,
) -> i64 {
    let flat_ix: usize = linearize_access_indexes_tensor(
        ArrayRef::from_raw_parts(ix_coord, broadcast_ndim),
        broadcast_ndim as ssize_t,
        index,
    );

    let idx_type: ScalarType = index.scalar_type();
    let index_val: i64;
    // Extract the index value
    if idx_type == ScalarType::Int {
        let index_ptr: *const i32 = index.const_data_ptr::<i32>();
        index_val = unsafe { *index_ptr.add(flat_ix) } as i64;
    } else {
        let index_ptr: *const i64 = index.const_data_ptr::<i64>();
        index_val = unsafe { *index_ptr.add(flat_ix) };
    }
    index_val
}

// [spec:et:def:advanced-index-util.torch.executor.check-index-args-fn]
// [spec:et:sem:advanced-index-util.torch.executor.check-index-args-fn]
pub fn check_index_args(in_: &Tensor, indices: TensorOptList, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(check_indices_dtypes(indices));
    et_check_or_return_false!(
        indices.size() as ssize_t <= in_.dim(),
        "Indexing too many dimensions"
    );
    et_log_and_return_if_false!(check_mask_indices(in_, indices));
    true
}

// [spec:et:def:advanced-index-util.torch.executor.count-index-blocks-fn]
// [spec:et:sem:advanced-index-util.torch.executor.count-index-blocks-fn]
pub fn count_index_blocks(indices: TensorOptList) -> usize {
    let mut block_count: usize = 0;
    let mut in_block: bool = false;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            if !in_block {
                in_block = true;
                block_count += 1;
            }
        } else {
            in_block = false;
        }
    }
    block_count
}

// [spec:et:def:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn]
///
/// # Safety
/// `ix_sizes` must point to at least `kTensorDimensionLimit` valid
/// `TensorSizesType` elements and `ix_ndim` to a valid `usize`.
pub unsafe fn get_indices_broadcast_shape(
    indices: TensorOptList,
    ix_sizes: *mut TensorSizesType,
    ix_ndim: *mut usize,
) -> bool {
    // Holds the (reversed) broadcasted shape of the indices.
    let mut rev_ix_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut curr_ndim: ssize_t = 0;

    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            if is_mask_index(index) {
                let len: TensorSizesType = count_trues_in_mask_index(index) as TensorSizesType;
                if curr_ndim == 0 {
                    curr_ndim = 1;
                    rev_ix_sizes[0] = len;
                } else if rev_ix_sizes[0] == 1 {
                    rev_ix_sizes[0] = len;
                } else if len != 1 && rev_ix_sizes[0] != len {
                    et_check_or_return_false!(false, "Broadcast of mask index failed.");
                }
            } else {
                for j in 0..index.dim() {
                    let rev_j_size: TensorSizesType =
                        index.size(index.dim() - j - 1) as TensorSizesType;
                    if j >= curr_ndim {
                        curr_ndim = j + 1;
                        rev_ix_sizes[j as usize] = rev_j_size;
                    } else if rev_ix_sizes[j as usize] == 1 {
                        rev_ix_sizes[j as usize] = rev_j_size;
                    } else if rev_j_size != 1 && rev_ix_sizes[j as usize] != rev_j_size {
                        et_check_or_return_false!(false, "Broadcast of index failed.");
                    }
                }
            }
        }
    }

    for i in 0..curr_ndim {
        unsafe {
            *ix_sizes.add(i as usize) = rev_ix_sizes[(curr_ndim - i - 1) as usize];
        }
    }
    unsafe {
        *ix_ndim = curr_ndim as usize;
    }
    true
}

// [spec:et:def:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn]
pub fn get_indices_broadcast_ndim(indices: TensorOptList) -> usize {
    let mut ndim: ssize_t = 0;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            if is_mask_index(index) {
                if ndim == 0 {
                    ndim = 1;
                }
            } else {
                if ndim < index.dim() {
                    ndim = index.dim();
                }
            }
        }
    }
    ndim as usize
}

// [spec:et:def:advanced-index-util.torch.executor.get-num-indexed-dims-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-num-indexed-dims-fn]
pub fn get_num_indexed_dims(indices: TensorOptList) -> usize {
    let mut num_indexed_dims: usize = 0;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            if is_mask_index(index) {
                num_indexed_dims += index.dim() as usize;
            } else {
                num_indexed_dims += 1;
            }
        }
    }
    num_indexed_dims
}

// [spec:et:def:advanced-index-util.torch.executor.get-num-null-indices-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-num-null-indices-fn]
pub fn get_num_null_indices(indices: TensorOptList) -> usize {
    let mut num_null_indices: usize = 0;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_none() {
            num_null_indices += 1;
        }
    }
    num_null_indices
}

// [spec:et:def:advanced-index-util.torch.executor.get-num-leading-null-indices-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-num-leading-null-indices-fn]
pub fn get_num_leading_null_indices(indices: TensorOptList) -> usize {
    let mut start: usize = 0;
    while unsafe { indices.index(start) }.is_none() {
        start += 1;
    }
    start
}

// [spec:et:def:advanced-index-util.torch.executor.get-index-out-target-size-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-index-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `kTensorDimensionLimit` valid
/// `TensorSizesType` elements and `out_ndim` to a valid `usize`.
pub unsafe fn get_index_out_target_size(
    in_: &Tensor,
    indices: TensorOptList,
    adjacent: bool,
    out_sizes: *mut TensorSizesType,
    out_ndim: *mut usize,
) -> bool {
    let mut broadcast_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut broadcast_ndim: usize = 0;
    if !unsafe {
        get_indices_broadcast_shape(indices, broadcast_sizes.as_mut_ptr(), &mut broadcast_ndim)
    } {
        return false;
    }

    let num_null_indices: usize = get_num_null_indices(indices);
    let num_indexed_dims: usize = get_num_indexed_dims(indices);

    et_check_or_return_false!(
        (num_null_indices + num_indexed_dims) as ssize_t <= in_.dim(),
        "Indexing too many dimensions; num_null_indices = {}, num_indexed_dims = {}, in.dim() = {}",
        num_null_indices,
        num_indexed_dims,
        in_.dim()
    );

    et_check_or_return_false!(
        in_.dim() as usize + broadcast_ndim - num_indexed_dims <= K_TENSOR_DIMENSION_LIMIT,
        "Out tensor would exceed number of allowed dimensions; in.dim() = {}, broadcast_ndim = {}, num_indexed_dims = {}, kTensorDimensionLimit = {}",
        in_.dim(),
        broadcast_ndim,
        num_indexed_dims,
        K_TENSOR_DIMENSION_LIMIT
    );

    unsafe {
        *out_ndim = in_.dim() as usize + broadcast_ndim - num_indexed_dims;
    }

    if adjacent {
        let start: usize = get_num_leading_null_indices(indices);
        for i in 0..start {
            unsafe {
                *out_sizes.add(i) = in_.size(i as ssize_t) as TensorSizesType;
            }
        }
        for i in 0..broadcast_ndim {
            unsafe {
                *out_sizes.add(i + start) = broadcast_sizes[i];
            }
        }
        for i in (num_indexed_dims + start)..(in_.dim() as usize) {
            unsafe {
                *out_sizes.add(i + broadcast_ndim - num_indexed_dims) =
                    in_.size(i as ssize_t) as TensorSizesType;
            }
        }
    } else {
        for i in 0..broadcast_ndim {
            unsafe {
                *out_sizes.add(i) = broadcast_sizes[i];
            }
        }
        let mut in_i: usize = 0;
        let mut out_i: usize = broadcast_ndim;
        for i in 0..indices.size() {
            if unsafe { indices.index(i) }.is_none() {
                unsafe {
                    *out_sizes.add(out_i) = in_.size(in_i as ssize_t) as TensorSizesType;
                }
                out_i += 1;
                in_i += 1;
            } else {
                let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
                if is_mask_index(index) {
                    in_i += index.dim() as usize;
                } else {
                    in_i += 1;
                }
            }
        }
        for i in (num_indexed_dims + num_null_indices)..(in_.dim() as usize) {
            unsafe {
                *out_sizes.add(i + broadcast_ndim - num_indexed_dims) =
                    in_.size(i as ssize_t) as TensorSizesType;
            }
        }
    }
    true
}

// dim_map maps non-indexed input dimensions to the corresponding output
// dimensions. Indexed dimensions are mapped to -1.
// [spec:et:def:advanced-index-util.torch.executor.compute-dim-map-fn]
// [spec:et:sem:advanced-index-util.torch.executor.compute-dim-map-fn]
///
/// # Safety
/// `dim_map` must point to at least `in.dim()` valid `i32` elements.
pub unsafe fn compute_dim_map(
    in_: &Tensor,
    indices: TensorOptList,
    dim_map: *mut i32,
    adjacent: bool,
) {
    let broadcast_ndim: usize = get_indices_broadcast_ndim(indices);
    let start: usize = get_num_leading_null_indices(indices);
    let num_indexed_dims: usize = get_num_indexed_dims(indices);
    let num_null_indices: usize = get_num_null_indices(indices);

    if adjacent {
        for i in 0..start {
            unsafe {
                *dim_map.add(i) = i as i32;
            }
        }
        for i in start..(start + num_indexed_dims) {
            unsafe {
                *dim_map.add(i) = -1;
            }
        }
        for i in (start + num_indexed_dims)..(in_.dim() as usize) {
            unsafe {
                *dim_map.add(i) = (i - num_indexed_dims + broadcast_ndim) as i32;
            }
        }
    } else {
        let mut in_i: usize = 0;
        let mut out_i: usize = broadcast_ndim;
        for i in 0..indices.size() {
            if unsafe { indices.index(i) }.is_none() {
                unsafe {
                    *dim_map.add(in_i) = out_i as i32;
                }
                in_i += 1;
                out_i += 1;
            } else {
                let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
                if is_mask_index(index) {
                    for _j in 0..index.dim() {
                        unsafe {
                            *dim_map.add(in_i) = -1;
                        }
                        in_i += 1;
                    }
                } else {
                    unsafe {
                        *dim_map.add(in_i) = -1;
                    }
                    in_i += 1;
                }
            }
        }
        for i in (num_indexed_dims + num_null_indices)..(in_.dim() as usize) {
            unsafe {
                *dim_map.add(i) = (i - num_indexed_dims + broadcast_ndim) as i32;
            }
        }
    }
}

// ix_map maps indexed input dimensions to the corresponding index.
// Non-indexed dimensions are mapped to -1.
// [spec:et:def:advanced-index-util.torch.executor.compute-index-map-fn]
// [spec:et:sem:advanced-index-util.torch.executor.compute-index-map-fn]
///
/// # Safety
/// `ix_map` must point to at least `in.dim()` valid `i32` elements.
pub unsafe fn compute_index_map(in_: &Tensor, indices: TensorOptList, ix_map: *mut i32) {
    for i in 0..in_.dim() {
        unsafe {
            *ix_map.add(i as usize) = -1;
        }
    }
    let mut in_i: usize = 0;
    for i in 0..indices.size() {
        if unsafe { indices.index(i) }.is_some() {
            let index: &Tensor = unsafe { indices.index(i) }.as_ref().unwrap();
            if is_mask_index(index) {
                for _j in 0..index.dim() {
                    unsafe {
                        *ix_map.add(in_i) = i as i32;
                    }
                    in_i += 1;
                }
            } else {
                unsafe {
                    *ix_map.add(in_i) = i as i32;
                }
                in_i += 1;
            }
        } else {
            in_i += 1;
        }
    }
}

// [spec:et:def:advanced-index-util.torch.executor.get-in-coord-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn]
///
/// # Safety
/// `dim_map` / `ix_map` must point to at least `in.dim()` valid `i32` elements;
/// `out_coord` must be a valid delinearized output coordinate; `in_coord` must
/// point to at least `in.dim()` valid `usize` elements.
pub unsafe fn get_in_coord(
    in_: &Tensor,
    indices: TensorOptList,
    start: usize,
    broadcast_ndim: usize,
    dim_map: *mut i32,
    ix_map: *mut i32,
    out_coord: *mut usize,
    in_coord: *mut usize,
) -> bool {
    let mut i: ssize_t = 0;
    while i < in_.dim() {
        if unsafe { *dim_map.add(i as usize) } >= 0 {
            unsafe {
                *in_coord.add(i as usize) = *out_coord.add(*dim_map.add(i as usize) as usize);
            }
        } else {
            let index: &Tensor = unsafe { indices.index(*ix_map.add(i as usize) as usize) }
                .as_ref()
                .unwrap();

            let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
            for j in 0..broadcast_ndim {
                ix_coord[j] = unsafe { *out_coord.add(j + start) };
            }

            if is_mask_index(index) {
                let query_ix: usize = ix_coord[broadcast_ndim - 1];
                let mut query_result: [usize; K_TENSOR_DIMENSION_LIMIT] =
                    [0; K_TENSOR_DIMENSION_LIMIT];
                unsafe { query_mask_index(index, query_ix, query_result.as_mut_ptr()) };
                for j in 0..index.dim() {
                    unsafe {
                        *in_coord.add((i + j) as usize) = query_result[j as usize];
                    }
                }
                i += index.dim() - 1;
            } else {
                let mut index_val: i64 =
                    unsafe { query_integral_index(index, ix_coord.as_mut_ptr(), broadcast_ndim) };
                if index_val < 0 {
                    index_val += in_.size(i) as i64;
                }
                et_check_or_return_false!(
                    index_val >= 0 && index_val < in_.size(i) as i64,
                    "Index {} is out of bounds for input dimension {} with size {}.",
                    index_val,
                    i,
                    in_.size(i)
                );
                unsafe {
                    *in_coord.add(i as usize) = index_val as usize;
                }
            }
        }
        i += 1;
    }
    true
}

// [spec:et:def:advanced-index-util.torch.executor.get-in-ix-fn]
// [spec:et:sem:advanced-index-util.torch.executor.get-in-ix-fn]
///
/// # Safety
/// `dim_map` / `ix_map` must point to at least `in.dim()` valid `i32` elements.
pub unsafe fn get_in_ix(
    in_: &Tensor,
    indices: TensorOptList,
    out: &Tensor,
    out_ix: usize,
    start: usize,
    broadcast_ndim: usize,
    dim_map: *mut i32,
    ix_map: *mut i32,
) -> (usize, bool) {
    let mut out_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    delinearize_index_tensor(
        out_ix,
        out,
        out_coord.as_mut_ptr(),
        K_TENSOR_DIMENSION_LIMIT,
    );

    let mut in_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let success: bool = unsafe {
        get_in_coord(
            in_,
            indices,
            start,
            broadcast_ndim,
            dim_map,
            ix_map,
            out_coord.as_mut_ptr(),
            in_coord.as_mut_ptr(),
        )
    };
    if !success {
        return (0, false);
    }
    (unsafe { coordinateToIndex(in_, in_coord.as_ptr()) }, true)
}
