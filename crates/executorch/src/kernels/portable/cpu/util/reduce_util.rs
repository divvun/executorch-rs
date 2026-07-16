//! Literal port of kernels/portable/cpu/util/reduce_util.cpp + kernels/portable/cpu/util/reduce_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{is_floating_type, is_integral_type};
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, dim_is_valid, getLeadingDims, resize_tensor,
    tensor_has_non_empty_dim, tensor_is_default_or_channels_last_dim_order,
    tensor_is_floating_type, tensors_have_same_dtype2, tensors_have_same_shape2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::core::portable_type::tensor_impl::ssize_t;

// PORT-NOTE: `ET_CHECK` / `ET_CHECK_MSG` are the C++ fatal checks; mirrored with
// a local abort on failure (message dropped since a fatal abort follows),
// matching the established pattern in tensor_util.rs.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// #define ET_CHECK_VALID_DIM(dim, upper_bound) ...
// PORT-NOTE: `ET_CHECK_VALID_DIM(dim, upper_bound)` fatally checks that `dim` is
// in `[-upper_bound, upper_bound)`. Mirrored via `et_check!` over the same range
// predicate (fatal abort on failure).
macro_rules! et_check_valid_dim {
    ($dim:expr, $upper_bound:expr) => {
        et_check!($dim >= -($upper_bound) && $dim < ($upper_bound))
    };
}

// #define ET_NORMALIZE_IX(IX, UPPER_BOUND) IX < 0 ? IX + UPPER_BOUND : IX
macro_rules! et_normalize_ix {
    ($ix:expr, $upper_bound:expr) => {
        if $ix < 0 { $ix + $upper_bound } else { $ix }
    };
}

// PORT-NOTE: the crate-level `et_check_or_return_false!` drops all
// caller-supplied format arguments after the leading literal; this local macro
// mirrors the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully. Only used by the
// `#[cfg(not(feature = "aten"))]` check_* helpers.
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

//
// Anonymous-namespace iteration helpers.
//

// [spec:et:def:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn]
// [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn]
fn apply_on_flat_ix_with_stride_and_base<Fn: FnMut(usize)>(
    mut fn_: Fn,
    stride: usize,
    base: usize,
    start: usize,
    end: usize,
) {
    let mut i: usize = start;
    while i <= end {
        fn_(base + i * stride);
        i += 1;
    }
}

// [spec:et:def:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]
// [spec:et:sem:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]
fn apply_on_flat_and_dim_ix_with_stride_and_base<Fn: FnMut(usize, usize)>(
    mut fn_: Fn,
    stride: usize,
    base: usize,
    start: usize,
    end: usize,
) {
    let mut i: usize = start;
    while i <= end {
        fn_(base + i * stride, i);
        i += 1;
    }
}

// [spec:et:def:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn]
// [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn]
///
/// # Safety
/// `dim_mask` must point to at least `in.dim()` valid `bool` elements.
unsafe fn apply_on_flat_ix_with_dim_mask_and_base<Fn: FnMut(usize)>(
    mut fn_: Fn,
    in_: &Tensor,
    dim_mask: *const bool,
    base: usize,
    start: usize,
    end: usize,
) {
    // Compute innermost dim from dim list
    let mut inner_dim: ssize_t = in_.dim() - 1;
    while !unsafe { *dim_mask.add(inner_dim as usize) } {
        inner_dim -= 1;
    }

    // Initialize array of indices per dimension. This array is used to maintain
    // the per-dimension index of the element in `in` that is being reduced over
    // Only the dims that are in the dim list are relevant.
    let mut dim_index: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    for d in 0..in_.dim() {
        dim_index[d as usize] = 0;
    }

    // Gather strides
    let strides = in_.strides();

    // curr_index will always be index of the element from `in` we are currently
    // reducing. Initialized to the first index from `in` that maps to `out_ix`
    let mut curr_index: usize = base;

    let mut apply_fun_counter: usize = 0;
    loop {
        // Apply reduction to current index
        if apply_fun_counter >= start && apply_fun_counter <= end {
            fn_(curr_index);
        }
        apply_fun_counter += 1;
        if apply_fun_counter > end {
            return;
        }

        // Next index to reduce. Increase dim_index[inner_dim] by 1, and
        // curr_index by strides[inner_dim].
        dim_index[inner_dim as usize] += 1;
        curr_index += *strides.at(inner_dim as usize) as usize;

        // Check if we have reached the end of the innermost dimension
        if dim_index[inner_dim as usize] == in_.size(inner_dim) as i64 {
            // If we reached the end, we need to update the indices in dim_index.
            // We do this by resetting dim_index[inner_dim] to 0, and then
            // incrementing the index of the next innermost dimension from the
            // dim list by 1. If when we do this increment, we also reach the end
            // of that dimension, we need to keep repeating that procedure.
            // This is similar to doing the carry over when adding 1 to a number.

            // curr_dim will be the dim from the dim list we are currently
            // updating
            let mut curr_dim: ssize_t = inner_dim;

            while dim_index[curr_dim as usize] == in_.size(curr_dim) as i64 {
                if curr_dim == 0 {
                    // Exit function if we've reached the end of the outermost
                    // dimension
                    return;
                }
                // Reset dim_index[curr_dim] to 0. We need to update curr_index
                // accordingly. Reseting dim_index[curr_dim] from in.size(curr_dim)
                // to 0 means we need to subtract in.size(curr_dim) *
                // strides[curr_dim] from curr_index. However in.size(curr_dim) *
                // strides[curr_dim] is equal to strides[curr_dim - 1]. Notice
                // that curr_dim > 0 at this point in the execution
                dim_index[curr_dim as usize] = 0;
                curr_index -= *strides.at((curr_dim - 1) as usize) as usize;

                // Decrease current dim
                curr_dim -= 1;
                while curr_dim >= 0 {
                    // Stop if curr_dim is in the dim list
                    if unsafe { *dim_mask.add(curr_dim as usize) } {
                        break;
                    }
                    // Keep decreasing if curr_dim is not in the dim list
                    curr_dim -= 1;
                }
                // Exit function if curr_dim was decreased to -1. This means we
                // have reduced over all the elements we needed to.
                if curr_dim < 0 {
                    return;
                }

                // At this point in the execution, curr_dim is the next innermost
                // dimension. Increase dim_index[curr_dim] by 1 and update
                // curr_index accordingly.
                dim_index[curr_dim as usize] += 1;
                curr_index += *strides.at(curr_dim as usize) as usize;
            }
        }
    }
}

//
// Helper Functions
//

// Normalize the dimension by adding in_dim if d < 0; for 0-D, clamp to 0
// [spec:et:def:reduce-util.torch.executor.normalize-non-neg-d-fn]
// [spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn]
#[allow(non_snake_case)]
fn _normalize_non_neg_d(d: ssize_t, in_dim: ssize_t) -> usize {
    if in_dim == 0 && (d == 0 || d == -1) {
        return 0;
    }
    if d < 0 {
        return (d + in_dim) as usize;
    }
    d as usize
}

// [spec:et:def:reduce-util.torch.executor.check-dim-list-is-valid-fn]
// [spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn]
#[must_use]
pub fn check_dim_list_is_valid(in_: &Tensor, dim_list: &Option<ArrayRef<i64>>) -> bool {
    if let Some(reduce_dims) = dim_list {
        if reduce_dims.size() != 0 {
            let mut dim_exist: [bool; K_TENSOR_DIMENSION_LIMIT] = [false; K_TENSOR_DIMENSION_LIMIT];
            for k in 0..reduce_dims.size() {
                let d: i64 = *reduce_dims.at(k);
                if in_.dim() == 0 {
                    et_log_and_return_if_false!(d == 0 || d == -1);
                } else {
                    et_log_and_return_if_false!(dim_is_valid(d, in_.dim() as i64));
                }

                let non_neg_d: usize = _normalize_non_neg_d(d as ssize_t, in_.dim());
                et_log_and_return_if_false!(non_neg_d < K_TENSOR_DIMENSION_LIMIT);

                et_check_or_return_false!(
                    dim_exist[non_neg_d] == false,
                    "dim {} appears multiple times in the list of dims",
                    non_neg_d
                );
                dim_exist[non_neg_d] = true;
            }
        }
    }

    true
}

// [spec:et:def:reduce-util.torch.executor.check-dim-in-dim-list-fn]
// [spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn]
pub fn check_dim_in_dim_list(dim: usize, max_dim: usize, dim_list: &ArrayRef<i64>) -> bool {
    for k in 0..dim_list.size() {
        let d: i64 = *dim_list.at(k);
        let non_neg_dim: usize = _normalize_non_neg_d(d as ssize_t, max_dim as ssize_t);
        if dim == non_neg_dim {
            return true;
        }
    }
    false
}

/// Returns the product of the sizes of all reduction dims (single-`dim`
/// overload).
pub fn get_reduced_dim_product_dim(in_: &Tensor, dim: &Option<i64>) -> usize {
    if in_.dim() == 0 {
        return 1;
    }
    match dim {
        None => in_.numel() as usize,
        Some(dim_val) => {
            let d: usize = _normalize_non_neg_d(*dim_val as ssize_t, in_.dim());
            in_.size(d as ssize_t) as usize
        }
    }
}

/// Returns the product of the sizes of all reduction dims (`dim_list` overload).
// [spec:et:def:reduce-util.torch.executor.get-reduced-dim-product-fn]
// [spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]
pub fn get_reduced_dim_product(in_: &Tensor, dim_list: &Option<ArrayRef<i64>>) -> usize {
    if in_.dim() == 0 {
        return 1;
    }
    match dim_list {
        None => return in_.numel() as usize,
        Some(dl) if dl.size() == 0 => return in_.numel() as usize,
        Some(_) => {}
    }
    let mut dim_product: usize = 1;
    let dl = dim_list.as_ref().unwrap();
    for k in 0..dl.size() {
        let d: i64 = *dl.at(k);
        let non_neg_d: usize = _normalize_non_neg_d(d as ssize_t, in_.dim());
        dim_product *= in_.size(non_neg_d as ssize_t) as usize;
    }
    dim_product
}

/// Returns the number of elements of the output of reducing `in` over `dim`
/// (single-`dim` overload).
// [spec:et:def:reduce-util.torch.executor.get-out-numel-fn]
// [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn]
pub fn get_out_numel_dim(in_: &Tensor, dim: &Option<i64>) -> usize {
    let mut out_numel: usize = 1;
    if let Some(dim_val) = dim {
        let dim_val = *dim_val;
        if in_.dim() == 0 {
            et_check!(dim_val == 0 || dim_val == -1);
        } else {
            et_check_valid_dim!(dim_val, in_.dim() as i64);
        }
        let non_neg_dim: usize = _normalize_non_neg_d(dim_val as ssize_t, in_.dim());
        for d in 0..(in_.dim() as usize) {
            if d != non_neg_dim {
                out_numel *= in_.size(d as ssize_t) as usize;
            }
        }
    }
    out_numel
}

/// Returns the number of elements of the output of reducing `in` over
/// `dim_list`.
pub fn get_out_numel(in_: &Tensor, dim_list: &Option<ArrayRef<i64>>) -> usize {
    let mut out_numel: usize = 1;
    if let Some(dl) = dim_list {
        if dl.size() != 0 {
            for d in 0..(in_.dim() as usize) {
                if !check_dim_in_dim_list(d, in_.dim() as usize, dl) {
                    out_numel *= in_.size(d as ssize_t) as usize;
                }
            }
        }
    }
    out_numel
}

/// Returns the index of the first element in `in` that maps to `out_ix` when
/// reducing over `dim`. Single-`dim` overload.
// [spec:et:def:reduce-util.torch.executor.get-init-index-fn]
// [spec:et:sem:reduce-util.torch.executor.get-init-index-fn]
pub fn get_init_index_dim(in_: &Tensor, dim: &Option<i64>, out_ix: usize) -> usize {
    let dim_val = match dim {
        None => return 0,
        Some(v) => *v,
    };
    if in_.dim() == 0 {
        et_check!(dim_val == 0 || dim_val == -1);
    } else {
        et_check_valid_dim!(dim_val, in_.dim() as i64);
    }
    let non_neg_dim: usize = _normalize_non_neg_d(dim_val as ssize_t, in_.dim());
    let mut init_ix: usize = 0;
    let mut mutable_out_ix: usize = out_ix;
    let strides = in_.strides();
    let mut d: ssize_t = in_.dim() - 1;
    while d >= 0 {
        if d != non_neg_dim as ssize_t {
            init_ix += (mutable_out_ix % in_.size(d) as usize) * *strides.at(d as usize) as usize;
            mutable_out_ix /= in_.size(d) as usize;
        }
        d -= 1;
    }
    init_ix
}

/// Returns the index of the first element in `in` that maps to `out_ix` when
/// reducing over the list of dimensions in `dim_list`. `dim_list` overload.
pub fn get_init_index(in_: &Tensor, dim_list: &Option<ArrayRef<i64>>, out_ix: usize) -> usize {
    match dim_list {
        None => return 0,
        Some(dl) if dl.size() == 0 => return 0,
        Some(_) => {}
    }
    let dl = dim_list.as_ref().unwrap();
    let mut init_ix: usize = 0;
    let mut mutable_out_ix: usize = out_ix;
    let strides = in_.strides();
    let mut d: ssize_t = in_.dim() - 1;
    while d >= 0 {
        if !check_dim_in_dim_list(d as usize, in_.dim() as usize, dl) {
            init_ix += (mutable_out_ix % in_.size(d) as usize) * *strides.at(d as usize) as usize;
            mutable_out_ix /= in_.size(d) as usize;
        }
        d -= 1;
    }
    init_ix
}

//
// Iteration Functions
//

/// Reduce a tensor `in` over a given dimension `dim` using the reduce function
/// `fn`, with signature `fn(size, stride, base_ix)`. (3-arg overload.)
pub fn apply_over_dim_whole<Fn: FnMut(usize, usize, usize)>(
    mut fn_: Fn,
    in_: &Tensor,
    dim: &Option<i64>,
) {
    // If dim is null, apply fn over the entire tensor
    let dim_val = match dim {
        None => {
            fn_(in_.numel() as usize, 1, 0);
            return;
        }
        Some(v) => *v,
    };

    if in_.dim() != 0 {
        et_check_valid_dim!(dim_val, in_.dim() as i64);
    } else {
        // Special handling for 0-D tensor; 0 or -1 is valid for PyTorch code
        // `torch.mean(torch.tensor(2, dtype=float), dim=-1)`
        et_check!(dim_val == 0 || dim_val == -1);
        fn_(in_.numel() as usize, 1, 0);
        return;
    }

    if in_.numel() == 0 {
        return;
    }

    let d: usize = _normalize_non_neg_d(dim_val as ssize_t, in_.dim());

    let size: usize = in_.size(d as ssize_t) as usize;
    let stride: usize = *in_.strides().at(d) as usize;
    let outer_size: usize = getLeadingDims(in_, d as i64);
    let outer_stride: usize = size * stride;
    // Loop through all outer dimensions
    for outer_idx in 0..outer_size {
        let outer: usize = outer_idx * outer_stride;
        // Loop through all inner dimensions
        for inner_idx in 0..stride {
            let base: usize = outer + inner_idx;
            fn_(size, stride, base);
        }
    }
}

/// Reduce a tensor `in` over a given dimension `dim` for the output element at
/// index `out_ix` using the reduce function `fn`, with signature
/// `fn(in_ix, dim_ix)`. (6-arg overload with default `start = 0`, `end = -1`.)
// [spec:et:def:reduce-util.torch.executor.apply-over-dim-fn]
// [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn]
pub fn apply_over_dim<Fn: FnMut(usize, usize)>(
    fn_: Fn,
    in_: &Tensor,
    dim: &Option<i64>,
    out_ix: usize,
    start: i64,
    end: i64,
) {
    if let Some(dim_val) = dim {
        if in_.dim() != 0 {
            et_check_valid_dim!(*dim_val, in_.dim() as i64);
        } else {
            et_check!(*dim_val == 0 || *dim_val == -1);
        }
    }
    et_check_msg!(
        out_ix < get_out_numel_dim(in_, dim),
        "Out index {} is out of bounds",
        out_ix
    );

    if in_.numel() == 0 {
        return;
    }

    let iter_length: usize = get_reduced_dim_product_dim(in_, dim);
    let normalized_start: usize = et_normalize_ix!(start, iter_length as i64) as usize;
    let normalized_end: usize = et_normalize_ix!(end, iter_length as i64) as usize;
    let ustart: usize = core::cmp::max(normalized_start, 0usize);
    let uend: usize = core::cmp::min(normalized_end, iter_length - 1);

    // If dim is null, iterate over the entire tensor
    if dim.is_none() {
        apply_on_flat_and_dim_ix_with_stride_and_base(
            fn_, /*stride=*/ 1, /*base=*/ 0, ustart, uend,
        );
        return;
    }

    // Compute the starting base index
    let base: usize = get_init_index_dim(in_, dim, out_ix);

    // Compute non-negative dimension value from dim value
    let d: usize = _normalize_non_neg_d(dim.unwrap() as ssize_t, in_.dim());

    if in_.dim() == 0 {
        let mut fn_ = fn_;
        fn_(base, ustart);
    } else {
        apply_on_flat_and_dim_ix_with_stride_and_base(
            fn_,
            *in_.strides().at(d) as usize,
            base,
            ustart,
            uend,
        );
    }
}

/// Execution plan for repeated apply_over_dim_list with the same function,
/// input tensor, dim list, start, and end but varying out_ix.
// [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.execution-mode]
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExecutionMode {
    // Empty input, no work to do.
    NothingToDo,
    // Iterate over the entire tensor with apply_on_flat_ix_with_stride_and_base.
    NoDimMaskOrZeroDimension,
    // dim_list has size 1, iterate with
    // apply_on_flat_and_dim_ix_with_stride_and_base
    OnlyOneDim,
    // General mode, iterate with apply_on_flat_ix_with_dim_mask_and_base.
    NormalDimMask,
}

// [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan]
pub struct ApplyOverDimListPlan<'a> {
    // Start argument to apply_on_flat_ix_with_{stride,dim_mask}_and_base.
    ustart_: usize,
    // End argument to apply_on_flat_ix_with_{stride,dim_mask}_and_base.
    uend_: usize,
    mode_: ExecutionMode,
    out_numel_: usize,
    dim_list_: Option<ArrayRef<i64>>,
    is_in_dim_list_: [bool; K_TENSOR_DIMENSION_LIMIT],
    in_: &'a Tensor<'a>,
}

impl<'a> ApplyOverDimListPlan<'a> {
    // [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]
    pub fn new(
        in_: &'a Tensor<'a>,
        // If set, lifetime must last until execute() returns.
        dim_list: &Option<ArrayRef<i64>>,
        start: i64,
        end: i64,
    ) -> Self {
        let mut plan = ApplyOverDimListPlan {
            ustart_: 0,
            uend_: 0,
            mode_: ExecutionMode::NothingToDo,
            out_numel_: 0,
            dim_list_: *dim_list,
            is_in_dim_list_: [false; K_TENSOR_DIMENSION_LIMIT],
            in_,
        };
        et_check!(check_dim_list_is_valid(in_, dim_list));
        plan.out_numel_ = get_out_numel(plan.in_, dim_list);
        if in_.numel() == 0 {
            plan.mode_ = ExecutionMode::NothingToDo;
            return plan;
        }
        let iter_length: usize = get_reduced_dim_product(in_, dim_list);
        let normalized_start: usize = et_normalize_ix!(start, iter_length as i64) as usize;
        let normalized_end: usize = et_normalize_ix!(end, iter_length as i64) as usize;
        plan.ustart_ = core::cmp::max(normalized_start, 0usize);
        plan.uend_ = core::cmp::min(normalized_end, iter_length - 1);
        let has_nonempty_dim_list = match dim_list {
            Some(dl) => dl.size() != 0,
            None => false,
        };
        if !has_nonempty_dim_list || in_.dim() == 0 {
            plan.mode_ = ExecutionMode::NoDimMaskOrZeroDimension;
            return plan;
        }
        plan.dim_list_ = Some(*dim_list.as_ref().unwrap());
        if plan.dim_list_.as_ref().unwrap().size() == 1 {
            plan.mode_ = ExecutionMode::OnlyOneDim;
            return plan;
        }
        plan.is_in_dim_list_.fill(false);
        let dl = dim_list.as_ref().unwrap();
        for k in 0..dl.size() {
            let d: i64 = *dl.at(k);
            let non_neg_d: usize = if d < 0 {
                (d + in_.dim() as i64) as usize
            } else {
                d as usize
            };
            plan.is_in_dim_list_[non_neg_d] = true;
        }

        plan.mode_ = ExecutionMode::NormalDimMask;
        plan
    }

    // [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]
    pub fn execute<Fn: FnMut(usize)>(&self, fn_: Fn, out_ix: usize) {
        et_check_msg!(
            out_ix < self.out_numel_,
            "Out index {} is out of bounds",
            out_ix
        );

        match self.mode_ {
            ExecutionMode::NothingToDo => (),
            ExecutionMode::NoDimMaskOrZeroDimension => {
                apply_on_flat_ix_with_stride_and_base(
                    fn_,
                    /*stride=*/ 1,
                    /*base=*/ 0,
                    self.ustart_,
                    self.uend_,
                );
            }
            ExecutionMode::OnlyOneDim => {
                let dl = self.dim_list_.as_ref().unwrap();
                let mut fn_ = fn_;
                apply_on_flat_and_dim_ix_with_stride_and_base(
                    |in_ix, _dim_ix| fn_(in_ix),
                    *self
                        .in_
                        .strides()
                        .at(_normalize_non_neg_d(*dl.at(0) as ssize_t, self.in_.dim()))
                        as usize,
                    get_init_index(self.in_, &self.dim_list_, out_ix),
                    self.ustart_,
                    self.uend_,
                );
            }
            ExecutionMode::NormalDimMask => unsafe {
                apply_on_flat_ix_with_dim_mask_and_base(
                    fn_,
                    self.in_,
                    self.is_in_dim_list_.as_ptr(),
                    get_init_index(self.in_, &self.dim_list_, out_ix),
                    self.ustart_,
                    self.uend_,
                );
            },
        }
    }

    // [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.get-input-tensor-fn]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-input-tensor-fn]
    pub fn get_input_tensor(&self) -> &'a Tensor<'a> {
        self.in_
    }

    // [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.get-dim-list-fn]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-dim-list-fn]
    pub fn get_dim_list(&self) -> &Option<ArrayRef<i64>> {
        &self.dim_list_
    }
}

/// Reduce a tensor `in` over a given list of dimensions `dim_list` for the
/// output element at index `out_ix` using the reduce function `fn`, with
/// signature `fn(in_ix)`. (Default `start = 0`, `end = -1`.)
// [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-fn]
// [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn]
pub fn apply_over_dim_list<Fn: FnMut(usize)>(
    fn_: Fn,
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    out_ix: usize,
    start: i64,
    end: i64,
) {
    let plan = ApplyOverDimListPlan::new(in_, dim_list, start, end);
    plan.execute(fn_, out_ix);
}

//
// Reduce Functions
//

/// Reduce a tensor `in` over a dimension `dim` for the output element at index
/// `out_ix`, first applying `map_fun` (`CTYPE_IN -> CTYPE_OUT`) then reducing
/// with `reduce_fun` (`(CTYPE_OUT, i64, CTYPE_OUT, i64) -> (CTYPE_OUT, i64)`).
// [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-fn]
// [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn]
pub fn map_reduce_over_dim<CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp>(
    map_fun: MapOp,
    reduce_fun: ReduceOp,
    in_: &Tensor,
    dim: &Option<i64>,
    out_ix: usize,
) -> (CTYPE_OUT, i64)
where
    CTYPE_IN: Copy,
    CTYPE_OUT: Copy,
    MapOp: Fn(CTYPE_IN) -> CTYPE_OUT,
    ReduceOp: Fn(CTYPE_OUT, i64, CTYPE_OUT, i64) -> (CTYPE_OUT, i64),
{
    if let Some(dim_val) = dim {
        if in_.dim() != 0 {
            et_check_valid_dim!(*dim_val, in_.dim() as i64);
        } else {
            et_check!(*dim_val == 0 || *dim_val == -1);
        }
    }

    et_check_msg!(
        out_ix < get_out_numel_dim(in_, dim),
        "Out index {} is out of bounds",
        out_ix
    );

    et_check_msg!(in_.numel() > 0, "Input tensor must be nonempty");

    let init_index: usize = get_init_index_dim(in_, dim, out_ix);

    let in_data: *const CTYPE_IN = in_.const_data_ptr::<CTYPE_IN>();
    let mut acc_val: CTYPE_OUT = map_fun(unsafe { *in_data.add(init_index) });
    let mut acc_ix: i64 = 0;

    if in_.numel() == 1 {
        return (acc_val, acc_ix);
    }

    apply_over_dim(
        |in_ix: usize, dim_ix: usize| {
            let res: (CTYPE_OUT, i64) = reduce_fun(
                map_fun(unsafe { *in_data.add(in_ix) }),
                dim_ix as i64,
                acc_val,
                acc_ix,
            );
            acc_val = res.0;
            acc_ix = res.1;
        },
        in_,
        dim,
        out_ix,
        1,
        -1,
    );

    (acc_val, acc_ix)
}

/// Execution plan for repeated map_reduce_over_dim_list with the same function,
/// input tensor, and dim_list but varying out_ix.
// [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan]
pub struct MapReduceOverDimListPlan<'a> {
    plan_: ApplyOverDimListPlan<'a>,
}

impl<'a> MapReduceOverDimListPlan<'a> {
    // [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]
    pub fn new(in_: &'a Tensor<'a>, dim_list: &Option<ArrayRef<i64>>) -> Self {
        let plan = MapReduceOverDimListPlan {
            plan_: ApplyOverDimListPlan::new(in_, dim_list, 1, -1),
        };
        et_check_msg!(in_.numel() > 0, "Input tensor must be nonempty");
        plan
    }

    // [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]
    pub fn execute<CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp>(
        &self,
        map_fun: MapOp,
        reduce_fun: ReduceOp,
        out_ix: usize,
    ) -> CTYPE_OUT
    where
        CTYPE_IN: Copy,
        CTYPE_OUT: Copy,
        MapOp: Fn(CTYPE_IN) -> CTYPE_OUT,
        ReduceOp: Fn(CTYPE_OUT, CTYPE_OUT) -> CTYPE_OUT,
    {
        et_check_msg!(
            self.plan_.get_input_tensor().numel() > 0,
            "Input tensor must be nonempty"
        );

        let init_index: usize = get_init_index(
            self.plan_.get_input_tensor(),
            self.plan_.get_dim_list(),
            out_ix,
        );

        let in_data: *const CTYPE_IN = self.plan_.get_input_tensor().const_data_ptr::<CTYPE_IN>();
        let mut acc_val: CTYPE_OUT = map_fun(unsafe { *in_data.add(init_index) });

        if self.plan_.get_input_tensor().numel() == 1 {
            return acc_val;
        }

        self.plan_.execute(
            |in_ix: usize| {
                acc_val = reduce_fun(map_fun(unsafe { *in_data.add(in_ix) }), acc_val);
            },
            out_ix,
        );
        acc_val
    }
}

/// Reduce a tensor `in` over a given list of dimensions `dim_list` for the
/// output element at index `out_ix`, mapping with `map_fun` then reducing with
/// `reduce_fun`.
// [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-fn]
// [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-fn]
pub fn map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp>(
    map_fun: MapOp,
    reduce_fun: ReduceOp,
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    out_ix: usize,
) -> CTYPE_OUT
where
    CTYPE_IN: Copy,
    CTYPE_OUT: Copy,
    MapOp: Fn(CTYPE_IN) -> CTYPE_OUT,
    ReduceOp: Fn(CTYPE_OUT, CTYPE_OUT) -> CTYPE_OUT,
{
    let plan = MapReduceOverDimListPlan::new(in_, dim_list);
    plan.execute::<CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp>(map_fun, reduce_fun, out_ix)
}

/// Reduce a tensor `in` over a dimension `dim` for the output element at index
/// `out_ix` using `reduce_fun`.
// [spec:et:def:reduce-util.torch.executor.reduce-over-dim-fn]
// [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-fn]
pub fn reduce_over_dim<CTYPE, ReduceOp>(
    reduce_fun: ReduceOp,
    in_: &Tensor,
    dim: &Option<i64>,
    out_ix: usize,
) -> (CTYPE, i64)
where
    CTYPE: Copy,
    ReduceOp: Fn(CTYPE, i64, CTYPE, i64) -> (CTYPE, i64),
{
    map_reduce_over_dim::<CTYPE, CTYPE, _, _>(|v: CTYPE| v, reduce_fun, in_, dim, out_ix)
}

/// Execution plan for repeated reduce_over_dim_list with the same function,
/// input tensor, and dim_list but varying out_ix.
// [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan]
pub struct ReduceOverDimListPlan<'a> {
    plan_: MapReduceOverDimListPlan<'a>,
}

impl<'a> ReduceOverDimListPlan<'a> {
    // [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn]
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn]
    pub fn new(in_: &'a Tensor<'a>, dim_list: &Option<ArrayRef<i64>>) -> Self {
        ReduceOverDimListPlan {
            plan_: MapReduceOverDimListPlan::new(in_, dim_list),
        }
    }

    // [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn]
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn]
    pub fn execute<CTYPE, ReduceOp>(&self, reduce_fun: ReduceOp, out_ix: usize) -> CTYPE
    where
        CTYPE: Copy,
        ReduceOp: Fn(CTYPE, CTYPE) -> CTYPE,
    {
        self.plan_
            .execute::<CTYPE, CTYPE, _, _>(|v: CTYPE| v, reduce_fun, out_ix)
    }
}

/// Reduce a tensor `in` over a given list of dimensions `dim_list` for the
/// output element at index `out_ix` using `reduce_fun`.
// [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-fn]
// [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-fn]
pub fn reduce_over_dim_list<CTYPE, ReduceOp>(
    reduce_fun: ReduceOp,
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    out_ix: usize,
) -> CTYPE
where
    CTYPE: Copy,
    ReduceOp: Fn(CTYPE, CTYPE) -> CTYPE,
{
    let plan = ReduceOverDimListPlan::new(in_, dim_list);
    plan.execute::<CTYPE, ReduceOp>(reduce_fun, out_ix)
}

//
// Compute reduced out tensor size and dim
//

/// (single-`dim` overload)
pub fn compute_reduced_out_dim_dim(in_: &Tensor, dim: &Option<i64>, keepdim: bool) -> ssize_t {
    if keepdim {
        in_.dim()
    } else if dim.is_some() && in_.dim() != 0 {
        in_.dim() - 1
    } else {
        0
    }
}

// [spec:et:def:reduce-util.torch.executor.compute-reduced-out-dim-fn]
// [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-dim-fn]
pub fn compute_reduced_out_dim(
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    keepdim: bool,
) -> ssize_t {
    if keepdim {
        in_.dim()
    } else if dim_list.is_some() && dim_list.as_ref().unwrap().size() != 0 && in_.dim() != 0 {
        in_.dim() - dim_list.as_ref().unwrap().size() as ssize_t
    } else {
        0
    }
}

/// (single-`dim` overload)
///
/// # Safety
/// `sizes_arr` must point to at least `kTensorDimensionLimit` valid
/// `TensorSizesType` elements.
pub unsafe fn compute_reduced_out_size_dim(
    in_: &Tensor,
    dim: &Option<i64>,
    keepdim: bool,
    sizes_arr: *mut TensorSizesType,
) -> usize {
    let in_dim = in_.dim();
    let mut out_dim: usize = in_dim as usize;

    if let Some(dim_val) = dim {
        let non_neg_dim: usize = _normalize_non_neg_d(*dim_val as ssize_t, in_dim);
        for i in 0..non_neg_dim {
            unsafe {
                *sizes_arr.add(i) = in_.size(i as ssize_t) as TensorSizesType;
            }
        }
        if keepdim {
            unsafe {
                *sizes_arr.add(non_neg_dim) = 1;
            }
            let mut i: ssize_t = non_neg_dim as ssize_t + 1;
            while i < in_dim {
                unsafe {
                    *sizes_arr.add(i as usize) = in_.size(i) as TensorSizesType;
                }
                i += 1;
            }
        } else {
            let mut i: ssize_t = non_neg_dim as ssize_t;
            while i < in_dim - 1 {
                unsafe {
                    *sizes_arr.add(i as usize) = in_.size(i + 1) as TensorSizesType;
                }
                i += 1;
            }
            out_dim = if in_dim == 0 {
                0
            } else {
                (in_dim - 1) as usize
            };
        }
    } else {
        if keepdim {
            for i in 0..(in_dim as usize) {
                unsafe {
                    *sizes_arr.add(i) = 1;
                }
            }
        } else {
            out_dim = 0;
        }
    }
    out_dim
}

// [spec:et:def:reduce-util.torch.executor.compute-reduced-out-size-fn]
// [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-size-fn]
///
/// # Safety
/// `sizes_arr` must point to at least `kTensorDimensionLimit` valid
/// `TensorSizesType` elements.
pub unsafe fn compute_reduced_out_size(
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    keepdim: bool,
    sizes_arr: *mut TensorSizesType,
) -> usize {
    // check_dim_in_dim_list and later comparisons expect in_dim to be size_t,
    // so cast it here
    let in_dim: usize = in_.dim() as usize;
    let mut out_dim: usize = in_dim;

    let has_nonempty = match dim_list {
        Some(dl) => dl.size() != 0,
        None => false,
    };
    if has_nonempty {
        let reduce_dims = dim_list.as_ref().unwrap();
        if keepdim {
            for i in 0..in_dim {
                if check_dim_in_dim_list(i, in_dim, reduce_dims) {
                    unsafe {
                        *sizes_arr.add(i) = 1;
                    }
                } else {
                    unsafe {
                        *sizes_arr.add(i) = in_.size(i as ssize_t) as TensorSizesType;
                    }
                }
            }
        } else {
            let mut out_i: usize = 0;
            for in_i in 0..in_dim {
                if !check_dim_in_dim_list(in_i, in_dim, reduce_dims) {
                    unsafe {
                        *sizes_arr.add(out_i) = in_.size(in_i as ssize_t) as TensorSizesType;
                    }
                    out_i += 1;
                }
            }
            out_dim = out_i;
        }
    } else {
        if keepdim {
            for i in 0..in_dim {
                unsafe {
                    *sizes_arr.add(i) = 1;
                }
            }
        } else {
            out_dim = 0;
        }
    }
    out_dim
}

/// (single-`dim` overload)
#[must_use]
pub fn resize_reduction_out_dim(
    in_: &Tensor,
    dim: &Option<i64>,
    keepdim: bool,
    out: &Tensor,
) -> Error {
    let mut sizes_arr: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let out_dim =
        unsafe { compute_reduced_out_size_dim(in_, dim, keepdim, sizes_arr.as_mut_ptr()) };
    let out_size: ArrayRef<TensorSizesType> = ArrayRef::from_raw_parts(sizes_arr.as_ptr(), out_dim);
    resize_tensor(out, out_size)
}

// [spec:et:def:reduce-util.torch.executor.resize-reduction-out-fn]
// [spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]
#[must_use]
pub fn resize_reduction_out(
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    keepdim: bool,
    out: &Tensor,
) -> Error {
    let mut sizes_arr: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let out_dim =
        unsafe { compute_reduced_out_size(in_, dim_list, keepdim, sizes_arr.as_mut_ptr()) };
    let out_size: ArrayRef<TensorSizesType> = ArrayRef::from_raw_parts(sizes_arr.as_ptr(), out_dim);
    resize_tensor(out, out_size)
}

//
// Argument validation (compiled only when USE_ATEN_LIB is not defined; mirrored
// via the `aten` feature being disabled).
//

// [spec:et:def:reduce-util.torch.executor.check-reduction-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_reduction_args(
    in_: &Tensor,
    dim_list: &Option<ArrayRef<i64>>,
    _keepdim: bool,
    dtype: Option<ScalarType>,
    out: &Tensor,
) -> bool {
    if let Some(dtype) = dtype {
        et_log_and_return_if_false!(dtype == out.scalar_type());
    }
    et_log_and_return_if_false!(check_dim_list_is_valid(in_, dim_list));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));

    true
}

// [spec:et:def:reduce-util.torch.executor.check-reduction-args-single-dim-fn]
// [spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn]
#[cfg(not(feature = "aten"))]
pub fn check_reduction_args_single_dim(
    in_: &Tensor,
    dim: Option<i64>,
    _keepdim: bool,
    dtype: Option<ScalarType>,
    out: &Tensor,
    allow_empty_dim: bool,
) -> bool {
    if let Some(dtype) = dtype {
        et_log_and_return_if_false!(dtype == out.scalar_type());
    }
    if in_.dim() == 0 {
        if let Some(dim_val) = dim {
            et_log_and_return_if_false!(dim_val == 0 || dim_val == -1);
        }
        return true;
    }

    if let Some(dim_val) = dim {
        et_log_and_return_if_false!(dim_is_valid(dim_val, in_.dim() as i64));
        if !allow_empty_dim {
            et_log_and_return_if_false!(tensor_has_non_empty_dim(in_, dim_val));
        }
    }

    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));

    true
}

// [spec:et:def:reduce-util.torch.executor.check-mean-dim-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-mean-dim-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_mean_dim_args(
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    keepdim: bool,
    dtype: Option<ScalarType>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(check_reduction_args(in_, &dim_list, keepdim, dtype, out));

    if let Some(dtype) = dtype {
        crate::et_log!(Error, "dtype is {}", dtype as i8);
        et_log_and_return_if_false!(is_floating_type(dtype));
        et_log_and_return_if_false!(out.scalar_type() == dtype);
    } else {
        et_log_and_return_if_false!(tensor_is_floating_type(in_));
        et_log_and_return_if_false!(tensor_is_floating_type(out));
    }

    true
}

// [spec:et:def:reduce-util.torch.executor.check-amin-amax-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-amin-amax-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_amin_amax_args(
    in_: &Tensor,
    dim_list: ArrayRef<i64>,
    keepdim: bool,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(check_reduction_args(
        in_,
        &Some(dim_list),
        keepdim,
        None,
        out
    ));
    et_log_and_return_if_false!(in_.scalar_type() == out.scalar_type());

    true
}

// [spec:et:def:reduce-util.torch.executor.check-argmin-argmax-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-argmin-argmax-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_argmin_argmax_args(
    in_: &Tensor,
    dim: Option<i64>,
    keepdim: bool,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(check_reduction_args_single_dim(
        in_, dim, keepdim, None, out, false
    ));

    et_log_and_return_if_false!(out.scalar_type() == ScalarType::Long);

    true
}

// [spec:et:def:reduce-util.torch.executor.check-min-max-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-min-max-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_min_max_args(
    in_: &Tensor,
    dim: i64,
    keepdim: bool,
    max: &Tensor,
    max_indices: &Tensor,
) -> bool {
    et_log_and_return_if_false!(check_reduction_args_single_dim(
        in_,
        Some(dim),
        keepdim,
        None,
        max,
        false
    ));
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, max));
    et_log_and_return_if_false!(tensors_have_same_shape2(max, max_indices));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(max_indices));
    et_log_and_return_if_false!(max_indices.scalar_type() == ScalarType::Long);

    true
}

// [spec:et:def:reduce-util.torch.executor.check-prod-out-args-fn]
// [spec:et:sem:reduce-util.torch.executor.check-prod-out-args-fn]
#[cfg(not(feature = "aten"))]
pub fn check_prod_out_args(in_: &Tensor, dtype: Option<ScalarType>, out: &Tensor) -> bool {
    if let Some(dtype) = dtype {
        et_log_and_return_if_false!(dtype == out.scalar_type());
    } else if is_integral_type(in_.scalar_type(), /*includeBool*/ true) {
        et_log_and_return_if_false!(out.scalar_type() == ScalarType::Long);
    } else {
        et_log_and_return_if_false!(out.scalar_type() == in_.scalar_type());
    }

    true
}

//
// parallel_for wrappers.
//
// PORT-NOTE: `executorch::extension::parallel_for` and `GRAIN_SIZE` map to the
// ported `runtime/kernel/thread_parallel_interface`. The C++ `#ifdef
// ET_USE_THREADPOOL` selects a reduction-size-scaled grain size; the Rust
// `thread_parallel_interface` mirrors the threadpool build, so the threadpool
// branch is ported. `func` is a `&dyn Fn(i64, i64)` matching the `parallel_for`
// closure contract; the `[[nodiscard]]` bool result is returned as-is.

/// parallel_for wrapper for reductions that call reduce_over_dim or
/// map_reduce_over_dim for each output element. Automatically calculates
/// appropriate grain size.
// [spec:et:def:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn]
// [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn]
#[must_use]
pub fn parallel_for_each_reduce_over_dim_output_index<Func>(
    in_: &Tensor,
    dim: Option<i64>,
    out: &Tensor,
    func: &Func,
) -> bool
where
    Func: Fn(i64, i64),
{
    let reduction_size: ssize_t = get_reduced_dim_product_dim(in_, &dim) as ssize_t;
    let grain_size: ssize_t = if reduction_size == 0 {
        1
    } else {
        core::cmp::max(
            1,
            crate::runtime::kernel::thread_parallel_interface::internal::GRAIN_SIZE as ssize_t
                / reduction_size,
        )
    };
    crate::runtime::kernel::thread_parallel_interface::parallel_for(
        0,
        out.numel() as i64,
        grain_size as i64,
        func,
    )
}

/// parallel_for wrapper for reductions that call reduce_over_dim_list or
/// map_reduce_over_dim_list for each output element. Automatically calculates
/// appropriate grain size.
// [spec:et:def:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]
// [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]
#[must_use]
pub fn parallel_for_each_reduce_over_dim_list_output_index<Func>(
    in_: &Tensor,
    dim_list: Option<ArrayRef<i64>>,
    out: &Tensor,
    func: &Func,
) -> bool
where
    Func: Fn(i64, i64),
{
    let reduction_size: ssize_t = get_reduced_dim_product(in_, &dim_list) as ssize_t;
    let grain_size: ssize_t = if reduction_size == 0 {
        1
    } else {
        core::cmp::max(
            1,
            crate::runtime::kernel::thread_parallel_interface::internal::GRAIN_SIZE as ssize_t
                / reduction_size,
        )
    };
    crate::runtime::kernel::thread_parallel_interface::parallel_for(
        0,
        out.numel() as i64,
        grain_size as i64,
        func,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    // Mirrors the C++ `_apply_over_dim` free helper.
    fn _apply_over_dim(in_: &Tensor, dim: &Option<i64>) {
        let in_data: *mut i64 = in_.mutable_data_ptr::<i64>();
        for out_ix in 0..get_out_numel_dim(in_, dim) {
            apply_over_dim(
                |in_ix: usize, _: usize| unsafe {
                    *in_data.add(in_ix) = out_ix as i64;
                },
                in_,
                dim,
                out_ix,
                0,
                -1,
            );
        }
    }

    // Mirrors the C++ `_apply_over_dim_list` free helper.
    fn _apply_over_dim_list(in_: &Tensor, dim_list: &Option<ArrayRef<i64>>) {
        let in_data: *mut i64 = in_.mutable_data_ptr::<i64>();
        for out_ix in 0..get_out_numel(in_, dim_list) {
            apply_over_dim_list(
                |in_ix: usize| unsafe {
                    *in_data.add(in_ix) = out_ix as i64;
                },
                in_,
                dim_list,
                out_ix,
                0,
                -1,
            );
        }
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn/test]
    // also verifies get_out_numel_dim, get_init_index_dim, get_reduced_dim_product_dim,
    // _normalize_non_neg_d, and the flat+dim-ix iteration helper it drives.
    // [spec:et:sem:reduce-util.torch.executor.get-init-index-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim() {
        let tf = TensorFactory::<i64>::new();
        let mut in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        _apply_over_dim(&in_, &Some(0));
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
            45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,

             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
            45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        _apply_over_dim(&in_, &Some(1));
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        _apply_over_dim(&in_, &Some(2));
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
             9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

            12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,
            15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,
            18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,
            21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        _apply_over_dim(&in_, &Some(3));
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
            10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
            15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

            20, 20, 20,   21, 21, 21,   22, 22, 22,   23, 23, 23,   24, 24, 24,
            25, 25, 25,   26, 26, 26,   27, 27, 27,   28, 28, 28,   29, 29, 29,
            30, 30, 30,   31, 31, 31,   32, 32, 32,   33, 33, 33,   34, 34, 34,
            35, 35, 35,   36, 36, 36,   37, 37, 37,   38, 38, 38,   39, 39, 39,
        ]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn/test]
    // Null dim_list drives ApplyOverDimListPlan in NoDimMaskOrZeroDimension mode,
    // exercising the plan ctor/execute and the stride-only flat helper.
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_null() {
        let tf = TensorFactory::<i64>::new();
        let null_dim_list: Option<ArrayRef<i64>> = None;

        let in_ = tf.ones_default(vec![2, 4, 5, 3]);
        _apply_over_dim_list(&in_, &null_dim_list);
        assert_tensor_eq!(in_, tf.zeros_default(vec![2, 4, 5, 3]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_zero_dim_list_empty() {
        let tf = TensorFactory::<i64>::new();
        let null_dim_list: Option<ArrayRef<i64>> = None;

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim_list(&in_, &null_dim_list);
        assert_tensor_eq!(in_, tf.zeros_default(vec![]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_zero_dim() {
        let tf = TensorFactory::<i64>::new();
        let dim_array_0 = [0i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_0.as_ptr(), 1));

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim_list(&in_, &dim_list);
        assert_tensor_eq!(in_, tf.zeros_default(vec![]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_empty() {
        let tf = TensorFactory::<i64>::new();
        let empty: [i64; 0] = [];
        let empty_dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(empty.as_ptr(), 0));

        let in_ = tf.ones_default(vec![2, 4, 5, 3]);
        _apply_over_dim_list(&in_, &empty_dim_list);
        assert_tensor_eq!(in_, tf.zeros_default(vec![2, 4, 5, 3]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    // Single-dim lists drive ApplyOverDimListPlan in OnlyOneDim mode, which
    // routes through the flat+dim-ix helper; get_out_numel drives check_dim_in_dim_list.
    // [spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_length1() {
        let tf = TensorFactory::<i64>::new();

        let mut in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_0 = [0i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_0.as_ptr(), 1));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
            45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,

             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            30, 31, 32,   33, 34, 35,   36, 37, 38,   39, 40, 41,   42, 43, 44,
            45, 46, 47,   48, 49, 50,   51, 52, 53,   54, 55, 56,   57, 58, 59,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_1 = [1i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_1.as_ptr(), 1));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
            15, 16, 17,   18, 19, 20,   21, 22, 23,   24, 25, 26,   27, 28, 29,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_2 = [2i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_2.as_ptr(), 1));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
             9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

            12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,   12, 13, 14,
            15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,   15, 16, 17,
            18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,   18, 19, 20,
            21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,   21, 22, 23,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_3 = [3i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_3.as_ptr(), 1));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
            10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
            15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

            20, 20, 20,   21, 21, 21,   22, 22, 22,   23, 23, 23,   24, 24, 24,
            25, 25, 25,   26, 26, 26,   27, 27, 27,   28, 28, 28,   29, 29, 29,
            30, 30, 30,   31, 31, 31,   32, 32, 32,   33, 33, 33,   34, 34, 34,
            35, 35, 35,   36, 36, 36,   37, 37, 37,   38, 38, 38,   39, 39, 39,
        ]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    // Multi-dim lists drive ApplyOverDimListPlan in NormalDimMask mode, exercising
    // the dim-mask flat iteration helper and its carry-over index arithmetic.
    // [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_length2() {
        let tf = TensorFactory::<i64>::new();

        let mut in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_01 = [0i64, 1];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_01.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,

             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
             0,  1,  2,    3,  4,  5,    6,  7,  8,    9, 10, 11,   12, 13, 14,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_02 = [0i64, 2];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_02.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
             9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,

             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,    6,  7,  8,
             9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,    9, 10, 11,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_03 = [0i64, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_03.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
            10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
            15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,

             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
            10, 10, 10,   11, 11, 11,   12, 12, 12,   13, 13, 13,   14, 14, 14,
            15, 15, 15,   16, 16, 16,   17, 17, 17,   18, 18, 18,   19, 19, 19,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_12 = [1i64, 2];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_12.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,
             0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,    0,  1,  2,

             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
             3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,    3,  4,  5,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_13 = [1i64, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_13.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,
             0,  0,  0,    1,  1,  1,    2,  2,  2,    3,  3,  3,    4,  4,  4,

             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
             5,  5,  5,    6,  6,  6,    7,  7,  7,    8,  8,  8,    9,  9,  9,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_23 = [2i64, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_23.as_ptr(), 2));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
            3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,

            4, 4, 4,   4, 4, 4,   4, 4, 4,   4, 4, 4,   4, 4, 4,
            5, 5, 5,   5, 5, 5,   5, 5, 5,   5, 5, 5,   5, 5, 5,
            6, 6, 6,   6, 6, 6,   6, 6, 6,   6, 6, 6,   6, 6, 6,
            7, 7, 7,   7, 7, 7,   7, 7, 7,   7, 7, 7,   7, 7, 7,
        ]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_length3() {
        let tf = TensorFactory::<i64>::new();

        let mut in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_012 = [0i64, 1, 2];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_012.as_ptr(), 3));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,

            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
            0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,   0, 1, 2,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_013 = [0i64, 1, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_013.as_ptr(), 3));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,

            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
            0, 0, 0,   1, 1, 1,   2, 2, 2,   3, 3, 3,   4, 4, 4,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_023 = [0i64, 2, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_023.as_ptr(), 3));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
            3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,

            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,   2, 2, 2,
            3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,   3, 3, 3,
        ]));

        in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_123 = [1i64, 2, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_123.as_ptr(), 3));
        _apply_over_dim_list(&in_, &dim_list);
        #[rustfmt::skip]
        assert_tensor_eq!(in_, tf.make_default(vec![2, 4, 5, 3], vec![
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,
            0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0,

            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
            1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,   1, 1, 1,
        ]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_over_dim_list_length4() {
        let tf = TensorFactory::<i64>::new();

        let in_ = tf.ones_default(vec![2, 4, 5, 3]);
        let dim_array_0123 = [0i64, 1, 2, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_0123.as_ptr(), 4));
        _apply_over_dim_list(&in_, &dim_list);
        assert_tensor_eq!(in_, tf.zeros_default(vec![2, 4, 5, 3]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn/test]
    #[test]
    fn reduce_util_test_apply_on_zero_dim_tensor_over_dim() {
        let tf = TensorFactory::<i64>::new();

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim(&in_, &Some(0));
        assert_tensor_eq!(in_, tf.make_default(vec![], vec![0]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_on_zero_dim_tensor_over_dim_list_null() {
        let tf = TensorFactory::<i64>::new();
        let null_dim_list: Option<ArrayRef<i64>> = None;

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim_list(&in_, &null_dim_list);
        assert_tensor_eq!(in_, tf.make_default(vec![], vec![0]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_on_zero_dim_tensor_over_dim_list_empty() {
        let tf = TensorFactory::<i64>::new();
        let empty: [i64; 0] = [];
        let empty_dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(empty.as_ptr(), 0));

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim_list(&in_, &empty_dim_list);
        assert_tensor_eq!(in_, tf.make_default(vec![], vec![0]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    fn reduce_util_test_apply_on_zero_dim_tensor_over_dim_list_non_empty() {
        let tf = TensorFactory::<i64>::new();
        let dim_array_0 = [0i64];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_0.as_ptr(), 1));

        let in_ = tf.ones_default(vec![]);
        _apply_over_dim_list(&in_, &dim_list);
        assert_tensor_eq!(in_, tf.make_default(vec![], vec![0]));
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn/test]
    #[test]
    fn reduce_util_test_apply_on_empty_tensor_over_dim() {
        let tf = TensorFactory::<i64>::new();

        let in_ = tf.zeros_default(vec![2, 0, 5, 3]);
        let out = tf.zeros_default(vec![2, 5, 3]);

        // dim = 1
        let dim: Option<i64> = Some(1);
        assert!(in_.numel() == 0);
        assert!(out.numel() == 30 && out.numel() as usize == get_out_numel_dim(&in_, &dim));

        let in_data: *mut i64 = in_.mutable_data_ptr::<i64>();
        let out_data: *mut i64 = out.mutable_data_ptr::<i64>();
        for out_ix in 0..get_out_numel_dim(&in_, &dim) {
            unsafe {
                *out_data.add(out_ix) = 1;
            }
            apply_over_dim(
                |in_ix: usize, _: usize| unsafe {
                    *in_data.add(in_ix) = out_ix as i64; // Should be ignored.
                    *out_data.add(out_ix) = 2; // Should be ignored.
                },
                &in_,
                &dim,
                out_ix,
                0,
                -1,
            );
        }
        assert_tensor_eq!(out, tf.ones_default(vec![2, 5, 3]));

        // dim = 0
        let dim: Option<i64> = Some(0);
        assert!(in_.numel() == 0);
        assert!(get_out_numel_dim(&in_, &dim) == 0);
        // ET_EXPECT_DEATH: see
        // reduce_util_test_apply_on_empty_tensor_over_dim_death.
    }

    // PORT-NOTE: death test; `runtime_abort` -> `libc::abort()` terminates the
    // process, so `#[should_panic]` cannot catch it; ported and `#[ignore]`d per
    // the established convention.
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_on_empty_tensor_over_dim_death() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 0, 5, 3]);
        let dim: Option<i64> = Some(0);
        apply_over_dim(|_in_ix: usize, _: usize| {}, &in_, &dim, 0, 0, -1);
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn/test]
    #[test]
    fn reduce_util_test_apply_on_empty_tensor_over_dim_list() {
        let tf = TensorFactory::<i64>::new();

        let in_ = tf.zeros_default(vec![2, 0, 5, 3]);
        let out = tf.zeros_default(vec![5, 3]);

        // dim list = {0, 1}
        let dim_array_01 = [0i64, 1];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_01.as_ptr(), 2));

        assert!(in_.numel() == 0);
        assert!(out.numel() == 15 && out.numel() as usize == get_out_numel(&in_, &dim_list));

        let in_data: *mut i64 = in_.mutable_data_ptr::<i64>();
        let out_data: *mut i64 = out.mutable_data_ptr::<i64>();
        for out_ix in 0..get_out_numel(&in_, &dim_list) {
            unsafe {
                *out_data.add(out_ix) = 1;
            }
            apply_over_dim_list(
                |in_ix: usize| unsafe {
                    *in_data.add(in_ix) = out_ix as i64; // Should be ignored.
                    *out_data.add(out_ix) = 2; // Should be ignored.
                },
                &in_,
                &dim_list,
                out_ix,
                0,
                -1,
            );
        }
        assert_tensor_eq!(out, tf.ones_default(vec![5, 3]));

        // dim list = {0, 2}
        let dim_array_02 = [0i64, 2];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_02.as_ptr(), 2));

        assert!(in_.numel() == 0);
        assert!(get_out_numel(&in_, &dim_list) == 0);
        // ET_EXPECT_DEATH: see
        // reduce_util_test_apply_on_empty_tensor_over_dim_list_death.
    }

    // PORT-NOTE: death test; see note above.
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_on_empty_tensor_over_dim_list_death() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 0, 5, 3]);
        let dim_array_02 = [0i64, 2];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_02.as_ptr(), 2));
        apply_over_dim_list(|_in_ix: usize| {}, &in_, &dim_list, 0, 0, -1);
    }

    // PORT-NOTE: death tests; see note above. The four `ET_EXPECT_DEATH` cases
    // (dim {0,9} out-of-bounds, {0,-5,3} negative, {0,1,1} duplicate, {1,-3})
    // become individual `#[ignore]`d `#[should_panic]` tests.
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_over_dim_list_invalid_oob() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_09 = [0i64, 9];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_09.as_ptr(), 2));
        apply_over_dim_list(|_in_ix: usize| {}, &in_, &dim_list, 0, 0, -1);
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_over_dim_list_invalid_neg() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_neg = [0i64, -5, 3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_neg.as_ptr(), 3));
        apply_over_dim_list(|_in_ix: usize| {}, &in_, &dim_list, 0, 0, -1);
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_over_dim_list_invalid_dup() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_011 = [0i64, 1, 1];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_011.as_ptr(), 3));
        apply_over_dim_list(|_in_ix: usize| {}, &in_, &dim_list, 0, 0, -1);
    }

    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn/test]
    #[test]
    #[should_panic]
    #[ignore]
    fn reduce_util_test_apply_over_dim_list_invalid_neg2() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 4, 5, 3]);
        let dim_array_1_3 = [1i64, -3];
        let dim_list: Option<ArrayRef<i64>> =
            Some(ArrayRef::from_raw_parts(dim_array_1_3.as_ptr(), 2));
        apply_over_dim_list(|_in_ix: usize| {}, &in_, &dim_list, 0, 0, -1);
    }

    // PORT-NOTE: focused Wave-3 unit tests for reduce_util functions not reached
    // by the ported C++ ReduceUtilTest suite (which only drives apply_over_dim /
    // apply_over_dim_list). Semantics referee: reduce_util.h + docs/spec/port sem
    // rules. Values are hand-computed against the C++ definitions.

    fn dl(v: &[i64]) -> Option<ArrayRef<i64>> {
        Some(ArrayRef::from_raw_parts(v.as_ptr(), v.len()))
    }

    // Sum over `dim` for each output element, checked element by element.
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn/test]
    #[test]
    fn reduce_util_test_reduce_over_dim() {
        let tf = TensorFactory::<i64>::new();
        // 0 1 2
        // 3 4 5
        let in_ = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
        let dim: Option<i64> = Some(1);

        // out shape after reducing dim 1 is [2]; out_ix in {0,1}.
        let (v0, _ix0): (i64, i64) =
            reduce_over_dim(|v, _iv, acc, aix| (acc + v, aix), &in_, &dim, 0);
        let (v1, _ix1): (i64, i64) =
            reduce_over_dim(|v, _iv, acc, aix| (acc + v, aix), &in_, &dim, 1);
        assert_eq!(v0, 0 + 1 + 2);
        assert_eq!(v1, 3 + 4 + 5);

        // Reduce over dim 0 instead: out shape [3], out_ix in {0,1,2}.
        let dim0: Option<i64> = Some(0);
        let (c0, _): (i64, i64) =
            reduce_over_dim(|v, _iv, acc, aix| (acc + v, aix), &in_, &dim0, 0);
        let (c1, _): (i64, i64) =
            reduce_over_dim(|v, _iv, acc, aix| (acc + v, aix), &in_, &dim0, 1);
        let (c2, _): (i64, i64) =
            reduce_over_dim(|v, _iv, acc, aix| (acc + v, aix), &in_, &dim0, 2);
        assert_eq!(c0, 0 + 3);
        assert_eq!(c1, 1 + 4);
        assert_eq!(c2, 2 + 5);
    }

    // map_reduce_over_dim with a non-identity map (double) then sum.
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn/test]
    #[test]
    fn reduce_util_test_map_reduce_over_dim() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
        let dim: Option<i64> = Some(1);
        let (v0, _): (i64, i64) = map_reduce_over_dim::<i64, i64, _, _>(
            |v| v * 2,
            |v, _iv, acc, aix| (acc + v, aix),
            &in_,
            &dim,
            0,
        );
        assert_eq!(v0, (0 + 1 + 2) * 2);
    }

    // reduce_over_dim_list / map_reduce_over_dim_list and their plans.
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-input-tensor-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-dim-list-fn/test]
    #[test]
    fn reduce_util_test_reduce_over_dim_list() {
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);

        // Reduce over both dims -> single output element = sum of all.
        let d01 = dl(&[0, 1]);
        let total: i64 = reduce_over_dim_list(|v, acc| v + acc, &in_, &d01, 0);
        assert_eq!(total, 0 + 1 + 2 + 3 + 4 + 5);

        // Reduce over dim 1 -> out shape [2]; two output indices.
        let d1 = dl(&[1]);
        let r0: i64 = reduce_over_dim_list(|v, acc| v + acc, &in_, &d1, 0);
        let r1: i64 = reduce_over_dim_list(|v, acc| v + acc, &in_, &d1, 1);
        assert_eq!(r0, 0 + 1 + 2);
        assert_eq!(r1, 3 + 4 + 5);

        // ReduceOverDimListPlan reused across out indices matches the free fn.
        let plan = ReduceOverDimListPlan::new(&in_, &d1);
        let p0: i64 = plan.execute(|v, acc| v + acc, 0);
        let p1: i64 = plan.execute(|v, acc| v + acc, 1);
        assert_eq!(p0, r0);
        assert_eq!(p1, r1);

        // map_reduce_over_dim_list: square each element then sum over dim 1.
        let m0: i64 =
            map_reduce_over_dim_list::<i64, i64, _, _>(|v| v * v, |v, acc| v + acc, &in_, &d1, 0);
        assert_eq!(m0, 0 * 0 + 1 * 1 + 2 * 2);

        // MapReduceOverDimListPlan directly, verifying its accessors are wired.
        let mplan = MapReduceOverDimListPlan::new(&in_, &d1);
        assert_eq!(mplan.plan_.get_input_tensor().numel(), in_.numel());
        assert!(mplan.plan_.get_dim_list().is_some());
        let mp1: i64 = mplan.execute::<i64, i64, _, _>(|v| v * v, |v, acc| v + acc, 1);
        assert_eq!(mp1, 3 * 3 + 4 * 4 + 5 * 5);
    }

    // compute_reduced_out_dim / compute_reduced_out_size / resize_reduction_out.
    // [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-dim-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-size-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn/test]
    #[test]
    fn reduce_util_test_compute_and_resize_reduced_out() {
        crate::runtime::platform::platform::pal_init();
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 3, 4]);

        // keepdim reduces rank unchanged.
        let d1 = dl(&[1]);
        assert_eq!(compute_reduced_out_dim(&in_, &d1, /*keepdim=*/ true), 3);
        // non-keepdim over one dim drops one dim.
        assert_eq!(compute_reduced_out_dim(&in_, &d1, /*keepdim=*/ false), 2);
        // non-keepdim over two dims drops two dims.
        let d01 = dl(&[0, 1]);
        assert_eq!(compute_reduced_out_dim(&in_, &d01, /*keepdim=*/ false), 1);
        // empty/None dim_list, non-keepdim -> scalar.
        let none: Option<ArrayRef<i64>> = None;
        assert_eq!(compute_reduced_out_dim(&in_, &none, /*keepdim=*/ false), 0);
        // None + keepdim -> unchanged rank.
        assert_eq!(compute_reduced_out_dim(&in_, &none, /*keepdim=*/ true), 3);

        // compute_reduced_out_size, keepdim: reduced dims become 1.
        let mut sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        let out_dim = unsafe { compute_reduced_out_size(&in_, &d1, true, sizes.as_mut_ptr()) };
        assert_eq!(out_dim, 3);
        assert_eq!(&sizes[..3], &[2, 1, 4]);

        // compute_reduced_out_size, non-keepdim: reduced dims removed.
        let out_dim = unsafe { compute_reduced_out_size(&in_, &d1, false, sizes.as_mut_ptr()) };
        assert_eq!(out_dim, 2);
        assert_eq!(&sizes[..2], &[2, 4]);

        // resize_reduction_out resizes `out` to the computed reduced shape.
        // Rank is immutable, so `out` starts at rank 2 with enough capacity.
        use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
        let out = tf.zeros(vec![8, 1], TensorShapeDynamism::DYNAMIC_BOUND);
        let err = resize_reduction_out(&in_, &d1, /*keepdim=*/ false, &out);
        assert_eq!(err, Error::Ok);
        assert_eq!(out.dim(), 2);
        assert_eq!(out.size(0), 2);
        assert_eq!(out.size(1), 4);
    }

    // check_reduction_args and the argument validators layered on top of it.
    // [spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-mean-dim-args-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-amin-amax-args-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-argmin-argmax-args-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-min-max-args-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.check-prod-out-args-fn/test]
    #[cfg(not(feature = "aten"))]
    #[test]
    fn reduce_util_test_check_args() {
        crate::runtime::platform::platform::pal_init();
        let tf_f = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();
        let in_f = tf_f.zeros_default(vec![2, 3, 4]);
        let out_f = tf_f.zeros_default(vec![2, 1, 4]);
        let d1 = dl(&[1]);

        // Valid float reduction with matching dtype.
        assert!(check_reduction_args(
            &in_f,
            &d1,
            true,
            Some(ScalarType::Float),
            &out_f
        ));
        // Mismatched requested dtype vs out dtype fails.
        assert!(!check_reduction_args(
            &in_f,
            &d1,
            true,
            Some(ScalarType::Double),
            &out_f
        ));
        // Invalid dim list (out of range) fails.
        let bad = dl(&[9]);
        assert!(!check_reduction_args(&in_f, &bad, true, None, &out_f));

        // Single-dim variant: valid dim, non-empty.
        assert!(check_reduction_args_single_dim(
            &in_f,
            Some(1),
            true,
            None,
            &out_f,
            false
        ));
        // Out-of-range dim fails.
        assert!(!check_reduction_args_single_dim(
            &in_f,
            Some(5),
            true,
            None,
            &out_f,
            false
        ));

        // mean requires floating types; float in/out passes.
        assert!(check_mean_dim_args(&in_f, d1, true, None, &out_f));
        // mean with an integral out (dtype requested non-float) fails.
        let out_l = tf_l.zeros_default(vec![2, 1, 4]);
        assert!(!check_mean_dim_args(
            &in_f,
            d1,
            true,
            Some(ScalarType::Long),
            &out_l
        ));

        // amin/amax: out dtype must equal in dtype.
        let d1_ref = ArrayRef::from_raw_parts([1i64].as_ptr(), 1);
        assert!(check_amin_amax_args(&in_f, d1_ref, true, &out_f));
        assert!(!check_amin_amax_args(&in_f, d1_ref, true, &out_l));

        // argmin/argmax: index output must be Long.
        let out_idx = tf_l.zeros_default(vec![2, 1, 4]);
        assert!(check_argmin_argmax_args(&in_f, Some(1), true, &out_idx));
        assert!(!check_argmin_argmax_args(&in_f, Some(1), true, &out_f));

        // min/max: values same dtype as in, indices Long and same shape as values.
        assert!(check_min_max_args(&in_f, 1, true, &out_f, &out_idx));
        assert!(!check_min_max_args(&in_f, 1, true, &out_l, &out_idx));

        // prod: integral in with default dtype requires Long out.
        let in_l = tf_l.zeros_default(vec![2, 3]);
        let scalar_l = tf_l.zeros_default(vec![]);
        let scalar_f = tf_f.zeros_default(vec![]);
        assert!(check_prod_out_args(&in_l, None, &scalar_l));
        assert!(!check_prod_out_args(&in_l, None, &scalar_f));
        // float in with default dtype requires same-dtype out.
        assert!(check_prod_out_args(&in_f, None, &scalar_f));
    }

    // parallel_for_each_reduce_over_dim(_list)_output_index run the closure once
    // per output element and return true on success.
    // [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn/test]
    // [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn/test]
    #[test]
    fn reduce_util_test_parallel_for_each_reduce_output_index() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        let tf = TensorFactory::<i64>::new();
        let in_ = tf.zeros_default(vec![2, 3, 4]);
        let out = tf.zeros_default(vec![2, 4]); // reduce dim 1

        let count = AtomicUsize::new(0);
        let ok =
            parallel_for_each_reduce_over_dim_output_index(&in_, Some(1), &out, &|begin, end| {
                count.fetch_add((end - begin) as usize, Ordering::Relaxed);
            });
        assert!(ok);
        assert_eq!(count.load(Ordering::Relaxed), out.numel() as usize);

        let count2 = AtomicUsize::new(0);
        let d1 = dl(&[1]);
        let ok2 =
            parallel_for_each_reduce_over_dim_list_output_index(&in_, d1, &out, &|begin, end| {
                count2.fetch_add((end - begin) as usize, Ordering::Relaxed);
            });
        assert!(ok2);
        assert_eq!(count2.load(Ordering::Relaxed), out.numel() as usize);
    }
}
