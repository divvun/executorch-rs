//! Literal port of kernels/portable/cpu/util/kernel_ops_util.cpp + kernels/portable/cpu/util/kernel_ops_util.h.

use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::dim_order_to_stride_nocheck;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    can_cast, is_integral_type, to_string,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, calculate_linear_index, dim_is_valid, resize_tensor_same_type,
    tensor_is_default_dim_order, tensor_is_default_or_channels_last_dim_order, tensor_is_rank,
    tensors_have_same_dtype, tensors_have_same_dtype2, tensors_have_same_rank,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, ssize_t,
};

// PORT-NOTE: local check macros mirroring the C++ `ET_LOG_AND_RETURN_IF_FALSE`
// and `ET_CHECK_OR_RETURN_FALSE`; the crate-level `et_check_or_return_false!`
// drops caller format args, so this module carries its own (as tensor_util.rs
// does) to keep the messages literal.
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

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

/// Extracts a value at index i from an int array. If the array length is 1, then
/// the first element will be returned regardless of what i is requested to
/// simulate broadcasting.
// [spec:et:def:kernel-ops-util.torch.executor.val-at-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.val-at-fn]
pub fn val_at(array: IntArrayRef, i: usize, default_value: i64) -> i64 {
    if array.size() == 1 {
        *array.at(0)
    } else if array.size() > 1 {
        *array.at(i)
    } else {
        default_value
    }
}

// PORT-NOTE: C++ `val_at` has a default argument `default_value = 1`. Rust has
// no default args; this thin wrapper preserves the two-argument call sites.
pub fn val_at_default(array: IntArrayRef, i: usize) -> i64 {
    val_at(array, i, 1)
}

// PORT-NOTE: file-local (anonymous namespace) helper in the C++ .cpp.
// [spec:et:def:kernel-ops-util.torch.executor.param-array-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.param-array-is-valid-fn]
fn param_array_is_valid(
    name: &str,
    array: IntArrayRef,
    min_val: i64,
    length: usize,
    allow_empty: bool,
) -> bool {
    let size = array.size();
    if allow_empty {
        et_check_or_return_false!(
            size == 0 || size == 1 || size == length,
            "Expected {} to have size 0, 1 or {} but got {}",
            name,
            length,
            size
        );
    } else {
        et_check_or_return_false!(
            size == 1 || size == length,
            "Expected {} to have size 1 or {} but got {}",
            name,
            length,
            size
        );
    }
    et_log_and_return_if_false!(int_array_all_ge(array, min_val));
    true
}

// PORT-NOTE: file-local (anonymous namespace) helper in the C++ .cpp.
// [spec:et:def:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn]
///
/// # Safety
/// `kernel_size` must point to at least `weight.dim() - 2` valid `i64` elements;
/// `kernel_ndim` to a valid `usize`.
unsafe fn fill_convolution_kernel_size(
    weight: &Tensor,
    kernel_size: *mut i64,
    kernel_ndim: *mut usize,
) {
    unsafe {
        *kernel_ndim = (weight.dim() - 2) as usize;
        for i in 0..*kernel_ndim {
            *kernel_size.add(i) = weight.size((i + 2) as ssize_t) as i64;
        }
    }
}

/// Checks that all elements of an IntArray are greater than or equal to `val`.
// [spec:et:def:kernel-ops-util.torch.executor.int-array-all-ge-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.int-array-all-ge-fn]
pub fn int_array_all_ge(array: IntArrayRef, val: i64) -> bool {
    for i in 0..array.size() {
        if *array.at(i) < val {
            crate::et_log!(
                Error,
                "Expected array[{}] > {}, found {}",
                i,
                val,
                *array.at(i)
            );
            return false;
        }
    }
    true
}

// [spec:et:def:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.kernel-size-is-valid-fn]
pub fn kernel_size_is_valid(kernel_size: IntArrayRef, kernel_ndim: usize) -> bool {
    param_array_is_valid("kernel_size", kernel_size, 1, kernel_ndim, false)
}

// [spec:et:def:kernel-ops-util.torch.executor.stride-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.stride-is-valid-fn]
pub fn stride_is_valid(stride: IntArrayRef, kernel_ndim: usize, allow_empty: bool) -> bool {
    param_array_is_valid("stride", stride, 1, kernel_ndim, allow_empty)
}

// [spec:et:def:kernel-ops-util.torch.executor.padding-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.padding-is-valid-fn]
pub fn padding_is_valid(
    padding: IntArrayRef,
    kernel_size: IntArrayRef,
    kernel_ndim: usize,
    enforce_half_kernel: bool,
) -> bool {
    let valid = param_array_is_valid("padding", padding, 0, kernel_ndim, false);
    if !valid {
        return false;
    }

    if enforce_half_kernel {
        // Padding must be at most half of kernel size.
        for i in 0..padding.size() {
            if *padding.at(i) > val_at_default(kernel_size, i) / 2 {
                crate::et_log!(
                    Error,
                    "Padding should be at most half of kernel size, but got padding[{}] = {} > kernel_size[{}] = {}",
                    i,
                    *padding.at(i),
                    i,
                    val_at_default(kernel_size, i)
                );
                return false;
            }
        }
    }
    true
}

// [spec:et:def:kernel-ops-util.torch.executor.dilation-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.dilation-is-valid-fn]
pub fn dilation_is_valid(dilation: IntArrayRef, kernel_ndim: usize) -> bool {
    param_array_is_valid("dilation", dilation, 1, kernel_ndim, false)
}

// [spec:et:def:kernel-ops-util.torch.executor.output-padding-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.output-padding-is-valid-fn]
pub fn output_padding_is_valid(
    output_padding: IntArrayRef,
    stride: IntArrayRef,
    dilation: IntArrayRef,
    kernel_ndim: usize,
) -> bool {
    et_log_and_return_if_false!(param_array_is_valid(
        "output_padding",
        output_padding,
        0,
        kernel_ndim,
        false
    ));

    for i in 0..kernel_ndim {
        let op_i: i64 = val_at_default(output_padding, i);
        let s_i: i64 = val_at_default(stride, i);
        let d_i: i64 = val_at_default(dilation, i);
        et_check_or_return_false!(
            op_i < s_i || op_i < d_i,
            "output padding must be smaller than either stride or dilation"
        );
    }
    true
}

// [spec:et:def:kernel-ops-util.torch.executor.output-size-is-valid-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.output-size-is-valid-fn]
pub fn output_size_is_valid(output_size: ArrayRef<SizesType>, kernel_ndim: usize) -> bool {
    let mut valid = true;
    let out_dim = output_size.size();
    for i in 0..(out_dim - kernel_ndim) {
        if *output_size.at(i) < 0 {
            valid = false;
        }
    }
    for i in (out_dim - kernel_ndim)..out_dim {
        if *output_size.at(i) <= 0 {
            valid = false;
        }
    }
    if !valid {
        crate::et_log!(
            Error,
            "The provided combination of input and kernel parameters produces an invalid output size:"
        );
        for d in 0..output_size.size() {
            crate::et_log!(Error, "    size({}): {}", d, *output_size.at(d) as usize);
        }
    }
    valid
}

// [spec:et:def:kernel-ops-util.torch.executor.get-unsqueezed-sizes-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-sizes-fn]
///
/// # Safety
/// `sizes_arr` must point to at least `t.dim() + 1` valid `SizesType` elements.
pub unsafe fn get_unsqueezed_sizes(
    t: &Tensor,
    unsqueeze_dim: i64,
    sizes_arr: *mut SizesType,
    ndim: &mut usize,
) {
    unsafe {
        *ndim = (t.dim() + 1) as usize;
        for d in 0..unsqueeze_dim {
            *sizes_arr.add(d as usize) = t.size(d as ssize_t) as SizesType;
        }
        *sizes_arr.add(unsqueeze_dim as usize) = 1;
        for d in (unsqueeze_dim + 1)..(*ndim as i64) {
            *sizes_arr.add(d as usize) = t.size((d - 1) as ssize_t) as SizesType;
        }
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.get-unsqueezed-dim-order-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-dim-order-fn]
///
/// # Safety
/// `dim_order_arr` must point to at least `t.dim() + 1` valid `DimOrderType`
/// elements.
pub unsafe fn get_unsqueezed_dim_order(
    t: &Tensor,
    unsqueeze_dim: DimOrderType,
    dim_order_arr: *mut DimOrderType,
) {
    let mut offset: i32 = 0;
    for i in 0..t.dim() {
        let dim: DimOrderType = *t.dim_order().at(i as usize);
        if dim == unsqueeze_dim {
            unsafe {
                *dim_order_arr.add(i as usize) = dim;
                *dim_order_arr.add((i + 1) as usize) = dim + 1;
            }
            offset = 1;
        } else {
            unsafe {
                *dim_order_arr.add((i as i32 + offset) as usize) =
                    if dim > unsqueeze_dim { dim + 1 } else { dim };
            }
        }
    }
}

// PORT-NOTE: file-local helper in the C++ .cpp (`_kernel_output_size_helper`).
// [spec:et:def:kernel-ops-util.torch.executor.kernel-output-size-helper-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.kernel-output-size-helper-fn]
fn kernel_output_size_helper(
    input_size: usize,
    kernel_size: i64,
    pad: i64,
    stride: i64,
    dilation: i64,
    ceil_mode: bool,
    transposed: bool,
    output_padding: i64,
) -> i64 {
    if transposed {
        return (input_size as i64 - 1) * stride - 2 * pad
            + dilation * (kernel_size - 1)
            + output_padding
            + 1;
    }
    let numerator: i64 = input_size as i64 + 2 * pad - dilation * (kernel_size - 1) - 1
        + (if ceil_mode { stride - 1 } else { 0 });
    let mut output_size: i64 = numerator / stride + 1;
    if ceil_mode {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if (output_size - 1) * stride >= input_size as i64 + pad {
            output_size -= 1;
        }
    }
    output_size
}

/// Given an input tensor and N-dim kernel parameters, calculates the output size
/// of the N-dim kernel region.
// [spec:et:def:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.calculate-kernel-output-sizes-fn]
///
/// # Safety
/// `out_sizes` must point to valid `SizesType` elements at each index
/// `in.dim() - (kernel_ndim - i)` for `i` in `[0, kernel_ndim)`.
pub unsafe fn calculate_kernel_output_sizes(
    in_: &Tensor,
    kernel_ndim: usize,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    out_sizes: *mut SizesType,
    ceil_mode: bool,
    transposed: bool,
    output_padding: IntArrayRef,
) {
    for i in 0..kernel_ndim {
        let dim: ssize_t = in_.dim() - (kernel_ndim - i) as ssize_t;
        let k: i64 = val_at_default(kernel_size, i);
        let s: i64 = val_at(stride, i, k);
        let d: i64 = val_at(dilation, i, 1);
        let p: i64 = val_at(padding, i, 0);
        let op: i64 = if transposed {
            val_at(output_padding, i, 0)
        } else {
            0
        };

        unsafe {
            *out_sizes.add(dim as usize) = kernel_output_size_helper(
                in_.size(dim) as usize,
                k,
                p,
                s,
                d,
                ceil_mode,
                transposed,
                op,
            ) as SizesType;
        }
    }
}

//
// Utility functions to apply reduction over a N-dimensional kernel window
//

// PORT-NOTE: `kernel_reduction_then_map_2d` and `apply_kernel_2d_reduce_then_map_fn`
// are header-only C++ templates over `<CTYPE, ReduceOp, MapOp>`. `CTYPE` needs
// only a zero value here (`CTYPE accum = 0` / `CTYPE in_val = 0`); all other
// element operations go through the caller-supplied `reduce_fn`/`map_fn`. The
// zero is modeled via the local `KernelCtype` trait.
pub trait KernelCtype: Copy {
    fn zero() -> Self;
}
macro_rules! impl_kernel_ctype {
    ($($t:ty),*) => {$(
        impl KernelCtype for $t {
            fn zero() -> Self { 0 as $t }
        }
    )*};
}
impl_kernel_ctype!(u8, i8, i16, i32, i64, f32, f64);
impl KernelCtype for crate::runtime::core::portable_type::Half {
    fn zero() -> Self {
        crate::runtime::core::portable_type::Half::from_f32(0.0)
    }
}
impl KernelCtype for crate::runtime::core::portable_type::BFloat16 {
    fn zero() -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(0.0)
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.kernel-reduction-then-map-2d-fn]
///
/// # Safety
/// `in_ptr`/`out_ptr` must be valid data buffers for the described sweep;
/// `indices_ptr` must be valid for `out` or null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn kernel_reduction_then_map_2d<
    CTYPE: KernelCtype,
    ReduceOp: Fn(CTYPE, i64, CTYPE, i64) -> (CTYPE, i64),
    MapOp: Fn(i64, CTYPE) -> CTYPE,
>(
    reduce_fn: &ReduceOp,
    map_fn: &MapOp,
    include_pad: bool,
    in_ptr: *const CTYPE,
    in_sizes: ArrayRef<SizesType>,
    in_strides: ArrayRef<StridesType>,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    out_ptr: *mut CTYPE,
    out_sizes: ArrayRef<SizesType>,
    out_strides: ArrayRef<StridesType>,
    indices_ptr: *mut i64,
    batch: usize,
    out_c: usize,
) {
    let in_dim: usize = in_sizes.size();
    let out_dim: usize = out_sizes.size();

    let out_h: usize = *out_sizes.at(in_dim - 2) as usize;
    let in_h: usize = *in_sizes.at(in_dim - 2) as usize;

    let out_w: usize = *out_sizes.at(in_dim - 1) as usize;
    let in_w: usize = *in_sizes.at(in_dim - 1) as usize;

    let mut in_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    if in_dim == 4 {
        in_coord[0] = batch as SizesType;
        out_coord[0] = batch as SizesType;
    }
    in_coord[in_dim - 3] = out_c as SizesType;
    out_coord[in_dim - 3] = out_c as SizesType;

    let k_h: i64 = val_at_default(kernel_size, 0);
    let k_w: i64 = val_at_default(kernel_size, 1);
    let s_h: i64 = val_at(stride, 0, k_h);
    let s_w: i64 = val_at(stride, 1, k_w);
    let p_h: i64 = val_at(padding, 0, 0);
    let p_w: i64 = val_at(padding, 1, 0);
    let d_h: i64 = val_at(dilation, 0, 1);
    let d_w: i64 = val_at(dilation, 1, 1);

    // Compute 2D output region
    for out_y in 0..out_h {
        out_coord[in_dim - 2] = out_y as SizesType;
        for out_x in 0..out_w {
            out_coord[in_dim - 1] = out_x as SizesType;

            let mut accum_initialized: bool = false;
            let mut accum: CTYPE = CTYPE::zero();
            let mut accum_idx: i64 = 0;
            let mut count: i64 = 0;

            let mut ih0: i64 = out_y as i64 * s_h - p_h;
            let mut iw0: i64 = out_x as i64 * s_w - p_w;
            let mut ih1: i64 = core::cmp::min(ih0 + k_h, in_h as i64 + p_h);
            let mut iw1: i64 = core::cmp::min(iw0 + k_w, in_w as i64 + p_w);
            let pool_size: i64 = (ih1 - ih0) * (iw1 - iw0);
            ih0 = core::cmp::max(ih0, 0i64);
            iw0 = core::cmp::max(iw0, 0i64);
            ih1 = core::cmp::min(ih1, in_h as i64);
            iw1 = core::cmp::min(iw1, in_w as i64);

            if ih0 >= ih1 || iw0 >= iw1 {
                continue;
            }

            if include_pad {
                count = pool_size;
            } else {
                count = (ih1 - ih0) * (iw1 - iw0);
            }

            for w_y in 0..k_h {
                let stride_y: i64 = s_h;
                let padding_y: i64 = p_h;
                let dilation_y: i64 = d_h;

                let in_y: isize = (stride_y * out_y as i64 + dilation_y * w_y - padding_y) as isize;
                in_coord[in_dim - 2] = in_y as SizesType;

                for w_x in 0..k_w {
                    let stride_x: i64 = s_w;
                    let padding_x: i64 = p_w;
                    let dilation_x: i64 = d_w;

                    let in_x: isize =
                        (stride_x * out_x as i64 + dilation_x * w_x - padding_x) as isize;
                    in_coord[in_dim - 1] = in_x as SizesType;

                    let x_in_bound: bool = in_x >= 0 && in_x < in_w as isize;
                    let y_in_bound: bool = in_y >= 0 && in_y < in_h as isize;
                    let xy_in_bound: bool = x_in_bound && y_in_bound;

                    let mut in_val: CTYPE = CTYPE::zero();
                    if xy_in_bound {
                        let in_idx: usize = unsafe {
                            calculate_linear_index(in_coord.as_ptr(), in_strides.data(), in_dim)
                        };
                        in_val = unsafe { *in_ptr.add(in_idx) };
                    }

                    let mut idx: i64 = in_y as i64 * in_w as i64 + in_x as i64;
                    if include_pad {
                        idx = in_y as i64
                            + padding_y * (in_w as i64 + 2 * padding_x)
                            + (in_x as i64 + padding_x);
                    }

                    if xy_in_bound {
                        if !accum_initialized {
                            accum = in_val;
                            accum_idx = idx;
                            accum_initialized = true;
                        } else {
                            let ret: (CTYPE, i64) = reduce_fn(in_val, idx, accum, accum_idx);
                            accum = ret.0;
                            accum_idx = ret.1;
                        }
                    }
                }
            }

            let out_idx: usize =
                unsafe { calculate_linear_index(out_coord.as_ptr(), out_strides.data(), out_dim) };
            unsafe {
                *out_ptr.add(out_idx) = map_fn(count, accum);
            }
            if !indices_ptr.is_null() {
                unsafe {
                    *indices_ptr.add(out_idx) = accum_idx;
                }
            }
        }
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.apply-kernel-2d-reduce-then-map-fn-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.apply-kernel-2d-reduce-then-map-fn-fn]
#[allow(clippy::too_many_arguments)]
pub fn apply_kernel_2d_reduce_then_map_fn<
    CTYPE: KernelCtype,
    ReduceOp: Fn(CTYPE, i64, CTYPE, i64) -> (CTYPE, i64),
    MapOp: Fn(i64, CTYPE) -> CTYPE,
>(
    reduce_fn: &ReduceOp,
    map_fn: &MapOp,
    include_pad: bool,
    in_: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    out: &Tensor,
    indices: Option<Tensor>,
) {
    let in_sizes: ArrayRef<SizesType> = in_.sizes();
    let out_sizes: ArrayRef<SizesType> = out.sizes();

    let in_dim_order: ArrayRef<DimOrderType> = in_.dim_order();
    let out_dim_order: ArrayRef<DimOrderType> = out.dim_order();

    let mut in_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        dim_order_to_stride_nocheck(
            in_sizes.data(),
            in_dim_order.data(),
            in_sizes.size(),
            in_strides.as_mut_ptr(),
        );
    }

    let mut out_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        dim_order_to_stride_nocheck(
            out_sizes.data(),
            out_dim_order.data(),
            out_sizes.size(),
            out_strides.as_mut_ptr(),
        );
    }

    let out_ptr: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();
    let in_ptr: *const CTYPE = in_.const_data_ptr::<CTYPE>();

    let mut indices_ptr: *mut i64 = core::ptr::null_mut();
    if let Some(ref indices) = indices {
        indices_ptr = indices.mutable_data_ptr::<i64>();
    }

    let mut batch_size: usize = 1;
    if in_.dim() == 4 {
        batch_size = *in_sizes.at(0) as usize;
    }
    for batch in 0..batch_size {
        for channel in 0..(*in_sizes.at((in_.dim() - 3) as usize) as usize) {
            unsafe {
                kernel_reduction_then_map_2d(
                    reduce_fn,
                    map_fn,
                    include_pad,
                    in_ptr,
                    in_sizes,
                    ArrayRef::from_raw_parts(in_strides.as_ptr(), 4),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    out_ptr,
                    out_sizes,
                    ArrayRef::from_raw_parts(out_strides.as_ptr(), 4),
                    indices_ptr,
                    batch,
                    channel,
                );
            }
        }
    }
}

//
// Operator specific utility functions
//

// [spec:et:def:kernel-ops-util.torch.executor.check-arange-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-arange-args-fn]
pub fn check_arange_args(start: f64, end: f64, step: f64, out: &Tensor) -> bool {
    et_check_or_return_false!(
        out.dim() == 1,
        "out should be a 1-d tensor, but got a {}-d tensor",
        out.dim()
    );

    et_check_or_return_false!(
        (step > 0.0 && (end >= start)) || (step < 0.0 && (end <= start)),
        "upper bound and larger bound inconsistent with step sign; step = {:.6}, start = {:.6}, end = {:.6}",
        step,
        start,
        end
    );

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn]
pub fn check_adaptive_avg_pool2d_args(
    in_: &Tensor,
    output_size: IntArrayRef,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));

    et_check_or_return_false!(
        (in_.dim() == 3 && in_.size(0) > 0 && in_.size(1) > 0 && in_.size(2) > 0)
            || (in_.dim() == 4 && in_.size(1) > 0 && in_.size(2) > 0 && in_.size(3) > 0),
        "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input; in.dim() = {}",
        in_.dim()
    );

    et_check_or_return_false!(
        output_size.size() == 2,
        "output_size must have exactly 2 elements, but got {}",
        output_size.size()
    );

    et_check_or_return_false!(
        *output_size.at(0) > 0 && *output_size.at(1) > 0,
        "output_size must be positive, but got ({}, {})",
        *output_size.at(0),
        *output_size.at(1)
    );

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.get-adaptive-avg-pool2d-out-target-size-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-adaptive-avg-pool2d-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in.dim()` valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_adaptive_avg_pool2d_out_target_size(
    in_: &Tensor,
    output_size: IntArrayRef,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;

        if in_.dim() == 4 {
            *out_sizes.add(0) = in_.size(0) as SizesType;
            *out_sizes.add(1) = in_.size(1) as SizesType;
        } else {
            *out_sizes.add(0) = in_.size(0) as SizesType;
        }

        *out_sizes.add(*out_ndim - 2) = *output_size.at(0) as SizesType;
        *out_sizes.add(*out_ndim - 1) = *output_size.at(1) as SizesType;
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-avg-pool2d-args-fn]
pub fn check_avg_pool2d_args(
    in_: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    _ceil_mode: bool,
    _count_include_pad: bool,
    divisor_override: &Option<i64>,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));

    et_check_or_return_false!(
        (in_.dim() == 3 && in_.size(0) > 0 && in_.size(1) > 0 && in_.size(2) > 0)
            || (in_.dim() == 4 && in_.size(1) > 0 && in_.size(2) > 0 && in_.size(3) > 0),
        "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input; in.dim() = {}",
        in_.dim()
    );

    et_log_and_return_if_false!(kernel_size_is_valid(kernel_size, 2));
    // PORT-NOTE: preexisting C++ quirk — `kernel_size` (not `stride`) is passed
    // as the array to validate, so the real `stride` argument is not shape/
    // min-value checked here. Reproduced for conformance.
    et_log_and_return_if_false!(stride_is_valid(kernel_size, 2, true));
    et_log_and_return_if_false!(padding_is_valid(padding, kernel_size, 2, true));

    if let Some(divisor_override) = divisor_override {
        et_check_or_return_false!(
            *divisor_override != 0,
            "divisor_override must be non-zero, but found {}",
            *divisor_override
        );
    }

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.get-avg-pool2d-out-target-size-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-avg-pool2d-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in.dim()` valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_avg_pool2d_out_target_size(
    in_: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    ceil_mode: bool,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;

        // Batch dim is optional, so in can be either 3 or 4 dim.
        if in_.dim() == 4 {
            *out_sizes.add(0) = in_.size(0) as SizesType;
            *out_sizes.add(1) = in_.size(1) as SizesType;
        } else {
            *out_sizes.add(0) = in_.size(0) as SizesType;
        }

        calculate_kernel_output_sizes(
            in_,
            2,
            kernel_size,
            stride,
            padding,
            ArrayRef::new(),
            out_sizes,
            ceil_mode,
            false,
            ArrayRef::new(),
        );
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.check-convolution-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-convolution-args-fn]
pub fn check_convolution_args(
    in_: &Tensor,
    weight: &Tensor,
    bias: &Option<Tensor>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    output_padding: IntArrayRef,
    groups: i64,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype(in_, weight, out));

    et_check_or_return_false!(
        in_.dim() == 3 || in_.dim() == 4 || in_.dim() == 5,
        "Expect input tensor to be 3-D, 4-D or 5-D, but got, {}.",
        in_.dim() as usize
    );
    et_log_and_return_if_false!(tensor_is_rank(weight, in_.dim() as usize));
    et_log_and_return_if_false!(tensor_is_rank(out, in_.dim() as usize));

    if in_.dim() == 5 {
        et_log_and_return_if_false!(tensor_is_default_dim_order(in_));
        et_log_and_return_if_false!(tensor_is_default_dim_order(weight));
        et_log_and_return_if_false!(tensor_is_default_dim_order(out));
        et_check_or_return_false!(
            !transposed,
            "Transposed 3D convolution is not yet supported on portable."
        );
    } else {
        et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
        et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(weight));
        et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));
    }

    if let Some(bias) = bias {
        et_log_and_return_if_false!(tensor_is_rank(bias, 1));
        // PORT-NOTE: latent C++ bug reproduced for conformance. Due to C
        // operator precedence (`==` binds tighter than `?:`) the source
        // `bias.value().size(0) == transposed ? groups * weight.size(1) :
        // weight.size(0)` parses as
        // `(bias.size(0) == transposed) ? (groups*weight.size(1)) : weight.size(0)`.
        // `ET_CHECK_OR_RETURN_FALSE` then tests `!(<that integer>)`, so the check
        // only fails when the selected branch evaluates to 0 — i.e. it passes for
        // essentially every real conv shape. The condition is the *integer ternary
        // result* treated as a boolean, not the `==` comparison.
        et_check_or_return_false!(
            (if bias.size(0) as i64 == (transposed as i64) {
                groups * weight.size(1) as i64
            } else {
                weight.size(0) as i64
            }) != 0,
            "bias length must equal number of output channels, but got {}; expected {}",
            bias.size(0),
            if transposed {
                groups * weight.size(1) as i64
            } else {
                weight.size(0) as i64
            }
        );
    }

    let mut kernel_size: [i64; 3] = [0; 3];
    let mut kernel_ndim: usize = 0;
    unsafe {
        fill_convolution_kernel_size(weight, kernel_size.as_mut_ptr(), &mut kernel_ndim);
    }
    et_log_and_return_if_false!(kernel_size_is_valid(
        ArrayRef::from_raw_parts(kernel_size.as_ptr(), kernel_ndim),
        kernel_ndim
    ));
    et_log_and_return_if_false!(stride_is_valid(stride, kernel_ndim, false));
    et_log_and_return_if_false!(padding_is_valid(
        padding,
        ArrayRef::from_raw_parts(kernel_size.as_ptr(), kernel_ndim),
        kernel_ndim,
        false
    ));
    et_log_and_return_if_false!(dilation_is_valid(dilation, kernel_ndim));
    if transposed {
        et_log_and_return_if_false!(output_padding_is_valid(
            output_padding,
            stride,
            dilation,
            kernel_ndim
        ));
    }

    et_check_or_return_false!(
        weight.size(0) as i64 >= groups,
        "Given groups={}, expected weight to be at least {} at dimension 0, but got weight.size(0) = {} instead",
        groups,
        groups,
        weight.size(0)
    );
    et_check_or_return_false!(
        weight.size(0) as i64 % groups == 0,
        "Given groups={}, expected weight to be divisible by {} at dimension 0, but got weight.size(0) = {} instead",
        groups,
        groups,
        weight.size(0)
    );

    if !transposed {
        et_check_or_return_false!(
            in_.size(1) as i64 == groups * weight.size(1) as i64,
            "Given groups={} and weight.size(1) = {}, expected input to have {} channels, but got {}",
            groups,
            weight.size(1),
            groups * weight.size(1) as i64,
            in_.size(1)
        );
    } else {
        et_check_or_return_false!(
            in_.size(1) == weight.size(0),
            "input channels must match weight.size(0) in transposed convolution; in.size(1) = {}, weight.size(0) = {}",
            in_.size(1),
            weight.size(0)
        );
    }

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.get-convolution-out-target-size-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-convolution-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in.dim()` valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_convolution_out_target_size(
    in_: &Tensor,
    weight: &Tensor,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    output_padding: IntArrayRef,
    groups: i64,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;

        // batch dim
        *out_sizes.add(0) = in_.size(0) as SizesType;

        // channel dim
        if !transposed {
            *out_sizes.add(1) = if in_.size(1) == 0 {
                0
            } else {
                weight.size(0) as SizesType
            };
        } else {
            *out_sizes.add(1) = if in_.size(1) == 0 {
                0
            } else {
                (groups * weight.size(1) as i64) as SizesType
            };
        }

        let mut kernel_size: [i64; 3] = [0; 3];
        let mut kernel_ndim: usize = 0;
        fill_convolution_kernel_size(weight, kernel_size.as_mut_ptr(), &mut kernel_ndim);
        calculate_kernel_output_sizes(
            in_,
            kernel_ndim,
            ArrayRef::from_raw_parts(kernel_size.as_ptr(), kernel_ndim),
            stride,
            padding,
            dilation,
            out_sizes,
            false,
            transposed,
            output_padding,
        );
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.check-cumsum-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-cumsum-args-fn]
pub fn check_cumsum_args(in_: &Tensor, dim: i64, dtype: Option<ScalarType>, out: &Tensor) -> bool {
    et_log_and_return_if_false!(dim_is_valid(dim, in_.dim() as i64));

    if let Some(dtype) = dtype {
        et_log_and_return_if_false!(dtype == out.scalar_type());
    }

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.check-max-pool2d-with-indices-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-max-pool2d-with-indices-args-fn]
pub fn check_max_pool2d_with_indices_args(
    in_: &Tensor,
    kernel_size: IntArrayRef,
    _stride: IntArrayRef,
    padding: IntArrayRef,
    _dilation: IntArrayRef,
    _ceil_mode: bool,
    out: &Tensor,
    indices: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_check_or_return_false!(
        indices.scalar_type() == ScalarType::Long,
        "Expected indices to have type of Long, but found {}",
        to_string(indices.scalar_type())
    );

    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));

    et_check_or_return_false!(
        (in_.dim() == 3 && in_.size(0) > 0 && in_.size(1) > 0 && in_.size(2) > 0)
            || (in_.dim() == 4 && in_.size(1) > 0 && in_.size(2) > 0 && in_.size(3) > 0),
        "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input; in.dim() = {}",
        in_.dim()
    );

    et_log_and_return_if_false!(kernel_size_is_valid(kernel_size, 2));
    // PORT-NOTE: preexisting C++ quirk — `kernel_size` (not `stride`) is passed.
    et_log_and_return_if_false!(stride_is_valid(kernel_size, 2, true));
    et_log_and_return_if_false!(padding_is_valid(padding, kernel_size, 2, true));
    // PORT-NOTE: preexisting C++ quirk — `kernel_size` (not `dilation`) is passed.
    et_log_and_return_if_false!(dilation_is_valid(kernel_size, 2));

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.get-max-pool2d-with-indices-out-target-size-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.get-max-pool2d-with-indices-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in.dim()` valid `SizesType` elements and
/// `out_ndim` to a valid `usize`.
pub unsafe fn get_max_pool2d_with_indices_out_target_size(
    in_: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    ceil_mode: bool,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;

        // Batch dim is optional, so in can be either 3 or 4 dim.
        if in_.dim() == 4 {
            *out_sizes.add(0) = in_.size(0) as SizesType;
            *out_sizes.add(1) = in_.size(1) as SizesType;
        } else {
            *out_sizes.add(0) = in_.size(0) as SizesType;
        }

        calculate_kernel_output_sizes(
            in_,
            2,
            kernel_size,
            stride,
            padding,
            dilation,
            out_sizes,
            ceil_mode,
            false,
            ArrayRef::new(),
        );
    }
}

// [spec:et:def:kernel-ops-util.torch.executor.check-masked-fill-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-masked-fill-args-fn]
pub fn check_masked_fill_args(in_: &Tensor, mask: &Tensor, _value: &Scalar, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(mask.scalar_type() == ScalarType::Bool);

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.check-constant-pad-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-constant-pad-args-fn]
pub fn check_constant_pad_args(
    in_: &Tensor,
    pad: IntArrayRef,
    _value: &Scalar,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));

    et_log_and_return_if_false!(tensors_have_same_rank(in_, out));

    et_check_or_return_false!(
        pad.size() % 2 == 0,
        "Padding array must be a multiple of 2; pad.size() = {}",
        pad.size()
    );

    et_check_or_return_false!(
        (pad.size() / 2) as ssize_t <= in_.dim(),
        "Padding array contains too many elements; pad.size()/2 = {}, in.dim() = {}",
        pad.size() / 2,
        in_.dim()
    );

    for i in 0..pad.size() {
        et_check_or_return_false!(
            *pad.at(i) >= 0,
            "Padding values must be non-negative, but got pad[{}] = {}",
            i,
            *pad.at(i)
        );
    }

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.resize-constant-pad-output-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.resize-constant-pad-output-fn]
#[must_use]
pub fn resize_constant_pad_output(in_: &Tensor, pad: IntArrayRef, out: &Tensor) -> Error {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];

    let mut pad_i: i32 = (in_.dim() - 1) as i32;
    for i in 0..in_.dim() {
        expected_output_size[i as usize] = in_.size(i) as SizesType;
        if pad_i >= 0 && (pad_i as usize) < pad.size() / 2 {
            expected_output_size[i as usize] +=
                (*pad.at((2 * pad_i) as usize) + *pad.at((2 * pad_i + 1) as usize)) as SizesType;
        }
        pad_i -= 1;
    }

    let output_size: ArrayRef<SizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), in_.dim() as usize);
    let error = resize_tensor_same_type(out, output_size);

    error
}

// [spec:et:def:kernel-ops-util.torch.executor.check-embedding-args-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-embedding-args-fn]
pub fn check_embedding_args(weight: &Tensor, indices: &Tensor, out: &Tensor) -> bool {
    // Ensure weight is 2-D. It could be empty.
    et_check_or_return_false!(weight.dim() == 2, "weight.dim() {} != 2", weight.dim());

    // Ensure out is k+1 dimension tensor where k is the indices.dim()
    // out's first k dimension shall be same as indices, and the last dim shall
    // equal weight's last dim
    et_check_or_return_false!(
        out.dim() == indices.dim() + 1,
        "out.dim() {} != indices.dim() {} + 1",
        out.dim(),
        indices.dim()
    );

    // Ensure dtype is the same for out and weight
    et_log_and_return_if_false!(tensors_have_same_dtype2(weight, out));

    true
}

// [spec:et:def:kernel-ops-util.torch.executor.resize-embedding-output-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.resize-embedding-output-fn]
#[must_use]
pub fn resize_embedding_output(weight: &Tensor, indices: &Tensor, out: &Tensor) -> Error {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    for i in 0..indices.dim() {
        expected_output_size[i as usize] = indices.size(i) as SizesType;
    }
    let embedding_dim: usize = weight.size(1) as usize;
    expected_output_size[(out.dim() - 1) as usize] = embedding_dim as SizesType;

    let output_size: ArrayRef<SizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize);

    resize_tensor_same_type(out, output_size)
}

// [spec:et:def:kernel-ops-util.torch.executor.check-alpha-type-fn]
// [spec:et:sem:kernel-ops-util.torch.executor.check-alpha-type-fn]
pub fn check_alpha_type(alpha_type: ScalarType, common_type: ScalarType) -> bool {
    // Verify that alpha type is compatible with common type,
    // as used by ops such as add and sub.
    et_log_and_return_if_false!(
        can_cast(alpha_type, common_type)
            || (common_type == ScalarType::Bool && is_integral_type(alpha_type, true))
    );

    true
}
