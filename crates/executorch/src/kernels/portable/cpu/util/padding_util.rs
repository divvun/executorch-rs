//! Literal port of kernels/portable/cpu/util/padding_util.cpp + kernels/portable/cpu/util/padding_util.h.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::{
    getLeadingDims, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: local `et_log_and_return_if_false!` mirroring the C++
// `ET_LOG_AND_RETURN_IF_FALSE(cond)` (== `ET_CHECK_OR_RETURN_FALSE(cond, "")`),
// same reasoning as tensor_util.rs (the crate-level check macro drops format
// args).
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}

// PORT-NOTE: `ET_CHECK(cond)` fatally aborts. Mirrors the local `et_check!`
// pattern used in tensor_util.rs / scalar_type_util.rs.
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:padding-util.torch.executor.check-padding-args-fn]
// [spec:et:sem:padding-util.torch.executor.check-padding-args-fn]
pub fn check_padding_args(
    n: i64,
    in_: &Tensor,
    padding: ArrayRef<i64>,
    out: &Tensor,
    reflection: bool,
) -> bool {
    et_log_and_return_if_false!(padding.size() as i64 == 2 * n);
    et_log_and_return_if_false!(in_.dim() as i64 == n + 1 || in_.dim() as i64 == n + 2);
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    for i in 1..(n + 1) {
        et_log_and_return_if_false!(
            in_.size((in_.dim() as i64 - i) as ssize_t) as i64
                + *padding.at((2 * i - 2) as usize)
                + *padding.at((2 * i - 1) as usize)
                >= 0
        );
        if reflection {
            et_log_and_return_if_false!(
                *padding.at((2 * i - 2) as usize)
                    < in_.size((in_.dim() as i64 - i) as ssize_t) as i64
                    && *padding.at((2 * i - 1) as usize)
                        < in_.size((in_.dim() as i64 - i) as ssize_t) as i64
            );
        }
    }
    true
}

// [spec:et:def:padding-util.torch.executor.get-padding-out-target-size-fn]
// [spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]
///
/// # Safety
/// `out_sizes` must point to at least `in_.dim()` valid `SizesType` elements
/// and `out_ndim` to a valid `usize`.
pub unsafe fn get_padding_out_target_size(
    n: i64,
    in_: &Tensor,
    padding: ArrayRef<i64>,
    out_sizes: *mut SizesType,
    out_ndim: *mut usize,
) {
    unsafe {
        *out_ndim = in_.dim() as usize;
        for i in 0..in_.dim() {
            *out_sizes.add(i as usize) = in_.size(i) as SizesType;
        }
        for i in 1..(n + 1) {
            *out_sizes.add((in_.dim() as i64 - i) as usize) =
                (in_.size((in_.dim() as i64 - i) as ssize_t) as i64
                    + *padding.at((2 * i - 2) as usize)
                    + *padding.at((2 * i - 1) as usize)) as SizesType;
        }
    }
}

// [spec:et:def:padding-util.torch.executor.replication-ix-fn]
// [spec:et:sem:padding-util.torch.executor.replication-ix-fn]
pub fn replication_ix(j: i64, size: i64, pad: i64) -> i64 {
    if j < pad {
        0
    } else if j >= pad && j < size + pad {
        j - pad
    } else {
        size - 1
    }
}

// [spec:et:def:padding-util.torch.executor.reflection-ix-fn]
// [spec:et:sem:padding-util.torch.executor.reflection-ix-fn]
pub fn reflection_ix(j: i64, size: i64, pad: i64) -> i64 {
    if j < pad {
        pad - j
    } else if j >= pad && j < size + pad {
        j - pad
    } else {
        2 * size + pad - j - 2
    }
}

// [spec:et:def:padding-util.torch.executor.pad1d-fn]
// [spec:et:sem:padding-util.torch.executor.pad1d-fn]
// PORT-NOTE: `PaddingIx` is a C++ callable `(out_index, in_size, pad_low) ->
// in_index`; modeled as `impl Fn(i64, i64, i64) -> i64`. `CTYPE` is copied
// verbatim (pure copy, no arithmetic) so only `Copy` is required.
pub fn pad1d<CTYPE: Copy, PaddingIx: Fn(i64, i64, i64) -> i64>(
    padding_ix: &PaddingIx,
    in_: &Tensor,
    out: &Tensor,
    padding: ArrayRef<i64>,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let dim = in_.dim() - 1;
    let outer = getLeadingDims(out, dim as i64);
    let in_width = in_.size(dim);
    let out_width = out.size(dim);
    let pad_left = *padding.at(0);
    for i in 0..outer {
        let out_i_base: usize = i * out_width as usize;
        let in_i_base: usize = i * in_width as usize;
        for w in 0..out_width {
            let in_w_idx: i64 = padding_ix(w as i64, in_width as i64, pad_left);
            et_check!(in_w_idx >= 0 && in_w_idx < in_width as i64);
            unsafe {
                *out_data.add(out_i_base + w as usize) =
                    *in_data.add(in_i_base + in_w_idx as usize);
            }
        }
    }
}

// [spec:et:def:padding-util.torch.executor.pad2d-fn]
// [spec:et:sem:padding-util.torch.executor.pad2d-fn]
pub fn pad2d<CTYPE: Copy, PaddingIx: Fn(i64, i64, i64) -> i64>(
    padding_ix: &PaddingIx,
    in_: &Tensor,
    out: &Tensor,
    padding: ArrayRef<i64>,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let dim = in_.dim() - 2;
    let outer = getLeadingDims(out, dim as i64);
    let in_height = in_.size(dim);
    let in_width = in_.size(dim + 1);
    let out_height = out.size(dim);
    let out_width = out.size(dim + 1);
    let pad_left = *padding.at(0);
    let pad_top = *padding.at(2);

    for i in 0..outer {
        let out_i_base: usize = i * out_height as usize * out_width as usize;
        let in_i_base: usize = i * in_height as usize * in_width as usize;
        for h in 0..out_height {
            let out_h_base: usize = out_i_base + h as usize * out_width as usize;
            let in_h_idx: i64 = padding_ix(h as i64, in_height as i64, pad_top);
            et_check!(in_h_idx >= 0 && in_h_idx < in_height as i64);
            let in_h_base: usize = in_i_base + in_h_idx as usize * in_width as usize;
            for w in 0..out_width {
                let in_w_idx: i64 = padding_ix(w as i64, in_width as i64, pad_left);
                et_check!(in_w_idx >= 0 && in_w_idx < in_width as i64);
                unsafe {
                    *out_data.add(out_h_base + w as usize) =
                        *in_data.add(in_h_base + in_w_idx as usize);
                }
            }
        }
    }
}

// [spec:et:def:padding-util.torch.executor.pad3d-fn]
// [spec:et:sem:padding-util.torch.executor.pad3d-fn]
pub fn pad3d<CTYPE: Copy, PaddingIx: Fn(i64, i64, i64) -> i64>(
    padding_ix: &PaddingIx,
    in_: &Tensor,
    out: &Tensor,
    padding: ArrayRef<i64>,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let dim = in_.dim() - 3;
    let outer = getLeadingDims(out, dim as i64);
    let in_depth = in_.size(dim);
    let in_height = in_.size(dim + 1);
    let in_width = in_.size(dim + 2);
    let out_depth = out.size(dim);
    let out_height = out.size(dim + 1);
    let out_width = out.size(dim + 2);
    let pad_left = *padding.at(0);
    let pad_top = *padding.at(2);
    let pad_front = *padding.at(4);

    for i in 0..outer {
        let out_i_base: usize = i * out_depth as usize * out_height as usize * out_width as usize;
        let in_i_base: usize = i * in_depth as usize * in_height as usize * in_width as usize;
        for d in 0..out_depth {
            let out_d_base: usize =
                out_i_base + d as usize * out_height as usize * out_width as usize;
            let in_d_base_padding: i64 = padding_ix(d as i64, in_depth as i64, pad_front);
            et_check!(in_d_base_padding >= 0 && in_d_base_padding < in_depth as i64);
            let in_d_base: usize =
                in_i_base + in_d_base_padding as usize * in_height as usize * in_width as usize;
            for h in 0..out_height {
                let out_h_base: usize = out_d_base + h as usize * out_width as usize;
                let in_h_base_padding: i64 = padding_ix(h as i64, in_height as i64, pad_top);
                et_check!(in_h_base_padding >= 0 && in_h_base_padding < in_height as i64);
                let in_h_base: usize = in_d_base + in_h_base_padding as usize * in_width as usize;
                for w in 0..out_width {
                    let in_w_base_padding: i64 = padding_ix(w as i64, in_width as i64, pad_left);
                    et_check!(in_w_base_padding >= 0 && in_w_base_padding < in_width as i64);
                    unsafe {
                        *out_data.add(out_h_base + w as usize) =
                            *in_data.add(in_h_base + in_w_base_padding as usize);
                    }
                }
            }
        }
    }
}
