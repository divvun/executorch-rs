//! Literal port of kernels/portable/cpu/util/slice_util.cpp.

use crate::runtime::core::exec_aten::util::tensor_util::{
    dim_is_valid, getLeadingDims, getTrailingDims, tensor_has_dim, tensors_have_same_dtype2,
    tensors_have_same_rank, tensors_have_same_shape_and_dtype2, tensors_have_same_size_at_dims,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface;

// PORT-NOTE: the crate-level `et_check_or_return_false!` (runtime/core/error.rs)
// drops all caller-supplied format arguments after the leading literal. This
// local override mirrors the C++ `ET_CHECK_OR_RETURN_FALSE` faithfully.
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

// [spec:et:def:slice-util.torch.executor.check-narrow-copy-args-fn]
// [spec:et:sem:slice-util.torch.executor.check-narrow-copy-args-fn]
pub fn check_narrow_copy_args(
    in_: &Tensor,
    dim: i64,
    mut start: i64,
    length: i64,
    out: &Tensor,
) -> bool {
    et_log_and_return_if_false!(in_.dim() > 0);
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    et_check_or_return_false!(
        length >= 0,
        "length must be non-negative; length = {}",
        length
    );
    et_log_and_return_if_false!(start >= -(in_.size(dim as ssize_t) as i64));
    et_log_and_return_if_false!(start <= in_.size(dim as ssize_t) as i64);
    if start < 0 {
        start += in_.size(dim as ssize_t) as i64;
    }
    et_log_and_return_if_false!(start + length <= in_.size(dim as ssize_t) as i64);
    true
}

// [spec:et:def:slice-util.torch.executor.get-narrow-copy-out-target-size-fn]
// [spec:et:sem:slice-util.torch.executor.get-narrow-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_narrow_copy_out_target_size(
    in_: &Tensor,
    dim: i64,
    length: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    *out_ndim = in_.dim() as usize;

    for d in 0..in_.dim() {
        unsafe { *out_sizes.add(d as usize) = in_.size(d) as SizesType };
    }
    unsafe { *out_sizes.add(dim as usize) = length as SizesType };
}

// [spec:et:def:slice-util.torch.executor.check-slice-copy-args-fn]
// [spec:et:sem:slice-util.torch.executor.check-slice-copy-args-fn]
pub fn check_slice_copy_args(in_: &Tensor, dim: i64, step: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(in_.dim() > 0);
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    et_check_or_return_false!(
        step > 0,
        "slice step must be greater than zero; step = {}",
        step
    );
    true
}

// [spec:et:def:slice-util.torch.executor.get-slice-copy-out-target-size-fn]
// [spec:et:sem:slice-util.torch.executor.get-slice-copy-out-target-size-fn]
//
// # Safety
// `out_sizes` must point to at least `in.dim()` writable elements.
pub unsafe fn get_slice_copy_out_target_size(
    in_: &Tensor,
    dim: i64,
    length: i64,
    out_sizes: *mut SizesType,
    out_ndim: &mut usize,
) {
    unsafe { get_narrow_copy_out_target_size(in_, dim, length, out_sizes, out_ndim) };
}

// [spec:et:def:slice-util.torch.executor.check-slice-scatter-args-fn]
// [spec:et:sem:slice-util.torch.executor.check-slice-scatter-args-fn]
//
// PORT-NOTE: C++ takes `Tensor output` by value; the ported `Tensor` handle is
// passed by shared reference here.
pub fn check_slice_scatter_args(
    input: &Tensor,
    src: &Tensor,
    dim: i64,
    num_values: i64,
    step: i64,
    output: &Tensor,
) -> bool {
    et_log_and_return_if_false!(input.dim() > 0);

    // Check dim. The dim planned to be selected on shall exist in input
    et_log_and_return_if_false!(dim_is_valid(dim, input.dim() as i64));

    // Input and output tensors should be the same shape and dtype
    et_log_and_return_if_false!(tensors_have_same_shape_and_dtype2(input, output));

    // The input.dim() shall equal to src.dim()
    et_log_and_return_if_false!(tensors_have_same_rank(input, src));

    // Check step. Step must be greater than zero
    et_check_or_return_false!(
        step > 0,
        "slice step must be greater than zero; step = {}",
        step
    );

    // The size of src tensor should follow these rules:
    // - src.size(i) shall equal to input.size(i) if i != dim,
    // - src.size(dim) shall equal to num_values
    for d in 0..input.dim() {
        if d as i64 != dim {
            et_log_and_return_if_false!(tensors_have_same_size_at_dims(
                input, d as usize, src, d as usize
            ));
        } else {
            et_check_or_return_false!(
                src.size(d) as i64 == num_values,
                "input.size({}) {} != num_values {} | dim = {})",
                d,
                input.size(d),
                num_values,
                dim
            );
        }
    }

    true
}

// [spec:et:def:slice-util.torch.executor.adjust-slice-indices-fn]
// [spec:et:sem:slice-util.torch.executor.adjust-slice-indices-fn]
pub fn adjust_slice_indices(dim_length: i64, start: &mut i64, end: &mut i64, step: i64) -> i64 {
    let num_values: i64;

    // Update start and end index
    // First convert it to c++ style from python style if needed.
    *start = if *start < 0 {
        *start + dim_length
    } else {
        *start
    };
    *end = if *end < 0 { *end + dim_length } else { *end };
    // Second, if start or end still negative, which means user want to start or
    // end slicing from very beginning, so set it to zero
    *start = if *start < 0 { 0 } else { *start };
    *end = if *end < 0 { 0 } else { *end };
    // Last, if start or end larger than maximum value (dim_length - 1), indicates
    // user want to start slicing after end or slicing until the end, so update it
    // to dim_length
    *start = if *start > dim_length {
        dim_length
    } else {
        *start
    };
    *end = if *end > dim_length { dim_length } else { *end };

    if *start >= dim_length || *end <= 0 || *start >= *end {
        // Set num_values to 0 if interval [start, end) is non-exist or do not
        // overlap with [0, dim_length)
        num_values = 0;
    } else {
        // Update num_values to min(max_num_values, num_values)
        num_values = (*end - 1 - *start) / step + 1;
    }
    num_values
}

// [spec:et:def:slice-util.torch.executor.compute-slice-fn]
// [spec:et:sem:slice-util.torch.executor.compute-slice-fn]
pub fn compute_slice(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    start: i64,
    length: i64,
    step: i64,
    out: &Tensor,
) {
    // No slicing requested.
    if length <= 0 {
        return;
    }

    crate::et_kernel_check_msg!(
        ctx,
        dim < in_.dim() as i64,
        InvalidArgument,
        (),
        "Requested dim is larger than input tensor dim"
    );
    let dim_length: usize = in_.size(dim as ssize_t) as usize;
    crate::et_kernel_check_msg!(
        ctx,
        start >= 0 && length >= 0 && step >= 0,
        InvalidArgument,
        (),
        "Input args should be >= 0."
    );
    let requested_slice: i64 = start + (length - 1) * step;
    crate::et_kernel_check_msg!(
        ctx,
        (requested_slice as u64) < (dim_length as u64),
        InvalidArgument,
        (),
        "Requested slice is larger than the dim size"
    );

    let leading_dims: usize = getLeadingDims(in_, dim);
    let trailing_dims: usize = getTrailingDims(in_, dim);

    if trailing_dims == 0 {
        return;
    }

    let length_per_step: usize = trailing_dims * in_.element_size() as usize;

    let input_data: *const u8 = in_.const_data_ptr::<u8>();
    let dest: *mut u8 = out.mutable_data_ptr::<u8>();

    crate::et_kernel_check_msg!(
        ctx,
        out.nbytes() >= (length as usize * leading_dims * length_per_step),
        InvalidArgument,
        (),
        "out.nbytes() is smaller than the expected slice size."
    );
    // Thresholds for enabling multithreading:
    // - Minimum number of leading dimensions: 8
    // - Minimum total elements to copy: 32768 (GRAIN_SIZE)
    const MIN_LEADING_DIMS_FOR_MT: i64 = 8;
    const MIN_ELEMENTS_FOR_MT: i64 = thread_parallel_interface::internal::GRAIN_SIZE;

    let total_elements: i64 = (leading_dims * length as usize * trailing_dims) as i64;
    let use_multithreading: bool =
        leading_dims as i64 >= MIN_LEADING_DIMS_FOR_MT && total_elements >= MIN_ELEMENTS_FOR_MT;

    if use_multithreading {
        // Use parallel_for to distribute work across leading dimensions
        // Calculate grain size based on number of elements per leading dimension
        let grain_size: i64 = MIN_LEADING_DIMS_FOR_MT;

        // PORT-NOTE: raw pointers `input_data`/`dest` are not `Send`/`Sync`; the
        // no-threadpool `parallel_for` runs `func` synchronously on the caller
        // thread, so capturing them by copy into the closure is sound and
        // mirrors the C++ `[&]` capture (the multithreaded threadpool variant is
        // out of scope, matching thread_parallel_interface.rs).
        let input_data_addr = input_data as usize;
        let dest_addr = dest as usize;
        thread_parallel_interface::parallel_for(
            0,
            leading_dims as i64,
            grain_size,
            &|begin, end| {
                let input_data = input_data_addr as *const u8;
                let dest = dest_addr as *mut u8;
                for i in begin..end {
                    let mut src: *const u8 = unsafe {
                        input_data.add((i as usize * dim_length + start as usize) * length_per_step)
                    };
                    let mut local_dest: *mut u8 =
                        unsafe { dest.add(i as usize * length as usize * length_per_step) };
                    for _j in 0..length {
                        unsafe {
                            core::ptr::copy_nonoverlapping(src, local_dest, length_per_step);
                            src = src.add(step as usize * length_per_step);
                            local_dest = local_dest.add(length_per_step);
                        }
                    }
                }
            },
        );
    } else {
        // Single-threaded path for small workloads
        let mut dest: *mut u8 = dest;
        for i in 0..leading_dims {
            let mut src: *const u8 =
                unsafe { input_data.add((i * dim_length + start as usize) * length_per_step) };
            for _j in 0..length {
                unsafe {
                    core::ptr::copy_nonoverlapping(src, dest, length_per_step);
                    src = src.add(step as usize * length_per_step);
                    dest = dest.add(length_per_step);
                }
            }
        }
    }
}
