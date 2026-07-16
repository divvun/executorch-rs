//! Literal port of kernels/portable/cpu/util/activation_ops_util.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, dim_is_valid, resize_tensor_same_type, tensor_has_dim,
    tensor_is_default_or_channels_last_dim_order, tensor_is_floating_type,
    tensors_have_same_dtype2, tensors_have_same_rank,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: local `et_log_and_return_if_false!` / `et_check_or_return_false!`
// mirroring the C++ check macros. The crate-level `et_check_or_return_false!`
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

// [spec:et:def:activation-ops-util.torch.executor.check-gelu-args-fn]
// [spec:et:sem:activation-ops-util.torch.executor.check-gelu-args-fn]
// PORT-NOTE: C++ `std::string_view approximate` -> `&str`.
pub fn check_gelu_args(in_: &Tensor, approximate: &str, out: &Tensor) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(in_.scalar_type() != ScalarType::Bool);
    et_check_or_return_false!(
        approximate == "tanh" || approximate == "none",
        "Invalid approximation format: {} for gelu",
        approximate
    );
    true
}

// [spec:et:def:activation-ops-util.torch.executor.check-glu-args-fn]
// [spec:et:sem:activation-ops-util.torch.executor.check-glu-args-fn]
pub fn check_glu_args(in_: &Tensor, dim: i64, out: &Tensor) -> bool {
    et_log_and_return_if_false!(dim_is_valid(dim, in_.dim() as i64));
    et_log_and_return_if_false!(tensor_is_floating_type(in_));

    let non_negative_dim: usize = if dim < 0 {
        (dim + in_.dim() as i64) as usize
    } else {
        dim as usize
    };
    let dim_size: ssize_t = in_.size(non_negative_dim as ssize_t);

    et_check_or_return_false!(
        dim_size % 2 == 0,
        "Halving dimension must be even, but dimension {} is size {}",
        non_negative_dim,
        dim_size
    );

    et_log_and_return_if_false!(tensor_is_floating_type(out));
    et_log_and_return_if_false!(tensors_have_same_rank(in_, out));
    et_check_or_return_false!(
        out.size(non_negative_dim as ssize_t) == dim_size / 2,
        "output tensor must have half the size of the input tensor along the specified dimension; out.size({}) = {}, dim_size = {}",
        non_negative_dim,
        out.size(non_negative_dim as ssize_t),
        dim_size
    );

    for i in 0..in_.dim() {
        if i as usize != non_negative_dim {
            if out.size(i) != in_.size(i) {
                if crate::runtime::platform::log::ET_LOG_ENABLED {
                    let out_shape_str = crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                        crate::runtime::core::span::Span::<SizesType>::from_raw_parts(
                            out.sizes().data() as *mut SizesType,
                            out.sizes().size(),
                        ),
                    );
                    let in_shape_str = crate::runtime::core::exec_aten::util::tensor_shape_to_c_string::tensor_shape_to_c_string(
                        crate::runtime::core::span::Span::<SizesType>::from_raw_parts(
                            in_.sizes().data() as *mut SizesType,
                            in_.sizes().size(),
                        ),
                    );
                    crate::et_log!(
                        Error,
                        "output tensor must have the same size as the input tensor in all dimensions except for the specified dimension. (output shape: {} input shape: {})",
                        c_string_data(&out_shape_str),
                        c_string_data(&in_shape_str)
                    );
                }
                return false;
            }
        }
    }

    true
}

// [spec:et:def:activation-ops-util.torch.executor.check-log-softmax-args-fn]
// [spec:et:sem:activation-ops-util.torch.executor.check-log-softmax-args-fn]
pub fn check_log_softmax_args(in_: &Tensor, dim: i64, half_to_float: bool, out: &Tensor) -> bool {
    et_check_or_return_false!(
        !half_to_float,
        "half to float conversion is not supported on CPU"
    );
    et_log_and_return_if_false!(tensors_have_same_dtype2(in_, out));
    et_log_and_return_if_false!(tensor_has_dim(in_, dim));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(in_));
    et_log_and_return_if_false!(tensor_is_default_or_channels_last_dim_order(out));
    true
}

// [spec:et:def:activation-ops-util.torch.executor.check-softmax-args-fn]
// [spec:et:sem:activation-ops-util.torch.executor.check-softmax-args-fn]
pub fn check_softmax_args(in_: &Tensor, dim: i64, half_to_float: bool, out: &Tensor) -> bool {
    check_log_softmax_args(in_, dim, half_to_float, out)
}

// [spec:et:def:activation-ops-util.torch.executor.resize-glu-out-fn]
// [spec:et:sem:activation-ops-util.torch.executor.resize-glu-out-fn]
#[must_use]
pub fn resize_glu_out(in_: &Tensor, dim: i64, out: &Tensor) -> Error {
    let mut expected_output_size: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];

    let non_negative_dim: usize = if dim < 0 {
        (dim + in_.dim() as i64) as usize
    } else {
        dim as usize
    };
    for i in 0..in_.dim() {
        expected_output_size[i as usize] = if i as usize == non_negative_dim {
            (in_.size(i) / 2) as SizesType
        } else {
            in_.size(i) as SizesType
        };
    }

    let output_size: ArrayRef<SizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize);

    resize_tensor_same_type(out, output_size)
}

// PORT-NOTE: `tensor_shape_to_c_string` returns a `[c_char; N]`; `.data()` in
// C++ yields the C-string used by the `%s` format. `c_string_data` mirrors that
// `.data()` accessor (same helper as tensor_impl.rs).
fn c_string_data<const N: usize>(buf: &[core::ffi::c_char; N]) -> &str {
    let bytes = unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const u8, N) };
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(N);
    core::str::from_utf8(&bytes[..end]).unwrap_or("")
}
