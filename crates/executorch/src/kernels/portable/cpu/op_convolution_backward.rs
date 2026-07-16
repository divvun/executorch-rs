//! Literal port of kernels/portable/cpu/op_convolution_backward.cpp.

use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_convolution_args, get_convolution_out_target_size, output_size_is_valid, val_at,
    val_at_default,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, calculate_linear_index, resize_tensor_same_type,
    tensor_has_expected_size, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: accumulation over FLOATHBF16 ({Half, Float, Double, BFloat16})
// requires `Mul`/`AddAssign` (from the ctypes' operator impls) plus a `0`-fill
// for memset, which `write_bytes(.., 0, ..)` provides directly. No zero literal
// is needed here; the trait bound just gathers the arithmetic ops.
trait ConvScalar: Copy + core::ops::Mul<Output = Self> + core::ops::AddAssign {}
impl ConvScalar for f32 {}
impl ConvScalar for f64 {}
impl ConvScalar for Half {}
impl ConvScalar for BFloat16 {}

// PORT-NOTE: local check macros mirroring the C++ `ET_CHECK_OR_RETURN_FALSE` /
// `ET_LOG_AND_RETURN_IF_FALSE`, matching kernel_ops_util.rs's approach (the
// crate-level `et_check_or_return_false!` drops format args).
macro_rules! et_log_and_return_if_false {
    ($cond:expr) => {{
        if !($cond) {
            crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            return false;
        }
    }};
}
macro_rules! et_check_or_return_false {
    ($cond:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        if !($cond) {
            crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond)
                $(, $($arg)*)?
            );
            return false;
        }
    }};
}

// [spec:et:def:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn]
// [spec:et:sem:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn]
#[allow(clippy::too_many_arguments)]
fn check_convolution_backward_args(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    _bias_sizes_opt: Option<IntArrayRef>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    output_padding: IntArrayRef,
    groups: i64,
    output_mask: ArrayRef<bool>,
    grad_input: &Tensor,
    grad_weight: &Tensor,
    grad_bias: &Tensor,
) -> bool {
    et_check_or_return_false!(
        transposed == false,
        "Transposed Convolution Backward not supported yet"
    );
    et_check_or_return_false!(
        weight.dim() == 4,
        "Only 2D Convolution Backward supported for now; weight.dim() = {}",
        weight.dim()
    );

    et_log_and_return_if_false!(tensors_have_same_dtype2(weight, input));
    et_log_and_return_if_false!(tensors_have_same_dtype2(grad_output, input));

    if *output_mask.at(0) {
        et_log_and_return_if_false!(tensors_have_same_dtype2(grad_input, input));
    }

    if *output_mask.at(1) {
        et_log_and_return_if_false!(tensors_have_same_dtype2(grad_weight, input));
    }

    if *output_mask.at(2) {
        et_log_and_return_if_false!(tensors_have_same_dtype2(grad_bias, input));
    }

    et_check_or_return_false!(
        check_convolution_args(
            input,
            weight,
            &None,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            grad_output,
        ),
        "Invalid convolution arguments"
    );

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_convolution_out_target_size(
            input,
            weight,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_sizes.as_mut_ptr(),
            &mut output_ndim,
        );
    }

    et_log_and_return_if_false!(output_size_is_valid(
        ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim),
        (input.dim() - 2) as usize
    ));

    et_check_or_return_false!(
        grad_output.dim() == input.dim(),
        "grad_output should have same number of dimensions as input; grad_output.dim() = {}, input.dim() = {}",
        grad_output.dim(),
        input.dim()
    );

    et_log_and_return_if_false!(tensor_has_expected_size(
        grad_output,
        ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
    ));

    true
}

// [spec:et:def:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn]
// [spec:et:sem:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn]
#[allow(clippy::too_many_arguments)]
fn conv2d_backward_impl<CTYPE: ConvScalar>(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    groups: i64,
    output_mask: ArrayRef<bool>,
    grad_input: &Tensor,
    grad_weight: &Tensor,
    grad_bias: &Tensor,
) {
    let batch_size = input.size(0);
    let in_channels = input.size(1);
    let out_channels = weight.size(0);
    let in_height = input.size(2);
    let in_width = input.size(3);
    let out_height = grad_output.size(2);
    let out_width = grad_output.size(3);
    let kernel_height = weight.size(2);
    let kernel_width = weight.size(3);

    let stride_h = val_at_default(stride, 0);
    let padding_h = val_at(padding, 0, /*default_value=*/ 0);
    let dilation_h = val_at_default(dilation, 0);
    let stride_w = val_at_default(stride, 1);
    let padding_w = val_at(padding, 1, /*default_value=*/ 0);
    let dilation_w = val_at_default(dilation, 1);

    let in_channels_per_group = in_channels / groups as isize;
    let out_channels_per_group = out_channels / groups as isize;

    let grad_output_data: *const CTYPE = grad_output.const_data_ptr::<CTYPE>();
    let input_data: *const CTYPE = input.const_data_ptr::<CTYPE>();
    let weight_data: *const CTYPE = weight.const_data_ptr::<CTYPE>();

    let mut grad_input_data: *mut CTYPE = core::ptr::null_mut();
    let mut grad_weight_data: *mut CTYPE = core::ptr::null_mut();
    let mut grad_bias_data: *mut CTYPE = core::ptr::null_mut();

    if *output_mask.at(0) {
        grad_input_data = grad_input.mutable_data_ptr::<CTYPE>();
        unsafe {
            core::ptr::write_bytes(grad_input_data as *mut u8, 0, grad_input.nbytes());
        }
    }

    if *output_mask.at(1) {
        grad_weight_data = grad_weight.mutable_data_ptr::<CTYPE>();
        unsafe {
            core::ptr::write_bytes(grad_weight_data as *mut u8, 0, grad_weight.nbytes());
        }
    }

    if *output_mask.at(2) {
        grad_bias_data = grad_bias.mutable_data_ptr::<CTYPE>();
        unsafe {
            core::ptr::write_bytes(grad_bias_data as *mut u8, 0, grad_bias.nbytes());
        }
    }

    let mut out_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut in_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut weight_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    // Compute gradients
    for b in 0..batch_size {
        // Loop over each batch
        in_coord[0] = b as SizesType;
        out_coord[0] = b as SizesType;
        for g in 0..groups {
            // Loop over each group
            for h in 0..out_height {
                // Loop over each output row
                out_coord[2] = h as SizesType;
                for w in 0..out_width {
                    // Loop over each output col
                    out_coord[3] = w as SizesType;

                    // Loop over each output channel in the group
                    for oc in 0..out_channels_per_group {
                        let oc_global = oc + g as isize * out_channels_per_group;
                        weight_coord[0] = oc_global as SizesType;
                        out_coord[1] = oc_global as SizesType;

                        let out_idx = unsafe {
                            calculate_linear_index(
                                out_coord.as_ptr(),
                                grad_output.strides().data(),
                                4,
                            )
                        };

                        // Accumulate the gradient with respect to the bias if
                        // required
                        if *output_mask.at(2) {
                            unsafe {
                                *grad_bias_data.offset(oc_global) += *grad_output_data.add(out_idx);
                            }
                        }

                        // Loop over each input channel in the group
                        for ic in 0..in_channels_per_group {
                            let ic_global = ic + g as isize * in_channels_per_group;
                            in_coord[1] = ic_global as SizesType;
                            weight_coord[1] = ic as SizesType;

                            // Loop over each element
                            for kh in 0..kernel_height {
                                let in_h = h * stride_h as isize - padding_h as isize
                                    + kh * dilation_h as isize;
                                if in_h >= 0 && in_h < in_height {
                                    in_coord[2] = in_h as SizesType;
                                    weight_coord[2] = kh as SizesType;

                                    for kw in 0..kernel_width {
                                        let in_w = w * stride_w as isize - padding_w as isize
                                            + kw * dilation_w as isize;
                                        if in_w >= 0 && in_w < in_width {
                                            in_coord[3] = in_w as SizesType;
                                            weight_coord[3] = kw as SizesType;

                                            let in_idx = unsafe {
                                                calculate_linear_index(
                                                    in_coord.as_ptr(),
                                                    input.strides().data(),
                                                    4,
                                                )
                                            };

                                            let weight_idx = unsafe {
                                                calculate_linear_index(
                                                    weight_coord.as_ptr(),
                                                    weight.strides().data(),
                                                    4,
                                                )
                                            };

                                            // Gradient with respect to the input
                                            // if required
                                            if *output_mask.at(0) {
                                                unsafe {
                                                    *grad_input_data.add(in_idx) +=
                                                        *grad_output_data.add(out_idx)
                                                            * *weight_data.add(weight_idx);
                                                }
                                            }
                                            // Gradient with respect to the weight
                                            // if required
                                            if *output_mask.at(1) {
                                                unsafe {
                                                    *grad_weight_data.add(weight_idx) +=
                                                        *grad_output_data.add(out_idx)
                                                            * *input_data.add(in_idx);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// [spec:et:def:op-convolution-backward.torch.executor.native.convolution-backward-out-fn]
// [spec:et:sem:op-convolution-backward.torch.executor.native.convolution-backward-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn convolution_backward_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias_sizes_opt: Option<IntArrayRef>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    output_padding: IntArrayRef,
    groups: i64,
    output_mask: ArrayRef<bool>,
    grad_input: &'a Tensor<'b>,
    grad_weight: &'a Tensor<'b>,
    grad_bias: &'a Tensor<'b>,
) -> (&'a Tensor<'b>, &'a Tensor<'b>, &'a Tensor<'b>) {
    let ret_val = (grad_input, grad_weight, grad_bias);

    crate::et_kernel_check!(
        ctx,
        check_convolution_backward_args(
            grad_output,
            input,
            weight,
            bias_sizes_opt,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
            grad_input,
            grad_weight,
            grad_bias,
        ),
        InvalidArgument,
        ret_val
    );

    if *output_mask.at(0) {
        crate::et_kernel_check!(
            ctx,
            resize_tensor_same_type(grad_input, input.sizes()) == Error::Ok,
            InvalidArgument,
            ret_val
        );
    }

    if *output_mask.at(1) {
        crate::et_kernel_check!(
            ctx,
            resize_tensor_same_type(grad_weight, weight.sizes()) == Error::Ok,
            InvalidArgument,
            ret_val
        );
    }

    if bias_sizes_opt.is_some() && *output_mask.at(2) {
        crate::et_kernel_check!(
            ctx,
            resize_tensor_same_type_i64(grad_bias, bias_sizes_opt.unwrap()) == Error::Ok,
            InvalidArgument,
            ret_val
        );
    }

    let name = "convolution_backward.out";

    crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, name, CTYPE, {
        conv2d_backward_impl::<CTYPE>(
            grad_output,
            input,
            weight,
            stride,
            padding,
            dilation,
            groups,
            output_mask,
            grad_input,
            grad_weight,
            grad_bias,
        );
    });

    ret_val
}

// PORT-NOTE: `resize_tensor(grad_bias, bias_sizes_opt.value())` resizes from an
// `IntArrayRef` (i64) rather than `SizesType`; the generic `resize_tensor<i64>`
// path handles the lossy cast (SizesType: TryFromLossy<i64>).
#[must_use]
fn resize_tensor_same_type_i64(t: &Tensor, new_sizes: IntArrayRef) -> Error {
    crate::runtime::core::exec_aten::util::tensor_util::resize_tensor(t, new_sizes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close_with_tol;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn ir(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    impl FromI32 for f32 {
        fn from_i32(v: i32) -> Self {
            v as f32
        }
    }
    impl FromI32 for f64 {
        fn from_i32(v: i32) -> Self {
            v as f64
        }
    }
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn v<T: FromI32>(vals: &[i32]) -> Vec<T> {
        vals.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        let grad_output_data = [
            10, 12, 87, 13, 34, 87, 55, 22, 48, 33, 29, 38, 60, 49, 88, 30, 99, 19, 42, 37, 61, 31,
            33, 58, 38, 23, 2, 33, 3, 21, 32, 2, 30, 72, 10, 67, 92, 19, 11, 16, 65, 37, 60, 74, 4,
            19, 45, 37,
        ];
        let input_data = [
            9, 89, 45, 39, 25, 2, 97, 55, 80, 24, 18, 33, 28, 89, 19, 16, 19, 33, 69, 61, 34, 84,
            58, 30, 33, 18, 75, 30, 6, 33, 42, 10, 80, 41, 66, 64, 47, 51, 67, 62, 58, 10, 97, 71,
            24, 44, 84, 34, 33, 54, 8, 73, 90, 15, 21, 92, 55, 22, 56, 12, 10, 63, 32, 76, 65, 38,
            95, 92, 22, 15, 37, 12, 67, 14, 60, 44, 73, 74, 23, 4, 56, 64, 88, 90, 82, 32, 91, 3,
            6, 87, 55, 95, 7, 14, 24, 69, 52, 44, 14, 37, 75, 52, 37, 40, 25, 54, 4, 15, 97, 51,
            46, 28, 65, 95, 50, 82, 23, 39, 50, 55, 97, 52, 91, 16, 19, 49, 61, 50, 42, 47, 87, 99,
            9, 60, 22, 71, 47, 17, 0, 80, 28, 88, 93, 43, 65, 25, 88, 67, 21, 89, 24, 81, 3, 71,
            20, 34, 17, 17, 94, 10, 82, 25, 10, 11, 7, 28, 77, 39, 74, 79, 17, 40, 67, 54, 49, 54,
            21, 89, 17, 7, 52, 64, 68, 80, 7, 72, 44, 35, 92, 47, 4, 13, 10, 43, 64, 66, 83, 49,
            81, 78, 58, 22, 86, 48, 35, 64, 98, 79, 8, 52, 56, 23, 38, 74, 16, 63, 51, 70, 44, 28,
            43, 13, 51, 85, 42, 29, 64, 26, 54, 91, 9, 96, 41, 56, 7, 52, 27, 22, 69, 13, 8, 20,
            22, 49, 66, 98, 77, 42, 54, 38, 70, 83, 13, 8, 21, 56, 78, 37, 28, 69, 42, 30, 91, 5,
            28, 15, 20, 14, 16, 39, 95, 66, 4, 72, 52, 35, 54, 93, 87, 77, 3, 49, 82, 70, 84, 3,
            73, 99, 32, 95, 58, 65, 32, 75, 34, 22, 12, 84, 63, 72, 85, 66, 63, 27, 3, 73, 45, 37,
            61, 52, 41, 16, 37, 14, 80, 17, 48, 8, 87, 98, 69, 63, 92, 68, 42, 63, 5, 22, 66, 91,
            74, 11, 17, 45, 45, 33, 40, 85, 26, 75, 73, 81, 54, 27, 80, 1, 44, 66, 10, 21, 15, 10,
            76, 96, 0, 43, 39, 3, 57, 79, 45, 64, 58, 92, 44, 42, 7, 28, 94, 4, 8, 22, 22, 31, 75,
            44, 3, 70, 83, 72, 87, 12, 20, 55, 84, 31, 50, 34, 25, 49, 29, 71, 57, 97, 25, 82, 84,
            42, 86, 41, 54, 92, 34, 30, 52, 34, 84, 25, 54, 37, 38, 26, 76, 82, 34, 14, 85, 28, 93,
            9,
        ];
        let weight_data = [
            2, 54, 9, 37, 0, 47, 70, 9, 84, 69, 56, 79, 25, 35, 54, 13, 65, 46, 38, 28, 74, 27, 66,
            61, 20, 60, 62, 58, 15, 44, 75, 55, 7, 52, 13, 36, 39, 64, 62, 45, 100, 6, 79, 63, 63,
            52, 37, 60, 78, 12, 69, 2, 74, 56, 93, 39, 62, 22, 55, 67, 68, 74, 12, 69, 15, 73, 28,
            70, 86, 20, 90, 49, 52, 26, 58, 2, 82, 17, 70, 55, 54, 83, 70, 11, 27, 9, 5, 42, 34,
            62, 29, 94, 69, 81, 54, 4,
        ];
        let expected_grad_input_data = [
            1134, 7578, 686, 2682, 0, 4148, 7136, 2406, 8698, 0, 3759, 6003, 2163, 2395, 0, 2929,
            5830, 3469, 6955, 0, 720, 6201, 495, 2063, 0, 5260, 5989, 3060, 7079, 0, 9690, 3423,
            3385, 1932, 0, 7644, 8499, 1323, 2613, 0, 4334, 6624, 8532, 9719, 0, 5496, 8601, 1157,
            2215, 0, 4676, 7600, 6524, 10069, 0, 4047, 6117, 1612, 2567, 0, 5931, 5651, 5669, 6623,
            0, 7674, 3291, 2748, 1654, 0, 10455, 4290, 4145, 796, 0, 9835, 5483, 11649, 5952, 0,
            7098, 5460, 3101, 2443, 0, 7788, 5909, 8582, 6298, 0, 9462, 4845, 3041, 2067, 0, 7038,
            6336, 10438, 6377, 0, 7518, 8187, 2079, 2773, 0, 10036, 2642, 3952, 1166, 0, 16014,
            2250, 10025, 1908, 0, 9610, 298, 3868, 122, 0, 16629, 4338, 11335, 3527, 0, 11514,
            5965, 4762, 2207, 0, 18552, 10755, 13309, 5996, 0, 12454, 6787, 4960, 2875, 0, 8750,
            6999, 3534, 3233, 0, 14160, 9399, 9595, 8922, 0, 9110, 6567, 3820, 2351, 0, 12969,
            11814, 9436, 5870, 0, 7631, 7061, 2877, 2499, 0, 8553, 13527, 3631, 6863, 0, 1361,
            8634, 515, 3372, 0, 3394, 10206, 1504, 4112, 0, 5505, 17421, 4702, 11891, 0, 4233,
            11894, 1739, 5014, 0, 11787, 14634, 8981, 10759, 0, 11777, 6701, 4719, 3111, 0, 18459,
            7761, 12044, 7627, 0, 11214, 4556, 4374, 1594, 0, 604, 1908, 1506, 6102, 0, 2532, 4024,
            1713, 6121, 0, 1878, 1814, 4761, 5397, 0, 1127, 3885, 4373, 5832, 0, 450, 1414, 1080,
            4719, 0, 5210, 2683, 2765, 4252, 0, 2390, 1668, 7710, 4257, 0, 378, 1698, 3276, 6021,
            0, 2866, 4881, 3547, 6822, 0, 502, 1238, 2784, 5199, 0, 2496, 3975, 2700, 5004, 0,
            1220, 1990, 3633, 5763, 0, 4501, 2679, 4504, 5412, 0, 1968, 1376, 6246, 3669, 0, 3130,
            272, 9345, 1950, 0, 5167, 3278, 9097, 2138, 0, 2446, 1946, 6942, 5460, 0, 5732, 3404,
            7919, 5534, 0, 2038, 1614, 6978, 4635, 0, 4544, 4839, 7367, 5574, 0, 1242, 1922, 4842,
            6333, 0, 1066, 236, 2236, 686, 0, 17238, 2254, 10413, 1592, 0, 991, 30, 2206, 70, 0,
            18823, 6392, 12173, 2470, 0, 1142, 684, 2742, 1219, 0, 21256, 11293, 12719, 7512, 0,
            1303, 649, 2818, 1669, 0, 898, 574, 2018, 1929, 0, 15720, 11989, 10517, 5972, 0, 885,
            781, 2210, 1281, 0, 14601, 12198, 7915, 4958, 0, 856, 850, 1601, 1355, 0, 7039, 14083,
            4113, 7490, 0, 152, 927, 287, 1902, 0, 301, 1051, 886, 2346, 0, 6821, 19615, 4491,
            13281, 0, 424, 1146, 999, 2906, 0, 15177, 15480, 8849, 12442, 0, 1222, 544, 2687, 1859,
            0, 20215, 9693, 11441, 4964, 0, 1206, 555, 2466, 860, 0,
        ];
        let expected_grad_weight_data = [
            9246, 22073, 12431, 19714, 11179, 19032, 8458, 6495, 18707, 13830, 20445, 17089, 17124,
            18710, 11827, 17236, 16824, 9008, 14086, 18834, 17419, 16759, 13152, 9339, 13801,
            20888, 13976, 27277, 13010, 23949, 9838, 11220, 17658, 15019, 25337, 17583, 13270,
            21754, 16908, 20563, 20732, 13413, 20868, 27521, 19537, 21170, 15888, 10034, 19195,
            16370, 40243, 25890, 40472, 30460, 21228, 21625, 13289, 24435, 19876, 29816, 24188,
            23619, 13752, 16251, 18741, 19368, 24517, 34261, 27054, 31257, 21238, 18909, 15776,
            16881, 34604, 22534, 28101, 23834, 18479, 16469, 12852, 16551, 14204, 29983, 20167,
            24150, 14281, 17501, 15897, 16019, 21661, 32765, 23874, 26527, 20463, 18661,
        ];
        let expected_grad_bias_data = [363, 438, 585, 501];

        let grad_output = tf.make_default(vec![2, 4, 3, 2], v(&grad_output_data));
        let input = tf.make_default(vec![2, 6, 7, 5], v(&input_data));
        let weight = tf.make_default(vec![4, 3, 4, 2], v(&weight_data));
        let bias_sizes = [4i64];
        let stride = [1i64, 2];
        let padding = [1i64, 0];
        let dilation = [2i64, 1];
        let transposed = false;
        let output_padding = [0i64, 0];
        let groups = 2i64;
        let output_mask_a = [true, true, true];
        let grad_input = tf.zeros_default(vec![2, 6, 7, 5]);
        let grad_weight = tf.zeros_default(vec![4, 3, 4, 2]);
        let grad_bias = tf.zeros_default(vec![4]);

        let mut ctx = context();
        convolution_backward_out(
            &mut ctx,
            &grad_output,
            &input,
            &weight,
            Some(ir(&bias_sizes)),
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            transposed,
            ir(&output_padding),
            groups,
            ArrayRef::from_raw_parts(output_mask_a.as_ptr(), output_mask_a.len()),
            &grad_input,
            &grad_weight,
            &grad_bias,
        );

        let expected_grad_input = tf.make_default(vec![2, 6, 7, 5], v(&expected_grad_input_data));
        let expected_grad_weight = tf.make_default(vec![4, 3, 4, 2], v(&expected_grad_weight_data));
        let expected_grad_bias = tf.make_default(vec![4], v(&expected_grad_bias_data));

        if T::VALUE == ScalarType::Half || T::VALUE == ScalarType::BFloat16 {
            assert_tensor_close_with_tol!(grad_input, expected_grad_input, 1e-2, 1e-8);
            assert_tensor_close_with_tol!(grad_weight, expected_grad_weight, 2e-2, 1e-8);
            assert_tensor_close_with_tol!(grad_bias, expected_grad_bias, 1e-2, 1e-8);
        } else {
            crate::assert_tensor_close!(grad_input, expected_grad_input);
            crate::assert_tensor_close!(grad_weight, expected_grad_weight);
            crate::assert_tensor_close!(grad_bias, expected_grad_bias);
        }
    }

    // Computing grad_input/grad_weight/grad_bias pins conv2d_backward_impl and
    // flows through check_convolution_backward_args on the valid (accepting) path.
    // [spec:et:sem:op-convolution-backward.torch.executor.native.convolution-backward-out-fn/test]
    // [spec:et:sem:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn/test]
    // [spec:et:sem:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn/test]
    #[test]
    fn op_convolution_backward_out_test_smoke_test() {
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }
}
