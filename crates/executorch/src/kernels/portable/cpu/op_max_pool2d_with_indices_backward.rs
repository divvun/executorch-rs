//! Literal port of kernels/portable/cpu/op_max_pool2d_with_indices_backward.cpp.

use core::ops::AddAssign;

use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_max_pool2d_with_indices_args, get_max_pool2d_with_indices_out_target_size,
    output_size_is_valid,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_has_expected_size,
    tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: local check macros mirroring the C++ `ET_LOG_AND_RETURN_IF_FALSE`
// and `ET_CHECK_OR_RETURN_FALSE`; the crate-level `et_check_or_return_false!`
// drops caller format args, so this module carries its own (as kernel_ops_util.rs
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

// [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn]
// [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn]
#[allow(clippy::too_many_arguments)]
fn check_max_pool2d_backward_args(
    grad_output: &Tensor,
    input: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    ceil_mode: bool,
    indices: &Tensor,
    grad_input: &Tensor,
) -> bool {
    et_log_and_return_if_false!(tensors_have_same_dtype2(grad_output, input));
    et_log_and_return_if_false!(tensors_have_same_dtype2(grad_input, input));

    et_check_or_return_false!(
        check_max_pool2d_with_indices_args(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            grad_output,
            indices,
        ),
        "Invalid max_pool_2d arguments"
    );

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_max_pool2d_with_indices_out_target_size(
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_sizes.as_mut_ptr(),
            &mut output_ndim,
        );
    }

    et_log_and_return_if_false!(output_size_is_valid(
        ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim),
        2
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

// PORT-NOTE: the C++ `template <typename CTYPE, bool is_3d>` compile-time flag
// becomes a runtime `is_3d: bool` parameter; the ported call site passes `false`
// (max_pool2d), so the `is_3d`-guarded branches never execute — the `size(-3)`
// negative-dim accesses in those branches are unreached, matching the C++ where
// `is_3d == false` selects the constant arm of each `?:`.
// [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn]
// [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn]
fn max_pool_backward_impl<CTYPE>(
    is_3d: bool,
    grad_input: &Tensor,
    grad_output: &Tensor,
    indices: &Tensor,
) where
    CTYPE: Copy + AddAssign,
{
    let grad_output_data: *const CTYPE = grad_output.const_data_ptr::<CTYPE>();
    let indices_data: *const i64 = indices.const_data_ptr::<i64>();
    let grad_input_data: *mut CTYPE = grad_input.mutable_data_ptr::<CTYPE>();

    // treat batch size and channels as one dimension
    //
    // MaxPool2d:
    //   ndim == 3: CHW
    //   ndim == 4: NCHW
    //
    // MaxPool3d:
    //   ndim == 4: CDHW
    //   ndim == 5: NCDHW
    let ndim: i64 = grad_output.dim() as i64;
    let channels: i64;
    if is_3d {
        channels = if ndim == 4 {
            grad_output.size(0) as i64
        } else {
            (grad_output.size(0) * grad_output.size(1)) as i64
        };
    } else {
        channels = if ndim == 3 {
            grad_output.size(0) as i64
        } else {
            (grad_output.size(0) * grad_output.size(1)) as i64
        };
    }
    let input_depth: i64 = if is_3d { grad_input.size(-3) as i64 } else { 1 };

    let input_height: i64 = grad_input.size((ndim - 2) as isize) as i64;
    let input_width: i64 = grad_input.size((ndim - 1) as isize) as i64;
    let output_depth: i64 = if is_3d {
        grad_output.size((ndim - 3) as isize) as i64
    } else {
        1
    };
    let output_height: i64 = grad_output.size((ndim - 2) as isize) as i64;
    let output_width: i64 = grad_output.size((ndim - 1) as isize) as i64;

    let mut c: i64 = 0;
    while c < channels {
        let grad_input_ptr: *mut CTYPE = unsafe {
            grad_input_data.offset((c * input_depth * input_height * input_width) as isize)
        };
        let grad_output_ptr: *const CTYPE = unsafe {
            grad_output_data.offset((c * output_depth * output_height * output_width) as isize)
        };
        let indices_ptr: *const i64 = unsafe {
            indices_data.offset((c * output_depth * output_height * output_width) as isize)
        };

        let mut od: i64 = 0;
        while od < output_depth {
            let mut oh: i64 = 0;
            while oh < output_height {
                let mut ow: i64 = 0;
                while ow < output_width {
                    // retrieve position of max
                    let index: i64 = od * output_height * output_width + oh * output_width + ow;
                    let maxindex: i64 = unsafe { *indices_ptr.offset(index as isize) };
                    if maxindex != -1 {
                        // update gradient
                        unsafe {
                            *grad_input_ptr.offset(maxindex as isize) +=
                                *grad_output_ptr.offset(index as isize);
                        }
                    }
                    ow += 1;
                }
                oh += 1;
            }
            od += 1;
        }
        c += 1;
    }
}

// PORT-NOTE: `Tensor& grad_input` / returned `Tensor&` become `&'a Tensor`
// (interior mutation through `*mut TensorImpl`). `ET_UNUSED` params are kept
// (prefixed `_` where unused after validation is factored in) — they are still
// forwarded to `check_max_pool2d_backward_args` for validation.
// [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn]
// [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_with_indices_backward_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    grad_output: &Tensor,
    input: &Tensor,
    kernel_size: IntArrayRef,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    ceil_mode: bool,
    indices: &Tensor,
    grad_input: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_max_pool2d_backward_args(
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
            grad_input,
        ),
        InvalidArgument,
        grad_input
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(grad_input, input.sizes()) == Error::Ok,
        InvalidArgument,
        grad_input
    );

    let name = "max_pool2d_with_indices_backward.grad_input";

    crate::et_switch_floathbf16_types!(input.scalar_type(), ctx, name, CTYPE, {
        max_pool_backward_impl::<CTYPE>(false, grad_input, grad_output, indices);
    });

    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn iarr(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64Elem for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64Elem for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn op_backward_out<'a, 'b>(
        grad_output: &Tensor,
        input: &Tensor,
        kernel_size: IntArrayRef,
        stride: IntArrayRef,
        padding: IntArrayRef,
        dilation: IntArrayRef,
        ceil_mode: bool,
        indices: &Tensor,
        grad_input: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        max_pool2d_with_indices_backward_out(
            &mut ctx,
            grad_output,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices,
            grad_input,
        )
    }

    fn test_4d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let tf_long = TensorFactory::<i64>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x as f64)).collect() };

        let grad_output = tf.make_default(
            vec![2, 3, 4, 4],
            d(&[
                69, 97, 97, 99, 69, 97, 97, 99, 12, 79, 85, 85, 77, 77, 85, 85, 87, 73, 73, 68, 87,
                94, 94, 68, -30, 94, 94, 8, 71, 74, 77, 77, 4, -8, -12, -46, 87, 90, 90, -45, 87,
                90, 90, 17, 63, 28, 88, 88, 83, 83, 61, 61, 83, 83, 47, 49, 16, 47, 47, 74, 90, 90,
                73, 74, 41, 81, 81, 29, 84, 81, 81, 17, 84, 45, 99, 99, 16, 45, 99, 99, 54, 54, 5,
                29, 54, 68, 68, 29, 90, 90, 68, 90, 99, 99, 65, 90,
            ]),
        );
        let input = tf.make_default(
            vec![2, 3, 5, 5],
            d(&[
                28, -38, -7, -13, 70, 53, 69, 97, 25, 99, -72, -87, 79, 42, -24, -15, 12, -86, 85,
                0, 67, 77, 53, -61, 50, 3, 42, -37, 51, -60, 87, 32, 73, 68, -84, -98, -30, 94, 1,
                -86, -56, -68, 74, -51, 8, 71, -53, 4, 77, -89, 4, -46, -46, -92, -85, -23, -8,
                -12, -46, -88, 66, 87, 90, -45, -78, 63, 28, 28, -30, 17, -16, 5, 11, 88, -47, 72,
                32, -7, 61, -63, -22, 83, -40, -78, 49, -39, -89, 47, -61, 7, 16, -96, -22, 8, 74,
                12, 90, 73, -71, -10, 41, 1, 10, -34, 29, -27, 26, 81, -8, 17, 84, -23, -53, -26,
                -67, -90, 16, 45, 99, 56, -87, -65, -79, 31, 79, 6, 44, -55, -5, -68, -38, 54, -3,
                5, 29, -39, 26, 68, -24, -53, 51, 90, 65, 43, 90, -41, 99, 6, -31, -94,
            ]),
        );
        let indices = tf_long.make_default(
            vec![2, 3, 4, 4],
            vec![
                6, 7, 7, 9, 6, 7, 7, 9, 16, 12, 18, 18, 21, 21, 18, 18, 5, 7, 7, 8, 5, 12, 12, 8,
                11, 12, 12, 19, 20, 17, 23, 23, 0, 6, 7, 8, 11, 12, 12, 13, 11, 12, 12, 19, 15, 16,
                23, 23, 6, 6, 3, 3, 6, 6, 12, 9, 15, 12, 12, 19, 21, 21, 22, 19, 0, 7, 7, 4, 10, 7,
                7, 9, 10, 17, 18, 18, 16, 17, 18, 18, 6, 6, 8, 9, 6, 12, 12, 9, 16, 16, 12, 19, 21,
                21, 17, 19,
            ],
        );
        let grad_input = tf.zeros_default(vec![2, 3, 5, 5]);
        let grad_input_expected = tf.make_default(
            vec![2, 3, 5, 5],
            d(&[
                0, 0, 0, 0, 0, 0, 138, 388, 0, 198, 0, 0, 79, 0, 0, 0, 12, 0, 340, 0, 0, 154, 0, 0,
                0, 0, 0, 0, 0, 0, 174, 0, 146, 136, 0, 0, -30, 376, 0, 0, 0, 0, 74, 0, 8, 71, 0, 0,
                154, 0, 4, 0, 0, 0, 0, 0, -8, -12, -46, 0, 0, 174, 360, -45, 0, 63, 28, 0, 0, 17,
                0, 0, 0, 176, 0, 0, 0, 0, 122, 0, 0, 332, 0, 0, 49, 0, 0, 141, 0, 0, 16, 0, 0, 0,
                148, 0, 180, 73, 0, 0, 41, 0, 0, 0, 29, 0, 0, 324, 0, 17, 168, 0, 0, 0, 0, 0, 16,
                90, 396, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 5, 58, 0, 0, 204, 0, 0, 0,
                180, 65, 0, 180, 0, 198, 0, 0, 0,
            ]),
        );
        op_backward_out(
            &grad_output,
            &input,
            iarr(&[2, 2]),
            iarr(&[1, 1]),
            iarr(&[0, 0]),
            iarr(&[1, 1]),
            false,
            &indices,
            &grad_input,
        );
        assert_tensor_close!(grad_input, grad_input_expected);
    }

    fn test_3d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let tf_long = TensorFactory::<i64>::new();
        let d = |v: &[i64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x as f64)).collect() };

        let grad_output = tf.make_default(
            vec![2, 5, 5],
            d(&[
                89, 89, 89, 20, 20, 89, 89, 86, 49, 80, 89, 89, 99, 99, 99, 84, 84, 86, 86, 86, 51,
                86, 86, 86, 62, 42, 67, 85, 85, 85, 75, 75, 42, 42, 74, 75, 98, 98, 98, 61, 95, 98,
                98, 98, 93, 88, 88, 13, 13, 67,
            ]),
        );
        let input = tf.make_default(
            vec![2, 12, 12],
            d(&[
                73, 15, 30, 89, -55, -62, 25, -50, -47, 12, -73, -89, 53, -63, -44, 86, 53, -84,
                -6, 20, -24, -43, -11, -34, -7, -13, 74, 33, -44, 49, -59, -88, -46, -33, 48, 80,
                38, -58, 0, -48, -46, -87, -66, 14, -68, -77, -50, -15, 86, 89, -37, 7, -16, -6,
                55, 40, -83, -77, -55, 32, -17, -83, 43, 17, 2, -51, 20, -77, -68, -72, -47, -78,
                -49, -52, -7, -25, -77, -8, -3, 99, 71, 19, 21, -47, 44, -90, -75, -87, 79, -42,
                -90, 22, 2, 73, -65, -50, -71, 19, -60, -91, -43, -60, 16, 86, -93, -78, 82, 14,
                20, 19, 33, 84, 60, 41, 2, -4, -52, 74, -40, -60, 88, 51, -59, 49, -81, -93, 43,
                -99, 40, -84, 76, 27, 59, -19, -55, -50, 81, 86, -19, 51, 70, -90, 74, 62, 0, -31,
                -71, 42, 42, 67, 26, 85, -11, -34, -97, 5, -45, -50, 74, -62, -81, -84, 70, 33,
                -27, -54, 94, 74, -30, 16, 39, 0, 0, -80, 85, 42, 13, -82, -30, -95, 34, -60, -51,
                -10, -30, -65, -96, -95, 60, -33, 67, -88, -26, 75, 29, -27, -28, 21, -2, -29, 11,
                -68, -36, -85, -4, 9, -31, -63, 98, -1, 17, 61, -50, 41, -18, -92, -50, -40, 14,
                18, 22, 10, 58, -86, -9, 5, -69, -50, -26, 26, 57, -94, -53, 98, 37, 35, -20, -9,
                -13, -41, 41, 95, 82, -71, -43, -37, -91, -14, -55, 52, -30, 93, -26, 83, 2, -63,
                52, 31, 57, 42, -2, -45, 99, -18, 38, 88, 36, -36, -35, 13, -31, -50, 10, -38, 1,
                67, 3, -87, 42, -31, -77, -7, -94, -99, 24, -21, -98, 15,
            ]),
        );
        let indices = tf_long.make_default(
            vec![2, 5, 5],
            vec![
                3, 3, 3, 19, 19, 49, 49, 15, 29, 35, 49, 49, 79, 79, 79, 111, 111, 103, 103, 103,
                121, 137, 137, 137, 143, 3, 5, 7, 7, 7, 49, 49, 31, 31, 23, 49, 89, 89, 89, 67, 97,
                89, 89, 89, 107, 121, 121, 125, 125, 131,
            ],
        );
        let grad_input = tf.zeros_default(vec![2, 12, 12]);
        let grad_input_expected = tf.make_default(
            vec![2, 12, 12],
            d(&[
                0, 0, 0, 267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 49, 0, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 356, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 258, 0,
                0, 0, 0, 0, 0, 0, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 258, 0, 0, 0, 0, 0, 62, 0, 0, 0, 42, 0, 67, 0, 255, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 84, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                61, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 588, 0, 0, 0, 0,
                0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                176, 0, 0, 0, 26, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]),
        );
        op_backward_out(
            &grad_output,
            &input,
            iarr(&[4, 3]),
            iarr(&[3, 2]),
            iarr(&[2, 1]),
            iarr(&[1, 2]),
            false,
            &indices,
            &grad_input,
        );
        assert_tensor_close!(grad_input, grad_input_expected);
    }

    // also verifies check_max_pool2d_backward_args (valid path: it must pass or the op
    // aborts leaving grad_input all-zeros) and max_pool_backward_impl (the exact scattered
    // gradient values, including accumulated repeated max-indices like 138/388, pin the
    // channel-flattened `grad_input[maxindex] += grad_output[index]` loop).
    // [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn/test]
    // [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn/test]
    // [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn/test]
    #[test]
    fn op_max_pool2d_with_indices_backward_out_test_sanity_test_4d() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_4d_dtype::<f32>();
        test_4d_dtype::<f64>();
        test_4d_dtype::<Half>();
        test_4d_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn/test]
    #[test]
    fn op_max_pool2d_with_indices_backward_out_test_sanity_test_3d() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_3d_dtype::<f32>();
        test_3d_dtype::<f64>();
        test_3d_dtype::<Half>();
        test_3d_dtype::<BFloat16>();
    }
}
