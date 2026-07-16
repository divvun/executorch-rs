//! Literal port of kernels/portable/cpu/op_convolution.cpp.

use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::dtype_util::internal::{
    ComputeCast, LoadToComputeFn, get_load_to_compute_fn,
};
use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_convolution_args, get_convolution_out_target_size, get_unsqueezed_dim_order,
    get_unsqueezed_sizes, output_size_is_valid, val_at, val_at_default,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::dim_order_to_stride_nocheck;
use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, calculate_linear_index, resize_tensor_same_type,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{DimOrderType, SizesType, StridesType};
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

type SizesArrayRef = ArrayRef<SizesType>;
type StridesArrayRef = ArrayRef<StridesType>;

// PORT-NOTE: `CTYPE accum = 0.0f;` and `accum += in_val * w_val;` require a zero
// value and `Mul`/`AddAssign` over REALHBF16 (integers + Half/Float/Double/
// BFloat16). Modeled as a `ConvScalar` trait supplying the `0.0f`-cast zero;
// `Mul`/`AddAssign` come from the ctypes' own operator impls.
trait ConvScalar: Copy + core::ops::Mul<Output = Self> + core::ops::AddAssign {
    fn zero() -> Self;
}
macro_rules! impl_conv_scalar {
    ($t:ty, $z:expr) => {
        impl ConvScalar for $t {
            fn zero() -> Self {
                $z
            }
        }
    };
}
impl_conv_scalar!(u8, 0);
impl_conv_scalar!(i8, 0);
impl_conv_scalar!(i16, 0);
impl_conv_scalar!(i32, 0);
impl_conv_scalar!(i64, 0);
impl_conv_scalar!(f32, 0.0);
impl_conv_scalar!(f64, 0.0);
impl ConvScalar for Half {
    fn zero() -> Self {
        Half::from_f32_const(0.0)
    }
}
impl ConvScalar for BFloat16 {
    fn zero() -> Self {
        BFloat16::from_f32_const(0.0)
    }
}

// [spec:et:def:op-convolution.torch.executor.native.conv2d-impl-fn]
// [spec:et:sem:op-convolution.torch.executor.native.conv2d-impl-fn]
#[allow(clippy::too_many_arguments)]
fn conv2d_impl<CTYPE: ConvScalar>(
    in_ptr: *const CTYPE,
    in_sizes: SizesArrayRef,
    in_strides: StridesArrayRef,
    w_ptr: *const CTYPE,
    w_sizes: SizesArrayRef,
    w_strides: StridesArrayRef,
    bias: &Option<Tensor>,
    bias_ptr: *const u8,
    load_bias: Option<LoadToComputeFn<CTYPE>>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    groups: i64,
    out_ptr: *mut CTYPE,
    out_sizes: SizesArrayRef,
    out_strides: StridesArrayRef,
    batch: usize,
    group: usize,
    out_c: usize,
    transposed: bool,
) {
    let in_C = *in_sizes.at(1) as usize;
    let out_C = *out_sizes.at(1) as usize;

    let out_H = *out_sizes.at(2) as usize;
    let in_H = *in_sizes.at(2) as usize;
    let w_H = *w_sizes.at(2) as usize;

    let out_W = *out_sizes.at(3) as usize;
    let in_W = *in_sizes.at(3) as usize;
    let w_W = *w_sizes.at(3) as usize;

    let in_C_per_group = in_C / groups as usize;
    let in_c_start = group * in_C_per_group;

    let out_C_per_group = out_C / groups as usize;
    let out_c_start = group * out_C_per_group;

    let mut in_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    in_coord[0] = batch as SizesType;
    let mut out_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    out_coord[0] = batch as SizesType;
    out_coord[1] = out_c as SizesType;
    let mut w_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    let stride_y = val_at_default(stride, 0);
    let padding_y = val_at(padding, 0, /*default_value=*/ 0);
    let dilation_y = val_at_default(dilation, 0);
    let stride_x = val_at_default(stride, 1);
    let padding_x = val_at(padding, 1, /*default_value=*/ 0);
    let dilation_x = val_at_default(dilation, 1);

    if !transposed {
        w_coord[0] = out_c as SizesType;
        // Compute 2D output region
        for out_y in 0..out_H {
            out_coord[2] = out_y as SizesType;
            for out_x in 0..out_W {
                out_coord[3] = out_x as SizesType;

                let mut accum = CTYPE::zero();
                for in_c in in_c_start..(in_c_start + in_C_per_group) {
                    in_coord[1] = in_c as SizesType;
                    w_coord[1] = (in_c - in_c_start) as SizesType;

                    for w_y in 0..w_H {
                        w_coord[2] = w_y as SizesType;

                        let in_y: isize = stride_y as isize * out_y as isize
                            + dilation_y as isize * w_y as isize
                            - padding_y as isize;
                        in_coord[2] = in_y as SizesType;
                        // Only proceed if input y coordinate is within bounds
                        if in_y >= 0 && in_y < in_H as isize {
                            for w_x in 0..w_W {
                                w_coord[3] = w_x as SizesType;

                                let in_x: isize = stride_x as isize * out_x as isize
                                    + dilation_x as isize * w_x as isize
                                    - padding_x as isize;
                                in_coord[3] = in_x as SizesType;

                                // Only proceed if input x coordinate is within
                                // bounds
                                if in_x >= 0 && in_x < in_W as isize {
                                    let in_idx = unsafe {
                                        calculate_linear_index(
                                            in_coord.as_ptr(),
                                            in_strides.data(),
                                            4,
                                        )
                                    };
                                    let in_val = unsafe { *in_ptr.add(in_idx) };

                                    let w_idx = unsafe {
                                        calculate_linear_index(
                                            w_coord.as_ptr(),
                                            w_strides.data(),
                                            4,
                                        )
                                    };
                                    let w_val = unsafe { *w_ptr.add(w_idx) };

                                    accum += in_val * w_val;
                                }
                            }
                        }
                    }
                }

                if !bias_ptr.is_null() {
                    let bias_es = bias.as_ref().unwrap().element_size() as usize;
                    accum += (load_bias.unwrap())(unsafe {
                        bias_ptr.add(out_c * bias_es) as *const core::ffi::c_void
                    });
                }
                let out_idx =
                    unsafe { calculate_linear_index(out_coord.as_ptr(), out_strides.data(), 4) };
                unsafe {
                    *out_ptr.add(out_idx) = accum;
                }
            }
        }
    } else {
        // transposed convolution
        w_coord[1] = (out_c - out_c_start) as SizesType;

        for in_y in 0..in_H {
            in_coord[2] = in_y as SizesType;

            for in_x in 0..in_W {
                in_coord[3] = in_x as SizesType;

                for in_c in in_c_start..(in_c_start + in_C_per_group) {
                    in_coord[1] = in_c as SizesType;

                    let in_idx =
                        unsafe { calculate_linear_index(in_coord.as_ptr(), in_strides.data(), 4) };
                    let in_val = unsafe { *in_ptr.add(in_idx) };

                    w_coord[0] = in_c as SizesType;
                    for w_y in 0..w_H {
                        w_coord[2] = w_y as SizesType;
                        let out_y: isize = stride_y as isize * in_y as isize
                            + dilation_y as isize * w_y as isize
                            - padding_y as isize;
                        out_coord[2] = out_y as SizesType;

                        // Only proceed if output y coordinate is within bounds
                        if out_y >= 0 && out_y < out_H as isize {
                            for w_x in 0..w_W {
                                w_coord[3] = w_x as SizesType;
                                let out_x: isize = stride_x as isize * in_x as isize
                                    + dilation_x as isize * w_x as isize
                                    - padding_x as isize;
                                out_coord[3] = out_x as SizesType;

                                // Only proceed if output x coordinate is within
                                // bounds
                                if out_x >= 0 && out_x < out_W as isize {
                                    let w_idx = unsafe {
                                        calculate_linear_index(
                                            w_coord.as_ptr(),
                                            w_strides.data(),
                                            4,
                                        )
                                    };
                                    let w_val = unsafe { *w_ptr.add(w_idx) };

                                    let out_idx = unsafe {
                                        calculate_linear_index(
                                            out_coord.as_ptr(),
                                            out_strides.data(),
                                            4,
                                        )
                                    };

                                    unsafe {
                                        *out_ptr.add(out_idx) += in_val * w_val;
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

// [spec:et:def:op-convolution.torch.executor.native.conv3d-impl-fn]
// [spec:et:sem:op-convolution.torch.executor.native.conv3d-impl-fn]
#[allow(clippy::too_many_arguments)]
fn conv3d_impl<CTYPE: ConvScalar>(
    in_ptr: *const CTYPE,
    in_sizes: SizesArrayRef,
    in_strides: StridesArrayRef,
    w_ptr: *const CTYPE,
    w_sizes: SizesArrayRef,
    w_strides: StridesArrayRef,
    bias: &Option<Tensor>,
    bias_ptr: *const u8,
    load_bias: Option<LoadToComputeFn<CTYPE>>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    groups: i64,
    out_ptr: *mut CTYPE,
    out_sizes: SizesArrayRef,
    out_strides: StridesArrayRef,
    batch: usize,
    group: usize,
    out_c: usize,
) {
    let in_C = *in_sizes.at(1) as usize;
    let out_D = *out_sizes.at(2) as usize;
    let in_D = *in_sizes.at(2) as usize;
    let w_D = *w_sizes.at(2) as usize;

    let out_H = *out_sizes.at(3) as usize;
    let in_H = *in_sizes.at(3) as usize;
    let w_H = *w_sizes.at(3) as usize;

    let out_W = *out_sizes.at(4) as usize;
    let in_W = *in_sizes.at(4) as usize;
    let w_W = *w_sizes.at(4) as usize;

    let in_C_per_group = in_C / groups as usize;
    let in_c_start = group * in_C_per_group;

    let mut in_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    in_coord[0] = batch as SizesType;
    let mut out_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    out_coord[0] = batch as SizesType;
    out_coord[1] = out_c as SizesType;
    let mut w_coord: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];

    let stride_z = val_at_default(stride, 0);
    let padding_z = val_at(padding, 0, /*default_value=*/ 0);
    let dilation_z = val_at_default(dilation, 0);
    let stride_y = val_at_default(stride, 1);
    let padding_y = val_at(padding, 1, /*default_value=*/ 0);
    let dilation_y = val_at_default(dilation, 1);
    let stride_x = val_at_default(stride, 2);
    let padding_x = val_at(padding, 2, /*default_value=*/ 0);
    let dilation_x = val_at_default(dilation, 2);

    w_coord[0] = out_c as SizesType;
    for out_z in 0..out_D {
        out_coord[2] = out_z as SizesType;
        for out_y in 0..out_H {
            out_coord[3] = out_y as SizesType;
            for out_x in 0..out_W {
                out_coord[4] = out_x as SizesType;

                let mut accum = CTYPE::zero();
                for in_c in in_c_start..(in_c_start + in_C_per_group) {
                    in_coord[1] = in_c as SizesType;
                    w_coord[1] = (in_c - in_c_start) as SizesType;

                    for w_z in 0..w_D {
                        w_coord[2] = w_z as SizesType;
                        let in_z: isize = stride_z as isize * out_z as isize
                            + dilation_z as isize * w_z as isize
                            - padding_z as isize;
                        in_coord[2] = in_z as SizesType;
                        if in_z < 0 || in_z >= in_D as isize {
                            continue;
                        }

                        for w_y in 0..w_H {
                            w_coord[3] = w_y as SizesType;
                            let in_y: isize = stride_y as isize * out_y as isize
                                + dilation_y as isize * w_y as isize
                                - padding_y as isize;
                            in_coord[3] = in_y as SizesType;
                            if in_y < 0 || in_y >= in_H as isize {
                                continue;
                            }

                            for w_x in 0..w_W {
                                w_coord[4] = w_x as SizesType;
                                let in_x: isize = stride_x as isize * out_x as isize
                                    + dilation_x as isize * w_x as isize
                                    - padding_x as isize;
                                in_coord[4] = in_x as SizesType;
                                if in_x >= 0 && in_x < in_W as isize {
                                    let in_idx = unsafe {
                                        calculate_linear_index(
                                            in_coord.as_ptr(),
                                            in_strides.data(),
                                            5,
                                        )
                                    };
                                    let w_idx = unsafe {
                                        calculate_linear_index(
                                            w_coord.as_ptr(),
                                            w_strides.data(),
                                            5,
                                        )
                                    };
                                    accum += unsafe { *in_ptr.add(in_idx) * *w_ptr.add(w_idx) };
                                }
                            }
                        }
                    }
                }

                if !bias_ptr.is_null() {
                    let bias_es = bias.as_ref().unwrap().element_size() as usize;
                    accum += (load_bias.unwrap())(unsafe {
                        bias_ptr.add(out_c * bias_es) as *const core::ffi::c_void
                    });
                }
                let out_idx =
                    unsafe { calculate_linear_index(out_coord.as_ptr(), out_strides.data(), 5) };
                unsafe {
                    *out_ptr.add(out_idx) = accum;
                }
            }
        }
    }
}

// [spec:et:def:op-convolution.torch.executor.native.convolution-wrapper-fn]
// [spec:et:sem:op-convolution.torch.executor.native.convolution-wrapper-fn]
#[allow(clippy::too_many_arguments)]
fn convolution_wrapper<CTYPE: ConvScalar>(
    in_: &Tensor,
    weight: &Tensor,
    bias: &Option<Tensor>,
    load_bias: Option<LoadToComputeFn<CTYPE>>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    groups: i64,
    out: &Tensor,
) {
    let mut in_sizes: SizesArrayRef = in_.sizes();
    let mut weight_sizes: SizesArrayRef = weight.sizes();
    let mut out_sizes: SizesArrayRef = out.sizes();

    let mut in_dim_order: ArrayRef<DimOrderType> = in_.dim_order();
    let mut weight_dim_order: ArrayRef<DimOrderType> = weight.dim_order();
    let mut out_dim_order: ArrayRef<DimOrderType> = out.dim_order();

    let mut stride_: IntArrayRef = stride;
    let mut padding_: IntArrayRef = padding;
    let mut dilation_: IntArrayRef = dilation;

    // Define arrays for modified sizes, etc. which will potentially be used
    let mut in_sizes_arr: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut in_dim_order_arr: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut in_ndim: usize = 0;
    let mut weight_sizes_arr: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut weight_dim_order_arr: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut weight_ndim: usize = 0;
    let mut out_sizes_arr: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_dim_order_arr: [DimOrderType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut out_ndim: usize = 0;

    let mut stride_arr: [i64; 2] = [0; 2];
    let mut padding_arr: [i64; 2] = [0; 2];
    let mut dilation_arr: [i64; 2] = [0; 2];

    // If in has a dim of 3, then a 1D convolution will be performed. See C++.
    if in_.dim() == 3 {
        unsafe {
            get_unsqueezed_sizes(in_, 2, in_sizes_arr.as_mut_ptr(), &mut in_ndim);
        }
        in_sizes = ArrayRef::from_raw_parts(in_sizes_arr.as_ptr(), in_ndim);
        unsafe {
            get_unsqueezed_dim_order(in_, 2, in_dim_order_arr.as_mut_ptr());
        }
        in_dim_order = ArrayRef::from_raw_parts(in_dim_order_arr.as_ptr(), in_ndim);

        unsafe {
            get_unsqueezed_sizes(weight, 2, weight_sizes_arr.as_mut_ptr(), &mut weight_ndim);
        }
        weight_sizes = ArrayRef::from_raw_parts(weight_sizes_arr.as_ptr(), weight_ndim);
        unsafe {
            get_unsqueezed_dim_order(weight, 2, weight_dim_order_arr.as_mut_ptr());
        }
        weight_dim_order = ArrayRef::from_raw_parts(weight_dim_order_arr.as_ptr(), weight_ndim);

        unsafe {
            get_unsqueezed_sizes(out, 2, out_sizes_arr.as_mut_ptr(), &mut out_ndim);
        }
        out_sizes = ArrayRef::from_raw_parts(out_sizes_arr.as_ptr(), out_ndim);
        unsafe {
            get_unsqueezed_dim_order(out, 2, out_dim_order_arr.as_mut_ptr());
        }
        out_dim_order = ArrayRef::from_raw_parts(out_dim_order_arr.as_ptr(), out_ndim);

        stride_arr[0] = 1;
        stride_arr[1] = *stride.at(0);
        stride_ = ArrayRef::from_raw_parts(stride_arr.as_ptr(), 2);

        padding_arr[0] = 0;
        padding_arr[1] = *padding.at(0);
        padding_ = ArrayRef::from_raw_parts(padding_arr.as_ptr(), 2);

        dilation_arr[0] = 1;
        if dilation.size() > 0 {
            dilation_arr[1] = *dilation.at(0);
        } else {
            dilation_arr[1] = 1;
        }
        dilation_ = ArrayRef::from_raw_parts(dilation_arr.as_ptr(), 2);
    }

    let mut in_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        dim_order_to_stride_nocheck(
            in_sizes.data(),
            in_dim_order.data(),
            in_sizes.size(),
            in_strides.as_mut_ptr(),
        );
    }

    let mut weight_strides: [StridesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        dim_order_to_stride_nocheck(
            weight_sizes.data(),
            weight_dim_order.data(),
            weight_sizes.size(),
            weight_strides.as_mut_ptr(),
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
    let w_ptr: *const CTYPE = weight.const_data_ptr::<CTYPE>();
    let bias_ptr: *const u8 = if bias.is_some() {
        bias.as_ref().unwrap().const_data_ptr::<u8>()
    } else {
        core::ptr::null()
    };

    let out_N = out.size(0) as usize;
    let out_C = out.size(1) as usize;
    let out_C_per_group = out_C / groups as usize;
    let is_conv3d = in_sizes.size() == 5;

    if transposed {
        // For transposed convolution, we need to initialized the output before
        // we can accumulate into it.
        if bias_ptr.is_null() {
            // If bias is not present, we need to initialize the output to 0
            unsafe {
                core::ptr::write_bytes(out_ptr as *mut u8, 0, out.nbytes());
            }
        } else {
            // If bias is present, we initialize the output to the bias value
            let bias_es = bias.as_ref().unwrap().element_size() as usize;
            for out_ix in 0..(out.numel() as usize) {
                unsafe {
                    *out_ptr.add(out_ix) = (load_bias.unwrap())(
                        bias_ptr.add(((out_ix / out_strides[1] as usize) % out_C) * bias_es)
                            as *const core::ffi::c_void,
                    );
                }
            }
        }
    }

    for batch in 0..out_N {
        for group in 0..(groups as usize) {
            // Align channel offset based on the group
            let out_c_start = group * out_C_per_group;
            // Populate all the out channels in the group
            for out_c in out_c_start..(out_c_start + out_C_per_group) {
                if is_conv3d {
                    conv3d_impl::<CTYPE>(
                        in_ptr,
                        in_sizes,
                        ArrayRef::from_raw_parts(in_strides.as_ptr(), 5),
                        w_ptr,
                        weight_sizes,
                        ArrayRef::from_raw_parts(weight_strides.as_ptr(), 5),
                        bias,
                        bias_ptr,
                        load_bias,
                        stride_,
                        padding_,
                        dilation_,
                        groups,
                        out_ptr,
                        out_sizes,
                        ArrayRef::from_raw_parts(out_strides.as_ptr(), 5),
                        batch,
                        group,
                        out_c,
                    );
                } else {
                    conv2d_impl::<CTYPE>(
                        in_ptr,
                        in_sizes,
                        ArrayRef::from_raw_parts(in_strides.as_ptr(), 4),
                        w_ptr,
                        weight_sizes,
                        ArrayRef::from_raw_parts(weight_strides.as_ptr(), 4),
                        bias,
                        bias_ptr,
                        load_bias,
                        stride_,
                        padding_,
                        dilation_,
                        groups,
                        out_ptr,
                        out_sizes,
                        ArrayRef::from_raw_parts(out_strides.as_ptr(), 4),
                        batch,
                        group,
                        out_c,
                        transposed,
                    );
                }
            }
        }
    }
}

// [spec:et:def:op-convolution.torch.executor.native.convolution-out-fn]
// [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn]
#[executorch_macros::et_kernel("aten::convolution.out")]
#[allow(clippy::too_many_arguments)]
pub fn convolution_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    weight: &Tensor,
    bias: &Option<Tensor>,
    stride: IntArrayRef,
    padding: IntArrayRef,
    dilation: IntArrayRef,
    transposed: bool,
    output_padding: IntArrayRef,
    groups: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_convolution_args(
            in_,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            out,
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_convolution_out_target_size(
            in_,
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

    crate::et_kernel_check!(
        ctx,
        output_size_is_valid(
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim),
            (in_.dim() - 2) as usize
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if out.numel() == 0 {
        return out;
    }

    let name = "convolution.out";

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        let load_bias: Option<LoadToComputeFn<CTYPE>> = if bias.is_some() {
            Some(get_load_to_compute_fn::<CTYPE>(
                ctx,
                bias.as_ref().unwrap(),
                SupportedTensorDtypes::REALHBF16,
                name,
            ))
        } else {
            None
        };
        convolution_wrapper::<CTYPE>(
            in_, weight, bias, load_bias, stride, padding, dilation, transposed, groups, out,
        );
    });

    out
}

// PORT-NOTE: `get_load_to_compute_fn::<CTYPE>` requires `CTYPE: ComputeCast +
// CppTypeToScalarType`, which hold for every REALHBF16 arm.
#[allow(dead_code)]
fn _assert_bounds<T: ComputeCast + CppTypeToScalarType>() {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor_impl::SizesType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    fn ir(v: &[i64]) -> IntArrayRef {
        IntArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
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

    fn test_conv3d_dtype<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + FactoryValue
            + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        let input = tf.full(
            vec![1, 2, 3, 3, 3],
            T::from_i32(1),
            TensorShapeDynamism::STATIC,
        );
        let weight = tf.full(
            vec![4, 2, 2, 2, 2],
            T::from_i32(1),
            TensorShapeDynamism::STATIC,
        );
        let bias: Option<Tensor> = None;
        let expected = tf.full(
            vec![1, 4, 2, 2, 2],
            T::from_i32(16),
            TensorShapeDynamism::STATIC,
        );
        let out = tf.zeros_default(vec![1, 4, 2, 2, 2]);

        let stride = [1i64, 1, 1];
        let padding = [0i64, 0, 0];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    fn test_dynamic_shape(out_shape: Vec<SizesType>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();

        let input = tf.make_default(
            vec![1, 2, 5],
            vec![5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3],
        );
        let weight = tf.make_default(
            vec![4, 2, 3],
            vec![
                8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1,
                8.9, 9.9, 7.9, 1.0, 1.1, 8.2, 6.3, 7.0,
            ],
        );
        let bias: Option<Tensor> = Some(tf.make_default(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let expected = tf.make_default(
            vec![1, 4, 2],
            vec![
                172.11, 237.72, 102.24, 132.28, 248.51, 320.18, 189.38, 236.07,
            ],
        );
        let out = tf.zeros(out_shape, dynamism);

        let stride = [2i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_generic_smoke_test() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.make_default(vec![1, 2, 5], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let weight = tf.make_default(
            vec![4, 2, 3],
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23,
            ],
        );
        let bias: Option<Tensor> = Some(tf.ones_default(vec![4]));
        let expected = tf.make_default(vec![1, 4, 2], vec![80, 110, 206, 308, 332, 506, 458, 704]);
        let out = tf.zeros_default(vec![1, 4, 2]);

        let stride = [2i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_non_zero_padding() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.make_default(
            vec![1, 2, 5],
            vec![5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3],
        );
        let weight = tf.make_default(
            vec![4, 2, 3],
            vec![
                8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1,
                8.9, 9.9, 7.9, 1.0, 1.1, 8.2, 6.3, 7.0,
            ],
        );
        let bias: Option<Tensor> = Some(tf.make_default(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let expected = tf.make_default(
            vec![1, 4, 4],
            vec![
                61.78, 172.11, 237.72, 79.7, 44.77, 102.24, 132.28, 34.87, 108.37, 248.51, 320.18,
                81.16, 62.24, 189.38, 236.07, 102.73,
            ],
        );
        let out = tf.zeros(vec![1, 4, 4], TensorShapeDynamism::STATIC);

        let stride = [2i64];
        let padding = [2i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_multiple_input_batches() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.make_default(
            vec![3, 2, 5],
            vec![
                5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3, 8.1, 6.6, 1.6, 4.9, 3.8, 6.6,
                4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0,
            ],
        );
        let weight = tf.make_default(
            vec![4, 2, 3],
            vec![
                1.1, 8.2, 6.3, 7.0, 6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3, 2.6, 8.9, 1.1, 8.7,
                2.3, 3.5, 4.2, 7.1, 5.0, 3.9, 3.3, 4.1,
            ],
        );
        let bias: Option<Tensor> = Some(tf.make_default(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let expected = tf.make_default(
            vec![3, 4, 4],
            vec![
                54.77, 168.21, 208.92, 57.93, 55.01, 241.19, 312.18, 121.3, 34.59, 143.87, 201.88,
                78.29, 60.39, 154.12, 194.07, 51.73, 68.53, 157.21, 105.33, 14.28, 75.19, 244.22,
                135.66, 48.70, 33.01, 160.36, 87.38, 22.19, 68.56, 142.28, 85.68, 22.03, 36.43,
                206.27, 235.96, 13.94, 36.79, 243.91, 338.66, 60.48, 22.81, 153.47, 210.56, 23.74,
                39.91, 174.16, 190.44, 27.58,
            ],
        );
        let out = tf.zeros(vec![3, 4, 4], TensorShapeDynamism::STATIC);

        let stride = [2i64];
        let padding = [2i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    // 2D numeric convolution spans arg validation, out-target-size, kernel-size
    // fill and the unsqueeze helpers used to normalize the 2D case; a wrong helper
    // fails the expected-tensor comparison.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-convolution-args-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.get-convolution-out-target-size-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.fill-convolution-kernel-size-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-sizes-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.get-unsqueezed-dim-order-fn/test]
    // The 2D convolution output pins convolution_wrapper's dispatch and conv2d_impl.
    // [spec:et:sem:op-convolution.torch.executor.native.convolution-wrapper-fn/test]
    // [spec:et:sem:op-convolution.torch.executor.native.conv2d-impl-fn/test]
    #[test]
    fn op_conv_correctness_test_2d_sanity_check() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.make_default(
            vec![1, 4, 8, 8],
            vec![
                5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3, 8.1, 6.6, 1.6, 4.9, 3.8, 6.6,
                4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0, 1.1, 8.2,
                6.3, 7.0, 6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3, 2.6, 8.9, 1.1, 8.7, 2.3, 3.5,
                4.2, 7.1, 5.0, 3.9, 3.3, 4.1, 8.1, 6.0, 3.3, 8.6, 6.6, 5.7, 5.9, 8.6, 7.3, 3.4,
                9.5, 6.0, 6.8, 6.2, 1.8, 3.2, 2.7, 7.5, 7.0, 8.0, 2.8, 5.1, 4.9, 8.6, 1.1, 9.0,
                4.2, 9.9, 2.4, 5.3, 4.9, 9.3, 2.9, 5.3, 8.9, 4.8, 9.5, 2.3, 9.2, 3.8, 6.5, 9.6,
                2.6, 3.5, 2.7, 9.2, 1.5, 7.6, 5.6, 8.5, 5.4, 7.0, 8.8, 5.1, 2.7, 1.8, 7.5, 4.4,
                2.4, 4.8, 1.4, 3.4, 8.9, 4.0, 4.7, 3.4, 2.5, 8.3, 8.3, 1.7, 2.3, 9.0, 2.9, 2.9,
                5.3, 7.1, 3.8, 7.1, 1.7, 9.8, 2.4, 4.1, 6.0, 8.4, 4.0, 1.4, 7.9, 7.7, 4.0, 4.0,
                9.1, 7.4, 4.9, 3.9, 3.5, 8.9, 2.2, 3.2, 8.2, 7.1, 5.4, 2.9, 8.1, 5.1, 3.0, 9.3,
                2.0, 3.6, 8.7, 6.6, 9.9, 3.1, 7.6, 3.4, 4.1, 5.0, 8.5, 9.2, 7.5, 5.8, 6.1, 5.8,
                4.1, 4.2, 9.8, 2.0, 7.3, 2.8, 7.9, 8.2, 9.7, 9.0, 4.8, 7.8, 6.6, 5.8, 4.5, 7.8,
                4.6, 8.5, 7.2, 4.4, 1.2, 7.7, 2.2, 2.4, 2.9, 1.8, 2.5, 2.6, 3.4, 6.3, 9.3, 8.4,
                3.0, 8.2, 1.5, 2.1, 3.2, 5.8, 5.2, 6.4, 1.8, 7.3, 7.6, 1.5, 2.8, 7.8, 9.0, 5.5,
                4.1, 2.3, 3.0, 8.8, 7.1, 7.1, 9.1, 3.7, 6.2, 6.2, 2.2, 1.3, 4.3, 5.6, 8.7, 6.8,
                5.0, 9.5, 5.0, 5.3, 5.5, 4.5, 3.3, 6.6, 6.2, 8.2, 5.5, 8.5, 2.9, 9.4, 8.3, 8.3,
            ],
        );
        let weight = tf.make_default(
            vec![2, 4, 3, 3],
            vec![
                4.7, 1.3, 7.8, 3.0, 9.7, 2.5, 3.8, 5.2, 4.4, 7.7, 2.3, 6.2, 1.5, 9.5, 6.3, 4.9,
                8.1, 9.8, 2.0, 6.6, 4.7, 2.4, 6.7, 5.6, 2.9, 1.3, 7.8, 5.4, 2.4, 6.9, 6.4, 1.4,
                8.9, 7.9, 7.5, 6.7, 4.0, 8.3, 5.2, 4.0, 4.8, 7.6, 7.1, 5.9, 9.1, 9.6, 3.9, 6.8,
                7.6, 2.5, 8.1, 7.3, 7.5, 7.5, 9.3, 5.6, 5.2, 4.7, 4.5, 8.7, 8.7, 1.3, 4.1, 4.5,
                4.9, 6.5, 7.9, 4.6, 7.0, 8.0, 1.6, 3.5,
            ],
        );
        let bias: Option<Tensor> = Some(tf.make_default(vec![2], vec![1.0, 1.0]));
        let expected = tf.make_default(
            vec![1, 2, 4, 4],
            vec![
                642.33, 714.6, 687.96, 717.12, 859.79, 939.27, 996.79, 1189.59, 700.73, 1083.28,
                1010.33, 1167.78, 776.33, 1138.92, 1073.43, 1140.64, 539.83, 851.42, 754.16,
                815.01, 822.66, 1191.95, 1063.46, 1330.28, 662.97, 1240.69, 1254.52, 1281.46,
                766.25, 1273.41, 1148.57, 1217.47,
            ],
        );
        let out = tf.zeros(vec![1, 2, 4, 4], TensorShapeDynamism::STATIC);

        let stride = [2i64, 2];
        let padding = [1i64, 1];
        let dilation = [1i64, 1];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_2d_sanity_check_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let contiguous_input = vec![
            5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3, 8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6,
            2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0, 1.1, 8.2, 6.3, 7.0,
            6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3, 2.6, 8.9, 1.1, 8.7, 2.3, 3.5, 4.2, 7.1, 5.0,
            3.9, 3.3, 4.1, 8.1, 6.0, 3.3, 8.6, 6.6, 5.7, 5.9, 8.6, 7.3, 3.4, 9.5, 6.0, 6.8, 6.2,
            1.8, 3.2, 2.7, 7.5, 7.0, 8.0, 2.8, 5.1, 4.9, 8.6, 1.1, 9.0, 4.2, 9.9, 2.4, 5.3, 4.9,
            9.3, 2.9, 5.3, 8.9, 4.8, 9.5, 2.3, 9.2, 3.8, 6.5, 9.6, 2.6, 3.5, 2.7, 9.2, 1.5, 7.6,
            5.6, 8.5, 5.4, 7.0, 8.8, 5.1, 2.7, 1.8, 7.5, 4.4, 2.4, 4.8, 1.4, 3.4, 8.9, 4.0, 4.7,
            3.4, 2.5, 8.3, 8.3, 1.7, 2.3, 9.0, 2.9, 2.9, 5.3, 7.1, 3.8, 7.1, 1.7, 9.8, 2.4, 4.1,
            6.0, 8.4, 4.0, 1.4, 7.9, 7.7, 4.0, 4.0, 9.1, 7.4, 4.9, 3.9, 3.5, 8.9, 2.2, 3.2, 8.2,
            7.1, 5.4, 2.9, 8.1, 5.1, 3.0, 9.3, 2.0, 3.6, 8.7, 6.6, 9.9, 3.1, 7.6, 3.4, 4.1, 5.0,
            8.5, 9.2, 7.5, 5.8, 6.1, 5.8, 4.1, 4.2, 9.8, 2.0, 7.3, 2.8, 7.9, 8.2, 9.7, 9.0, 4.8,
            7.8, 6.6, 5.8, 4.5, 7.8, 4.6, 8.5, 7.2, 4.4, 1.2, 7.7, 2.2, 2.4, 2.9, 1.8, 2.5, 2.6,
            3.4, 6.3, 9.3, 8.4, 3.0, 8.2, 1.5, 2.1, 3.2, 5.8, 5.2, 6.4, 1.8, 7.3, 7.6, 1.5, 2.8,
            7.8, 9.0, 5.5, 4.1, 2.3, 3.0, 8.8, 7.1, 7.1, 9.1, 3.7, 6.2, 6.2, 2.2, 1.3, 4.3, 5.6,
            8.7, 6.8, 5.0, 9.5, 5.0, 5.3, 5.5, 4.5, 3.3, 6.6, 6.2, 8.2, 5.5, 8.5, 2.9, 9.4, 8.3,
            8.3,
        ];
        let input = tf.make_channels_last(
            vec![1, 4, 8, 8],
            contiguous_input,
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let contiguous_weight = vec![
            4.7, 1.3, 7.8, 3.0, 9.7, 2.5, 3.8, 5.2, 4.4, 7.7, 2.3, 6.2, 1.5, 9.5, 6.3, 4.9, 8.1,
            9.8, 2.0, 6.6, 4.7, 2.4, 6.7, 5.6, 2.9, 1.3, 7.8, 5.4, 2.4, 6.9, 6.4, 1.4, 8.9, 7.9,
            7.5, 6.7, 4.0, 8.3, 5.2, 4.0, 4.8, 7.6, 7.1, 5.9, 9.1, 9.6, 3.9, 6.8, 7.6, 2.5, 8.1,
            7.3, 7.5, 7.5, 9.3, 5.6, 5.2, 4.7, 4.5, 8.7, 8.7, 1.3, 4.1, 4.5, 4.9, 6.5, 7.9, 4.6,
            7.0, 8.0, 1.6, 3.5,
        ];
        let weight = tf.make_channels_last(
            vec![2, 4, 3, 3],
            contiguous_weight,
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let bias: Option<Tensor> = Some(tf.make_default(vec![2], vec![1.0, 1.0]));
        let contiguous_expected = vec![
            624.92, 656.07, 710.91, 800.45, 622.48, 596.14, 831.26, 882.43, 812.8, 947.49, 1069.65,
            1155.81, 964.84, 1057.19, 1121.77, 1328.68, 748.23, 799.7, 1090.23, 1203.45, 1043.71,
            1124.75, 1140.41, 1265.35, 688.62, 807.57, 1073.07, 1109.53, 1110., 1221.82, 1210.86,
            1324.26,
        ];
        let expected = tf.make_channels_last(
            vec![1, 2, 4, 4],
            contiguous_expected,
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let out = tf.full_channels_last(vec![1, 2, 4, 4], 0.0, TensorShapeDynamism::STATIC);

        let stride = [2i64, 2];
        let padding = [1i64, 1];
        let dilation = [1i64, 1];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // The 3D convolution output pins conv3d_impl (via convolution_wrapper's dispatch).
    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    // [spec:et:sem:op-convolution.torch.executor.native.conv3d-impl-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_default_params() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![1, 2, 3, 3, 3], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![4, 2, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let expected = tf.full(vec![1, 4, 2, 2, 2], 16.0, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![1, 4, 2, 2, 2]);

        let stride = [1i64, 1, 1];
        let padding = [0i64, 0, 0];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_padding() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![1, 1, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![1, 1, 3, 3, 3], 1.0, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let expected = tf.full(vec![1, 1, 2, 2, 2], 8.0, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![1, 1, 2, 2, 2]);

        let stride = [1i64, 1, 1];
        let padding = [1i64, 1, 1];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_grouped_stride_and_bias() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![1, 4, 4, 4, 4], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![4, 2, 2, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = Some(tf.make_default(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
        let mut expected_data: Vec<f32> = Vec::new();
        for value in [9.0f32, 10.0, 11.0, 12.0] {
            for _ in 0..8 {
                expected_data.push(value);
            }
        }
        let expected = tf.make_default(vec![1, 4, 2, 2, 2], expected_data);
        let out = tf.zeros_default(vec![1, 4, 2, 2, 2]);

        let stride = [2i64, 2, 2];
        let padding = [0i64, 0, 0];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            2,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_dilation() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![1, 1, 7, 7, 7], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![1, 1, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let expected = tf.full(vec![1, 1, 5, 5, 5], 8.0, TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![1, 1, 5, 5, 5]);

        let stride = [1i64, 1, 1];
        let padding = [0i64, 0, 0];
        let dilation = [2i64, 2, 2];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_all_dtypes_supported() {
        test_conv3d_dtype::<u8>();
        test_conv3d_dtype::<i8>();
        test_conv3d_dtype::<i16>();
        test_conv3d_dtype::<i32>();
        test_conv3d_dtype::<i64>();
        test_conv3d_dtype::<f32>();
        test_conv3d_dtype::<f64>();
        test_conv3d_dtype::<Half>();
        test_conv3d_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_transposed_unsupported() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![1, 2, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![2, 1, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let out = tf.zeros_default(vec![1, 2, 3, 3, 3]);

        let stride = [1i64, 1, 1];
        let padding = [0i64, 0, 0];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            convolution_out(
                &mut ctx,
                &input,
                &weight,
                &bias,
                ir(&stride),
                ir(&padding),
                ir(&dilation),
                true,
                ir(&output_padding),
                2,
                &out,
            )
        );
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_conv3d_channels_last_unsupported() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full_channels_last(vec![1, 1, 3, 3, 3], 1.0, TensorShapeDynamism::STATIC);
        let weight = tf.full_channels_last(vec![1, 1, 2, 2, 2], 1.0, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let out = tf.zeros_channels_last(vec![1, 1, 2, 2, 2], TensorShapeDynamism::STATIC);

        let stride = [1i64, 1, 1];
        let padding = [0i64, 0, 0];
        let dilation = [1i64, 1, 1];
        let output_padding = [0i64, 0, 0];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            convolution_out(
                &mut ctx,
                &input,
                &weight,
                &bias,
                ir(&stride),
                ir(&padding),
                ir(&dilation),
                false,
                ir(&output_padding),
                1,
                &out,
            )
        );
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![1, 4, 2], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!output_resize, ...)`. The portable
    // (non-ATen) build sets `output_resize: false`, so this is skipped there.
    // Ignored to match.
    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    #[ignore = "output_resize unsupported in portable build (ET_SKIP_IF !output_resize)"]
    fn op_conv_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_invalid_input_shape() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.ones_default(vec![2, 4, 4, 5]);
        let weight = tf.ones_default(vec![8, 3, 2, 2]);
        let bias: Option<Tensor> = None;
        let out = tf.zeros_default(vec![2, 8, 3, 4]);

        let stride = [1i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];
        let groups = 2i64;

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            convolution_out(
                &mut ctx,
                &input,
                &weight,
                &bias,
                ir(&stride),
                ir(&padding),
                ir(&dilation),
                false,
                ir(&output_padding),
                groups,
                &out,
            )
        );

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            convolution_out(
                &mut ctx,
                &input,
                &weight,
                &bias,
                ir(&stride),
                ir(&padding),
                ir(&dilation),
                true,
                ir(&output_padding),
                groups,
                &out,
            )
        );
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_transposed_default_params() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![2, 4, 3, 2], 2.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![4, 1, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let out = tf.full(vec![2, 2, 4, 3], 0.7, TensorShapeDynamism::STATIC);
        let expected = tf.make_default(
            vec![2, 2, 4, 3],
            vec![
                2., 4., 2., 4., 8., 4., 4., 8., 4., 2., 4., 2., 2., 4., 2., 4., 8., 4., 4., 8., 4.,
                2., 4., 2., 2., 4., 2., 4., 8., 4., 4., 8., 4., 2., 4., 2., 2., 4., 2., 4., 8., 4.,
                4., 8., 4., 2., 4., 2.,
            ],
        );

        let stride = [1i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            true,
            ir(&output_padding),
            2,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_transposed_non_default_params() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![2, 6, 4, 5], 2.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![6, 1, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = Some(tf.make_default(vec![3], vec![1., 2., 3.]));
        let out = tf.full(vec![2, 3, 3, 6], 0.7, TensorShapeDynamism::STATIC);
        let expected = tf.make_default(
            vec![2, 3, 3, 6],
            vec![
                1., 1., 1., 1., 1., 1., 1., 3., 3., 1., 3., 3., 1., 3., 3., 1., 3., 3., 2., 2., 2.,
                2., 2., 2., 2., 4., 4., 2., 4., 4., 2., 4., 4., 2., 4., 4., 3., 3., 3., 3., 3., 3.,
                3., 5., 5., 3., 5., 5., 3., 5., 5., 3., 5., 5., 1., 1., 1., 1., 1., 1., 1., 3., 3.,
                1., 3., 3., 1., 3., 3., 1., 3., 3., 2., 2., 2., 2., 2., 2., 2., 4., 4., 2., 4., 4.,
                2., 4., 4., 2., 4., 4., 3., 3., 3., 3., 3., 3., 3., 5., 5., 3., 5., 5., 3., 5., 5.,
                3., 5., 5.,
            ],
        );

        let stride = [3i64];
        let padding = [7i64];
        let dilation = [5i64];
        let output_padding = [2i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            true,
            ir(&output_padding),
            3,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: the C++ `get_channels_last_data(expected)` physically reorders
    // contiguous data into channels-last, then `make_channels_last` applies the CL
    // dim order — logically equal to `channels_last_like(expected)`, used here.
    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_transposed_default_params_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full_channels_last(vec![2, 4, 3, 2], 2.0, TensorShapeDynamism::STATIC);
        let weight = tf.full_channels_last(vec![4, 1, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = None;
        let out = tf.full_channels_last(vec![2, 2, 4, 3], 0.7, TensorShapeDynamism::STATIC);
        let expected = tf.make_default(
            vec![2, 2, 4, 3],
            vec![
                2., 4., 2., 4., 8., 4., 4., 8., 4., 2., 4., 2., 2., 4., 2., 4., 8., 4., 4., 8., 4.,
                2., 4., 2., 2., 4., 2., 4., 8., 4., 4., 8., 4., 2., 4., 2., 2., 4., 2., 4., 8., 4.,
                4., 8., 4., 2., 4., 2.,
            ],
        );
        let expected_channels_last = tf.channels_last_like(&expected, TensorShapeDynamism::STATIC);

        let stride = [1i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            true,
            ir(&output_padding),
            2,
            &out,
        );
        assert_tensor_close!(out, expected_channels_last);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_transposed_non_default_params_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full_channels_last(vec![2, 6, 4, 5], 2.0, TensorShapeDynamism::STATIC);
        let weight = tf.full_channels_last(vec![6, 1, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = Some(tf.make_default(vec![3], vec![1., 2., 3.]));
        let out = tf.full_channels_last(vec![2, 3, 3, 6], 0.7, TensorShapeDynamism::STATIC);
        let expected = tf.make_default(
            vec![2, 3, 3, 6],
            vec![
                1., 1., 1., 1., 1., 1., 1., 3., 3., 1., 3., 3., 1., 3., 3., 1., 3., 3., 2., 2., 2.,
                2., 2., 2., 2., 4., 4., 2., 4., 4., 2., 4., 4., 2., 4., 4., 3., 3., 3., 3., 3., 3.,
                3., 5., 5., 3., 5., 5., 3., 5., 5., 3., 5., 5., 1., 1., 1., 1., 1., 1., 1., 3., 3.,
                1., 3., 3., 1., 3., 3., 1., 3., 3., 2., 2., 2., 2., 2., 2., 2., 4., 4., 2., 4., 4.,
                2., 4., 4., 2., 4., 4., 3., 3., 3., 3., 3., 3., 3., 5., 5., 3., 5., 5., 3., 5., 5.,
                3., 5., 5.,
            ],
        );
        let expected_channels_last = tf.channels_last_like(&expected, TensorShapeDynamism::STATIC);

        let stride = [3i64];
        let padding = [7i64];
        let dilation = [5i64];
        let output_padding = [2i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            true,
            ir(&output_padding),
            3,
            &out,
        );
        assert_tensor_close!(out, expected_channels_last);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    // Transposed conv with output_padding >= both stride and dilation must be
    // rejected — this is exactly the output_padding_is_valid predicate failing.
    // [spec:et:sem:kernel-ops-util.torch.executor.output-padding-is-valid-fn/test]
    #[test]
    fn op_conv_correctness_test_invalid_output_padding() {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![2, 6, 4, 5], 2.0, TensorShapeDynamism::STATIC);
        let weight = tf.full(vec![6, 1, 2, 2], 0.5, TensorShapeDynamism::STATIC);
        let bias: Option<Tensor> = Some(tf.make_default(vec![3], vec![1., 2., 3.]));
        let out = tf.zeros_default(vec![2, 3, 6, 9]);

        let stride = [3i64];
        let padding = [7i64];
        let dilation = [5i64];
        let output_padding = [5i64];
        let groups = 3i64;

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            convolution_out(
                &mut ctx,
                &input,
                &weight,
                &bias,
                ir(&stride),
                ir(&padding),
                ir(&dilation),
                true,
                ir(&output_padding),
                groups,
                &out,
            )
        );
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_half_type_smoke_test() {
        let tf = TensorFactory::<Half>::new();

        let input = tf.make_default(
            vec![1, 2, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .map(|&x: &f64| Half::from_f32(x as f32))
                .collect(),
        );
        let weight = tf.make_default(
            vec![2, 2, 2],
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
                .iter()
                .map(|&x: &f64| Half::from_f32(x as f32))
                .collect(),
        );
        let bias: Option<Tensor> = None;
        let expected = tf.make_default(
            vec![1, 2, 2],
            [6.0, 8.0, 12.0, 16.0]
                .iter()
                .map(|&x: &f64| Half::from_f32(x as f32))
                .collect(),
        );
        let out = tf.zeros_default(vec![1, 2, 2]);

        let stride = [1i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn/test]
    #[test]
    fn op_conv_correctness_test_bfloat16_type_smoke_test() {
        let tf = TensorFactory::<BFloat16>::new();

        let input = tf.make_default(
            vec![1, 2, 3],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
                .iter()
                .map(|&x: &f64| BFloat16::from_f32(x as f32))
                .collect(),
        );
        let weight = tf.make_default(
            vec![2, 2, 2],
            [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
                .iter()
                .map(|&x: &f64| BFloat16::from_f32(x as f32))
                .collect(),
        );
        let bias: Option<Tensor> = None;
        let expected = tf.make_default(
            vec![1, 2, 2],
            [6.0, 8.0, 12.0, 16.0]
                .iter()
                .map(|&x: &f64| BFloat16::from_f32(x as f32))
                .collect(),
        );
        let out = tf.zeros_default(vec![1, 2, 2]);

        let stride = [1i64];
        let padding = [0i64];
        let dilation = [1i64];
        let output_padding = [0i64];

        let mut ctx = context();
        convolution_out(
            &mut ctx,
            &input,
            &weight,
            &bias,
            ir(&stride),
            ir(&padding),
            ir(&dilation),
            false,
            ir(&output_padding),
            1,
            &out,
        );
        assert_tensor_close!(out, expected);
    }
}
