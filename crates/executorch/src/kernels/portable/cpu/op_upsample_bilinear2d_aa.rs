//! Literal port of kernels/portable/cpu/op_upsample_bilinear2d_aa.cpp.

use crate::kernels::portable::cpu::util::upsample_util::{
    OptionalArrayRef, area_pixel_compute_scale, check_upsample_bilinear2d_args,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::dim_order_util::is_contiguous_dim_order;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `const float scale_h/scale_w` keep `f32`;
// the entry point narrows the `f64` kernel ratio to `f32` at the call.
//
// PORT-NOTE: the accumulator `CTYPE value` narrows back to `CTYPE` on every
// `value += ...` (c10::Half/BFloat16 `operator+=` re-narrows each step; `int +=`
// truncates each step). The `AaArith` trait carries the promoted term type
// `Acc` (`f32` for every CTYPE except Double `f64`, matching `CTYPE*float`
// promotion) and the per-`+=` widen/narrow so this precision behaviour is
// reproduced bit-for-bit.
//
// PORT-NOTE: `T` in `compute_aa_weights_for_pixel<T>` and `bilinear_aa_filter<T>`
// is instantiated with `float`; ported as concrete `f32` fns rather than a
// generic, matching the single instantiation the kernel uses.

trait AaArith: Copy {
    type Acc: Copy + core::ops::Add<Output = Self::Acc> + core::ops::Mul<Output = Self::Acc>;
    // `CTYPE value = 0;`
    fn zero() -> Self;
    // static_cast<Acc>(self)
    fn to_acc(self) -> Self::Acc;
    fn acc_from_f32(w: f32) -> Self::Acc;
    // narrow the accumulator back to CTYPE (assignment of the `Acc` result).
    fn from_acc(v: Self::Acc) -> Self;
}

macro_rules! impl_aa_arith_f32 {
    ($t:ty) => {
        impl AaArith for $t {
            type Acc = f32;
            #[inline]
            fn zero() -> Self {
                0 as $t
            }
            #[inline]
            fn to_acc(self) -> f32 {
                self as f32
            }
            #[inline]
            fn acc_from_f32(w: f32) -> f32 {
                w
            }
            #[inline]
            fn from_acc(v: f32) -> Self {
                v as $t
            }
        }
    };
}
impl_aa_arith_f32!(u8);
impl_aa_arith_f32!(i8);
impl_aa_arith_f32!(i16);
impl_aa_arith_f32!(i32);
impl_aa_arith_f32!(i64);
impl_aa_arith_f32!(f32);

impl AaArith for f64 {
    type Acc = f64;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn to_acc(self) -> f64 {
        self
    }
    #[inline]
    fn acc_from_f32(w: f32) -> f64 {
        w as f64
    }
    #[inline]
    fn from_acc(v: f64) -> Self {
        v
    }
}

impl AaArith for crate::runtime::core::portable_type::Half {
    type Acc = f32;
    #[inline]
    fn zero() -> Self {
        Self::from_f32(0.0)
    }
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn acc_from_f32(w: f32) -> f32 {
        w
    }
    #[inline]
    fn from_acc(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl AaArith for crate::runtime::core::portable_type::BFloat16 {
    type Acc = f32;
    #[inline]
    fn zero() -> Self {
        Self::from_f32(0.0)
    }
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn acc_from_f32(w: f32) -> f32 {
        w
    }
    #[inline]
    fn from_acc(v: f32) -> Self {
        Self::from_f32(v)
    }
}

// Anti-aliasing filter matching PyTorch's implementation exactly
// [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn]
// [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn]
#[inline]
fn bilinear_aa_filter(mut x: f32) -> f32 {
    x = x.abs();
    if x < 1.0f32 { 1.0f32 - x } else { 0.0f32 }
}

// Compute anti-aliasing weights exactly matching PyTorch's algorithm
// [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn]
// [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn]
fn compute_aa_weights_for_pixel(
    output_idx: i64,
    scale: f32,
    input_size: i64,
    indices: &mut [i64; 4],
    weights: &mut [f32; 4],
    num_contributors: &mut i64,
) {
    // Use the provided scale directly instead of recalculating

    // PyTorch's center calculation for anti-aliasing
    // Always uses scale * (i + 0.5) for anti-aliasing, regardless of
    // align_corners
    let center: f32 = scale * (output_idx as f32 + 0.5f32);

    // PyTorch's support calculation for bilinear anti-aliasing
    // interp_size = 2 for bilinear, so base support = 1.0
    let support: f32 = if scale >= 1.0f32 {
        1.0f32 * scale
    } else {
        1.0f32
    };

    // PyTorch's exact range calculation
    let xmin: i64 = core::cmp::max((center - support + 0.5f32) as i64, 0i64);
    let xmax: i64 = core::cmp::min((center + support + 0.5f32) as i64, input_size);

    *num_contributors = core::cmp::min(xmax - xmin, 4i64);

    // Ensure we have at least one contributor
    if *num_contributors <= 0 {
        *num_contributors = 1;
        indices[0] = core::cmp::max(0i64, core::cmp::min(center as i64, input_size - 1));
        weights[0] = 1.0f32;
        // Clear unused weight slots
        let mut j: i64 = 1;
        while j < 4 {
            weights[j as usize] = 0.0f32;
            j += 1;
        }
        return;
    }

    // PyTorch's weight computation
    let mut total_weight: f32 = 0.0f32;
    let invscale: f32 = if scale >= 1.0f32 {
        1.0f32 / scale
    } else {
        1.0f32
    };

    let mut j: i64 = 0;
    while j < *num_contributors {
        let x: i64 = xmin + j;
        // PyTorch's exact weight formula: (j + xmin - center + 0.5) * invscale
        let arg: f32 = (j as f32 + xmin as f32 - center + 0.5f32) * invscale;
        let weight: f32 = bilinear_aa_filter(arg);
        indices[j as usize] = x;
        weights[j as usize] = weight;
        total_weight += weight;
        j += 1;
    }

    // Normalize weights to sum to 1 (PyTorch does this)
    if total_weight > 0.0f32 {
        let mut j: i64 = 0;
        while j < *num_contributors {
            weights[j as usize] /= total_weight;
            j += 1;
        }
    } else {
        // Fallback: if total weight is 0, set equal weights
        let equal_weight: f32 = 1.0f32 / (*num_contributors as f32);
        let mut j: i64 = 0;
        while j < *num_contributors {
            weights[j as usize] = equal_weight;
            j += 1;
        }
    }

    // Clear unused weight slots
    let mut j: i64 = *num_contributors;
    while j < 4 {
        weights[j as usize] = 0.0f32;
        j += 1;
    }
}

// [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn]
// [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn]
fn upsample_bilinear2d_aa_kernel_impl<CTYPE: AaArith>(
    _ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    _align_corners: bool,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let is_nchw: bool =
        unsafe { is_contiguous_dim_order(in_.dim_order().data(), in_.dim_order().size()) };

    if is_nchw {
        // NCHW layout
        let mut n: i64 = 0;
        while n < out.size(0) as i64 {
            let mut c: i64 = 0;
            while c < out.size(1) as i64 {
                let in_plane: *const CTYPE = unsafe {
                    in_data.offset(
                        ((n * in_.size(1) as i64 + c) * in_.size(2) as i64 * in_.size(3) as i64)
                            as isize,
                    )
                };
                let out_plane: *mut CTYPE = unsafe {
                    out_data.offset(
                        ((n * out.size(1) as i64 + c) * out.size(2) as i64 * out.size(3) as i64)
                            as isize,
                    )
                };

                let mut oh: i64 = 0;
                while oh < out.size(2) as i64 {
                    // Compute height weights for this output row
                    let mut h_indices: [i64; 4] = [0; 4];
                    let mut h_weights: [f32; 4] = [0.0; 4];
                    let mut h_num_contributors: i64 = 0;
                    compute_aa_weights_for_pixel(
                        oh,
                        scale_h,
                        in_.size(2) as i64,
                        &mut h_indices,
                        &mut h_weights,
                        &mut h_num_contributors,
                    );

                    let mut ow: i64 = 0;
                    while ow < out.size(3) as i64 {
                        // Compute width weights for this output column
                        let mut w_indices: [i64; 4] = [0; 4];
                        let mut w_weights: [f32; 4] = [0.0; 4];
                        let mut w_num_contributors: i64 = 0;
                        compute_aa_weights_for_pixel(
                            ow,
                            scale_w,
                            in_.size(3) as i64,
                            &mut w_indices,
                            &mut w_weights,
                            &mut w_num_contributors,
                        );

                        let mut value: CTYPE = CTYPE::zero();

                        // Apply anti-aliased interpolation
                        let mut ih_idx: i64 = 0;
                        while ih_idx < h_num_contributors {
                            let ih: i64 = h_indices[ih_idx as usize];
                            let h_weight: f32 = h_weights[ih_idx as usize];

                            let mut iw_idx: i64 = 0;
                            while iw_idx < w_num_contributors {
                                let iw: i64 = w_indices[iw_idx as usize];
                                let w_weight: f32 = w_weights[iw_idx as usize];

                                let in_val: CTYPE = unsafe {
                                    *in_plane.offset((ih * in_.size(3) as i64 + iw) as isize)
                                };
                                let term = in_val.to_acc()
                                    * CTYPE::acc_from_f32(h_weight)
                                    * CTYPE::acc_from_f32(w_weight);
                                value = CTYPE::from_acc(value.to_acc() + term);
                                iw_idx += 1;
                            }
                            ih_idx += 1;
                        }

                        unsafe {
                            *out_plane.offset((oh * out.size(3) as i64 + ow) as isize) = value;
                        }
                        ow += 1;
                    }
                    oh += 1;
                }
                c += 1;
            }
            n += 1;
        }
    } else {
        // NHWC layout
        let mut n: i64 = 0;
        while n < out.size(0) as i64 {
            let in_batch: *const CTYPE = unsafe {
                in_data.offset(
                    (n * in_.size(1) as i64 * in_.size(2) as i64 * in_.size(3) as i64) as isize,
                )
            };
            let out_batch: *mut CTYPE = unsafe {
                out_data.offset(
                    (n * out.size(1) as i64 * out.size(2) as i64 * out.size(3) as i64) as isize,
                )
            };

            let mut oh: i64 = 0;
            while oh < out.size(2) as i64 {
                // Compute height weights for this output row
                let mut h_indices: [i64; 4] = [0; 4];
                let mut h_weights: [f32; 4] = [0.0; 4];
                let mut h_num_contributors: i64 = 0;
                compute_aa_weights_for_pixel(
                    oh,
                    scale_h,
                    in_.size(2) as i64,
                    &mut h_indices,
                    &mut h_weights,
                    &mut h_num_contributors,
                );

                let mut ow: i64 = 0;
                while ow < out.size(3) as i64 {
                    // Compute width weights for this output column
                    let mut w_indices: [i64; 4] = [0; 4];
                    let mut w_weights: [f32; 4] = [0.0; 4];
                    let mut w_num_contributors: i64 = 0;
                    compute_aa_weights_for_pixel(
                        ow,
                        scale_w,
                        in_.size(3) as i64,
                        &mut w_indices,
                        &mut w_weights,
                        &mut w_num_contributors,
                    );

                    let mut c: i64 = 0;
                    while c < out.size(1) as i64 {
                        let mut value: CTYPE = CTYPE::zero();

                        // Apply anti-aliased interpolation
                        let mut ih_idx: i64 = 0;
                        while ih_idx < h_num_contributors {
                            let ih: i64 = h_indices[ih_idx as usize];
                            let h_weight: f32 = h_weights[ih_idx as usize];

                            let mut iw_idx: i64 = 0;
                            while iw_idx < w_num_contributors {
                                let iw: i64 = w_indices[iw_idx as usize];
                                let w_weight: f32 = w_weights[iw_idx as usize];

                                let in_val: CTYPE = unsafe {
                                    *in_batch.offset(
                                        ((ih * in_.size(3) as i64 + iw) * in_.size(1) as i64 + c)
                                            as isize,
                                    )
                                };
                                let term = in_val.to_acc()
                                    * CTYPE::acc_from_f32(h_weight)
                                    * CTYPE::acc_from_f32(w_weight);
                                value = CTYPE::from_acc(value.to_acc() + term);
                                iw_idx += 1;
                            }
                            ih_idx += 1;
                        }

                        unsafe {
                            *out_batch.offset(
                                ((oh * out.size(3) as i64 + ow) * out.size(1) as i64 + c) as isize,
                            ) = value;
                        }
                        c += 1;
                    }
                    ow += 1;
                }
                oh += 1;
            }
            n += 1;
        }
    }
}

// Check function for anti-aliased bilinear upsampling
// [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.check-upsample-bilinear2d-aa-args-fn]
// [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.check-upsample-bilinear2d-aa-args-fn]
pub fn check_upsample_bilinear2d_aa_args(
    in_: &Tensor,
    output_size: &OptionalArrayRef<i64>,
    align_corners: bool,
    scale_factors: &OptionalArrayRef<f64>,
    out: &Tensor,
) -> bool {
    // Use the same checks as regular bilinear upsampling
    check_upsample_bilinear2d_args(in_, output_size, align_corners, scale_factors, out)
}

// Main entry point for anti-aliased bilinear upsampling
// [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn]
// [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn]
pub fn _upsample_bilinear2d_aa_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    output_size: ArrayRef<i64>,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Preconditions (checked in check_..._args):
    //  In and out tensors have same dtype.
    //  In and out tensors are rank 4 and have same dim[0] and dim[1].
    //  In and out tensors are NHWC or NCHW dim order.

    // Custom validation for our specific interface (ArrayRef + optional
    // individual scales)
    crate::et_kernel_check!(ctx, in_.dim() == 4, InvalidArgument, out);
    crate::et_kernel_check!(ctx, out.dim() == 4, InvalidArgument, out);
    crate::et_kernel_check!(
        ctx,
        in_.scalar_type() == out.scalar_type(),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, output_size.size() == 2, InvalidArgument, out);
    crate::et_kernel_check!(
        ctx,
        *output_size.at(0) > 0 && *output_size.at(1) > 0,
        InvalidArgument,
        out
    );

    // Ensure output tensor has correct dimensions
    crate::et_kernel_check!(ctx, out.size(0) == in_.size(0), InvalidArgument, out); // batch
    crate::et_kernel_check!(ctx, out.size(1) == in_.size(1), InvalidArgument, out); // channels
    crate::et_kernel_check!(
        ctx,
        out.size(2) as i64 == *output_size.at(0),
        InvalidArgument,
        out
    ); // height
    crate::et_kernel_check!(
        ctx,
        out.size(3) as i64 == *output_size.at(1),
        InvalidArgument,
        out
    ); // width

    // Compute final scales - use provided scales if available, otherwise compute
    // from sizes
    let final_scale_h: f64;
    let final_scale_w: f64;
    if scale_h.is_some() && scale_w.is_some() {
        final_scale_h = scale_h.unwrap();
        final_scale_w = scale_w.unwrap();
    } else {
        // Compute scales from input/output sizes
        final_scale_h = *output_size.at(0) as f64 / in_.size(2) as f64;
        final_scale_w = *output_size.at(1) as f64 / in_.size(3) as f64;
    }

    let kernel_scale_h: f64 = area_pixel_compute_scale::<f64>(
        *in_.sizes().at(2) as i64,
        *out.sizes().at(2) as i64,
        align_corners,
        &Some(final_scale_h),
    );
    let kernel_scale_w: f64 = area_pixel_compute_scale::<f64>(
        *in_.sizes().at(3) as i64,
        *out.sizes().at(3) as i64,
        align_corners,
        &Some(final_scale_w),
    );

    crate::et_switch_realhbf16_types!(
        in_.scalar_type(),
        ctx,
        "_upsample_bilinear2d_aa.out",
        CTYPE,
        {
            upsample_bilinear2d_aa_kernel_impl::<CTYPE>(
                ctx,
                in_,
                align_corners,
                kernel_scale_h as f32,
                kernel_scale_w as f32,
                out,
            );
        }
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn op_upsample_bilinear2d_aa_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        input: &Tensor,
        output_size: &[i64],
        align_corners: bool,
        scales_h: Option<f64>,
        scales_w: Option<f64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        _upsample_bilinear2d_aa_out(
            ctx,
            input,
            ArrayRef::from_raw_parts(output_size.as_ptr(), output_size.len()),
            align_corners,
            scales_h,
            scales_w,
            out,
        )
    }

    // PORT-NOTE: `check_upsample_bilinear2d_aa_args` is a public helper that is
    // NOT invoked by `_upsample_bilinear2d_aa_out` (which does its own inline
    // validation) and has no dedicated C++ test. It is a pure forwarder to
    // `check_upsample_bilinear2d_args`, so this focused unit test pins that
    // forwarding: valid rank-4 same-dtype tensors with an output_size pass, a
    // rank-3 input fails.
    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.check-upsample-bilinear2d-aa-args-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_check_args_forwards() {
        let tf = TensorFactory::<f32>::new();
        let output_size: [i64; 2] = [4, 4];
        let os: OptionalArrayRef<i64> = Some(ArrayRef::from_raw_parts(
            output_size.as_ptr(),
            output_size.len(),
        ));
        let no_scales: OptionalArrayRef<f64> = None;

        let in_ok = tf.ones_default(vec![1, 1, 2, 2]);
        let out_ok = tf.zeros_default(vec![1, 1, 4, 4]);
        assert!(check_upsample_bilinear2d_aa_args(
            &in_ok, &os, false, &no_scales, &out_ok
        ));

        // rank-3 input fails the `in.dim() == 4` gate in the common args check.
        let in_bad = tf.ones_default(vec![1, 2, 2]);
        assert!(!check_upsample_bilinear2d_aa_args(
            &in_bad, &os, false, &no_scales, &out_ok
        ));
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_smoke_test2x_upsample_nchw() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros_default(vec![1, 1, 4, 4]);
        let output_size: [i64; 2] = [4, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 4);
        assert_eq!(out.size(3), 4);

        let out_data = out.const_data_ptr::<f32>();
        let mut has_non_zero = false;
        for i in 0..16 {
            if unsafe { *out_data.add(i) } != 0.0 {
                has_non_zero = true;
                break;
            }
        }
        assert!(has_non_zero);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_with_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 2, 3, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        );
        let out = tf.zeros_default(vec![1, 2, 6, 6]);
        let output_size: [i64; 2] = [6, 6];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, true, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 2);
        assert_eq!(out.size(2), 6);
        assert_eq!(out.size(3), 6);

        let in_data = input.const_data_ptr::<f32>();
        let out_data = out.const_data_ptr::<f32>();

        // Relaxed tolerance due to implementation differences.
        assert!((unsafe { *out_data } - unsafe { *in_data }).abs() <= 0.35);
        assert!((unsafe { *out_data.add(5) } - unsafe { *in_data.add(2) }).abs() <= 0.35);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_downsample() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 1, 4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let out = tf.zeros_default(vec![1, 1, 2, 2]);
        let output_size: [i64; 2] = [2, 2];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 2);
        assert_eq!(out.size(3), 2);

        let out_data = out.const_data_ptr::<f32>();
        for i in 0..4 {
            let v = unsafe { *out_data.add(i) };
            assert!(v > 0.0);
            assert!(v < 17.0);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_batched_input() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![2, 3, 2, 2],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        );
        let out = tf.zeros_default(vec![2, 3, 4, 4]);
        let output_size: [i64; 2] = [4, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 2);
        assert_eq!(out.size(1), 3);
        assert_eq!(out.size(2), 4);
        assert_eq!(out.size(3), 4);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_with_scale_factors() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let output_size: [i64; 2] = [6, 6];
        let out = tf.zeros_default(vec![1, 1, 6, 6]);

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(
            &mut ctx,
            &input,
            &output_size,
            false,
            Some(2.0),
            Some(2.0),
            &out,
        );

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 6);
        assert_eq!(out.size(3), 6);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_asymmetric_scaling() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 2, 3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        );
        let out = tf.zeros_default(vec![1, 2, 6, 12]);
        let output_size: [i64; 2] = [6, 12];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 2);
        assert_eq!(out.size(2), 6);
        assert_eq!(out.size(3), 12);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_edge_case_one_by_one() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 3, 1, 1], vec![1.0, 2.0, 3.0]);
        let out = tf.zeros_default(vec![1, 3, 4, 4]);
        let output_size: [i64; 2] = [4, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 3);
        assert_eq!(out.size(2), 4);
        assert_eq!(out.size(3), 4);

        let in_data = input.const_data_ptr::<f32>();
        let out_data = out.const_data_ptr::<f32>();
        for c in 0..3 {
            for i in 0..16 {
                let d = (unsafe { *out_data.add(c * 16 + i) } - unsafe { *in_data.add(c) }).abs();
                assert!(d <= 0.01);
            }
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    // also verifies upsample_bilinear2d_aa_kernel_impl (NCHW anti-aliased
    // accumulation) plus compute_aa_weights_for_pixel and bilinear_aa_filter:
    // a 3x3 -> 3x3 identity transform must reproduce the input within 0.01, which
    // only holds if the per-pixel weight indices/values and the triangular filter
    // are computed correctly and normalize to 1.
    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn/test]
    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn/test]
    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_identity_transform() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let out = tf.zeros_default(vec![1, 1, 3, 3]);
        let output_size: [i64; 2] = [3, 3];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        let in_data = input.const_data_ptr::<f32>();
        let out_data = out.const_data_ptr::<f32>();
        for i in 0..9 {
            let d = (unsafe { *out_data.add(i) } - unsafe { *in_data.add(i) }).abs();
            assert!(d <= 0.01);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_large_downsample() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_default(vec![1, 1, 8, 8]);
        let in_data = input.mutable_data_ptr::<f32>();
        for i in 0..64 {
            unsafe { *in_data.add(i) = i as f32 };
        }

        let out = tf.zeros_default(vec![1, 1, 2, 2]);
        let output_size: [i64; 2] = [2, 2];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 2);
        assert_eq!(out.size(3), 2);

        let out_data = out.const_data_ptr::<f32>();
        for i in 0..4 {
            let v = unsafe { *out_data.add(i) };
            assert!(v > 0.0);
            assert!(v < 64.0);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_double_data_type() {
        let tf = TensorFactory::<f64>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let out = tf.zeros_default(vec![1, 1, 4, 4]);
        let output_size: [i64; 2] = [4, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 4);
        assert_eq!(out.size(3), 4);

        let out_data = out.const_data_ptr::<f64>();
        assert!(unsafe { *out_data } > 0.0);
        assert!(unsafe { *out_data } < 5.0);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_uint8_data_type() {
        let tf = TensorFactory::<u8>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![50, 100, 150, 200]);
        let out = tf.zeros_default(vec![1, 1, 4, 4]);
        let output_size: [i64; 2] = [4, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 4);
        assert_eq!(out.size(3), 4);

        let out_data = out.const_data_ptr::<u8>();
        for i in 0..16 {
            let v = unsafe { *out_data.add(i) };
            assert!(v >= 40);
            assert!(v <= 210);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_fractional_downsample() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_default(vec![1, 2, 5, 7]);
        let in_data = input.mutable_data_ptr::<f32>();
        for i in 0..70 {
            unsafe { *in_data.add(i) = i as f32 };
        }

        let out = tf.zeros_default(vec![1, 2, 3, 4]);
        let output_size: [i64; 2] = [3, 4];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 2);
        assert_eq!(out.size(2), 3);
        assert_eq!(out.size(3), 4);

        let out_data = out.const_data_ptr::<f32>();
        for i in 0..24 {
            let v = unsafe { *out_data.add(i) };
            assert!(v >= 0.0);
            assert!(v <= 70.0);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_large_batch_size() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_default(vec![5, 8, 4, 4]);
        let in_data = input.mutable_data_ptr::<f32>();
        for n in 0..5 {
            for c in 0..8 {
                for i in 0..16 {
                    unsafe {
                        *in_data.add(n * 8 * 16 + c * 16 + i) = (n * 100 + c * 10 + i) as f32;
                    }
                }
            }
        }

        let out = tf.zeros_default(vec![5, 8, 2, 2]);
        let output_size: [i64; 2] = [2, 2];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 5);
        assert_eq!(out.size(1), 8);
        assert_eq!(out.size(2), 2);
        assert_eq!(out.size(3), 2);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_extreme_downsample() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_default(vec![1, 1, 16, 16]);
        let in_data = input.mutable_data_ptr::<f32>();
        for h in 0..16 {
            for w in 0..16 {
                unsafe {
                    *in_data.add(h * 16 + w) = if (h + w) % 2 == 0 { 1.0 } else { 0.0 };
                }
            }
        }

        let out = tf.zeros_default(vec![1, 1, 1, 1]);
        let output_size: [i64; 2] = [1, 1];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 1);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 1);
        assert_eq!(out.size(3), 1);

        let out_data = out.const_data_ptr::<f32>();
        assert!(unsafe { *out_data } > 0.3);
        assert!(unsafe { *out_data } < 0.7);
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_consistency_between_scales_and_output_size() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 2, 3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        );

        let out1 = tf.zeros_default(vec![1, 2, 6, 8]);
        let output_size: [i64; 2] = [6, 8];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out1);

        let out2 = tf.zeros_default(vec![1, 2, 6, 8]);
        op_upsample_bilinear2d_aa_out(
            &mut ctx,
            &input,
            &output_size,
            false,
            Some(2.0),
            Some(2.0),
            &out2,
        );

        let out1_data = out1.const_data_ptr::<f32>();
        let out2_data = out2.const_data_ptr::<f32>();
        for i in 0..48 {
            let d = (unsafe { *out1_data.add(i) } - unsafe { *out2_data.add(i) }).abs();
            assert!(d <= 1e-4);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_non_square_input_output() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![2, 1, 2, 6],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
        );
        let out = tf.zeros_default(vec![2, 1, 5, 3]);
        let output_size: [i64; 2] = [5, 3];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out);

        assert_eq!(out.size(0), 2);
        assert_eq!(out.size(1), 1);
        assert_eq!(out.size(2), 5);
        assert_eq!(out.size(3), 3);

        let out_data = out.const_data_ptr::<f32>();
        for i in 0..30 {
            let v = unsafe { *out_data.add(i) };
            assert!(v >= 0.0);
            assert!(v <= 25.0);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_precision_consistency() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![1, 1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );

        let out1 = tf.zeros_default(vec![1, 1, 7, 7]);
        let out2 = tf.zeros_default(vec![1, 1, 7, 7]);
        let output_size: [i64; 2] = [7, 7];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out1);
        op_upsample_bilinear2d_aa_out(&mut ctx, &input, &output_size, false, None, None, &out2);

        let out1_data = out1.const_data_ptr::<f32>();
        let out2_data = out2.const_data_ptr::<f32>();
        for i in 0..49 {
            assert_eq!(unsafe { *out1_data.add(i) }, unsafe { *out2_data.add(i) });
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_aa_out_test_test_specific_input_case() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_default(vec![8, 2, 7, 1]);
        let in_data = input.mutable_data_ptr::<f32>();
        for i in 0..(8 * 2 * 7 * 1) {
            unsafe { *in_data.add(i) = i as f32 * 0.1 };
        }

        let out = tf.zeros_default(vec![8, 2, 7, 2]);
        let output_size: [i64; 2] = [7, 2];

        let mut ctx = context();
        op_upsample_bilinear2d_aa_out(
            &mut ctx,
            &input,
            &output_size,
            false,
            Some(0.010000000000000002),
            Some(10.0),
            &out,
        );

        assert_eq!(out.size(0), 8);
        assert_eq!(out.size(1), 2);
        assert_eq!(out.size(2), 7);
        assert_eq!(out.size(3), 2);

        let out_data = out.const_data_ptr::<f32>();
        for i in 0..(8 * 2 * 7 * 2) {
            let v = unsafe { *out_data.add(i) };
            assert!(!v.is_nan());
            assert!(!v.is_infinite());
        }
    }
}
