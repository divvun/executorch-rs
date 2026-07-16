//! Literal port of kernels/portable/cpu/op_upsample_bilinear2d.cpp.

use crate::kernels::portable::cpu::util::upsample_util::{
    OptionalArrayRef, area_pixel_compute_scale, check_upsample_bilinear2d_args,
    compute_source_index_and_lambda, resize_upsample_2d,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). The `const float scale_h/scale_w` kernel
// params keep `f32`; the entry point narrows the `f64` kernel ratio to `f32` at
// the call (mirroring the C++ `double`->`float` argument narrowing).
//
// PORT-NOTE: the interpolation math follows C++ `auto` deduction over
// `CTYPE * float`. c10::Half/BFloat16 `operator*(float)` return `float`, and
// `int/Half/BFloat16/float * float` promote to `float`, while `double * float`
// promotes to `double`. So the accumulation type is `f32` for every CTYPE
// except Double (`f64`). The `UpsampleArith` trait carries this promoted `Acc`
// plus the widen/narrow conversions the C++ overloads perform implicitly.

trait UpsampleArith: Copy {
    type Acc: Copy + core::ops::Add<Output = Self::Acc> + core::ops::Mul<Output = Self::Acc>;
    // static_cast<Acc>(self): CTYPE -> promoted float.
    fn to_acc(self) -> Self::Acc;
    // The `float` weight widened to `Acc` for the multiply.
    fn weight_to_acc(w: f32) -> Self::Acc;
    // Assigning the `Acc` result back into `*out_data` (narrow to CTYPE).
    fn from_acc(v: Self::Acc) -> Self;
}

macro_rules! impl_upsample_arith_f32 {
    ($t:ty) => {
        impl UpsampleArith for $t {
            type Acc = f32;
            #[inline]
            fn to_acc(self) -> f32 {
                self as f32
            }
            #[inline]
            fn weight_to_acc(w: f32) -> f32 {
                w
            }
            #[inline]
            fn from_acc(v: f32) -> Self {
                v as $t
            }
        }
    };
}
impl_upsample_arith_f32!(u8);
impl_upsample_arith_f32!(i8);
impl_upsample_arith_f32!(i16);
impl_upsample_arith_f32!(i32);
impl_upsample_arith_f32!(i64);
impl_upsample_arith_f32!(f32);

impl UpsampleArith for f64 {
    type Acc = f64;
    #[inline]
    fn to_acc(self) -> f64 {
        self
    }
    #[inline]
    fn weight_to_acc(w: f32) -> f64 {
        w as f64
    }
    #[inline]
    fn from_acc(v: f64) -> Self {
        v
    }
}

impl UpsampleArith for crate::runtime::core::portable_type::Half {
    type Acc = f32;
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn weight_to_acc(w: f32) -> f32 {
        w
    }
    #[inline]
    fn from_acc(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl UpsampleArith for crate::runtime::core::portable_type::BFloat16 {
    type Acc = f32;
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn weight_to_acc(w: f32) -> f32 {
        w
    }
    #[inline]
    fn from_acc(v: f32) -> Self {
        Self::from_f32(v)
    }
}

// [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]
// [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn]
fn upsample_bilinear2d_kernel_impl_nchw<CTYPE: UpsampleArith>(
    in_: &Tensor,
    align_corners: bool,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let mut out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    let mut in_plane: *const CTYPE = in_data;
    for _n in 0..out.size(0) {
        for _c in 0..out.size(1) {
            for h in 0..out.size(2) {
                // Compute source index and weights.
                let mut in_h1: i64 = 0;
                let mut in_h2: i64 = 0;
                let mut weight_h: f32 = 0.0;
                let mut inv_weight_h: f32 = 0.0;

                compute_source_index_and_lambda::<f32, f32>(
                    &mut in_h1,
                    &mut in_h2,
                    &mut weight_h,
                    &mut inv_weight_h,
                    scale_h,
                    h as i64,
                    *in_.sizes().at(2) as i64,
                    *out.sizes().at(2) as i64,
                    align_corners,
                );

                for w in 0..out.size(3) {
                    let mut in_w1: i64 = 0;
                    let mut in_w2: i64 = 0;
                    let mut weight_w: f32 = 0.0;
                    let mut inv_weight_w: f32 = 0.0;

                    compute_source_index_and_lambda::<f32, f32>(
                        &mut in_w1,
                        &mut in_w2,
                        &mut weight_w,
                        &mut inv_weight_w,
                        scale_w,
                        w as i64,
                        *in_.sizes().at(3) as i64,
                        *out.sizes().at(3) as i64,
                        align_corners,
                    );

                    let stride2: i64 = *in_.strides().at(2) as i64;
                    let stride3: i64 = *in_.strides().at(3) as i64;
                    let top_left: CTYPE =
                        unsafe { *in_plane.offset((in_h1 * stride2 + in_w1 * stride3) as isize) };
                    let top_right: CTYPE =
                        unsafe { *in_plane.offset((in_h1 * stride2 + in_w2 * stride3) as isize) };
                    let bottom_left: CTYPE =
                        unsafe { *in_plane.offset((in_h2 * stride2 + in_w1 * stride3) as isize) };
                    let bottom_right: CTYPE =
                        unsafe { *in_plane.offset((in_h2 * stride2 + in_w2 * stride3) as isize) };

                    let top = top_left.to_acc() * CTYPE::weight_to_acc(weight_w)
                        + top_right.to_acc() * CTYPE::weight_to_acc(inv_weight_w);
                    let bottom = bottom_left.to_acc() * CTYPE::weight_to_acc(weight_w)
                        + bottom_right.to_acc() * CTYPE::weight_to_acc(inv_weight_w);
                    let val = top * CTYPE::weight_to_acc(weight_h)
                        + bottom * CTYPE::weight_to_acc(inv_weight_h);

                    unsafe {
                        *out_data = CTYPE::from_acc(val);
                        out_data = out_data.add(1);
                    }
                }
            }

            in_plane = unsafe { in_plane.offset(*in_.strides().at(1) as isize) };
        }
    }
}

// [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn]
// [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn]
fn upsample_bilinear2d_kernel_impl_nhwc<CTYPE: UpsampleArith>(
    in_: &Tensor,
    align_corners: bool,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    let mut in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let mut out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    for _n in 0..out.size(0) {
        for h in 0..out.size(2) {
            // Compute source index and weights.
            let mut in_h1: i64 = 0;
            let mut in_h2: i64 = 0;
            let mut weight_h: f32 = 0.0;
            let mut inv_weight_h: f32 = 0.0;

            compute_source_index_and_lambda::<f32, f32>(
                &mut in_h1,
                &mut in_h2,
                &mut weight_h,
                &mut inv_weight_h,
                scale_h,
                h as i64,
                *in_.sizes().at(2) as i64,
                *out.sizes().at(2) as i64,
                align_corners,
            );

            for w in 0..out.size(3) {
                let mut in_w1: i64 = 0;
                let mut in_w2: i64 = 0;
                let mut weight_w: f32 = 0.0;
                let mut inv_weight_w: f32 = 0.0;

                compute_source_index_and_lambda::<f32, f32>(
                    &mut in_w1,
                    &mut in_w2,
                    &mut weight_w,
                    &mut inv_weight_w,
                    scale_w,
                    w as i64,
                    *in_.sizes().at(3) as i64,
                    *out.sizes().at(3) as i64,
                    align_corners,
                );

                let stride1: i64 = *in_.strides().at(1) as i64;
                let stride2: i64 = *in_.strides().at(2) as i64;
                let stride3: i64 = *in_.strides().at(3) as i64;
                for c in 0..out.size(1) {
                    let c: i64 = c as i64;
                    let top_left: CTYPE = unsafe {
                        *in_data.offset((in_h1 * stride2 + in_w1 * stride3 + c * stride1) as isize)
                    };
                    let top_right: CTYPE = unsafe {
                        *in_data.offset((in_h1 * stride2 + in_w2 * stride3 + c * stride1) as isize)
                    };
                    let bottom_left: CTYPE = unsafe {
                        *in_data.offset((in_h2 * stride2 + in_w1 * stride3 + c * stride1) as isize)
                    };
                    let bottom_right: CTYPE = unsafe {
                        *in_data.offset((in_h2 * stride2 + in_w2 * stride3 + c * stride1) as isize)
                    };

                    let top = top_left.to_acc() * CTYPE::weight_to_acc(weight_w)
                        + top_right.to_acc() * CTYPE::weight_to_acc(inv_weight_w);
                    let bottom = bottom_left.to_acc() * CTYPE::weight_to_acc(weight_w)
                        + bottom_right.to_acc() * CTYPE::weight_to_acc(inv_weight_w);
                    let val = top * CTYPE::weight_to_acc(weight_h)
                        + bottom * CTYPE::weight_to_acc(inv_weight_h);

                    unsafe {
                        *out_data = CTYPE::from_acc(val);
                        out_data = out_data.add(1);
                    }
                }
            }
        }

        in_data = unsafe { in_data.offset(*in_.strides().at(0) as isize) };
    }
}

// [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn]
// [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn]
fn upsample_bilinear2d_kernel_impl<CTYPE: UpsampleArith>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    align_corners: bool,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    if unsafe { is_contiguous_dim_order(in_.dim_order().data(), in_.dim_order().size()) } {
        upsample_bilinear2d_kernel_impl_nchw::<CTYPE>(in_, align_corners, scale_h, scale_w, out);
    } else if unsafe { is_channels_last_dim_order(in_.dim_order().data(), in_.dim_order().size()) }
    {
        upsample_bilinear2d_kernel_impl_nhwc::<CTYPE>(in_, align_corners, scale_h, scale_w, out);
    } else {
        // Shouldn't be reachable because of args checks, but just in case.
        crate::et_log!(Error, "Unsupported dim order");
        ctx.fail(Error::InvalidArgument);
    }
}

// Signatures are auto-generated, so disable pass-by-value lint.
// [spec:et:def:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn]
// [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn]
pub fn upsample_bilinear2d_vec_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    output_size: OptionalArrayRef<i64>,
    align_corners: bool,
    scale_factors: OptionalArrayRef<f64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Preconditions (checked in check_..._args):
    //  In and out tensors have same dtype.
    //  In and out tensors are rank 4 and have same dim[0] and dim[1].
    //  In and out tensors are NHWC or NCHW dim order.
    crate::et_kernel_check!(
        ctx,
        check_upsample_bilinear2d_args(in_, &output_size, align_corners, &scale_factors, out),
        InvalidArgument,
        out
    );

    let mut scale_h: f64 = 0.0;
    let mut scale_w: f64 = 0.0;

    crate::et_kernel_check_msg!(
        ctx,
        resize_upsample_2d(
            in_,
            &output_size,
            &scale_factors,
            &mut scale_h,
            &mut scale_w,
            out
        ) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor"
    );

    let kernel_scale_h: f64 = area_pixel_compute_scale::<f64>(
        *in_.sizes().at(2) as i64,
        *out.sizes().at(2) as i64,
        align_corners,
        &Some(scale_h),
    );
    let kernel_scale_w: f64 = area_pixel_compute_scale::<f64>(
        *in_.sizes().at(3) as i64,
        *out.sizes().at(3) as i64,
        align_corners,
        &Some(scale_w),
    );

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "upsample_bilinear2d.out", CTYPE, {
        upsample_bilinear2d_kernel_impl::<CTYPE>(
            ctx,
            in_,
            align_corners,
            kernel_scale_h as f32,
            kernel_scale_w as f32,
            out,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
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

    #[allow(clippy::too_many_arguments)]
    fn op_upsample_bilinear2d_vec_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        in_: &Tensor,
        output_size: OptionalArrayRef<i64>,
        align_corners: bool,
        scale_factors: OptionalArrayRef<f64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        upsample_bilinear2d_vec_out(ctx, in_, output_size, align_corners, scale_factors, out)
    }

    trait FromI64 {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI64 for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }

    fn ar_i64(v: &[i64]) -> OptionalArrayRef<i64> {
        Some(ArrayRef::from_raw_parts(v.as_ptr(), v.len()))
    }
    fn ar_f64(v: &[f64]) -> OptionalArrayRef<f64> {
        Some(ArrayRef::from_raw_parts(v.as_ptr(), v.len()))
    }

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, internal::K_DEFAULT_RTOL, None));
    }
    fn assert_close_with_tol(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) {
        assert!(tensors_are_close(a, b, rtol, Some(atol)));
    }
    fn assert_eq_t(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, 0.0, Some(0.0)));
    }
    // Mirrors EXPECT_FLOAT_EQ's ~4 ULP tolerance for the numerics checks.
    fn expect_float_eq(expected: f32, actual: f32) {
        let diff = (expected - actual).abs();
        let tol = 4.0 * f32::EPSILON * expected.abs().max(actual.abs()).max(1.0);
        assert!(diff <= tol, "expected {expected} got {actual}");
    }

    // template test_upsample_bilinear2d_dtype<CTYPE, DTYPE>
    fn test_upsample_bilinear2d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let input = tf.make_default(vec![1, 1, 1, 2], make_i64(&[1, 4]));
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let expected = tf.make_default(vec![1, 1, 1, 4], make_i64(&[1, 2, 3, 4]));
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    // also verifies the source-index/lambda helpers on the non-align-corners path:
    // 1x2 -> 1x4 yields interpolated values 1.0, 1.75, 3.25, 4.0, pinning
    // area_pixel_compute_source_index (real index incl. the <0 clamp),
    // guard_index_and_lambda (index floor/clamp + lambda = frac), and
    // compute_source_index_and_lambda (index0/index1 + lambda0/lambda1 pairing).
    // [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-source-index-fn/test]
    // [spec:et:sem:upsample-util.torch.executor.guard-index-and-lambda-fn/test]
    // [spec:et:sem:upsample-util.torch.executor.compute-source-index-and-lambda-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple1x2_to1x4() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 1, 2], vec![1.0, 4.0]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_default(vec![1, 1, 1, 4], vec![1.0, 1.75, 3.25, 4.0]);
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple1x2_to1x4_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 1], vec![1.0, 4.0]);
        let output_size: [i64; 2] = [4, 1];
        let out = tf.zeros_default(vec![1, 1, 4, 1]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let expected = tf.make_default(vec![1, 1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple2x1_to4x1() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 1], vec![1.0, 4.0]);
        let output_size: [i64; 2] = [4, 1];
        let out = tf.zeros_default(vec![1, 1, 4, 1]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_default(vec![1, 1, 4, 1], vec![1.0, 1.75, 3.25, 4.0]);
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple2x1_to4x1_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 1], vec![1.0, 4.0]);
        let output_size: [i64; 2] = [4, 1];
        let out = tf.zeros_default(vec![1, 1, 4, 1]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let expected = tf.make_default(vec![1, 1, 4, 1], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    // also verifies upsample_bilinear2d_kernel_impl (contiguous dispatch) and
    // upsample_bilinear2d_kernel_impl_nchw: the 2x3 -> 3x4 default-dim-order
    // bilinear expansion pins the exact 1.0,1.625,2.375,... interpolated grid.
    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-fn/test]
    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nchw-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_smoke_test() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output_size: [i64; 2] = [3, 4];
        let out = tf.zeros_default(vec![1, 1, 3, 4]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_default(
            vec![1, 1, 3, 4],
            vec![
                1.0, 1.625, 2.375, 3.0, 2.5, 3.125, 3.875, 4.5, 4.0, 4.625, 5.375, 6.0,
            ],
        );
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_smoke_test_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output_size: [i64; 2] = [3, 4];
        let out = tf.zeros_default(vec![1, 1, 3, 4]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let expected = tf.make_default(
            vec![1, 1, 3, 4],
            vec![
                1.0, 1.6667, 2.3333, 3.0, 2.5, 3.1667, 3.8333, 4.5, 4.0, 4.6667, 5.3333, 6.0,
            ],
        );
        assert_close_with_tol(&out, &expected, 0.0, 0.0001);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    // also verifies compute_scales_value: scale_factors given, so area_pixel_compute_scale
    // (non-align) returns 1/scale, producing the same interpolation as the
    // output_size-driven smoke_test.
    // [spec:et:sem:upsample-util.torch.executor.compute-scales-value-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_smoke_test_scales() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf.zeros_default(vec![1, 1, 3, 4]);
        let scale_factors: [f64; 2] = [3.0 / 2.0, 4.0 / 3.0];

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, None, false, ar_f64(&scale_factors), &out);

        let expected = tf.make_default(
            vec![1, 1, 3, 4],
            vec![
                1.0, 1.625, 2.375, 3.0, 2.5, 3.125, 3.875, 4.5, 4.0, 4.625, 5.375, 6.0,
            ],
        );
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_smoke_test_align_corners_scales() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf.zeros_default(vec![1, 1, 3, 4]);
        let scale_factors: [f64; 2] = [3.0 / 2.0, 4.0 / 3.0];

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, None, true, ar_f64(&scale_factors), &out);

        let expected = tf.make_default(
            vec![1, 1, 3, 4],
            vec![
                1.0, 1.6667, 2.3333, 3.0, 2.5, 3.1667, 3.8333, 4.5, 4.0, 4.6667, 5.3333, 6.0,
            ],
        );
        assert_close_with_tol(&out, &expected, 0.0, 0.0001);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_d_type() {
        test_upsample_bilinear2d_dtype::<u8>();
        test_upsample_bilinear2d_dtype::<i8>();
        test_upsample_bilinear2d_dtype::<i16>();
        test_upsample_bilinear2d_dtype::<i32>();
        test_upsample_bilinear2d_dtype::<i64>();
        test_upsample_bilinear2d_dtype::<f32>();
        test_upsample_bilinear2d_dtype::<f64>();
        test_upsample_bilinear2d_dtype::<Half>();
        test_upsample_bilinear2d_dtype::<BFloat16>();
    }

    // PORT-NOTE: `ET_SKIP_IF(output_resize, ...)`: portable output_resize is false,
    // so the failure body runs.
    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_mismatched_output_size_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                ar_i64(&output_size),
                false,
                None,
                &out
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    // also verifies check_upsample_bilinear2d_args (-> check_upsample_2d_common_args):
    // a rank-3 input fails the `in.dim() == 4` guard.
    // [spec:et:sem:upsample-util.torch.executor.check-upsample-bilinear2d-args-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_invalid_input_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                ar_i64(&output_size),
                false,
                None,
                &out
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_invalid_output_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                ar_i64(&output_size),
                false,
                None,
                &out
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_missing_output_size_or_scale_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let out = tf.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(&mut ctx, &input, None, false, None, &out)
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_both_output_size_and_scale_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let scale_factors: [f64; 2] = [2.0, 1.0];
        let out = tf.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                ar_i64(&output_size),
                false,
                ar_f64(&scale_factors),
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_mismatched_d_type_dies() {
        let tf = TensorFactory::<f32>::new();
        let tf2 = TensorFactory::<i64>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf2.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                ar_i64(&output_size),
                false,
                None,
                &out
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_computed_output_size_matches_expected() {
        let tf = TensorFactory::<f32>::new();

        let test_cases: [(i32, i32, f64, f64, i32, i32); 2] = [
            (10, 10, 9.99999, 9.55, 99, 95),
            (10, 10, 9.99999999, 0.1, 99, 1),
        ];

        for &(in_h, in_w, scale_h, scale_w, out_h, out_w) in test_cases.iter() {
            let input = tf.ones_default(vec![1, 1, in_h, in_w]);
            let out = tf.zeros_default(vec![1, 1, out_h, out_w]);
            let scale_factors: [f64; 2] = [scale_h, scale_w];

            let mut ctx = context();
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                None,
                false,
                ar_f64(&scale_factors),
                &out,
            );

            let expected = tf.ones_default(vec![1, 1, out_h, out_w]);
            assert_close(&out, &expected);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_zero_computed_output_size_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let out = tf.zeros_default(vec![1, 1, 1, 4]);
        let scale_factors: [f64; 2] = [1.0, 0.25];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                None,
                false,
                ar_f64(&scale_factors),
                &out
            )
        );
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure body runs.
    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_mismatched_dim_order_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let out = tf.zeros_channels_last(vec![1, 1, 1, 4], TensorShapeDynamism::STATIC);
        let scale_factors: [f64; 2] = [2.0, 2.0];

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_bilinear2d_vec_out(
                &mut ctx,
                &input,
                None,
                false,
                ar_f64(&scale_factors),
                &out
            )
        );
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_numerics_check() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![3, 7, 47, 99]);
        let out = tf.zeros_default(vec![3, 7, 291, 512]);
        let output_size: [i64; 2] = [291, 512];

        let input_ptr = input.mutable_data_ptr::<f32>();
        for i in 0..input.numel() as usize {
            unsafe { *input_ptr.add(i) = i as f32 };
        }

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let test_values: [(usize, usize, usize, usize, f32); 3] = [
            (0, 2, 60, 200, 10262.14453125),
            (1, 6, 5, 503, 60624.30078125),
            (2, 0, 111, 300, 66932.953125),
        ];

        let output_data = out.const_data_ptr::<f32>();
        let s = out.strides();
        for &(n, c, h, w, expected) in test_values.iter() {
            let idx = n * *s.at(0) as usize
                + c * *s.at(1) as usize
                + h * *s.at(2) as usize
                + w * *s.at(3) as usize;
            let actual = unsafe { *output_data.add(idx) };
            expect_float_eq(expected, actual);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_numerics_check_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![3, 7, 47, 99]);
        let out = tf.zeros_default(vec![3, 7, 291, 512]);
        let output_size: [i64; 2] = [291, 512];

        let input_ptr = input.mutable_data_ptr::<f32>();
        for i in 0..input.numel() as usize {
            unsafe { *input_ptr.add(i) = i as f32 };
        }

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let test_values: [(usize, usize, usize, usize, f32); 3] = [
            (0, 2, 60, 200, 10286.5634765625),
            (1, 6, 5, 503, 60663.98046875),
            (2, 0, 111, 300, 66942.625),
        ];

        let output_data = out.const_data_ptr::<f32>();
        let s = out.strides();
        for &(n, c, h, w, expected) in test_values.iter() {
            let idx = n * *s.at(0) as usize
                + c * *s.at(1) as usize
                + h * *s.at(2) as usize
                + w * *s.at(3) as usize;
            let actual = unsafe { *output_data.add(idx) };
            expect_float_eq(expected, actual);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple5x1_to4x1() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output_size: [i64; 2] = [4, 1];
        let out = tf.zeros_default(vec![1, 1, 4, 1]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_default(vec![1, 1, 4, 1], vec![1.125, 2.375, 3.625, 4.875]);
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple5x1_to4x1_align_corners() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output_size: [i64; 2] = [4, 1];
        let out = tf.zeros_default(vec![1, 1, 4, 1]);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let expected = tf.make_default(vec![1, 1, 4, 1], vec![1.0, 2.333333, 3.666667, 5.0]);
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_simple1x2_to1x4_channels_last() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_channels_last(
            vec![1, 1, 1, 2],
            vec![1.0, 4.0],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_channels_last(vec![1, 1, 1, 4], TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_channels_last(
            vec![1, 1, 1, 4],
            vec![1.0, 1.75, 3.25, 4.0],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    // also verifies upsample_bilinear2d_kernel_impl_nhwc: a channels-last 1x2x3x4
    // -> 1x2x6x8 expansion is dispatched to the NHWC path and pins the full
    // interleaved-channel interpolated output.
    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-kernel-impl-nhwc-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_smoke_test_channels_last() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_channels_last(
            vec![1, 2, 3, 4],
            vec![
                0.0, 12.0, 1.0, 13.0, 2.0, 14.0, 3.0, 15.0, 4.0, 16.0, 5.0, 17.0, 6.0, 18.0, 7.0,
                19.0, 8.0, 20.0, 9.0, 21.0, 10.0, 22.0, 11.0, 23.0,
            ],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let output_size: [i64; 2] = [6, 8];
        let out = tf.zeros_channels_last(vec![1, 2, 6, 8], TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let expected = tf.make_channels_last(
            vec![1, 2, 6, 8],
            vec![
                0.0, 12.0, 0.25, 12.25, 0.75, 12.75, 1.25, 13.25, 1.75, 13.75, 2.25, 14.25, 2.75,
                14.75, 3.0, 15.0, 1.0, 13.0, 1.25, 13.25, 1.75, 13.75, 2.25, 14.25, 2.75, 14.75,
                3.25, 15.25, 3.75, 15.75, 4.0, 16.0, 3.0, 15.0, 3.25, 15.25, 3.75, 15.75, 4.25,
                16.25, 4.75, 16.75, 5.25, 17.25, 5.75, 17.75, 6.0, 18.0, 5.0, 17.0, 5.25, 17.25,
                5.75, 17.75, 6.25, 18.25, 6.75, 18.75, 7.25, 19.25, 7.75, 19.75, 8.0, 20.0, 7.0,
                19.0, 7.25, 19.25, 7.75, 19.75, 8.25, 20.25, 8.75, 20.75, 9.25, 21.25, 9.75, 21.75,
                10.0, 22.0, 8.0, 20.0, 8.25, 20.25, 8.75, 20.75, 9.25, 21.25, 9.75, 21.75, 10.25,
                22.25, 10.75, 22.75, 11.0, 23.0,
            ],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_numerics_check_channels_last() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_channels_last(vec![3, 7, 47, 99], TensorShapeDynamism::STATIC);
        let out = tf.zeros_channels_last(vec![3, 7, 291, 512], TensorShapeDynamism::STATIC);
        let output_size: [i64; 2] = [291, 512];

        let input_ptr = input.mutable_data_ptr::<f32>();
        for i in 0..input.numel() as usize {
            unsafe { *input_ptr.add(i) = i as f32 };
        }

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), false, None, &out);

        let test_values: [(usize, usize, usize, usize, f32); 3] = [
            (0, 2, 60, 200, 6695.0137),
            (1, 6, 5, 503, 33524.098),
            (2, 0, 111, 300, 77678.68),
        ];

        let output_data = out.const_data_ptr::<f32>();
        let s = out.strides();
        for &(n, c, h, w, expected) in test_values.iter() {
            let idx = n * *s.at(0) as usize
                + c * *s.at(1) as usize
                + h * *s.at(2) as usize
                + w * *s.at(3) as usize;
            let actual = unsafe { *output_data.add(idx) };
            expect_float_eq(expected, actual);
        }
    }

    // [spec:et:sem:op-upsample-bilinear2d.torch.executor.native.upsample-bilinear2d-vec-out-fn/test]
    #[test]
    fn op_upsample_bilinear2d_test_numerics_check_align_corners_channels_last() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.zeros_channels_last(vec![3, 7, 47, 99], TensorShapeDynamism::STATIC);
        let out = tf.zeros_channels_last(vec![3, 7, 291, 512], TensorShapeDynamism::STATIC);
        let output_size: [i64; 2] = [291, 512];

        let input_ptr = input.mutable_data_ptr::<f32>();
        for i in 0..input.numel() as usize {
            unsafe { *input_ptr.add(i) = i as f32 };
        }

        let mut ctx = context();
        op_upsample_bilinear2d_vec_out(&mut ctx, &input, ar_i64(&output_size), true, None, &out);

        let test_values: [(usize, usize, usize, usize, f32); 3] = [
            (0, 2, 60, 200, 6865.9414),
            (1, 6, 5, 503, 33801.883),
            (2, 0, 111, 300, 77746.32),
        ];

        let output_data = out.const_data_ptr::<f32>();
        let s = out.strides();
        for &(n, c, h, w, expected) in test_values.iter() {
            let idx = n * *s.at(0) as usize
                + c * *s.at(1) as usize
                + h * *s.at(2) as usize
                + w * *s.at(3) as usize;
            let actual = unsafe { *output_data.add(idx) };
            expect_float_eq(expected, actual);
        }
    }
}
