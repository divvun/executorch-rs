//! Literal port of kernels/portable/cpu/op_upsample_nearest2d.cpp.

use crate::kernels::portable::cpu::util::upsample_util::{
    OptionalArrayRef, area_pixel_compute_scale, check_upsample_nearest2d_args,
    nearest_neighbor_compute_source_index, resize_upsample_2d,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::{
    is_channels_last_dim_order, is_contiguous_dim_order,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `const float scale_h/scale_w` params keep
// their `f32` type; the caller narrows the `f64` kernel ratio to `f32` at the
// call site (mirroring the C++ `double`->`float` argument narrowing).
// PORT-NOTE: `out_data` is advanced by 1 per write (a running `*mut CTYPE`
// cursor), mirroring the C++ `out_data++`.

// [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn]
// [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn]
fn upsample_nearest2d_kernel_impl_nchw<CTYPE: Copy>(
    in_: &Tensor,
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
                for w in 0..out.size(3) {
                    let in_h = nearest_neighbor_compute_source_index(
                        scale_h,
                        h as i64,
                        *in_.sizes().at(2) as i64,
                    );
                    let in_w = nearest_neighbor_compute_source_index(
                        scale_w,
                        w as i64,
                        *in_.sizes().at(3) as i64,
                    );

                    let stride2: i64 = *in_.strides().at(2) as i64;
                    let stride3: i64 = *in_.strides().at(3) as i64;
                    unsafe {
                        *out_data = *in_plane.offset((in_h * stride2 + in_w * stride3) as isize);
                        out_data = out_data.add(1);
                    }
                }
            }

            in_plane = unsafe { in_plane.offset(*in_.strides().at(1) as isize) };
        }
    }
}

// [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn]
// [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn]
fn upsample_nearest2d_kernel_impl_nhwc<CTYPE: Copy>(
    in_: &Tensor,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    let mut in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let mut out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    for _n in 0..out.size(0) {
        for h in 0..out.size(2) {
            let in_h =
                nearest_neighbor_compute_source_index(scale_h, h as i64, *in_.sizes().at(2) as i64);
            for w in 0..out.size(3) {
                let in_w = nearest_neighbor_compute_source_index(
                    scale_w,
                    w as i64,
                    *in_.sizes().at(3) as i64,
                );
                let stride1: i64 = *in_.strides().at(1) as i64;
                let stride2: i64 = *in_.strides().at(2) as i64;
                let stride3: i64 = *in_.strides().at(3) as i64;
                for c in 0..out.size(1) {
                    let c: i64 = c as i64;
                    unsafe {
                        *out_data = *in_data
                            .offset((in_h * stride2 + in_w * stride3 + c * stride1) as isize);
                        out_data = out_data.add(1);
                    }
                }
            }
        }

        in_data = unsafe { in_data.offset(*in_.strides().at(0) as isize) };
    }
}

// [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn]
// [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn]
fn upsample_nearest2d_kernel_impl<CTYPE: Copy>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    scale_h: f32,
    scale_w: f32,
    out: &Tensor,
) {
    if unsafe { is_contiguous_dim_order(in_.dim_order().data(), in_.dim_order().size()) } {
        upsample_nearest2d_kernel_impl_nchw::<CTYPE>(in_, scale_h, scale_w, out);
    } else if unsafe { is_channels_last_dim_order(in_.dim_order().data(), in_.dim_order().size()) }
    {
        upsample_nearest2d_kernel_impl_nhwc::<CTYPE>(in_, scale_h, scale_w, out);
    } else {
        // Shouldn't be reachable because of args checks, but just in case.
        crate::et_log!(Error, "Unsupported dim order");
        ctx.fail(Error::InvalidArgument);
    }
}

// [spec:et:def:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn]
// [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn]
pub fn upsample_nearest2d_vec_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    output_size: OptionalArrayRef<i64>,
    scale_factors: OptionalArrayRef<f64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Preconditions (checked in check_..._args):
    //  In and out tensors have same dtype.
    //  In and out tensors are rank 4 and have same dim[0] and dim[1].
    //  In and out tensors are default dim order (NCHW).
    crate::et_kernel_check!(
        ctx,
        check_upsample_nearest2d_args(in_, &output_size, &scale_factors, out),
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
        false,
        &Some(scale_h),
    );
    let kernel_scale_w: f64 = area_pixel_compute_scale::<f64>(
        *in_.sizes().at(3) as i64,
        *out.sizes().at(3) as i64,
        false,
        &Some(scale_w),
    );

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "upsample_nearest2d.out", CTYPE, {
        upsample_nearest2d_kernel_impl::<CTYPE>(
            ctx,
            in_,
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

    fn op_upsample_nearest2d_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        in_: &Tensor,
        output_size: OptionalArrayRef<i64>,
        scale_factors: OptionalArrayRef<f64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        upsample_nearest2d_vec_out(ctx, in_, output_size, scale_factors, out)
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

    fn assert_close(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, internal::K_DEFAULT_RTOL, None));
    }

    fn assert_eq_t(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, 0.0, Some(0.0)));
    }

    // template test_upsample_nearest2d_dtype<CTYPE, DTYPE>
    // PORT-NOTE: the C++ ATen dtype-skip is a no-op in the portable build.
    fn test_upsample_nearest2d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();

        let input = tf.make_default(vec![1, 1, 2, 2], make_i64(&[1, 2, 3, 4]));
        let output_size: [i64; 2] = [4, 4];
        let out = tf.zeros_default(vec![1, 1, 4, 4]);

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            Some(ArrayRef::from_raw_parts(
                output_size.as_ptr(),
                output_size.len(),
            )),
            None,
            &out,
        );

        let expected = tf.make_default(
            vec![1, 1, 4, 4],
            make_i64(&[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]),
        );
        assert_close(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies upsample_nearest2d_kernel_impl (contiguous dispatch) and
    // upsample_nearest2d_kernel_impl_nchw: the 2x2 -> 4x4 default-dim-order
    // nearest expansion pins the exact replicated-pixel output grid.
    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-fn/test]
    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nchw-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_smoke_test() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![0.1, 0.2, 1.1, 1.2]);
        let output_size: [i64; 2] = [4, 4];
        let out = tf.zeros_default(vec![1, 1, 4, 4]);

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            Some(ArrayRef::from_raw_parts(
                output_size.as_ptr(),
                output_size.len(),
            )),
            None,
            &out,
        );

        let expected = tf.make_default(
            vec![1, 1, 4, 4],
            vec![
                0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.2, 1.2,
            ],
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies area_pixel_compute_scale: with scale_factors given and
    // align_corners=false it returns compute_scales_value = 1/scale (here the
    // 2x scale drives the 2->4 source-index mapping).
    // [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_smoke_test_scale() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![0.1, 0.2, 1.1, 1.2]);
        let out = tf.zeros_default(vec![1, 1, 4, 4]);
        let scale_factors: [f64; 2] = [2.0, 2.0];

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            None,
            Some(ArrayRef::from_raw_parts(
                scale_factors.as_ptr(),
                scale_factors.len(),
            )),
            &out,
        );

        let expected = tf.make_default(
            vec![1, 1, 4, 4],
            vec![
                0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.2, 1.2,
            ],
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies nearest_neighbor_compute_source_index: the fractional 2->5 and
    // 2->9 mappings pin the `min(floor(dst*scale), in-1)` index picks (each output
    // row/col selects the expected input pixel).
    // [spec:et:sem:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_upsample_simple_fractional() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![0.1, 0.2, 1.1, 1.2]);
        let output_size: [i64; 2] = [5, 9];
        let out = tf.zeros_default(vec![1, 1, 5, 9]);

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            Some(ArrayRef::from_raw_parts(
                output_size.as_ptr(),
                output_size.len(),
            )),
            None,
            &out,
        );

        let expected = tf.make_default(
            vec![1, 1, 5, 9],
            vec![
                0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2,
                0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 1.1, 1.1, 1.1, 1.1, 1.1,
                1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2,
            ],
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_upsample_simple_fractional_scale() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(vec![1, 1, 2, 2], vec![0.1, 0.2, 1.1, 1.2]);
        let out = tf.zeros_default(vec![1, 1, 5, 9]);
        let scale_factors: [f64; 2] = [5.0 / 2.0, 9.0 / 2.0];

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            None,
            Some(ArrayRef::from_raw_parts(
                scale_factors.as_ptr(),
                scale_factors.len(),
            )),
            &out,
        );

        let expected = tf.make_default(
            vec![1, 1, 5, 9],
            vec![
                0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2,
                0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 1.1, 1.1, 1.1, 1.1, 1.1,
                1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2,
            ],
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_multi_batch_and_channel() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_default(
            vec![2, 2, 2, 2],
            vec![
                0.1, 0.2, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2, 7.1, 7.2,
            ],
        );
        let output_size: [i64; 2] = [4, 4];
        let out = tf.zeros_default(vec![2, 2, 4, 4]);

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            Some(ArrayRef::from_raw_parts(
                output_size.as_ptr(),
                output_size.len(),
            )),
            None,
            &out,
        );

        let expected = tf.make_default(
            vec![2, 2, 4, 4],
            vec![
                0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.2, 1.2,
                2.1, 2.1, 2.2, 2.2, 2.1, 2.1, 2.2, 2.2, 3.1, 3.1, 3.2, 3.2, 3.1, 3.1, 3.2, 3.2,
                4.1, 4.1, 4.2, 4.2, 4.1, 4.1, 4.2, 4.2, 5.1, 5.1, 5.2, 5.2, 5.1, 5.1, 5.2, 5.2,
                6.1, 6.1, 6.2, 6.2, 6.1, 6.1, 6.2, 6.2, 7.1, 7.1, 7.2, 7.2, 7.1, 7.1, 7.2, 7.2,
            ],
        );
        assert_eq_t(&out, &expected);
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_d_type() {
        test_upsample_nearest2d_dtype::<u8>();
        test_upsample_nearest2d_dtype::<i8>();
        test_upsample_nearest2d_dtype::<i16>();
        test_upsample_nearest2d_dtype::<i32>();
        test_upsample_nearest2d_dtype::<i64>();
        test_upsample_nearest2d_dtype::<f32>();
        test_upsample_nearest2d_dtype::<f64>();
        test_upsample_nearest2d_dtype::<Half>();
        test_upsample_nearest2d_dtype::<BFloat16>();
    }

    // PORT-NOTE: `ET_SKIP_IF(output_resize, ...)`: portable output_resize is false,
    // so the failure body runs.
    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_mismatched_output_size_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 5]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                Some(ArrayRef::from_raw_parts(
                    output_size.as_ptr(),
                    output_size.len()
                )),
                None,
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies check_upsample_nearest2d_args -> check_upsample_2d_common_args:
    // a rank-3 input fails the `in.dim() == 4` guard.
    // [spec:et:sem:upsample-util.torch.executor.check-upsample-nearest2d-args-fn/test]
    // [spec:et:sem:upsample-util.torch.executor.check-upsample-2d-common-args-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_invalid_input_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                Some(ArrayRef::from_raw_parts(
                    output_size.as_ptr(),
                    output_size.len()
                )),
                None,
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_invalid_output_rank_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                Some(ArrayRef::from_raw_parts(
                    output_size.as_ptr(),
                    output_size.len()
                )),
                None,
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_missing_output_size_or_scale_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let out = tf.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(&mut ctx, &input, None, None, &out)
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_both_output_size_and_scale_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let scale_factors: [f64; 2] = [1.0, 2.0];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                Some(ArrayRef::from_raw_parts(
                    output_size.as_ptr(),
                    output_size.len()
                )),
                Some(ArrayRef::from_raw_parts(
                    scale_factors.as_ptr(),
                    scale_factors.len()
                )),
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_mismatched_d_type_dies() {
        let tf = TensorFactory::<f32>::new();
        let tf2 = TensorFactory::<i64>::new();
        let input = tf.ones_default(vec![1, 1, 2]);
        let output_size: [i64; 2] = [1, 4];
        let out = tf2.zeros_default(vec![1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                Some(ArrayRef::from_raw_parts(
                    output_size.as_ptr(),
                    output_size.len()
                )),
                None,
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies resize_upsample_2d: with scale_factors given, target H/W =
    // (in * scale) truncated to SizesType (e.g. 10 * 9.99999 -> 99, 10 * 0.1 -> 1),
    // matching the pre-sized out tensor.
    // [spec:et:sem:upsample-util.torch.executor.resize-upsample-2d-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_computed_output_size_matches_expected() {
        let tf = TensorFactory::<f32>::new();

        // { in_h, in_w, scale_h, scale_w, out_h, out_w }
        let test_cases: [(i32, i32, f64, f64, i32, i32); 2] = [
            (10, 10, 9.99999, 9.55, 99, 95),
            (10, 10, 9.99999999, 0.1, 99, 1),
        ];

        for &(in_h, in_w, scale_h, scale_w, out_h, out_w) in test_cases.iter() {
            let input = tf.ones_default(vec![1, 1, in_h, in_w]);
            let scale_factors: [f64; 2] = [scale_h, scale_w];
            let out = tf.zeros_default(vec![1, 1, out_h, out_w]);

            let mut ctx = context();
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                None,
                Some(ArrayRef::from_raw_parts(
                    scale_factors.as_ptr(),
                    scale_factors.len(),
                )),
                &out,
            );

            let expected = tf.ones_default(vec![1, 1, out_h, out_w]);
            assert_eq_t(&out, &expected);
        }
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_zero_computed_output_size_dies() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.ones_default(vec![1, 1, 1, 2]);
        let scale_factors: [f64; 2] = [1.0, 0.25];
        let out = tf.zeros_default(vec![1, 1, 1, 4]);

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_upsample_nearest2d_out(
                &mut ctx,
                &input,
                None,
                Some(ArrayRef::from_raw_parts(
                    scale_factors.as_ptr(),
                    scale_factors.len()
                )),
                &out,
            )
        );
    }

    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-vec-out-fn/test]
    // also verifies upsample_nearest2d_kernel_impl_nhwc: a channels-last 1x2x2x2
    // -> 1x2x4x4 nearest expansion is dispatched to the NHWC path and pins the
    // full interleaved-channel replicated output.
    // [spec:et:sem:op-upsample-nearest2d.torch.executor.native.upsample-nearest2d-kernel-impl-nhwc-fn/test]
    #[test]
    fn op_upsample_nearest2d_test_smoke_test_channels_last() {
        let tf = TensorFactory::<f32>::new();
        let input = tf.make_channels_last(
            vec![1, 2, 2, 2],
            vec![0.1, 2.1, 0.2, 2.2, 1.1, 3.1, 1.2, 3.2],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        let output_size: [i64; 2] = [4, 4];
        let out = tf.zeros_channels_last(vec![1, 2, 4, 4], TensorShapeDynamism::STATIC);

        let mut ctx = context();
        op_upsample_nearest2d_out(
            &mut ctx,
            &input,
            Some(ArrayRef::from_raw_parts(
                output_size.as_ptr(),
                output_size.len(),
            )),
            None,
            &out,
        );

        let expected = tf.make_channels_last(
            vec![1, 2, 4, 4],
            vec![
                0.1, 2.1, 0.1, 2.1, 0.2, 2.2, 0.2, 2.2, 0.1, 2.1, 0.1, 2.1, 0.2, 2.2, 0.2, 2.2,
                1.1, 3.1, 1.1, 3.1, 1.2, 3.2, 1.2, 3.2, 1.1, 3.1, 1.1, 3.1, 1.2, 3.2, 1.2, 3.2,
            ],
            vec![],
            TensorShapeDynamism::STATIC,
        );
        assert_eq_t(&out, &expected);
    }
}
