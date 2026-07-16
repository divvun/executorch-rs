//! Literal port of kernels/portable/cpu/op_pixel_unshuffle.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_pixel_unshuffle_args, get_pixel_unshuffle_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, getLeadingDims, resize_tensor_same_type,
};
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn]
// [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn]
fn pixel_unshuffle_impl(in_: &Tensor, downscale_factor: i64, out: &Tensor) {
    let in_data: *const u8 = in_.const_data_ptr::<u8>();
    let out_data: *mut u8 = out.mutable_data_ptr::<u8>();
    let elem_size = in_.element_size();

    let leading_dims = getLeadingDims(in_, (in_.dim() - 3) as i64);
    let channels = out.size(in_.dim() - 3) as i64;
    let height = out.size(in_.dim() - 2) as i64;
    let width = out.size(in_.dim() - 1) as i64;

    let s = downscale_factor;
    let sub_channels = channels / (s * s);

    // output strides
    let stride_n = channels * height * width;
    let stride_c = s * s * height * width;
    let stride_s1 = s * height * width;
    let stride_s2 = height * width;
    let stride_h = width;

    // input tensor shape of [n, c, h, s1, w, s2]
    // output tensor shape of [n, c, s1, s2, h, w]
    let mut i: usize = 0;
    for n in 0..leading_dims as i64 {
        for c in 0..sub_channels {
            for h in 0..height {
                for s1 in 0..s {
                    for w in 0..width {
                        for s2 in 0..s {
                            let output_offset: i64 = n * stride_n
                                + c * stride_c
                                + s1 * stride_s1
                                + s2 * stride_s2
                                + h * stride_h
                                + w;
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    in_data.add(i * elem_size as usize),
                                    out_data.add(output_offset as usize * elem_size as usize),
                                    elem_size as usize,
                                );
                            }
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn]
// [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn]
pub fn pixel_unshuffle_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    downscale_factor: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_pixel_unshuffle_args(in_, downscale_factor, out),
        InvalidArgument,
        out
    );

    let mut expected_out_size: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_pixel_unshuffle_out_target_size(
            in_,
            downscale_factor,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }

    // Make sure the output tensor is the right size.
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    pixel_unshuffle_impl(in_, downscale_factor, out);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
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

    fn op_pixel_unshuffle_out<'a, 'b>(
        self_: &Tensor,
        upscale_factor: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        pixel_unshuffle_out(&mut ctx, self_, upscale_factor, out)
    }

    trait FromI64: Copy {
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

    fn test_pixel_unshuffle<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf_in = TensorFactory::<T>::new();

        let sizes = vec![1, 1, 4, 4];
        let out_sizes = vec![1, 4, 2, 2];

        let out = tf_in.zeros_default(out_sizes.clone());

        op_pixel_unshuffle_out(
            &tf_in.make_default(
                sizes,
                [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
                    .iter()
                    .map(|&v| T::from_i64(v))
                    .collect(),
            ),
            2,
            &out,
        );
        assert_tensor_eq!(
            out,
            tf_in.make_default(out_sizes, (0..16).map(T::from_i64).collect())
        );
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    // also verifies pixel_unshuffle_impl: the shuffled input [0,4,1,5,...] must map back
    // to 0..16, which fails unless the nested channel/height/width index math is exact.
    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-impl-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_all_real_dtypes_supported() {
        // ET_FORALL_REALHBBF16_TYPES: Byte,Char,Short,Int,Long,Float,Double,Half,BFloat16.
        test_pixel_unshuffle::<u8>();
        test_pixel_unshuffle::<i8>();
        test_pixel_unshuffle::<i16>();
        test_pixel_unshuffle::<i32>();
        test_pixel_unshuffle::<i64>();
        test_pixel_unshuffle::<f32>();
        test_pixel_unshuffle::<f64>();
        test_pixel_unshuffle::<Half>();
        test_pixel_unshuffle::<BFloat16>();
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    // also verifies check_pixel_unshuffle_args (arg gate) and
    // get_pixel_unshuffle_out_target_size (leading-dim copy + channel*factor^2
    // + h/factor + w/factor, out {1,4,1,4,2,2})
    // [spec:et:sem:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_larger_input_rank() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![1, 4, 1, 1, 4, 4]);

        let out_sizes = vec![1, 4, 1, 4, 2, 2];
        let out = tf.zeros_default(out_sizes.clone());

        op_pixel_unshuffle_out(&a, 2, &out);
        assert_tensor_eq!(out, tf.ones_default(out_sizes));
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_invalid_input_shape_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![1, 1, 7, 8]);
        let out = tf.zeros_default(vec![1, 4, 4, 4]);

        let mut ctx = context();
        pixel_unshuffle_out(&mut ctx, &a, 2, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_wrong_input_rank_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![1, 2]);
        let out = tf.zeros_default(vec![1, 2]);

        let mut ctx = context();
        pixel_unshuffle_out(&mut ctx, &a, 2, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_different_dtype_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![1, 2, 12, 12]);
        let out = tf_float.zeros_default(vec![1, 18, 4, 4]);

        let mut ctx = context();
        pixel_unshuffle_out(&mut ctx, &a, 3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-pixel-unshuffle.torch.executor.native.pixel-unshuffle-out-fn/test]
    #[test]
    fn op_pixel_unshuffle_out_test_negative_upscale_factor_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![1, 2, 12, 12]);
        let out = tf.zeros_default(vec![1, 18, 4, 4]);

        let mut ctx = context();
        pixel_unshuffle_out(&mut ctx, &a, -3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
