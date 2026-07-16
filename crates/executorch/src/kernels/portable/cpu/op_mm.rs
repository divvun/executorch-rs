//! Literal port of kernels/portable/cpu/op_mm.cpp.

use crate::kernels::portable::cpu::util::matmul_ops_util::{check_mm_args, get_mm_out_target_size};
use crate::kernels::portable::cpu::vec_ops::vec_matmul;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `kTensorDimensionLimit` maps to
// `K_TENSOR_DIMENSION_LIMIT`, `SizesType` to `tensor_impl::SizesType`.
// `ET_SWITCH_REAL_TYPES_AND2(Half, BFloat16, ...)` maps to the established
// `et_switch_realhbf16_types!` (real dtypes + Half + BFloat16, no Bool), as in
// op_addmm. `get_mm_out_target_size` and `vec_matmul` are `unsafe` (raw pointer
// output), so their calls are wrapped in `unsafe`.

// [spec:et:def:op-mm.torch.executor.native.mm-out-fn]
// [spec:et:sem:op-mm.torch.executor.native.mm-out-fn]
#[executorch_macros::et_kernel("aten::mm.out")]
pub fn mm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mat2: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(ctx, check_mm_args(in_, mat2, out), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_mm_out_target_size(in_, mat2, output_sizes.as_mut_ptr(), &mut output_ndim);
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, mat2, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "mm.out", CTYPE, {
        let m: i64 = in_.size(0) as i64;
        let n: i64 = in_.size(1) as i64;
        let p: i64 = mat2.size(1) as i64;

        unsafe {
            vec_matmul::<CTYPE>(
                out.mutable_data_ptr::<CTYPE>(),
                in_.const_data_ptr::<CTYPE>(),
                mat2.const_data_ptr::<CTYPE>(),
                m,
                n,
                p,
            );
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_mm_out<'a, 'b>(self_: &Tensor, mat2: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        mm_out(&mut ctx, self_, mat2, out)
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        // matmul gives 4 * 2 * 3 = 24
        let x = tf.full(vec![3, 4], T::from_f64(2.0), TensorShapeDynamism::STATIC);
        let y = tf.full(vec![4, 5], T::from_f64(3.0), TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![3, 5]);
        op_mm_out(&x, &y, &out);
        let expected = tf.full(vec![3, 5], T::from_f64(24.0), TensorShapeDynamism::STATIC);
        assert_tensor_eq!(out, expected);
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
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

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    // also verifies get_mm_out_target_size (out resized to [3,5] = [in.size(0), mat2.size(1)])
    // [spec:et:sem:matmul-ops-util.torch.executor.get-mm-out-target-size-fn/test]
    #[test]
    fn op_mm_out_test_output_dim() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.ones_default(vec![3, 4]);
        let y = tf.ones_default(vec![4, 5]);
        let out = tf.zeros_default(vec![3, 5]);
        let ret = op_mm_out(&x, &y, &out);
        assert_tensor_eq!(*ret, out);
        let expected = tf.full(vec![3, 5], 4, TensorShapeDynamism::STATIC);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_all_dtypes_supported() {
        // ET_FORALL_REALHBF16_TYPES
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_empty_input_with_empty_out_tensor_passes() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![0, 3], vec![]);
        let y = tf.make_default(vec![3, 0], vec![]);
        let out = tf.make_default(vec![0, 0], vec![]);
        let expected = tf.make_default(vec![0, 0], vec![]);
        assert_tensor_eq!(*op_mm_out(&x, &y, &out), expected);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_infinity_tensor_passes() {
        let tff = TensorFactory::<f32>::new();
        let x = tff.full(vec![3, 4], f32::INFINITY, TensorShapeDynamism::STATIC);
        let y = tff.full(vec![4, 5], 3.0f32, TensorShapeDynamism::STATIC);
        let out = tff.zeros_default(vec![3, 5]);
        let expected = tff.full(vec![3, 5], f32::INFINITY, TensorShapeDynamism::STATIC);
        assert_tensor_eq!(*op_mm_out(&x, &y, &out), expected);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    // also verifies check_mm_args rejects mat2 with mismatched inner dim (in.size(1) != mat2.size(0))
    // [spec:et:sem:matmul-ops-util.torch.executor.check-mm-args-fn/test]
    #[test]
    fn op_mm_out_test_mismatched_dimensions_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![2, 2], 3, TensorShapeDynamism::STATIC);
        let wrong_y = tf.full(vec![3, 1], 1, TensorShapeDynamism::STATIC);
        let right_y = tf.full(vec![2, 2], 1, TensorShapeDynamism::STATIC);
        let out = tf.full(vec![2, 2], 0, TensorShapeDynamism::STATIC);
        let expected = tf.full(vec![2, 2], 6, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        mm_out(&mut ctx, &x, &wrong_y, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        assert_tensor_eq!(*op_mm_out(&x, &right_y, &out), expected);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_mismatched_dimension_size_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.full(vec![2, 2], 3, TensorShapeDynamism::STATIC);
        let wrong_y = tf.full(vec![2, 2, 2], 1, TensorShapeDynamism::STATIC);
        let right_y = tf.full(vec![2, 2], 1, TensorShapeDynamism::STATIC);
        let right_out = tf.ones_default(vec![2, 2]);
        let wrong_out = tf.ones_default(vec![2, 2, 3]);

        let mut ctx = context();
        mm_out(&mut ctx, &x, &right_y, &wrong_out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        mm_out(&mut ctx, &x, &wrong_y, &right_out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_wrong_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.ones_default(vec![10, 3]);
        let y = tf.ones_default(vec![3, 4]);
        let right_out = tf.ones_default(vec![10, 4]);
        let wrong_out = tf.ones_default(vec![7, 5]);

        let mut ctx = context();
        mm_out(&mut ctx, &x, &y, &wrong_out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        assert_tensor_eq!(
            *op_mm_out(&x, &y, &right_out),
            tf.full(vec![10, 4], 3, TensorShapeDynamism::STATIC)
        );
    }

    fn dyn_shape_data<'a>(tf: &'a TensorFactory<f32>) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.17412060499191284,
                0.34793388843536377,
                0.8187907934188843,
                0.9979893565177917,
                0.7049332857131958,
                0.4255824089050293,
            ],
        );
        let y = tf.make_default(
            vec![2, 4],
            vec![
                0.8071839213371277,
                0.13667285442352295,
                0.9002121090888977,
                0.9070476293563843,
                0.31638312339782715,
                0.3691965937614441,
                0.09420186281204224,
                0.9310881495475769,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 4],
            vec![
                0.2506277561187744,
                0.15225356817245483,
                0.18952149152755737,
                0.48189279437065125,
                0.976661741733551,
                0.480360746383667,
                0.8310978412628174,
                1.6718982458114624,
                0.703657865524292,
                0.2534688115119934,
                0.6746801733970642,
                1.0356627702713013,
            ],
        );
        (x, y, expected_result)
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        op_mm_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    fn op_mm_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_mm_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED_DynamicShapeUnbound in C++; ported and #[ignore]d.
    // [spec:et:sem:op-mm.torch.executor.native.mm-out-fn/test]
    #[test]
    #[ignore]
    fn op_mm_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected_result) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_mm_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }
}
