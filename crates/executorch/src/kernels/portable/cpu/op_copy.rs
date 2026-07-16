//! Literal port of kernels/portable/cpu/op_copy.cpp.

use crate::kernels::portable::cpu::util::broadcast_indexes_range::sizes_match_ignoring_leading_1s;
use crate::kernels::portable::cpu::util::broadcast_util::tensor_is_broadcastable_to_tensors;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_bitensor_elementwise_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// copy.out(const Tensor& in, const Tensor& src, bool non_blocking, Tensor(a!)
// out) -> Tensor(a!), see caffe2/aten/src/ATen/native/Copy.cpp
// TODO: We actually shouldn't see this op with the proper functionalization,
// and this op needs to be deleted
// [spec:et:def:op-copy.torch.executor.native.copy-out-fn]
// [spec:et:sem:op-copy.torch.executor.native.copy-out-fn]
pub fn copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    src: &Tensor,
    non_blocking: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Right now we only support blocking data transfer
    crate::et_kernel_check!(ctx, non_blocking == false, InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_broadcastable_to_tensors(src, in_),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let op_name = "copy.out";

    // Use direct copy fast path if broadcast is not needed and tensors are
    // non-empty
    if sizes_match_ignoring_leading_1s(out.sizes(), src.sizes())
        && src.numel() > 0
        && out.nbytes() >= src.nbytes()
        && tensors_have_same_dtype2(src, out)
    {
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.const_data_ptr::<u8>(),
                out.mutable_data_ptr::<u8>(),
                src.nbytes(),
            );
        }
    } else {
        crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
            apply_bitensor_elementwise_fn::<CTYPE, _>(
                |vals: &[CTYPE]| {
                    let _ = vals[0];
                    let val_src = vals[1];
                    val_src
                },
                ctx,
                in_,
                SupportedTensorDtypes::REALHBBF16,
                src,
                SupportedTensorDtypes::REALHBBF16,
                out,
                SupportedTensorDtypes::REALHBBF16,
                /*support_noncontiguous*/ false,
            );
        });
    }

    out
}

// [spec:et:def:op-copy.torch.executor.native.copy-fn]
// [spec:et:sem:op-copy.torch.executor.native.copy-fn]
pub fn copy_<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &'a Tensor<'b>,
    src: &Tensor,
    non_blocking: bool,
) -> &'a Tensor<'b> {
    // Right now we only support blocking data transfer
    crate::et_kernel_check!(ctx, non_blocking == false, InvalidArgument, in_);

    crate::et_kernel_check!(
        ctx,
        tensor_is_broadcastable_to_tensors(src, in_),
        InvalidArgument,
        in_
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, src),
        InvalidArgument,
        in_
    );

    let op_name = "copy_";

    // Use direct copy fast path if broadcast is not needed and tensors are
    // non-empty
    if sizes_match_ignoring_leading_1s(in_.sizes(), src.sizes())
        && src.numel() > 0
        && in_.nbytes() >= src.nbytes()
        && tensors_have_same_dtype2(src, in_)
    {
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.const_data_ptr::<u8>(),
                in_.mutable_data_ptr::<u8>(),
                src.nbytes(),
            );
        }
    } else {
        crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
            apply_bitensor_elementwise_fn::<CTYPE, _>(
                |vals: &[CTYPE]| {
                    let _ = vals[0];
                    let val_src = vals[1];
                    val_src
                },
                ctx,
                in_,
                SupportedTensorDtypes::REALHBBF16,
                src,
                SupportedTensorDtypes::REALHBBF16,
                in_,
                SupportedTensorDtypes::REALHBBF16,
                /*support_noncontiguous*/ false,
            );
        });
    }

    in_
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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_copy_out<'a, 'b>(
        self_: &Tensor,
        src: &Tensor,
        non_blocking: bool,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        copy_out(&mut ctx, self_, src, non_blocking, out)
    }

    fn op_copy_<'a, 'b>(self_: &'a Tensor<'b>, src: &Tensor, non_blocking: bool) -> &'a Tensor<'b> {
        let mut ctx = context();
        copy_(&mut ctx, self_, src, non_blocking)
    }

    // PORT-NOTE: local `from_i32` bridge for the REALHBF16 element types used by
    // the copy dtype suites.
    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_num {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_num!(u8, i8, i16, i32, i64, f32, f64);
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

    // test if copy.out works well under all kinds of legal input type.
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let data: Vec<T> = [2, 3, 2, 4, 1, 5, 1, 6]
            .iter()
            .map(|&v| T::from_i32(v))
            .collect();
        let self_ = tf.make_default(vec![2, 4], data.clone());
        let src = tf.make_default(vec![2, 4], data);
        let non_blocking = false;
        let out_nullopt = tf.zeros_default(vec![2, 4]);
        let out_contiguous = tf.zeros_default(vec![2, 4]);

        // we only support contiguous memory, the memory type shall be either
        // nullopt or MemoryFormat::Contiguous.
        let out_nullopt_ret = op_copy_out(&self_, &src, non_blocking, &out_nullopt);

        // The original tensor a should share same value with the out variable and
        // return variable of copy function
        assert_tensor_eq!(src, out_nullopt);
        assert_tensor_eq!(src, out_nullopt_ret);

        let out_contiguous_ret = op_copy_out(&self_, &src, non_blocking, &out_contiguous);
        assert_tensor_eq!(src, out_contiguous);
        assert_tensor_eq!(src, out_contiguous_ret);
    }

    fn test_empty_input<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();
        let self_ = tf.make_default(vec![3, 0, 1, 2], vec![]);
        let src = tf.make_default(vec![3, 0, 1, 2], vec![]);
        let non_blocking = false;
        let out = tf.zeros_default(vec![3, 0, 1, 2]);
        op_copy_out(&self_, &src, non_blocking, &out);
        // check a and out share same value, but are different object
        assert_tensor_eq!(src, out);
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.make_default(vec![3, 4], vec![4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6]);
        let src = tf.make_default(vec![3, 4], vec![6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4]);
        let expected = tf.make_default(vec![3, 4], vec![6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4]);
        let out = tf.zeros(out_shape, dynamism);

        op_copy_out(&self_, &src, false, &out);
        assert_tensor_eq!(out, expected);
    }

    // regular test for copy.out
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_all_real_dtypes_supported() {
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

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_empty_input_supported() {
        test_empty_input::<u8>();
        test_empty_input::<i8>();
        test_empty_input::<i16>();
        test_empty_input::<i32>();
        test_empty_input::<i64>();
        test_empty_input::<f32>();
        test_empty_input::<f64>();
        test_empty_input::<Half>();
        test_empty_input::<BFloat16>();
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_broad_cast_src_supported() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let src = tf.make_default(vec![1, 2], vec![3, 3]);
        let non_blocking = false;
        let out = tf.zeros_default(vec![2, 2]);
        op_copy_out(&self_, &src, non_blocking, &out);
        let out_expected = tf.make_default(vec![2, 2], vec![3, 3, 3, 3]);
        assert_tensor_eq!(out, out_expected);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_broad_cast_src_missing_dim_supported() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let src = tf.make_default(vec![1, 2], vec![3, 3]);
        let non_blocking = false;
        let out = tf.zeros_default(vec![2, 2]);
        op_copy_out(&self_, &src, non_blocking, &out);
        let out_expected = tf.make_default(vec![2, 2], vec![3, 3, 3, 3]);
        assert_tensor_eq!(out, out_expected);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_broad_cast_selfc_supported_die() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![1, 2], vec![3, 3]);
        let src = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let non_blocking = false;
        let out = tf.zeros_default(vec![2, 2]);
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_mismatch_self_src_type_supported() {
        let tf_self = TensorFactory::<i32>::new();
        let tf_src = TensorFactory::<f32>::new();
        let self_ = tf_self.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let src = tf_src.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf_src.zeros_default(vec![3, 0, 1, 2]);
        let non_blocking = false;
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ guards this under `#ifndef USE_ATEN_LIB`; the ported runtime
    // is never ATen so the test runs.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_resize_out_supported() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let src = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros(vec![4, 2, 2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let non_blocking = false;
        op_copy_out(&self_, &src, non_blocking, &out);
        let out_expected = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        assert_tensor_eq!(out, out_expected);
    }

    // PORT-NOTE: C++ guards this under `#ifndef USE_ATEN_LIB`; the ported runtime
    // is never ATen so the test runs.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_resize_out_die() {
        let tf_self = TensorFactory::<i32>::new();
        let tf_src = TensorFactory::<f32>::new();
        let self_ = tf_self.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let src = tf_src.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf_src.zeros(vec![3, 2, 0], TensorShapeDynamism::DYNAMIC_BOUND);
        let non_blocking = false;
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: the ported runtime is never ATen,
    // so the failure path runs.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let self_ = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let src = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let non_blocking = false;
        let out = tf.zeros_default(vec![3, 2, 1, 1]);
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_mismatched_src_out_types_die() {
        let tf_in = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let self_ = tf_in.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let src = tf_in.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let non_blocking = false;
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // Only contiguous memory is supported, the memory type other than nullopt or
    // MemoryFormat::Contiguous should not be allowed.
    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure runs.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_blocking_die() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let self_ = tf_in.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let src = tf_in.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let non_blocking = true;
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);
        let mut ctx = context();
        copy_out(&mut ctx, &self_, &src, non_blocking, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    fn op_copy_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: portable's `output_resize`
    // SupportedFeature is false, so this test is skipped in the portable build.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_copy_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: skipped in portable build.
    // [spec:et:sem:op-copy.torch.executor.native.copy-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_copy_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // OpCopyInplaceTest
    // [spec:et:sem:op-copy.torch.executor.native.copy-fn/test]
    #[test]
    fn op_copy_inplace_test_smoke_test() {
        let tf = TensorFactory::<i32>::new();
        let in_ = tf.zeros_default(vec![2, 2]);
        let src = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let non_blocking = false;
        op_copy_(&in_, &src, non_blocking);
        let expected = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        assert_tensor_eq!(in_, expected);
    }

    // [spec:et:sem:op-copy.torch.executor.native.copy-fn/test]
    #[test]
    fn op_copy_inplace_test_broad_cast_src_supported() {
        let tf = TensorFactory::<i32>::new();
        let in_ = tf.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let src = tf.make_default(vec![1, 2], vec![3, 3]);
        let non_blocking = false;
        op_copy_(&in_, &src, non_blocking);
        let expected = tf.make_default(vec![2, 2], vec![3, 3, 3, 3]);
        assert_tensor_eq!(in_, expected);
    }
}
