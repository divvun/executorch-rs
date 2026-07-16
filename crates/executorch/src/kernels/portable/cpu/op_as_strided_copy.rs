//! Literal port of kernels/portable/cpu/op_as_strided_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    as_strided_copy, check_as_strided_copy_args,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). `ArrayRef<int64_t>`
// maps to `ArrayRef<i64>`; `optional<int64_t>` to `Option<i64>`. The C++
// `size_t offset` is carried as `i64` to match the ported `as_strided_copy`
// (which takes `offset: i64`).

// [spec:et:def:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn]
// [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn]
pub fn as_strided_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    size: ArrayRef<i64>,
    stride: ArrayRef<i64>,
    storage_offset: Option<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_as_strided_copy_args(in_, size, stride, storage_offset, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, size) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    if in_.numel() == 0 {
        return out;
    }

    let offset: i64 = if storage_offset.is_some() {
        storage_offset.unwrap()
    } else {
        0
    };

    crate::et_switch_all_types!(in_.scalar_type(), ctx, "as_strided_copy_out", CTYPE, {
        as_strided_copy::<CTYPE>(in_, size, stride, offset, out);
    });

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
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for bool {
        fn from_i64(v: i64) -> Self {
            v != 0
        }
    }

    fn test_detach_copy_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let out_sizes = vec![2, 2, 2];

        let in_ = tf.make_default(vec![3, 3], (1..=9).map(T::from_i64).collect());
        let out = tf.zeros_default(out_sizes.clone());

        let storage_offset: Option<i64> = None;
        let sizes: [i64; 3] = [2, 2, 2];
        let stride: [i64; 3] = [1, 2, 3];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(
                out_sizes.clone(),
                [1, 4, 3, 6, 2, 5, 4, 7]
                    .iter()
                    .map(|&v| T::from_i64(v))
                    .collect()
            )
        );

        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            Some(2),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(
                out_sizes,
                [3, 6, 5, 8, 4, 7, 6, 9]
                    .iter()
                    .map(|&v| T::from_i64(v))
                    .collect()
            )
        );
    }

    fn test_as_strided_copy_out_invalid_parameters<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64,
    {
        let tf = TensorFactory::<T>::new();
        let out_sizes = vec![2, 2, 2];

        let in_ = tf.ones_default(vec![3, 3]);
        let out = tf.zeros_default(out_sizes);
        let storage_offset: Option<i64> = None;
        let sizes: [i64; 3] = [2, 2, 2];
        let stride: [i64; 3] = [1, 2, 3];

        // Mismatch strides and shape should die
        let stride_short: [i64; 2] = [1, 2];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride_short.as_ptr(), 2),
            storage_offset,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Negative strides should die
        let stride_negative: [i64; 3] = [1, 2, -1];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride_negative.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Mismatch output tensor shape and size should die
        let size_invalid: [i64; 3] = [2, 2, 1];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(size_invalid.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Invalid storage offset should die
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            Some(-1),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);

        // Out of bound storage access of `in` should die
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            Some(3),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: the C++ `test_detach_copy_out<ScalarType::Bool>` specialization is
    // defined but never invoked (`AllScalarInputOutputSupport` dispatches over
    // `ET_FORALL_INT_TYPES`, which excludes Bool/Float). Ported for fidelity, not
    // called.
    #[allow(dead_code)]
    fn test_detach_copy_out_bool() {
        let tf = TensorFactory::<bool>::new();
        let out_sizes = vec![2, 2, 2];
        let in_ = tf.make_default(
            vec![3, 3],
            vec![false, true, false, true, false, true, false, true, false],
        );
        let out = tf.zeros_default(out_sizes.clone());

        let sizes: [i64; 3] = [2, 2, 2];
        let stride: [i64; 3] = [1, 2, 3];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            Some(2),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(
                out_sizes,
                vec![false, true, false, true, true, false, true, false]
            )
        );
    }

    // PORT-NOTE: the C++ `test_detach_copy_out<ScalarType::Float>` specialization is
    // defined but never invoked (see above). Ported for fidelity, not called.
    #[allow(dead_code)]
    fn test_detach_copy_out_float() {
        use crate::assert_tensor_close;
        let tf = TensorFactory::<f32>::new();
        let out_sizes = vec![2, 2, 2];

        let in_ = tf.make_default(
            vec![3, 3],
            vec![
                3.14,
                2.33,
                42.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                -3.14,
                -2.33,
                -42.0,
            ],
        );
        let out = tf.zeros_default(out_sizes.clone());

        let sizes: [i64; 3] = [2, 2, 2];
        let stride: [i64; 3] = [1, 2, 3];
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            Some(2),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(
                out_sizes,
                vec![
                    42.0,
                    f32::NAN,
                    f32::NEG_INFINITY,
                    2.33,
                    f32::INFINITY,
                    -3.14,
                    f32::NAN,
                    -42.0,
                ]
            )
        );
    }

    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    // Also drives as_strided_copy<CTYPE>: strides [1,2,3] over sizes [2,2,2]
    // gather distinct expected values, pinning the recursive strided copy.
    // [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-fn/test]
    // also verifies check_as_strided_copy_args (dtype/stride/offset/bounds gate,
    // offset=2) and as_strided_copy_compute_storage_nbytes (nonzero storage size
    // from sizes {2,2,2}/strides {1,2,3} feeds the bounds check)
    // [spec:et:sem:copy-ops-util.torch.executor.check-as-strided-copy-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_all_scalar_input_output_support() {
        test_detach_copy_out::<u8>();
        test_detach_copy_out::<i8>();
        test_detach_copy_out::<i16>();
        test_detach_copy_out::<i32>();
        test_detach_copy_out::<i64>();
    }

    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_invalid_parameters_dies() {
        test_as_strided_copy_out_invalid_parameters::<u8>();
        test_as_strided_copy_out_invalid_parameters::<i8>();
        test_as_strided_copy_out_invalid_parameters::<i16>();
        test_as_strided_copy_out_invalid_parameters::<i32>();
        test_as_strided_copy_out_invalid_parameters::<i64>();
        test_as_strided_copy_out_invalid_parameters::<f32>();
        test_as_strided_copy_out_invalid_parameters::<f64>();
        test_as_strided_copy_out_invalid_parameters::<bool>();
    }

    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_mismatched_input_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();
        let out_sizes = vec![2, 2, 2];

        let in_ = tf_byte.make_default(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let out = tf_char.zeros_default(out_sizes);
        let storage_offset: Option<i64> = None;
        let sizes: [i64; 3] = [2, 2, 2];
        let stride: [i64; 3] = [1, 2, 3];

        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &in_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), 3),
            ArrayRef::from_raw_parts(stride.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
            ],
        );
        let expected = tf.make_default(
            vec![2, 2, 2],
            vec![
                0.49625658988952637,
                0.13203048706054688,
                0.08847743272781372,
                0.6340786814689636,
                0.7682217955589294,
                0.30742281675338745,
                0.13203048706054688,
                0.4900934100151062,
            ],
        );

        let sizev: [i64; 3] = [2, 2, 2];
        let stridev: [i64; 3] = [1, 2, 3];
        let storage_offset: Option<i64> = None;

        let out = tf.zeros(vec![2, 2, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &x,
            ArrayRef::from_raw_parts(sizev.as_ptr(), 3),
            ArrayRef::from_raw_parts(stridev.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
            ],
        );
        let expected = tf.make_default(
            vec![2, 2, 2],
            vec![
                0.49625658988952637,
                0.13203048706054688,
                0.08847743272781372,
                0.6340786814689636,
                0.7682217955589294,
                0.30742281675338745,
                0.13203048706054688,
                0.4900934100151062,
            ],
        );

        let sizev: [i64; 3] = [2, 2, 2];
        let stridev: [i64; 3] = [1, 2, 3];
        let storage_offset: Option<i64> = None;

        let out = tf.zeros(vec![5, 5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        as_strided_copy_out(
            &mut ctx,
            &x,
            ArrayRef::from_raw_parts(sizev.as_ptr(), 3),
            ArrayRef::from_raw_parts(stridev.as_ptr(), 3),
            storage_offset,
            &out,
        );
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: `DynamicShapeUnbound` is `ET_SKIP_IF(!output_resize, ...)`; the
    // portable build lacks output resize for unbound, so it is skipped. Ported as a
    // no-op skip.
    // [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn/test]
    #[test]
    fn op_as_strided_copy_out_test_dynamic_shape_unbound() {
        // Dynamic shape unbound not supported in the portable build; skipped.
    }
}
