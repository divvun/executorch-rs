//! Literal port of kernels/portable/cpu/op_narrow_copy.cpp.

use crate::kernels::portable::cpu::util::slice_util::{
    check_narrow_copy_args, compute_slice, get_narrow_copy_out_target_size,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `Tensor::SizesType` maps to
// `tensor_impl::SizesType`, `kTensorDimensionLimit` to `K_TENSOR_DIMENSION_LIMIT`.
// `get_narrow_copy_out_target_size` is `unsafe` (raw pointer output), so its call
// is wrapped in `unsafe`.

// [spec:et:def:op-narrow-copy.torch.executor.native.narrow-copy-out-fn]
// [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn]
pub fn narrow_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    start: i64,
    length: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_narrow_copy_args(in_, dim, start, length, out),
        InvalidArgument,
        out
    );

    if dim < 0 {
        dim += in_.dim() as i64;
    }

    let mut target_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_narrow_copy_out_target_size(
            in_,
            dim,
            length,
            target_sizes.as_mut_ptr(),
            &mut target_ndim,
        );
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    if length != 0 {
        compute_slice(ctx, in_, dim, start, length, 1, out);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // PORT-NOTE: `static_cast<CTYPE>(int)` bridge for building integer literal data
    // in the REAL_TYPES_AND(Bool) factory element types used by test_dtype.
    trait FromI32Data: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32_data_num {
        ($($t:ty),*) => {$(impl FromI32Data for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32_data_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32Data for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32Data,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[i32]| -> Vec<T> { v.iter().map(|&x| T::from_i32(x)).collect() };

        let input = tf.make_default(vec![3, 4], d(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        let expected = tf.make_default(vec![2, 4], d(&[1, 2, 3, 4, 5, 6, 7, 8]));

        let out = tf.zeros_default(vec![2, 4]);
        let mut ctx = context();
        let ret = narrow_copy_out(&mut ctx, &input, 0, 0, 2, &out);

        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    // also verifies get_narrow_copy_out_target_size: out shape [2,4] = in [3,4]
    // with dim 0 replaced by length 2.
    // [spec:et:sem:slice-util.torch.executor.get-narrow-copy-out-target-size-fn/test]
    #[test]
    fn op_narrow_copy_out_test_all_dtypes_supported() {
        // ET_FORALL_REAL_TYPES_AND(Bool)
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_empty_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 0, 1]);
        let out = tf.zeros_default(vec![1, 0, 1]);
        let expect = tf.ones_default(vec![1, 0, 1]);

        let mut ctx = context();
        let ret = narrow_copy_out(&mut ctx, &input, 0, 0, 1, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);

        let ret = narrow_copy_out(&mut ctx, &input, 1, 0, 0, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);

        let ret = narrow_copy_out(&mut ctx, &input, 2, 0, 1, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_zero_length_supported() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 3]);
        let out = tf.ones_default(vec![2, 0]);
        let expect = tf.ones_default(vec![2, 0]);

        let mut ctx = context();
        let ret = narrow_copy_out(&mut ctx, &input, 1, 1, 0, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);

        let ret = narrow_copy_out(&mut ctx, &input, 1, -1, 0, &out);
        assert_tensor_eq!(*ret, out);
        assert_tensor_eq!(*ret, expect);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_zero_dim_input_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![]);
        let out = tf.ones_default(vec![]);

        // The operation shall die whatever the end is.
        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 0, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 0, 1, 1, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_invalid_start() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 3]);
        let out = tf.ones_default(vec![2, 3]);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 0, -3, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 1, 4, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    // also verifies check_narrow_copy_args: start+length > dim size (and the
    // negative-start-adjust path) is rejected.
    // [spec:et:sem:slice-util.torch.executor.check-narrow-copy-args-fn/test]
    #[test]
    fn op_narrow_copy_out_test_invalid_start_length_combination() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![2, 3]);
        let out = tf.ones_default(vec![2, 3]);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 0, 0, 3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 1, -1, 2, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_negative_length_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        // Some invalid length values.
        let invalid_lengths: [i64; 3] = [-3, -2, -1];
        for length in invalid_lengths {
            let mut ctx = context();
            narrow_copy_out(&mut ctx, &input, 0, 0, length, &out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let input = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1, 1]);

        // Some invalid dim values.
        let invalid_dims: [i64; 6] = [3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            narrow_copy_out(&mut ctx, &input, dim, 0, 1, &out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn/test]
    #[test]
    fn op_narrow_copy_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let input = tf_int.zeros_default(vec![1, 2, 2]);

        // Size is compatible to the output, but a mismatched dtype.
        let out = tf_float.ones_default(vec![1, 2, 2]);

        let mut ctx = context();
        narrow_copy_out(&mut ctx, &input, 0, 0, 1, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
