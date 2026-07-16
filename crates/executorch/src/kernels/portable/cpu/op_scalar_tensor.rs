//! Literal port of kernels/portable/cpu/op_scalar_tensor.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped. The C++
// `resize_tensor(out, {})` passes an empty `initializer_list<SizesType>`, ported
// as an empty `ArrayRef<TensorSizesType>` (0-dim, 1-element tensor).

// [spec:et:def:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn]
// [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn]
#[executorch_macros::et_kernel("aten::scalar_tensor.out")]
pub fn scalar_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    s: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::<TensorSizesType>::new()) == Error::Ok,
        InvalidArgument,
        out
    );

    let out_type: ScalarType = out.scalar_type();

    let name = "scalar_tensor.out";

    crate::et_switch_realhbbf16_types!(out_type, ctx, name, CTYPE, {
        let opt_val_casted = check_overflow_scalar_cast::<CTYPE>(s);
        crate::et_kernel_check!(ctx, opt_val_casted.is_some(), InvalidArgument, out);
        unsafe {
            *out.mutable_data_ptr::<CTYPE>().add(0) = opt_val_casted.unwrap();
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_scalar_tensor_out<'a, 'b>(s: &Scalar, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        scalar_tensor_out(&mut ctx, s, out)
    }

    // Bridges an integer test literal (CTYPE `value` in the C++ template) to both
    // a factory element value (C++ implicit conversion to CTYPE) and the Scalar
    // the C++ implicitly constructs from that CTYPE value.
    trait ScalarTensorElem: CppTypeToScalarType + FactoryValue {
        fn from_i64(v: i64) -> Self;
        fn to_scalar(self) -> Scalar;
    }
    macro_rules! impl_ste_int {
        ($($t:ty),*) => {$(impl ScalarTensorElem for $t {
            fn from_i64(v: i64) -> Self { v as $t }
            fn to_scalar(self) -> Scalar { Scalar::from_i64(self as i64) }
        })*};
    }
    impl_ste_int!(u8, i8, i16, i32, i64);
    macro_rules! impl_ste_float {
        ($($t:ty),*) => {$(impl ScalarTensorElem for $t {
            fn from_i64(v: i64) -> Self { v as $t }
            fn to_scalar(self) -> Scalar { Scalar::from_double(self as f64) }
        })*};
    }
    impl_ste_float!(f32, f64);
    impl ScalarTensorElem for bool {
        fn from_i64(v: i64) -> Self {
            v != 0
        }
        fn to_scalar(self) -> Scalar {
            Scalar::from_bool(self)
        }
    }
    impl ScalarTensorElem for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
        fn to_scalar(self) -> Scalar {
            Scalar::from_double(self.to_f32() as f64)
        }
    }
    impl ScalarTensorElem for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn to_scalar(self) -> Scalar {
            Scalar::from_double(self.to_f32() as f64)
        }
    }

    // test_scalar_tensor_out_0d<CTYPE, DTYPE>(value)
    fn test_scalar_tensor_out_0d<T: ScalarTensorElem>(value: T) {
        let tf = TensorFactory::<T>::new();

        let sizes: Vec<i32> = vec![];
        let expected = tf.make_default(sizes.clone(), vec![value]);

        let out = tf.ones_default(sizes);
        op_scalar_tensor_out(&value.to_scalar(), &out);

        assert_tensor_eq!(out, expected);
    }

    // test_scalar_tensor_out_1d<CTYPE, DTYPE>(value)
    fn test_scalar_tensor_out_1d<T: ScalarTensorElem>(value: T) {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![1];
        let out = tf.ones_default(sizes);

        let mut ctx = context();
        scalar_tensor_out(&mut ctx, &value.to_scalar(), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // test_scalar_tensor_out_2d<CTYPE, DTYPE>(value)
    fn test_scalar_tensor_out_2d<T: ScalarTensorElem>(value: T) {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![1, 1];
        let out = tf.ones_default(sizes);

        let mut ctx = context();
        scalar_tensor_out(&mut ctx, &value.to_scalar(), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // test_scalar_tensor_out_3d<CTYPE, DTYPE>(value)
    fn test_scalar_tensor_out_3d<T: ScalarTensorElem>(value: T) {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![1, 1, 1];
        let out = tf.ones_default(sizes);

        let mut ctx = context();
        scalar_tensor_out(&mut ctx, &value.to_scalar(), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // expect_bad_scalar_value_dies<DTYPE>(bad_value)
    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let out = tf.zeros_default(vec![]);

        let mut ctx = context();
        scalar_tensor_out(&mut ctx, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // GENERATE_TEST_0D(ctype, dtype) over ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16)
    fn generate_test_0d<T: ScalarTensorElem>() {
        test_scalar_tensor_out_0d::<T>(T::from_i64(4));
        test_scalar_tensor_out_0d::<T>(T::from_i64(8));
        test_scalar_tensor_out_0d::<T>(T::from_i64(9));
    }

    // GENERATE_TEST(ctype, dtype) over ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16)
    fn generate_test<T: ScalarTensorElem>() {
        test_scalar_tensor_out_1d::<T>(T::from_i64(2));
        test_scalar_tensor_out_2d::<T>(T::from_i64(2));
        test_scalar_tensor_out_3d::<T>(T::from_i64(2));
        test_scalar_tensor_out_1d::<T>(T::from_i64(4));
        test_scalar_tensor_out_2d::<T>(T::from_i64(4));
        test_scalar_tensor_out_3d::<T>(T::from_i64(4));
        test_scalar_tensor_out_1d::<T>(T::from_i64(7));
        test_scalar_tensor_out_2d::<T>(T::from_i64(7));
        test_scalar_tensor_out_3d::<T>(T::from_i64(7));
    }

    // ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, GENERATE_TEST_0D)
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_byte_tensors_dim0() {
        generate_test_0d::<u8>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_char_tensors_dim0() {
        generate_test_0d::<i8>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_short_tensors_dim0() {
        generate_test_0d::<i16>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_int_tensors_dim0() {
        generate_test_0d::<i32>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_long_tensors_dim0() {
        generate_test_0d::<i64>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_float_tensors_dim0() {
        generate_test_0d::<f32>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_double_tensors_dim0() {
        generate_test_0d::<f64>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_half_tensors_dim0() {
        generate_test_0d::<Half>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_bool_tensors_dim0() {
        generate_test_0d::<bool>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_b_float16_tensors_dim0() {
        generate_test_0d::<BFloat16>();
    }

    // ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, GENERATE_TEST)
    // PORT-NOTE: C++ guards each with ET_SKIP_IF(is_aten); ET-mode port runs them.
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_byte_tensors() {
        generate_test::<u8>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_char_tensors() {
        generate_test::<i8>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_short_tensors() {
        generate_test::<i16>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_int_tensors() {
        generate_test::<i32>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_long_tensors() {
        generate_test::<i64>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_float_tensors() {
        generate_test::<f32>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_double_tensors() {
        generate_test::<f64>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_half_tensors() {
        generate_test::<Half>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_bool_tensors() {
        generate_test::<bool>();
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_b_float16_tensors() {
        generate_test::<BFloat16>();
    }

    // PORT-NOTE: C++ guards with ET_SKIP_IF(is_aten); ET-mode port runs it.
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_invalid_out_shape_fails() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![1, 2, 1];

        let out = tf.ones_default(sizes);
        let mut ctx = context();
        scalar_tensor_out(&mut ctx, &Scalar::from_i64(7), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_half_support() {
        let tf = TensorFactory::<Half>::new();
        let out = tf.zeros_default(vec![]);

        op_scalar_tensor_out(&Scalar::from_bool(false), &out);
        assert_tensor_close!(out, tf.make_default(vec![], vec![Half::from_f32(0.0)]));

        op_scalar_tensor_out(&Scalar::from_bool(true), &out);
        assert_tensor_close!(out, tf.make_default(vec![], vec![Half::from_f32(1.0)]));

        op_scalar_tensor_out(&Scalar::from_i64(7), &out);
        assert_tensor_close!(out, tf.make_default(vec![], vec![Half::from_f32(7.0)]));

        op_scalar_tensor_out(&Scalar::from_double(2.5), &out);
        assert_tensor_close!(out, tf.make_default(vec![], vec![Half::from_f32(2.5)]));

        op_scalar_tensor_out(&Scalar::from_double(f64::INFINITY), &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![], vec![Half::from_f32(f32::INFINITY)])
        );
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpScalarTensorOutTest)
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_char_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_short_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_float_tensor_too_small_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn/test]
    #[test]
    fn op_scalar_tensor_out_test_float_tensor_too_large_scalar_dies() {
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
