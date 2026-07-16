//! Literal port of kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_floathbf16.cpp.

use crate::kernels::portable::cpu::pattern::pattern::{AsF32, AsF64, FromF32, FromF64};
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_floating_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` — see
// unary_ufunc_realhbf16.rs for the interior-mutation rationale.

// [spec:et:def:unary-ufunc-realhbbf16-to-floathbf16.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
// [spec:et:sem:unary-ufunc-realhbbf16-to-floathbf16.torch.executor.native.internal.unary-ufunc-realhbbf16-to-floathbf16-fn]
pub fn unary_ufunc_realhbbf16_to_floathbf16<'a, 'b>(
    fn_float: fn(f32) -> f32,
    fn_double: fn(f64) -> f64,
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(ctx, tensor_is_floating_type(out), InvalidArgument, out);

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();
    let out_type = out.scalar_type();

    crate::et_switch_realhbbf16_types!(
        in_type,
        ctx,
        "unary_ufunc_realhbbf16_to_floathbf16",
        CTYPE_IN,
        {
            crate::et_switch_floathbf16_types!(
                out_type,
                ctx,
                "unary_ufunc_realhbbf16_to_floathbf16",
                CTYPE_OUT,
                {
                    apply_unary_map_fn(
                        |val_in: CTYPE_IN| -> CTYPE_OUT {
                            if CTYPE_IN::IS_DOUBLE {
                                let xi: f64 = val_in.as_f64();
                                CTYPE_OUT::from_f64(fn_double(xi))
                            } else {
                                let xi: f32 = val_in.as_f32();
                                CTYPE_OUT::from_f32(fn_float(xi))
                            }
                        },
                        in_.const_data_ptr::<CTYPE_IN>(),
                        out.mutable_data_ptr::<CTYPE_OUT>(),
                        in_.numel() as i64,
                        1,
                    );
                }
            );
        }
    );

    out
}

/// Port of kernels/test/UnaryUfuncRealHBBF16ToFloatHBF16Test.{h,cpp}.
///
/// This is the generic gtest harness shared by the op_acos/op_acosh (and other)
/// unary-ufunc test suites. The C++ `UnaryUfuncRealHBBF16ToFloatHBF16Test`
/// fixture is templated over an overridden `op_out` and `op_reference`; here the
/// harness functions take those two as parameters and each concrete op module's
/// `#[cfg(test)] mod tests` invokes them (mirroring the per-op `TEST_F` cases
/// produced by `IMPLEMENT_UNARY_UFUNC_REALHB_TO_FLOATH_TEST`).
#[cfg(test)]
pub(crate) mod test_harness {
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    // Mirrors `OperatorTest::SetUp()`'s `runtime_init()`.
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

    /// The op function under test: `torch::executor::aten::<op>_outf(context_, self, out)`.
    pub(crate) type OpOut =
        for<'a, 'b> fn(&mut KernelRuntimeContext, &Tensor, &'a Tensor<'b>) -> &'a Tensor<'b>;

    /// `static_cast<IN_CTYPE>(int)` — build the `{0,1,3,5,10,100}` (and the `2`
    /// replacement) test vector in the input element type.
    pub(crate) trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    /// `static_cast<OUT_CTYPE>(double)` — narrow the reference result.
    pub(crate) trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
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

    /// `to_f64` for the input element type, feeding `op_reference`.
    pub(crate) trait ToF64: Copy {
        fn to_f64(self) -> f64;
    }
    macro_rules! impl_to_f64_num {
        ($($t:ty),*) => {$(impl ToF64 for $t { fn to_f64(self) -> f64 { self as f64 } })*};
    }
    impl_to_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl ToF64 for Half {
        fn to_f64(self) -> f64 {
            self.to_f32() as f64
        }
    }
    impl ToF64 for BFloat16 {
        fn to_f64(self) -> f64 {
            self.to_f32() as f64
        }
    }

    fn test_floating_point_op_out<IN, OUT>(
        op_out: OpOut,
        op_reference: fn(f64) -> f64,
        out_shape: &[i32],
        dynamism: TensorShapeDynamism,
    ) where
        IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let out = tf_out.zeros(out_shape.to_vec(), dynamism);

        let mut test_vector: Vec<IN> = vec![
            IN::from_i32(0),
            IN::from_i32(1),
            IN::from_i32(3),
            IN::from_i32(5),
            IN::from_i32(10),
            IN::from_i32(100),
        ];
        let mut expected_vector: Vec<OUT> = Vec::new();
        for ii in 0..test_vector.len() {
            let mut ref_result = op_reference(test_vector[ii].to_f64());
            // Drop test cases with high magnitude results due to precision issues.
            if ref_result.abs() > 1e30 || ref_result.abs() < -1e30 {
                test_vector[ii] = IN::from_i32(2);
                ref_result = op_reference(2.0);
            }
            expected_vector.push(OUT::from_f64(ref_result));
        }

        let mut ctx = context();
        op_out(&mut ctx, &tf_in.make_default(vec![1, 6], test_vector), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);

        let expected = tf_out.make_default(vec![1, 6], expected_vector);
        if IN::VALUE == ScalarType::BFloat16 || OUT::VALUE == ScalarType::BFloat16 {
            // Raise tolerance because both we and ATen run these computations at
            // internal float32 precision rather than float64.
            let rtol = 3e-3;
            assert!(tensors_are_close(
                &out,
                &expected,
                rtol,
                Some(internal::K_DEFAULT_BFLOAT16_ATOL)
            ));
        } else if IN::VALUE == ScalarType::Half || OUT::VALUE == ScalarType::Half {
            let rtol = 1e-3;
            assert!(tensors_are_close(
                &out,
                &expected,
                rtol,
                Some(internal::K_DEFAULT_HALF_ATOL)
            ));
        } else {
            assert!(tensors_are_close(
                &out,
                &expected,
                internal::K_DEFAULT_RTOL,
                None
            ));
        }
    }

    fn test_op_invalid_output_dtype_dies<IN, OUT>(op_out: OpOut)
    where
        IN: CppTypeToScalarType + FactoryValue,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 5];

        let in_ = tf.ones_default(sizes.clone());
        let out = tf_out.zeros_default(sizes);

        // ET_EXPECT_KERNEL_FAILURE(context_, op_out(in, out))
        let mut ctx = context();
        op_out(&mut ctx, &in_, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    /// `TEST_F(TestName, HandleBoolInput)` — `test_bool_input()`.
    pub(crate) fn test_bool_input(op_out: OpOut, op_reference: fn(f64) -> f64) {
        let tf_bool = TensorFactory::<bool>::new();
        let tf_float = TensorFactory::<f32>::new();

        let sizes = vec![1, 2];

        let a = tf_bool.make_default(sizes.clone(), vec![false, true]);
        let out = tf_float.zeros_default(sizes.clone());
        let res = tf_float.make_default(
            sizes,
            vec![op_reference(0.0) as f32, op_reference(1.0) as f32],
        );

        let mut ctx = context();
        let got = op_out(&mut ctx, &a, &out);
        assert!(tensors_are_close(got, &res, internal::K_DEFAULT_RTOL, None));
    }

    /// `TEST_F(TestName, MismatchedInputShapesDies)`.
    ///
    /// PORT-NOTE: the C++ test skips when `is_aten`; the ported runtime is never
    /// ATen, so the skip never triggers and the failure path is always exercised.
    pub(crate) fn test_mismatched_input_shapes_dies(op_out: OpOut) {
        let tf = TensorFactory::<f32>::new();

        let a = tf.ones_default(vec![4]);
        let out = tf.ones_default(vec![2, 2]);

        let mut ctx = context();
        op_out(&mut ctx, &a, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ET_FORALL_REALHBF16_TYPES: Byte,Char,Short,Int,Long,Float,Double,Half,BFloat16
    // ET_FORALL_REALH_TYPES:     Byte,Char,Short,Int,Long,Float,Double,Half
    // Each `test_all_*` below is the literal expansion of the corresponding
    // FORALL loop over its IN dtype set with the fixed OUT dtype.

    pub(crate) fn test_all_real_input_half_output_static_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, Half>(op_out, r, &[1, 6], TensorShapeDynamism::STATIC);
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
        e::<BFloat16>(op_out, r);
    }

    pub(crate) fn test_all_real_input_bfloat16_output_static_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, BFloat16>(
                op_out,
                r,
                &[1, 6],
                TensorShapeDynamism::STATIC,
            );
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
        e::<BFloat16>(op_out, r);
    }

    // ET_FORALL_REALH_TYPES: Byte,Char,Short,Int,Long,Float,Double,Half
    pub(crate) fn test_all_real_input_float_output_static_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, f32>(op_out, r, &[1, 6], TensorShapeDynamism::STATIC);
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
    }

    pub(crate) fn test_all_real_input_double_output_static_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, f64>(op_out, r, &[1, 6], TensorShapeDynamism::STATIC);
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
    }

    pub(crate) fn test_all_real_input_half_output_bound_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, Half>(
                op_out,
                r,
                &[10, 10],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
        e::<BFloat16>(op_out, r);
    }

    pub(crate) fn test_all_real_input_bfloat16_output_bound_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, BFloat16>(
                op_out,
                r,
                &[10, 10],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
        e::<BFloat16>(op_out, r);
    }

    pub(crate) fn test_all_real_input_float_output_bound_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, f32>(
                op_out,
                r,
                &[10, 10],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
    }

    pub(crate) fn test_all_real_input_double_output_bound_dynamism_support(
        op_out: OpOut,
        r: fn(f64) -> f64,
    ) {
        fn e<IN>(op_out: OpOut, r: fn(f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_floating_point_op_out::<IN, f64>(
                op_out,
                r,
                &[10, 10],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );
        }
        e::<u8>(op_out, r);
        e::<i8>(op_out, r);
        e::<i16>(op_out, r);
        e::<i32>(op_out, r);
        e::<i64>(op_out, r);
        e::<f32>(op_out, r);
        e::<f64>(op_out, r);
        e::<Half>(op_out, r);
    }

    /// `test_all_real_input_float_output_unbound_dynamism_support` — C++ skips
    /// unless `is_aten`; the ported runtime is never ATen, so this is a no-op.
    pub(crate) fn test_all_real_input_float_output_unbound_dynamism_support(
        _op_out: OpOut,
        _r: fn(f64) -> f64,
    ) {
        // ET_SKIP_IF(!is_aten, ...) -> skipped in the non-ATen build.
    }

    /// `test_all_real_input_double_output_unbound_dynamism_support` — skipped in
    /// the non-ATen build (see above).
    pub(crate) fn test_all_real_input_double_output_unbound_dynamism_support(
        _op_out: OpOut,
        _r: fn(f64) -> f64,
    ) {
    }

    // ET_FORALL_INT_TYPES: Byte,Char,Short,Int,Long
    pub(crate) fn test_non_float_output_dtype_dies(op_out: OpOut) {
        test_op_invalid_output_dtype_dies::<f32, u8>(op_out);
        test_op_invalid_output_dtype_dies::<f32, i8>(op_out);
        test_op_invalid_output_dtype_dies::<f32, i16>(op_out);
        test_op_invalid_output_dtype_dies::<f32, i32>(op_out);
        test_op_invalid_output_dtype_dies::<f32, i64>(op_out);
    }
}
