//! Literal port of kernels/portable/cpu/pattern/logical_op.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::apply_bitensor_elementwise_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` — the ported
// `Tensor` handle mutates through an interior `*mut TensorImpl`, and all the
// helpers here take `&Tensor`.
//
// PORT-NOTE (cross-module): the compile-time `op_name` template parameter of the
// C++ `logical_tensor_out<op_name>` is dropped — the ported
// `apply_bitensor_elementwise_fn` takes no op-name argument (its dtype gating no
// longer keys on the op name). `support_noncontiguous_tensors` is fixed `true`,
// matching the C++ pattern's `apply_bitensor_elementwise_fn` invocation. The
// compute closure returns `bool` (the fixed `CTYPE_COMPUTE` here), which the
// framework stores into `out`.

// [spec:et:def:logical-op.torch.executor.native.internal.logical-tensor-out-fn]
// [spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn]
pub fn logical_tensor_out<'a, 'b>(
    fn_: fn(bool, bool) -> bool,
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    apply_bitensor_elementwise_fn::<bool, _>(
        // TODO: rewrite this to be vectorization-capable.
        |vals: &[bool]| fn_(vals[0], vals[1]),
        ctx,
        a,
        SupportedTensorDtypes::REALHBBF16,
        b,
        SupportedTensorDtypes::REALHBBF16,
        out,
        SupportedTensorDtypes::REALHBBF16,
        true,
    );

    out
}

/// Port of kernels/test/BinaryLogicalOpTest.{h,cpp}.
///
/// The C++ `BinaryLogicalOpTest` fixture is templated over an overridden `op_out`
/// and `op_reference`; here the harness function takes those two as parameters and
/// each concrete op module's `#[cfg(test)] mod tests` invokes it (mirroring the
/// per-op `TEST_F(SimpleTestAllTypes)` produced by `IMPLEMENT_BINARY_LOGICAL_OP_TEST`).
#[cfg(test)]
pub(crate) mod test_harness {
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    // Mirrors `OperatorTest::SetUp()`'s `runtime_init()`.
    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    /// The op function under test: `torch::executor::aten::<op>_outf(context_, lhs, rhs, out)`.
    pub(crate) type OpOut = for<'a, 'b> fn(
        &mut KernelRuntimeContext,
        &Tensor,
        &Tensor,
        &'a Tensor<'b>,
    ) -> &'a Tensor<'b>;

    /// `static_cast<CTYPE>(int)` for the input factory element types.
    pub(crate) trait FromI32: Copy {
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

    /// The implicit `CTYPE -> double` conversion at the `op_reference(...)` call.
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

    /// `static_cast<OUT>(double)` — narrow the reference result to the output type.
    pub(crate) trait FromF64Elem: Copy {
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

    fn test_op_out<IN, IN2, OUT>(op_out: OpOut, op_reference: fn(f64, f64) -> f64)
    where
        IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        IN2: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_in2 = TensorFactory::<IN2>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let out = tf_out.zeros_default(vec![1, 4]);

        let test_vector1: Vec<IN> = vec![
            IN::from_i32(0),
            IN::from_i32(-1),
            IN::from_i32(0),
            IN::from_i32(31),
        ];
        let test_vector2: Vec<IN2> = vec![
            IN2::from_i32(0),
            IN2::from_i32(0),
            IN2::from_i32(15),
            IN2::from_i32(12),
        ];

        let mut expected_vector: Vec<OUT> = Vec::new();
        for ii in 0..test_vector1.len() {
            expected_vector.push(OUT::from_f64(op_reference(
                test_vector1[ii].to_f64(),
                test_vector2[ii].to_f64(),
            )));
        }

        let mut ctx = context();
        op_out(
            &mut ctx,
            &tf_in.make_default(vec![1, 4], test_vector1),
            &tf_in2.make_default(vec![1, 4], test_vector2),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf_out.make_default(vec![1, 4], expected_vector),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    /// `test_all_dtypes()` — three `ET_FORALL_REALHBF16_TYPES` sweeps: input1 over
    /// all dtypes (in2/out fixed Double), input2 over all dtypes (in1/out fixed
    /// Double), and output over all dtypes (in1/in2 fixed Double).
    pub(crate) fn test_all_dtypes(op_out: OpOut, op_reference: fn(f64, f64) -> f64) {
        // input1 varies, in2 = Double, out = Double
        fn e1<IN>(op_out: OpOut, r: fn(f64, f64) -> f64)
        where
            IN: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_op_out::<IN, f64, f64>(op_out, r);
        }
        e1::<u8>(op_out, op_reference);
        e1::<i8>(op_out, op_reference);
        e1::<i16>(op_out, op_reference);
        e1::<i32>(op_out, op_reference);
        e1::<i64>(op_out, op_reference);
        e1::<f32>(op_out, op_reference);
        e1::<f64>(op_out, op_reference);
        e1::<Half>(op_out, op_reference);
        e1::<BFloat16>(op_out, op_reference);

        // input2 varies, in1 = Double, out = Double
        fn e2<IN2>(op_out: OpOut, r: fn(f64, f64) -> f64)
        where
            IN2: CppTypeToScalarType + FactoryValue + FromI32 + ToF64,
        {
            test_op_out::<f64, IN2, f64>(op_out, r);
        }
        e2::<u8>(op_out, op_reference);
        e2::<i8>(op_out, op_reference);
        e2::<i16>(op_out, op_reference);
        e2::<i32>(op_out, op_reference);
        e2::<i64>(op_out, op_reference);
        e2::<f32>(op_out, op_reference);
        e2::<f64>(op_out, op_reference);
        e2::<Half>(op_out, op_reference);
        e2::<BFloat16>(op_out, op_reference);

        // output varies, in1 = Double, in2 = Double
        fn e3<OUT>(op_out: OpOut, r: fn(f64, f64) -> f64)
        where
            OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
        {
            test_op_out::<f64, f64, OUT>(op_out, r);
        }
        e3::<u8>(op_out, op_reference);
        e3::<i8>(op_out, op_reference);
        e3::<i16>(op_out, op_reference);
        e3::<i32>(op_out, op_reference);
        e3::<i64>(op_out, op_reference);
        e3::<f32>(op_out, op_reference);
        e3::<f64>(op_out, op_reference);
        e3::<Half>(op_out, op_reference);
        e3::<BFloat16>(op_out, op_reference);
    }
}
