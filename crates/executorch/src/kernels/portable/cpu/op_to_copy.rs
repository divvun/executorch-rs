//! Literal port of kernels/portable/cpu/op_to_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::check_to_copy_args;
use crate::kernels::portable::cpu::util::dtype_util::internal::StaticCast;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_options::MemoryFormat;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `static_cast<OUT_CTYPE>(self_data[i])` cast over the
// heterogeneous REALHBBF16 set uses the dtype_util `StaticCast` trait (as
// op_cat / other dtype-cast ports do), reproducing the per-pair `static_cast`.

// [spec:et:def:op-to-copy.torch.executor.native.to-impl-fn]
// [spec:et:sem:op-to-copy.torch.executor.native.to-impl-fn]
fn _to_impl<SELF_CTYPE, OUT_CTYPE>(self_: &Tensor, out: &Tensor)
where
    OUT_CTYPE: StaticCast<SELF_CTYPE>,
    SELF_CTYPE: Copy,
{
    let self_data: *mut SELF_CTYPE = self_.mutable_data_ptr::<SELF_CTYPE>();
    let out_data: *mut OUT_CTYPE = out.mutable_data_ptr::<OUT_CTYPE>();

    for i in 0..self_.numel() {
        unsafe {
            *out_data.offset(i as isize) =
                <OUT_CTYPE as StaticCast<SELF_CTYPE>>::static_cast(*self_data.offset(i as isize));
        }
    }
}

// to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat?
// memory_format=None, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-to-copy.torch.executor.native.to-copy-out-fn]
// [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn]
#[executorch_macros::et_kernel("aten::_to_copy.out")]
pub fn to_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    non_blocking: bool,
    memory_format: Option<MemoryFormat>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_to_copy_args(self_, non_blocking, memory_format, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(self_),
        InvalidArgument,
        out
    );

    crate::et_switch_realhbbf16_types!(self_.scalar_type(), ctx, "to_copy", CTYPE_IN, {
        crate::et_switch_realhbbf16_types!(out.scalar_type(), ctx, "to_copy", CTYPE_OUT, {
            _to_impl::<CTYPE_IN, CTYPE_OUT>(self_, out);
        });
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::tensor::tensor_ptr::NumericCast;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // `static_cast<T>(f64)` for the element ctypes (mirrors `vector_type_cast`).
    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64 for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    struct ToTestCase {
        sizes: Vec<i32>,
        data_in: Vec<f64>,
    }

    // Mirrors `test_runner_static_cast<IN, IN_DTYPE, OUT, OUT_DTYPE>`. `data_out`
    // is `static_cast<OUT>(static_cast<IN>(data_in))`.
    fn test_runner_static_cast<IN, OUT>(test_cases: &[ToTestCase])
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64 + NumericCast<OUT>,
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        for test_case in test_cases {
            let data_in: Vec<IN> = test_case.data_in.iter().map(|&x| IN::from_f64(x)).collect();
            let data_out: Vec<OUT> = data_in.iter().map(|&x| x.numeric_cast()).collect();

            let input = tf_in.make_default(test_case.sizes.clone(), data_in);
            let output = tf_out.zeros_like(&input, TensorShapeDynamism::STATIC);

            let mut ctx = context();
            let ret = to_copy_out(
                &mut ctx,
                &input,
                false,
                Some(MemoryFormat::Contiguous),
                &output,
            );

            let expected = tf_out.make_default(test_case.sizes.clone(), data_out);

            assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
            assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
        }
    }

    fn test_runner_to_bool<IN>(test_case: Vec<f64>, data_out: Vec<bool>)
    where
        IN: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<bool>::new();

        let data_in: Vec<IN> = test_case.iter().map(|&x| IN::from_f64(x)).collect();

        let input = tf_in.make_default(vec![test_case.len() as i32], data_in);
        let output = tf_out.zeros_like(&input, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        let ret = to_copy_out(
            &mut ctx,
            &input,
            false,
            Some(MemoryFormat::Contiguous),
            &output,
        );

        let expected = tf_out.make_default(vec![data_out.len() as i32], data_out);

        assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    fn test_runner_from_bool<OUT>(test_case: Vec<bool>, out: Vec<f64>)
    where
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_in = TensorFactory::<bool>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let data_out: Vec<OUT> = out.iter().map(|&x| OUT::from_f64(x)).collect();

        let input = tf_in.make_default(vec![test_case.len() as i32], test_case);
        let output = tf_out.zeros_like(&input, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        let ret = to_copy_out(
            &mut ctx,
            &input,
            false,
            Some(MemoryFormat::Contiguous),
            &output,
        );

        let expected = tf_out.make_default(vec![data_out.len() as i32], data_out);

        assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );

        let non_blocking = false;
        let memory_format: Option<MemoryFormat> = None;

        let out = tf.zeros(out_shape, dynamism);
        let mut ctx = context();
        to_copy_out(&mut ctx, &x, non_blocking, memory_format, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    fn all_dtypes_cases() -> Vec<ToTestCase> {
        vec![
            ToTestCase {
                sizes: vec![2, 4],
                data_in: vec![2.11, 3.2, 2.3, 4.0, 1.1, 5.2, 1.1, 6.3],
            },
            ToTestCase {
                sizes: vec![3, 4, 0, 5],
                data_in: vec![],
            },
            ToTestCase {
                sizes: vec![],
                data_in: vec![10.0],
            },
        ]
    }

    // ET_FORALL_REALHBF16_TYPES_WITH2: enumerate the OUT dtype for a fixed IN.
    fn static_cast_enumerate_out<IN>(cases: &[ToTestCase])
    where
        IN: CppTypeToScalarType
            + FactoryValue
            + FromF64
            + NumericCast<u8>
            + NumericCast<i8>
            + NumericCast<i16>
            + NumericCast<i32>
            + NumericCast<i64>
            + NumericCast<f32>
            + NumericCast<f64>
            + NumericCast<Half>
            + NumericCast<BFloat16>,
    {
        test_runner_static_cast::<IN, u8>(cases);
        test_runner_static_cast::<IN, i8>(cases);
        test_runner_static_cast::<IN, i16>(cases);
        test_runner_static_cast::<IN, i32>(cases);
        test_runner_static_cast::<IN, i64>(cases);
        test_runner_static_cast::<IN, f32>(cases);
        test_runner_static_cast::<IN, f64>(cases);
        test_runner_static_cast::<IN, Half>(cases);
        test_runner_static_cast::<IN, BFloat16>(cases);
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    // also verifies check_to_copy_args (accepts non_blocking==false and
    // memory_format==Contiguous, which the runner passes)
    // [spec:et:sem:copy-ops-util.torch.executor.check-to-copy-args-fn/test]
    // also verifies _to_impl: every IN x OUT dtype pair drives the per-element
    // static_cast copy loop, and the asserted casted outputs would fail if it were wrong.
    // [spec:et:sem:op-to-copy.torch.executor.native.to-impl-fn/test]
    #[test]
    fn op_to_test_all_dtypes_supported() {
        let cases = all_dtypes_cases();
        // ET_FORALL_REALHBF16_TYPES(input) x ET_FORALL_REALHBF16_TYPES(output)
        static_cast_enumerate_out::<u8>(&cases);
        static_cast_enumerate_out::<i8>(&cases);
        static_cast_enumerate_out::<i16>(&cases);
        static_cast_enumerate_out::<i32>(&cases);
        static_cast_enumerate_out::<i64>(&cases);
        static_cast_enumerate_out::<f32>(&cases);
        static_cast_enumerate_out::<f64>(&cases);
        static_cast_enumerate_out::<Half>(&cases);
        static_cast_enumerate_out::<BFloat16>(&cases);
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_bool_tests() {
        let test_case_to_bool = vec![1.1, 2.2, 0.0];
        let result_to_bool = vec![true, true, false];
        // ET_FORALL_REALHBF16_TYPES
        test_runner_to_bool::<u8>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i8>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i16>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i32>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i64>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<f32>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<f64>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<Half>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<BFloat16>(test_case_to_bool, result_to_bool);

        let test_case_from_bool = vec![true, true, false];
        let result_from_bool = vec![1.0, 1.0, 0.0];
        test_runner_from_bool::<u8>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i8>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i16>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i32>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i64>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<f32>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<f64>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<Half>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<BFloat16>(test_case_from_bool, result_from_bool);
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_nan_inf_supported() {
        let float_infinity = f32::INFINITY as f64;
        let nan = f64::NAN;
        let cases = vec![ToTestCase {
            sizes: vec![2, 4],
            data_in: vec![
                2.0,
                3.0,
                nan,
                4.0,
                float_infinity,
                5.0,
                -float_infinity,
                6.0,
            ],
        }];

        // ET_FORALL_FLOATHBF16_TYPES(input) x ET_FORALL_FLOATHBF16_TYPES(output)
        fn enumerate_out<IN>(cases: &[ToTestCase])
        where
            IN: CppTypeToScalarType
                + FactoryValue
                + FromF64
                + NumericCast<f32>
                + NumericCast<f64>
                + NumericCast<Half>
                + NumericCast<BFloat16>,
        {
            test_runner_static_cast::<IN, f32>(cases);
            test_runner_static_cast::<IN, f64>(cases);
            test_runner_static_cast::<IN, Half>(cases);
            test_runner_static_cast::<IN, BFloat16>(cases);
        }
        enumerate_out::<f32>(&cases);
        enumerate_out::<f64>(&cases);
        enumerate_out::<Half>(&cases);
        enumerate_out::<BFloat16>(&cases);
    }

    // PORT-NOTE: `test_runner_hardcode_data<IN, OUT>` skips `OUT == uint8_t`
    // ("Would cause underflow"). Hardcoded input floating data and expected
    // integer outputs (from core PyTorch) match op__to_dim_order_copy.rs.
    fn hardcode_float_data() -> [f64; 15] {
        [
            -1.47900053955270172068,
            -4.59277735274143061872,
            2.15365796963871947156,
            -2.55494554556038755422,
            3.06999137834642255029,
            3.27460679459944969949,
            -3.98865109243288795682,
            -4.81065977167646074975,
            3.67902198302105531980,
            3.72226414774102742911,
            0.80567768667100203572,
            2.23788335717029518435,
            -0.52035578832931150828,
            -1.58493480710766210251,
            -0.30919688936285893988,
        ]
    }
    fn hardcode_float32_data() -> [f32; 15] {
        [
            -1.47900056838989257812,
            -4.59277725219726562500,
            2.15365791320800781250,
            -2.55494546890258789062,
            3.06999135017395019531,
            3.27460670471191406250,
            -3.98865103721618652344,
            -4.81065988540649414062,
            3.67902207374572753906,
            3.72226405143737792969,
            0.80567771196365356445,
            2.23788332939147949219,
            -0.52035576105117797852,
            -1.58493483066558837891,
            -0.30919688940048217773,
        ]
    }
    const HARDCODE_INT: [i64; 15] = [-1, -4, 2, -2, 3, 3, -3, -4, 3, 3, 0, 2, 0, -1, 0];

    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64);

    fn test_runner_hardcode<IN, OUT>(data_in: Vec<IN>)
    where
        IN: CppTypeToScalarType + FactoryValue,
        OUT: CppTypeToScalarType + FactoryValue + FromI64,
    {
        // if (typeid(OUT) == uint8_t) return;
        if OUT::VALUE == ScalarType::Byte {
            return;
        }

        let tf_in = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![3, 5];
        let data_out: Vec<OUT> = HARDCODE_INT.iter().map(|&v| OUT::from_i64(v)).collect();

        let input = tf_in.make_default(sizes.clone(), data_in);
        let output = tf_out.zeros_like(&input, TensorShapeDynamism::STATIC);

        let mut ctx = context();
        let ret = to_copy_out(
            &mut ctx,
            &input,
            false,
            Some(MemoryFormat::Contiguous),
            &output,
        );

        let expected = tf_out.make_default(sizes, data_out);

        assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_hardcode_float_convert_int() {
        // ET_FORALL_FLOATHBF16_TYPES(input) x ET_FORALL_INT_TYPES(output)
        fn enumerate_int_out<IN>(data_in: Vec<IN>)
        where
            IN: CppTypeToScalarType + FactoryValue + Clone,
        {
            test_runner_hardcode::<IN, u8>(data_in.clone());
            test_runner_hardcode::<IN, i8>(data_in.clone());
            test_runner_hardcode::<IN, i16>(data_in.clone());
            test_runner_hardcode::<IN, i32>(data_in.clone());
            test_runner_hardcode::<IN, i64>(data_in);
        }

        let f32_data: Vec<f32> = hardcode_float32_data().to_vec();
        enumerate_int_out::<f32>(f32_data);

        let f64_data: Vec<f64> = hardcode_float_data().to_vec();
        enumerate_int_out::<f64>(f64_data);

        let half_data: Vec<Half> = hardcode_float_data()
            .iter()
            .map(|&d| Half::from_f64(d))
            .collect();
        enumerate_int_out::<Half>(half_data);

        let bf16_data: Vec<BFloat16> = hardcode_float_data()
            .iter()
            .map(|&d| BFloat16::from_f64(d))
            .collect();
        enumerate_int_out::<BFloat16>(bf16_data);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 2, 1, 1]);

        let mut ctx = context();
        to_copy_out(
            &mut ctx,
            &input,
            false,
            Some(MemoryFormat::Contiguous),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen. The C++ uses
    // `static_cast<MemoryFormat>(55)` to construct an illegal memory format; the
    // ported `MemoryFormat` enum only has valid variants, so `Preserve` is used
    // (the only non-`Contiguous`, non-`None` value `check_to_copy_args` rejects).
    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_mismatched_memory_format_dies() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let input = tf_in.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);

        let mut ctx = context();
        to_copy_out(&mut ctx, &input, false, Some(MemoryFormat::Preserve), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // memory format can be null
        let mut ctx = context();
        let ret = to_copy_out(&mut ctx, &input, false, None, &out);
        assert!(tensors_are_close(ret, &input, 0.0, Some(0.0)));
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`; never ATen in the port.
    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_mismatched_blocking_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 1, 1, 2]);

        let mut ctx = context();
        to_copy_out(&mut ctx, &input, true, Some(MemoryFormat::Contiguous), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    fn op_to_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`; portable's `output_resize`
    // SupportedFeature is false, so this test is skipped. Ported as `#[ignore]`.
    // [spec:et:sem:op-to-copy.torch.executor.native.to-copy-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_to_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }
}
