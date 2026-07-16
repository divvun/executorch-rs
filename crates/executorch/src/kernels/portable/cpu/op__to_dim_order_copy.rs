//! Literal port of kernels/portable/cpu/op__to_dim_order_copy.cpp.

use crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesRange;
use crate::kernels::portable::cpu::util::copy_ops_util::{
    _to_dim_order_copy_impl, check__to_dim_order_copy_args,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

type OptionalArrayRefI64 = Option<ArrayRef<i64>>;

// PORT-NOTE: the real<->complex paths in C++ rely on implicit conversions
// between the real element type and the complex's underlying component type
// (`out.real_ = self`, `out = static_cast<CTYPE_OUT>(self.real_)`). The ported
// `Complex<T>` carries plain `real`/`imag` with no conversions, so these two
// local traits reproduce the C++ implicit widening/narrowing without redesigning
// `Complex`: `RealToComplex` builds a complex with `imag = 0`, `ComplexRealPart`
// extracts the real part cast to the destination real type.
trait FromReal<R> {
    fn from_real_with_zero_imag(r: R) -> Self;
}
trait RealPartAs<R> {
    fn real_part_as(self) -> R;
}

macro_rules! impl_complex_conv {
    ($comp:ty) => {
        // real (Half/f32/f64) -> Complex<$comp>: assign `.real_ = self`, `.imag_ = 0`.
        impl FromReal<Half> for Complex<$comp> {
            fn from_real_with_zero_imag(r: Half) -> Self {
                Complex {
                    real: real_to::<$comp>(r.to_f64()),
                    imag: real_to::<$comp>(0.0),
                }
            }
        }
        impl FromReal<f32> for Complex<$comp> {
            fn from_real_with_zero_imag(r: f32) -> Self {
                Complex {
                    real: real_to::<$comp>(r as f64),
                    imag: real_to::<$comp>(0.0),
                }
            }
        }
        impl FromReal<f64> for Complex<$comp> {
            fn from_real_with_zero_imag(r: f64) -> Self {
                Complex {
                    real: real_to::<$comp>(r),
                    imag: real_to::<$comp>(0.0),
                }
            }
        }
        // Complex<$comp> -> real: `static_cast<CTYPE_OUT>(self.real_)`.
        impl RealPartAs<Half> for Complex<$comp> {
            fn real_part_as(self) -> Half {
                Half::from_f64(comp_to_f64(self.real))
            }
        }
        impl RealPartAs<f32> for Complex<$comp> {
            fn real_part_as(self) -> f32 {
                comp_to_f64(self.real) as f32
            }
        }
        impl RealPartAs<f64> for Complex<$comp> {
            fn real_part_as(self) -> f64 {
                comp_to_f64(self.real)
            }
        }
    };
}
impl_complex_conv!(Half);
impl_complex_conv!(f32);
impl_complex_conv!(f64);

trait CompFromF64 {
    fn comp_from_f64(v: f64) -> Self;
    fn comp_as_f64(self) -> f64;
}
impl CompFromF64 for Half {
    fn comp_from_f64(v: f64) -> Self {
        Half::from_f64(v)
    }
    fn comp_as_f64(self) -> f64 {
        self.to_f64()
    }
}
impl CompFromF64 for f32 {
    fn comp_from_f64(v: f64) -> Self {
        v as f32
    }
    fn comp_as_f64(self) -> f64 {
        self as f64
    }
}
impl CompFromF64 for f64 {
    fn comp_from_f64(v: f64) -> Self {
        v
    }
    fn comp_as_f64(self) -> f64 {
        self
    }
}
fn real_to<C: CompFromF64>(v: f64) -> C {
    C::comp_from_f64(v)
}
fn comp_to_f64<C: CompFromF64>(v: C) -> f64 {
    v.comp_as_f64()
}

// _to_dim_order_copy.out(Tensor self, *, bool non_blocking=False, int[]?
// dim_order=None, Tensor(a!) out) -> Tensor(a!)
// [spec:et:def:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn]
// [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn]
pub fn _to_dim_order_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &'a Tensor<'b>,
    non_blocking: bool,
    dim_order: OptionalArrayRefI64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;
    crate::et_kernel_check!(
        ctx,
        check__to_dim_order_copy_args(self_, non_blocking, dim_order, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, self_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    if self_.numel() == 0 {
        return out;
    }

    let op_name = "dim_order_ops::_to_dim_order_copy.out";

    let in_is_complex = is_complex_type(self_.scalar_type());
    let out_is_complex = is_complex_type(out.scalar_type());

    if in_is_complex && out_is_complex {
        // Complex to complex: same type copy
        crate::et_switch_complexh_types!(self_.scalar_type(), ctx, op_name, CTYPE, {
            _to_dim_order_copy_impl::<CTYPE, CTYPE>(self_, out);
        });
    } else if !in_is_complex && out_is_complex {
        // Real to complex: convert real value to complex with zero imaginary part
        crate::et_switch_floath_types!(self_.scalar_type(), ctx, op_name, CTYPE_IN, {
            crate::et_switch_complexh_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
                let self_data = self_.mutable_data_ptr::<CTYPE_IN>();
                let out_data = out.mutable_data_ptr::<CTYPE_OUT>();
                for indexes in
                    BroadcastIndexesRange::<3>::new_with_support(self_, &[self_, out], true)
                {
                    let self_data_index = indexes[1];
                    let out_data_index = indexes[2];
                    unsafe {
                        *out_data.offset(out_data_index) =
                            <CTYPE_OUT as FromReal<CTYPE_IN>>::from_real_with_zero_imag(
                                *self_data.offset(self_data_index),
                            );
                    }
                }
            });
        });
    } else if in_is_complex && !out_is_complex {
        // Complex to real: take real part
        crate::et_switch_complexh_types!(self_.scalar_type(), ctx, op_name, CTYPE_IN, {
            crate::et_switch_floath_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
                let self_data = self_.mutable_data_ptr::<CTYPE_IN>();
                let out_data = out.mutable_data_ptr::<CTYPE_OUT>();
                for indexes in
                    BroadcastIndexesRange::<3>::new_with_support(self_, &[self_, out], true)
                {
                    let self_data_index = indexes[1];
                    let out_data_index = indexes[2];
                    unsafe {
                        *out_data.offset(out_data_index) =
                            <CTYPE_IN as RealPartAs<CTYPE_OUT>>::real_part_as(
                                *self_data.offset(self_data_index),
                            );
                    }
                }
            });
        });
    } else {
        // Real to real
        crate::et_switch_realhbbf16_types!(self_.scalar_type(), ctx, op_name, CTYPE_IN, {
            crate::et_switch_realhbbf16_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
                _to_dim_order_copy_impl::<CTYPE_IN, CTYPE_OUT>(self_, out);
            });
        });
    }

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
    use crate::runtime::core::portable_type::{BFloat16, ComplexDouble, ComplexFloat};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // `static_cast<T>(f64)` for the element ctypes.
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
    impl FromF64 for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    struct ToTestCase {
        sizes: Vec<i32>,
        data_in: Vec<f64>,
    }

    // Mirrors `test_runner_static_cast<IN, IN_DTYPE, OUT, OUT_DTYPE>`.
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

            let mut dim_order_vec: Vec<i64> = Vec::new();
            for i in 0..input.dim() {
                dim_order_vec.push(i as i64);
            }
            let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

            let mut ctx = context();
            let ret = _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &output);

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

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &output);

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

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &output);

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

        let out = tf.zeros(out_shape, dynamism);

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..x.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &x, non_blocking, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
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

    // For a fixed input dtype, enumerate the output dtype over ET_FORALL_REAL_TYPES.
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
            + NumericCast<f64>,
    {
        test_runner_static_cast::<IN, u8>(cases);
        test_runner_static_cast::<IN, i8>(cases);
        test_runner_static_cast::<IN, i16>(cases);
        test_runner_static_cast::<IN, i32>(cases);
        test_runner_static_cast::<IN, i64>(cases);
        test_runner_static_cast::<IN, f32>(cases);
        test_runner_static_cast::<IN, f64>(cases);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    // also verifies check__to_dim_order_copy_args (non_blocking==false gate,
    // Some(dim_order) branch: size == input.dim, contiguous dim-order accepted,
    // out dim_order matches the requested one)
    // [spec:et:sem:copy-ops-util.torch.executor.check-to-dim-order-copy-args-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_all_dtypes_supported() {
        let cases = all_dtypes_cases();
        static_cast_enumerate_out::<u8>(&cases);
        static_cast_enumerate_out::<i8>(&cases);
        static_cast_enumerate_out::<i16>(&cases);
        static_cast_enumerate_out::<i32>(&cases);
        static_cast_enumerate_out::<i64>(&cases);
        static_cast_enumerate_out::<f32>(&cases);
        static_cast_enumerate_out::<f64>(&cases);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_bool_tests() {
        let test_case_to_bool = vec![1.1, 2.2, 0.0];
        let result_to_bool = vec![true, true, false];
        // ET_FORALL_REAL_TYPES
        test_runner_to_bool::<u8>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i8>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i16>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i32>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<i64>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<f32>(test_case_to_bool.clone(), result_to_bool.clone());
        test_runner_to_bool::<f64>(test_case_to_bool, result_to_bool);

        let test_case_from_bool = vec![true, true, false];
        let result_from_bool = vec![1.0, 1.0, 0.0];
        test_runner_from_bool::<u8>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i8>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i16>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i32>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<i64>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<f32>(test_case_from_bool.clone(), result_from_bool.clone());
        test_runner_from_bool::<f64>(test_case_from_bool, result_from_bool);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_nan_inf_supported() {
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

        // ET_FORALL_FLOATHBF16_TYPES x ET_FORALL_FLOATHBF16_TYPES
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
    // ("Would cause underflow"). The hardcoded input floating data and expected
    // integer outputs are the same across dtypes, so it is expressed directly.
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

        let mut dim_order_vec: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            dim_order_vec.push(i as i64);
        }
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &output);

        let expected = tf_out.make_default(sizes, data_out);

        assert!(tensors_are_close(ret, &output, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    trait FromI64: Copy {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64);

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_hardcode_float_convert_int() {
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

    fn dim_order_vec_from(input: &Tensor) -> Vec<i64> {
        let mut v: Vec<i64> = Vec::new();
        for i in 0..input.dim() {
            v.push(i as i64);
        }
        v
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen; the failure path runs.
    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_mismatched_sizes_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 2, 1, 1]);
        let dim_order_vec = dim_order_vec_from(&input);
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_mismatched_memory_format_dies() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let input = tf_in.make_default(vec![3, 1, 1, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = tf_out.zeros_default(vec![3, 1, 1, 2]);

        let mut dim_order_vec = dim_order_vec_from(&input);
        // mutate dim_order_vec to create a illegal one.
        dim_order_vec[1] = 3;
        dim_order_vec[3] = 1;
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &input, false, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_mismatched_blocking_die() {
        let tf = TensorFactory::<i32>::new();
        let input = tf.make_default(vec![3, 1, 1, 2], vec![1, 2, 3, 4, 5, 6]);
        let out = tf.zeros_default(vec![3, 1, 1, 2]);

        let dim_order_vec = dim_order_vec_from(&input);
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), dim_order_vec.len());

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &input, true, Some(dim_order), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)`: portable's `output_resize`
    // SupportedFeature is false, so this test is skipped. Ported as `#[ignore]`.
    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    #[ignore = "SKIP_IF(!output_resize): portable kernel does not support output resize"]
    fn op_to_dim_order_copy_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    const CL_CONTIG_DATA: [f64; 60] = [
        0.2432, 0.5248, 0.5361, 0.8513, 0.8184, 0.8206, 0.7357, 0.9655, 0.6138, 0.1112, 0.2799,
        0.1079, 0.9680, 0.2548, 0.0393, 0.6002, 0.2257, 0.8766, 0.2715, 0.1595, 0.2029, 0.7026,
        0.6982, 0.8529, 0.4405, 0.6560, 0.9217, 0.6372, 0.2446, 0.6590, 0.3866, 0.7185, 0.4439,
        0.5346, 0.3179, 0.4492, 0.3491, 0.6970, 0.8456, 0.2516, 0.2345, 0.2924, 0.7695, 0.0911,
        0.8530, 0.8560, 0.6909, 0.7719, 0.8923, 0.5546, 0.6978, 0.8151, 0.3007, 0.3961, 0.8416,
        0.4296, 0.7203, 0.8963, 0.3597, 0.5552,
    ];

    const CL_CHANNELS_LAST_DATA: [f64; 60] = [
        0.2432, 0.8184, 0.6138, 0.9680, 0.2257, 0.5248, 0.8206, 0.1112, 0.2548, 0.8766, 0.5361,
        0.7357, 0.2799, 0.0393, 0.2715, 0.8513, 0.9655, 0.1079, 0.6002, 0.1595, 0.2029, 0.4405,
        0.2446, 0.4439, 0.3491, 0.7026, 0.6560, 0.6590, 0.5346, 0.6970, 0.6982, 0.9217, 0.3866,
        0.3179, 0.8456, 0.8529, 0.6372, 0.7185, 0.4492, 0.2516, 0.2345, 0.8530, 0.8923, 0.3007,
        0.7203, 0.2924, 0.8560, 0.5546, 0.3961, 0.8963, 0.7695, 0.6909, 0.6978, 0.8416, 0.3597,
        0.0911, 0.7719, 0.8151, 0.4296, 0.5552,
    ];

    fn f32_vec(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    // Also exercises _to_dim_order_copy_impl<f32,f32>: the contiguous->channels-
    // last reshuffle is exactly the dim-order-respecting index mapping the impl
    // produces via BroadcastIndexesRange(support_noncontiguous_input=true).
    // [spec:et:sem:copy-ops-util.torch.executor.to-dim-order-copy-impl-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_contiguous_to_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CONTIG_DATA),
            vec![],
            TensorShapeDynamism::STATIC,
        );

        let out = tf.full_channels_last(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);
        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let dim_order_vec: [i64; 4] = [0, 2, 3, 1];
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), 4);
        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &x, false, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_channels_last_to_contiguous() {
        let tf = TensorFactory::<f32>::new();

        let out = tf.full(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);
        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CONTIG_DATA),
            vec![],
            TensorShapeDynamism::STATIC,
        );

        let dim_order_vec: [i64; 4] = [0, 1, 2, 3];
        let dim_order = ArrayRef::from_raw_parts(dim_order_vec.as_ptr(), 4);
        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &x, false, Some(dim_order), &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_preserve_channels_last() {
        let tf = TensorFactory::<f32>::new();

        let out = tf.full_channels_last(vec![3, 5, 2, 2], 0.0, TensorShapeDynamism::STATIC);
        let x = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let expected = tf.make_with_dimorder(
            vec![3, 5, 2, 2],
            f32_vec(&CL_CHANNELS_LAST_DATA),
            vec![0, 2, 3, 1],
            TensorShapeDynamism::STATIC,
        );

        let mut ctx = context();
        let ret = _to_dim_order_copy_out(&mut ctx, &x, false, None, &out);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
        assert!(tensors_are_close(ret, &expected, 0.0, Some(0.0)));
    }

    //
    // Complex Type Tests
    //

    fn cf(re: f32, im: f32) -> ComplexFloat {
        ComplexFloat { real: re, imag: im }
    }
    fn cd(re: f64, im: f64) -> ComplexDouble {
        ComplexDouble { real: re, imag: im }
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_complex_float_to_complex_float() {
        let tf = TensorFactory::<ComplexFloat>::new();
        let sizes = vec![2, 2];

        let self_ = tf.make_default(
            sizes.clone(),
            vec![cf(1.0, 2.0), cf(3.0, 4.0), cf(5.0, 6.0), cf(7.0, 8.0)],
        );
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &self_, false, None, &out);

        assert!(tensors_are_close(&out, &self_, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_float_to_complex_float() {
        let tf_in = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<ComplexFloat>::new();
        let sizes = vec![2, 2];

        let self_ = tf_in.make_default(sizes.clone(), vec![1.0, 2.0, 3.0, 4.0]);
        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &self_, false, None, &out);

        let expected = tf_out.make_default(
            sizes,
            vec![cf(1.0, 0.0), cf(2.0, 0.0), cf(3.0, 0.0), cf(4.0, 0.0)],
        );

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_complex_float_to_float() {
        let tf_in = TensorFactory::<ComplexFloat>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];

        let self_ = tf_in.make_default(
            sizes.clone(),
            vec![cf(1.0, 10.0), cf(2.0, 20.0), cf(3.0, 30.0), cf(4.0, 40.0)],
        );
        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &self_, false, None, &out);

        let expected = tf_out.make_default(sizes, vec![1.0, 2.0, 3.0, 4.0]);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn/test]
    #[test]
    fn op_to_dim_order_copy_test_complex_double_to_complex_double() {
        let tf = TensorFactory::<ComplexDouble>::new();
        let sizes = vec![3];

        let self_ = tf.make_default(
            sizes.clone(),
            vec![cd(1.5, 2.5), cd(-3.5, 4.5), cd(0.0, -1.0)],
        );
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        _to_dim_order_copy_out(&mut ctx, &self_, false, None, &out);

        assert!(tensors_are_close(&out, &self_, 0.0, Some(0.0)));
    }
}
