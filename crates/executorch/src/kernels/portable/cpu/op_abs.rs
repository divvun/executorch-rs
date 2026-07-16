//! Literal port of kernels/portable/cpu/op_abs.cpp.

use crate::kernels::portable::cpu::pattern::pattern::FromF64;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ complex path computes `sqrt(v.real_*v.real_ +
// v.imag_*v.imag_)` in the underlying float element type and implicitly converts
// to CTYPE_OUT. The ported `Complex<T>` carries `real`/`imag` but no arithmetic;
// this module-local trait exposes the two components promoted to f64 so the
// magnitude can be formed without redesigning `Complex`. The math is performed in
// f64 and cast back via `FromF64`, matching the C++ conversion chain.
trait ComplexParts {
    fn real_f64(self) -> f64;
    fn imag_f64(self) -> f64;
}
impl ComplexParts for Complex<Half> {
    fn real_f64(self) -> f64 {
        self.real.to_f64()
    }
    fn imag_f64(self) -> f64 {
        self.imag.to_f64()
    }
}
impl ComplexParts for Complex<f32> {
    fn real_f64(self) -> f64 {
        self.real as f64
    }
    fn imag_f64(self) -> f64 {
        self.imag as f64
    }
}
impl ComplexParts for Complex<f64> {
    fn real_f64(self) -> f64 {
        self.real
    }
    fn imag_f64(self) -> f64 {
        self.imag
    }
}

// PORT-NOTE: the real path branches `if (val_in < 0) return -val_in; else return
// val_in;` and casts back with `static_cast<CTYPE>`. For the unsigned Byte ctype
// this is always the identity (never `< 0`) and `-val_in` never runs. This local
// trait reproduces that per-ctype behavior over the REALHBF16 set.
trait AbsReal: Copy {
    fn abs_val(self) -> Self;
}
macro_rules! impl_abs_real_signed {
    ($($t:ty),*) => {$(
        impl AbsReal for $t {
            fn abs_val(self) -> Self {
                if self < (0 as $t) {
                    (-self) as $t
                } else {
                    self as $t
                }
            }
        }
    )*};
}
impl_abs_real_signed!(i8, i16, i32, i64, f32, f64);
impl AbsReal for u8 {
    fn abs_val(self) -> Self {
        // Unsigned: `val_in < 0` is always false, so this returns `val_in`.
        if self < 0u8 {
            self.wrapping_neg()
        } else {
            self
        }
    }
}
impl AbsReal for Half {
    fn abs_val(self) -> Self {
        if self < Half::from_f32_const(0.0) {
            Half::from_f64(-self.to_f64())
        } else {
            self
        }
    }
}
impl AbsReal for crate::runtime::core::portable_type::BFloat16 {
    fn abs_val(self) -> Self {
        use crate::runtime::core::portable_type::BFloat16;
        if self < BFloat16::from_f32_const(0.0) {
            BFloat16::from_f64(-self.to_f64())
        } else {
            self
        }
    }
}

// [spec:et:def:op-abs.torch.executor.native.abs-out-fn]
// [spec:et:sem:op-abs.torch.executor.native.abs-out-fn]
pub fn abs_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = &ctx;

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let in_is_complex = is_complex_type(in_.scalar_type());
    crate::et_kernel_check!(
        ctx,
        in_is_complex || tensors_have_same_dtype2(in_, out),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let op_name = "abs.out";

    if in_is_complex {
        // NOTE: Elected not to add COMPLEXH to dtype_util.h for now
        // because I am not planning wide rollout of complex support; if
        // we do add SupportedTensorDtypes::COMPLEXH support, then we
        // should use it here.
        crate::et_switch_complexh_types!(in_.scalar_type(), ctx, op_name, CTYPE_IN, {
            crate::et_switch_floath_types!(out.scalar_type(), ctx, op_name, CTYPE_OUT, {
                apply_unary_map_fn(
                    |val_in: CTYPE_IN| -> CTYPE_OUT {
                        let re = val_in.real_f64();
                        let im = val_in.imag_f64();
                        CTYPE_OUT::from_f64((re * re + im * im).sqrt())
                    },
                    in_.const_data_ptr::<CTYPE_IN>(),
                    out.mutable_data_ptr::<CTYPE_OUT>(),
                    in_.numel() as i64,
                    1,
                );
            });
        });
    } else {
        crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, op_name, CTYPE, {
            apply_unary_map_fn(
                // `if (val_in < 0) return -val_in; else return val_in;` — folded
                // into `AbsReal::abs_val` per ctype (identity for unsigned Byte).
                |val_in: CTYPE| -> CTYPE { val_in.abs_val() },
                in_.const_data_ptr::<CTYPE>(),
                out.mutable_data_ptr::<CTYPE>(),
                in_.numel() as i64,
                1,
            );
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::to_real_value_type;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat, ComplexHalf};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

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

    fn op_abs_out<'a, 'b>(self_: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        abs_out(&mut ctx, self_, out)
    }

    // PORT-NOTE: `run_smoke_test<DTYPE>` templated helper. The C++ SmokeTest
    // dispatches over ET_FORALL_FLOATHBF16_TYPES (f32, f64, Half, BFloat16). Each
    // dtype instantiation is expanded as a separate call.
    fn run_smoke_test<T>()
    where
        T: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let in_ = tf.make_default(
            vec![1, 7],
            vec![
                T::from_f64(-3.0),
                T::from_f64(-2.5),
                T::from_f64(-1.01),
                T::from_f64(0.0),
                T::from_f64(1.01),
                T::from_f64(2.5),
                T::from_f64(3.0),
            ],
        );
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(
            vec![1, 7],
            vec![
                T::from_f64(3.0),
                T::from_f64(2.5),
                T::from_f64(1.01),
                T::from_f64(0.0),
                T::from_f64(1.01),
                T::from_f64(2.5),
                T::from_f64(3.0),
            ],
        );

        let ret = op_abs_out(&in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // abs is computed elementwise through apply_unary_map_fn; the per-element
    // results pin its map-over-strided-buffer behavior.
    // [spec:et:sem:op-abs.torch.executor.native.abs-out-fn/test]
    // [spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn/test]
    #[test]
    fn op_abs_test_smoke_test() {
        run_smoke_test::<f32>();
        run_smoke_test::<f64>();
        run_smoke_test::<Half>();
        run_smoke_test::<crate::runtime::core::portable_type::BFloat16>();
    }

    // PORT-NOTE: `run_complex_smoke_test<CTYPE, DTYPE>` templated helper. The C++
    // ComplexSmokeTest dispatches over ET_FORALL_COMPLEXH_TYPES (ComplexHalf,
    // ComplexFloat, ComplexDouble). The output factory uses toRealValueType(DTYPE)
    // -> the real element type (Half, f32, f64).
    fn run_complex_smoke_test<C, R>(mk: impl Fn(f64, f64) -> C, out_dtype: ScalarType)
    where
        C: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue,
        R: crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType
            + crate::runtime::core::exec_aten::testing_util::tensor_factory::FactoryValue
            + FromF64,
    {
        // Mirror `constexpr auto REAL_DTYPE = toRealValueType(DTYPE)`.
        assert_eq!(to_real_value_type(C::VALUE), out_dtype);
        let tf = TensorFactory::<C>::new();
        let tf_out = TensorFactory::<R>::new();
        let in_ = tf.make_default(vec![1, 2], vec![mk(3.0, 4.0), mk(5.0, 12.0)]);
        let out = tf_out.zeros_default(vec![1, 2]);
        let expected = tf_out.make_default(vec![1, 2], vec![R::from_f64(5.0), R::from_f64(13.0)]);
        let ret = op_abs_out(&in_, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &expected,
            crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-abs.torch.executor.native.abs-out-fn/test]
    #[test]
    fn op_abs_test_complex_smoke_test() {
        run_complex_smoke_test::<ComplexHalf, Half>(
            |re, im| ComplexHalf {
                real: Half::from_f64(re),
                imag: Half::from_f64(im),
            },
            ScalarType::Half,
        );
        run_complex_smoke_test::<ComplexFloat, f32>(
            |re, im| ComplexFloat {
                real: re as f32,
                imag: im as f32,
            },
            ScalarType::Float,
        );
        run_complex_smoke_test::<ComplexDouble, f64>(
            |re, im| ComplexDouble { real: re, imag: im },
            ScalarType::Double,
        );
    }

    // [spec:et:sem:op-abs.torch.executor.native.abs-out-fn/test]
    #[test]
    fn op_abs_test_memory_format_check() {
        let tf = TensorFactory::<f32>::new();

        let sizes = vec![2, 3, 1, 5];

        let input_contiguous = tf.make_default(
            sizes.clone(),
            vec![
                0.8737, 0.5359, 0.3743, -0.3040, -0.7800, -0.2306, -0.7684, -0.5364, 0.3478,
                -0.3289, 0.0829, 0.2939, -0.8211, 0.8572, -0.0802, 0.9252, -0.2093, 0.9013,
                -0.4197, 0.3987, -0.5291, -0.5567, 0.2691, 0.7819, -0.8009, -0.4286, -0.9299,
                0.2143, 0.2565, -0.5701,
            ],
        );
        let expected_contiguous = tf.make_default(
            sizes.clone(),
            vec![
                0.8737, 0.5359, 0.3743, 0.3040, 0.7800, 0.2306, 0.7684, 0.5364, 0.3478, 0.3289,
                0.0829, 0.2939, 0.8211, 0.8572, 0.0802, 0.9252, 0.2093, 0.9013, 0.4197, 0.3987,
                0.5291, 0.5567, 0.2691, 0.7819, 0.8009, 0.4286, 0.9299, 0.2143, 0.2565, 0.5701,
            ],
        );

        // ET_TEST_OP_SUPPORTS_MEMORY_FORMATS(tf, op_abs_out, ..., channels_last=true)
        test_op_supports_memory_formats(&tf, &input_contiguous, &expected_contiguous, true);
    }

    // Literal port of the `ET_TEST_OP_SUPPORTS_MEMORY_FORMATS` macro (non-ATen
    // branch) specialized to `op_abs_out`.
    fn test_op_supports_memory_formats(
        tf: &TensorFactory<f32>,
        input_contiguous: &Tensor,
        expected_contiguous: &Tensor,
        channels_last_support: bool,
    ) {
        let input_channels_last =
            tf.channels_last_like(input_contiguous, TensorShapeDynamism::STATIC);
        let expected_channel_last =
            tf.channels_last_like(expected_contiguous, TensorShapeDynamism::STATIC);

        let output_contiguous = tf.zeros_like(expected_contiguous, TensorShapeDynamism::STATIC);
        let output_channels_last =
            tf.channels_last_like(&output_contiguous, TensorShapeDynamism::STATIC);

        let ret = op_abs_out(&input_channels_last, &output_channels_last);
        if channels_last_support {
            assert!(tensors_are_close(
                &output_channels_last,
                &expected_channel_last,
                0.0,
                Some(0.0)
            ));
        } else {
            assert!(!tensors_are_close(
                &output_channels_last,
                &expected_channel_last,
                0.0,
                Some(0.0)
            ));
        }
        assert!(tensors_are_close(
            &output_channels_last,
            ret,
            0.0,
            Some(0.0)
        ));

        // ET_EXPECT_KERNEL_FAILURE(context_, op(input_channels_last, output_contiguous))
        let mut ctx = context();
        abs_out(&mut ctx, &input_channels_last, &output_contiguous);
        assert_ne!(ctx.failure_state(), Error::Ok);

        // ET_EXPECT_KERNEL_FAILURE(context_, op(input_contiguous, output_channels_last))
        let mut ctx = context();
        abs_out(&mut ctx, input_contiguous, &output_channels_last);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: local `from_f64` bridge for the FLOATHBF16 element types used by
    // the smoke tests (mirrors op_ceil.rs's test helper).
    trait FromF64 {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for crate::runtime::core::portable_type::BFloat16 {
        fn from_f64(v: f64) -> Self {
            crate::runtime::core::portable_type::BFloat16::from_f32(v as f32)
        }
    }
}
