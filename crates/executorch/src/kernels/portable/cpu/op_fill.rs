//! Literal port of kernels/portable/cpu/op_fill.cpp.

use crate::kernels::portable::cpu::scalar_utils::internal::check_overflow_scalar_cast;
use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensor_is_scalar, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped.

// PORT-NOTE: local `et_check_msg!` mirroring the C++ fatal `ET_CHECK_MSG` used by
// `ET_EXTRACT_SCALAR_TENSOR` (message dropped, a fatal abort follows), matching
// the established per-module definitions (see tensor_util.rs / scalar_type_util.rs).
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `ET_EXTRACT_SCALAR_TENSOR(b, b_val)` expands to
// `ET_CHECK_MSG(extract_scalar_tensor(b, &b_val), ...)`, where the overloaded
// `extract_scalar_tensor` is resolved by the target type `CTYPE_B`. Rust cannot
// select overloads by out-param type, so the resolution is modeled by an
// `ExtractScalarTensor` trait — one impl per REALHBBF16 ctype — each reproducing
// the C++ overload the compiler would have picked:
//   - integer ctypes: the `is_integral && !bool` overload (range-check widened to
//     i128 against `lowest()/max()`, narrowing static_cast);
//   - Float/Double/Half/BFloat16: the floating overload (widen element to f64,
//     finite range-check against `numeric_limits<FLOAT_T>::lowest()/max()`,
//     `static_cast<FLOAT_T>`);
//   - Bool: the bool overload.
// This reproduces the C++ overloads exactly, whereas the ported util's single
// `extract_scalar_tensor_float` only targets `f64` (f64 bounds, no per-FLOAT_T
// narrowing) — so it is NOT reused for the f32/Half/BFloat16 arms here. Util
// float-only-f64 limitation noted for the fixer.
trait ExtractScalarTensor: Sized {
    fn extract_scalar_tensor(tensor: &Tensor, out_val: &mut Self) -> bool;
}

macro_rules! impl_extract_int {
    ($t:ty) => {
        impl ExtractScalarTensor for $t {
            fn extract_scalar_tensor(tensor: &Tensor, out_val: &mut Self) -> bool {
                if tensor.numel() != 1 {
                    return false;
                }
                macro_rules! case_int_dtype {
                    ($tensor_ctype:ty) => {{
                        let val: $tensor_ctype =
                            unsafe { *tensor.const_data_ptr::<$tensor_ctype>() };
                        let val_i128: i128 = val as i128;
                        if val_i128 < <$t>::MIN as i128 || val_i128 > <$t>::MAX as i128 {
                            return false;
                        }
                        *out_val = val_i128 as $t;
                        return true;
                    }};
                }
                match tensor.scalar_type() {
                    ScalarType::Byte => case_int_dtype!(u8),
                    ScalarType::Char => case_int_dtype!(i8),
                    ScalarType::Short => case_int_dtype!(i16),
                    ScalarType::Int => case_int_dtype!(i32),
                    ScalarType::Long => case_int_dtype!(i64),
                    _ => false,
                }
            }
        }
    };
}
impl_extract_int!(u8);
impl_extract_int!(i8);
impl_extract_int!(i16);
impl_extract_int!(i32);
impl_extract_int!(i64);

// The floating overload: range-check against `numeric_limits<FLOAT_T>::lowest()`
// (= -MAX for floating types) and `max()`, narrowing to FLOAT_T. `$lowest`/`$max`
// are the FLOAT_T bounds widened to f64; `$from_f64` is `static_cast<FLOAT_T>`.
macro_rules! impl_extract_float {
    ($t:ty, $lowest:expr, $max:expr, $from_f64:expr) => {
        impl ExtractScalarTensor for $t {
            fn extract_scalar_tensor(tensor: &Tensor, out_val: &mut Self) -> bool {
                if tensor.numel() != 1 {
                    return false;
                }
                macro_rules! case_real_dtype {
                    ($tensor_ctype:ty, $to_f64:expr) => {{
                        let raw: $tensor_ctype =
                            unsafe { *tensor.const_data_ptr::<$tensor_ctype>() };
                        let val: f64 = $to_f64(raw);
                        if val.is_finite() && (val < $lowest || val > $max) {
                            return false;
                        }
                        *out_val = $from_f64(val);
                        return true;
                    }};
                }
                match tensor.scalar_type() {
                    ScalarType::Byte => case_real_dtype!(u8, |v: u8| v as f64),
                    ScalarType::Char => case_real_dtype!(i8, |v: i8| v as f64),
                    ScalarType::Short => case_real_dtype!(i16, |v: i16| v as f64),
                    ScalarType::Int => case_real_dtype!(i32, |v: i32| v as f64),
                    ScalarType::Long => case_real_dtype!(i64, |v: i64| v as f64),
                    ScalarType::Float => case_real_dtype!(f32, |v: f32| v as f64),
                    ScalarType::Double => case_real_dtype!(f64, |v: f64| v),
                    ScalarType::Half => case_real_dtype!(Half, |v: Half| v.to_f64()),
                    ScalarType::BFloat16 => case_real_dtype!(BFloat16, |v: BFloat16| v.to_f64()),
                    _ => false,
                }
            }
        }
    };
}
impl_extract_float!(f32, f32::MIN as f64, f32::MAX as f64, |v: f64| v as f32);
impl_extract_float!(f64, f64::MIN, f64::MAX, |v: f64| v);
impl_extract_float!(Half, -(Half::MAX.to_f64()), Half::MAX.to_f64(), |v: f64| {
    Half::from_f64(v)
});
impl_extract_float!(
    BFloat16,
    -(BFloat16::MAX.to_f64()),
    BFloat16::MAX.to_f64(),
    |v: f64| { BFloat16::from_f64(v) }
);

impl ExtractScalarTensor for bool {
    fn extract_scalar_tensor(tensor: &Tensor, out_val: &mut Self) -> bool {
        if tensor.scalar_type() != ScalarType::Bool {
            return false;
        }
        if tensor.numel() != 1 {
            return false;
        }
        let val: bool = unsafe { *tensor.const_data_ptr::<bool>() };
        *out_val = val;
        true
    }
}

// [spec:et:def:op-fill.torch.executor.native.fill-scalar-out-fn]
// [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn]
pub fn fill_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type: ScalarType = a.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    crate::et_kernel_check!(ctx, a_type == out_type, InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let op_name = "fill.Scalar_out";

    crate::et_switch_realhbbf16_types!(a_type, ctx, op_name, CTYPE_A, {
        let opt_b_casted = check_overflow_scalar_cast::<CTYPE_A>(b);
        crate::et_kernel_check!(ctx, opt_b_casted.is_some(), InvalidArgument, out);
        let b_casted = opt_b_casted.unwrap();

        apply_unary_map_fn(
            |_val_a: CTYPE_A| -> CTYPE_A { b_casted },
            a.const_data_ptr::<CTYPE_A>(),
            out.mutable_data_ptr::<CTYPE_A>(),
            out.numel() as i64,
            1,
        );
    });

    out
}

// [spec:et:def:op-fill.torch.executor.native.fill-tensor-out-fn]
// [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn]
pub fn fill_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Assert `b` must be a scalar tensor.
    crate::et_kernel_check!(ctx, tensor_is_scalar(b), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    let a_type: ScalarType = a.scalar_type();
    let b_type: ScalarType = b.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    crate::et_kernel_check!(ctx, a_type == out_type, InvalidArgument, out);

    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    let op_name = "fill.Tensor_out";

    crate::et_switch_realhbbf16_types!(a_type, ctx, op_name, CTYPE_A, {
        let mut b_casted: CTYPE_A = Default::default();
        crate::et_switch_realhbbf16_types!(b_type, ctx, op_name, CTYPE_B, {
            let mut b_val: CTYPE_B = Default::default();
            et_check_msg!(
                <CTYPE_B as ExtractScalarTensor>::extract_scalar_tensor(b, &mut b_val),
                "b could not be extracted: wrong type or out of range"
            );
            // b_casted = static_cast<CTYPE_A>(b_val)
            b_casted = <CTYPE_A as StaticCast<CTYPE_B>>::static_cast(b_val);
        });

        apply_unary_map_fn(
            |_val_a: CTYPE_A| -> CTYPE_A { b_casted },
            a.const_data_ptr::<CTYPE_A>(),
            out.mutable_data_ptr::<CTYPE_A>(),
            out.numel() as i64,
            1,
        );
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

    fn op_fill_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        fill_scalar_out(&mut ctx, self_, other, out)
    }

    fn op_fill_tensor_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        fill_tensor_out(&mut ctx, self_, other, out)
    }

    // PORT-NOTE: `DTYPE == Bool` selects `other = false` / `full(sizes, false)` in
    // the C++ template; the `Fill` trait captures the per-element-type "one" fill
    // value and matching scalar the same way (bool -> false, else -> 1).
    trait Fill: FactoryValue + CppTypeToScalarType {
        fn other_scalar() -> Scalar;
        fn exp_value() -> Self;
    }
    macro_rules! impl_fill_num {
        ($($t:ty),*) => {$(impl Fill for $t {
            fn other_scalar() -> Scalar { Scalar::from_i64(1) }
            fn exp_value() -> Self { 1 as $t }
        })*};
    }
    impl_fill_num!(u8, i8, i16, i32, i64, f32, f64);
    impl Fill for Half {
        fn other_scalar() -> Scalar {
            Scalar::from_i64(1)
        }
        fn exp_value() -> Self {
            Half::from_f32(1.0)
        }
    }
    impl Fill for BFloat16 {
        fn other_scalar() -> Scalar {
            Scalar::from_i64(1)
        }
        fn exp_value() -> Self {
            BFloat16::from_f32(1.0)
        }
    }
    impl Fill for bool {
        fn other_scalar() -> Scalar {
            Scalar::from_bool(false)
        }
        fn exp_value() -> Self {
            false
        }
    }

    fn test_fill_scalar_out<T>(sizes: Vec<i32>)
    where
        T: Fill,
    {
        let tf = TensorFactory::<T>::new();

        // Before: `out` consists of 0s.
        let self_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());

        // After: `out` consists of 1s.
        let other = T::other_scalar();
        op_fill_scalar_out(&self_, &other, &out);

        let exp_out = tf.full(sizes, T::exp_value(), TensorShapeDynamism::STATIC);

        assert!(tensors_are_close(&out, &exp_out, 0.0, Some(0.0)));
    }

    fn test_fill_tensor_out<T>(sizes: Vec<i32>)
    where
        T: Fill,
    {
        let tf = TensorFactory::<T>::new();

        // Before: `out` consists of 0s.
        let self_ = tf.zeros_default(sizes.clone());
        let out = tf.zeros_default(sizes.clone());

        // After: `out` consists of 1s.
        let other = tf.ones_default(vec![]);
        op_fill_tensor_out(&self_, &other, &out);

        // PORT-NOTE: unlike `test_fill_scalar_out`, the C++ tensor variant does NOT
        // special-case Bool: `exp_out = tf.full(sizes, 1)` unconditionally, so the
        // fill value is `1` (= `true` for Bool). Uses `T::one()`, not `exp_value()`.
        let exp_out = tf.full(sizes, T::one(), TensorShapeDynamism::STATIC);

        assert!(tensors_are_close(&out, &exp_out, 0.0, Some(0.0)));
    }

    fn expect_bad_scalar_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let a = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        fill_scalar_out(&mut ctx, &a, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // TEST_FILL_OUT(FN, DTYPE): FN({}); FN({1}); FN({1,1,1}); FN({2,0,4}); FN({2,3,4}).
    fn run_scalar_input_support<T: Fill>() {
        test_fill_scalar_out::<T>(vec![]);
        test_fill_scalar_out::<T>(vec![1]);
        test_fill_scalar_out::<T>(vec![1, 1, 1]);
        test_fill_scalar_out::<T>(vec![2, 0, 4]);
        test_fill_scalar_out::<T>(vec![2, 3, 4]);
    }

    fn run_tensor_input_support<T: Fill>() {
        test_fill_tensor_out::<T>(vec![]);
        test_fill_tensor_out::<T>(vec![1]);
        test_fill_tensor_out::<T>(vec![1, 1, 1]);
        test_fill_tensor_out::<T>(vec![2, 0, 4]);
        test_fill_tensor_out::<T>(vec![2, 3, 4]);
    }

    // ET_FORALL_REALHBBF16_TYPES scalar input support.
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_byte_scalar_input_support() {
        run_scalar_input_support::<u8>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_char_scalar_input_support() {
        run_scalar_input_support::<i8>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_short_scalar_input_support() {
        run_scalar_input_support::<i16>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_int_scalar_input_support() {
        run_scalar_input_support::<i32>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_long_scalar_input_support() {
        run_scalar_input_support::<i64>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_bool_scalar_input_support() {
        run_scalar_input_support::<bool>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_half_scalar_input_support() {
        run_scalar_input_support::<Half>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_bfloat16_scalar_input_support() {
        run_scalar_input_support::<BFloat16>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_float_scalar_input_support() {
        run_scalar_input_support::<f32>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_double_scalar_input_support() {
        run_scalar_input_support::<f64>();
    }

    // ET_FORALL_REALHBBF16_TYPES tensor input support.
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_byte_tensor_input_support() {
        run_tensor_input_support::<u8>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_char_tensor_input_support() {
        run_tensor_input_support::<i8>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_short_tensor_input_support() {
        run_tensor_input_support::<i16>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_int_tensor_input_support() {
        run_tensor_input_support::<i32>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_long_tensor_input_support() {
        run_tensor_input_support::<i64>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_bool_tensor_input_support() {
        run_tensor_input_support::<bool>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_half_tensor_input_support() {
        run_tensor_input_support::<Half>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_bfloat16_tensor_input_support() {
        run_tensor_input_support::<BFloat16>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_float_tensor_input_support() {
        run_tensor_input_support::<f32>();
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_double_tensor_input_support() {
        run_tensor_input_support::<f64>();
    }

    // [spec:et:sem:op-fill.torch.executor.native.fill-tensor-out-fn/test]
    #[test]
    fn op_fill_test_mismatched_other_properties_dies() {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.zeros_default(vec![1]);
        let out = tf.zeros_default(vec![1]);

        let other1 = tf.zeros_default(vec![1]);
        assert_eq!(other1.dim(), 1);
        assert_eq!(other1.numel(), 1);

        let other2 = tf.zeros_default(vec![2]);
        assert_eq!(other2.dim(), 1);
        assert_eq!(other2.numel(), 2);

        let other3 = tf.zeros_default(vec![3, 3]);
        assert_eq!(other3.dim(), 2);
        assert_eq!(other3.numel(), 9);

        let mut ctx = context();
        fill_tensor_out(&mut ctx, &self_, &other1, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        fill_tensor_out(&mut ctx, &self_, &other2, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);

        let mut ctx = context();
        fill_tensor_out(&mut ctx, &self_, &other3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)` — non-aten branch, so the body runs.
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let self_ = tf.zeros_default(vec![1]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        fill_scalar_out(&mut ctx, &self_, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_mismatched_output_dtype_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_byte.zeros_default(vec![2, 2]);
        let out = tf_float.ones_default(vec![2, 2]);

        let mut ctx = context();
        fill_scalar_out(&mut ctx, &self_, &Scalar::from_double(0.0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // GENERATE_SCALAR_OVERFLOW_TESTS(OpFillTest).
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_byte_tensor_too_large_scalar_dies() {
        // Cannot be represented by a uint8_t.
        expect_bad_scalar_value_dies::<u8>(Scalar::from_i64(256));
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_char_tensor_too_small_scalar_dies() {
        // Cannot be represented by a int8_t.
        expect_bad_scalar_value_dies::<i8>(Scalar::from_i64(-129));
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_short_tensor_too_large_scalar_dies() {
        // Cannot be represented by a int16_t.
        expect_bad_scalar_value_dies::<i16>(Scalar::from_i64(32768));
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_float_tensor_too_small_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }
    // [spec:et:sem:op-fill.torch.executor.native.fill-scalar-out-fn/test]
    #[test]
    fn op_fill_test_float_tensor_too_large_scalar_dies() {
        // Cannot be represented by a float.
        expect_bad_scalar_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }
}
