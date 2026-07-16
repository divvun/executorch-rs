//! Literal port of kernels/portable/cpu/op_clamp.cpp.

use crate::kernels::portable::cpu::scalar_utils::{
    get_scalar_dtype, promote_type_with_scalar, scalar_to,
};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size_3;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_tritensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::math_util::{max_override, min_override};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    can_cast, is_floating_type, is_integral_type, promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order4,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `is_out_of_bounds<CTYPE_OUT, CTYPE_CAST>` compares the already-held
// value against `numeric_limits<CTYPE_OUT>::lowest()/max()` in CTYPE_CAST. The
// C++ template pair (CTYPE_OUT integral / CTYPE_CAST=int64_t) and
// (CTYPE_OUT floating / CTYPE_CAST=double) are the only instantiations used;
// modeled as two traits carrying the per-CTYPE_OUT bounds cast to i64 / f64.
trait BoundsI64 {
    fn lowest_i64() -> i64;
    fn max_i64() -> i64;
}
trait BoundsF64 {
    fn lowest_f64() -> f64;
    fn max_f64() -> f64;
}

macro_rules! impl_bounds_i64 {
    ($t:ty) => {
        impl BoundsI64 for $t {
            fn lowest_i64() -> i64 {
                <$t>::MIN as i64
            }
            fn max_i64() -> i64 {
                <$t>::MAX as i64
            }
        }
    };
}
impl_bounds_i64!(u8);
impl_bounds_i64!(i8);
impl_bounds_i64!(i16);
impl_bounds_i64!(i32);
impl_bounds_i64!(i64);

macro_rules! impl_bounds_f64_native {
    ($t:ty) => {
        impl BoundsF64 for $t {
            fn lowest_f64() -> f64 {
                // numeric_limits<T>::lowest() is the most negative finite value.
                -(<$t>::MAX) as f64
            }
            fn max_f64() -> f64 {
                <$t>::MAX as f64
            }
        }
    };
}
impl_bounds_f64_native!(f32);
impl_bounds_f64_native!(f64);
impl BoundsF64 for Half {
    fn lowest_f64() -> f64 {
        (-Half::MAX).to_f64()
    }
    fn max_f64() -> f64 {
        Half::MAX.to_f64()
    }
}
impl BoundsF64 for BFloat16 {
    fn lowest_f64() -> f64 {
        (-BFloat16::MAX).to_f64()
    }
    fn max_f64() -> f64 {
        BFloat16::MAX.to_f64()
    }
}

/// Check if val, when cast to CTYPE_CAST, is not in the range of CTYPE_OUT
// [spec:et:def:op-clamp.torch.executor.native.is-out-of-bounds-fn]
// [spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn]
fn is_out_of_bounds_i64<CTYPE_OUT: BoundsI64>(val_cast: i64) -> bool {
    val_cast < CTYPE_OUT::lowest_i64() || val_cast > CTYPE_OUT::max_i64()
}
fn is_out_of_bounds_f64<CTYPE_OUT: BoundsF64>(val_cast: f64) -> bool {
    val_cast < CTYPE_OUT::lowest_f64() || val_cast > CTYPE_OUT::max_f64()
}

// [spec:et:def:op-clamp.torch.executor.native.check-bounds-fn]
// [spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn]
#[must_use]
fn check_bounds(
    ctx: &mut KernelRuntimeContext,
    val_scalar: Scalar,
    _val_type: ScalarType,
    out_type: ScalarType,
    val_name: &str,
) -> bool {
    let mut is_valid = true;

    let op_name = "clamp.out";

    if is_integral_type(out_type, /*includeBool=*/ false) {
        let val_long: i64 = scalar_to::<i64>(&val_scalar);
        crate::et_switch_int_types!(out_type, ctx, op_name, CTYPE_OUT, {
            if is_out_of_bounds_i64::<CTYPE_OUT>(val_long) {
                crate::et_log!(Error, "{} value out of bounds", val_name);
                is_valid = false;
            }
        });
    } else if is_floating_type(out_type) {
        crate::et_switch_floathbf16_types!(out_type, ctx, op_name, CTYPE_OUT, {
            let val_double: f64 = scalar_to::<f64>(&val_scalar);
            if val_double.is_finite() && is_out_of_bounds_f64::<CTYPE_OUT>(val_double) {
                crate::et_log!(Error, "{} value out of bounds", val_name);
                is_valid = false;
            }
        });
    }

    is_valid
}

// [spec:et:def:op-clamp.torch.executor.native.clamp-out-fn]
// [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn]
#[executorch_macros::et_kernel("aten::clamp.out")]
pub fn clamp_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    min_opt: Option<Scalar>,
    max_opt: Option<Scalar>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let has_min = min_opt.is_some();
    let has_max = max_opt.is_some();

    crate::et_kernel_check_msg!(
        ctx,
        has_min || has_max,
        InvalidArgument,
        out,
        "At least one of 'min' or 'max' must not be None"
    );

    // Input Dtypes
    let in_type = in_.scalar_type();
    let min_type = if has_min {
        get_scalar_dtype(min_opt.unwrap())
    } else {
        in_type
    };
    let max_type = if has_max {
        get_scalar_dtype(max_opt.unwrap())
    } else {
        in_type
    };
    let out_type = out.scalar_type();

    // Common Dtype
    let mut common_type = in_type;
    if has_min {
        common_type = promote_type_with_scalar(common_type, min_opt.unwrap(), false);
    }
    if has_max {
        common_type = promote_type_with_scalar(common_type, max_opt.unwrap(), false);
    }

    // Check Common Dtype
    crate::et_kernel_check!(ctx, common_type == out_type, InvalidArgument, out);

    // Check Scalar Bounds
    if has_min {
        crate::et_kernel_check!(
            ctx,
            check_bounds(ctx, min_opt.unwrap(), min_type, out_type, "minimum"),
            InvalidArgument,
            out
        );
    }
    if has_max {
        crate::et_kernel_check!(
            ctx,
            check_bounds(ctx, max_opt.unwrap(), max_type, out_type, "maximum"),
            InvalidArgument,
            out
        );
    }

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    let op_name = "clamp.out";

    crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                let val_in = vals[0];
                let mut val_out = val_in;
                if has_min {
                    val_out = max_override(val_out, scalar_to::<CTYPE_COMPUTE>(&min_opt.unwrap()));
                }
                if has_max {
                    val_out = min_override(val_out, scalar_to::<CTYPE_COMPUTE>(&max_opt.unwrap()));
                }
                val_out
            },
            ctx,
            in_,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            /*support_noncontiguous*/ false,
        );
    });

    out
}

// [spec:et:def:op-clamp.torch.executor.native.clamp-tensor-out-fn]
// [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn]
pub fn clamp_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    min_opt: Option<&Tensor>,
    max_opt: Option<&Tensor>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let has_min = min_opt.is_some();
    let has_max = max_opt.is_some();

    crate::et_kernel_check_msg!(
        ctx,
        has_min || has_max,
        InvalidArgument,
        out,
        "At least one of 'min' or 'max' must not be None"
    );

    let min: &Tensor = if has_min { min_opt.unwrap() } else { in_ };
    let max: &Tensor = if has_max { max_opt.unwrap() } else { in_ };

    // Common Dtype
    let mut common_type = in_.scalar_type();
    if has_min {
        common_type = promote_types(common_type, min.scalar_type(), false);
    }
    if has_max {
        common_type = promote_types(common_type, max.scalar_type(), false);
    }

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()),
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order4(in_, min, max, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size_3(in_, min, max, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    let op_name = "clamp.Tensor_out";

    crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_tritensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                let val_in = vals[0];
                let val_min = vals[1];
                let val_max = vals[2];
                // TODO: rewrite this to be vectorization-capable.
                let mut val_out = val_in;
                if has_min {
                    val_out = max_override(val_out, val_min);
                }
                if has_max {
                    val_out = min_override(val_out, val_max);
                }
                val_out
            },
            ctx,
            in_,
            SupportedTensorDtypes::REALHBBF16,
            min,
            SupportedTensorDtypes::REALHBBF16,
            max,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBBF16,
            /*support_noncontiguous*/ false,
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
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor_impl::SizesType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! et_expect_kernel_failure {
        ($ctx:expr, $stmt:expr) => {{
            let _ = $stmt;
            assert_ne!(
                $ctx.failure_state(),
                Error::Ok,
                "Expected kernel failure but found success."
            );
        }};
    }

    // Element conversion for typed tensor data. Integer dtypes truncate; the
    // integer/floating test data only ever holds representable values.
    trait Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_elem_int {
        ($($t:ty),*) => {$(impl Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_elem_int!(u8, i8, i16, i32, i64);
    impl Elem for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl Elem for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl Elem for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    struct ClampTestCase {
        _title: &'static str,
        sizes: Vec<SizesType>,
        input_data: Vec<f64>,
        min: Option<Scalar>,
        max: Option<Scalar>,
        expected_data: Vec<f64>,
    }

    fn run_test_cases<T>(test_cases: Vec<ClampTestCase>)
    where
        T: CppTypeToScalarType + FactoryValue + Elem,
    {
        let tf = TensorFactory::<T>::new();
        for test_case in test_cases {
            let in_ = tf.make_default(
                test_case.sizes.clone(),
                test_case
                    .input_data
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect(),
            );
            let out = tf.zeros_default(test_case.sizes.clone());
            let mut ctx = context();
            let ret = clamp_out(&mut ctx, &in_, test_case.min, test_case.max, &out);
            assert_tensor_eq!(out, *ret);

            let expected = tf.make_default(
                test_case.sizes.clone(),
                test_case
                    .expected_data
                    .iter()
                    .map(|&x| T::from_f64(x))
                    .collect(),
            );
            assert_tensor_eq!(out, expected);
        }
    }

    fn opt_i(v: i64) -> Option<Scalar> {
        Some(Scalar::from_i64(v))
    }
    fn opt_d(v: f64) -> Option<Scalar> {
        Some(Scalar::from_double(v))
    }

    fn arange(stop: usize) -> Vec<f64> {
        (0..stop).map(|i| i as f64).collect()
    }

    fn run_unsigned_integer_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + Elem,
    {
        let test_cases = vec![
            ClampTestCase {
                _title: "Simple clamp",
                sizes: vec![2, 2],
                input_data: vec![0., 1., 10., 100.],
                min: opt_i(1),
                max: opt_i(6),
                expected_data: vec![1., 1., 6., 6.],
            },
            ClampTestCase {
                _title: "No max",
                sizes: vec![2, 2],
                input_data: vec![0., 1., 10., 100.],
                min: opt_i(1),
                max: None,
                expected_data: vec![1., 1., 10., 100.],
            },
            ClampTestCase {
                _title: "No min",
                sizes: vec![2, 2],
                input_data: vec![0., 1., 10., 100.],
                min: None,
                max: opt_i(6),
                expected_data: vec![0., 1., 6., 6.],
            },
            ClampTestCase {
                _title: "min > max",
                sizes: vec![2, 2],
                input_data: vec![0., 1., 10., 100.],
                min: opt_i(10),
                max: opt_i(6),
                expected_data: vec![6., 6., 6., 6.],
            },
            ClampTestCase {
                _title: "Simple clamp larger data",
                sizes: vec![18],
                input_data: arange(18),
                min: opt_i(1),
                max: opt_i(6),
                expected_data: vec![
                    1., 1., 2., 3., 4., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
                ],
            },
        ];
        run_test_cases::<T>(test_cases);
    }

    fn run_signed_integer_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + Elem,
    {
        let test_cases = vec![
            ClampTestCase {
                _title: "Simple negative/positive clamp",
                sizes: vec![2, 2],
                input_data: vec![-10., -1., 1., 10.],
                min: opt_i(-5),
                max: opt_i(5),
                expected_data: vec![-5., -1., 1., 5.],
            },
            ClampTestCase {
                _title: "Simple negative-only clamp",
                sizes: vec![2, 2],
                input_data: vec![-10., -5., 1., 10.],
                min: opt_i(-6),
                max: opt_i(-1),
                expected_data: vec![-6., -5., -1., -1.],
            },
        ];
        run_test_cases::<T>(test_cases);
    }

    // PORT-NOTE: the C++ helper computes `kInfinity` from the element type and
    // wraps it in `OptScalar`. `Scalar` only stores i64/double/bool, so the
    // infinities are materialized as double scalars (matching the C++ conversion
    // through `opt_infinity_type` -> `OptScalar(double)`).
    fn run_floating_point_test_cases<T>()
    where
        T: CppTypeToScalarType + FactoryValue + Elem,
    {
        let k_infinity = f64::INFINITY;
        let test_cases = vec![
            ClampTestCase {
                _title: "Simple negative/positive clamp",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -1.1, 1.1, 10.1],
                min: opt_d(-5.5),
                max: opt_d(5.5),
                expected_data: vec![-5.5, -1.1, 1.1, 5.5],
            },
            ClampTestCase {
                _title: "Simple negative-only clamp",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -5.5, 1.1, 10.1],
                min: opt_d(-6.6),
                max: opt_d(-1.1),
                expected_data: vec![-6.6, -5.5, -1.1, -1.1],
            },
            ClampTestCase {
                _title: "Infinities are clamped",
                sizes: vec![2, 2],
                input_data: vec![-k_infinity, -1.1, 1.1, k_infinity],
                min: opt_d(-5.5),
                max: opt_d(5.5),
                expected_data: vec![-5.5, -1.1, 1.1, 5.5],
            },
            ClampTestCase {
                _title: "Infinite min",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -1.1, 1.1, 10.1],
                min: opt_d(-k_infinity),
                max: opt_d(5.5),
                expected_data: vec![-10.1, -1.1, 1.1, 5.5],
            },
            ClampTestCase {
                _title: "Infinite max",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -1.1, 1.1, 10.1],
                min: opt_d(-5.5),
                max: opt_d(k_infinity),
                expected_data: vec![-5.5, -1.1, 1.1, 10.1],
            },
            ClampTestCase {
                _title: "NaN entries preserved",
                sizes: vec![2, 2],
                input_data: vec![-10.1, f64::NAN, f64::NAN, 10.1],
                min: opt_d(0.0),
                max: opt_d(0.0),
                expected_data: vec![0.0, f64::NAN, f64::NAN, 0.0],
            },
            ClampTestCase {
                _title: "NaN min produces all NaN output",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -1.1, 1.1, 10.1],
                min: opt_d(f64::NAN),
                max: opt_d(5.5),
                expected_data: vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            },
            ClampTestCase {
                _title: "NaN max produces all NaN output",
                sizes: vec![2, 2],
                input_data: vec![-10.1, -1.1, 1.1, 10.1],
                min: opt_d(-5.5),
                max: opt_d(f64::NAN),
                expected_data: vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            },
        ];
        run_test_cases::<T>(test_cases);
    }

    fn expect_bad_clamp_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue + Elem,
    {
        let tf = TensorFactory::<T>::new();
        let in_ = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, clamp_out(&mut ctx, &in_, Some(bad_value), None, &out));
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, clamp_out(&mut ctx, &in_, None, Some(bad_value), &out));
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            clamp_out(&mut ctx, &in_, Some(bad_value), Some(bad_value), &out)
        );
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_byte_tensors() {
        run_unsigned_integer_test_cases::<u8>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_char_tensors() {
        run_unsigned_integer_test_cases::<i8>();
        run_signed_integer_test_cases::<i8>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_short_tensors() {
        run_unsigned_integer_test_cases::<i16>();
        run_signed_integer_test_cases::<i16>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_int_tensors() {
        run_unsigned_integer_test_cases::<i32>();
        run_signed_integer_test_cases::<i32>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_long_tensors() {
        run_unsigned_integer_test_cases::<i64>();
        run_signed_integer_test_cases::<i64>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_half_tensors() {
        run_unsigned_integer_test_cases::<Half>();
        run_signed_integer_test_cases::<Half>();
        run_floating_point_test_cases::<Half>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_float_tensors() {
        run_unsigned_integer_test_cases::<f32>();
        run_signed_integer_test_cases::<f32>();
        run_floating_point_test_cases::<f32>();
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_double_tensors() {
        run_unsigned_integer_test_cases::<f64>();
        run_signed_integer_test_cases::<f64>();
        run_floating_point_test_cases::<f64>();
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // Out-of-bounds integer clamp value drives check_bounds -> is_out_of_bounds_i64
    // to report failure; the kernel error would not fire if either were wrong.
    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn/test]
    #[test]
    fn op_clamp_out_test_byte_tensor_negative_clamp_dies() {
        expect_bad_clamp_value_dies::<u8>(Scalar::from_i64(-1));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn/test]
    #[test]
    fn op_clamp_out_test_byte_tensor_too_large_clamp_dies() {
        expect_bad_clamp_value_dies::<u8>(Scalar::from_i64(256));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_byte_tensor_floating_point_clamp_dies() {
        expect_bad_clamp_value_dies::<u8>(Scalar::from_double(2.2));
    }

    // PORT-NOTE: guarded by `#ifndef USE_ATEN_LIB`; this is the non-ATen build.
    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_int_tensor_too_small_clamp_dies() {
        expect_bad_clamp_value_dies::<i32>(Scalar::from_i64(-2147483649));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_int_tensor_too_large_clamp_dies() {
        expect_bad_clamp_value_dies::<i32>(Scalar::from_i64(2147483648));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_int_tensor_floating_point_clamp_dies() {
        expect_bad_clamp_value_dies::<i32>(Scalar::from_double(2.2));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_float_tensor_too_small_clamp_dies() {
        expect_bad_clamp_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }

    // Out-of-range float clamp value drives check_bounds -> is_out_of_bounds_f64.
    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn/test]
    // [spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn/test]
    #[test]
    fn op_clamp_out_test_float_tensor_too_large_clamp_dies() {
        expect_bad_clamp_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let y = opt_d(-0.5);
        let z = opt_d(0.5);
        let expected_result = tf.make_default(vec![10, 10], vec![0.5f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        let _ret = clamp_out(&mut ctx, &x, y, z, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    fn op_clamp_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6984410881996155,
                0.5675464272499084,
                0.8352431654930115,
                0.2055988311767578,
                0.593172013759613,
                0.11234724521636963,
            ],
        );
        let y = opt_d(-0.5);
        let z = opt_d(0.5);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = clamp_out(&mut ctx, &x, y, z, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED in C++ (dynamic shape not supported). Ported + ignored.
    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUpperBoundLargerThanExpected: dynamic shape not supported"]
    fn op_clamp_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6984410881996155,
                0.5675464272499084,
                0.8352431654930115,
                0.2055988311767578,
                0.593172013759613,
                0.11234724521636963,
            ],
        );
        let y = opt_d(-0.5);
        let z = opt_d(0.5);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963],
        );
        let out = tf.zeros(vec![6, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        let _ret = clamp_out(&mut ctx, &x, y, z, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: DISABLED in C++ (dynamic shape not supported). Ported + ignored.
    // [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape not supported"]
    fn op_clamp_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6984410881996155,
                0.5675464272499084,
                0.8352431654930115,
                0.2055988311767578,
                0.593172013759613,
                0.11234724521636963,
            ],
        );
        let y = opt_d(-0.5);
        let z = opt_d(0.5);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963],
        );
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        let mut ctx = context();
        let _ret = clamp_out(&mut ctx, &x, y, z, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn/test]
    #[test]
    fn op_clamp_tensor_out_test_smoke_test() {
        let tf_in = TensorFactory::<u8>::new();
        let tf_min = TensorFactory::<i32>::new();
        let tf_max = TensorFactory::<i8>::new();
        let tf_out = TensorFactory::<i16>::new();

        let in_ = tf_in.make_default(vec![1, 1], vec![3]);
        let min = tf_min.make_default(vec![1, 3], vec![0, 1, 4]);
        let max = tf_max.make_default(vec![2, 1], vec![2, 5]);
        let out = tf_out.zeros_default(vec![2, 3]);
        let expected = tf_out.make_default(vec![2, 3], vec![2, 2, 2, 3, 3, 4]);

        let mut ctx = context();
        clamp_tensor_out(&mut ctx, &in_, Some(&min), Some(&max), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn/test]
    #[test]
    fn op_clamp_tensor_out_test_downcasting_smoke_test() {
        let tf_in = TensorFactory::<u8>::new();
        let tf_min = TensorFactory::<i16>::new();
        let tf_max = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<i8>::new();

        let in_ = tf_in.make_default(vec![], vec![5]);
        let min = tf_min.make_default(vec![], vec![-129]);
        let max = tf_max.make_default(vec![], vec![300]);
        let out = tf_out.zeros_default(vec![]);
        let expected = tf_out.make_default(vec![], vec![5]);

        let mut ctx = context();
        clamp_tensor_out(&mut ctx, &in_, Some(&min), Some(&max), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn/test]
    #[test]
    fn op_clamp_tensor_out_test_downcasting_smoke_test2() {
        let tf_in = TensorFactory::<i16>::new();
        let tf_min = TensorFactory::<i16>::new();
        let tf_max = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<i8>::new();

        let in_ = tf_in.make_default(vec![], vec![301]);
        let min = tf_min.make_default(vec![], vec![-129]);
        let max = tf_max.make_default(vec![], vec![300]);
        let out = tf_out.zeros_default(vec![]);
        let expected = tf_out.make_default(vec![], vec![44]);

        let mut ctx = context();
        clamp_tensor_out(&mut ctx, &in_, Some(&min), Some(&max), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn/test]
    #[test]
    fn op_clamp_tensor_out_test_downcasting_smoke_test3() {
        let tf_in = TensorFactory::<i16>::new();
        let tf_min = TensorFactory::<i16>::new();
        let tf_max = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<i8>::new();

        let in_ = tf_in.make_default(vec![], vec![45]);
        let min = tf_min.make_default(vec![], vec![-129]);
        let max = tf_max.make_default(vec![], vec![300]);
        let out = tf_out.zeros_default(vec![]);
        let expected = tf_out.make_default(vec![], vec![45]);

        let mut ctx = context();
        clamp_tensor_out(&mut ctx, &in_, Some(&min), Some(&max), &out);
        assert_tensor_eq!(out, expected);
    }
}
