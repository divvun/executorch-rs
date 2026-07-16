//! Literal port of kernels/portable/cpu/op_div.cpp.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
// PORT-NOTE: `div_out_mode`/`div_scalar_mode_out` dispatch over REAL =
// {Byte, Char, Short, Int, Long, Float, Double} and call `utils::floor_divide`
// on the compute type, including `Byte` (u8). As translated, math_util.rs only
// impls `FloorDivide` for i8/i16/i32/i64/f32/f64 — NOT u8 — so the Byte arm's
// `floor_divide` does not resolve. The C++ integral `floor_divide` template
// accepts `unsigned char`; this is a math_util omission. Unresolved
// cross-module reference — noted for the fixer; kept literal here.
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::math_util::floor_divide;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    CppTypeToScalarType, can_cast, is_complex_type, is_floating_type, is_integral_type,
    promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat, ComplexHalf, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: div_out's complex path does `a_data[i] / b_data[i]` over
// ComplexHalf/ComplexFloat/ComplexDouble. The portable_type `Complex<T>` struct
// carries no arithmetic, and there is no `ET_SWITCH_COMPLEX_TYPES` macro yet, so
// both are supplied locally here (in-file, not a portable_type redesign): a
// `ComplexDiv` trait implementing the standard complex division formula and an
// `et_switch_complex_types!`-style match mirroring `et_internal_switch!`.
trait ComplexDiv: Copy {
    fn cdiv(self, b: Self) -> Self;
}
macro_rules! impl_complex_div {
    ($t:ty, $c:ty) => {
        impl ComplexDiv for $t {
            fn cdiv(self, b: Self) -> Self {
                let ar = self.real as f64;
                let ai = self.imag as f64;
                let br = b.real as f64;
                let bi = b.imag as f64;
                let denom = br * br + bi * bi;
                Self {
                    real: ((ar * br + ai * bi) / denom) as $c,
                    imag: ((ai * br - ar * bi) / denom) as $c,
                }
            }
        }
    };
}
impl_complex_div!(ComplexFloat, f32);
impl_complex_div!(ComplexDouble, f64);
impl ComplexDiv for ComplexHalf {
    fn cdiv(self, b: Self) -> Self {
        let ar = self.real.to_f64();
        let ai = self.imag.to_f64();
        let br = b.real.to_f64();
        let bi = b.imag.to_f64();
        let denom = br * br + bi * bi;
        ComplexHalf {
            real: Half::from_f64((ar * br + ai * bi) / denom),
            imag: Half::from_f64((ai * br - ar * bi) / denom),
        }
    }
}

// PORT-NOTE: for div_out_mode/div_scalar_mode the compute set is REAL (integers
// + Float/Double). The closure needs the compile-time `is_integral<CTYPE>::value`
// test, a zero literal, and `std::trunc`. Modeled as a `RealCompute` trait so
// the generic closure carries each per-type body verbatim.
trait RealCompute: Copy + CppTypeToScalarType {
    const IS_INTEGRAL: bool;
    fn is_zero(self) -> bool;
    fn zero() -> Self;
    fn trunc_(self) -> Self;
    fn div_(self, b: Self) -> Self;
}
macro_rules! impl_real_compute_int {
    ($t:ty) => {
        impl RealCompute for $t {
            const IS_INTEGRAL: bool = true;
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
            fn zero() -> Self {
                0 as $t
            }
            fn trunc_(self) -> Self {
                // std::trunc promotes an integer to double and truncates: no-op.
                self
            }
            fn div_(self, b: Self) -> Self {
                self / b
            }
        }
    };
}
impl_real_compute_int!(u8);
impl_real_compute_int!(i8);
impl_real_compute_int!(i16);
impl_real_compute_int!(i32);
impl_real_compute_int!(i64);
macro_rules! impl_real_compute_float {
    ($t:ty) => {
        impl RealCompute for $t {
            const IS_INTEGRAL: bool = false;
            fn is_zero(self) -> bool {
                self == 0 as $t
            }
            fn zero() -> Self {
                0 as $t
            }
            fn trunc_(self) -> Self {
                self.trunc()
            }
            fn div_(self, b: Self) -> Self {
                self / b
            }
        }
    };
}
impl_real_compute_float!(f32);
impl_real_compute_float!(f64);

// [spec:et:def:op-div.torch.executor.native.get-common-type-fn]
// [spec:et:sem:op-div.torch.executor.native.get-common-type-fn]
fn get_common_type(a_type: ScalarType, b_type: ScalarType) -> ScalarType {
    if is_complex_type(a_type) || is_complex_type(b_type) {
        promote_types(a_type, b_type, false)
    } else if is_floating_type(a_type) && is_floating_type(b_type) {
        promote_types(a_type, b_type, false)
    } else if is_floating_type(a_type) {
        a_type
    } else if is_floating_type(b_type) {
        b_type
    } else {
        ScalarType::Float
    }
}

// PORT-NOTE: local `ET_SWITCH_COMPLEX_TYPES` equivalent (only used by div_out).
// Mirrors `et_internal_switch!` restricted to the three complex arms.
macro_rules! et_switch_complex_types {
    ($type:expr, $ctx:expr, $name:expr, $ctype_alias:ident, $body:block) => {{
        let _st = $type;
        match _st {
            ScalarType::ComplexHalf => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = ComplexHalf;
                $body
            }
            ScalarType::ComplexFloat => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = ComplexFloat;
                $body
            }
            ScalarType::ComplexDouble => {
                #[allow(non_camel_case_types, dead_code)]
                type $ctype_alias = ComplexDouble;
                $body
            }
            _ => {
                $ctx.fail(Error::InvalidArgument);
                crate::et_log!(
                    Error,
                    "Unhandled dtype {} for {}",
                    crate::runtime::core::exec_aten::util::scalar_type_util::to_string(_st),
                    $name
                );
            }
        }
    }};
}

// [spec:et:def:op-div.torch.executor.native.div-out-fn]
// [spec:et:sem:op-div.torch.executor.native.div-out-fn]
pub fn div_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type = get_common_type(a.scalar_type(), b.scalar_type());

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    let op_name = "div.out";

    if is_complex_type(common_type) {
        et_switch_complex_types!(common_type, ctx, op_name, CTYPE, {
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            let out_data = out.mutable_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).cdiv(*b_data.offset(i));
                }
            }
        });
    } else {
        // Compute Dtype for real types
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);
        crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| {
                    let val_a = vals[0];
                    let val_b = vals[1];
                    val_a / val_b
                },
                ctx,
                a,
                SupportedTensorDtypes::REALHBBF16,
                b,
                SupportedTensorDtypes::REALHBBF16,
                out,
                SupportedTensorDtypes::FLOATHBF16,
                /*support_noncontiguous*/ false,
            );
        });
    }

    out
}

// [spec:et:def:op-div.torch.executor.native.div-out-mode-fn]
// [spec:et:sem:op-div.torch.executor.native.div-out-mode-fn]
pub fn div_out_mode<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    mode: Option<&str>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    if mode.is_none() {
        return div_out(ctx, a, b, out);
    }

    let mode_val = mode.unwrap();

    // Check mode
    crate::et_kernel_check!(
        ctx,
        mode_val == "trunc" || mode_val == "floor",
        InvalidArgument,
        out
    );

    // Common Dtype
    let mut common_type = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && common_type != ScalarType::Bool,
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    let op_name = "div.out_mode";

    let mode_is_trunc = mode_val == "trunc";
    // PORT-NOTE: the C++ lambda captures `&div_by_zero_error` and writes through
    // the reference while the closure object itself stays const (`Fn`). A `Cell`
    // reproduces that interior mutation without widening the util's `Fn` bound.
    let div_by_zero_error = core::cell::Cell::new(false);

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE {
                let val_a: CTYPE_COMPUTE = vals[0];
                let val_b: CTYPE_COMPUTE = vals[1];
                // TODO: rewrite this to be vectorization-capable.
                if <CTYPE_COMPUTE as RealCompute>::IS_INTEGRAL {
                    if val_b.is_zero() {
                        div_by_zero_error.set(true);
                        return <CTYPE_COMPUTE as RealCompute>::zero();
                    }
                }
                let mut value = val_a.div_(val_b);
                if mode_is_trunc {
                    value = value.trunc_();
                } else {
                    // We established above that the mode is either trunc or
                    // floor, so it must be floor.
                    value = floor_divide(val_a, val_b);
                }
                value
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
            /*support_noncontiguous*/ false,
        );
    });

    crate::et_kernel_check_msg!(
        ctx,
        !div_by_zero_error.get(),
        InvalidArgument,
        out,
        "Div mode operation encountered integer division by zero"
    );

    out
}

// [spec:et:def:op-div.torch.executor.native.div-scalar-out-fn]
// [spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn]
pub fn div_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type = if is_floating_type(a.scalar_type()) {
        a.scalar_type()
    } else {
        ScalarType::Float
    };

    // Check Common Dtype
    crate::et_kernel_check!(ctx, common_type == out.scalar_type(), InvalidArgument, out);

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    let op_name = "div.Scalar_out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                let val_a = vals[0];
                val_a / val_b
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            /*support_noncontiguous*/ false,
        );
    });

    out
}

// [spec:et:def:op-div.torch.executor.native.div-scalar-mode-out-fn]
// [spec:et:sem:op-div.torch.executor.native.div-scalar-mode-out-fn]
pub fn div_scalar_mode_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    mode: Option<&str>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    if mode.is_none() {
        return div_scalar_out(ctx, a, b, out);
    }

    let mode_val = mode.unwrap();

    // Check mode
    crate::et_kernel_check!(
        ctx,
        mode_val == "trunc" || mode_val == "floor",
        InvalidArgument,
        out
    );

    // Common Dtype
    let mut common_type = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()) && common_type != ScalarType::Bool,
        InvalidArgument,
        out
    );

    // Check for intergral division by zero
    crate::et_kernel_check_msg!(
        ctx,
        !(is_integral_type(common_type, true) && scalar_to::<f64>(b) == 0.0),
        InvalidArgument,
        out,
        "Div mode operation encountered integer division by zero"
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    let mode_is_trunc = mode_val == "trunc";

    let op_name = "div.Scalar_mode_out";

    crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                let val_a = vals[0];
                let mut value = val_a.div_(val_b);
                if mode_is_trunc {
                    value = value.trunc_();
                } else {
                    value = floor_divide(val_a, val_b);
                }
                value
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBF16,
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
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Complex};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_close_with_tol, assert_tensor_eq};

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

    fn op_div_out<'a, 'b>(a: &Tensor, b: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        div_out(&mut ctx, a, b, out)
    }

    fn op_div_scalar_out<'a, 'b>(a: &Tensor, b: &Scalar, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        div_scalar_out(&mut ctx, a, b, out)
    }

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

    fn test_div<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64,
        B: CppTypeToScalarType + FactoryValue + FromF64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        // Valid input should give the expected output
        op_div_out(
            &tf_a.make_default(
                sizes.clone(),
                vec![
                    A::from_f64(1.0),
                    A::from_f64(2.0),
                    A::from_f64(4.0),
                    A::from_f64(8.0),
                ],
            ),
            &tf_b.make_default(
                sizes.clone(),
                vec![
                    B::from_f64(8.0),
                    B::from_f64(4.0),
                    B::from_f64(2.0),
                    B::from_f64(1.0),
                ],
            ),
            &out,
        );

        assert_tensor_close!(
            out,
            tf_out.make_default(
                sizes,
                vec![
                    OUT::from_f64(0.125),
                    OUT::from_f64(0.5),
                    OUT::from_f64(2.0),
                    OUT::from_f64(8.0)
                ]
            )
        );
    }

    // template <> test_div<Float, Float, Float>
    fn test_div_float_float_float() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 5];

        let out = tf.zeros_default(sizes.clone());

        op_div_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    1.0,
                    2.0,
                    4.0,
                    8.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    1.0,
                    1.0,
                    1.0,
                ],
            ),
            &tf.make_default(
                sizes.clone(),
                vec![
                    8.0,
                    0.0,
                    2.0,
                    1.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                ],
            ),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(
                sizes,
                vec![
                    0.125,
                    f32::INFINITY,
                    2.0,
                    8.0,
                    f32::NAN,
                    f32::NAN,
                    f32::NAN,
                    0.0,
                    0.0,
                    f32::NAN,
                ]
            )
        );
    }

    // template <> test_div<Bool, Float, Float>
    fn test_div_bool_float_float() {
        let tf_b = TensorFactory::<bool>::new();
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        op_div_out(
            &tf_b.make_default(sizes.clone(), vec![true, true, true, true]),
            &tf.make_default(sizes.clone(), vec![4.0, 4.0, 2.0, 1.0]),
            &out,
        );

        assert_tensor_close!(out, tf.make_default(sizes, vec![0.25, 0.25, 0.5, 1.0]));
    }

    // PORT-NOTE: the `test_div<Float,Float,Float>` and `test_div<Bool,Float,Float>`
    // template specializations override the generic body. C++ reaches
    // `test_div<Float,Float,Float>` through the a/b/out enumeration
    // (`test_div_enumerate_b_types<Float>` → `..._out_types<Float,Float>` →
    // `test_div<Float,Float,Float>`), which resolves to the specialization. Here
    // the generic `test_div_enumerate_*` fold that dispatch into the `_spec`
    // helpers below, branching on the type ids to select the specialized body.
    fn test_div_specialized<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64,
        B: CppTypeToScalarType + FactoryValue + FromF64,
        OUT: CppTypeToScalarType + FactoryValue + FromF64,
    {
        if A::VALUE == ScalarType::Float
            && B::VALUE == ScalarType::Float
            && OUT::VALUE == ScalarType::Float
        {
            test_div_float_float_float();
        } else {
            test_div::<A, B, OUT>();
        }
    }

    fn test_div_enumerate_out_types_spec<A, B>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64,
        B: CppTypeToScalarType + FactoryValue + FromF64,
    {
        test_div_specialized::<A, B, f32>();
        test_div_specialized::<A, B, f64>();
        test_div_specialized::<A, B, Half>();
        test_div_specialized::<A, B, BFloat16>();
    }

    fn test_div_enumerate_b_types_spec<A>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64,
    {
        test_div_enumerate_out_types_spec::<A, u8>();
        test_div_enumerate_out_types_spec::<A, i8>();
        test_div_enumerate_out_types_spec::<A, i16>();
        test_div_enumerate_out_types_spec::<A, i32>();
        test_div_enumerate_out_types_spec::<A, i64>();
        test_div_enumerate_out_types_spec::<A, f32>();
        test_div_enumerate_out_types_spec::<A, f64>();
        test_div_enumerate_out_types_spec::<A, Half>();
        test_div_enumerate_out_types_spec::<A, BFloat16>();
    }

    fn test_div_invalid_output_dtype_dies<OUT>()
    where
        OUT: CppTypeToScalarType + FactoryValue,
    {
        let tf_float = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let sizes = vec![2, 5];
        let a = tf_float.ones_default(sizes.clone());
        let b = tf_float.ones_default(sizes.clone());
        let out = tf_out.zeros_default(sizes);
        let mut ctx = context();
        div_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn test_broadcast_3d<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf_a = TensorFactory::<T>::new();

        let a = tf_a.make_default(
            vec![2, 2, 3],
            (1..=12).map(|v| T::from_f64(v as f64)).collect(),
        );
        let b = tf_a.make_default(
            vec![2, 1, 3],
            vec![
                T::from_f64(2.0),
                T::from_f64(3.0),
                T::from_f64(4.0),
                T::from_f64(5.0),
                T::from_f64(6.0),
                T::from_f64(7.0),
            ],
        );

        let out = tf_a.make_default(
            vec![2, 2, 3],
            (1..=12).map(|v| T::from_f64(v as f64)).collect(),
        );
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            vec![
                T::from_f64(0.5000),
                T::from_f64(0.6667),
                T::from_f64(0.75002),
                T::from_f64(2.0000),
                T::from_f64(1.6667),
                T::from_f64(1.5000),
                T::from_f64(1.4000),
                T::from_f64(1.3333),
                T::from_f64(1.2857),
                T::from_f64(2.0000),
                T::from_f64(1.8333),
                T::from_f64(1.7143),
            ],
        );
        assert_tensor_close_with_tol!(op_div_out(&a, &b, &out), expected, 1e-4, 1e-4);
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            vec![
                T::from_f64(2.0000),
                T::from_f64(1.5000),
                T::from_f64(1.3333),
                T::from_f64(0.5000),
                T::from_f64(0.6000),
                T::from_f64(0.6667),
                T::from_f64(0.7143),
                T::from_f64(0.7500),
                T::from_f64(0.7778),
                T::from_f64(0.5000),
                T::from_f64(0.5455),
                T::from_f64(0.5833),
            ],
        );
        assert_tensor_close_with_tol!(op_div_out(&b, &a, &out), expected, 1e-4, 1e-4);
    }

    // Integer inputs with fractional expected results (0.125, 0.5) pin
    // get_common_type's non-floating fallback to Float; an integer common type
    // would truncate the division to 0.
    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    // [spec:et:sem:op-div.torch.executor.native.get-common-type-fn/test]
    #[test]
    fn op_div_out_test_all_real_dtypes_supported() {
        // test_div_enumerate_a_types()
        test_div_enumerate_b_types_spec::<u8>();
        test_div_enumerate_b_types_spec::<i8>();
        test_div_enumerate_b_types_spec::<i16>();
        test_div_enumerate_b_types_spec::<i32>();
        test_div_enumerate_b_types_spec::<i64>();
        test_div_enumerate_b_types_spec::<f32>();
        test_div_enumerate_b_types_spec::<f64>();
        test_div_enumerate_b_types_spec::<Half>();
        test_div_enumerate_b_types_spec::<BFloat16>();
        // test_div<Bool, Float, Float>()
        test_div_bool_float_float();
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_supported1() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 1, 2, 1], vec![4.0, 8.0, 12.0, 16.0]);
        let b = tf.make_default(vec![2, 1, 4], vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
        let out = tf.zeros_default(vec![2, 2, 2, 4]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(
            vec![2, 2, 2, 4],
            vec![
                4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0,
                12.0, 12.0, 12.0, 12.0, 16.0, 16.0, 16.0, 16.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0,
                8.0,
            ],
        );
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_supported2() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![3, 2, 1], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let b = tf.make_default(vec![1, 2, 1], vec![2.0, 2.0]);
        let out = tf.zeros_default(vec![3, 2, 1]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(vec![3, 2, 1], vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_scalar_supported1() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 1, 3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let b = tf.make_default(vec![1], vec![2.0]);
        let out = tf.zeros_default(vec![2, 1, 3]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(vec![2, 1, 3], vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_scalar_supported2() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![1, 1, 1], vec![8.0]);
        let b = tf.make_default(vec![3, 1, 1], vec![2.0, 4.0, 8.0]);
        let out = tf.zeros_default(vec![3, 1, 1]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(vec![3, 1, 1], vec![4.0, 2.0, 1.0]);
        assert_tensor_eq!(out, ret);

        // std::swap(a, b)
        let out = tf.zeros_default(vec![3, 1, 1]);
        op_div_out(&b, &a, &out);
        let ret = tf.make_default(vec![3, 1, 1], vec![0.25, 0.5, 1.0]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_supported3() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![5], vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![1, 5], vec![2.0, 1.0, 2.0, 2.0, 3.0]);
        let out = tf.zeros_default(vec![1, 5]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(vec![1, 5], vec![1.0, 3.0, 2.0, 2.5, 2.0]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_scalar_rank0_supported() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![1], vec![8.0]);
        let b = tf.make_default(vec![], vec![2.0]);
        let out = tf.zeros_default(vec![1]);
        op_div_out(&a, &b, &out);
        let ret = tf.make_default(vec![1], vec![4.0]);
        assert_tensor_eq!(out, ret);

        op_div_out(&b, &a, &out);
        let ret = tf.make_default(vec![1], vec![0.25]);
        assert_tensor_eq!(out, ret);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_dim_size_is_one_ab() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9403896331787109,
                0.33918434381484985,
                0.6973152756690979,
                0.7128887176513672,
                0.9746139049530029,
                0.3507251739501953,
            ],
        );
        let y = tf.make_default(vec![1, 2], vec![0.942541241645813, 0.0298004150390625]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.9977172017097473,
                11.381866455078125,
                0.7398247122764587,
                23.922107696533203,
                1.0340278148651123,
                11.769137382507324,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_dim_size_missing_ab() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9403896331787109,
                0.33918434381484985,
                0.6973152756690979,
                0.7128887176513672,
                0.9746139049530029,
                0.3507251739501953,
            ],
        );
        let y = tf.make_default(vec![2], vec![0.942541241645813, 0.0298004150390625]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.9977172017097473,
                11.381866455078125,
                0.7398247122764587,
                23.922107696533203,
                1.0340278148651123,
                11.769137382507324,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_dim_size_is_one_ba() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 2], vec![0.942541241645813, 0.0298004150390625]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.9403896331787109,
                0.33918434381484985,
                0.6973152756690979,
                0.7128887176513672,
                0.9746139049530029,
                0.3507251739501953,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.0022879838943481,
                0.08785904943943024,
                1.351671576499939,
                0.041802339255809784,
                0.9670919179916382,
                0.08496799319982529,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_dim_size_missing_ba() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 2], vec![0.942541241645813, 0.0298004150390625]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.9403896331787109,
                0.33918434381484985,
                0.6973152756690979,
                0.7128887176513672,
                0.9746139049530029,
                0.3507251739501953,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.0022879838943481,
                0.08785904943943024,
                1.351671576499939,
                0.041802339255809784,
                0.9670919179916382,
                0.08496799319982529,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(is_aten, ...)`: never ATen, so the failure runs.
    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_mismatched_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let a = tf_int.ones_default(vec![2]);
        let b = tf_int.ones_default(vec![4]);
        let out = tf_float.ones_default(vec![2, 2]);
        let mut ctx = context();
        div_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_all_non_float_output_d_type_dies() {
        test_div_invalid_output_dtype_dies::<u8>();
        test_div_invalid_output_dtype_dies::<i8>();
        test_div_invalid_output_dtype_dies::<i16>();
        test_div_invalid_output_dtype_dies::<i32>();
        test_div_invalid_output_dtype_dies::<i64>();
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9321315288543701,
                0.013347446918487549,
                0.42016714811325073,
                0.059867143630981445,
                0.951939046382904,
                0.8632845878601074,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.714946985244751,
                0.39985191822052,
                0.9640239477157593,
                0.06885606050491333,
                0.008897960186004639,
                0.468650221824646,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.3037770986557007,
                0.03338097408413887,
                0.4358472228050232,
                0.869453489780426,
                106.98396301269531,
                1.8420659303665161,
            ],
        );
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9321315288543701,
                0.013347446918487549,
                0.42016714811325073,
                0.059867143630981445,
                0.951939046382904,
                0.8632845878601074,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.714946985244751,
                0.39985191822052,
                0.9640239477157593,
                0.06885606050491333,
                0.008897960186004639,
                0.468650221824646,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.3037770986557007,
                0.03338097408413887,
                0.4358472228050232,
                0.869453489780426,
                106.98396301269531,
                1.8420659303665161,
            ],
        );
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_broadcast_nd_test() {
        test_broadcast_3d::<f32>();
        test_broadcast_3d::<Half>();
        test_broadcast_3d::<BFloat16>();
    }

    // DISABLED: Dynamic shape not supported
    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    #[ignore = "DISABLED_DynamicShapeUnbound: dynamic shape not supported"]
    fn op_div_out_test_disabled_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.9321315288543701,
                0.013347446918487549,
                0.42016714811325073,
                0.059867143630981445,
                0.951939046382904,
                0.8632845878601074,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.714946985244751,
                0.39985191822052,
                0.9640239477157593,
                0.06885606050491333,
                0.008897960186004639,
                0.468650221824646,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                1.3037770986557007,
                0.03338097408413887,
                0.4358472228050232,
                0.869453489780426,
                106.98396301269531,
                1.8420659303665161,
            ],
        );
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_div_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // OpDivScalarOutTest
    // [spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_sanity_check_int_scalar() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());
        op_div_scalar_out(
            &tf_a.make_default(sizes.clone(), vec![1, 2, 4, -9]),
            &Scalar::from_i64(2),
            &out,
        );
        assert_tensor_eq!(out, tf_out.make_default(sizes, vec![0.5, 1.0, 2.0, -4.5]));
    }

    // [spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_sanity_check_float_scalar() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());
        op_div_scalar_out(
            &tf_a.make_default(sizes.clone(), vec![1, 2, 4, -9]),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_eq!(out, tf_out.make_default(sizes, vec![0.5, 1.0, 2.0, -4.5]));
    }

    // [spec:et:sem:op-div.torch.executor.native.div-scalar-out-fn/test]
    #[test]
    fn op_div_scalar_out_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        op_div_scalar_out(
            &tf.make_default(sizes.clone(), vec![1.3, 2.1, 4.6, 8.2]),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![0.65, 1.05, 2.3, 4.1]));
    }

    // Complex Type Tests
    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_float_basic() {
        let tf = TensorFactory::<Complex<f32>>::new();
        let sizes = vec![2, 2];
        let mk = |re: f32, im: f32| Complex { real: re, imag: im };
        let a = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 2.0), mk(4.0, 4.0), mk(3.0, 4.0), mk(8.0, 0.0)],
        );
        let b = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 0.0), mk(2.0, 0.0), mk(1.0, -1.0), mk(2.0, 2.0)],
        );
        let out = tf.zeros_default(sizes.clone());
        op_div_out(&a, &b, &out);
        let expected = tf.make_default(
            sizes,
            vec![mk(1.0, 2.0), mk(2.0, 2.0), mk(-0.5, 3.5), mk(2.0, -2.0)],
        );
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_double_basic() {
        let tf = TensorFactory::<Complex<f64>>::new();
        let sizes = vec![2];
        let mk = |re: f64, im: f64| Complex { real: re, imag: im };
        let a = tf.make_default(sizes.clone(), vec![mk(6.0, 8.0), mk(4.0, 0.0)]);
        let b = tf.make_default(sizes.clone(), vec![mk(2.0, 0.0), mk(0.0, 2.0)]);
        let out = tf.zeros_default(sizes.clone());
        op_div_out(&a, &b, &out);
        let expected = tf.make_default(sizes, vec![mk(3.0, 4.0), mk(0.0, -2.0)]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-out-fn/test]
    #[test]
    fn op_div_out_test_complex_float_identity() {
        let tf = TensorFactory::<Complex<f32>>::new();
        let sizes = vec![3];
        let mk = |re: f32, im: f32| Complex { real: re, imag: im };
        let a = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 2.0), mk(3.0, 4.0), mk(-5.0, 6.0)],
        );
        let one = tf.make_default(
            sizes.clone(),
            vec![mk(1.0, 0.0), mk(1.0, 0.0), mk(1.0, 0.0)],
        );
        let out = tf.zeros_default(sizes);
        op_div_out(&a, &one, &out);
        assert!(tensors_are_close(&out, &a, internal::K_DEFAULT_RTOL, None));
    }

    // PORT-NOTE: the C++ op_div suite has no coverage for the rounding-mode
    // variants; these focused tests pin `div_out_mode` against PyTorch's
    // trunc (toward zero) and floor (toward -inf) rounding_mode semantics.
    // [spec:et:sem:op-div.torch.executor.native.div-out-mode-fn/test]
    #[test]
    fn op_div_out_mode_test_trunc_and_floor() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![4];
        let a = tf.make_default(sizes.clone(), vec![7, -7, 8, -8]);
        let b = tf.make_default(sizes.clone(), vec![2, 2, 3, 3]);

        // trunc rounds toward zero.
        let out = tf.zeros_default(sizes.clone());
        let mut ctx = context();
        div_out_mode(&mut ctx, &a, &b, Some("trunc"), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(sizes.clone(), vec![3, -3, 2, -2]));

        // floor rounds toward negative infinity.
        let out = tf.zeros_default(sizes.clone());
        let mut ctx = context();
        div_out_mode(&mut ctx, &a, &b, Some("floor"), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(sizes.clone(), vec![3, -4, 2, -3]));
    }

    // None mode delegates to div_out (true division).
    // [spec:et:sem:op-div.torch.executor.native.div-out-mode-fn/test]
    #[test]
    fn op_div_out_mode_test_none_mode_delegates() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let a = tf.make_default(sizes.clone(), vec![1.0, 2.0, 4.0, 8.0]);
        let b = tf.make_default(sizes.clone(), vec![8.0, 4.0, 2.0, 1.0]);
        let out = tf.zeros_default(sizes.clone());
        let mut ctx = context();
        div_out_mode(&mut ctx, &a, &b, None, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close!(out, tf.make_default(sizes, vec![0.125, 0.5, 2.0, 8.0]));
    }

    // Integer division by zero is reported (not aborted) by div_out_mode.
    // [spec:et:sem:op-div.torch.executor.native.div-out-mode-fn/test]
    #[test]
    fn op_div_out_mode_test_integer_div_by_zero_fails() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![1];
        let a = tf.make_default(sizes.clone(), vec![5]);
        let b = tf.make_default(sizes.clone(), vec![0]);
        let out = tf.zeros_default(sizes);
        let mut ctx = context();
        div_out_mode(&mut ctx, &a, &b, Some("trunc"), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-div.torch.executor.native.div-scalar-mode-out-fn/test]
    #[test]
    fn op_div_scalar_mode_out_test_trunc_and_floor() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![4];
        let a = tf.make_default(sizes.clone(), vec![7, -7, 8, -8]);
        let b = Scalar::from_i64(3);

        let out = tf.zeros_default(sizes.clone());
        let mut ctx = context();
        div_scalar_mode_out(&mut ctx, &a, &b, Some("trunc"), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(sizes.clone(), vec![2, -2, 2, -2]));

        let out = tf.zeros_default(sizes.clone());
        let mut ctx = context();
        div_scalar_mode_out(&mut ctx, &a, &b, Some("floor"), &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(sizes.clone(), vec![2, -3, 2, -3]));
    }

    // Integer scalar division by zero is reported by div_scalar_mode_out.
    // [spec:et:sem:op-div.torch.executor.native.div-scalar-mode-out-fn/test]
    #[test]
    fn op_div_scalar_mode_out_test_integer_div_by_zero_fails() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![1];
        let a = tf.make_default(sizes.clone(), vec![5]);
        let out = tf.zeros_default(sizes);
        let mut ctx = context();
        div_scalar_mode_out(&mut ctx, &a, &Scalar::from_i64(0), Some("trunc"), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
