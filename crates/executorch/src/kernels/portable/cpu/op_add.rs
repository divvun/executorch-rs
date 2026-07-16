//! Literal port of kernels/portable/cpu/op_add.cpp.

use crate::kernels::portable::cpu::scalar_utils::{extract_scalar, get_scalar_dtype, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, resize_to_broadcast_target_size,
};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::kernel_ops_util::check_alpha_type;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    can_cast, is_complex_type, promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the complex path computes `val_a + val_alpha * val_b` on
// `Complex<T>` and derives `val_alpha` from a scalar via `scalar_to<CTYPE>`. The
// ported `Complex<T>` carries no arithmetic and `scalar_to` has no complex
// instantiation, so these module-local helpers reproduce c10::complex's
// `operator+`/`operator*` and the scalar->complex (imag = 0) construction without
// redesigning `Complex` or `scalar_utils`.
trait ComplexArith: Copy {
    fn c_add(self, other: Self) -> Self;
    fn c_mul(self, other: Self) -> Self;
    fn from_scalar(s: &Scalar) -> Self;
}
macro_rules! impl_complex_arith {
    ($comp:ty, $to:expr, $from:expr) => {
        impl ComplexArith for Complex<$comp> {
            fn c_add(self, other: Self) -> Self {
                Complex {
                    real: $from($to(self.real) + $to(other.real)),
                    imag: $from($to(self.imag) + $to(other.imag)),
                }
            }
            fn c_mul(self, other: Self) -> Self {
                // (a+bi)(c+di) = (ac - bd) + (ad + bc)i
                let a = $to(self.real);
                let b = $to(self.imag);
                let c = $to(other.real);
                let d = $to(other.imag);
                Complex {
                    real: $from(a * c - b * d),
                    imag: $from(a * d + b * c),
                }
            }
            fn from_scalar(s: &Scalar) -> Self {
                let v: f64 = if s.is_floating_point() {
                    s.to_f64()
                } else if s.is_boolean() {
                    if s.to_bool_val() { 1.0 } else { 0.0 }
                } else {
                    s.to_i64() as f64
                };
                Complex {
                    real: $from(v),
                    imag: $from(0.0),
                }
            }
        }
    };
}
impl_complex_arith!(Half, |x: Half| x.to_f64(), |x: f64| Half::from_f64(x));
impl_complex_arith!(f32, |x: f32| x as f64, |x: f64| x as f32);
impl_complex_arith!(f64, |x: f64| x, |x: f64| x);

// PORT-NOTE: C++ `val_a + val_alpha * val_b` over the REALB compute set includes
// `Bool`, where C++ integer-promotes bool operands (`false`->0, `true`->1) before
// the arithmetic and truncates back. Rust `bool` has no `+`/`*`, so this
// module-local trait reproduces the promotion for the realb compute types; the
// primitive arms are the plain operators.
trait RealbArith: Copy {
    fn radd(self, other: Self) -> Self;
    fn rmul(self, other: Self) -> Self;
}
macro_rules! impl_realb_arith_prim {
    ($($t:ty),*) => {$(
        impl RealbArith for $t {
            fn radd(self, other: Self) -> Self { self + other }
            fn rmul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_realb_arith_prim!(u8, i8, i16, i32, i64, f32, f64);
impl RealbArith for bool {
    fn radd(self, other: Self) -> Self {
        ((self as i32) + (other as i32)) != 0
    }
    fn rmul(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
}

// [spec:et:def:op-add.torch.executor.native.add-out-fn]
// [spec:et:sem:op-add.torch.executor.native.add-out-fn]
pub fn add_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type())
            && check_alpha_type(get_scalar_dtype(*alpha), common_type),
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
    let mut common_type_mut = common_type;
    let compute_type = get_compute_type(&mut common_type_mut);

    let op_name = "add.out";

    if is_complex_type(a.scalar_type())
        || is_complex_type(b.scalar_type())
        || is_complex_type(out.scalar_type())
    {
        // TODO: The current support for complex dtype enforces that input and
        // output tensors have the same dtype. Support mixed dtypes in the future.
        crate::et_kernel_check!(
            ctx,
            a.scalar_type() == b.scalar_type() && a.scalar_type() == out.scalar_type(),
            InvalidArgument,
            out
        );
        crate::et_switch_complexh_types!(out.scalar_type(), ctx, op_name, CTYPE, {
            let val_alpha: CTYPE = <CTYPE as ComplexArith>::from_scalar(alpha);
            apply_binary_elementwise_fn::<CTYPE, CTYPE, CTYPE, _>(
                move |val_a: CTYPE, val_b: CTYPE| val_a.c_add(val_alpha.c_mul(val_b)),
                a,
                b,
                out,
            );
        });
    } else {
        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let mut val_alpha: CTYPE_COMPUTE = Default::default();
            crate::et_kernel_check!(
                ctx,
                extract_scalar(*alpha, &mut val_alpha),
                InvalidArgument,
                out
            );
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                move |vals: &[CTYPE_COMPUTE]| vals[0].radd(val_alpha.rmul(vals[1])),
                ctx,
                a,
                SupportedTensorDtypes::REALHBBF16,
                b,
                SupportedTensorDtypes::REALHBBF16,
                out,
                SupportedTensorDtypes::REALHBBF16,
                false,
            );
        });
    }

    out
}

// [spec:et:def:op-add.torch.executor.native.add-scalar-out-fn]
// [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn]
pub fn add_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type = crate::kernels::portable::cpu::scalar_utils::promote_type_with_scalar(
        a.scalar_type(),
        *b,
        false,
    );

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        common_type == out.scalar_type() && check_alpha_type(get_scalar_dtype(*alpha), common_type),
        InvalidArgument,
        out
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
        resize_tensor(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type = get_compute_type(&mut common_type_mut);

    let op_name = "add.Scalar_out";

    crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        let mut val_alpha: CTYPE_COMPUTE = Default::default();
        crate::et_kernel_check!(
            ctx,
            extract_scalar(*alpha, &mut val_alpha),
            InvalidArgument,
            out
        );
        let val_alpha_times_b = val_alpha.rmul(val_b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| vals[0].radd(val_alpha_times_b),
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
            false,
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
    use crate::runtime::core::exec_aten::util::scalar_type_util::{
        CppTypeToScalarType, is_integral_type,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, ComplexDouble, ComplexFloat, ComplexHalf};
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

    fn op_add_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        alpha: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        add_out(&mut ctx, self_, other, alpha, out)
    }

    fn op_add_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        alpha: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        add_scalar_out(&mut ctx, self_, other, alpha, out)
    }

    // PORT-NOTE: local `from_f64` bridge for the element types used across the
    // add suites (mirrors the op_ceil.rs / op_abs.rs test helper).
    trait FromF64Elem: Copy {
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
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    fn test_add<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];

        let out = tf_out.zeros_default(sizes.clone());

        op_add_out(
            &tf_a.make_default(
                sizes.clone(),
                vec![
                    A::from_f64(1.0),
                    A::from_f64(2.0),
                    A::from_f64(4.0),
                    A::from_f64(8.0),
                ],
            ),
            &tf_b.ones_default(sizes.clone()),
            &Scalar::from_i64(1),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf_out.make_default(
                sizes,
                vec![
                    OUT::from_f64(2.0),
                    OUT::from_f64(3.0),
                    OUT::from_f64(5.0),
                    OUT::from_f64(9.0),
                ],
            ),
            0.0,
            Some(0.0),
        ));
    }

    fn test_add_enumerate_out_types<A, B>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        test_add::<A, B, BFloat16>();
        test_add::<A, B, Half>();
        test_add::<A, B, f32>();
        test_add::<A, B, f64>();
        // Integral out type is only allowed if both inputs are integral types.
        if is_integral_type(A::VALUE, false) && is_integral_type(B::VALUE, false) {
            test_add::<A, B, i32>();
            test_add::<A, B, i64>();
        }
    }

    fn test_add_enumerate_b_types<A>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        // ET_FORALL_REALHBF16_TYPES
        test_add_enumerate_out_types::<A, u8>();
        test_add_enumerate_out_types::<A, i8>();
        test_add_enumerate_out_types::<A, i16>();
        test_add_enumerate_out_types::<A, i32>();
        test_add_enumerate_out_types::<A, i64>();
        test_add_enumerate_out_types::<A, f32>();
        test_add_enumerate_out_types::<A, f64>();
        test_add_enumerate_out_types::<A, Half>();
        test_add_enumerate_out_types::<A, BFloat16>();
    }

    fn test_add_enumerate_a_types() {
        // ET_FORALL_REALHBF16_TYPES
        test_add_enumerate_b_types::<u8>();
        test_add_enumerate_b_types::<i8>();
        test_add_enumerate_b_types::<i16>();
        test_add_enumerate_b_types::<i32>();
        test_add_enumerate_b_types::<i64>();
        test_add_enumerate_b_types::<f32>();
        test_add_enumerate_b_types::<f64>();
        test_add_enumerate_b_types::<Half>();
        test_add_enumerate_b_types::<BFloat16>();
    }

    fn test_add_complex_dtype<C>(mk: impl Fn(f64, f64) -> C)
    where
        C: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<C>::new();

        // Both inputs have the same shape.
        let x_0 = tf.make_default(vec![2], vec![mk(1.0, 2.1), mk(3.1, 4.0)]);
        let y_0 = tf.make_default(vec![2], vec![mk(5.2, 6.3), mk(7.0, 8.9)]);
        let out = tf.full(vec![2], mk(0.0, 0.0), TensorShapeDynamism::STATIC);
        op_add_out(&x_0, &y_0, &Scalar::from_i64(1), &out);
        let expected_0 = tf.make_default(vec![2], vec![mk(6.2, 8.4), mk(10.1, 12.9)]);
        assert!(tensors_are_close(&out, &expected_0, 0.0, Some(0.0)));

        // Other tensor has numel() = 1.
        let y_1 = tf.make_default(vec![1], vec![mk(2.0, 3.0)]);
        op_add_out(&x_0, &y_1, &Scalar::from_i64(2), &out);
        let expected_1 = tf.make_default(vec![2], vec![mk(5.0, 8.1), mk(7.1, 10.0)]);
        assert!(tensors_are_close(&out, &expected_1, 0.0, Some(0.0)));
    }

    fn test_add_enumerate_complex_types() {
        test_add_complex_dtype::<ComplexHalf>(|re, im| ComplexHalf {
            real: Half::from_f64(re),
            imag: Half::from_f64(im),
        });
        test_add_complex_dtype::<ComplexFloat>(|re, im| ComplexFloat {
            real: re as f32,
            imag: im as f32,
        });
        test_add_complex_dtype::<ComplexDouble>(|re, im| ComplexDouble { real: re, imag: im });
    }

    fn test_floating_point_add_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());

        op_add_out(
            &tf.make_default(
                sizes.clone(),
                vec![
                    T::from_f64(1.25),
                    T::from_f64(2.25),
                    T::from_f64(4.5),
                    T::from_f64(8.875),
                ],
            ),
            &tf.ones_default(sizes.clone()),
            &Scalar::from_double(1.25),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                sizes,
                vec![
                    T::from_f64(2.5),
                    T::from_f64(3.5),
                    T::from_f64(5.75),
                    T::from_f64(10.125),
                ],
            ),
            internal::K_DEFAULT_RTOL,
            None,
        ));
    }

    fn f32_vec(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    fn test_broadcast_3d<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };

        let a = tf_a.make_default(
            vec![2, 2, 3],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let b = tf_a.make_default(vec![2, 1, 3], d(&[2., 3., 4., 5., 6., 7.]));

        let out = tf_a.make_default(
            vec![2, 2, 3],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            d(&[3., 5., 7., 6., 8., 10., 12., 14., 16., 15., 17., 19.]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            d(&[3.5, 6., 8.5, 8., 10.5, 13., 15.5, 18., 20.5, 20., 22.5, 25.]),
        );
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.5), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    fn test_broadcast_4d<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };

        let a = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&(1..=60).map(|i| i as f64).collect::<Vec<_>>()),
        );
        let b = tf_a.make_default(
            vec![2, 1, 3, 5],
            d(&(1..=30).map(|i| i as f64).collect::<Vec<_>>()),
        );

        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&[
                2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 17., 19.,
                21., 23., 25., 27., 29., 31., 33., 35., 37., 39., 41., 43., 45., 47., 49., 51.,
                53., 55., 57., 59., 61., 63., 65., 67., 69., 71., 73., 75., 62., 64., 66., 68.,
                70., 72., 74., 76., 78., 80., 82., 84., 86., 88., 90.,
            ]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));

        let b = tf_a.make_default(
            vec![2, 2, 1, 5],
            d(&(1..=20).map(|i| i as f64).collect::<Vec<_>>()),
        );
        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&[
                2., 4., 6., 8., 10., 7., 9., 11., 13., 15., 12., 14., 16., 18., 20., 22., 24., 26.,
                28., 30., 27., 29., 31., 33., 35., 32., 34., 36., 38., 40., 42., 44., 46., 48.,
                50., 47., 49., 51., 53., 55., 52., 54., 56., 58., 60., 62., 64., 66., 68., 70.,
                67., 69., 71., 73., 75., 72., 74., 76., 78., 80.,
            ]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    fn test_broadcast_last_dim<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };

        let a = tf_a.make_default(
            vec![4, 3],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let b = tf_a.make_default(vec![4, 1], d(&[2., 3., 4., 5.]));

        let out = tf_a.zeros_default(vec![4, 3]);
        let expected = tf_a.make_default(
            vec![4, 3],
            d(&[3., 4., 5., 7., 8., 9., 11., 12., 13., 15., 16., 17.]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));

        let a = tf_a.make_default(
            vec![2, 2, 3],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let b = tf_a.make_default(vec![2, 2, 1], d(&[2., 3., 4., 5.]));

        let out = tf_a.zeros_default(vec![2, 2, 3]);
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            d(&[3., 4., 5., 7., 8., 9., 11., 12., 13., 15., 16., 17.]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));

        let a = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&(1..=60).map(|i| i as f64).collect::<Vec<_>>()),
        );
        let b = tf_a.make_default(
            vec![2, 2, 3, 1],
            d(&(1..=12).map(|i| i as f64).collect::<Vec<_>>()),
        );

        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&[
                2., 3., 4., 5., 6., 8., 9., 10., 11., 12., 14., 15., 16., 17., 18., 20., 21., 22.,
                23., 24., 26., 27., 28., 29., 30., 32., 33., 34., 35., 36., 38., 39., 40., 41.,
                42., 44., 45., 46., 47., 48., 50., 51., 52., 53., 54., 56., 57., 58., 59., 60.,
                62., 63., 64., 65., 66., 68., 69., 70., 71., 72.,
            ]),
        );

        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
        assert!(tensors_are_close(
            op_add_out(&b, &a, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // expect_bad_alpha_value_dies<DTYPE>(bad_value) for the tensor-tensor suite.
    fn expect_bad_alpha_value_dies<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        add_out(&mut ctx, &a, &b, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // expect_bad_alpha_value_dies<DTYPE>(bad_value) for the tensor-scalar suite.
    fn expect_bad_alpha_value_dies_scalar<T>(bad_value: Scalar)
    where
        T: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<T>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = Scalar::from_i64(1);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        add_scalar_out(&mut ctx, &a, &b, &bad_value, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ---- OpAddOutKernelTest ----

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_all_real_dtypes_supported() {
        test_add_enumerate_a_types();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_complex_tensors() {
        test_add_enumerate_complex_types();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_float_tensors() {
        test_floating_point_add_out::<f32>();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_double_tensors() {
        test_floating_point_add_out::<f64>();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_half_tensors() {
        test_floating_point_add_out::<Half>();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bfloat16_tensors() {
        test_floating_point_add_out::<BFloat16>();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_and_int_input_tensor() {
        let tf = TensorFactory::<bool>::new();
        let tfi = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tfi.make_default(sizes.clone(), vec![2, 4, 3, 3]);

        let out = tfi.zeros_default(sizes.clone());

        op_add_out(&a, &b, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &tfi.make_default(sizes, vec![2, 5, 3, 4]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_and_bool_input_tensor() {
        setup();
        let tf = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tf.make_default(sizes.clone(), vec![false, true, true, true]);

        let out = tf.zeros_default(sizes.clone());

        op_add_out(&a, &b, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, vec![false, true, true, true]),
            0.0,
            Some(0.0)
        ));
    }

    fn broadcast_dim_size_helper(y_sizes: Vec<i32>) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.5721208453178406,
                0.9629082083702087,
                0.19517338275909424,
                0.4107270836830139,
                0.945562481880188,
                0.8788509368896484,
            ]),
        );
        let y = tf.make_default(y_sizes, f32_vec(&[0.7453382015228271, 0.3131374716758728]));
        let expected_result = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                1.3174591064453125,
                1.2760456800460815,
                0.9405115842819214,
                0.7238645553588867,
                1.6909006834030151,
                1.191988468170166,
            ]),
        );

        let out = tf.zeros_default(vec![3, 2]);
        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_dim_size_is_one_ab() {
        broadcast_dim_size_helper(vec![1, 2]);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_dim_size_missing_ab() {
        broadcast_dim_size_helper(vec![2]);
    }

    fn broadcast_dim_size_ba_helper(x_sizes: Vec<i32>) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(x_sizes, f32_vec(&[0.7453382015228271, 0.3131374716758728]));
        let y = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.5721208453178406,
                0.9629082083702087,
                0.19517338275909424,
                0.4107270836830139,
                0.945562481880188,
                0.8788509368896484,
            ]),
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                1.3174591064453125,
                1.2760456800460815,
                0.9405115842819214,
                0.7238645553588867,
                1.6909006834030151,
                1.191988468170166,
            ]),
        );

        let out = tf.zeros_default(vec![3, 2]);
        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_dim_size_is_one_ba() {
        broadcast_dim_size_ba_helper(vec![1, 2]);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_dim_size_missing_ba() {
        broadcast_dim_size_ba_helper(vec![2]);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.zeros_default(vec![5, 1, 3, 1]);
        let b = tf.ones_default(vec![2, 1, 4]);

        let out = tf.zeros_default(vec![5, 2, 3, 4]);

        let ret = op_add_out(&a, &b, &Scalar::from_i64(1), &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![5, 2, 3, 4]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_tensor() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1], f32_vec(&[1.75]));
        let y = tf.make_default(vec![3, 2], f32_vec(&[-1.5, -1.0, -0.5, 0.0, 0.5, 1.5]));

        let out = tf.zeros_default(vec![3, 2]);

        let _ret = op_add_out(&x, &y, &Scalar::from_i64(1), &out);

        let expected = tf.make_default(vec![3, 2], f32_vec(&[0.25, 0.75, 1.25, 1.75, 2.25, 3.25]));

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));

        let out = op_add_out(&y, &x, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_tensor_type_promotion() {
        let tf = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let x = tf_double.make_default(vec![1], vec![1.75]);
        let y = tf.make_default(vec![3, 2], f32_vec(&[-1.5, -1.0, -0.5, 0.0, 0.5, 1.5]));

        let out = tf_double.zeros_default(vec![3, 2]);

        let _ret = op_add_out(&x, &y, &Scalar::from_i64(1), &out);

        let expected = tf_double.make_default(vec![3, 2], vec![0.25, 0.75, 1.25, 1.75, 2.25, 3.25]);

        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));

        let out = op_add_out(&y, &x, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(out, &expected, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_rank0_tensor() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![1], f32_vec(&[5.0]));
        let b = tf.make_default(vec![], f32_vec(&[2.0]));

        let out = tf.zeros_default(vec![1]);

        op_add_out(&a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![1], f32_vec(&[7.0]));
        assert!(tensors_are_close(&out, &ret, 0.0, Some(0.0)));

        op_add_out(&b, &a, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(&out, &ret, 0.0, Some(0.0)));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_nd_test() {
        test_broadcast_3d::<f32>();
        test_broadcast_3d::<Half>();
        test_broadcast_3d::<BFloat16>();

        test_broadcast_4d::<f32>();
        test_broadcast_4d::<Half>();
        test_broadcast_4d::<BFloat16>();

        test_broadcast_last_dim::<f32>();
        test_broadcast_last_dim::<Half>();
        test_broadcast_last_dim::<BFloat16>();
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_b_to_a() {
        let tf_a = TensorFactory::<f32>::new();
        let a = tf_a.make_default(vec![1, 3], f32_vec(&[1.0, 2.0, 3.0]));
        let b = tf_a.make_default(vec![1, 1, 3], f32_vec(&[3.2, 1.3, 5.5]));
        let out = tf_a.zeros_default(vec![1, 1, 3]);

        let expected = tf_a.make_default(vec![1, 1, 3], f32_vec(&[4.2, 3.3, 8.5]));
        assert!(tensors_are_close(
            op_add_out(&a, &b, &Scalar::from_double(1.0), &out),
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // ---- Death Tests ----

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    // Int common type + float alpha must fail: this is check_alpha_type rejecting
    // a non-castable alpha (can_cast(Double, Int) is false, common != Bool).
    // [spec:et:sem:kernel-ops-util.torch.executor.check-alpha-type-fn/test]
    #[test]
    fn op_add_out_kernel_test_int_inputs_float_alpha_dies() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        add_out(
            &mut ctx,
            &tf.ones_default(sizes.clone()),
            &tf.ones_default(sizes),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_inputs_float_alpha_dies() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        add_out(
            &mut ctx,
            &tf.ones_default(sizes.clone()),
            &tf.ones_default(sizes),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_int_output_with_float_input_dies() {
        let tfi = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];

        let a = tfi.make_default(sizes.clone(), vec![2, 4, 3, 3]);
        let b = tff.make_default(sizes.clone(), f32_vec(&[2.0, 4.0, 3.0, 3.0]));
        let out = tfi.zeros_default(sizes);

        let mut ctx = context();
        add_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_output_with_integral_input() {
        let tf = TensorFactory::<bool>::new();
        let tfi = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];

        let a = tfi.make_default(sizes.clone(), vec![0, 1, 1, 0]);
        let b = tfi.make_default(sizes.clone(), vec![2, 3, 4, 3]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        add_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_mismatched_non_broadcastable_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![4, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![8]);

        let mut ctx = context();
        add_out(&mut ctx, &a, &b, &Scalar::from_i64(0), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    //
    // PORT-NOTE: C++ `ET_SKIP_IF(SupportedFeatures::get()->output_resize, ...)`:
    // the portable kernel implicitly resizes the output, so the C++ test is
    // skipped for the portable build. Ported as a skip (no-op body).
    #[test]
    fn op_add_out_kernel_test_mismatched_output_shapes_dies() {
        // Skipped: portable kernel supports implicit output resize.
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_simple_generated_case() {
        setup();
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let y = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![2.0f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let _ret = op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    fn dynamic_shape_helper(out_sizes: Vec<i32>) {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.04024535417556763,
                0.6475827097892761,
                0.9623860716819763,
                0.6206040978431702,
                0.47623592615127563,
                0.4509747624397278,
            ]),
        );
        let y = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.7232733964920044,
                0.3614498972892761,
                0.15757757425308228,
                0.9975225925445557,
                0.09227871894836426,
                0.3320664167404175,
            ]),
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.763518750667572,
                1.0090326070785522,
                1.1199636459350586,
                1.618126630783081,
                0.5685146450996399,
                0.7830411791801453,
            ]),
        );

        let out = tf.zeros(out_sizes, TensorShapeDynamism::DYNAMIC_BOUND);
        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_dynamic_shape_upper_bound_same_as_expected() {
        dynamic_shape_helper(vec![3, 2]);
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_dynamic_shape_upper_bound_larger_than_expected() {
        dynamic_shape_helper(vec![10, 10]);
    }

    // PORT-NOTE: `DISABLED_DynamicShapeUnbound` is disabled in C++
    // ("Dynamic shape not supported"); ported as an ignored test.
    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    #[ignore = "DISABLED in C++: Dynamic shape unbound not supported"]
    fn op_add_out_kernel_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.04024535417556763,
                0.6475827097892761,
                0.9623860716819763,
                0.6206040978431702,
                0.47623592615127563,
                0.4509747624397278,
            ]),
        );
        let y = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.7232733964920044,
                0.3614498972892761,
                0.15757757425308228,
                0.9975225925445557,
                0.09227871894836426,
                0.3320664167404175,
            ]),
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            f32_vec(&[
                0.763518750667572,
                1.0090326070785522,
                1.1199636459350586,
                1.618126630783081,
                0.5685146450996399,
                0.7830411791801453,
            ]),
        );

        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert!(tensors_are_close(
            &out,
            &expected_result,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // ---- OpAddScalarOutKernelTest ----

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_sanity_check() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_add_scalar_out(
            &tf.make_default(sizes.clone(), vec![1, 2, 4, 8]),
            &Scalar::from_bool(true),
            &Scalar::from_i64(2),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, vec![3, 4, 6, 10]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_add_scalar_out(
            &tf.make_default(sizes.clone(), f32_vec(&[1.3, 2.1, 4.6, 8.2])),
            &Scalar::from_double(1.9),
            &Scalar::from_double(2.8),
            &out,
        );

        assert!(tensors_are_close(
            &out,
            &tf.make_default(sizes, f32_vec(&[6.62, 7.42, 9.92, 13.52])),
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_dtype_test_float16_bool_int_float16() {
        let tf_half = TensorFactory::<Half>::new();

        let self_ = tf_half.ones_default(vec![2, 2]);
        let other = Scalar::from_bool(true);
        let alpha = Scalar::from_i64(1);
        let out = tf_half.zeros_default(vec![2, 2]);
        let out_expected =
            tf_half.full(vec![2, 2], Half::from_f64(2.0), TensorShapeDynamism::STATIC);
        op_add_scalar_out(&self_, &other, &alpha, &out);
        assert!(tensors_are_close(
            &out,
            &out_expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // ---- ByteTensorFloatingPointAlphaDies / IntTensorFloatingPointAlphaDies ----

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_byte_tensor_floating_point_alpha_dies() {
        expect_bad_alpha_value_dies::<u8>(Scalar::from_double(2.2));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_int_tensor_floating_point_alpha_dies() {
        expect_bad_alpha_value_dies::<i32>(Scalar::from_double(2.2));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_byte_tensor_floating_point_alpha_dies() {
        expect_bad_alpha_value_dies_scalar::<u8>(Scalar::from_double(2.2));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_int_tensor_floating_point_alpha_dies() {
        expect_bad_alpha_value_dies_scalar::<i32>(Scalar::from_double(2.2));
    }

    // ---- GENERATE_SCALAR_OVERFLOW_TESTS(OpAddOutKernelTest) ----

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies::<u8>(Scalar::from_i64(256));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_char_tensor_too_small_scalar_dies() {
        expect_bad_alpha_value_dies::<i8>(Scalar::from_i64(-129));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_short_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies::<i16>(Scalar::from_i64(32768));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_float_tensor_too_small_scalar_dies() {
        expect_bad_alpha_value_dies::<f32>(Scalar::from_double(-3.41e+38));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_float_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies::<f32>(Scalar::from_double(3.41e+38));
    }

    // ---- GENERATE_SCALAR_OVERFLOW_TESTS(OpAddScalarOutKernelTest) ----

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_byte_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies_scalar::<u8>(Scalar::from_i64(256));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_char_tensor_too_small_scalar_dies() {
        expect_bad_alpha_value_dies_scalar::<i8>(Scalar::from_i64(-129));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_short_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies_scalar::<i16>(Scalar::from_i64(32768));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_float_tensor_too_small_scalar_dies() {
        expect_bad_alpha_value_dies_scalar::<f32>(Scalar::from_double(-3.41e+38));
    }

    // [spec:et:sem:op-add.torch.executor.native.add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_float_tensor_too_large_scalar_dies() {
        expect_bad_alpha_value_dies_scalar::<f32>(Scalar::from_double(3.41e+38));
    }
}
