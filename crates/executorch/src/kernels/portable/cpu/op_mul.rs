//! Literal port of kernels/portable/cpu/op_mul.cpp.

use crate::kernels::portable::cpu::scalar_utils::scalar_to;
use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, resize_to_broadcast_target_size,
};
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{
    can_cast, is_complex_type, is_real_type, promote_types,
};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{BFloat16, Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the complex path computes `val_a * val_b` on native `Complex<T>`.
// The ported `Complex<T>` carries no arithmetic, so this module-local trait
// reproduces c10::complex's `operator*` without redesigning `Complex` (mirroring
// the established pattern in op_add).
trait ComplexArith: Copy {
    fn c_mul(self, other: Self) -> Self;
}
macro_rules! impl_complex_arith {
    ($comp:ty, $to:expr, $from:expr) => {
        impl ComplexArith for Complex<$comp> {
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
        }
    };
}
impl_complex_arith!(Half, |x: Half| x.to_f64(), |x: f64| Half::from_f64(x));
impl_complex_arith!(f32, |x: f32| x as f64, |x: f64| x as f32);
impl_complex_arith!(f64, |x: f64| x, |x: f64| x);

// PORT-NOTE: C++ `val_a * val_b` over the REALB compute set includes `Bool`,
// where C++ integer-promotes bool operands before the multiply and truncates
// back. Rust `bool` has no `*`, so this module-local trait reproduces the
// promotion; the primitive arms are the plain operator.
trait RealbMul: Copy {
    fn rmul(self, other: Self) -> Self;
}
macro_rules! impl_realb_mul_int {
    ($($t:ty),*) => {$(
        impl RealbMul for $t {
            // Integer multiply wraps per two's-complement, mirroring the C++
            // integer-promotion-then-narrow semantics (sem rule: "integer
            // multiply wraps per two's-complement").
            fn rmul(self, other: Self) -> Self { self.wrapping_mul(other) }
        }
    )*};
}
impl_realb_mul_int!(u8, i8, i16, i32, i64);
macro_rules! impl_realb_mul_float {
    ($($t:ty),*) => {$(
        impl RealbMul for $t {
            fn rmul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_realb_mul_float!(f32, f64);
impl RealbMul for Half {
    fn rmul(self, other: Self) -> Self {
        Half::from_f32(self.to_f32() * other.to_f32())
    }
}
impl RealbMul for BFloat16 {
    fn rmul(self, other: Self) -> Self {
        BFloat16::from_f32(self.to_f32() * other.to_f32())
    }
}
impl RealbMul for bool {
    fn rmul(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
}

// [spec:et:def:op-mul.torch.executor.native.mul-out-fn]
// [spec:et:sem:op-mul.torch.executor.native.mul-out-fn]
#[executorch_macros::et_kernel("aten::mul.out")]
pub fn mul_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType = promote_types(a.scalar_type(), b.scalar_type(), false);

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
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "mul.out";

    crate::et_kernel_check!(
        ctx,
        is_real_type(compute_type)
            || is_complex_type(compute_type)
            || compute_type == ScalarType::Bool,
        InvalidArgument,
        out
    );

    if is_complex_type(compute_type) {
        crate::et_kernel_check!(
            ctx,
            a.scalar_type() == b.scalar_type() && a.scalar_type() == out.scalar_type(),
            InvalidArgument,
            out
        );
        crate::et_switch_complexh_types!(out.scalar_type(), ctx, "mul.out", CTYPE, {
            apply_binary_elementwise_fn::<CTYPE, CTYPE, CTYPE, _>(
                |val_a: CTYPE, val_b: CTYPE| -> CTYPE { val_a.c_mul(val_b) },
                a,
                b,
                out,
            );
        });
    } else {
        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { vals[0].rmul(vals[1]) },
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

// [spec:et:def:op-mul.torch.executor.native.mul-scalar-out-fn]
// [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn]
pub fn mul_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType =
        crate::kernels::portable::cpu::scalar_utils::promote_type_with_scalar(
            a.scalar_type(),
            *b,
            false,
        );

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
        resize_tensor(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type: ScalarType = get_compute_type(&mut common_type_mut);

    let op_name = "mul.Scalar_out";

    crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            move |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { vals[0].rmul(val_b) },
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
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::{
        CppTypeToScalarType, is_integral_type,
    };
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{ComplexDouble, ComplexFloat, ComplexHalf};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_mul_out<'a, 'b>(self_: &Tensor, other: &Tensor, out: &'a Tensor<'b>) -> &'a Tensor<'b> {
        let mut ctx = context();
        mul_out(&mut ctx, self_, other, out)
    }

    fn op_mul_scalar_out<'a, 'b>(
        self_: &Tensor,
        other: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        mul_scalar_out(&mut ctx, self_, other, out)
    }

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

    fn test_mul<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue + FromF64Elem,
        OUT: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let da = |v: &[f64]| -> Vec<A> { v.iter().map(|&x| A::from_f64(x)).collect() };
        let db = |v: &[f64]| -> Vec<B> { v.iter().map(|&x| B::from_f64(x)).collect() };
        let dout = |v: &[f64]| -> Vec<OUT> { v.iter().map(|&x| OUT::from_f64(x)).collect() };

        let out = tf_out.zeros_default(sizes.clone());

        op_mul_out(
            &tf_a.make_default(sizes.clone(), da(&[1., 2., 4., 8.])),
            &tf_b.ones_default(sizes.clone()),
            &out,
        );
        assert!(tensors_are_close(
            &out,
            &tf_out.make_default(sizes.clone(), dout(&[1., 2., 4., 8.])),
            0.0,
            Some(0.0)
        ));

        op_mul_out(
            &tf_a.make_default(sizes.clone(), da(&[1., 2., 4., 8.])),
            &tf_b.zeros_default(sizes.clone()),
            &out,
        );
        assert!(tensors_are_close(
            &out,
            &tf_out.make_default(sizes.clone(), dout(&[0., 0., 0., 0.])),
            0.0,
            Some(0.0)
        ));

        op_mul_out(
            &tf_a.make_default(sizes.clone(), da(&[1., 2., 4., 8.])),
            &tf_b.make_default(sizes.clone(), db(&[1., 2., 4., 8.])),
            &out,
        );
        assert!(tensors_are_close(
            &out,
            &tf_out.make_default(sizes.clone(), dout(&[1., 4., 16., 64.])),
            0.0,
            Some(0.0)
        ));

        let out = tf_out.zeros_default(vec![18]);
        op_mul_out(
            &tf_a.full(vec![18], A::from_f64(4.0), TensorShapeDynamism::STATIC),
            &tf_b.full(vec![18], B::from_f64(2.0), TensorShapeDynamism::STATIC),
            &out,
        );
        assert!(tensors_are_close(
            &out,
            &tf_out.full(vec![18], OUT::from_f64(8.0), TensorShapeDynamism::STATIC),
            0.0,
            Some(0.0)
        ));
    }

    fn test_mul_enumerate_out_types<A, B>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
        B: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        test_mul::<A, B, Half>();
        test_mul::<A, B, f32>();
        test_mul::<A, B, f64>();
        if is_integral_type(A::VALUE, false) && is_integral_type(B::VALUE, false) {
            test_mul::<A, B, i32>();
            test_mul::<A, B, i64>();
        }
    }

    fn test_mul_enumerate_b_types<A>()
    where
        A: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        // ET_FORALL_REALHBF16_TYPES
        test_mul_enumerate_out_types::<A, u8>();
        test_mul_enumerate_out_types::<A, i8>();
        test_mul_enumerate_out_types::<A, i16>();
        test_mul_enumerate_out_types::<A, i32>();
        test_mul_enumerate_out_types::<A, i64>();
        test_mul_enumerate_out_types::<A, f32>();
        test_mul_enumerate_out_types::<A, f64>();
        test_mul_enumerate_out_types::<A, Half>();
        test_mul_enumerate_out_types::<A, BFloat16>();
    }

    fn test_mul_enumerate_a_types() {
        // ET_FORALL_REALHBF16_TYPES
        test_mul_enumerate_b_types::<u8>();
        test_mul_enumerate_b_types::<i8>();
        test_mul_enumerate_b_types::<i16>();
        test_mul_enumerate_b_types::<i32>();
        test_mul_enumerate_b_types::<i64>();
        test_mul_enumerate_b_types::<f32>();
        test_mul_enumerate_b_types::<f64>();
        test_mul_enumerate_b_types::<Half>();
        test_mul_enumerate_b_types::<BFloat16>();
    }

    fn test_floating_point_mul_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_mul_out(
            &tf.make_default(sizes.clone(), d(&[1.25, 2.5, 4.75, 8.875])),
            &tf.ones_default(sizes.clone()),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(sizes.clone(), d(&[1.25, 2.5, 4.75, 8.875]))
        );

        op_mul_out(
            &tf.make_default(sizes.clone(), d(&[1.1, 2.2, 4.4, 8.8])),
            &tf.zeros_default(sizes.clone()),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(sizes.clone(), d(&[0.0, 0.0, 0.0, 0.0]))
        );

        op_mul_out(
            &tf.make_default(sizes.clone(), d(&[1.25, 2.5, 4.75, 8.875])),
            &tf.make_default(sizes.clone(), d(&[1.25, 2.5, 4.75, 8.875])),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(sizes, d(&[1.5625, 6.25, 22.5625, 78.765625]))
        );
    }

    fn test_optimized_path_ignores_leading_1_dimensions<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let sizes1 = vec![1, 1, 2, 2];
        let sizes2 = vec![1, 2, 2];
        let out = tf.zeros_default(sizes1.clone());
        op_mul_out(
            &tf.make_default(sizes1.clone(), d(&[1.1, 2.2, 4.4, 8.8])),
            &tf.ones_default(sizes2),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes1, d(&[1.1, 2.2, 4.4, 8.8])));
    }

    fn test_broadcast_a2b<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let b_sizeses: Vec<Vec<i32>> = vec![vec![2], vec![1, 2]];
        for b_sizes in b_sizeses {
            let a = tf_a.make_default(vec![2, 2], d(&[1., 2., 3., 4.]));
            let b = tf_a.make_default(b_sizes, d(&[2., 2.]));
            let out = tf_a.zeros_default(vec![2, 2]);
            assert_tensor_close!(
                op_mul_out(&a, &b, &out),
                tf_a.make_default(vec![2, 2], d(&[2., 4., 6., 8.]))
            );
        }
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
            d(&[2., 6., 12., 8., 15., 24., 35., 48., 63., 50., 66., 84.]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);
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
                1., 4., 9., 16., 25., 36., 49., 64., 81., 100., 121., 144., 169., 196., 225., 16.,
                34., 54., 76., 100., 126., 154., 184., 216., 250., 286., 324., 364., 406., 450.,
                496., 544., 594., 646., 700., 756., 814., 874., 936., 1000., 1066., 1134., 1204.,
                1276., 1350., 736., 799., 864., 931., 1000., 1071., 1144., 1219., 1296., 1375.,
                1456., 1539., 1624., 1711., 1800.,
            ]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);

        let b = tf_a.make_default(
            vec![2, 2, 1, 5],
            d(&(1..=20).map(|i| i as f64).collect::<Vec<_>>()),
        );
        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&[
                1., 4., 9., 16., 25., 6., 14., 24., 36., 50., 11., 24., 39., 56., 75., 96., 119.,
                144., 171., 200., 126., 154., 184., 216., 250., 156., 189., 224., 261., 300., 341.,
                384., 429., 476., 525., 396., 444., 494., 546., 600., 451., 504., 559., 616., 675.,
                736., 799., 864., 931., 1000., 816., 884., 954., 1026., 1100., 896., 969., 1044.,
                1121., 1200.,
            ]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);
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
            d(&[2., 4., 6., 12., 15., 18., 28., 32., 36., 50., 55., 60.]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);

        let a = tf_a.make_default(
            vec![2, 2, 3],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let b = tf_a.make_default(vec![2, 2, 1], d(&[2., 3., 4., 5.]));
        let out = tf_a.zeros_default(vec![2, 2, 3]);
        let expected = tf_a.make_default(
            vec![2, 2, 3],
            d(&[2., 4., 6., 12., 15., 18., 28., 32., 36., 50., 55., 60.]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);

        let a = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&(1..=60).map(|i| i as f64).collect::<Vec<_>>()),
        );
        let b = tf_a.make_default(
            vec![2, 2, 3, 1],
            d(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        );
        let out = tf_a.zeros_default(vec![2, 2, 3, 5]);
        let expected = tf_a.make_default(
            vec![2, 2, 3, 5],
            d(&[
                1., 2., 3., 4., 5., 12., 14., 16., 18., 20., 33., 36., 39., 42., 45., 64., 68.,
                72., 76., 80., 105., 110., 115., 120., 125., 156., 162., 168., 174., 180., 217.,
                224., 231., 238., 245., 288., 296., 304., 312., 320., 369., 378., 387., 396., 405.,
                460., 470., 480., 490., 500., 561., 572., 583., 594., 605., 672., 684., 696., 708.,
                720.,
            ]),
        );
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);
    }

    fn test_broadcast_b2a<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let a = tf_a.make_default(vec![2], d(&[2., 2.]));
        let b = tf_a.make_default(vec![2, 2], d(&[1., 2., 3., 4.]));
        let out = tf_a.zeros_default(vec![2, 2]);
        assert_tensor_close!(
            op_mul_out(&a, &b, &out),
            tf_a.make_default(vec![2, 2], d(&[2., 4., 6., 8.]))
        );
    }

    fn test_scalar_input_broadcast<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let a = tf_a.make_default(vec![2], d(&[2., 2.]));
        let b = tf_a.make_default(vec![], d(&[2.]));
        let out = tf_a.make_default(vec![2], d(&[2., 2.]));
        let expected = tf_a.make_default(vec![2], d(&[4., 4.]));
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);
    }

    fn test_both_scalar_input_broadcast<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf_a = TensorFactory::<T>::new();
        let d = |v: &[f64]| -> Vec<T> { v.iter().map(|&x| T::from_f64(x)).collect() };
        let a = tf_a.make_default(vec![1], d(&[2.]));
        let b = tf_a.make_default(vec![], d(&[2.]));
        let out = tf_a.make_default(vec![1], d(&[2.]));
        let expected = tf_a.make_default(vec![1], d(&[4.]));
        assert_tensor_close!(op_mul_out(&a, &b, &out), expected);
        assert_tensor_close!(op_mul_out(&b, &a, &out), expected);
    }

    fn test_complex_dtype<C>(mk: impl Fn(f64, f64) -> C)
    where
        C: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<C>::new();
        let sizes = vec![2, 2];
        let x = tf.make_default(
            sizes.clone(),
            vec![mk(1., 2.), mk(3., 4.), mk(5., 6.), mk(7., 8.)],
        );
        let y = tf.make_default(
            sizes.clone(),
            vec![mk(2., 3.), mk(4., 5.), mk(6., 7.), mk(8., 9.)],
        );
        let expected = tf.make_default(
            sizes.clone(),
            vec![mk(-4., 7.), mk(-8., 31.), mk(-12., 71.), mk(-16., 127.)],
        );
        let out = tf.make_default(
            vec![2, 2],
            vec![mk(0., 0.), mk(0., 0.), mk(0., 0.), mk(0., 0.)],
        );
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected);
    }

    // ---- OpMulOutTest ----

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_all_real_dtypes_supported() {
        test_mul_enumerate_a_types();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_float_tensors() {
        test_floating_point_mul_out::<f32>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_double_tensors() {
        test_floating_point_mul_out::<f64>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_half_tensors() {
        test_floating_point_mul_out::<Half>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_bfloat16_tensors() {
        test_floating_point_mul_out::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_bool_tensors() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_mul_out(
            &tf.make_default(sizes.clone(), vec![true, false, true, true]),
            &tf.ones_default(sizes.clone()),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![true, false, true, true])
        );

        op_mul_out(
            &tf.make_default(sizes.clone(), vec![true, false, true, true]),
            &tf.zeros_default(sizes.clone()),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(sizes.clone(), vec![false, false, false, false])
        );

        op_mul_out(
            &tf.make_default(sizes.clone(), vec![true, false, true, true]),
            &tf.make_default(sizes.clone(), vec![false, false, true, false]),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, vec![false, false, true, false]));
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_optimized_path_ignores_leading_1_dimensions() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_optimized_path_ignores_leading_1_dimensions::<f32>();
        test_optimized_path_ignores_leading_1_dimensions::<f64>();
        test_optimized_path_ignores_leading_1_dimensions::<Half>();
        test_optimized_path_ignores_leading_1_dimensions::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_mismatched_non_broadcastable_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![4, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![8]);
        let mut ctx = context();
        mul_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // Broadcasting multiply drives apply_binary_elementwise_fn over a
    // BroadcastIndexesRange; the per-element product assertions pin its behavior.
    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_a2b() {
        test_broadcast_a2b::<i32>();
        test_broadcast_a2b::<Half>();
        test_broadcast_a2b::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_b2a() {
        test_broadcast_b2a::<i32>();
        test_broadcast_b2a::<Half>();
        test_broadcast_b2a::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_nd() {
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

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_ab2c() {
        let tf_a = TensorFactory::<i32>::new();
        let a = tf_a.make_default(vec![2, 1], vec![1, 2]);
        let b = tf_a.make_default(vec![2, 1, 2], vec![1, 2, 3, 4]);
        let out = tf_a.zeros_default(vec![2, 2, 2]);
        assert_tensor_close!(
            op_mul_out(&a, &b, &out),
            tf_a.make_default(vec![2, 2, 2], vec![1, 2, 2, 4, 3, 4, 6, 8])
        );
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_scalar_input_broadcast() {
        test_scalar_input_broadcast::<i32>();
        test_scalar_input_broadcast::<Half>();
        test_scalar_input_broadcast::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_both_scalar_input_broadcast() {
        test_both_scalar_input_broadcast::<i32>();
        test_both_scalar_input_broadcast::<Half>();
        test_both_scalar_input_broadcast::<BFloat16>();
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_all_complex_dtypes_supported() {
        // ET_FORALL_COMPLEXH_TYPES (non-ATen build)
        test_complex_dtype::<ComplexHalf>(|re, im| ComplexHalf {
            real: Half::from_f64(re),
            imag: Half::from_f64(im),
        });
        test_complex_dtype::<ComplexFloat>(|re, im| ComplexFloat {
            real: re as f32,
            imag: im as f32,
        });
        test_complex_dtype::<ComplexDouble>(|re, im| ComplexDouble { real: re, imag: im });
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_mismatched_output_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];
        let a = tf.ones_default(sizes.clone());
        let b = tf.ones_default(sizes);
        let out = tf.zeros_default(vec![4]);
        let mut ctx = context();
        mul_out(&mut ctx, &a, &b, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dim_size_is_one_ab() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.3200607895851135,
                0.029979348182678223,
                0.27112698554992676,
                0.15423381328582764,
                0.6920414566993713,
                0.005174398422241211,
            ],
        );
        let y = tf.make_default(vec![1, 2], vec![0.9711773991584778, 0.8632034063339233]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3108358085155487,
                0.02587827481329441,
                0.2633123993873596,
                0.13313515484333038,
                0.672095000743866,
                0.004466558340936899,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dim_size_missing_ab() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.3200607895851135,
                0.029979348182678223,
                0.27112698554992676,
                0.15423381328582764,
                0.6920414566993713,
                0.005174398422241211,
            ],
        );
        let y = tf.make_default(vec![2], vec![0.9711773991584778, 0.8632034063339233]);
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3108358085155487,
                0.02587827481329441,
                0.2633123993873596,
                0.13313515484333038,
                0.672095000743866,
                0.004466558340936899,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dim_size_is_one_ba() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 2], vec![0.9711773991584778, 0.8632034063339233]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.3200607895851135,
                0.029979348182678223,
                0.27112698554992676,
                0.15423381328582764,
                0.6920414566993713,
                0.005174398422241211,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3108358085155487,
                0.02587827481329441,
                0.2633123993873596,
                0.13313515484333038,
                0.672095000743866,
                0.004466558340936899,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dim_size_missing_ba() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1, 2], vec![0.9711773991584778, 0.8632034063339233]);
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.3200607895851135,
                0.029979348182678223,
                0.27112698554992676,
                0.15423381328582764,
                0.6920414566993713,
                0.005174398422241211,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                0.3108358085155487,
                0.02587827481329441,
                0.2633123993873596,
                0.13313515484333038,
                0.672095000743866,
                0.004466558340936899,
            ],
        );
        let out = tf.zeros_default(vec![3, 2]);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected_result);
    }

    fn dyn_shape_data<'a>(tf: &'a TensorFactory<f32>) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6910695433616638,
                0.6540696620941162,
                0.8072559237480164,
                0.8218746185302734,
                0.9193597435951233,
                0.4525110721588135,
            ],
        );
        let y = tf.make_default(
            vec![3, 2],
            vec![
                0.9212601184844971,
                0.2030404806137085,
                0.34644562005996704,
                0.4489826560020447,
                0.5666958689689636,
                0.5006863474845886,
            ],
        );
        let expected = tf.make_default(
            vec![3, 2],
            vec![
                0.636654794216156,
                0.13280262053012848,
                0.27967026829719543,
                0.3690074384212494,
                0.5209973454475403,
                0.2265661209821701,
            ],
        );
        (x, y, expected)
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: DISABLED_DynamicShapeUnbound in C++ (dynamic shape unbound not
    // supported); ported and #[ignore]d to match.
    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    #[ignore]
    fn op_mul_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, y, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_mul_out(&x, &y, &out);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_mixed_integer_dtype_matches_aten() {
        let tf_in = TensorFactory::<i8>::new();
        let tf_out = TensorFactory::<i64>::new();
        let in_ = tf_in.make_default(vec![1], vec![100]);
        let out = tf_out.zeros_default(vec![1]);
        op_mul_out(&in_, &in_, &out);
        let expected = tf_out.make_default(vec![1], vec![16]);
        assert_tensor_close!(out, expected);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dimension_mismatch_fix() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![1, 1, 6], vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let out = tf.zeros_default(vec![1, 1, 6]);
        let result = op_mul_out(&a, &b, &out);
        assert_eq!(result.dim(), 3);
        assert_eq!(result.size(0), 1);
        assert_eq!(result.size(1), 1);
        assert_eq!(result.size(2), 6);
        let expected = tf.make_default(vec![1, 1, 6], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_tensor_close!(*result, expected);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dimension_mismatch_reversed() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![1, 1, 6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tf.make_default(vec![6], vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        let out = tf.zeros_default(vec![1, 1, 6]);
        let result = op_mul_out(&a, &b, &out);
        assert_eq!(result.dim(), 3);
        assert_eq!(result.size(0), 1);
        assert_eq!(result.size(1), 1);
        assert_eq!(result.size(2), 6);
        let expected = tf.make_default(vec![1, 1, 6], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_tensor_close!(*result, expected);
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-out-fn/test]
    #[test]
    fn op_mul_out_test_broadcast_dimension_mismatch_with_different_types() {
        {
            let tf = TensorFactory::<Half>::new();
            let d = |v: &[f64]| -> Vec<Half> { v.iter().map(|&x| Half::from_f64(x)).collect() };
            let a = tf.make_default(vec![4], d(&[1.0, 2.0, 3.0, 4.0]));
            let b = tf.make_default(vec![1, 1, 4], d(&[2.0, 2.0, 2.0, 2.0]));
            let out = tf.zeros_default(vec![1, 1, 4]);
            let result = op_mul_out(&a, &b, &out);
            assert_eq!(result.dim(), 3);
            assert_eq!(result.size(0), 1);
            assert_eq!(result.size(1), 1);
            assert_eq!(result.size(2), 4);
            let expected = tf.make_default(vec![1, 1, 4], d(&[2.0, 4.0, 6.0, 8.0]));
            assert_tensor_close!(*result, expected);
        }
        {
            let tf = TensorFactory::<BFloat16>::new();
            let d =
                |v: &[f64]| -> Vec<BFloat16> { v.iter().map(|&x| BFloat16::from_f64(x)).collect() };
            let a = tf.make_default(vec![4], d(&[1.0, 2.0, 3.0, 4.0]));
            let b = tf.make_default(vec![1, 1, 4], d(&[2.0, 2.0, 2.0, 2.0]));
            let out = tf.zeros_default(vec![1, 1, 4]);
            let result = op_mul_out(&a, &b, &out);
            assert_eq!(result.dim(), 3);
            assert_eq!(result.size(0), 1);
            assert_eq!(result.size(1), 1);
            assert_eq!(result.size(2), 4);
            let expected = tf.make_default(vec![1, 1, 4], d(&[2.0, 4.0, 6.0, 8.0]));
            assert_tensor_close!(*result, expected);
        }
        {
            let tf = TensorFactory::<i32>::new();
            let a = tf.make_default(vec![4], vec![1, 2, 3, 4]);
            let b = tf.make_default(vec![1, 1, 4], vec![2, 2, 2, 2]);
            let out = tf.zeros_default(vec![1, 1, 4]);
            let result = op_mul_out(&a, &b, &out);
            assert_eq!(result.dim(), 3);
            assert_eq!(result.size(0), 1);
            assert_eq!(result.size(1), 1);
            assert_eq!(result.size(2), 4);
            let expected = tf.make_default(vec![1, 1, 4], vec![2, 4, 6, 8]);
            assert_tensor_eq!(*result, expected);
        }
    }

    // ---- OpMulScalarOutTest ----

    // [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn/test]
    #[test]
    fn op_mul_scalar_out_test_sanity_check() {
        let tf_a = TensorFactory::<bool>::new();
        let tf_out = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf_out.zeros_default(sizes.clone());
        op_mul_scalar_out(
            &tf_a.make_default(sizes.clone(), vec![true, false, true, false]),
            &Scalar::from_double(2.3),
            &out,
        );
        assert_tensor_eq!(out, tf_out.make_default(sizes, vec![2.3, 0.0, 2.3, 0.0]));
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn/test]
    #[test]
    fn op_mul_scalar_out_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        op_mul_scalar_out(
            &tf.make_default(sizes.clone(), vec![1.3, 2.1, 4.6, 8.2]),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![2.6, 4.2, 9.2, 16.4]));
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn/test]
    #[test]
    fn op_mul_scalar_out_test_half_sanity_check() {
        let tf = TensorFactory::<Half>::new();
        let d = |v: &[f64]| -> Vec<Half> { v.iter().map(|&x| Half::from_f64(x)).collect() };
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        op_mul_scalar_out(
            &tf.make_default(sizes.clone(), d(&[1.3, 2.1, 4.6, 8.2])),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, d(&[2.6, 4.2, 9.2, 16.4])));
    }

    // [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn/test]
    #[test]
    fn op_mul_scalar_out_test_bfloat16_sanity_check() {
        let tf = TensorFactory::<BFloat16>::new();
        let d = |v: &[f64]| -> Vec<BFloat16> { v.iter().map(|&x| BFloat16::from_f64(x)).collect() };
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());
        op_mul_scalar_out(
            &tf.make_default(sizes.clone(), d(&[1.3, 2.1, 4.6, 8.2])),
            &Scalar::from_double(2.0),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, d(&[2.6, 4.2, 9.2, 16.4])));
    }
}
