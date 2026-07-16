//! Literal port of kernels/optimized/cpu/op_add.cpp.
//!
//! DEVIATION: `at::vec::map` over `Vectorized<CTYPE>` collapses to a scalar
//! loop (PORTING.md optimized-kernel substitution).

use crate::kernels::optimized::cpu::op_add_sub_impl::opt_add_sub_out_impl;
use crate::kernels::portable::cpu::scalar_utils::{
    extract_scalar, get_scalar_dtype, promote_type_with_scalar, scalar_to,
};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::{StaticCast, SupportedTensorDtypes};
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_unitensor_elementwise_fn, get_compute_type,
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

// PORT-NOTE: complex `x + Vec(alpha_val * b_val)` over `Complex<T>` — mirrors
// the `ComplexArith` trait from kernels/portable/cpu/op_add.rs.
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

// PORT-NOTE: REALB `x + Vec(alpha_val * b_casted)` — bool promotes via i32.
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

// [spec:et:def:op-add.torch.executor.native.opt-add-out-fn]
// [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn]
pub fn opt_add_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let out_type = out.scalar_type();
    let common_type = promote_types(a_type, b_type, false);

    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out_type) && check_alpha_type(get_scalar_dtype(*alpha), common_type),
        InvalidArgument,
        out
    );

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

    let op_name = "add.out";

    if b.numel() == 1 {
        if is_complex_type(a_type) || is_complex_type(b_type) || is_complex_type(out_type) {
            // TODO: The current support for complex dtype enforces that input
            // and output tensors have the same dtype. Support mixed dtypes in
            // the future.
            crate::et_kernel_check!(
                ctx,
                a_type == b_type && a_type == out_type,
                InvalidArgument,
                out
            );

            crate::et_switch_complexh_types!(out_type, ctx, op_name, CTYPE, {
                let alpha_val: CTYPE = <CTYPE as ComplexArith>::from_scalar(alpha);
                let b_val: CTYPE = unsafe { *b.const_data_ptr::<CTYPE>() };

                // DEVIATION: at::vec::map -> scalar loop; x + (alpha_val * b_val).
                let out_data = out.mutable_data_ptr::<CTYPE>();
                let a_data = a.const_data_ptr::<CTYPE>();
                let prod = alpha_val.c_mul(b_val);
                for i in 0..out.numel() {
                    unsafe {
                        *out_data.offset(i) = (*a_data.offset(i)).c_add(prod);
                    }
                }
            });
            return out;
        } else if a_type == b_type
            && a_type == out_type
            && a_type != ScalarType::Half
            && a_type != ScalarType::BFloat16
        {
            crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
                crate::et_switch_realb_types!(b_type, ctx, op_name, CTYPE_B, {
                    let mut alpha_val: CTYPE = Default::default();
                    crate::et_kernel_check!(
                        ctx,
                        extract_scalar(*alpha, &mut alpha_val),
                        InvalidArgument,
                        out
                    );
                    let b_val: CTYPE_B = unsafe { *b.const_data_ptr::<CTYPE_B>() };
                    let b_casted: CTYPE = <CTYPE as StaticCast<CTYPE_B>>::static_cast(b_val);

                    // DEVIATION: at::vec::map -> scalar loop; x + (alpha_val * b_casted).
                    let out_data = out.mutable_data_ptr::<CTYPE>();
                    let a_data = a.const_data_ptr::<CTYPE>();
                    let prod = alpha_val.rmul(b_casted);
                    for i in 0..out.numel() {
                        unsafe {
                            *out_data.offset(i) = (*a_data.offset(i)).radd(prod);
                        }
                    }
                });
            });
            return out;
        }
    } else if a.numel() == 1 {
        return opt_add_out(ctx, b, a, alpha, out);
    }

    opt_add_sub_out_impl(ctx, a, b, alpha, out, false, op_name)
}

// [spec:et:def:op-add.torch.executor.native.opt-add-scalar-out-fn]
// [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn]
pub fn opt_add_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let common_type = promote_type_with_scalar(a_type, *b, false);
    let out_type = out.scalar_type();

    crate::et_kernel_check!(
        ctx,
        common_type == a_type && check_alpha_type(get_scalar_dtype(*alpha), common_type),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let op_name = "add.Scalar_out";

    if a_type == common_type
        && a_type == out_type
        && a_type != ScalarType::Half
        && a_type != ScalarType::BFloat16
    {
        crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
            let b_casted: CTYPE = scalar_to::<CTYPE>(b);
            let mut alpha_val: CTYPE = Default::default();
            crate::et_kernel_check!(
                ctx,
                extract_scalar(*alpha, &mut alpha_val),
                InvalidArgument,
                out
            );

            // DEVIATION: at::vec::map -> scalar loop; x + (alpha_val * b_casted).
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let prod = alpha_val.rmul(b_casted);
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).radd(prod);
                }
            }
        });
    } else {
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

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
    }

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
    use crate::runtime::core::portable_type::{BFloat16, ComplexDouble, ComplexFloat, ComplexHalf};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_add_out<'a, 'b>(
        a: &'a Tensor<'b>,
        b: &'a Tensor<'b>,
        alpha: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        opt_add_out(&mut ctx, a, b, alpha, out)
    }

    fn op_add_scalar_out<'a, 'b>(
        a: &Tensor,
        b: &Scalar,
        alpha: &Scalar,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        opt_add_scalar_out(&mut ctx, a, b, alpha, out)
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
            Half::from_f64(v)
        }
    }
    impl FromF64Elem for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f64(v)
        }
    }

    fn d<T: FromF64Elem>(v: &[f64]) -> Vec<T> {
        v.iter().map(|&x| T::from_f64(x)).collect()
    }

    // op_add_test.cpp `test_add<DTYPE, DTYPE, DTYPE>` (same-dtype instances hit
    // the optimized kTreatAs1d path; Half/BFloat16 fall back to the
    // elementwise-util path because select_optimized_path rejects them).
    fn test_add<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());
        op_add_out(
            &tf.make_default(sizes.clone(), d(&[1.0, 2.0, 4.0, 8.0])),
            &tf.ones_default(sizes.clone()),
            &Scalar::from_i64(1),
            &out,
        );
        assert_tensor_eq!(out, tf.make_default(sizes, d(&[2.0, 3.0, 5.0, 9.0])));
    }

    // op_add_test.cpp `test_floating_point_add_out<DTYPE>`.
    fn test_floating_point_add_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();
        let sizes = vec![2, 2];

        let out = tf.zeros_default(sizes.clone());
        op_add_out(
            &tf.make_default(sizes.clone(), d(&[1.25, 2.25, 4.5, 8.875])),
            &tf.ones_default(sizes.clone()),
            &Scalar::from_double(1.25),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, d(&[2.5, 3.5, 5.75, 10.125])));
    }

    fn test_add_complex_dtype<C>(mk: impl Fn(f64, f64) -> C)
    where
        C: CppTypeToScalarType + FactoryValue,
    {
        let tf = TensorFactory::<C>::new();

        // Both inputs have the same shape (kTreatAs1d complex path in the
        // shared impl).
        let x_0 = tf.make_default(vec![2], vec![mk(1.0, 2.1), mk(3.1, 4.0)]);
        let y_0 = tf.make_default(vec![2], vec![mk(5.2, 6.3), mk(7.0, 8.9)]);
        let out = tf.full(vec![2], mk(0.0, 0.0), TensorShapeDynamism::STATIC);
        op_add_out(&x_0, &y_0, &Scalar::from_i64(1), &out);
        let expected_0 = tf.make_default(vec![2], vec![mk(6.2, 8.4), mk(10.1, 12.9)]);
        assert_tensor_eq!(out, expected_0);

        // Other tensor has numel() == 1 (opt_add_out's own complex branch).
        let y_1 = tf.make_default(vec![1], vec![mk(2.0, 3.0)]);
        op_add_out(&x_0, &y_1, &Scalar::from_i64(2), &out);
        let expected_1 = tf.make_default(vec![2], vec![mk(5.0, 8.1), mk(7.1, 10.0)]);
        assert_tensor_eq!(out, expected_1);
    }

    // ---- OpAddOutKernelTest ----

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_all_real_dtypes_supported() {
        test_add::<u8>();
        test_add::<i8>();
        test_add::<i16>();
        test_add::<i32>();
        test_add::<i64>();
        test_add::<f32>();
        test_add::<f64>();
        test_add::<Half>();
        test_add::<BFloat16>();
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_complex_tensors() {
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

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_float_tensors() {
        test_floating_point_add_out::<f32>();
        test_floating_point_add_out::<f64>();
        test_floating_point_add_out::<Half>();
        test_floating_point_add_out::<BFloat16>();
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_and_int_input_tensor() {
        let tf = TensorFactory::<bool>::new();
        let tfi = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tfi.make_default(sizes.clone(), vec![2, 4, 3, 3]);
        let out = tfi.zeros_default(sizes.clone());

        let mut ctx = context();
        opt_add_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, tfi.make_default(sizes, vec![2, 5, 3, 4]));
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_and_bool_input_tensor() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];

        let a = tf.make_default(sizes.clone(), vec![false, true, false, true]);
        let b = tf.make_default(sizes.clone(), vec![false, true, true, true]);
        let out = tf.zeros_default(sizes.clone());

        op_add_out(&a, &b, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, tf.make_default(sizes, vec![false, true, true, true]));
    }

    // op_add_test.cpp BroadcastOneElementTensor: b.numel() == 1 fast path plus
    // the a.numel() == 1 swapped recursion.
    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_tensor() {
        let tf = TensorFactory::<f32>::new();
        let x = tf.make_default(vec![1], vec![1.75]);
        let y = tf.make_default(vec![3, 2], vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.5]);

        let out = tf.zeros_default(vec![3, 2]);
        let expected = tf.make_default(vec![3, 2], vec![0.25, 0.75, 1.25, 1.75, 2.25, 3.25]);

        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, expected);

        op_add_out(&y, &x, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_tensor_type_promotion() {
        let tf = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let x = tf_double.make_default(vec![1], vec![1.75]);
        let y = tf.make_default(vec![3, 2], vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.5]);

        let out = tf_double.zeros_default(vec![3, 2]);
        let expected = tf_double.make_default(vec![3, 2], vec![0.25, 0.75, 1.25, 1.75, 2.25, 3.25]);

        op_add_out(&x, &y, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, expected);

        op_add_out(&y, &x, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_one_element_rank0_tensor() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(vec![1], vec![5.0]);
        let b = tf.make_default(vec![], vec![2.0]);

        let out = tf.zeros_default(vec![1]);
        op_add_out(&a, &b, &Scalar::from_i64(1), &out);

        let ret = tf.make_default(vec![1], vec![7.0]);
        assert_tensor_eq!(out, ret);

        op_add_out(&b, &a, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, ret);
    }

    // op_add_test.cpp test_broadcast_3D<Float> (kBroadcastNdByNd and reverse).
    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_nd() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(
            vec![2, 2, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = tf.make_default(vec![2, 1, 3], vec![2., 3., 4., 5., 6., 7.]);

        let out = tf.zeros_default(vec![2, 2, 3]);
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![3., 5., 7., 6., 8., 10., 12., 14., 16., 15., 17., 19.],
        );
        op_add_out(&a, &b, &Scalar::from_double(1.0), &out);
        assert_tensor_close!(out, expected);

        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![3.5, 6., 8.5, 8., 10.5, 13., 15.5, 18., 20.5, 20., 22.5, 25.],
        );
        op_add_out(&b, &a, &Scalar::from_double(1.5), &out);
        assert_tensor_close!(out, expected);
    }

    // op_add_test.cpp test_broadcast_last_dim<Float> (kBroadcastLastDim and
    // reverse).
    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_last_dim() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.make_default(
            vec![4, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = tf.make_default(vec![4, 1], vec![2., 3., 4., 5.]);

        let out = tf.zeros_default(vec![4, 3]);
        let expected = tf.make_default(
            vec![4, 3],
            vec![3., 4., 5., 7., 8., 9., 11., 12., 13., 15., 16., 17.],
        );

        op_add_out(&a, &b, &Scalar::from_double(1.0), &out);
        assert_tensor_close!(out, expected);
        op_add_out(&b, &a, &Scalar::from_double(1.0), &out);
        assert_tensor_close!(out, expected);
    }

    // op_add_test.cpp BroadcastSupported: a [5,1,3,1], b [2,1,4] (kNone -> the
    // portable elementwise fallback inside the shared impl).
    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_broadcast_supported() {
        let tf = TensorFactory::<f32>::new();

        let a = tf.zeros_default(vec![5, 1, 3, 1]);
        let b = tf.ones_default(vec![2, 1, 4]);
        let out = tf.zeros_default(vec![5, 2, 3, 4]);

        op_add_out(&a, &b, &Scalar::from_i64(1), &out);
        assert_tensor_eq!(out, tf.ones_default(vec![5, 2, 3, 4]));
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_int_inputs_float_alpha_dies() {
        let tf = TensorFactory::<i32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        opt_add_out(
            &mut ctx,
            &tf.ones_default(sizes.clone()),
            &tf.ones_default(sizes),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_bool_inputs_float_alpha_dies() {
        let tf = TensorFactory::<bool>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        let mut ctx = context();
        opt_add_out(
            &mut ctx,
            &tf.ones_default(sizes.clone()),
            &tf.ones_default(sizes),
            &Scalar::from_double(0.7),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_int_output_with_float_input_dies() {
        let tfi = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];

        let a = tfi.make_default(sizes.clone(), vec![2, 4, 3, 3]);
        let b = tff.make_default(sizes.clone(), vec![2., 4., 3., 3.]);
        let out = tfi.zeros_default(sizes);

        let mut ctx = context();
        opt_add_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn/test]
    #[test]
    fn op_add_out_kernel_test_mismatched_non_broadcastable_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.ones_default(vec![4, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![4, 2]);

        let mut ctx = context();
        opt_add_out(&mut ctx, &a, &b, &Scalar::from_i64(1), &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // ---- OpAddScalarOutKernelTest ----

    // [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn/test]
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
        assert_tensor_eq!(out, tf.make_default(sizes, vec![3, 4, 6, 10]));
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let sizes = vec![2, 2];
        let out = tf.zeros_default(sizes.clone());

        op_add_scalar_out(
            &tf.make_default(sizes.clone(), vec![1.3, 2.1, 4.6, 8.2]),
            &Scalar::from_double(1.9),
            &Scalar::from_double(2.8),
            &out,
        );
        assert_tensor_close!(out, tf.make_default(sizes, vec![6.62, 7.42, 9.92, 13.52]));
    }

    // op_add_test.cpp DtypeTest_float16_bool_int_float16 (Half input takes the
    // elementwise-util branch of opt_add_scalar_out).
    // [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_dtype_test_float16_bool_int_float16() {
        let tf_half = TensorFactory::<Half>::new();

        let self_ = tf_half.ones_default(vec![2, 2]);
        let out = tf_half.zeros_default(vec![2, 2]);
        let out_expected =
            tf_half.full(vec![2, 2], Half::from_f32(2.0), TensorShapeDynamism::STATIC);
        op_add_scalar_out(&self_, &Scalar::from_bool(true), &Scalar::from_i64(1), &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn/test]
    #[test]
    fn op_add_scalar_out_kernel_test_int_tensor_floating_point_alpha_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_add_scalar_out(
            &mut ctx,
            &a,
            &Scalar::from_i64(1),
            &Scalar::from_double(2.2),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }
}
