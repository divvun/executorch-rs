//! Literal port of kernels/optimized/cpu/op_add_sub_impl.h.
//!
//! DEVIATION: the C++ `at::vec::map2` / `handle_broadcast_elementwise` dispatch
//! `Vectorized<CTYPE>` lambdas. Per PORTING.md's optimized-kernel substitution,
//! the SIMD lane collapses to scalar `CTYPE`: `at::vec::map2` becomes a scalar
//! loop and the broadcast lambdas take/return scalar `CTYPE`.

use crate::kernels::optimized::cpu::binary_ops::{
    ElementwiseOptimizedPath, handle_broadcast_elementwise, select_optimized_path,
};
use crate::kernels::portable::cpu::scalar_utils::extract_scalar;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::exec_aten::util::scalar_type_util::{is_complex_type, promote_types};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::{Complex, Half};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the complex kTreatAs1d path computes `x + Vec(alpha_val) * y` over
// `Complex<T>`. Mirrors the `ComplexArith` trait established in
// kernels/portable/cpu/op_add.rs (c10::complex `operator+`/`operator*` and the
// scalar->complex construction with imag=0), plus a unary negate for `is_sub`.
trait ComplexArith: Copy {
    fn c_add(self, other: Self) -> Self;
    fn c_mul(self, other: Self) -> Self;
    fn c_neg(self) -> Self;
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
            fn c_neg(self) -> Self {
                Complex {
                    real: $from(-$to(self.real)),
                    imag: $from(-$to(self.imag)),
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

// PORT-NOTE: the REALB compute set includes `Bool`, where C++ integer-promotes
// bool operands before arithmetic and truncates back. Rust `bool` has no
// `+`/`-`/`*`/unary `-`; this trait reproduces the promotion (mirrors op_add.rs).
trait RealbArith: Copy {
    fn radd(self, other: Self) -> Self;
    fn rsub(self, other: Self) -> Self;
    fn rmul(self, other: Self) -> Self;
    fn rneg(self) -> Self;
}
macro_rules! impl_realb_arith_prim {
    ($($t:ty),*) => {$(
        impl RealbArith for $t {
            fn radd(self, other: Self) -> Self { self + other }
            fn rsub(self, other: Self) -> Self { self - other }
            fn rmul(self, other: Self) -> Self { self * other }
            fn rneg(self) -> Self { (0 as $t) - self }
        }
    )*};
}
impl_realb_arith_prim!(u8, i8, i16, i32, i64, f32, f64);
impl RealbArith for bool {
    fn radd(self, other: Self) -> Self {
        ((self as i32) + (other as i32)) != 0
    }
    fn rsub(self, other: Self) -> Self {
        ((self as i32) - (other as i32)) != 0
    }
    fn rmul(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
    fn rneg(self) -> Self {
        (-(self as i32)) != 0
    }
}

// [spec:et:def:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn]
// [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn]
//
// PORT-NOTE: the C++ template parameters `<bool is_sub, const char* op_name>`
// become ordinary runtime arguments; the ported switch macros already take a
// runtime op-name string, and `is_sub` gates the same branches at runtime.
pub fn opt_add_sub_out_impl<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
    is_sub: bool,
    op_name: &str,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let out_type = out.scalar_type();

    let selected_optimized_path = select_optimized_path(a, b, out);

    if is_complex_type(a_type) || is_complex_type(b_type) || is_complex_type(out_type) {
        // TODO: The current implementation for complex dtypes enforces that the
        // inputs and output tensors have same dtype and shape. Handle mixed
        // dtypes and broadcasting in the future.
        crate::et_kernel_check!(
            ctx,
            a_type == b_type
                && a_type == out_type
                && selected_optimized_path == ElementwiseOptimizedPath::KTreatAs1d,
            InvalidArgument,
            out
        );
        crate::et_switch_complexh_types!(out_type, ctx, op_name, CTYPE, {
            let mut alpha_val: CTYPE = <CTYPE as ComplexArith>::from_scalar(alpha);
            if is_sub {
                alpha_val = alpha_val.c_neg();
            }
            // DEVIATION: at::vec::map2 -> scalar loop; x + alpha_val * y.
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) =
                        (*a_data.offset(i)).c_add(alpha_val.c_mul(*b_data.offset(i)));
                }
            }
        });
        return out;
    }

    if selected_optimized_path == ElementwiseOptimizedPath::KTreatAs1d {
        crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
            let mut alpha_val: CTYPE = Default::default();
            crate::et_kernel_check!(
                ctx,
                extract_scalar(*alpha, &mut alpha_val),
                InvalidArgument,
                out
            );
            if is_sub {
                alpha_val = alpha_val.rneg();
            }
            // DEVIATION: at::vec::map2 -> scalar loop; x + alpha_val * y.
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) =
                        (*a_data.offset(i)).radd(alpha_val.rmul(*b_data.offset(i)));
                }
            }
        });
    } else if selected_optimized_path != ElementwiseOptimizedPath::KNone {
        // Cannot apply the trick of -alpha here because alpha is Scalar without
        // support for - operator. At least not right now.
        crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
            let mut alpha_val: CTYPE = Default::default();
            crate::et_kernel_check_msg!(
                ctx,
                extract_scalar(*alpha, &mut alpha_val),
                InvalidArgument,
                out,
                "Failed to extract scalar alpha."
            );
            let reverse = selected_optimized_path
                == ElementwiseOptimizedPath::KBroadcast2dBy1dReverseArguments
                || selected_optimized_path
                    == ElementwiseOptimizedPath::KBroadcastLastDimReverseArguments
                || selected_optimized_path
                    == ElementwiseOptimizedPath::KBroadcastNdByNdReverseArguments;
            if is_sub {
                if reverse {
                    let add_lambda = move |x: CTYPE, y: CTYPE| y.rsub(alpha_val.rmul(x));
                    handle_broadcast_elementwise::<CTYPE, _>(
                        ctx,
                        &add_lambda,
                        a,
                        b,
                        out,
                        selected_optimized_path,
                        Some(*alpha),
                    );
                } else {
                    let add_lambda = move |x: CTYPE, y: CTYPE| x.rsub(alpha_val.rmul(y));
                    handle_broadcast_elementwise::<CTYPE, _>(
                        ctx,
                        &add_lambda,
                        a,
                        b,
                        out,
                        selected_optimized_path,
                        Some(*alpha),
                    );
                }
            } else if reverse {
                // Reason we swap out args here is because
                // handle_broadcast_elementwise handles this
                // selected_optimized_path option a bit differently.
                let add_lambda = move |x: CTYPE, y: CTYPE| y.radd(alpha_val.rmul(x));
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &add_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    Some(*alpha),
                );
            } else {
                let add_lambda = move |x: CTYPE, y: CTYPE| x.radd(alpha_val.rmul(y));
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &add_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    Some(*alpha),
                );
            }
        });
    } else {
        let common_type = promote_types(a_type, b_type, false);
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let mut val_alpha: CTYPE_COMPUTE = Default::default();
            crate::et_kernel_check!(
                ctx,
                extract_scalar(*alpha, &mut val_alpha),
                InvalidArgument,
                out
            );
            if is_sub {
                val_alpha = val_alpha.rneg();
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::ComplexFloat;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // out = a + alpha * b over the same-shape (kTreatAs1d) fast path.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_add_treat_as_1d() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1., 2., 4., 8.]);
        let b = tf.make_default(vec![2, 2], vec![3., 5., 7., 9.]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            false,
            "add.out",
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![7., 12., 18., 26.]));
    }

    // is_sub negates alpha: out = a - alpha * b on the same fast path.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_sub_treat_as_1d() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1., 2., 4., 8.]);
        let b = tf.make_default(vec![2, 2], vec![3., 5., 7., 9.]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            true,
            "sub.out",
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![-5., -8., -10., -10.]));
    }

    // Integer dtype on the fast path.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_int_treat_as_1d() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![4], vec![10, 20, 30, 40]);
        let b = tf.make_default(vec![4], vec![1, 2, 3, 4]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(3),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_eq!(out, tf.make_default(vec![4], vec![7, 14, 21, 28]));
    }

    // kBroadcast2dBy1d: a [2,3] op b [3], plus the ReverseArguments variant.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_broadcast_2d_by_1d() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = tf.make_default(vec![3], vec![10., 20., 30.]);
        let out = tf.zeros_default(vec![2, 3]);

        // add: a + 2*b
        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            false,
            "add.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![21., 42., 63., 24., 45., 66.])
        );

        // add, reversed arguments: b + 2*a
        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &b,
            &a,
            &Scalar::from_i64(2),
            &out,
            false,
            "add.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![12., 24., 36., 18., 30., 42.])
        );

        // sub: a - 2*b
        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![-19., -38., -57., -16., -35., -54.])
        );

        // sub, reversed arguments: b - 2*a
        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &b,
            &a,
            &Scalar::from_i64(2),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![8., 16., 24., 2., 10., 18.])
        );
    }

    // kBroadcastLastDim: a [2,3] op b [2,1], plus reverse.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_broadcast_last_dim() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = tf.make_default(vec![2, 1], vec![10., 20.]);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(1),
            &out,
            false,
            "add.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![11., 12., 13., 24., 25., 26.])
        );

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &b,
            &a,
            &Scalar::from_i64(1),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 3], vec![9., 8., 7., 16., 15., 14.])
        );
    }

    // kBroadcastNdByNd: a [2,2,3] op b [2,1,3], plus reverse.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_broadcast_nd_by_nd() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(
            vec![2, 2, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        let b = tf.make_default(vec![2, 1, 3], vec![2., 3., 4., 5., 6., 7.]);
        let out = tf.zeros_default(vec![2, 2, 3]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_double(1.0),
            &out,
            false,
            "add.out",
        );
        assert_tensor_close!(
            out,
            tf.make_default(
                vec![2, 2, 3],
                vec![3., 5., 7., 6., 8., 10., 12., 14., 16., 15., 17., 19.]
            )
        );

        // Reverse arguments with sub: b - a (alpha 1).
        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &b,
            &a,
            &Scalar::from_double(1.0),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_close!(
            out,
            tf.make_default(
                vec![2, 2, 3],
                vec![1., 1., 1., -2., -2., -2., -2., -2., -2., -5., -5., -5.]
            )
        );
    }

    // Mixed dtypes force the kNone fallback (portable elementwise path with the
    // promoted compute type).
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_mixed_dtype_fallback() {
        let tfi = TensorFactory::<i32>::new();
        let tff = TensorFactory::<f32>::new();
        let a = tfi.make_default(vec![2, 2], vec![1, 2, 3, 4]);
        let b = tff.make_default(vec![2, 2], vec![0.5, 1.5, 2.5, 3.5]);
        let out = tff.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            false,
            "add.out",
        );
        assert_tensor_close!(out, tff.make_default(vec![2, 2], vec![2., 5., 8., 11.]));

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(2),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_close!(out, tff.make_default(vec![2, 2], vec![0., -1., -2., -3.]));
    }

    // Complex same-shape path: out = a + alpha * b / a - alpha * b.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_complex_treat_as_1d() {
        let tf = TensorFactory::<ComplexFloat>::new();
        let mk = |re: f32, im: f32| ComplexFloat { real: re, imag: im };
        let a = tf.make_default(vec![2], vec![mk(1.0, 2.0), mk(3.0, 4.0)]);
        let b = tf.make_default(vec![2], vec![mk(5.0, 6.0), mk(7.0, 8.0)]);
        let out = tf.full(vec![2], mk(0.0, 0.0), TensorShapeDynamism::STATIC);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(1),
            &out,
            false,
            "add.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2], vec![mk(6.0, 8.0), mk(10.0, 12.0)])
        );

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(1),
            &out,
            true,
            "sub.out",
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2], vec![mk(-4.0, -4.0), mk(-4.0, -4.0)])
        );
    }

    // Complex operands with mismatched shapes are rejected (the complex branch
    // requires kTreatAs1d and equal dtypes).
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_complex_broadcast_dies() {
        let tf = TensorFactory::<ComplexFloat>::new();
        let mk = |re: f32, im: f32| ComplexFloat { real: re, imag: im };
        let a = tf.make_default(vec![2, 1], vec![mk(1.0, 2.0), mk(3.0, 4.0)]);
        let b = tf.make_default(vec![1], vec![mk(5.0, 6.0)]);
        let out = tf.full(vec![2, 1], mk(0.0, 0.0), TensorShapeDynamism::STATIC);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(1),
            &out,
            false,
            "add.out",
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // Bool operands on the fast path integer-promote before arithmetic.
    // [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn/test]
    #[test]
    fn opt_add_sub_out_impl_bool_treat_as_1d() {
        let tf = TensorFactory::<bool>::new();
        let a = tf.make_default(vec![4], vec![false, true, false, true]);
        let b = tf.make_default(vec![4], vec![false, false, true, true]);
        let out = tf.zeros_default(vec![4]);

        let mut ctx = context();
        opt_add_sub_out_impl(
            &mut ctx,
            &a,
            &b,
            &Scalar::from_i64(1),
            &out,
            false,
            "add.out",
        );
        assert_tensor_eq!(out, tf.make_default(vec![4], vec![false, true, true, true]));
    }
}
