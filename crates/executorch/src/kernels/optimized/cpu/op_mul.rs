//! Literal port of kernels/optimized/cpu/op_mul.cpp.
//!
//! DEVIATION: `at::vec::map` / `at::vec::map2` / `handle_broadcast_elementwise`
//! over `Vectorized<CTYPE>` collapse to scalar loops (PORTING.md optimized-
//! kernel substitution).

use crate::kernels::optimized::cpu::binary_ops::{
    ElementwiseOptimizedPath, handle_broadcast_elementwise, select_optimized_path,
};
use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::{
    apply_binary_elementwise_fn, resize_to_broadcast_target_size,
};
use crate::kernels::portable::cpu::util::dtype_util::{StaticCast, SupportedTensorDtypes};
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
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

// PORT-NOTE: complex `x * y` over `Complex<T>` — mirrors op_add.rs's ComplexArith.
trait ComplexMul: Copy {
    fn c_mul(self, other: Self) -> Self;
}
macro_rules! impl_complex_mul {
    ($comp:ty, $to:expr, $from:expr) => {
        impl ComplexMul for Complex<$comp> {
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
        }
    };
}
impl_complex_mul!(Half, |x: Half| x.to_f64(), |x: f64| Half::from_f64(x));
impl_complex_mul!(f32, |x: f32| x as f64, |x: f64| x as f32);
impl_complex_mul!(f64, |x: f64| x, |x: f64| x);

// PORT-NOTE: REALB `x * y` — bool promotes via i32.
trait RealbMul: Copy {
    fn rmul(self, other: Self) -> Self;
}
macro_rules! impl_realb_mul_prim {
    ($($t:ty),*) => {$(
        impl RealbMul for $t {
            fn rmul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_realb_mul_prim!(u8, i8, i16, i32, i64, f32, f64);
impl RealbMul for bool {
    fn rmul(self, other: Self) -> Self {
        ((self as i32) * (other as i32)) != 0
    }
}

// [spec:et:def:op-mul.torch.executor.native.opt-mul-out-fn]
// [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn]
pub fn opt_mul_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let out_type = out.scalar_type();
    let common_type = promote_types(a_type, b_type, false);

    crate::et_kernel_check!(ctx, can_cast(common_type, out_type), InvalidArgument, out);

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

    let op_name = "mul.out";

    if b.numel() == 1 {
        if a_type == b_type
            && a_type == out_type
            && a_type != ScalarType::Half
            && a_type != ScalarType::BFloat16
        {
            crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
                crate::et_switch_realb_types!(b_type, ctx, op_name, CTYPE_B, {
                    let b_val: CTYPE_B = unsafe { *b.const_data_ptr::<CTYPE_B>() };
                    let b_casted: CTYPE = <CTYPE as StaticCast<CTYPE_B>>::static_cast(b_val);

                    // DEVIATION: at::vec::map -> scalar loop; x * b_casted.
                    let out_data = out.mutable_data_ptr::<CTYPE>();
                    let a_data = a.const_data_ptr::<CTYPE>();
                    for i in 0..out.numel() {
                        unsafe {
                            *out_data.offset(i) = (*a_data.offset(i)).rmul(b_casted);
                        }
                    }
                });
            });
            return out;
        }
    } else if a.numel() == 1 {
        return opt_mul_out(ctx, b, a, out);
    }

    let selected_optimized_path = select_optimized_path(a, b, out);
    if selected_optimized_path == ElementwiseOptimizedPath::KTreatAs1d {
        if is_complex_type(out_type) {
            crate::et_kernel_check!(
                ctx,
                a_type == b_type && a_type == out_type,
                InvalidArgument,
                out
            );

            crate::et_switch_complexh_types!(out_type, ctx, op_name, CTYPE, {
                // DEVIATION: at::vec::map2 -> scalar loop; x * y (complex).
                let out_data = out.mutable_data_ptr::<CTYPE>();
                let a_data = a.const_data_ptr::<CTYPE>();
                let b_data = b.const_data_ptr::<CTYPE>();
                for i in 0..out.numel() {
                    unsafe {
                        *out_data.offset(i) = (*a_data.offset(i)).c_mul(*b_data.offset(i));
                    }
                }
            });
        } else {
            crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
                // DEVIATION: at::vec::map2 -> scalar loop; x * y.
                let out_data = out.mutable_data_ptr::<CTYPE>();
                let a_data = a.const_data_ptr::<CTYPE>();
                let b_data = b.const_data_ptr::<CTYPE>();
                for i in 0..out.numel() {
                    unsafe {
                        *out_data.offset(i) = (*a_data.offset(i)).rmul(*b_data.offset(i));
                    }
                }
            });
        }
    } else if selected_optimized_path != ElementwiseOptimizedPath::KNone {
        if is_complex_type(out_type) {
            crate::et_kernel_check!(
                ctx,
                a_type == b_type && a_type == out_type,
                InvalidArgument,
                out
            );

            crate::et_switch_complexh_types!(out_type, ctx, op_name, CTYPE, {
                let mul_lambda = |x: CTYPE, y: CTYPE| x.c_mul(y);
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &mul_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    None,
                );
            });
        } else {
            crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
                let mul_lambda = |x: CTYPE, y: CTYPE| x.rmul(y);
                handle_broadcast_elementwise::<CTYPE, _>(
                    ctx,
                    &mul_lambda,
                    a,
                    b,
                    out,
                    selected_optimized_path,
                    None,
                );
            });
        }
    } else {
        if is_complex_type(a_type) || is_complex_type(b_type) || is_complex_type(out_type) {
            crate::et_kernel_check!(
                ctx,
                a_type == b_type && a_type == out_type,
                InvalidArgument,
                out
            );

            crate::et_switch_complexh_types!(out_type, ctx, op_name, CTYPE, {
                apply_binary_elementwise_fn::<CTYPE, CTYPE, CTYPE, _>(
                    |val_a: CTYPE, val_b: CTYPE| val_a.c_mul(val_b),
                    a,
                    b,
                    out,
                );
            });
        } else {
            let mut common_type_mut = common_type;
            let compute_type = get_compute_type(&mut common_type_mut);

            crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
                apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                    |vals: &[CTYPE_COMPUTE]| vals[0].rmul(vals[1]),
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
    }

    out
}

// [spec:et:def:op-mul.torch.executor.native.opt-mul-scalar-out-fn]
// [spec:et:sem:op-mul.torch.executor.native.opt-mul-scalar-out-fn]
pub fn opt_mul_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let common_type = promote_type_with_scalar(a_type, *b, false);
    let out_type = out.scalar_type();

    crate::et_kernel_check!(ctx, common_type == out_type, InvalidArgument, out);

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

    let op_name = "mul.Scalar_out";

    if a_type == common_type
        && a_type == out_type
        && a_type != ScalarType::Half
        && a_type != ScalarType::BFloat16
    {
        crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
            let b_casted: CTYPE = scalar_to::<CTYPE>(b);

            // DEVIATION: at::vec::map -> scalar loop; x * b_casted.
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).rmul(b_casted);
                }
            }
        });
    } else {
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
            apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                move |vals: &[CTYPE_COMPUTE]| vals[0].rmul(val_b),
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
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::{assert_tensor_close, assert_tensor_eq};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
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

    // OpMulOutTest.FloatTensors (same shape -> kTreatAs1d fast path).
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_float_tensors() {
        let tf = TensorFactory::<f32>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let mut ctx = context();

        opt_mul_out(
            &mut ctx,
            &tf.make_default(vec![2, 2], vec![1.25, 2.5, 4.75, 8.875]),
            &tf.ones_default(vec![2, 2]),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2], vec![1.25, 2.5, 4.75, 8.875])
        );

        opt_mul_out(
            &mut ctx,
            &tf.make_default(vec![2, 2], vec![1.25, 2.5, 4.75, 8.875]),
            &tf.make_default(vec![2, 2], vec![1.25, 2.5, 4.75, 8.875]),
            &out,
        );
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2], vec![1.5625, 6.25, 22.5625, 78.765625])
        );
    }

    // b.numel() == 1 vectorized fast path, and the swapped-argument recursion
    // when a.numel() == 1.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_scalar_tensor_fast_path() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.25, 2.5, 4.75, 8.875]);
        let b = tf.make_default(vec![1], vec![2.0]);
        let out = tf.zeros_default(vec![2, 2]);
        let expected = tf.make_default(vec![2, 2], vec![2.5, 5.0, 9.5, 17.75]);

        let mut ctx = context();
        opt_mul_out(&mut ctx, &a, &b, &out);
        assert_tensor_close!(out, expected);

        // a.numel() == 1: recurses with the operands swapped.
        let out2 = tf.zeros_default(vec![2, 2]);
        opt_mul_out(&mut ctx, &b, &a, &out2);
        assert_tensor_close!(out2, expected);
    }

    // OpMulOutTest.BoolTensors.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_bool_tensors() {
        let tf = TensorFactory::<bool>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let mut ctx = context();

        opt_mul_out(
            &mut ctx,
            &tf.make_default(vec![2, 2], vec![true, false, true, true]),
            &tf.make_default(vec![2, 2], vec![false, false, true, false]),
            &out,
        );
        assert_tensor_eq!(
            out,
            tf.make_default(vec![2, 2], vec![false, false, true, false])
        );
    }

    // [2,2,3] * [2,1,3] -> kBroadcastNdByNd optimized path.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_broadcast_nd() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(
            vec![2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let b = tf.make_default(vec![2, 1, 3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let out = tf.zeros_default(vec![2, 2, 3]);

        let mut ctx = context();
        opt_mul_out(&mut ctx, &a, &b, &out);
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                2.0, 6.0, 12.0, 8.0, 15.0, 24.0, 35.0, 48.0, 63.0, 50.0, 66.0, 84.0,
            ],
        );
        assert_tensor_close!(out, expected);
    }

    // OpMulOutTest.BroadcastAB2CTest: both inputs broadcast -> portable
    // fallback (kNone).
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_broadcast_ab2c() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 1], vec![1, 2]);
        let b = tf.make_default(vec![2, 1, 2], vec![1, 2, 3, 4]);
        let out = tf.zeros_default(vec![2, 2, 2]);

        let mut ctx = context();
        opt_mul_out(&mut ctx, &a, &b, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2, 2], vec![1, 2, 2, 4, 3, 4, 6, 8])
        );
    }

    // Half tensors skip every optimized path (kNone) and run the portable
    // elementwise fallback with a float compute type.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_half_portable_fallback() {
        let tf = TensorFactory::<Half>::new();
        let d = |v: f32| Half::from_f32(v);
        let a = tf.make_default(vec![2, 2], vec![d(1.5), d(2.5), d(3.5), d(4.5)]);
        let b = tf.make_default(vec![2, 2], vec![d(2.0), d(2.0), d(2.0), d(2.0)]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_mul_out(&mut ctx, &a, &b, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2], vec![d(3.0), d(5.0), d(7.0), d(9.0)])
        );
    }

    // Complex same-shape multiply: (1+2i)(3+4i) = -5+10i, (2-i)(5i) = 5+10i.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_complex_float() {
        let tf = TensorFactory::<Complex<f32>>::new();
        let c = |re: f32, im: f32| Complex { real: re, imag: im };
        let a = tf.make_default(vec![2], vec![c(1.0, 2.0), c(2.0, -1.0)]);
        let b = tf.make_default(vec![2], vec![c(3.0, 4.0), c(0.0, 5.0)]);
        let out = tf.zeros_default(vec![2]);

        let mut ctx = context();
        opt_mul_out(&mut ctx, &a, &b, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2], vec![c(-5.0, 10.0), c(5.0, 10.0)])
        );
    }

    // OpMulOutTest.MismatchedNonBroadcastableInputShapesDies.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn/test]
    #[test]
    fn opt_mul_out_mismatched_input_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![4, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![8]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_mul_out(&mut ctx, &a, &b, &out));
    }

    // OpMulScalarOutTest.OptimizedSanityCheck (float fast path).
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-scalar-out-fn/test]
    #[test]
    fn opt_mul_scalar_out_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.3, 2.1, 2.1, 4.2]);
        let out = tf.zeros_default(vec![2, 2]);
        let b = Scalar::from_double(2.0);

        let mut ctx = context();
        opt_mul_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_close!(out, tf.make_default(vec![2, 2], vec![2.6, 4.2, 4.2, 8.4]));
    }

    // Int tensor * Double scalar promotes to Float -> the non-fast branch with
    // a float compute type.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-scalar-out-fn/test]
    #[test]
    fn opt_mul_scalar_out_type_promotion() {
        let tf = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let out = tf_out.zeros_default(vec![2, 2]);
        let b = Scalar::from_double(0.5);

        let mut ctx = context();
        opt_mul_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_close!(
            out,
            tf_out.make_default(vec![2, 2], vec![0.5, 1.0, 2.0, 4.0])
        );
    }

    // Int fast path with an Int scalar.
    // [spec:et:sem:op-mul.torch.executor.native.opt-mul-scalar-out-fn/test]
    #[test]
    fn opt_mul_scalar_out_int_fast_path() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let out = tf.zeros_default(vec![2, 2]);
        let b = Scalar::from_i64(3);

        let mut ctx = context();
        opt_mul_scalar_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![3, 6, 12, 24]));
    }
}
