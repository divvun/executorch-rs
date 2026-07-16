//! Literal port of kernels/optimized/cpu/op_sub.cpp.
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
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the sub fast paths dispatch over ET_SWITCH_REAL (integers +
// Float/Double, no Bool), so plain primitive `-`/`*` suffice. `RealArith` keeps
// the closures generic across the switch-bound ctype.
trait RealArith: Copy {
    fn rsub(self, other: Self) -> Self;
    fn rmul(self, other: Self) -> Self;
}
macro_rules! impl_real_arith_prim {
    ($($t:ty),*) => {$(
        impl RealArith for $t {
            fn rsub(self, other: Self) -> Self { self - other }
            fn rmul(self, other: Self) -> Self { self * other }
        }
    )*};
}
impl_real_arith_prim!(u8, i8, i16, i32, i64, f32, f64);

// [spec:et:def:op-sub.torch.executor.native.opt-sub-out-fn]
// [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn]
pub fn opt_sub_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = b.scalar_type();
    let alpha_type = get_scalar_dtype(*alpha);
    let out_type = out.scalar_type();

    crate::et_kernel_check!(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    let common_type = promote_types(a_type, b_type, false);

    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out_type) && can_cast(alpha_type, common_type),
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

    let op_name = "sub.out";

    if a.numel() == 1 || b.numel() == 1 {
        if a_type == b_type
            && a_type == out_type
            && a_type != ScalarType::Half
            && a_type != ScalarType::BFloat16
        {
            let tensor: &Tensor;
            let scalar: &Tensor;
            let tensor_type: ScalarType;
            let scalar_type: ScalarType;
            if a.numel() == 1 {
                tensor = b;
                tensor_type = b_type;
                scalar = a;
                scalar_type = a_type;
            } else {
                tensor = a;
                tensor_type = a_type;
                scalar = b;
                scalar_type = b_type;
            }
            crate::et_switch_real_types!(tensor_type, ctx, op_name, CTYPE, {
                crate::et_switch_real_types!(scalar_type, ctx, op_name, CTYPE_SCALAR, {
                    let mut alpha_val: CTYPE = Default::default();
                    crate::et_kernel_check!(
                        ctx,
                        extract_scalar(*alpha, &mut alpha_val),
                        InvalidArgument,
                        out
                    );
                    let scalar_val: CTYPE_SCALAR =
                        unsafe { *scalar.const_data_ptr::<CTYPE_SCALAR>() };
                    let scalar_casted: CTYPE =
                        <CTYPE as StaticCast<CTYPE_SCALAR>>::static_cast(scalar_val);

                    // DEVIATION: at::vec::map -> scalar loop.
                    let out_data = out.mutable_data_ptr::<CTYPE>();
                    let tensor_data = tensor.const_data_ptr::<CTYPE>();
                    if a.numel() == 1 {
                        for i in 0..out.numel() {
                            unsafe {
                                *out_data.offset(i) =
                                    scalar_casted.rsub(alpha_val.rmul(*tensor_data.offset(i)));
                            }
                        }
                    } else {
                        let prod = alpha_val.rmul(scalar_casted);
                        for i in 0..out.numel() {
                            unsafe {
                                *out_data.offset(i) = (*tensor_data.offset(i)).rsub(prod);
                            }
                        }
                    }
                });
            });
            return out;
        }
    }

    opt_add_sub_out_impl(ctx, a, b, alpha, out, true, op_name)
}

// [spec:et:def:op-sub.torch.executor.native.opt-sub-scalar-out-fn]
// [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn]
pub fn opt_sub_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    alpha: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let common_type = promote_type_with_scalar(a_type, *b, false);
    let alpha_type = get_scalar_dtype(*alpha);
    let out_type = out.scalar_type();

    crate::et_kernel_check!(ctx, alpha_type != ScalarType::Bool, InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        common_type == out_type && can_cast(alpha_type, common_type),
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

    let op_name = "sub.Scalar_out";

    if a_type == common_type
        && a_type == out_type
        && a_type != ScalarType::Half
        && a_type != ScalarType::BFloat16
    {
        crate::et_switch_real_types!(a_type, ctx, op_name, CTYPE, {
            let b_casted: CTYPE = scalar_to::<CTYPE>(b);
            let mut alpha_val: CTYPE = Default::default();
            crate::et_kernel_check!(
                ctx,
                extract_scalar(*alpha, &mut alpha_val),
                InvalidArgument,
                out
            );

            // DEVIATION: at::vec::map -> scalar loop; x - (alpha_val * b_casted).
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let prod = alpha_val.rmul(b_casted);
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).rsub(prod);
                }
            }
        });
    } else {
        let mut common_type_mut = common_type;
        let compute_type = get_compute_type(&mut common_type_mut);

        crate::et_switch_real_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
            let val_alpha: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(alpha);
            let val_alpha_times_b = val_alpha.rmul(val_b);
            apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                move |vals: &[CTYPE_COMPUTE]| vals[0].rsub(val_alpha_times_b),
                ctx,
                a,
                SupportedTensorDtypes::REALHBF16,
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
    use crate::runtime::core::portable_type::Half;
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

    // OpSubOutTest.FloatTensors (same shape -> opt_add_sub_out_impl 1d path).
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_float_tensors() {
        let tf = TensorFactory::<f32>::new();
        let out = tf.zeros_default(vec![2, 2]);
        let alpha = Scalar::from_i64(1);

        let mut ctx = context();
        opt_sub_out(
            &mut ctx,
            &tf.make_default(vec![2, 2], vec![1.25, 2.25, 4.5, 8.875]),
            &tf.ones_default(vec![2, 2]),
            &alpha,
            &out,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2], vec![0.25, 1.25, 3.5, 7.875])
        );
    }

    // OpSubOutTest.BroadcastScalarSupported2: a.numel()==1 / b.numel()==1 fast
    // paths (out = a_scalar - alpha*b[i], and out = a[i] - alpha*b_scalar).
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_broadcast_scalar_operand() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![1, 1, 1], vec![8.0]);
        let b = tf.make_default(vec![3, 1, 1], vec![2.0, 4.0, 8.0]);
        let alpha = Scalar::from_i64(1);

        let out = tf.zeros_default(vec![3, 1, 1]);
        let mut ctx = context();
        opt_sub_out(&mut ctx, &a, &b, &alpha, &out);
        assert_tensor_eq!(out, tf.make_default(vec![3, 1, 1], vec![6.0, 4.0, 0.0]));

        let out2 = tf.zeros_default(vec![3, 1, 1]);
        opt_sub_out(&mut ctx, &b, &a, &alpha, &out2);
        assert_tensor_eq!(out2, tf.make_default(vec![3, 1, 1], vec![-6.0, -4.0, 0.0]));
    }

    // Alpha scaling in the b.numel()==1 fast path: out = a[i] - 1.5*2.0.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_scalar_operand_with_alpha() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = tf.make_default(vec![1], vec![2.0]);
        let alpha = Scalar::from_double(1.5);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_sub_out(&mut ctx, &a, &b, &alpha, &out);
        assert_tensor_close!(out, tf.make_default(vec![2, 2], vec![-2.0, -1.0, 0.0, 1.0]));
    }

    // OpSubOutTest test_broadcast_3D: [2,2,3] - [2,1,3] via the broadcast
    // optimized path.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_broadcast_3d() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(
            vec![2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let b = tf.make_default(vec![2, 1, 3], vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let alpha = Scalar::from_double(1.0);
        let out = tf.zeros_default(vec![2, 2, 3]);

        let mut ctx = context();
        opt_sub_out(&mut ctx, &a, &b, &alpha, &out);
        let expected = tf.make_default(
            vec![2, 2, 3],
            vec![
                -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0,
            ],
        );
        assert_tensor_close!(out, expected);
    }

    // OpSubOutTest.IntTensorFloatAlphaDies: floating alpha cannot cast to the
    // integral common type.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_int_tensor_float_alpha_dies() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = tf.ones_default(vec![2, 2]);
        let alpha = Scalar::from_double(0.7);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_sub_out(&mut ctx, &a, &b, &alpha, &out));
    }

    // OpSubOutTest.BoolInputTensorsFail: Int alpha cannot cast to Bool common
    // type.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn/test]
    #[test]
    fn opt_sub_out_bool_inputs_die() {
        let tf = TensorFactory::<bool>::new();
        let a = tf.make_default(vec![2, 2], vec![false, true, false, true]);
        let b = tf.make_default(vec![2, 2], vec![false, true, true, true]);
        let alpha = Scalar::from_i64(1);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_sub_out(&mut ctx, &a, &b, &alpha, &out));
    }

    // OpSubScalarOutTest.SanityCheck: Int tensor - Double scalar with Float out
    // (non-fast branch, float compute type).
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn/test]
    #[test]
    fn opt_sub_scalar_out_sanity_check() {
        let tf_a = TensorFactory::<i32>::new();
        let tf_out = TensorFactory::<f32>::new();
        let a = tf_a.make_default(vec![2, 2], vec![1, 2, 4, 8]);
        let b = Scalar::from_double(0.5);
        let alpha = Scalar::from_double(1.5);
        let out = tf_out.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_sub_scalar_out(&mut ctx, &a, &b, &alpha, &out);
        assert_tensor_eq!(
            out,
            tf_out.make_default(vec![2, 2], vec![0.25, 1.25, 3.25, 7.25])
        );
    }

    // OpSubScalarOutTest.OptimizedSanityCheck: float fast path,
    // out = a[i] - alpha*b.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn/test]
    #[test]
    fn opt_sub_scalar_out_optimized_sanity_check() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![6.3, 2.1, 5.6, 8.2]);
        let b = Scalar::from_double(1.9);
        let alpha = Scalar::from_double(2.8);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_sub_scalar_out(&mut ctx, &a, &b, &alpha, &out);
        assert_tensor_close!(
            out,
            tf.make_default(vec![2, 2], vec![0.98, -3.22, 0.28, 2.88])
        );
    }

    // OpSubScalarOutTest.DtypeTest_float16_float_int_float16: Half skips the
    // fast path and runs the portable unitensor fallback.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn/test]
    #[test]
    fn opt_sub_scalar_out_half() {
        let tf = TensorFactory::<Half>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = Scalar::from_double(-1.0);
        let alpha = Scalar::from_i64(1);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_sub_scalar_out(&mut ctx, &a, &b, &alpha, &out);
        let two = Half::from_f32(2.0);
        assert_tensor_close!(out, tf.make_default(vec![2, 2], vec![two, two, two, two]));
    }

    // Bool alpha is rejected up front.
    // [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn/test]
    #[test]
    fn opt_sub_scalar_out_bool_alpha_dies() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let b = Scalar::from_double(1.0);
        let alpha = Scalar::from_bool(true);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_sub_scalar_out(&mut ctx, &a, &b, &alpha, &out));
    }
}
