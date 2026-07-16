//! Literal port of kernels/optimized/cpu/op_le.cpp.
//!
//! DEVIATION: `at::vec::map` / `at::vec::map2` / `handle_broadcast_elementwise`
//! over `Vectorized<CTYPE>` collapse to scalar loops (PORTING.md optimized-
//! kernel substitution).

use crate::kernels::optimized::cpu::binary_ops::{
    ElementwiseOptimizedPath, handle_broadcast_elementwise, select_optimized_path,
};
use crate::kernels::portable::cpu::pattern::comparison_op::{
    ComparisonOp, comparison_scalar_out, comparison_tensor_out,
};
use crate::kernels::portable::cpu::scalar_utils::{get_scalar_dtype, promote_type_with_scalar};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::StaticCast;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `std::less_equal` functor passed to the portable comparison
// patterns becomes this zero-sized `ComparisonOp` (mirrors portable op_le.rs).
struct LessEqual;
impl ComparisonOp for LessEqual {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a <= b
    }
}

// PORT-NOTE: `at::vec::Vectorized<CTYPE>::le(other)` is a *normalized* boolean
// predicate, not a raw lane mask: the generic vec_base.h implementation is
// `binary_pred_bool(other, std::less_equal<T>())`, which stores
// `static_cast<T>(a <= b)` per lane, and the SIMD specializations match it by
// AND-ing the comparison mask with `Vectorized<T>(1)` (e.g. NEON:
// `(*this <= other) & Vectorized<float>(1.0f)`). So the optimized path writes
// CTYPE 1/0 into `out` — this trait reproduces exactly that.
trait VecLe: Copy {
    fn le(self, other: Self) -> Self;
}
macro_rules! impl_vec_le_num {
    ($($t:ty),*) => {$(
        impl VecLe for $t {
            fn le(self, other: Self) -> Self {
                if self <= other { 1 as $t } else { 0 as $t }
            }
        }
    )*};
}
impl_vec_le_num!(u8, i8, i16, i32, i64, f32, f64);
impl VecLe for bool {
    fn le(self, other: Self) -> Self {
        self <= other
    }
}

// [spec:et:def:op-le.torch.executor.native.opt-le-tensor-out-fn]
// [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn]
pub fn opt_le_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let out_type = out.scalar_type();

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

    let op_name = "le.Tensor_out";

    // Check for optimized broadcast paths
    let selected_optimized_path = select_optimized_path(a, b, out);
    if selected_optimized_path == ElementwiseOptimizedPath::KTreatAs1d {
        crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
            // DEVIATION: at::vec::map2 -> scalar loop; x.le(y).
            let out_data = out.mutable_data_ptr::<CTYPE>();
            let a_data = a.const_data_ptr::<CTYPE>();
            let b_data = b.const_data_ptr::<CTYPE>();
            for i in 0..out.numel() {
                unsafe {
                    *out_data.offset(i) = (*a_data.offset(i)).le(*b_data.offset(i));
                }
            }
        });
    } else if selected_optimized_path != ElementwiseOptimizedPath::KNone {
        // Handle optimized broadcast cases
        crate::et_switch_realb_types!(out_type, ctx, op_name, CTYPE, {
            let le_lambda = |x: CTYPE, y: CTYPE| x.le(y);
            handle_broadcast_elementwise::<CTYPE, _>(
                ctx,
                &le_lambda,
                a,
                b,
                out,
                selected_optimized_path,
                None,
            );
        });
    } else {
        comparison_tensor_out::<LessEqual>(ctx, a, b, out);
    }

    out
}

// [spec:et:def:op-le.torch.executor.native.opt-le-scalar-out-fn]
// [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn]
pub fn opt_le_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let a_type = a.scalar_type();
    let b_type = get_scalar_dtype(*b);
    let common_type = promote_type_with_scalar(a_type, *b, false);
    let out_type = out.scalar_type();

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

    let op_name = "le.Scalar_out";

    if a_type == common_type
        && a_type == out_type
        && a_type != ScalarType::Half
        && a_type != ScalarType::BFloat16
    {
        crate::et_switch_realb_types!(a_type, ctx, op_name, CTYPE, {
            crate::et_switch_realb_types!(b_type, ctx, op_name, CTYPE_B, {
                let mut b_val: CTYPE_B = Default::default();
                crate::et_extract_scalar!(*b, b_val);
                let b_casted: CTYPE = <CTYPE as StaticCast<CTYPE_B>>::static_cast(b_val);
                // DEVIATION: at::vec::map -> scalar loop; x.le(b_casted).
                let out_data = out.mutable_data_ptr::<CTYPE>();
                let a_data = a.const_data_ptr::<CTYPE>();
                for i in 0..a.numel() {
                    unsafe {
                        *out_data.offset(i) = (*a_data.offset(i)).le(b_casted);
                    }
                }
            });
        });
    } else {
        comparison_scalar_out::<LessEqual>(ctx, a, b, out);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

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

    // Same shape + same dtype (Int) -> kTreatAs1d fast path. Vectorized::le is
    // a normalized predicate, so the Int output holds 1/0, not lane masks.
    // [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn/test]
    #[test]
    fn opt_le_tensor_out_treat_as_1d_int() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 4]);
        let b = tf.make_default(vec![2, 2], vec![1, 4, 2, 3]);
        let out = tf.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_le_tensor_out(&mut ctx, &a, &b, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![0, 1, 1, 0]));
    }

    // Float kTreatAs1d fast path: NEON `le` is `(a <= b) & Vectorized(1.0f)`,
    // i.e. the float output is exactly 1.0/0.0.
    // [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn/test]
    #[test]
    fn opt_le_tensor_out_treat_as_1d_float() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![1, 4], vec![2.0, 3.0, 2.0, 4.0]);
        let b = tf.make_default(vec![1, 4], vec![1.0, 4.0, 2.0, 3.0]);
        let out = tf.zeros_default(vec![1, 4]);

        let mut ctx = context();
        opt_le_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(out, tf.make_default(vec![1, 4], vec![0.0, 1.0, 1.0, 0.0]));
    }

    // Mixed dtypes (Int inputs, Bool out) never select an optimized path and
    // fall through to the portable comparison_tensor_out.
    // [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn/test]
    #[test]
    fn opt_le_tensor_out_bool_out_portable_fallback() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 4]);
        let b = tf.make_default(vec![2, 2], vec![1, 4, 2, 3]);
        let out = tf_bool.zeros_default(vec![2, 2]);

        let mut ctx = context();
        opt_le_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(vec![2, 2], vec![false, true, true, false])
        );
    }

    // [3,4] vs [3,1] with matching dtypes -> kBroadcastLastDim optimized path
    // (handle_broadcast_elementwise with the le lambda).
    // [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn/test]
    #[test]
    fn opt_le_tensor_out_broadcast_last_dim() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let b = tf.make_default(vec![3, 1], vec![2, 5, 50]);
        let out = tf.zeros_default(vec![3, 4]);

        let mut ctx = context();
        opt_le_tensor_out(&mut ctx, &a, &b, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(vec![3, 4], vec![1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
        );
    }

    // [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn/test]
    #[test]
    fn opt_le_tensor_out_mismatched_shapes_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf.ones_default(vec![4]);
        let b = tf.ones_default(vec![2, 2]);
        let out = tf_bool.ones_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_le_tensor_out(&mut ctx, &a, &b, &out));
    }

    // Int tensor vs Int scalar with Int out -> vectorized fast path; output is
    // 1/0 in the input dtype.
    // [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn/test]
    #[test]
    fn opt_le_scalar_out_int_fast_path() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 2], vec![3, 1, 2, 4]);
        let out = tf.ones_default(vec![2, 2]);
        let other = Scalar::from_i64(2);

        let mut ctx = context();
        opt_le_scalar_out(&mut ctx, &a, &other, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![0, 1, 1, 0]));
    }

    // Float fast path with an integer Scalar (b_type Long, casted to float).
    // [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn/test]
    #[test]
    fn opt_le_scalar_out_float_fast_path() {
        let tf = TensorFactory::<f32>::new();
        let a = tf.make_default(vec![2, 2], vec![3.0, 1.0, 2.0, 4.0]);
        let out = tf.zeros_default(vec![2, 2]);
        let other = Scalar::from_i64(2);

        let mut ctx = context();
        opt_le_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]));
    }

    // Bool input with a Double scalar: common type promotes to Float != a_type,
    // so this takes the portable comparison_scalar_out branch.
    // [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn/test]
    #[test]
    fn opt_le_scalar_out_bool_input() {
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf_bool.make_default(vec![2, 2], vec![false, true, false, true]);
        let out = tf_bool.zeros_default(vec![2, 2]);
        let other = Scalar::from_double(0.5);

        let mut ctx = context();
        opt_le_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(vec![2, 2], vec![true, false, true, false])
        );
    }

    // [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn/test]
    #[test]
    fn opt_le_scalar_out_mismatched_out_shape_dies() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let a = tf.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);
        let other = Scalar::from_i64(3);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, opt_le_scalar_out(&mut ctx, &a, &other, &out));
    }
}
