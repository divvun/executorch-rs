//! Literal port of kernels/portable/cpu/op_where.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size_3;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_tritensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::promote_types;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order4;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the compute lambda is `val_c ? val_a : val_b`, relying on C++'s
// implicit conversion of `CTYPE_COMPUTE val_c` to bool (nonzero -> true) over the
// REALB compute set (u8, i8, i16, i32, i64, f32, f64, bool). Rust has no implicit
// truthiness, so this module-local trait reproduces the "nonzero" predicate for
// each compute type; for `bool` the value is used directly.
trait Truthy: Copy {
    fn is_truthy(self) -> bool;
}
macro_rules! impl_truthy_num {
    ($($t:ty),*) => {$(
        impl Truthy for $t {
            fn is_truthy(self) -> bool {
                self != 0 as $t
            }
        }
    )*};
}
impl_truthy_num!(u8, i8, i16, i32, i64, f32, f64);
impl Truthy for bool {
    fn is_truthy(self) -> bool {
        self
    }
}

// [spec:et:def:op-where.torch.executor.native.where-out-fn]
// [spec:et:sem:op-where.torch.executor.native.where-out-fn]
#[executorch_macros::et_kernel("aten::where.self_out")]
pub fn where_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    cond: &Tensor,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(ctx, common_type == out.scalar_type(), InvalidArgument, out);

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order4(cond, a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size_3(a, b, cond, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let mut common_type_mut = common_type;
    let compute_type = get_compute_type(&mut common_type_mut);

    let op_name = "where.self_out";

    crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_tritensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                let val_a = vals[0];
                let val_b = vals[1];
                let val_c = vals[2];
                if val_c.is_truthy() { val_a } else { val_b }
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            cond,
            SupportedTensorDtypes::BOOL_OR_BYTE,
            out,
            SupportedTensorDtypes::SAME_AS_COMMON,
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
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

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

    fn op_where_self_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        condition: &Tensor,
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        where_out(ctx, condition, self_, other, out)
    }

    trait FromI64 {
        fn from_i64(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_num {
        ($($t:ty),*) => {$(impl FromI64 for $t { fn from_i64(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI64 for bool {
        fn from_i64(v: i64) -> Self {
            v != 0
        }
    }
    impl FromI64 for Half {
        fn from_i64(v: i64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI64 for BFloat16 {
        fn from_i64(v: i64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn make_i64<T: FromI64>(vals: &[i64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_i64(v)).collect()
    }

    fn test_where<A, B, OUT>()
    where
        A: CppTypeToScalarType + FactoryValue + FromI64,
        B: CppTypeToScalarType + FactoryValue + FromI64,
        OUT: CppTypeToScalarType + FactoryValue + FromI64,
    {
        if OUT::VALUE == ScalarType::Byte || OUT::VALUE == ScalarType::Char {
            return;
        }
        let tf_condition = TensorFactory::<bool>::new();
        let tf_condition_byte = TensorFactory::<u8>::new();
        let tf_a = TensorFactory::<A>::new();
        let tf_b = TensorFactory::<B>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let condition_sizes = vec![12];
        let sizes = vec![1, 12];

        let out = tf_out.zeros_default(sizes.clone());

        let condition_data_u8: [u8; 12] = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0];
        let condition_data_bool: Vec<bool> = condition_data_u8.iter().map(|&v| v != 0).collect();

        let a_tensor = tf_a.make_default(
            sizes.clone(),
            make_i64(&[1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]),
        );
        let b_tensor = tf_b.make_default(
            sizes.clone(),
            make_i64(&[6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6]),
        );

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes.clone(), condition_data_bool),
            &a_tensor,
            &b_tensor,
            &out,
        );

        let expected_out = tf_out.make_default(
            sizes.clone(),
            make_i64(&[6, 2, 4, 4, 5, 1, 1, 5, 3, 3, 2, 6]),
        );
        assert_tensor_close(&out, &expected_out);

        op_where_self_out(
            &mut ctx,
            &tf_condition_byte.make_default(condition_sizes, condition_data_u8.to_vec()),
            &a_tensor,
            &b_tensor,
            &out,
        );
        assert_tensor_close(&out, &expected_out);
    }

    fn assert_tensor_close(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, internal::K_DEFAULT_RTOL, None));
    }

    // PORT-NOTE: defined in the C++ fixture but invoked by no TEST_F.
    #[allow(dead_code)]
    fn test_where_enumerate_out_types<A, B>()
    where
        A: CppTypeToScalarType + FactoryValue + FromI64,
        B: CppTypeToScalarType + FactoryValue + FromI64,
    {
        // ET_FORALL_REALHBF16_TYPES
        test_where::<A, B, u8>();
        test_where::<A, B, i8>();
        test_where::<A, B, i16>();
        test_where::<A, B, i32>();
        test_where::<A, B, i64>();
        test_where::<A, B, f32>();
        test_where::<A, B, f64>();
        test_where::<A, B, Half>();
        test_where::<A, B, BFloat16>();
    }

    // PORT-NOTE: defined in the C++ fixture but invoked by no TEST_F.
    #[allow(dead_code)]
    fn test_where_enumerate_b_types<A>()
    where
        A: CppTypeToScalarType + FactoryValue + FromI64,
    {
        // ET_FORALL_REALHBBF16_TYPES
        test_where::<A, u8, A>();
        test_where::<A, i8, A>();
        test_where::<A, i16, A>();
        test_where::<A, i32, A>();
        test_where::<A, i64, A>();
        test_where::<A, f32, A>();
        test_where::<A, f64, A>();
        test_where::<A, bool, A>();
        test_where::<A, Half, A>();
        test_where::<A, BFloat16, A>();
    }

    // PORT-NOTE: defined in the C++ fixture but invoked by no TEST_F.
    #[allow(dead_code)]
    fn test_where_enumerate_a_types() {
        // ET_FORALL_REALHBBF16_TYPES
        test_where_enumerate_b_types::<u8>();
        test_where_enumerate_b_types::<i8>();
        test_where_enumerate_b_types::<i16>();
        test_where_enumerate_b_types::<i32>();
        test_where_enumerate_b_types::<i64>();
        test_where_enumerate_b_types::<f32>();
        test_where_enumerate_b_types::<f64>();
        test_where_enumerate_b_types::<bool>();
        test_where_enumerate_b_types::<Half>();
        test_where_enumerate_b_types::<BFloat16>();
    }

    fn test_where_enumerate_a_types_aten() {
        // ET_FORALL_REALHBF16_TYPES: test_where<dtype, dtype, dtype>()
        test_where::<u8, u8, u8>();
        test_where::<i8, i8, i8>();
        test_where::<i16, i16, i16>();
        test_where::<i32, i32, i32>();
        test_where::<i64, i64, i64>();
        test_where::<f32, f32, f32>();
        test_where::<f64, f64, f64>();
        test_where::<Half, Half, Half>();
        test_where::<BFloat16, BFloat16, BFloat16>();
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf_bool = TensorFactory::<bool>::new();
        let tf = TensorFactory::<f32>::new();

        let condition = tf_bool.make_default(
            vec![2, 3, 4],
            vec![
                true, false, true, true, true, false, false, true, false, true, true, false, false,
                false, false, false, false, false, true, true, false, false, true, true,
            ],
        );
        let input = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.41940832138061523,
                0.5529070496559143,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.37341737747192383,
                0.3051000237464905,
                0.9320003986358643,
                0.17591017484664917,
                0.2698335647583008,
                0.15067976713180542,
                0.03171950578689575,
                0.20812976360321045,
                0.9297990202903748,
                0.7231091856956482,
                0.7423362731933594,
                0.5262957811355591,
                0.24365824460983276,
                0.584592342376709,
                0.033152639865875244,
                0.13871687650680542,
                0.242235004901886,
                0.815468966960907,
                0.793160617351532,
            ],
        );
        let other = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.2782524824142456,
                0.48195880651474,
                0.8197803497314453,
                0.9970665574073792,
                0.6984410881996155,
                0.5675464272499084,
                0.8352431654930115,
                0.2055988311767578,
                0.593172013759613,
                0.11234724521636963,
                0.1534569263458252,
                0.24170821905136108,
                0.7262365221977234,
                0.7010802030563354,
                0.2038237452507019,
                0.6510535478591919,
                0.7744860053062439,
                0.4368913173675537,
                0.5190907716751099,
                0.6158523559570312,
                0.8101882934570312,
                0.9800970554351807,
                0.1146882176399231,
                0.3167651295661926,
            ],
        );
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.41940832138061523,
                0.48195880651474,
                0.9527381062507629,
                0.036164820194244385,
                0.1852310299873352,
                0.5675464272499084,
                0.8352431654930115,
                0.9320003986358643,
                0.593172013759613,
                0.2698335647583008,
                0.15067976713180542,
                0.24170821905136108,
                0.7262365221977234,
                0.7010802030563354,
                0.2038237452507019,
                0.6510535478591919,
                0.7744860053062439,
                0.4368913173675537,
                0.584592342376709,
                0.033152639865875244,
                0.8101882934570312,
                0.9800970554351807,
                0.815468966960907,
                0.793160617351532,
            ],
        );
        let out = tf.zeros(out_shape, dynamism);

        let mut ctx = context();
        op_where_self_out(&mut ctx, &condition, &input, &other, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    //
    // Correctness Test
    //

    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_all_real_dtypes_supported() {
        test_where_enumerate_a_types_aten();
    }

    // Condition is true, all items will be from x
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_all_true_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let condition_sizes = vec![1];
        let sizes = vec![1, 12];

        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes, vec![true]),
            &tf_x.make_default(
                sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(
                sizes.clone(),
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                sizes,
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
        );
    }

    // Condition is false, all items will be from y
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_all_false_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let condition_sizes = vec![1];
        let sizes = vec![1, 12];

        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes, vec![false]),
            &tf_x.make_default(
                sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(
                sizes.clone(),
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                sizes,
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
        );
    }

    // Choosing based on condition[i] ? x[i] : y[i]
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_mixed_true_false_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let condition_sizes = vec![12];
        let sizes = vec![1, 12];

        let out = tf_out.zeros_default(sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(
                condition_sizes,
                vec![
                    false, true, false, true, true, false, false, true, false, true, true, false,
                ],
            ),
            &tf_x.make_default(
                sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(
                sizes.clone(),
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                sizes,
                vec![
                    0.1, 1.0, 2.1, 3.0, 4.0, 5.1, 6.1, 7.0, 8.1, 9.0, 10.0, 100.1,
                ],
            ),
        );
    }

    // Choosing based on condition[i] ? x[i] : y[i]
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_broadcast_condition_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let condition_sizes = vec![3, 1];
        let x_sizes = vec![3, 4];
        let y_sizes = vec![3, 4];

        let out = tf_out.zeros_default(x_sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes, vec![false, true, false]),
            &tf_x.make_default(
                x_sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(
                y_sizes,
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                x_sizes,
                vec![
                    0.1, 1.1, 2.1, 3.1, 4.0, 5.0, 6.0, 7.0, 8.1, 9.1, 10.1, 100.1,
                ],
            ),
        );
    }

    // Choosing based on condition[i] ? x[i] : y[i]
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_broadcast_condition_and_broad_cast_y_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f32>::new();
        let tf_out = TensorFactory::<f32>::new();

        let condition_sizes = vec![3, 1];
        let x_sizes = vec![3, 4];
        let y_sizes = vec![3, 1];

        let out = tf_out.zeros_default(x_sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes, vec![false, true, false]),
            &tf_x.make_default(
                x_sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(y_sizes, vec![0.1, 4.1, 8.1]),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                x_sizes,
                vec![0.1, 0.1, 0.1, 0.1, 4.0, 5.0, 6.0, 7.0, 8.1, 8.1, 8.1, 8.1],
            ),
        );
    }

    // Choosing based on condition[i] ? x[i] : y[i]
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_double_type_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f64>::new();
        let tf_y = TensorFactory::<f64>::new();
        let tf_out = TensorFactory::<f64>::new();

        let condition_sizes = vec![3, 1];
        let x_sizes = vec![3, 4];
        let y_sizes = vec![3, 1];

        let out = tf_out.zeros_default(x_sizes.clone());

        let mut ctx = context();
        op_where_self_out(
            &mut ctx,
            &tf_condition.make_default(condition_sizes, vec![false, true, false]),
            &tf_x.make_default(
                x_sizes.clone(),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0,
                ],
            ),
            &tf_y.make_default(y_sizes, vec![0.1, 4.1, 8.1]),
            &out,
        );

        assert_tensor_close(
            &out,
            &tf_out.make_default(
                x_sizes,
                vec![0.1, 0.1, 0.1, 0.1, 4.0, 5.0, 6.0, 7.0, 8.1, 8.1, 8.1, 8.1],
            ),
        );
    }

    // Choosing based on condition[i] ? x[i] : y[i]
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_mismatched_shape_test() {
        let tf_condition = TensorFactory::<bool>::new();
        let tf_x = TensorFactory::<f32>::new();
        let tf_y = TensorFactory::<f64>::new();
        let tf_out = TensorFactory::<f64>::new();

        let condition_sizes = vec![3, 1];
        let x_sizes = vec![3, 4];
        let y_sizes = vec![4, 1];

        let out = tf_out.zeros_default(x_sizes.clone());

        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            op_where_self_out(
                &mut ctx,
                &tf_condition.make_default(condition_sizes, vec![false, true, false]),
                &tf_x.make_default(
                    x_sizes,
                    vec![
                        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0
                    ],
                ),
                &tf_y.make_default(y_sizes, vec![0.1, 4.1, 8.1, 11.1]),
                &out,
            )
        );
    }

    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (SupportedFeatures::output_resize default false).
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips this test for the
    // portable kernel (output_resize default false).
    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // [spec:et:sem:op-where.torch.executor.native.where-out-fn/test]
    #[test]
    fn op_where_out_test_half_support() {
        let tb = TensorFactory::<bool>::new();
        let tf = TensorFactory::<Half>::new();
        let cond = tb.make_default(vec![2, 3], vec![true, false, true, false, true, false]);
        let a = tf.full(vec![2, 3], Half::from_f32(1.5), TensorShapeDynamism::STATIC);
        let b = tf.full(vec![2, 3], Half::from_f32(2.5), TensorShapeDynamism::STATIC);
        let out = tf.zeros_default(vec![2, 3]);

        let mut ctx = context();
        op_where_self_out(&mut ctx, &cond, &a, &b, &out);
        let expected: Vec<Half> = [1.5f32, 2.5, 1.5, 2.5, 1.5, 2.5]
            .iter()
            .map(|&v| Half::from_f32(v))
            .collect();
        assert_tensor_close(&out, &tf.make_default(vec![2, 3], expected));
    }
}
