//! Literal port of kernels/optimized/cpu/op_where.cpp.

use crate::kernels::portable::cpu::util::broadcast_indexes_range::BroadcastIndexesRange;
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size_3;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_tritensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::promote_types;
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order4;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
use crate::runtime::kernel::thread_parallel_interface::internal::GRAIN_SIZE;
use crate::runtime::kernel::thread_parallel_interface::parallel_for;

// [spec:et:def:op-where.torch.executor.native.opt-where-out-fn]
// [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn]
pub fn opt_where_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    cond: &'a Tensor<'b>,
    a: &'a Tensor<'b>,
    b: &'a Tensor<'b>,
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

    if a.scalar_type() == b.scalar_type()
        && a.scalar_type() == out.scalar_type()
        && a.scalar_type() == compute_type
        // Using a Byte tensor for cond has been deprecated for a long time.
        && cond.scalar_type() == ScalarType::Bool
    {
        let out_numel = out.numel();
        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            let data_a = a.const_data_ptr::<CTYPE_COMPUTE>();
            let data_b = b.const_data_ptr::<CTYPE_COMPUTE>();
            let data_cond = cond.const_data_ptr::<bool>();
            let data_out = out.mutable_data_ptr::<CTYPE_COMPUTE>();
            parallel_for(0, out_numel as i64, GRAIN_SIZE, &|begin: i64, end: i64| {
                // NOTE: the Rust `BroadcastIndexesRange<NT>` const param is the
                // value_type array length (num_inputs + 1), so the C++
                // `BroadcastIndexesRange<3>` (3 inputs) maps to `<4>` here.
                let range = BroadcastIndexesRange::<4>::new(out, &[a, b, cond]);
                let mut begin_it = range.begin();
                begin_it.add_assign(begin as isize);
                while begin_it.output_index() < end as isize {
                    let idxs = *begin_it.deref();
                    let out_index = idxs[0];
                    let a_index = idxs[1];
                    let b_index = idxs[2];
                    let cond_index = idxs[3];
                    unsafe {
                        *data_out.offset(out_index) = if *data_cond.offset(cond_index) {
                            *data_a.offset(a_index)
                        } else {
                            *data_b.offset(b_index)
                        };
                    }
                    begin_it.increment();
                }
            });
        });
    } else {
        // Fall back for mixed dtype to keep code size and compile time
        // reasonable.
        crate::et_switch_realb_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
            apply_tritensor_elementwise_fn::<CTYPE_COMPUTE, _>(
                |vals: &[CTYPE_COMPUTE]| {
                    // (val_a, val_b, val_c) -> val_c ? val_a : val_b
                    if vals[2] != <CTYPE_COMPUTE as WhereZero>::zero() {
                        vals[0]
                    } else {
                        vals[1]
                    }
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
                false,
            );
        });
    }

    out
}

// PORT-NOTE: the C++ fallback closure evaluates `val_c ? val_a : val_b`, where
// `val_c` is a `CTYPE_COMPUTE` (the cond value cast into the compute type). Rust
// has no implicit truthiness, so `WhereZero` supplies the per-type zero to test
// `val_c != 0`, matching the C++ conversion-to-bool of a numeric compute value.
trait WhereZero: Copy {
    fn zero() -> Self;
}
macro_rules! impl_where_zero_prim {
    ($($t:ty),*) => {$(
        impl WhereZero for $t {
            fn zero() -> Self { 0 as $t }
        }
    )*};
}
impl_where_zero_prim!(u8, i8, i16, i32, i64, f32, f64);
impl WhereZero for bool {
    fn zero() -> Self {
        false
    }
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
        condition: &'a Tensor<'b>,
        self_: &'a Tensor<'b>,
        other: &'a Tensor<'b>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        opt_where_out(ctx, condition, self_, other, out)
    }

    fn assert_tensor_close(a: &Tensor, b: &Tensor) {
        assert!(tensors_are_close(a, b, internal::K_DEFAULT_RTOL, None));
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

    // With a bool condition and uniform dtypes this hits the optimized fast
    // path; the byte-condition second call exercises the mixed-dtype fallback.
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

    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
    #[test]
    fn op_where_out_test_all_real_dtypes_supported() {
        test_where_enumerate_a_types_aten();
    }

    // Condition is true, all items will be from x
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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

    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
    #[test]
    fn op_where_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(!output_resize, ...)` skips
    // DynamicShapeUpperBoundLargerThanExpected and DynamicShapeUnbound for
    // non-aten kernels (SupportedFeatures::output_resize default false), so
    // they are not ported here.

    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
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

    // Not in the C++ suite: the optimized fast path iterates with
    // BroadcastIndexesRange under parallel_for, so pin a case where every
    // input (cond, a, b) broadcasts along a different axis and hits the
    // uniform-dtype bool-cond branch, then cross-check the byte-cond
    // fallback on equivalent data.
    // [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn/test]
    #[test]
    fn op_where_out_test_fast_path_broadcast_all_inputs() {
        let tb = TensorFactory::<bool>::new();
        let tu8 = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();

        let cond = tb.make_default(vec![2, 1], vec![true, false]);
        let a = tf.make_default(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let b = tf.make_default(vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let out = tf.zeros_default(vec![2, 3]);

        let expected = tf.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 40.0, 50.0, 60.0]);

        let mut ctx = context();
        op_where_self_out(&mut ctx, &cond, &a, &b, &out);
        assert_tensor_close(&out, &expected);

        // Same computation through the fallback branch (byte condition).
        let cond_byte = tu8.make_default(vec![2, 1], vec![1, 0]);
        let out2 = tf.zeros_default(vec![2, 3]);
        op_where_self_out(&mut ctx, &cond_byte, &a, &b, &out2);
        assert_tensor_close(&out2, &expected);
    }
}
