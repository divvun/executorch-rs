//! Literal port of kernels/portable/cpu/op_le.cpp.

use crate::kernels::portable::cpu::pattern::comparison_op::{
    ComparisonOp, comparison_scalar_out, comparison_tensor_out,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `std::less_equal` functor template passed to the comparison
// patterns becomes a zero-sized `ComparisonOp` implementor whose `apply` yields
// `a <= b`, mirroring `Comparison<CTYPE_COMPUTE>()(val_a, val_b)`.
struct LessEqual;

impl ComparisonOp for LessEqual {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool {
        a <= b
    }
}

// [spec:et:def:op-le.torch.executor.native.le-tensor-out-fn]
// [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn]
#[executorch_macros::et_kernel("aten::le.Tensor_out")]
pub fn le_tensor_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    // op_name = "le.Tensor_out"
    comparison_tensor_out::<LessEqual>(ctx, a, b, out)
}

// [spec:et:def:op-le.torch.executor.native.le-scalar-out-fn]
// [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn]
pub fn le_scalar_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    // op_name = "le.Scalar_out"
    comparison_scalar_out::<LessEqual>(ctx, a, b, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
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

    // Element builder: `{true,false,...}` / integer initializer lists coerce to
    // 0/1 for the numeric factories and to bool for the Bool factory.
    trait FromCmp: Copy {
        fn from_i32(v: i32) -> Self;
        fn from_bool(v: bool) -> Self;
    }
    macro_rules! impl_from_cmp_num {
        ($($t:ty),*) => {$(impl FromCmp for $t {
            fn from_i32(v: i32) -> Self { v as $t }
            fn from_bool(v: bool) -> Self { v as i32 as $t }
        })*};
    }
    impl_from_cmp_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromCmp for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            Half::from_f32(v as i32 as f32)
        }
    }
    impl FromCmp for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
        fn from_bool(v: bool) -> Self {
            BFloat16::from_f32(v as i32 as f32)
        }
    }
    impl FromCmp for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
        fn from_bool(v: bool) -> Self {
            v
        }
    }

    fn b<T: FromCmp>(v: &[bool]) -> Vec<T> {
        v.iter().map(|&x| T::from_bool(x)).collect()
    }
    fn i<T: FromCmp>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_le_scalar_out<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
        OUT: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        let tf = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();

        let sizes = vec![2, 2];
        let out = tf_out.ones_default(sizes.clone());
        let other = Scalar::from_i64(2);

        // Valid input should give the expected output
        let a = tf.make_default(sizes.clone(), i::<IN>(&[3, 1, 2, 4]));
        let mut ctx = context();
        le_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_out.make_default(sizes, b::<OUT>(&[false, true, true, false]))
        );
    }

    fn test_dtype<IN, OUT>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
        OUT: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        let tf_input = TensorFactory::<IN>::new();
        let tf_out = TensorFactory::<OUT>::new();
        let a = tf_input.make_default(vec![2, 2], i::<IN>(&[2, 3, 2, 4]));
        let bb = tf_input.make_default(vec![2, 2], i::<IN>(&[1, 4, 2, 3]));
        let out = tf_out.zeros_default(vec![2, 2]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);
        assert_tensor_eq!(
            out,
            tf_out.make_default(vec![2, 2], b::<OUT>(&[false, true, true, false]))
        );
    }

    // ET_FORALL_REALHBF16_TYPES x {that dtype, Bool}.
    fn forall_realhbf16_out<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        test_le_scalar_out::<IN, u8>();
        test_le_scalar_out::<IN, i8>();
        test_le_scalar_out::<IN, i16>();
        test_le_scalar_out::<IN, i32>();
        test_le_scalar_out::<IN, i64>();
        test_le_scalar_out::<IN, Half>();
        test_le_scalar_out::<IN, BFloat16>();
        test_le_scalar_out::<IN, f32>();
        test_le_scalar_out::<IN, f64>();
        test_le_scalar_out::<IN, bool>();
    }

    fn forall_realhbf16_out_tensor<IN>()
    where
        IN: CppTypeToScalarType + FactoryValue + FromCmp,
    {
        test_dtype::<IN, u8>();
        test_dtype::<IN, i8>();
        test_dtype::<IN, i16>();
        test_dtype::<IN, i32>();
        test_dtype::<IN, i64>();
        test_dtype::<IN, Half>();
        test_dtype::<IN, BFloat16>();
        test_dtype::<IN, f32>();
        test_dtype::<IN, f64>();
        test_dtype::<IN, bool>();
    }

    // [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn/test]
    // [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn/test]
    #[test]
    fn op_le_scalar_out_test_all_real_input_bool_output_support() {
        forall_realhbf16_out::<u8>();
        forall_realhbf16_out::<i8>();
        forall_realhbf16_out::<i16>();
        forall_realhbf16_out::<i32>();
        forall_realhbf16_out::<i64>();
        forall_realhbf16_out::<Half>();
        forall_realhbf16_out::<BFloat16>();
        forall_realhbf16_out::<f32>();
        forall_realhbf16_out::<f64>();
    }

    // [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn/test]
    #[test]
    fn op_le_scalar_out_test_bool_input_dtype() {
        let tf_bool = TensorFactory::<bool>::new();

        let sizes = vec![2, 2];
        let a = tf_bool.make_default(sizes.clone(), vec![false, true, false, true]);
        let out = tf_bool.zeros_default(sizes.clone());
        let other = Scalar::from_double(0.5);

        let mut ctx = context();
        le_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(
            out,
            tf_bool.make_default(sizes, vec![true, false, true, false])
        );
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn/test]
    #[test]
    fn op_le_scalar_out_test_mismatched_in_out_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);
        let other = Scalar::from_i64(3);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, le_scalar_out(&mut ctx, &a, &other, &out));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-scalar-out-fn/test]
    #[test]
    fn op_le_scalar_out_test_dynamic_out_shape_test() {
        let tf = TensorFactory::<i32>::new();

        let sizes = vec![2, 2];
        let out_sizes = vec![4, 1];

        let out = tf.zeros(out_sizes, TensorShapeDynamism::DYNAMIC_BOUND);
        let other = Scalar::from_i64(2);

        let a = tf.make_default(sizes.clone(), vec![3, 1, 2, 4]);
        let mut ctx = context();
        le_scalar_out(&mut ctx, &a, &other, &out);
        assert_tensor_eq!(out, tf.make_default(sizes, vec![0, 1, 1, 0]));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    // [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_all_dtypes_supported() {
        forall_realhbf16_out_tensor::<u8>();
        forall_realhbf16_out_tensor::<i8>();
        forall_realhbf16_out_tensor::<i16>();
        forall_realhbf16_out_tensor::<i32>();
        forall_realhbf16_out_tensor::<i64>();
        forall_realhbf16_out_tensor::<Half>();
        forall_realhbf16_out_tensor::<BFloat16>();
        forall_realhbf16_out_tensor::<f32>();
        forall_realhbf16_out_tensor::<f64>();
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_mismatched_in_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let bb = tf_int.ones_default(vec![2, 2]);
        let out = tf_bool.ones_default(vec![4]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, le_tensor_out(&mut ctx, &a, &bb, &out));
    }

    // PORT-NOTE: guarded in C++ by `ET_SKIP_IF(is_aten, ...)`; non-ATen build runs.
    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_mismatched_in_out_shapes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf_int.ones_default(vec![4]);
        let bb = tf_int.ones_default(vec![4]);
        let out = tf_bool.ones_default(vec![2, 2]);

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, le_tensor_out(&mut ctx, &a, &bb, &out));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_dynamic_out_shape_test() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![2, 2], vec![2, 3, 2, 4]);
        let bb = tf.make_default(vec![2, 2], vec![1, 4, 2, 3]);

        let out = tf.zeros(vec![1, 4], TensorShapeDynamism::DYNAMIC_BOUND);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2, 2], vec![0, 1, 1, 0]));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_test() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![4], vec![2, 3, 2, 4]);
        let bb = tf.make_default(vec![1, 1], vec![3]);

        let out = tf.zeros_default(vec![1, 4]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);
        assert_tensor_eq!(out, tf.make_default(vec![1, 4], vec![1, 1, 1, 0]));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast2_d_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![1, 10], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let bb = tf.make_default(vec![6, 1], vec![2, 4, 6, 8, 10, 12]);

        let out = tf_bool.zeros_default(vec![6, 10]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            true, true, false, false, false, false, false, false, false, false, // Row 0 (b=2)
            true, true, true, true, false, false, false, false, false, false, // Row 1 (b=4)
            true, true, true, true, true, true, false, false, false, false, // Row 2 (b=6)
            true, true, true, true, true, true, true, true, false, false, // Row 3 (b=8)
            true, true, true, true, true, true, true, true, true, true, // Row 4 (b=10)
            true, true, true, true, true, true, true, true, true, true, // Row 5 (b=12)
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![6, 10], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast1_d_to2_d_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![6, 1], vec![2, 4, 6, 8, 10, 12]);
        let bb = tf.make_default(vec![1, 10], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let out = tf_bool.zeros_default(vec![6, 10]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            false, true, true, true, true, true, true, true, true, true, // Row 0 (a=2)
            false, false, false, true, true, true, true, true, true, true, // Row 1 (a=4)
            false, false, false, false, false, true, true, true, true, true, // Row 2 (a=6)
            false, false, false, false, false, false, false, true, true, true, // Row 3 (a=8)
            false, false, false, false, false, false, false, false, false,
            true, // Row 4 (a=10)
            false, false, false, false, false, false, false, false, false,
            false, // Row 5 (a=12)
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![6, 10], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_reverse_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![6, 1], vec![2, 4, 6, 8, 10, 12]);
        let bb = tf.make_default(vec![1, 10], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let out = tf_bool.zeros_default(vec![6, 10]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            false, true, true, true, true, true, true, true, true, true, // Row 0 (a=2)
            false, false, false, true, true, true, true, true, true, true, // Row 1 (a=4)
            false, false, false, false, false, true, true, true, true, true, // Row 2 (a=6)
            false, false, false, false, false, false, false, true, true, true, // Row 3 (a=8)
            false, false, false, false, false, false, false, false, false,
            true, // Row 4 (a=10)
            false, false, false, false, false, false, false, false, false,
            false, // Row 5 (a=12)
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![6, 10], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_last_dim_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![3, 4, 1], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let bb = tf.make_default(
            vec![3, 4, 5],
            vec![
                1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, // slice 0
                5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, // slice 1
                9, 10, 11, 12, 13, 10, 11, 12, 13, 14, 11, 12, 13, 14, 15, 12, 13, 14, 15,
                16, // slice 2
            ],
        );

        let out = tf_bool.zeros_default(vec![3, 4, 5]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![true; 60];
        assert_tensor_eq!(out, tf_bool.make_default(vec![3, 4, 5], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_last_dim_reverse_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(
            vec![3, 4, 5],
            vec![
                1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, // slice 0
                5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, // slice 1
                9, 10, 11, 12, 13, 10, 11, 12, 13, 14, 11, 12, 13, 14, 15, 12, 13, 14, 15,
                16, // slice 2
            ],
        );
        let bb = tf.make_default(
            vec![3, 4, 1],
            vec![5, 5, 5, 5, 10, 10, 10, 10, 15, 15, 15, 15],
        );

        let out = tf_bool.zeros_default(vec![3, 4, 5]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            true, true, true, true, true, // [1,2,3,4,5] <= 5
            true, true, true, true, false, // [2,3,4,5,6] <= 5
            true, true, true, false, false, // [3,4,5,6,7] <= 5
            true, true, false, false, false, // [4,5,6,7,8] <= 5
            true, true, true, true, true, // [5,6,7,8,9] <= 10
            true, true, true, true, true, // [6,7,8,9,10] <= 10
            true, true, true, true, false, // [7,8,9,10,11] <= 10
            true, true, true, false, false, // [8,9,10,11,12] <= 10
            true, true, true, true, true, // [9,10,11,12,13] <= 15
            true, true, true, true, true, // [10,11,12,13,14] <= 15
            true, true, true, true, true, // [11,12,13,14,15] <= 15
            true, true, true, true, false, // [12,13,14,15,16] <= 15
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![3, 4, 5], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_nd_by_nd_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![2, 1, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let bb = tf.make_default(
            vec![2, 3, 4],
            vec![
                1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, // slice 0
                5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, // slice 1
            ],
        );

        let out = tf_bool.zeros_default(vec![2, 3, 4]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![true; 24];
        assert_tensor_eq!(out, tf_bool.make_default(vec![2, 3, 4], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast_nd_by_nd_reverse_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(
            vec![2, 3, 4],
            vec![
                1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, // slice 0
                5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, // slice 1
            ],
        );
        let bb = tf.make_default(vec![2, 1, 4], vec![2, 3, 4, 5, 6, 7, 8, 9]);

        let out = tf_bool.zeros_default(vec![2, 3, 4]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            true, true, true, true, // [1,2,3,4] <= [2,3,4,5]
            true, true, true, true, // [2,3,4,5] <= [2,3,4,5]
            false, false, false, false, // [3,4,5,6] <= [2,3,4,5]
            true, true, true, true, // [5,6,7,8] <= [6,7,8,9]
            true, true, true, true, // [6,7,8,9] <= [6,7,8,9]
            false, false, false, false, // [7,8,9,10] <= [6,7,8,9]
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![2, 3, 4], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast2d_by1d_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let bb = tf.make_default(vec![4], vec![2, 4, 6, 8]);

        let out = tf_bool.zeros_default(vec![3, 4]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            true, true, true, true, // [1,2,3,4] <= [2,4,6,8]
            false, false, false, true, // [5,6,7,8] <= [2,4,6,8]
            false, false, false, false, // [9,10,11,12] <= [2,4,6,8]
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![3, 4], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast1_d_to2_d_shape_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![6], vec![1, 3, 5, 7, 9, 11]);
        let bb = tf.make_default(vec![1, 6], vec![2, 4, 6, 8, 10, 12]);

        let out = tf_bool.zeros_default(vec![1, 6]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![true, true, true, true, true, true];
        assert_tensor_eq!(out, tf_bool.make_default(vec![1, 6], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast2_d_by1_d_shape_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![10], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let bb = tf.make_default(vec![6, 1], vec![2, 4, 6, 8, 10, 12]);

        let out = tf_bool.zeros_default(vec![6, 10]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            true, true, false, false, false, false, false, false, false, false, // Row 0 (b=2)
            true, true, true, true, false, false, false, false, false, false, // Row 1 (b=4)
            true, true, true, true, true, true, false, false, false, false, // Row 2 (b=6)
            true, true, true, true, true, true, true, true, false, false, // Row 3 (b=8)
            true, true, true, true, true, true, true, true, true, true, // Row 4 (b=10)
            true, true, true, true, true, true, true, true, true, true, // Row 5 (b=12)
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![6, 10], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_broadcast22d_by1d_reverse_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let a = tf.make_default(vec![4], vec![2, 4, 6, 8]);
        let bb = tf.make_default(vec![3, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let out = tf_bool.zeros_default(vec![3, 4]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &a, &bb, &out);

        let expected_data: Vec<bool> = vec![
            false, false, false, false, // [2,4,6,8] <= [1,2,3,4]
            true, true, true, true, // [2,4,6,8] <= [5,6,7,8]
            true, true, true, true, // [2,4,6,8] <= [9,10,11,12]
        ];
        assert_tensor_eq!(out, tf_bool.make_default(vec![3, 4], expected_data));
    }

    // [spec:et:sem:op-le.torch.executor.native.le-tensor-out-fn/test]
    #[test]
    fn op_le_tensor_out_test_monotonic_increasing_vs_scalar_broadcast_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();

        let lhs_data: Vec<i32> = (0..64).collect();

        let lhs = tf.make_default(vec![64], lhs_data);
        let mut rhs = tf.make_default(vec![1, 1], vec![2]);
        let out = tf_bool.zeros_default(vec![1, 64]);

        let mut ctx = context();
        le_tensor_out(&mut ctx, &lhs, &rhs, &out);

        let expected_data: Vec<bool> = (0..64).map(|i| i <= 2).collect();
        assert_tensor_eq!(out, tf_bool.make_default(vec![1, 64], expected_data));

        // Test with rhs value 4
        rhs = tf.make_default(vec![1, 1], vec![4]);
        let out = tf_bool.zeros_default(vec![1, 64]);

        le_tensor_out(&mut ctx, &lhs, &rhs, &out);

        let expected_data: Vec<bool> = (0..64).map(|i| i <= 4).collect();
        assert_tensor_eq!(out, tf_bool.make_default(vec![1, 64], expected_data));

        // Test with rhs value 10
        rhs = tf.make_default(vec![1, 1], vec![10]);
        let out = tf_bool.zeros_default(vec![1, 64]);

        le_tensor_out(&mut ctx, &lhs, &rhs, &out);

        let expected_data: Vec<bool> = (0..64).map(|i| i <= 10).collect();
        assert_tensor_eq!(out, tf_bool.make_default(vec![1, 64], expected_data));

        // Test with rhs value 32
        rhs = tf.make_default(vec![1, 1], vec![32]);
        let out = tf_bool.zeros_default(vec![1, 64]);

        le_tensor_out(&mut ctx, &lhs, &rhs, &out);

        let expected_data: Vec<bool> = (0..64).map(|i| i <= 32).collect();
        assert_tensor_eq!(out, tf_bool.make_default(vec![1, 64], expected_data));
    }
}
