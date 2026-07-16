//! Literal port of kernels/portable/cpu/op_stack.cpp.

use crate::kernels::portable::cpu::util::stack_util::stack_out_impl;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-stack.torch.executor.native.stack-out-fn]
// [spec:et:sem:op-stack.torch.executor.native.stack-out-fn]
pub fn stack_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    tensors: ArrayRef<Tensor>,
    dim: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    stack_out_impl(ctx, tensors, dim, out)
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

    fn list<'t>(v: &'t [Tensor]) -> ArrayRef<Tensor<'t>> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    trait FromI32 {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
    impl FromI32 for bool {
        fn from_i32(v: i32) -> Self {
            v != 0
        }
    }
    impl FromI32 for Half {
        fn from_i32(v: i32) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromI32 for BFloat16 {
        fn from_i32(v: i32) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.ones_default(vec![3, 4]);
        let y = tf.zeros_default(vec![3, 4]);
        let inputs = vec![x, y];

        let out = tf.ones_default(vec![3, 2, 4]);
        let mut ctx = context();
        stack_out(&mut ctx, list(&inputs), 1, &out);

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 2, 4],
            vec![
                T::from_i32(1), T::from_i32(1), T::from_i32(1), T::from_i32(1),
                T::from_i32(0), T::from_i32(0), T::from_i32(0), T::from_i32(0),
                T::from_i32(1), T::from_i32(1), T::from_i32(1), T::from_i32(1),
                T::from_i32(0), T::from_i32(0), T::from_i32(0), T::from_i32(0),
                T::from_i32(1), T::from_i32(1), T::from_i32(1), T::from_i32(1),
                T::from_i32(0), T::from_i32(0), T::from_i32(0), T::from_i32(0),
            ],
        );

        assert_tensor_eq!(out, expected);
    }

    // Runs stacking experiments along the given dim (the `run_stack_tests`
    // helper). `out_size` is derived from `expected`'s sizes.
    fn run_stack_tests(inputs: &[Tensor], dim: i64, expected: &Tensor, out_size: Vec<i32>) {
        let inputs_array = list(inputs);

        let tf = TensorFactory::<f64>::new();
        let out = tf.zeros_default(out_size);

        let mut ctx = context();
        let ret = stack_out(&mut ctx, inputs_array, dim, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, *expected);

        let ret = stack_out(&mut ctx, inputs_array, dim - out.dim() as i64, &out);
        assert_tensor_eq!(out, *ret);
        assert_tensor_eq!(out, *expected);
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_insert_front() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![3, 4],
            vec![
                1.,  2.,  3.,  4.,
                5.,  6.,  7.,  8.,
                9., 10., 11., 12.,
            ],
        );
        #[rustfmt::skip]
        let y = tf.make_default(
            vec![3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let inputs = vec![x, y];

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![2, 3, 4],
            vec![
                1.,   2.,   3.,   4.,
                5.,   6.,   7.,   8.,
                9.,  10.,  11.,  12.,
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        run_stack_tests(&inputs, 0, &expected, vec![2, 3, 4]);
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    // via stack_out_impl -> also verifies check_stack_args (arg gate) and
    // get_stack_out_target_size (dim=1 exercises d<dim, d==dim inserted count,
    // and d>dim shift branches; out {3,2,4})
    // stack_out_impl itself is the entry point exercised here: negative-dim
    // normalization, the per-input dim-order checks, resize, and the
    // outer/inner interleave-copy loop over the realhbbf16 dtype switch.
    // [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-impl-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.check-stack-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-stack-out-target-size-fn/test]
    #[test]
    fn op_stack_out_test_insert_middle() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![3, 4],
            vec![
                1.,  2.,  3.,  4.,
                5.,  6.,  7.,  8.,
                9., 10., 11., 12.,
            ],
        );
        #[rustfmt::skip]
        let y = tf.make_default(
            vec![3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let inputs = vec![x, y];

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 2, 4],
            vec![
                1.,   2.,   3.,   4.,
                -1.,  -2.,  -3.,  -4.,
                5.,   6.,   7.,   8.,
                -5.,  -6.,  -7.,  -8.,
                9.,  10.,  11.,  12.,
                -9., -10., -11., -12.,
            ],
        );

        run_stack_tests(&inputs, 1, &expected, vec![3, 2, 4]);
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_insert_end() {
        let tf = TensorFactory::<f64>::new();

        #[rustfmt::skip]
        let x = tf.make_default(
            vec![3, 4],
            vec![
                1.,  2.,  3.,  4.,
                5.,  6.,  7.,  8.,
                9., 10., 11., 12.,
            ],
        );
        #[rustfmt::skip]
        let y = tf.make_default(
            vec![3, 4],
            vec![
                -1.,  -2.,  -3.,  -4.,
                -5.,  -6.,  -7.,  -8.,
                -9., -10., -11., -12.,
            ],
        );

        let inputs = vec![x, y];

        #[rustfmt::skip]
        let expected = tf.make_default(
            vec![3, 4, 2],
            vec![
                1.,  -1.,
                2.,  -2.,
                3.,  -3.,
                4.,  -4.,
                5.,  -5.,
                6.,  -6.,
                7.,  -7.,
                8.,  -8.,
                9.,  -9.,
                10., -10.,
                11., -11.,
                12., -12.,
            ],
        );

        run_stack_tests(&inputs, 2, &expected, vec![3, 4, 2]);
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
        test_dtype::<Half>();
        test_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_no_input_tensors_with_empty_out_tensor_fails() {
        let tf = TensorFactory::<i32>::new();

        let out = tf.make_default(vec![0], vec![]);
        assert_eq!(out.numel(), 0);

        let empty: [Tensor; 0] = [];
        let mut ctx = context();
        et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&empty), 0, &out));
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_all_empty_input_tensors() {
        let tf = TensorFactory::<i32>::new();

        let empty0 = tf.make_default(vec![0, 10, 3], vec![]);
        let empty1 = tf.make_default(vec![0, 10, 3], vec![]);
        let empty2 = tf.make_default(vec![0, 10, 3], vec![]);
        assert_eq!(empty0.numel(), 0);
        let inputs = vec![empty0, empty1, empty2];

        let out = tf.make_default(vec![3, 0, 10, 3], vec![]);
        assert_eq!(out.numel(), 0);

        let mut ctx = context();
        let ret = stack_out(&mut ctx, list(&inputs), 0, &out);
        assert_eq!(ret.numel(), 0);
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![1, 1]);
        let inputs = vec![x];

        let out = tf.zeros_default(vec![1, 1, 1]);

        let invalid_dims: [i64; 6] = [3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&inputs), dim, &out));
        }
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let out = tf_int.zeros_default(vec![1, 2, 2]);

        let inputs = vec![tf_float.ones_default(vec![2, 2])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_out_match_numel_with_extra_dim_at_end_dies() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![1, 2, 2, 1]);

        let inputs = vec![tf.ones_default(vec![2, 2])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_out_match_numel_lack_dim_at_front_dies() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 2]);

        let inputs = vec![tf.ones_default(vec![2, 2])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&inputs), 0, &out));
    }

    // PORT-NOTE: the C++ `ET_SKIP_IF(is_aten, ...)` is a no-op here.
    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_out_regular_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let out = tf.zeros_default(vec![2, 4, 5]);

        let inputs = vec![tf.ones_default(vec![2, 3]), tf.ones_default(vec![2, 3])];

        let mut ctx = context();
        et_expect_kernel_failure!(ctx, stack_out(&mut ctx, list(&inputs), 0, &out));
    }

    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();

        let xv = vec![
            tf.make_default(vec![2, 3], vec![4, 9, 3, 0, 3, 9]),
            tf.make_default(vec![2, 3], vec![7, 3, 7, 3, 1, 6]),
            tf.make_default(vec![2, 3], vec![6, 9, 8, 6, 6, 8]),
            tf.make_default(vec![2, 3], vec![4, 3, 6, 9, 1, 4]),
        ];
        let expected = tf.make_default(
            vec![4, 2, 3],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );

        let out = tf.zeros(vec![4, 2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        stack_out(&mut ctx, list(&xv), 0, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)` skips this for the portable
    // kernel (output_resize default false). Mirrored as an early skip.
    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }

    // PORT-NOTE: C++ `ET_SKIP_IF(!output_resize, ...)` skips this for the portable
    // kernel. Mirrored as an early skip.
    // [spec:et:sem:op-stack.torch.executor.native.stack-out-fn/test]
    #[test]
    fn op_stack_out_test_dynamic_shape_unbound() {
        println!("Skipping: portable kernel does not support output_resize (dynamic shape)");
    }
}
