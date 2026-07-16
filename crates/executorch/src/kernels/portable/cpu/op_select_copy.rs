//! Literal port of kernels/portable/cpu/op_select_copy.cpp.

use crate::kernels::portable::cpu::util::select_copy_util::select_copy_util;
use crate::runtime::core::error::Error;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-select-copy.torch.executor.native.select-copy-int-out-fn]
// [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn]
#[executorch_macros::et_kernel("aten::select_copy.int_out")]
pub fn select_copy_int_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: i64,
    index: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let err: Error = select_copy_util(in_, dim, index, out);
    if err != Error::Ok {
        ctx.fail(err);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::tensor::Tensor as PtTensor;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_select_copy_int_out<'a, 'b>(
        self_: &Tensor,
        dim: i64,
        index: i64,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        select_copy_int_out(&mut ctx, self_, dim, index, out)
    }

    trait FromF64Elem: Copy {
        fn from_f64(v: f64) -> Self;
    }
    macro_rules! impl_from_f64_num {
        ($($t:ty),*) => {$(impl FromF64Elem for $t { fn from_f64(v: f64) -> Self { v as $t } })*};
    }
    impl_from_f64_num!(u8, i8, i16, i32, i64, f32, f64);
    impl FromF64Elem for bool {
        fn from_f64(v: f64) -> Self {
            v != 0.0
        }
    }

    // test_dtype<CTYPE, DTYPE>
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Elem,
    {
        let tf = TensorFactory::<T>::new();

        let x = tf.make_default(
            vec![3, 2, 4],
            vec![
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(1.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
                T::from_f64(0.0),
            ],
        );

        let out_0 = tf.zeros_default(vec![3, 4]);
        let out_1 = tf.ones_default(vec![3, 4]);
        let ret_0 = op_select_copy_int_out(&x, 1, 0, &out_0);
        let ret_1 = op_select_copy_int_out(&x, 1, 1, &out_1);

        assert_tensor_eq!(*ret_0, out_0);
        assert_tensor_eq!(*ret_1, out_1);

        assert_tensor_eq!(*ret_0, tf.ones_default(vec![3, 4]));
        assert_tensor_eq!(*ret_1, tf.zeros_default(vec![3, 4]));
    }

    // Run the test by selecting Tensor x on given dim and all available indexes.
    fn run_test_cases(x: &Tensor, dim: isize, expected: &[PtTensor]) {
        let tf = TensorFactory::<f64>::new();

        let out_size: Vec<i32> = (0..expected[0].sizes().size())
            .map(|i| *expected[0].sizes().at(i))
            .collect();
        let out = tf.ones_default(out_size);

        for idx in 0..x.size(dim) {
            let ret = op_select_copy_int_out(x, dim as i64, idx as i64, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(out, expected[idx as usize]);

            let ret = op_select_copy_int_out(x, dim as i64, (idx - x.size(dim)) as i64, &out);
            assert_tensor_eq!(out, *ret);
            assert_tensor_eq!(out, expected[idx as usize]);
        }
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    // also verifies select_copy_util: negative-index normalization, resize to the
    // dim-dropped shape, and the leading/trailing-dims memcpy loop.
    // [spec:et:sem:select-copy-util.torch.executor.select-copy-util-fn/test]
    #[test]
    fn op_select_copy_int_out_test_select_front_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let out_size = vec![3, 4];

        let _out = tf.zeros_default(out_size.clone());

        let expected_rets = vec![
            tf.make_default(
                out_size.clone(),
                vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            ),
            tf.make_default(
                out_size,
                vec![
                    -1., -2., -3., -4., -5., -6., -7., -8., -9., -10., -11., -12.,
                ],
            ),
        ];

        run_test_cases(&x, 0, &expected_rets);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    // via select_copy_util -> also verifies check_select_copy_out_args (rank/dim/
    // index/dtype gate) and get_select_copy_out_target_size (dim=1 on {2,3,4}
    // drops the selected dim, exercising d<dim and d>=dim branches, out {2,4})
    // [spec:et:sem:copy-ops-util.torch.executor.check-select-copy-out-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn/test]
    #[test]
    fn op_select_copy_int_out_test_select_middle_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let out_size = vec![2, 4];

        let _out = tf.zeros_default(out_size.clone());

        let expected_rets = vec![
            tf.make_default(out_size.clone(), vec![1., 2., 3., 4., -1., -2., -3., -4.]),
            tf.make_default(out_size.clone(), vec![5., 6., 7., 8., -5., -6., -7., -8.]),
            tf.make_default(out_size, vec![9., 10., 11., 12., -9., -10., -11., -12.]),
        ];

        run_test_cases(&x, 1, &expected_rets);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_select_end_dim_all_indexes() {
        let tf = TensorFactory::<f64>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6.,
                -7., -8., -9., -10., -11., -12.,
            ],
        );

        let out_size = vec![2, 3];

        let _out = tf.zeros_default(out_size.clone());

        let expected_rets = vec![
            tf.make_default(out_size.clone(), vec![1., 5., 9., -1., -5., -9.]),
            tf.make_default(out_size.clone(), vec![2., 6., 10., -2., -6., -10.]),
            tf.make_default(out_size.clone(), vec![3., 7., 11., -3., -7., -11.]),
            tf.make_default(out_size, vec![4., 8., 12., -4., -8., -12.]),
        ];

        run_test_cases(&x, 2, &expected_rets);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_all_dtypes_supported() {
        // ET_FORALL_REAL_TYPES_AND(Bool)
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<i32>();
        test_dtype::<i64>();
        test_dtype::<f32>();
        test_dtype::<f64>();
        test_dtype::<bool>();
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_vector_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![10], vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let out = tf.make_default(vec![], vec![0]);
        assert_eq!(out.numel(), 1);

        let expect = tf.make_default(vec![], vec![5]);
        op_select_copy_int_out(&x, 0, 5, &out);
        assert_tensor_eq!(out, expect);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_empty_tensor_non_zero_n_dims_input_supported() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![3, 0, 10, 3], vec![]);
        assert_eq!(x.numel(), 0);

        let out = tf.make_default(vec![3, 0, 3], vec![]);
        assert_eq!(out.numel(), 0);

        let ret = op_select_copy_int_out(&x, 2, 3, &out);
        assert_eq!(ret.numel(), 0);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_empty_tensor_zero_n_dims_input_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(vec![0], vec![]);
        assert_eq!(x.numel(), 0);

        let out = tf.make_default(vec![], vec![0]);
        assert_eq!(out.numel(), 1);

        let mut ctx = context();
        select_copy_int_out(&mut ctx, &x, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_dim_out_of_bound_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.ones_default(vec![1, 1, 1]);
        let out = tf.zeros_default(vec![1, 1]);

        let invalid_dims: Vec<i64> = vec![3, 4, 5, -4, -5, -6];
        for dim in invalid_dims {
            let mut ctx = context();
            select_copy_int_out(&mut ctx, &x, dim, 0, &out);
            assert_ne!(ctx.failure_state(), Error::Ok);
        }
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_mismatched_dtypes_dies() {
        let tf_int = TensorFactory::<i32>::new();
        let tf_float = TensorFactory::<f32>::new();
        let x = tf_int.zeros_default(vec![1, 2, 2]);

        let out = tf_float.ones_default(vec![2, 2]);

        let mut ctx = context();
        select_copy_int_out(&mut ctx, &x, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_out_match_numel_lack_dim_at_end_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.zeros_default(vec![1, 2, 2, 1]);

        let out = tf.ones_default(vec![2, 2]);

        let mut ctx = context();
        select_copy_int_out(&mut ctx, &x, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_out_match_numel_extra_dim_at_front_dies() {
        let tf = TensorFactory::<i32>::new();
        let x = tf.zeros_default(vec![2, 2]);

        let out = tf.ones_default(vec![1, 2]);

        let mut ctx = context();
        select_copy_int_out(&mut ctx, &x, 0, 0, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_out_size_mismatch_dim_dies() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.zeros_default(vec![2, 4, 7, 5]);

        let out = tf.zeros_default(vec![2, 4, 7]);

        let mut ctx = context();
        select_copy_int_out(&mut ctx, &x, 2, 3, &out);
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    fn dyn_shape_data(tf: &TensorFactory<f32>) -> (PtTensor, PtTensor) {
        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.4900934100151062,
                0.8964447379112244,
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.022325754165649414,
                0.16885894536972046,
                0.2938884496688843,
                0.518521785736084,
                0.6976675987243652,
                0.800011396408081,
                0.16102945804595947,
                0.28226858377456665,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
            ],
        );
        let expected = tf.make_default(
            vec![2, 4],
            vec![
                0.455627977848053,
                0.6323062777519226,
                0.3488934636116028,
                0.40171730518341064,
                0.6816085577011108,
                0.9151939749717712,
                0.39709991216659546,
                0.8741558790206909,
            ],
        );
        (x, expected)
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![2, 4], TensorShapeDynamism::DYNAMIC_BOUND);
        op_select_copy_int_out(&x, 1, 2, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    fn op_select_copy_int_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        op_select_copy_int_out(&x, 1, 2, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: C++ guards this with ET_SKIP_IF(!output_resize); the portable
    // build does not support DYNAMIC_UNBOUND resize, so the test is #[ignore]d.
    // [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn/test]
    #[test]
    #[ignore = "DynamicShapeUnbound: dynamic shape not supported"]
    fn op_select_copy_int_out_test_dynamic_shape_unbound() {
        let tf = TensorFactory::<f32>::new();
        let (x, expected) = dyn_shape_data(&tf);
        let out = tf.zeros(vec![1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_select_copy_int_out(&x, 1, 2, &out);
        assert_tensor_eq!(out, expected);
    }
}
