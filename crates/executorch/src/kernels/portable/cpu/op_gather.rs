//! Literal port of kernels/portable/cpu/op_gather.cpp.

use crate::kernels::portable::cpu::util::index_util::check_gather_args;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndex, indexToCoordinate, nonzero_dim, resize_tensor,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ `gather_helper` is a private template on `CTYPE`. Rust
// generic function reproduces the per-ctype instantiation dispatched by the
// switch in `gather_out`.

// [spec:et:def:op-gather.torch.executor.native.gather-helper-fn]
// [spec:et:sem:op-gather.torch.executor.native.gather-helper-fn]
fn gather_helper<CTYPE: Copy>(in_: &Tensor, index: &Tensor, out: &Tensor, dim: i64) {
    let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
    let index_data: *const i64 = index.const_data_ptr::<i64>();
    let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

    if index.dim() == 0 {
        unsafe {
            *out_data.add(0) = *in_data.add(*index_data.add(0) as usize);
        }
        return;
    }

    for ix in 0..index.numel() {
        let mut ix_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        unsafe {
            indexToCoordinate(index, ix as usize, ix_coord.as_mut_ptr());
        }

        let mut in_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
        for i in 0..out.dim() {
            if i == dim as isize {
                in_coord[i as usize] = unsafe { *index_data.add(ix as usize) } as usize;
            } else {
                in_coord[i as usize] = ix_coord[i as usize];
            }
        }

        let in_ix: usize = unsafe { coordinateToIndex(in_, in_coord.as_ptr()) };
        let out_ix: usize = unsafe { coordinateToIndex(out, ix_coord.as_ptr()) };

        unsafe {
            *out_data.add(out_ix) = *in_data.add(in_ix);
        }
    }
}

// [spec:et:def:op-gather.torch.executor.native.gather-out-fn]
// [spec:et:sem:op-gather.torch.executor.native.gather-out-fn]
pub fn gather_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mut dim: i64,
    index: &Tensor,
    sparse_grad: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_gather_args(in_, dim, index, sparse_grad, out),
        InvalidArgument,
        out
    );

    if dim < 0 {
        dim += nonzero_dim(in_) as i64;
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, index.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let name = "gather.out";

    crate::et_switch_realhbbf16_types!(in_.scalar_type(), ctx, name, CTYPE, {
        gather_helper::<CTYPE>(in_, index, out, dim);
    });

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
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::{assert_tensor_close, assert_tensor_eq};

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

    trait FromI32: Copy {
        fn from_i32(v: i32) -> Self;
    }
    macro_rules! impl_from_i32 {
        ($($t:ty),*) => {$(impl FromI32 for $t { fn from_i32(v: i32) -> Self { v as $t } })*};
    }
    impl_from_i32!(u8, i8, i16, i32, i64, f32, f64);
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

    fn i<T: FromI32>(v: &[i32]) -> Vec<T> {
        v.iter().map(|&x| T::from_i32(x)).collect()
    }

    fn test_gather_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();
        let sizes = vec![2, 3];
        let self_ = tf_data.make_default(vec![2, 5], i::<T>(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let out = tf_data.zeros_default(sizes.clone());
        let sparse_grad = false;
        let index = tf_index.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);

        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(sizes.clone(), i::<T>(&[1, 7, 3, 6, 2, 8]))
        );

        gather_out(&mut ctx, &self_, 1, &index, sparse_grad, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(sizes, i::<T>(&[1, 2, 1, 7, 6, 7]))
        );

        let self_ = tf_data.make_default(
            vec![2, 3, 3],
            i::<T>(&[
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ]),
        );
        let index = tf_index.make_default(vec![1, 3, 2], vec![0, 1, 1, 2, 0, 2]);
        let out = tf_data.zeros_default(vec![1, 3, 2]);

        gather_out(&mut ctx, &self_, 1, &index, sparse_grad, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(vec![1, 3, 2], i::<T>(&[1, 5, 4, 8, 1, 8]))
        );

        let out = tf_data.zeros_default(vec![1, 3, 2]);
        gather_out(&mut ctx, &self_, 2, &index, sparse_grad, &out);
        assert_tensor_eq!(
            out,
            tf_data.make_default(vec![1, 3, 2], i::<T>(&[1, 2, 5, 6, 7, 9]))
        );
    }

    fn test_gather_out_invalid_dim<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI32,
    {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<T>::new();
        let self_ = tf_data.make_default(vec![2, 5], i::<T>(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let sizes = vec![2, 3];
        let index = tf_index.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);
        let sparse_grad = false;
        let out = tf_data.zeros_default(sizes.clone());

        let mut ctx = context();
        // Invalid dim should die
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, -3, &index, sparse_grad, &out)
        );
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 2, &index, sparse_grad, &out)
        );

        // Self and index should have same number of dimensions
        let index = tf_index.zeros_default(vec![2, 2, 2]);
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );

        // Size of dimension of index should be smaller than the size of that
        // dimension of self if dimension != dim
        let index = tf_index.zeros_default(vec![3, 5]);
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 1, &index, sparse_grad, &out)
        );

        // Index out of bound for self in dim
        let index = tf_index.make_default(vec![2, 3], vec![0, 1, 2, 0, 1, 2]);
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );
    }

    fn test_dynamic_shape(out_shape: Vec<i32>, dynamism: TensorShapeDynamism) {
        let tf = TensorFactory::<i32>::new();
        let tf_index = TensorFactory::<i64>::new();

        let input = tf.ones_default(vec![2, 3, 4]);
        let index = tf_index.zeros_default(vec![2, 3, 4]);
        let sparse_grad = false;
        let expected = tf.ones_default(vec![2, 3, 4]);
        let out = tf.zeros(out_shape, dynamism);

        let mut ctx = context();
        gather_out(&mut ctx, &input, 2, &index, sparse_grad, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    // also verifies check_gather_args (valid path returns true).
    // Gathering along dims 0/1/2 pins gather_helper's index-driven strided copy.
    // [spec:et:sem:index-util.torch.executor.check-gather-args-fn/test]
    // [spec:et:sem:op-gather.torch.executor.native.gather-helper-fn/test]
    #[test]
    fn op_gather_out_test_all_valid_input_output_support() {
        // ET_FORALL_REALHBF16_TYPES
        test_gather_out::<u8>();
        test_gather_out::<i8>();
        test_gather_out::<i16>();
        test_gather_out::<i32>();
        test_gather_out::<i64>();
        test_gather_out::<Half>();
        test_gather_out::<f32>();
        test_gather_out::<f64>();
        test_gather_out::<BFloat16>();
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_infinity_and_nan_test() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();
        let self_ = tf_data.make_default(
            vec![2, 5],
            vec![
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                2.33,
                3.14,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                3.14,
                2.33,
            ],
        );
        let sizes = vec![2, 3];
        let index = tf_index.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);
        let sparse_grad = false;
        let out = tf_data.zeros_default(sizes.clone());

        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_close!(
            out,
            tf_data.make_default(
                sizes,
                vec![
                    f32::INFINITY,
                    f32::INFINITY,
                    f32::NAN,
                    f32::NAN,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                ]
            )
        );
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_invalid_dimensions_dies() {
        // ET_FORALL_REAL_TYPES
        test_gather_out_invalid_dim::<u8>();
        test_gather_out_invalid_dim::<i8>();
        test_gather_out_invalid_dim::<i16>();
        test_gather_out_invalid_dim::<i32>();
        test_gather_out_invalid_dim::<i64>();
        test_gather_out_invalid_dim::<f32>();
        test_gather_out_invalid_dim::<f64>();
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_mismatched_input_dtypes_dies() {
        let tf_byte = TensorFactory::<u8>::new();
        let tf_char = TensorFactory::<i8>::new();
        let tf_long = TensorFactory::<i64>::new();

        let self_ = tf_char.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let sizes = vec![2, 3];
        let index = tf_byte.make_default(sizes.clone(), vec![0, 1, 0, 0, 1, 0]);
        let sparse_grad = false;
        let out = tf_char.zeros_default(sizes.clone());

        let mut ctx = context();
        // Types other than long for index should die
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );

        // Mismatched dtype of self and out should die
        let self_ = tf_byte.make_default(vec![2, 5], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let index = tf_long.make_default(sizes.clone(), vec![0, 1, 0, 1, 0, 1]);
        let out = tf_char.zeros_default(sizes);
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_dynamic_shape_upper_bound_same_as_expected() {
        test_dynamic_shape(vec![2, 3, 4], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        test_dynamic_shape(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
    }

    // PORT-NOTE: C++ guards with `ET_SKIP_IF(!output_resize, ...)`; the portable
    // kernel's `output_resize` supported-feature defaults to false (unbound
    // dynamic resize unsupported), so the C++ test is skipped. Ported + ignored.
    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    #[ignore]
    fn op_gather_out_test_dynamic_shape_unbound() {
        test_dynamic_shape(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_empty_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![2, 5]);
        let sizes = vec![2, 0, 3];
        let index = tf_index.zeros_default(sizes.clone());
        let sparse_grad = false;
        let out = tf_data.zeros_default(sizes.clone());
        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_close!(out, tf_data.zeros_default(sizes));
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_valid_zero_dim() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let index = tf_index.zeros_default(vec![]);
        let sparse_grad = false;
        let out = tf_data.zeros_default(vec![]);
        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![3.14]));
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_invalid_zero_dim_input() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.ones_default(vec![]);
        let sizes = vec![2, 3];
        let index = tf_index.make_default(sizes.clone(), vec![0, 0, 0, 0, 0, 0]);
        let sparse_grad = false;
        let out = tf_data.zeros_default(sizes);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_invalid_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let sizes: Vec<i32> = vec![];
        let index = tf_index.make_default(sizes.clone(), vec![2]);
        let sparse_grad = false;
        let out = tf_data.zeros_default(sizes);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 1, &index, sparse_grad, &out)
        );
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_valid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let sizes = vec![3];
        let index = tf_index.make_default(sizes.clone(), vec![0, 0, 0]);
        let sparse_grad = false;
        let out = tf_data.make_default(vec![3], vec![2.71, 2.71, 2.71]);
        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![3], vec![3.14, 3.14, 3.14]));
        let _ = sizes;
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_valid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let sizes: Vec<i32> = vec![];
        let index = tf_index.make_default(sizes.clone(), vec![2]);
        let sparse_grad = false;
        let out = tf_data.make_default(sizes, vec![1729.]);
        let mut ctx = context();
        gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out);
        assert_tensor_close!(out, tf_data.make_default(vec![], vec![30.]));
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_invalid_zero_dim_input_and_one_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![], vec![3.14]);
        let sizes = vec![3];
        let index = tf_index.make_default(sizes.clone(), vec![10, 100, 1000]);
        let sparse_grad = false;
        let out = tf_data.make_default(vec![3], vec![2.71, 2.71, 2.71]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );
        let _ = sizes;
    }

    // [spec:et:sem:op-gather.torch.executor.native.gather-out-fn/test]
    #[test]
    fn op_gather_out_test_invalid_one_dim_input_and_zero_dim_index() {
        let tf_index = TensorFactory::<i64>::new();
        let tf_data = TensorFactory::<f32>::new();

        let self_ = tf_data.make_default(vec![3], vec![10., 20., 30.]);
        let sizes: Vec<i32> = vec![];
        let index = tf_index.make_default(sizes.clone(), vec![100]);
        let sparse_grad = false;
        let out = tf_data.make_default(sizes, vec![1729.]);
        let mut ctx = context();
        et_expect_kernel_failure!(
            ctx,
            gather_out(&mut ctx, &self_, 0, &index, sparse_grad, &out)
        );
    }
}
