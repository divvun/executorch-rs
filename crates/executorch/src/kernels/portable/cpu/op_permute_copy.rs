//! Literal port of kernels/portable/cpu/op_permute_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_permute_copy_args, get_permute_copy_out_target_size,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, coordinateToIndexWithTrailingDimsMemo, memoizeTrailingDims,
    resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::core::portable_type::tensor_impl::ssize_t;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn]
// [spec:et:sem:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn]
///
/// # Safety
/// `coordinate` must point to at least `tensor.dim()` valid `usize` elements.
unsafe fn increment_coordinate_permuted(
    tensor: &Tensor,
    coordinate: *mut usize,
    dims: IntArrayRef,
) {
    let mut i: i32 = dims.size() as i32 - 1;
    while i >= 0 {
        let d: usize = if *dims.at(i as usize) >= 0 {
            *dims.at(i as usize) as usize
        } else {
            (*dims.at(i as usize) + tensor.dim() as i64) as usize
        };
        unsafe {
            *coordinate.add(d) += 1;
            if *coordinate.add(d) as ssize_t == tensor.size(d as ssize_t) {
                *coordinate.add(d) = 0;
            } else {
                return;
            }
        }
        i -= 1;
    }
}

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-permute-copy.torch.executor.native.permute-copy-out-fn]
// [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn]
#[executorch_macros::et_kernel("aten::permute_copy.out")]
pub fn permute_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dims: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_permute_copy_args(in_, dims, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let mut expected_out_size: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut expected_out_dim: usize = 0;
    unsafe {
        get_permute_copy_out_target_size(
            in_,
            dims,
            expected_out_size.as_mut_ptr(),
            &mut expected_out_dim,
        );
    }
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(expected_out_size.as_ptr(), expected_out_dim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type = out.scalar_type();

    let mut in_coord: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut trailing_dims_memo: [usize; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        memoizeTrailingDims(in_, trailing_dims_memo.as_mut_ptr());
    }

    // in and out must be the same dtype
    crate::et_switch_all_types!(in_type, ctx, "permute_copy.out", CTYPE, {
        let in_data: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_data: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        for i in 0..out.numel() {
            unsafe {
                *out_data.add(i as usize) = *in_data.add(coordinateToIndexWithTrailingDimsMemo(
                    in_,
                    in_coord.as_ptr(),
                    trailing_dims_memo.as_ptr(),
                ));
                increment_coordinate_permuted(in_, in_coord.as_mut_ptr(), dims);
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_permute_copy_out<'a, 'b>(
        self_: &Tensor,
        dims: &[i64],
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            self_,
            IntArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            out,
        )
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_one_d_permute() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64];
        let sizes = vec![2];
        let t_int = tf.make_default(sizes.clone(), vec![1, 2]);
        let out = tf.zeros_default(sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(out, tf.make_default(sizes, vec![1, 2]));
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_permute_with_no_data_reorder() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [1i64, 0, 2];
        let t_int = tf.make_default(
            vec![1, 4, 5],
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
        );
        let new_sizes = vec![4, 1, 5];
        let out = tf.zeros_default(new_sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(
                new_sizes,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
                ]
            )
        );
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    // also verifies check_permute_copy_args (arg gate) and
    // get_permute_copy_out_target_size (permuted out-shape {3,2} asserted below)
    // [spec:et:sem:copy-ops-util.torch.executor.check-permute-copy-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn/test]
    // also verifies increment_coordinate_permuted: transposing [[0,1,2],[3,4,5]] with
    // dims [1,0] to [0,3,1,4,2,5] depends entirely on the permuted per-dim carry order.
    // [spec:et:sem:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn/test]
    #[test]
    fn op_permute_copy_test_two_d_permute() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [1i64, 0];
        let t_int = tf.make_default(vec![2, 3], vec![0, 1, 2, 3, 4, 5]);
        let new_sizes = vec![3, 2];
        let out = tf.zeros_default(new_sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(out, tf.make_default(new_sizes, vec![0, 3, 1, 4, 2, 5]));
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_three_d_permute() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [2i64, 0, 1];
        let t_int = tf.make_default(vec![2, 1, 3], vec![0, 1, 2, 3, 4, 5]);
        let new_sizes = vec![3, 2, 1];
        let out = tf.zeros_default(new_sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(out, tf.make_default(new_sizes, vec![0, 3, 1, 4, 2, 5]));
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_four_d_permute() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64, 3, 2, 1];
        let t_int = tf.make_default(
            vec![2, 3, 3, 4],
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71,
            ],
        );
        let new_sizes = vec![2, 4, 3, 3];
        let out = tf.zeros_default(new_sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(
                new_sizes,
                vec![
                    0, 12, 24, 4, 16, 28, 8, 20, 32, 1, 13, 25, 5, 17, 29, 9, 21, 33, 2, 14, 26, 6,
                    18, 30, 10, 22, 34, 3, 15, 27, 7, 19, 31, 11, 23, 35, 36, 48, 60, 40, 52, 64,
                    44, 56, 68, 37, 49, 61, 41, 53, 65, 45, 57, 69, 38, 50, 62, 42, 54, 66, 46, 58,
                    70, 39, 51, 63, 43, 55, 67, 47, 59, 71,
                ]
            )
        );
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_five_d_permute() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [4i64, 3, 2, 1, 0];
        let sizes = vec![2, 2, 2, 2, 2];
        let t_int = tf.make_default(
            sizes.clone(),
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ],
        );
        let out = tf.zeros_default(sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(
            out,
            tf.make_default(
                sizes,
                vec![
                    0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21,
                    13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
                ]
            )
        );
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_all_dimensions_size_one() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [4i64, 3, 2, 1, 0];
        let sizes = vec![1, 1, 1, 1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes.clone());

        op_permute_copy_out(&t_int, &new_dim, &out);
        assert_tensor_eq!(out, tf.make_default(sizes, vec![1]));
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dupe_dimension_pos() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64, 1, 1];
        let sizes = vec![1, 1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            &t_int,
            IntArrayRef::from_raw_parts(new_dim.as_ptr(), new_dim.len()),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dupe_dimension_pos2() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [1i64, 1, 1];
        let sizes = vec![1, 1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            &t_int,
            IntArrayRef::from_raw_parts(new_dim.as_ptr(), new_dim.len()),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dupe_dimension_neg() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64, 1, -2];
        let sizes = vec![1, 1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            &t_int,
            IntArrayRef::from_raw_parts(new_dim.as_ptr(), new_dim.len()),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dupe_dimension_neg2() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64, 1, -5];
        let sizes = vec![1, 1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            &t_int,
            IntArrayRef::from_raw_parts(new_dim.as_ptr(), new_dim.len()),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_mismatch_dim() {
        let tf = TensorFactory::<i32>::new();

        let new_dim = [0i64, 1, 2];
        let sizes = vec![1, 1];
        let t_int = tf.make_default(sizes.clone(), vec![1]);
        let out = tf.zeros_default(sizes);

        let mut ctx = context();
        permute_copy_out(
            &mut ctx,
            &t_int,
            IntArrayRef::from_raw_parts(new_dim.as_ptr(), new_dim.len()),
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );
        let expected = tf.make_default(
            vec![4, 2, 3],
            vec![
                4, 3, 7, 6, 6, 6, 9, 9, 3, 9, 8, 9, 3, 7, 1, 8, 4, 1, 0, 3, 6, 6, 3, 4,
            ],
        );

        let perm = [2i64, 0, 1];
        let out = tf.zeros(vec![4, 2, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        op_permute_copy_out(&x, &perm, &out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dynamic_shape_upper_bound_larger_than_expected() {
        let tf = TensorFactory::<i32>::new();

        let x = tf.make_default(
            vec![2, 3, 4],
            vec![
                4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6, 6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4,
            ],
        );
        let expected = tf.make_default(
            vec![4, 2, 3],
            vec![
                4, 3, 7, 6, 6, 6, 9, 9, 3, 9, 8, 9, 3, 7, 1, 8, 4, 1, 0, 3, 6, 6, 3, 4,
            ],
        );

        let perm = [2i64, 0, 1];
        let out = tf.zeros(vec![5, 5, 5], TensorShapeDynamism::DYNAMIC_BOUND);
        op_permute_copy_out(&x, &perm, &out);
        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: `ET_SKIP_IF(!SupportedFeatures::get()->output_resize, ...)`. In
    // the non-ATen (portable) build `output_resize` defaults to false
    // (kernels/test/supported_features.yaml), so this test is skipped. Ported as
    // a no-op body to preserve the case.
    // [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn/test]
    #[test]
    fn op_permute_copy_test_dynamic_shape_unbound() {
        // ET_SKIP_IF(!output_resize, ...) -> skipped in the non-ATen build.
    }
}
