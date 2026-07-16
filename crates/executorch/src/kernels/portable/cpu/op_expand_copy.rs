//! Literal port of kernels/portable/cpu/op_expand_copy.cpp.

use crate::kernels::portable::cpu::util::copy_ops_util::{
    check_expand_copy_args, get_expand_copy_out_target_size,
};
use crate::kernels::portable::cpu::util::repeat_util::repeat_tensor;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor, tensor_is_default_dim_order, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

const K_TENSOR_DIMENSION_LIMIT: usize = 16;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `(void)ctx;` dropped.
//
// PORT-NOTE: `repeats` is a raw `*mut i64` buffer (as in C++); the aligned
// descending walk (`i`/`j` unsigned countdowns) and the leading-dim tail loop are
// preserved verbatim. `expand_sizes[j] == self_sizes[i]` compares `int64_t` to a
// `SizesType` (i32) promoted to `int64_t` — mirrored by casting `self_sizes[i]`
// to `i64`.
//
// # Safety
// `repeats` must point to at least `expand_sizes.size()` writable elements.
// [spec:et:def:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn]
// [spec:et:sem:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn]
unsafe fn map_expand_to_repeats(
    self_sizes: ArrayRef<SizesType>,
    expand_sizes: ArrayRef<i64>,
    repeats: *mut i64,
    _repeats_size: usize,
) -> usize {
    let mut j: usize = expand_sizes.size();
    let mut i: usize = self_sizes.size();
    while i > 0 && j > 0 {
        i -= 1;
        j -= 1;

        // Default, just copy the expand size to repeat
        unsafe {
            *repeats.add(j) = *expand_sizes.at(j);
        }
        if *expand_sizes.at(j) == -1 || *expand_sizes.at(j) == *self_sizes.at(i) as i64 {
            unsafe {
                *repeats.add(j) = 1;
            }
        }
    }

    while j > 0 {
        j -= 1;
        unsafe {
            *repeats.add(j) = *expand_sizes.at(j);
        }
    }

    expand_sizes.size()
}

// [spec:et:def:op-expand-copy.torch.executor.native.expand-copy-out-fn]
// [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn]
#[executorch_macros::et_kernel("aten::expand_copy.out")]
pub fn expand_copy_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    self_: &Tensor,
    expand_sizes: ArrayRef<i64>,
    implicit: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_expand_copy_args(self_, expand_sizes, implicit, out),
        InvalidArgument,
        out
    );

    let self_sizes = self_.sizes();

    // Holds the result of converting -1 to the original dim sizes
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut output_rank: usize = 0;
    crate::et_kernel_check!(
        ctx,
        unsafe {
            get_expand_copy_out_target_size(
                self_sizes,
                expand_sizes,
                output_sizes.as_mut_ptr(),
                &mut output_rank,
            )
        },
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_rank)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(self_, out),
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        tensor_is_default_dim_order(self_),
        InvalidArgument,
        out
    );

    // Holds the result of expand_sizes converted to repeat sizes
    let mut repeats: [i64; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let repeats_size = unsafe {
        map_expand_to_repeats(
            self_sizes,
            expand_sizes,
            repeats.as_mut_ptr(),
            K_TENSOR_DIMENSION_LIMIT,
        )
    };

    crate::et_kernel_check!(
        ctx,
        repeat_tensor(
            self_,
            ArrayRef::from_raw_parts(repeats.as_ptr(), repeats_size),
            out
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_expand_copy_out<'a, 'b>(
        self_: &Tensor,
        sizes: &[i64],
        implicit: bool,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            self_,
            ArrayRef::from_raw_parts(sizes.as_ptr(), sizes.len()),
            implicit,
            out,
        )
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_no_op() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);
        let dims: Vec<i64> = vec![2, 2];

        let ret = op_expand_copy_out(&a, &dims, false, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 2]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    // also verifies check_expand_copy_args (arg gate) and
    // get_expand_copy_out_target_size (leading-dim prepend + expand shape).
    // Prepending leading dims drives map_expand_to_repeats' second (j>i) loop.
    // [spec:et:sem:copy-ops-util.torch.executor.check-expand-copy-args-fn/test]
    // [spec:et:sem:copy-ops-util.torch.executor.get-expand-copy-out-target-size-fn/test]
    // [spec:et:sem:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn/test]
    #[test]
    fn op_expand_out_test_prepend_dims() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![3, 3, 3, 2, 2]);
        let dims: Vec<i64> = vec![3, 3, 3, 2, 2];

        let ret = op_expand_copy_out(&a, &dims, false, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![3, 3, 3, 2, 2]),
            0.0,
            Some(0.0)
        ));
    }

    // Growing a size-1 dim to 92 drives map_expand_to_repeats' "expand != self &&
    // != -1 -> repeat = expand" branch; a wrong repeat count would mis-size the out.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    // [spec:et:sem:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn/test]
    #[test]
    fn op_expand_out_test_grow_existing_dim() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 1]);
        let out = tf.zeros_default(vec![2, 92]);
        let dims: Vec<i64> = vec![2, 92];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 92]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_all_negative_ones() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 4, 12]);
        let out = tf.zeros_default(vec![2, 4, 12]);
        let dims: Vec<i64> = vec![-1, -1, -1];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 4, 12]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_all_negative_ones2() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 1, 12]);
        let out = tf.zeros_default(vec![2, 1, 12]);
        let dims: Vec<i64> = vec![-1, -1, -1];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 1, 12]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_ends_negative_ones() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 1, 12]);
        let out = tf.zeros_default(vec![2, 14, 12]);
        let dims: Vec<i64> = vec![-1, 14, -1];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 14, 12]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_more_negative_ones() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 14, 1]);
        let out = tf.zeros_default(vec![2, 14, 12]);
        let dims: Vec<i64> = vec![-1, -1, 12];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.ones_default(vec![2, 14, 12]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_bad_expand_dims_too_small() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 14, 1]);
        let out = tf.ones_default(vec![2, 14]); // undefined
        let dims: Vec<i64> = vec![2];

        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            &a,
            ArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            false,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_bad_leading_negative_ones() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 14, 1]);
        let out = tf.ones_default(vec![2, 14, 1]); // undefined
        let dims: Vec<i64> = vec![-1, -1, -1, -1, 2, 14, 1];

        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            &a,
            ArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            false,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_expand_dims_one_to_n() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 1], vec![3, 3]);
        let out = tf.ones_default(vec![2, 6]);
        let dims: Vec<i64> = vec![2, 6];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![2, 6], vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_expand_one_to_n_plus_new_dim_uniform() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 1], vec![3, 3]);
        let out = tf.ones_default(vec![2, 2, 6]);
        let dims: Vec<i64> = vec![2, 2, 6];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                vec![2, 2, 6],
                vec![
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                ]
            ),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_expand_one_to_n_plus_new_dim_different() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 1], vec![1, 2]);
        let out = tf.ones_default(vec![2, 2, 6]);
        let dims: Vec<i64> = vec![2, 2, 6];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                vec![2, 2, 6],
                vec![
                    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                ]
            ),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_expand_one_to_n_plus_new_dim_different_two() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![1, 2], vec![42, 96]);
        let out = tf.ones_default(vec![2, 6, 2]);
        let dims: Vec<i64> = vec![2, 6, 2];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                vec![2, 6, 2],
                vec![
                    42, 96, 42, 96, 42, 96, 42, 96, 42, 96, 42, 96, 42, 96, 42, 96, 42, 96, 42, 96,
                    42, 96, 42, 96,
                ]
            ),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_bad_out_data_type_good_shape_death() {
        let tf_int = TensorFactory::<i32>::new();
        let a = tf_int.make_default(vec![1, 2], vec![42, 96]);

        let tf_float = TensorFactory::<f32>::new();
        let out = tf_float.ones_default(vec![2, 6, 2]);
        let dims: Vec<i64> = vec![2, 6, 2];

        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            &a,
            ArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            false,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)` — non-aten branch, so the body runs.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_bad_out_shape_good_data_type_death() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![1, 2], vec![42, 96]);
        let out = tf.ones_default(vec![2, 6, 4]);
        let dims: Vec<i64> = vec![2, 6, 2];

        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            &a,
            ArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            false,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_single_to_many() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![1], vec![42]);
        let out = tf.ones_default(vec![4, 4, 4]);
        let dims: Vec<i64> = vec![4, 4, 4];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![4, 4, 4], vec![42i32; 64]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_zero_dim_input_expand_1() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![], vec![3]);
        let out = tf.ones_default(vec![6]);
        let dims: Vec<i64> = vec![6];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![6], vec![3, 3, 3, 3, 3, 3]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_zero_dim_input_expand_2() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![], vec![3]);
        let out = tf.ones_default(vec![6, 2]);
        let dims: Vec<i64> = vec![6, 2];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![6, 2], vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
            0.0,
            Some(0.0)
        ));
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_zero_dim_input_zero_dim_output_expand() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![], vec![3]);
        let out = tf.ones_default(vec![]);
        let dims: Vec<i64> = vec![];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(vec![], vec![3]),
            0.0,
            Some(0.0)
        ));
    }

    // PORT-NOTE: `#ifndef USE_ATEN_LIB` — non-aten branch, so this test is present.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_resized_output() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.make_default(vec![2, 1, 1], vec![42, 42]);

        let out = tf.zeros(vec![2, 6, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let dims: Vec<i64> = vec![2, 3, 4];

        let ret = op_expand_copy_out(&a, &dims, false, &out);
        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &tf.make_default(
                vec![2, 3, 4],
                vec![
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42,
                ]
            ),
            0.0,
            Some(0.0)
        ));
    }

    // PORT-NOTE: `ET_SKIP_IF(is_aten, ...)` — non-aten branch, so the body runs.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_implicit_true() {
        let tf = TensorFactory::<i32>::new();
        let a = tf.ones_default(vec![2, 2]);
        let out = tf.zeros_default(vec![2, 2]);
        let dims: Vec<i64> = vec![2, 2];

        let mut ctx = context();
        expand_copy_out(
            &mut ctx,
            &a,
            ArrayRef::from_raw_parts(dims.as_ptr(), dims.len()),
            true,
            &out,
        );
        assert_ne!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 1, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 5, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );

        let dims: Vec<i64> = vec![2, 5, 3];
        let out = tf.zeros(vec![2, 5, 3], TensorShapeDynamism::DYNAMIC_BOUND);
        op_expand_copy_out(&x, &dims, false, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // PORT-NOTE: `ET_SKIP_IF(!output_resize, ...)` — the portable (non-aten) kernel
    // reports `output_resize = false`, so this test is SKIPPED. Body preserved for
    // correspondence.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_dynamic_shape_upper_bound_larger_than_expected() {
        const OUTPUT_RESIZE: bool = false;
        if !OUTPUT_RESIZE {
            return;
        }
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 1, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 5, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );

        let dims: Vec<i64> = vec![2, 5, 3];
        let out = tf.zeros(vec![10, 10, 10], TensorShapeDynamism::DYNAMIC_BOUND);
        op_expand_copy_out(&x, &dims, false, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }

    // PORT-NOTE: `ET_SKIP_IF(!output_resize, ...)` — the portable (non-aten) kernel
    // reports `output_resize = false`, so this test is SKIPPED. `DYNAMIC_UNBOUND`
    // output resize is genuinely unsupported by the portable kernel. Body preserved
    // for correspondence.
    // [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn/test]
    #[test]
    fn op_expand_out_test_dynamic_shape_unbound() {
        const OUTPUT_RESIZE: bool = false;
        if !OUTPUT_RESIZE {
            return;
        }
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![2, 1, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );
        let expected = tf.make_default(
            vec![2, 5, 3],
            vec![
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.49625658988952637,
                0.7682217955589294,
                0.08847743272781372,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
                0.13203048706054688,
                0.30742281675338745,
                0.6340786814689636,
            ],
        );

        let dims: Vec<i64> = vec![2, 5, 3];
        let out = tf.zeros(vec![1, 1, 1], TensorShapeDynamism::DYNAMIC_UNBOUND);
        op_expand_copy_out(&x, &dims, false, &out);
        assert!(tensors_are_close(&out, &expected, 0.0, Some(0.0)));
    }
}
