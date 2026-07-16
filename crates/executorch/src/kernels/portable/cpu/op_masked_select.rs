//! Literal port of kernels/portable/cpu/op_masked_select.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::{
    get_broadcast_target_size_tensors, linearize_access_indexes_tensor,
    tensors_are_broadcastable_between_tensors,
};
use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_realhbbf16_type,
    tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`). `memcpy` becomes
// `core::ptr::copy_nonoverlapping` over raw `u8` pointers, mirroring the C++
// byte copy of a single element.

// [spec:et:def:op-masked-select.torch.executor.native.masked-select-out-fn]
// [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn]
pub fn masked_select_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    mask: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let in_type: ScalarType = in_.scalar_type();

    crate::et_kernel_check!(ctx, tensor_is_realhbbf16_type(in_), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        mask.scalar_type() == ScalarType::Bool,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, out.scalar_type() == in_type, InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(in_, mask, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_are_broadcastable_between_tensors(in_, mask),
        InvalidArgument,
        out
    );

    // If input or mask is empty, the output should be empty
    if in_.numel() == 0 || mask.numel() == 0 {
        crate::et_kernel_check!(
            ctx,
            resize_tensor_same_type(out, ArrayRef::from_raw_parts([0 as SizesType].as_ptr(), 1))
                == Error::Ok,
            InvalidArgument,
            out
        );
        return out;
    }

    // Compute the shape resulting from broadcasting the mask against the input
    let mut broadcast_ndim: usize = 0;
    let mut broadcast_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let err: Error = get_broadcast_target_size_tensors(
        in_,
        mask,
        broadcast_sizes.as_mut_ptr(),
        K_TENSOR_DIMENSION_LIMIT,
        &mut broadcast_ndim,
    );
    if err != Error::Ok {
        crate::et_kernel_check_msg!(
            ctx,
            false,
            InvalidArgument,
            out,
            "Failed to broadcast input and mask"
        );
    }
    let mut broadcast_numel: usize = 1;
    for i in 0..broadcast_ndim {
        broadcast_numel *= broadcast_sizes[i] as usize;
    }

    // Compute the number of out elements
    let mut mask_true_count: usize = 0;
    let mask_data: *const bool = mask.const_data_ptr::<bool>();
    for i in 0..mask.numel() {
        if unsafe { *mask_data.offset(i as isize) } {
            mask_true_count += 1;
        }
    }
    let out_numel: SizesType =
        (mask_true_count * (broadcast_numel / mask.numel() as usize)) as SizesType;

    // Resize the out tensor
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, ArrayRef::from_raw_parts([out_numel].as_ptr(), 1))
            == Error::Ok,
        InvalidArgument,
        out
    );

    let in_data: *const u8 = in_.const_data_ptr::<u8>();
    let out_data: *mut u8 = out.mutable_data_ptr::<u8>();
    let elem_size = in_.element_size();

    // Figure out if `in` is broadcasted
    let mut in_is_broadcasted: bool = false;
    if in_.dim() != broadcast_ndim as isize {
        in_is_broadcasted = true;
    } else {
        for i in 0..in_.dim() {
            if in_.size(i) != broadcast_sizes[i as usize] as isize {
                in_is_broadcasted = true;
            }
        }
    }

    // Figure out if `mask` is broadcasted
    let mut mask_is_broadcasted: bool = false;
    if mask.dim() != broadcast_ndim as isize {
        mask_is_broadcasted = true;
    } else {
        for i in 0..mask.dim() {
            if mask.size(i) != broadcast_sizes[i as usize] as isize {
                mask_is_broadcasted = true;
            }
        }
    }

    // Figure out if either `in` or `mask` is broadcasted
    let any_is_broadcasted: bool = in_is_broadcasted || mask_is_broadcasted;

    let mut out_ix: usize = 0;
    for i in 0..broadcast_numel {
        let mut in_linear_index: usize = i;
        let mut mask_linear_index: usize = i;

        // If either `in` or `mask` is broadcasted, we need to compute the indexes
        // in the broadcasted space.
        if any_is_broadcasted {
            let mut broadcast_indexes: [usize; K_TENSOR_DIMENSION_LIMIT] =
                [0; K_TENSOR_DIMENSION_LIMIT];
            delinearize_index(
                i,
                ArrayRef::from_raw_parts(broadcast_sizes.as_ptr(), broadcast_ndim),
                broadcast_indexes.as_mut_ptr(),
                K_TENSOR_DIMENSION_LIMIT,
            );

            if in_is_broadcasted {
                in_linear_index = linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(broadcast_indexes.as_ptr(), broadcast_ndim),
                    broadcast_ndim as isize,
                    in_,
                );
            }
            if mask_is_broadcasted {
                mask_linear_index = linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(broadcast_indexes.as_ptr(), broadcast_ndim),
                    broadcast_ndim as isize,
                    mask,
                );
            }
        }

        // If the mask is true, copy the value from `in` to `out` and increment the
        // `out_ix`
        if unsafe { *mask_data.offset(mask_linear_index as isize) } {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    in_data.offset((in_linear_index as isize) * (elem_size as isize)),
                    out_data.offset((out_ix as isize) * (elem_size as isize)),
                    elem_size as usize,
                );
            }
            out_ix += 1;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_masked_select_out<'a, 'b>(
        in_: &Tensor,
        mask: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        masked_select_out(&mut ctx, in_, mask, out)
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_smoke_test() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let out = tf.zeros_default(vec![3]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![3], vec![1, 4, 6]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_broadcast_input() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![3], vec![1, 2, 3]);
        let mask = tf_bool.make_default(vec![2, 3], vec![true, false, false, true, false, true]);
        let out = tf.zeros_default(vec![3]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![3], vec![1, 1, 3]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_broadcast_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let mask = tf_bool.make_default(vec![3], vec![false, true, false]);
        let out = tf.zeros_default(vec![2]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![2], vec![2, 5]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_broadcast_input_and_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.ones_default(vec![2, 3, 4, 1]);
        let mask = tf_bool.ones_default(vec![2, 1, 1, 5]);
        let out = tf.zeros_default(vec![120]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.ones_default(vec![120]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_empty_input() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![2, 0], vec![]);
        let mask = tf_bool.make_default(vec![2, 1], vec![true, true]);
        let out = tf.zeros_default(vec![0]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![0], vec![]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_empty_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![2, 1], vec![100, 200]);
        let mask = tf_bool.make_default(vec![2, 0], vec![]);
        let out = tf.zeros_default(vec![0]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![0], vec![]));
    }

    // [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn/test]
    #[test]
    fn op_masked_select_out_test_empty_input_and_mask() {
        let tf = TensorFactory::<i32>::new();
        let tf_bool = TensorFactory::<bool>::new();
        let in_ = tf.make_default(vec![2, 0], vec![]);
        let mask = tf_bool.make_default(vec![0], vec![]);
        let out = tf.zeros_default(vec![0]);
        op_masked_select_out(&in_, &mask, &out);
        assert_tensor_eq!(out, tf.make_default(vec![0], vec![]));
    }
}
