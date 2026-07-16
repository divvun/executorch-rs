//! Literal port of kernels/portable/cpu/op_reflection_pad2d.cpp.

use crate::kernels::portable::cpu::util::padding_util::{
    check_padding_args, get_padding_out_target_size, pad2d, reflection_ix,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-reflection-pad2d.torch.executor.native.reflection-pad2d-out-fn]
// [spec:et:sem:op-reflection-pad2d.torch.executor.native.reflection-pad2d-out-fn]
pub fn reflection_pad2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    padding: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_padding_args(2, in_, padding, out, /*reflection*/ true),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut target_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_padding_out_target_size(2, in_, padding, target_sizes.as_mut_ptr(), &mut target_ndim);
    }

    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(
            out,
            ArrayRef::from_raw_parts(target_sizes.as_ptr(), target_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type: ScalarType = in_.scalar_type();
    let name = "reflection_pad2d.out";

    crate::et_switch_all_types!(in_type, ctx, name, CTYPE, {
        pad2d::<CTYPE, _>(&reflection_ix, in_, out, padding);
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_reflection_pad2d_out<'a, 'b>(
        input: &Tensor,
        padding: &[i64],
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        reflection_pad2d_out(
            &mut ctx,
            input,
            ArrayRef::from_raw_parts(padding.as_ptr(), padding.len()),
            out,
        )
    }

    // [spec:et:sem:op-reflection-pad2d.torch.executor.native.reflection-pad2d-out-fn/test]
    // also verifies pad2d (2D reflection padding output asserted exactly)
    // [spec:et:sem:padding-util.torch.executor.pad2d-fn/test]
    #[test]
    fn op_reflection_pad2d_out_test_smoke_test() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data = [1i64, 1, 2, 1];
        let out = tf_float.zeros_default(vec![2, 6, 4]);
        let out_expected = tf_float.make_default(
            vec![2, 6, 4],
            vec![
                5.0, 4.0, 5.0, 4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 3.0, 2.0,
                5.0, 4.0, 5.0, 4.0, 3.0, 2.0, 3.0, 2.0, 11.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0, 8.0,
                7.0, 6.0, 7.0, 6.0, 9.0, 8.0, 9.0, 8.0, 11.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0, 8.0,
            ],
        );
        op_reflection_pad2d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-reflection-pad2d.torch.executor.native.reflection-pad2d-out-fn/test]
    #[test]
    fn op_reflection_pad2d_out_test_smoke_test_neg_top_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data = [1i64, 1, -2, 0];
        let out = tf_float.zeros_default(vec![2, 1, 4]);
        let out_expected = tf_float.make_default(
            vec![2, 1, 4],
            vec![5.0, 4.0, 5.0, 4.0, 11.0, 10.0, 11.0, 10.0],
        );
        op_reflection_pad2d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-reflection-pad2d.torch.executor.native.reflection-pad2d-out-fn/test]
    #[test]
    fn op_reflection_pad2d_out_test_smoke_test_neg_bottom_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data = [1i64, 1, 1, -3];
        let out = tf_float.zeros_default(vec![2, 1, 4]);
        let out_expected =
            tf_float.make_default(vec![2, 1, 4], vec![3.0, 2.0, 3.0, 2.0, 9.0, 8.0, 9.0, 8.0]);
        op_reflection_pad2d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }
}
