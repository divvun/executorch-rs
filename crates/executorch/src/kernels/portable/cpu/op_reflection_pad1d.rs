//! Literal port of kernels/portable/cpu/op_reflection_pad1d.cpp.

use crate::kernels::portable::cpu::util::padding_util::{
    check_padding_args, get_padding_out_target_size, pad1d, reflection_ix,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer).

// [spec:et:def:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn]
// [spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn]
pub fn reflection_pad1d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    padding: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // (void)ctx;

    crate::et_kernel_check!(
        ctx,
        check_padding_args(1, in_, padding, out, /*reflection*/ true),
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

    let mut target_sizes: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut target_ndim: usize = 0;
    unsafe {
        get_padding_out_target_size(1, in_, padding, target_sizes.as_mut_ptr(), &mut target_ndim);
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
    let name = "reflection_pad1d.out";

    crate::et_switch_all_types!(in_type, ctx, name, CTYPE, {
        pad1d::<CTYPE, _>(&reflection_ix, in_, out, padding);
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

    fn op_reflection_pad1d_out<'a, 'b>(
        input: &Tensor,
        padding: &[i64],
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        reflection_pad1d_out(
            &mut ctx,
            input,
            ArrayRef::from_raw_parts(padding.as_ptr(), padding.len()),
            out,
        )
    }

    // [spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn/test]
    // also verifies pad1d, reflection_ix (reflected output values [1,0,1,2,1,0,...]),
    // check_padding_args (accept path), and get_padding_out_target_size ([2,3] -> [2,6])
    // [spec:et:sem:padding-util.torch.executor.pad1d-fn/test]
    // [spec:et:sem:padding-util.torch.executor.reflection-ix-fn/test]
    // [spec:et:sem:padding-util.torch.executor.check-padding-args-fn/test]
    // [spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn/test]
    #[test]
    fn op_reflection_pad1d_out_test_smoke_test() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let padding_data = [1i64, 2];
        let out = tf_float.zeros_default(vec![2, 6]);
        let out_expected = tf_float.make_default(
            vec![2, 6],
            vec![1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0],
        );
        op_reflection_pad1d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn/test]
    #[test]
    fn op_reflection_pad1d_out_test_smoke_test_neg_left_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let padding_data = [-1i64, 1];
        let out = tf_float.zeros_default(vec![2, 3]);
        let out_expected = tf_float.make_default(vec![2, 3], vec![1.0, 2.0, 1.0, 4.0, 5.0, 4.0]);
        op_reflection_pad1d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn/test]
    #[test]
    fn op_reflection_pad1d_out_test_smoke_test_neg_right_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let padding_data = [2i64, -4];
        let out = tf_float.zeros_default(vec![2, 1]);
        let out_expected = tf_float.make_default(vec![2, 1], vec![2.0, 5.0]);
        op_reflection_pad1d_out(&self_, &padding_data, &out);
        assert_tensor_close!(out, out_expected);
    }
}
