//! Literal port of kernels/portable/cpu/op_replication_pad2d.cpp.

use crate::kernels::portable::cpu::util::padding_util::{
    check_padding_args, get_padding_out_target_size, pad2d, replication_ix,
};
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor_same_type,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn]
// [spec:et:sem:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn]
pub fn replication_pad2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    padding: ArrayRef<i64>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_padding_args(2, in_, padding, out, false),
        InvalidArgument,
        out
    );

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
    let name = "replication_pad2d.out";

    crate::et_switch_all_types!(in_type, ctx, name, CTYPE, {
        pad2d::<CTYPE, _>(&replication_ix, in_, out, padding);
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
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_replication_pad2d_out<'a, 'b>(
        input: &Tensor,
        padding: ArrayRef<i64>,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        replication_pad2d_out(&mut ctx, input, padding, out)
    }

    fn ir(v: &[i64]) -> ArrayRef<i64> {
        ArrayRef::from_raw_parts(v.as_ptr(), v.len())
    }

    // [spec:et:sem:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn/test]
    #[test]
    fn smoke_test() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data: [i64; 4] = [1, 2, 2, 1];
        let padding = ir(&padding_data);
        let out = tf_float.zeros_default(vec![2, 6, 5]);
        #[rustfmt::skip]
        let out_expected = tf_float.make_default(
            vec![2, 6, 5],
            vec![
                0.,  0.,  1.,  1.,  1.,
                0.,  0.,  1.,  1.,  1.,
                0.,  0.,  1.,  1.,  1.,
                2.,  2.,  3.,  3.,  3.,
                4.,  4.,  5.,  5.,  5.,
                4.,  4.,  5.,  5.,  5.,

                6.,  6.,  7.,  7.,  7.,
                6.,  6.,  7.,  7.,  7.,
                6.,  6.,  7.,  7.,  7.,
                8.,  8.,  9.,  9.,  9.,
                10., 10., 11., 11., 11.,
                10., 10., 11., 11., 11.,
            ],
        );
        op_replication_pad2d_out(&self_, padding, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn/test]
    #[test]
    fn smoke_test_neg_top_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data: [i64; 4] = [1, 1, -2, 0];
        let padding = ir(&padding_data);
        let out = tf_float.zeros_default(vec![2, 1, 4]);
        let out_expected = tf_float.make_default(
            vec![2, 1, 4],
            vec![4.0, 4.0, 5.0, 5.0, 10.0, 10.0, 11.0, 11.0],
        );
        op_replication_pad2d_out(&self_, padding, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn/test]
    #[test]
    fn smoke_test_neg_bottom_pad() {
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_float.make_default(
            vec![2, 3, 2],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let padding_data: [i64; 4] = [1, 1, 3, -5];
        let padding = ir(&padding_data);
        let out = tf_float.zeros_default(vec![2, 1, 4]);
        let out_expected =
            tf_float.make_default(vec![2, 1, 4], vec![0.0, 0.0, 1.0, 1.0, 6.0, 6.0, 7.0, 7.0]);
        op_replication_pad2d_out(&self_, padding, &out);
        assert_tensor_close!(out, out_expected);
    }
}
