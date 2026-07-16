//! Literal port of kernels/portable/cpu/op_maximum.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::math_util::max_override;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{can_cast, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE (cross-module): the compile-time `op_name` template parameter
// ("maximum.out") of the C++ `apply_bitensor_elementwise_fn` is dropped — the
// ported wrapper takes no op-name argument. `support_noncontiguous_tensors` is
// fixed `true`, matching the C++ default. The compute closure takes the loaded
// inputs as a `&[CTYPE_COMPUTE]` slice (`vals[0]`, `vals[1]`), mirroring the C++
// generic-lambda `(val_a, val_b)`. `utils::get_compute_type` maps to
// `elementwise_util::get_compute_type`; `utils::max_override` to
// `math_util::max_override`.

// [spec:et:def:op-maximum.torch.executor.native.maximum-out-fn]
// [spec:et:sem:op-maximum.torch.executor.native.maximum-out-fn]
pub fn maximum_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type: ScalarType = promote_types(a.scalar_type(), b.scalar_type(), false);

    // Check Common Dtype
    crate::et_kernel_check!(
        ctx,
        can_cast(common_type, out.scalar_type()),
        InvalidArgument,
        out
    );

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order3(a, b, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type: ScalarType = get_compute_type(&mut common_type);

    crate::et_switch_realb_types!(compute_type, ctx, "maximum.out", CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { max_override(vals[0], vals[1]) },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBBF16,
            true,
        );
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_maximum_out<'a, 'b>(
        self_: &Tensor,
        other: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        let mut ctx = context();
        maximum_out(&mut ctx, self_, other, out)
    }

    // [spec:et:sem:op-maximum.torch.executor.native.maximum-out-fn/test]
    // also verifies max_override (max(-86.125, 36) == 36)
    // [spec:et:sem:math-util.torch.executor.native.utils.max-override-fn/test]
    #[test]
    fn op_maximum_out_test_smoke_test() {
        let tf_double = TensorFactory::<f64>::new();
        let tf_float = TensorFactory::<f32>::new();
        let tf_short = TensorFactory::<i16>::new();

        let self_ = tf_float.full(vec![], -86.125f32, TensorShapeDynamism::STATIC);
        let other = tf_short.full(vec![], 36i16, TensorShapeDynamism::STATIC);
        let out = tf_double.zeros_default(vec![]);
        let out_expected = tf_double.full(vec![], 36.0f64, TensorShapeDynamism::STATIC);
        op_maximum_out(&self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-maximum.torch.executor.native.maximum-out-fn/test]
    #[test]
    fn op_maximum_out_test_smoke_test_larger() {
        let tf_float = TensorFactory::<f32>::new();

        let a: Vec<f32> = (0..18).map(|i| (i as f32) - 8.0).collect();
        let self_ = tf_float.make_default(vec![18], a);
        let other = tf_float.full(vec![18], 4.0f32, TensorShapeDynamism::STATIC);
        let out = tf_float.zeros_default(vec![18]);
        let out_expected = tf_float.make_default(
            vec![18],
            vec![
                4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 6., 7., 8., 9.,
            ],
        );
        op_maximum_out(&self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }
}
