//! Literal port of kernels/portable/cpu/op_atan2.cpp.

use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::SupportedTensorDtypes;
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, get_compute_type,
};
use crate::kernels::portable::cpu::util::vectorized_math::atan2 as math_atan2;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{is_floating_type, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::tensors_have_same_dim_order3;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-atan2.torch.executor.native.get-common-type-fn]
// [spec:et:sem:op-atan2.torch.executor.native.get-common-type-fn]
fn get_common_type(a_type: ScalarType, b_type: ScalarType) -> ScalarType {
    if is_floating_type(a_type) && is_floating_type(b_type) {
        return promote_types(a_type, b_type, /*half_to_float=*/ false);
    } else if is_floating_type(a_type) {
        return a_type;
    } else if is_floating_type(b_type) {
        return b_type;
    }
    ScalarType::Float
}

// PORT-NOTE: C++ `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through the non-owning handle's raw pointer). The C++ binary functor
// `(val_a, val_b)` becomes a `&[CTYPE_COMPUTE]` closure (`vals[0]`, `vals[1]`),
// following the ported `apply_bitensor_elementwise_fn` contract.

// [spec:et:def:op-atan2.torch.executor.native.atan2-out-fn]
// [spec:et:sem:op-atan2.torch.executor.native.atan2-out-fn]
pub fn atan2_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let common_type: ScalarType = get_common_type(a.scalar_type(), b.scalar_type());

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
    let compute_type: ScalarType = get_compute_type(&mut common_type.clone());

    let op_name = "atan2.out";

    crate::et_switch_float_types!(compute_type, ctx, op_name, CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| -> CTYPE_COMPUTE { math_atan2(vals[0], vals[1]) },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            b,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::FLOATHBF16,
            /*support_noncontiguous=*/ false,
        );
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

    // [spec:et:sem:op-atan2.torch.executor.native.atan2-out-fn/test]
    #[test]
    fn op_atan2_out_test_smoke_test() {
        let tf_double = TensorFactory::<f64>::new();
        let tf_float = TensorFactory::<f32>::new();

        let self_ = tf_double.make_default(
            vec![3, 2],
            vec![20.25, 42.5, 51.625, -46.125, 80.375, -35.75],
        );
        let other = tf_double.make_default(vec![2], vec![-0.625, -2.25]);
        let out = tf_float.zeros_default(vec![3, 2]);
        let out_expected = tf_float.make_default(
            vec![3, 2],
            vec![
                1.6016507148742676,
                1.6236881017684937,
                1.5829023122787476,
                -1.6195381879806519,
                1.5785722732543945,
                -1.633650541305542,
            ],
        );
        let mut ctx = context();
        atan2_out(&mut ctx, &self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }

    // Double x Double inputs drive get_common_type -> Double, so the f64 compute
    // path is selected; the high-precision expected values would fail if
    // get_common_type widened wrongly.
    // [spec:et:sem:op-atan2.torch.executor.native.atan2-out-fn/test]
    // [spec:et:sem:op-atan2.torch.executor.native.get-common-type-fn/test]
    #[test]
    fn op_atan2_out_test_smoke_test_no_broadcasting_same_dtype() {
        let tf_double = TensorFactory::<f64>::new();

        // std::iota(a.begin(), a.end(), -8) over 18 elements.
        let a: Vec<f64> = (0..18).map(|i| (i as f64) - 8.0).collect();
        let b: Vec<f64> = vec![2.0; 18];
        let self_ = tf_double.make_default(vec![18], a);
        let other = tf_double.make_default(vec![18], b);
        let out = tf_double.zeros_default(vec![18]);
        let out_expected = tf_double.make_default(
            vec![18],
            vec![
                -1.3258176636680326,
                -1.2924966677897853,
                -1.2490457723982544,
                -1.1902899496825317,
                -1.1071487177940904,
                -0.9827937232473291,
                -0.7853981633974483,
                -0.4636476090008061,
                0.0000000000000000,
                0.4636476090008061,
                0.7853981633974483,
                0.9827937232473291,
                1.1071487177940904,
                1.1902899496825317,
                1.2490457723982544,
                1.2924966677897853,
                1.3258176636680326,
                1.3521273809209546,
            ],
        );
        let mut ctx = context();
        atan2_out(&mut ctx, &self_, &other, &out);
        assert_tensor_close!(out, out_expected);
    }
}
