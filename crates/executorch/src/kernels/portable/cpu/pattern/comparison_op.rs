//! Literal port of kernels/portable/cpu/pattern/comparison_op.h.

use crate::kernels::portable::cpu::scalar_utils::{promote_type_with_scalar, scalar_to};
use crate::kernels::portable::cpu::util::broadcast_util::resize_to_broadcast_target_size;
use crate::kernels::portable::cpu::util::dtype_util::{StaticCast, SupportedTensorDtypes};
use crate::kernels::portable::cpu::util::elementwise_util::{
    apply_bitensor_elementwise_fn, apply_unitensor_elementwise_fn, get_compute_type,
};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::{is_floating_type, promote_types};
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_dim_order3,
};
use crate::runtime::core::portable_type::scalar::Scalar;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ patterns are templated on a template-template `Comparison`
// functor (`std::greater`, `std::less`, `std::equal_to`, ...) instantiated per
// compute ctype as `Comparison<CTYPE_COMPUTE>()`. Rust has no template-template
// parameters, so the functor is modeled as a `ComparisonOp` trait with a generic
// `apply<T>` method; each op file supplies a zero-sized type implementing it and
// passes it as the `C` type parameter, reproducing `Comparison<CTYPE_COMPUTE>()`
// via `C::apply::<CTYPE_COMPUTE>`.
pub trait ComparisonOp {
    fn apply<T: PartialOrd>(a: T, b: T) -> bool;
}

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` (interior
// mutation through `*mut TensorImpl`).
//
// PORT-NOTE (cross-module): the compile-time `op_name` template parameter is
// dropped (the ported `apply_*_elementwise_fn` take no op-name argument).
// `support_noncontiguous_tensors` is fixed `true`, matching the C++ pattern. The
// compute closure returns `CTYPE_COMPUTE`, so the boolean comparison result is
// cast to `CTYPE_COMPUTE` via `StaticCast` before the framework stores it into
// the (Bool) `out` tensor — bug-for-bug equivalent to the C++ bool→out cast.

// [spec:et:def:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]
// [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-tensor-out-fn]
pub fn comparison_tensor_out<'a, 'b, C: ComparisonOp>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type = promote_types(a.scalar_type(), b.scalar_type(), false);
    if is_floating_type(common_type) && a.scalar_type() != b.scalar_type() {
        common_type = ScalarType::Float;
    }

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
    let compute_type = get_compute_type(&mut common_type);

    crate::et_switch_realb_types!(compute_type, ctx, "comparison_tensor_out", CTYPE_COMPUTE, {
        apply_bitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            // TODO: rewrite this to be vectorization-capable.
            |vals: &[CTYPE_COMPUTE]| {
                <CTYPE_COMPUTE as StaticCast<bool>>::static_cast(C::apply::<CTYPE_COMPUTE>(
                    vals[0], vals[1],
                ))
            },
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

// [spec:et:def:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]
// [spec:et:sem:comparison-op.torch.executor.native.internal.comparison-scalar-out-fn]
pub fn comparison_scalar_out<'a, 'b, C: ComparisonOp>(
    ctx: &mut KernelRuntimeContext,
    a: &Tensor,
    b: &Scalar,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Common Dtype
    let mut common_type = promote_type_with_scalar(a.scalar_type(), *b, false);

    // Check Dim Order
    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(a, out),
        InvalidArgument,
        out
    );

    // Resize
    crate::et_kernel_check!(
        ctx,
        resize_tensor_same_type(out, a.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    // Compute Dtype
    let compute_type = get_compute_type(&mut common_type);

    crate::et_switch_realb_types!(compute_type, ctx, "comparison_scalar_out", CTYPE_COMPUTE, {
        let val_b: CTYPE_COMPUTE = scalar_to::<CTYPE_COMPUTE>(b);
        apply_unitensor_elementwise_fn::<CTYPE_COMPUTE, _>(
            |vals: &[CTYPE_COMPUTE]| {
                // TODO: rewrite this to be vectorization-capable.
                <CTYPE_COMPUTE as StaticCast<bool>>::static_cast(C::apply::<CTYPE_COMPUTE>(
                    vals[0], val_b,
                ))
            },
            ctx,
            a,
            SupportedTensorDtypes::REALHBBF16,
            out,
            SupportedTensorDtypes::REALHBBF16,
            true,
        );
    });

    out
}
