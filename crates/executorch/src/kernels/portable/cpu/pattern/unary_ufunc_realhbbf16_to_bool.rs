//! Literal port of kernels/portable/cpu/pattern/unary_ufunc_realhbbf16_to_bool.cpp.

use crate::kernels::portable::cpu::pattern::pattern::{AsF32, AsF64};
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `Tensor& out` / returned `Tensor&` become `&'a Tensor` — see
// unary_ufunc_realhbf16.rs for the interior-mutation rationale.

// [spec:et:def:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
// [spec:et:sem:unary-ufunc-realhbbf16-to-bool.torch.executor.native.internal.unary-ufunc-realhbbf16-to-bool-fn]
pub fn unary_ufunc_realhbbf16_to_bool<'a, 'b>(
    fn_float: fn(f32) -> bool,
    fn_double: fn(f64) -> bool,
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Resize for dynamic shape
    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor_same_type(out, in_.sizes()) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor."
    );

    // PORT-NOTE: the C++ message interpolates the actual dtype
    // (`... but got %PRId8 instead.`, out.scalar_type()). The ported
    // `et_kernel_check_msg!` only accepts a message string with no format args
    // (it passes only the stringified condition), so the runtime dtype value is
    // dropped from the log text here.
    crate::et_kernel_check_msg!(
        ctx,
        out.scalar_type() == ScalarType::Bool,
        InvalidArgument,
        out,
        "Expected out tensor to have dtype Bool."
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();

    crate::et_switch_realhbbf16_types!(in_type, ctx, "unary_ufunc_realhbbf16_to_bool", CTYPE_IN, {
        apply_unary_map_fn(
            |val_in: CTYPE_IN| -> bool {
                if CTYPE_IN::IS_DOUBLE {
                    let xi: f64 = val_in.as_f64();
                    fn_double(xi)
                } else {
                    let xi: f32 = val_in.as_f32();
                    fn_float(xi)
                }
            },
            in_.const_data_ptr::<CTYPE_IN>(),
            out.mutable_data_ptr::<bool>(),
            in_.numel() as i64,
            1,
        );
    });

    out
}
