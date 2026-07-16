//! Literal port of kernels/portable/cpu/pattern/unary_ufunc_realhbf16.cpp.

use crate::kernels::portable::cpu::pattern::pattern::{AsF32, AsF64, FromF32, FromF64};
use crate::kernels::portable::cpu::util::functional_util::apply_unary_map_fn;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{
    resize_tensor_same_type, tensors_have_same_dim_order2, tensors_have_same_shape_and_dtype2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: the C++ takes `Tensor& out` and returns `Tensor&`. The ported
// `Tensor` handle mutates its data/shape through an interior `*mut TensorImpl`
// (`resize_tensor`, `mutable_data_ptr`, and the dim-order/shape checks all take
// `&Tensor`), so `out` is `&'a Tensor` and the same handle is returned — an
// exclusive `&mut Tensor` would conflict with the shared reads these helpers do.

// [spec:et:def:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
// [spec:et:sem:unary-ufunc-realhbf16.torch.executor.native.internal.unary-ufunc-realhbf16-fn]
pub fn unary_ufunc_realhbf16<'a, 'b>(
    fn_float: fn(f32) -> f32,
    fn_double: fn(f64) -> f64,
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

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_shape_and_dtype2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_switch_realhbf16_types!(in_.scalar_type(), ctx, "unary_ufunc_realhbf16", CTYPE, {
        apply_unary_map_fn(
            |val_in: CTYPE| {
                if CTYPE::IS_DOUBLE {
                    let xi: f64 = val_in.as_f64();
                    // C++: `return fn_double(xi);` (CTYPE is double on this arm).
                    CTYPE::from_f64(fn_double(xi))
                } else {
                    let xi: f32 = val_in.as_f32();
                    CTYPE::from_f32(fn_float(xi))
                }
            },
            in_.const_data_ptr::<CTYPE>(),
            out.mutable_data_ptr::<CTYPE>(),
            in_.numel() as i64,
            1,
        );
    });

    out
}
