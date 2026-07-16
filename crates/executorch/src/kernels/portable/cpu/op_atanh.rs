//! Literal port of kernels/portable/cpu/op_atanh.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(atanh_out, std::atanh)`
// expands to a kernel fn forwarding to
// `internal::unary_ufunc_realhbbf16_to_floathbf16(std::atanh, std::atanh, ...)`.
// The `define_unary_ufunc_*!` macro form is sibling in-flight work in pattern.rs;
// this is the literal expansion of that macro. `std::atanh` is passed as both the
// float and double function pointers (`f32::atanh` / `f64::atanh`).

// [spec:et:def:op-atanh.torch.executor.native.atanh-out-fn]
// [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn]
pub fn atanh_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(f32::atanh, f64::atanh, ctx, in_, out)
}

// PORT-NOTE: see op_atan.rs for the shared-harness rationale. Harness inlined via
// `#[path]` include with `op_out = atanh_out` and `op_reference = f64::atanh`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as harness;

    fn op_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        atanh_out(ctx, self_, out)
    }

    fn op_reference(x: f64) -> f64 {
        x.atanh()
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_handle_bool_input() {
        harness::test_bool_input(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_half_output_static_dynamism_support() {
        harness::test_all_real_input_half_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_float_output_static_dynamism_support() {
        harness::test_all_real_input_float_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_double_output_static_dynamism_support() {
        harness::test_all_real_input_double_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_float_output_bound_dynamism_support() {
        harness::test_all_real_input_float_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_double_output_bound_dynamism_support() {
        harness::test_all_real_input_double_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_float_output_unbound_dynamism_support() {
        harness::test_all_real_input_float_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_real_input_double_output_unbound_dynamism_support() {
        harness::test_all_real_input_double_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_all_non_float_output_dtype_dies() {
        harness::test_non_float_output_dtype_dies(op_out);
    }

    // [spec:et:sem:op-atanh.torch.executor.native.atanh-out-fn/test]
    #[test]
    fn op_atanh_out_test_mismatched_input_shapes_dies() {
        harness::test_mismatched_input_shapes_dies(op_out);
    }
}
