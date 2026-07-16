//! Literal port of kernels/portable/cpu/op_asinh.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(asinh_out, std::asinh)`
// expands to a kernel fn forwarding to
// `internal::unary_ufunc_realhbbf16_to_floathbf16(std::asinh, std::asinh, ...)`.
// `std::asinh` is passed as both the float and double function pointers
// (`f32::asinh` / `f64::asinh`).

// [spec:et:def:op-asinh.torch.executor.native.asinh-out-fn]
// [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn]
pub fn asinh_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(f32::asinh, f64::asinh, ctx, in_, out)
}

// PORT-NOTE: see op_atan.rs for the shared-harness rationale. Harness inlined via
// `#[path]` include with `op_out = asinh_out` and `op_reference = f64::asinh`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as harness;

    fn op_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        asinh_out(ctx, self_, out)
    }

    fn op_reference(x: f64) -> f64 {
        x.asinh()
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_handle_bool_input() {
        harness::test_bool_input(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_half_output_static_dynamism_support() {
        harness::test_all_real_input_half_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_float_output_static_dynamism_support() {
        harness::test_all_real_input_float_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_double_output_static_dynamism_support() {
        harness::test_all_real_input_double_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_float_output_bound_dynamism_support() {
        harness::test_all_real_input_float_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_double_output_bound_dynamism_support() {
        harness::test_all_real_input_double_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_float_output_unbound_dynamism_support() {
        harness::test_all_real_input_float_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_real_input_double_output_unbound_dynamism_support() {
        harness::test_all_real_input_double_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_all_non_float_output_dtype_dies() {
        harness::test_non_float_output_dtype_dies(op_out);
    }

    // [spec:et:sem:op-asinh.torch.executor.native.asinh-out-fn/test]
    #[test]
    fn op_asinh_out_test_mismatched_input_shapes_dies() {
        harness::test_mismatched_input_shapes_dies(op_out);
    }
}
