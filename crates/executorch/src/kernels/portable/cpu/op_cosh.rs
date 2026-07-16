//! Literal port of kernels/portable/cpu/op_cosh.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(cosh_out, std::cosh)`
// expands to a kernel fn delegating to `internal::unary_ufunc_realhbbf16_to_floathbf16`.
// `f32::cosh` / `f64::cosh` are the concrete overloads of `std::cosh`.

// [spec:et:def:op-cosh.torch.executor.native.cosh-out-fn]
// [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn]
pub fn cosh_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(f32::cosh, f64::cosh, ctx, in_, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.cosh()
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_handle_bool_input() {
        h::test_bool_input(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(cosh_out, op_reference);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(cosh_out);
    }

    // [spec:et:sem:op-cosh.torch.executor.native.cosh-out-fn/test]
    #[test]
    fn op_cosh_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(cosh_out);
    }
}
