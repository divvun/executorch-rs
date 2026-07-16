//! Literal port of kernels/portable/cpu/op_sin.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(sin_out, std::sin)`
// expands to a kernel fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16`. `std::sin` resolves to the
// `float` and `double` overloads; `f32::sin` / `f64::sin` are the concrete Rust
// equivalents.

// [spec:et:def:op-sin.torch.executor.native.sin-out-fn]
// [spec:et:sem:op-sin.torch.executor.native.sin-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(sin_out, |x: f32| x.sin(), |x: f64| x.sin());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as harness;
    use crate::runtime::core::portable_type::tensor::Tensor;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn op_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        sin_out(ctx, self_, out)
    }

    fn op_reference(x: f64) -> f64 {
        x.sin()
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_handle_bool_input() {
        harness::test_bool_input(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_half_output_static_dynamism_support() {
        harness::test_all_real_input_half_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_float_output_static_dynamism_support() {
        harness::test_all_real_input_float_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_double_output_static_dynamism_support() {
        harness::test_all_real_input_double_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_float_output_bound_dynamism_support() {
        harness::test_all_real_input_float_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_double_output_bound_dynamism_support() {
        harness::test_all_real_input_double_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_float_output_unbound_dynamism_support() {
        harness::test_all_real_input_float_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_real_input_double_output_unbound_dynamism_support() {
        harness::test_all_real_input_double_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_all_non_float_output_dtype_dies() {
        harness::test_non_float_output_dtype_dies(op_out);
    }

    // [spec:et:sem:op-sin.torch.executor.native.sin-out-fn/test]
    #[test]
    fn op_sin_out_test_mismatched_input_shapes_dies() {
        harness::test_mismatched_input_shapes_dies(op_out);
    }
}
