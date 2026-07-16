//! Literal port of kernels/portable/cpu/op_exp.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(exp_out, std::exp)`
// expands to a kernel fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16`. `std::exp` resolves to the
// `float` and `double` overloads; `f32::exp` / `f64::exp` are the concrete Rust
// equivalents.

// [spec:et:def:op-exp.torch.executor.native.exp-out-fn]
// [spec:et:sem:op-exp.torch.executor.native.exp-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(exp_out, |x: f32| x.exp(), |x: f64| x.exp());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.exp()
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_handle_bool_input() {
        h::test_bool_input(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(exp_out, op_reference);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(exp_out);
    }

    // [spec:et:sem:op-exp.torch.executor.native.exp-out-fn/test]
    #[test]
    fn op_exp_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(exp_out);
    }
}
