//! Literal port of kernels/portable/cpu/op_expm1.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(expm1_out, std::expm1)`
// expands to a kernel fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16`. `std::expm1` resolves to the
// `float` and `double` overloads; `f32::exp_m1` / `f64::exp_m1` are the concrete
// Rust equivalents (exp(x)-1 with improved accuracy near 0).

// [spec:et:def:op-expm1.torch.executor.native.expm1-out-fn]
// [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(expm1_out, |x: f32| x.exp_m1(), |x: f64| x
    .exp_m1());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.exp_m1()
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_handle_bool_input() {
        h::test_bool_input(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(expm1_out, op_reference);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(expm1_out);
    }

    // [spec:et:sem:op-expm1.torch.executor.native.expm1-out-fn/test]
    #[test]
    fn op_expm1_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(expm1_out);
    }
}
