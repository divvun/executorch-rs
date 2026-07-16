//! Literal port of kernels/portable/cpu/op_acosh.cpp.

// C++: DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(acosh_out, std::acosh)
// The overloaded `std::acosh` resolves to the `float` and `double` variants at
// the two internal call sites; supplied here as the f32 and f64 versions.
// [spec:et:def:op-acosh.torch.executor.native.acosh-out-fn]
// [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(acosh_out, |x: f32| x.acosh(), |x: f64| x
    .acosh());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.acosh()
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_handle_bool_input() {
        h::test_bool_input(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(acosh_out, op_reference);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(acosh_out);
    }

    // [spec:et:sem:op-acosh.torch.executor.native.acosh-out-fn/test]
    #[test]
    fn op_acosh_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(acosh_out);
    }
}
