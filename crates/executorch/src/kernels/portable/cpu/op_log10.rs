//! Literal port of kernels/portable/cpu/op_log10.cpp.

// [spec:et:def:op-log10.torch.executor.native.log10-out-fn]
// [spec:et:sem:op-log10.torch.executor.native.log10-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(log10_out, |x: f32| x.log10(), |x: f64| x
    .log10());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.log10()
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_handle_bool_input() {
        h::test_bool_input(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(log10_out, op_reference);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(log10_out);
    }

    // [spec:et:sem:op-log10.torch.executor.native.log10-out-fn/test]
    #[test]
    fn op_log10_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(log10_out);
    }
}
