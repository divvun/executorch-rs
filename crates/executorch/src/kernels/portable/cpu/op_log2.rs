//! Literal port of kernels/portable/cpu/op_log2.cpp.

// [spec:et:def:op-log2.torch.executor.native.log2-out-fn]
// [spec:et:sem:op-log2.torch.executor.native.log2-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(log2_out, |x: f32| x.log2(), |x: f64| x.log2());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.log2()
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_handle_bool_input() {
        h::test_bool_input(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(log2_out, op_reference);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(log2_out);
    }

    // [spec:et:sem:op-log2.torch.executor.native.log2-out-fn/test]
    #[test]
    fn op_log2_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(log2_out);
    }
}
