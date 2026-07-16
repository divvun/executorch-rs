//! Literal port of kernels/portable/cpu/op_sinh.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(sinh_out, std::sinh)`
// expands to a kernel fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16`. `std::sinh` resolves to the
// `float` and `double` overloads; `f32::sinh` / `f64::sinh` are the concrete
// Rust equivalents.

// [spec:et:def:op-sinh.torch.executor.native.sinh-out-fn]
// [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(sinh_out, |x: f32| x.sinh(), |x: f64| x.sinh());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.sinh()
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_handle_bool_input() {
        h::test_bool_input(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(sinh_out, op_reference);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(sinh_out);
    }

    // [spec:et:sem:op-sinh.torch.executor.native.sinh-out-fn/test]
    #[test]
    fn op_sinh_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(sinh_out);
    }
}
