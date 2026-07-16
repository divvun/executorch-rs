//! Literal port of kernels/portable/cpu/op_cos.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(cos_out, std::cos)`
// expands to a kernel fn delegating to `internal::unary_ufunc_realhbbf16_to_floathbf16`.
// `std::cos` is passed for both the float and double overloads, mirroring the
// C++ where the same overloaded name resolves per argument type; here `f32::cos`
// and `f64::cos` are the concrete overloads.

// [spec:et:def:op-cos.torch.executor.native.cos-out-fn]
// [spec:et:sem:op-cos.torch.executor.native.cos-out-fn]
#[executorch_macros::et_kernel("aten::cos.out")]
pub fn cos_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(f32::cos, f64::cos, ctx, in_, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        x.cos()
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_handle_bool_input() {
        h::test_bool_input(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(cos_out, op_reference);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(cos_out);
    }

    // [spec:et:sem:op-cos.torch.executor.native.cos-out-fn/test]
    #[test]
    fn op_cos_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(cos_out);
    }
}
