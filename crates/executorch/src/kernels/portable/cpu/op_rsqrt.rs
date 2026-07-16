//! Literal port of kernels/portable/cpu/op_rsqrt.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: `T rsqrt(T x) { return 1.0 / std::sqrt(x); }` is a single template
// instantiated at `float` and `double`. The two ufunc-pattern function pointers
// require concrete `fn(f32)->f32` / `fn(f64)->f64`, so the template is expanded
// into the two monomorphizations here, reproducing the C++ mixed-precision
// arithmetic exactly: at `float`, `std::sqrt(x)` runs at float precision, then
// `1.0 / <float>` promotes to `double` (the `1.0` numerator is `double`) and
// narrows back to `float` on return; at `double` everything is `double`.

// [spec:et:def:op-rsqrt.torch.executor.native.rsqrt-fn]
// [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-fn]
fn rsqrt_float(x: f32) -> f32 {
    (1.0 / (x.sqrt() as f64)) as f32
}

fn rsqrt_double(x: f64) -> f64 {
    1.0 / x.sqrt()
}

// [spec:et:def:op-rsqrt.torch.executor.native.rsqrt-out-fn]
// [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn]
// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(rsqrt_out, rsqrt)`
// expands to a body delegating to `internal::unary_ufunc_realhbbf16_to_floathbf16`
// with the ctype-specialized function pointers.
pub fn rsqrt_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(rsqrt_float, rsqrt_double, ctx, in_, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as harness;

    fn op_out<'a, 'b>(
        ctx: &mut KernelRuntimeContext,
        self_: &Tensor,
        out: &'a Tensor<'b>,
    ) -> &'a Tensor<'b> {
        rsqrt_out(ctx, self_, out)
    }

    fn op_reference(x: f64) -> f64 {
        1.0 / x.sqrt()
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_handle_bool_input() {
        harness::test_bool_input(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_half_output_static_dynamism_support() {
        harness::test_all_real_input_half_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    // also verifies rsqrt_float: the float-output path computes each element via rsqrt_float.
    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_float_output_static_dynamism_support() {
        harness::test_all_real_input_float_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    // also verifies rsqrt_double: the double-output path computes each element via rsqrt_double.
    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_double_output_static_dynamism_support() {
        harness::test_all_real_input_double_output_static_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        harness::test_all_real_input_bfloat16_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_float_output_bound_dynamism_support() {
        harness::test_all_real_input_float_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_double_output_bound_dynamism_support() {
        harness::test_all_real_input_double_output_bound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_float_output_unbound_dynamism_support() {
        harness::test_all_real_input_float_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_real_input_double_output_unbound_dynamism_support() {
        harness::test_all_real_input_double_output_unbound_dynamism_support(op_out, op_reference);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_all_non_float_output_dtype_dies() {
        harness::test_non_float_output_dtype_dies(op_out);
    }

    // [spec:et:sem:op-rsqrt.torch.executor.native.rsqrt-out-fn/test]
    #[test]
    fn op_rsqrt_out_test_mismatched_input_shapes_dies() {
        harness::test_mismatched_input_shapes_dies(op_out);
    }
}
