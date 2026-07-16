//! Literal port of kernels/portable/cpu/op_reciprocal.cpp.

use crate::kernels::portable::cpu::pattern::pattern::unary_ufunc_realhbbf16_to_floathbf16;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// template <typename T>
// [spec:et:def:op-reciprocal.torch.executor.native.reciprocal-fn]
// [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-fn]
//
// PORT-NOTE: the C++ `reciprocal<T>(T x) { return 1.0 / x; }` is instantiated for
// `float` and `double` by the DEFINE macro below (which passes `fn, fn` to the
// pattern's `float(*)(float)` / `double(*)(double)` parameters). The literal
// `1.0` is a `double`, so the `float` instantiation divides in double then
// narrows on return; the `double` instantiation stays in double. Modeled as two
// concrete functions matching those two instantiations.
fn reciprocal_f32(x: f32) -> f32 {
    (1.0f64 / x as f64) as f32
}
fn reciprocal_f64(x: f64) -> f64 {
    1.0f64 / x
}

// [spec:et:def:op-reciprocal.torch.executor.native.reciprocal-out-fn]
// [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn]
//
// PORT-NOTE: the C++ macro
// `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(reciprocal_out, reciprocal)`
// expands to a full `reciprocal_out(ctx, in, out)` fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16(fn, fn, ctx, in, out)`. The
// `DEFINE_*` macro_rules! that PORTING assigns to `pattern.rs` is not yet
// available, so the expansion is written out here verbatim. Unresolved
// cross-module reference (macro home is `pattern.rs`).
pub fn reciprocal_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    unary_ufunc_realhbbf16_to_floathbf16(reciprocal_f32, reciprocal_f64, ctx, in_, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;

    fn op_reference(x: f64) -> f64 {
        1.0 / x
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_handle_bool_input() {
        h::test_bool_input(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(
            reciprocal_out,
            op_reference,
        );
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    // also verifies reciprocal_f32: the float-output path computes each element via reciprocal_f32.
    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    // also verifies reciprocal_f64: the double-output path computes each element via reciprocal_f64.
    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(reciprocal_out, op_reference);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(reciprocal_out);
    }

    // [spec:et:sem:op-reciprocal.torch.executor.native.reciprocal-out-fn/test]
    #[test]
    fn op_reciprocal_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(reciprocal_out);
    }
}
