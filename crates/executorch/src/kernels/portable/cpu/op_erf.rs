//! Literal port of kernels/portable/cpu/op_erf.cpp.

// PORT-NOTE: `DEFINE_UNARY_UFUNC_REALHBBF16_TO_FLOATHBF16(erf_out, std::erf)`
// expands to a kernel fn delegating to
// `internal::unary_ufunc_realhbbf16_to_floathbf16`. `std::erf` resolves to the
// `float` and `double` overloads at the two internal call sites. Core Rust has no
// `f32::erf` / `f64::erf`, so the two overloads are the `libm` erf functions
// (`erff` / `erf`), mirroring `std::erf`'s float/double dispatch.

// [spec:et:def:op-erf.torch.executor.native.erf-out-fn]
// [spec:et:sem:op-erf.torch.executor.native.erf-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(erf_out, |x: f32| libm::erff(x), |x: f64| {
    libm::erf(x)
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::{internal, tensors_are_close};
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn setup() {
        crate::runtime::platform::platform::pal_init();
    }

    fn context() -> KernelRuntimeContext<'static> {
        setup();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_reference(x: f64) -> f64 {
        libm::erf(x)
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_sanity_check() {
        let tf = TensorFactory::<f32>::new();

        let in_ = tf.make_default(vec![1, 7], vec![-3.0, -2.99, -1.01, 0.0, 1.01, 2.99, 3.0]);
        let out = tf.zeros_default(vec![1, 7]);
        let expected = tf.make_default(
            vec![1, 7],
            vec![
                -0.999978, -0.999976, -0.846811, 0.000000, 0.846811, 0.999976, 0.999978,
            ],
        );

        let mut ctx = context();
        let ret = erf_out(&mut ctx, &in_, &out);

        assert!(tensors_are_close(&out, ret, 0.0, Some(0.0)));
        assert!(tensors_are_close(
            &out,
            &expected,
            internal::K_DEFAULT_RTOL,
            None
        ));
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_handle_bool_input() {
        h::test_bool_input(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(erf_out, op_reference);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(erf_out);
    }

    // [spec:et:sem:op-erf.torch.executor.native.erf-out-fn/test]
    #[test]
    fn op_erf_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(erf_out);
    }
}
