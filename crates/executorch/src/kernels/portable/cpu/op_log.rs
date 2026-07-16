//! Literal port of kernels/portable/cpu/op_log.cpp.

// [spec:et:def:op-log.torch.executor.native.log-out-fn]
// [spec:et:sem:op-log.torch.executor.native.log-out-fn]
crate::define_unary_ufunc_realhbbf16_to_floathbf16!(log_out, |x: f32| x.ln(), |x: f64| x.ln());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness as h;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    fn op_reference(x: f64) -> f64 {
        x.ln()
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_handle_bool_input() {
        h::test_bool_input(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_half_output_static_dynamism_support() {
        h::test_all_real_input_half_output_static_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_bfloat16_output_static_dynamism_support() {
        h::test_all_real_input_bfloat16_output_static_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_float_output_static_dynamism_support() {
        h::test_all_real_input_float_output_static_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_double_output_static_dynamism_support() {
        h::test_all_real_input_double_output_static_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_bfloat16_output_bound_dynamism_support() {
        h::test_all_real_input_bfloat16_output_bound_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_float_output_bound_dynamism_support() {
        h::test_all_real_input_float_output_bound_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_double_output_bound_dynamism_support() {
        h::test_all_real_input_double_output_bound_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_float_output_unbound_dynamism_support() {
        h::test_all_real_input_float_output_unbound_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_real_input_double_output_unbound_dynamism_support() {
        h::test_all_real_input_double_output_unbound_dynamism_support(log_out, op_reference);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_all_non_float_output_d_type_dies() {
        h::test_non_float_output_dtype_dies(log_out);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_mismatched_input_shapes_dies() {
        h::test_mismatched_input_shapes_dies(log_out);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_simple_generated_case() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(vec![10, 10], vec![1.0f32; 100]);
        let expected_result = tf.make_default(vec![10, 10], vec![0.0f32; 100]);

        let out = tf.zeros_default(vec![10, 10]);
        let mut ctx = context();
        log_out(&mut ctx, &x, &out);
        assert_tensor_close!(out, expected_result);
    }

    // [spec:et:sem:op-log.torch.executor.native.log-out-fn/test]
    #[test]
    fn op_log_out_test_dynamic_shape_upper_bound_same_as_expected() {
        let tf = TensorFactory::<f32>::new();

        let x = tf.make_default(
            vec![3, 2],
            vec![
                0.6879220604896545,
                0.8289883136749268,
                0.7889447808265686,
                0.6339777112007141,
                0.8719115853309631,
                0.4185197353363037,
            ],
        );
        let expected_result = tf.make_default(
            vec![3, 2],
            vec![
                -0.37407973408699036,
                -0.18754921853542328,
                -0.23705895245075226,
                -0.4557414948940277,
                -0.1370672583580017,
                -0.8710312247276306,
            ],
        );

        let out = tf.zeros(vec![3, 2], TensorShapeDynamism::DYNAMIC_BOUND);
        let mut ctx = context();
        log_out(&mut ctx, &x, &out);
        assert_tensor_close!(out, expected_result);
    }
}
