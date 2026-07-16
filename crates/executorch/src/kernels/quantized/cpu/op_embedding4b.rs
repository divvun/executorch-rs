//! Literal port of kernels/quantized/cpu/op_embedding4b.cpp.

use crate::kernels::quantized::cpu::embeddingxb::{
    quantized_embedding_xbit_dtype_out, quantized_embedding_xbit_dtype_out_nocontext,
    quantized_embedding_xbit_out, quantized_embedding_xbit_out_nocontext,
};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

/// Retrieves the embeddings specified by indices, dequantizes them, and stores
/// them in out. The weight is quantized per channel, with a scale and zero_point
/// for each embedding.
///
/// Corresponds as the out variant to torch.ops.quantized.embedding_4bit
///
/// NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
/// metadata that is passed around which can be useful for pattern matching.
// [spec:et:def:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn]
// [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_4bit_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    quantized_embedding_xbit_out_nocontext(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out,
        4,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_4bit_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    quantized_embedding_xbit_out(
        context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out,
        4,
    )
}

// [spec:et:def:op-embedding4b.torch.executor.native.quantized-embedding-4bit-dtype-out-fn]
// [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-dtype-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_4bit_dtype_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    quantized_embedding_xbit_dtype_out_nocontext(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
        4,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_4bit_dtype_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    quantized_embedding_xbit_dtype_out(
        context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
        4,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::BFloat16;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::runtime::runtime_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    macro_rules! assert_tensor_eq {
        ($t1:expr, $t2:expr) => {
            assert!(
                tensors_are_close(&$t1, &$t2, 0.0, Some(0.0)),
                "tensors are not equal"
            )
        };
    }
    macro_rules! assert_tensor_close {
        ($t1:expr, $t2:expr) => {
            assert!(
                tensors_are_close(
                    &$t1,
                    &$t2,
                    crate::runtime::core::exec_aten::testing_util::tensor_util::internal::K_DEFAULT_RTOL,
                    None
                ),
                "tensors are not close"
            )
        };
    }

    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn/test]
    #[test]
    fn op_quantized_embedding4b_test_test_group_wise_quantized_embedding() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -8;
        let quant_max: i64 = 7;

        let mut weight_scales = tf.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let mut weight_zero_points = tf.make_default(vec![3], vec![1.0, -5.0, 0.0]);

        let qweight = tfb.make_default(vec![3, 2], vec![89, 239, 163, 72, 11, 126]);

        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let mut expected = tf.make_default(
            vec![3, 4],
            vec![
                -2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0,
            ],
        );

        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_eq!(out, expected);

        out = tf.zeros(vec![3, 4], STATIC);
        let mut ctx = context();
        quantized_embedding_4bit_out(
            &mut ctx,
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_eq!(out, expected);

        // Groupwise quantization. groupsize = 2
        weight_scales = tf.make_default(vec![3, 2], vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        weight_zero_points = tf.make_default(vec![3, 2], vec![1.0, -5.0, 0.0, 2.0, -3.0, -1.0]);

        out = tf.zeros(vec![3, 4], STATIC);
        expected = tf.make_default(
            vec![3, 4],
            vec![
                -2.0, 0.0, 11.0, 12.0, -12.5, 15.0, 0.0, 21.0, 3.0, -7.5, -12.0, -4.0,
            ],
        );

        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn/test]
    #[test]
    fn op_quantized_embedding4b_test_test_group_wise_quantized_embedding_int32_indices() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfi = TensorFactory::<i32>::new();

        let quant_min: i64 = -8;
        let quant_max: i64 = 7;

        let weight_scales = tf.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let weight_zero_points = tf.make_default(vec![3], vec![1.0, -5.0, 0.0]);

        let qweight = tfb.make_default(vec![3, 2], vec![89, 239, 163, 72, 11, 126]);

        let indices = tfi.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let expected = tf.make_default(
            vec![3, 4],
            vec![
                -2.0, 0.0, 2.5, 3.0, -12.0, 4.5, -1.5, 9.0, 7.0, 0.0, 1.0, 5.0,
            ],
        );

        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_eq!(out, expected);

        out = tf.zeros(vec![3, 4], STATIC);
        let mut ctx = context();
        quantized_embedding_4bit_out(
            &mut ctx,
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // PORT-NOTE: `ET_EXPECT_DEATH` — arg-shape mismatch is caught by
    // `check_embedding_xbit_args` -> `et_check_msg!` -> `runtime_abort()` ->
    // `libc::abort()`, which terminates the process rather than unwinding, so
    // `#[should_panic]` cannot catch it. Ported and `#[ignore]`d.
    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding4b_test_test_group_wise_quantized_embedding_death1() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -8;
        let quant_max: i64 = 7;

        let weight_scales = tf.make_default(vec![4], vec![0.5, 1.0, 1.5, 3.3]);
        let weight_zero_points = tf.make_default(vec![4], vec![1.0, 5.0, 7.0, 5.0]);
        let qweight = tfb.make_default(vec![3, 2], vec![89, 239, 163, 72, 11, 126]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 4], STATIC);
        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }

    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn/test]
    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-dtype-out-fn/test]
    #[test]
    fn op_quantized_embedding4b_test_test_group_wise_quantized_embedding_b_float16() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<BFloat16>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -8;
        let quant_max: i64 = 7;

        let bf = |v: f32| BFloat16::from_f32(v);
        let weight_scales = tf.make_default(vec![3], vec![bf(0.5), bf(1.0), bf(1.5)]);
        let weight_zero_points = tf.make_default(vec![3], vec![bf(1.0), bf(-5.0), bf(0.0)]);
        let qweight = tfb.make_default(vec![3, 2], vec![89, 239, 163, 72, 11, 126]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let expected = tf.make_default(
            vec![3, 4],
            vec![
                bf(-2.0),
                bf(0.0),
                bf(2.5),
                bf(3.0),
                bf(-12.0),
                bf(4.5),
                bf(-1.5),
                bf(9.0),
                bf(7.0),
                bf(0.0),
                bf(1.0),
                bf(5.0),
            ],
        );

        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_close!(out, expected);

        // Same values through the dtype_out variant.
        out = tf.zeros(vec![3, 4], STATIC);
        quantized_embedding_4bit_dtype_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            Some(ScalarType::BFloat16),
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    // PORT-NOTE: abort-based death test; see death1. `#[ignore]`d.
    // [spec:et:sem:op-embedding4b.torch.executor.native.quantized-embedding-4bit-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding4b_test_test_group_wise_quantized_embedding_death2() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -8;
        let quant_max: i64 = 7;

        let weight_scales = tf.make_default(vec![2], vec![0.5, 1.0]);
        let weight_zero_points = tf.make_default(vec![2], vec![1.0, 5.0]);
        let qweight = tfb.make_default(vec![3, 2], vec![89, 239, 163, 72, 11, 126]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 4], STATIC);
        quantized_embedding_4bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }
}
