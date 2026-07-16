//! Literal port of kernels/quantized/cpu/op_embedding2b.cpp.

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
/// Corresponds as the out variant to torch.ops.quantized.embedding_2bit
///
/// NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
/// metadata that is passed around which can be useful for pattern matching.
// [spec:et:def:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn]
// [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_2bit_out_nocontext<'a, 'b>(
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
        2,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_2bit_out<'a, 'b>(
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
        2,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_2bit_dtype_out_nocontext<'a, 'b>(
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
        2,
    )
}

// [spec:et:def:op-embedding2b.torch.executor.native.quantized-embedding-2bit-dtype-out-fn]
// [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-dtype-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_2bit_dtype_out<'a, 'b>(
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
        2,
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

    // Exercises the whole embeddingxb.cpp pipeline reached through
    // quantized_embedding_2bit_out: xbit out, arg-checking, out-tensor resize,
    // embedding-dim math, the per-channel dequant loop, and the 2-bit weight
    // unpacking.
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.resize-out-tensor-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.weight-value-fn/test]
    #[test]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let mut weight_scales = tf.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let mut weight_zero_points = tf.make_default(vec![3], vec![1.0, -2.0, 0.0]);

        let mut qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);

        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let mut expected = tf.make_default(
            vec![3, 4],
            vec![
                -1.5, 0.0, -0.5, 0.0, -3.0, -1.5, 0.0, 1.5, 2.0, 1.0, 0.0, 2.0,
            ],
        );

        quantized_embedding_2bit_out_nocontext(
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
        quantized_embedding_2bit_out(
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
        weight_zero_points = tf.make_default(vec![3, 2], vec![1.0, -2.0, 0.0, 1.0, -2.0, -1.0]);

        qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);

        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        out = tf.zeros(vec![3, 4], STATIC);
        expected = tf.make_default(
            vec![3, 4],
            vec![
                -1.5, 0.0, 2.0, 3.0, 0.0, 2.5, 3.0, 6.0, 0.0, -1.5, -6.0, -2.0,
            ],
        );

        quantized_embedding_2bit_out_nocontext(
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

    // The dtype_out variant path routes through quantized_embedding_xbit_dtype_out.
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-dtype-out-fn/test]
    // [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn/test]
    #[test]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding_b_float16() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<BFloat16>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let weight_scales = tf.make_default(
            vec![3],
            vec![
                BFloat16::from_f32(0.5),
                BFloat16::from_f32(1.0),
                BFloat16::from_f32(1.5),
            ],
        );
        let weight_zero_points = tf.make_default(
            vec![3],
            vec![
                BFloat16::from_f32(1.0),
                BFloat16::from_f32(-2.0),
                BFloat16::from_f32(0.0),
            ],
        );
        let qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let bf = |v: f32| BFloat16::from_f32(v);
        let expected = tf.make_default(
            vec![3, 4],
            vec![
                bf(-1.5),
                bf(0.0),
                bf(-0.5),
                bf(0.0),
                bf(-3.0),
                bf(-1.5),
                bf(0.0),
                bf(1.5),
                bf(2.0),
                bf(1.0),
                bf(0.0),
                bf(2.0),
            ],
        );

        quantized_embedding_2bit_out_nocontext(
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
        quantized_embedding_2bit_dtype_out_nocontext(
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

    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    #[test]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding_int32_indices() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfi = TensorFactory::<i32>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let weight_scales = tf.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let weight_zero_points = tf.make_default(vec![3], vec![1.0, -2.0, 0.0]);

        let qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);

        let indices = tfi.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let expected = tf.make_default(
            vec![3, 4],
            vec![
                -1.5, 0.0, -0.5, 0.0, -3.0, -1.5, 0.0, 1.5, 2.0, 1.0, 0.0, 2.0,
            ],
        );

        quantized_embedding_2bit_out_nocontext(
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
        quantized_embedding_2bit_out(
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
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding_death1() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let weight_scales = tf.make_default(vec![4], vec![0.5, 1.0, 1.5, 3.3]);
        let weight_zero_points = tf.make_default(vec![4], vec![1.0, -2.0, 1.0, 0.0]);
        let qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);
        let out = tf.zeros(vec![3, 4], STATIC);

        // qvals are incompatible shape with scales/zeros
        quantized_embedding_2bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }

    // PORT-NOTE: abort-based death test; see death1. `#[ignore]`d.
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding_death2() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let weight_scales = tf.make_default(vec![2], vec![0.5, 1.0]);
        let weight_zero_points = tf.make_default(vec![2], vec![1.0, -2.0]);
        let qweight = tfb.make_default(vec![3, 1], vec![236, 134, 228]);
        let indices = tfl.make_default(vec![3], vec![0, 2, 1]);
        let out = tf.zeros(vec![3, 4], STATIC);

        // qvals are incompatible shape with scales/zeros
        quantized_embedding_2bit_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }

    // PORT-NOTE: abort-based death test; see death1. `#[ignore]`d.
    // [spec:et:sem:op-embedding2b.torch.executor.native.quantized-embedding-2bit-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding2b_test_test_group_wise_quantized_embedding_death3() {
        crate::runtime::platform::runtime::runtime_init();
        let tfb = TensorFactory::<u8>::new();
        let tf = TensorFactory::<f32>::new();
        let tfl = TensorFactory::<i64>::new();

        let quant_min: i64 = -2;
        let quant_max: i64 = 1;

        let weight_scales = tf.make_default(vec![2, 3], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let weight_zero_points = tf.make_default(vec![2, 3], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let qweight = tfb.make_default(vec![2, 1], vec![236, 134]);
        let indices = tfl.make_default(vec![2], vec![0, 2]);
        let out = tf.zeros(vec![2, 8], STATIC);

        // scales/zeros imply 3 groups, which does not divide embed dimension from
        // qvals (8)
        quantized_embedding_2bit_out_nocontext(
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
