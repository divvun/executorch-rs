//! Literal port of kernels/quantized/cpu/op_embedding.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: header inline convenience overloads that omit `ctx` construct a
// fresh default-constructed `KernelRuntimeContext ctx;`. Mirrors the established
// `embeddingxb.rs` pattern: null `dyn` fat pointers built from
// `core::ptr::null_mut::<Concrete>() as *mut dyn Trait`.
fn default_context() -> KernelRuntimeContext<'static> {
    KernelRuntimeContext::new(
        crate::extension::module::module::null_event_tracer(),
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
    )
}

// PORT-NOTE: `ET_CHECK_MSG` / `ET_CHECK` are C++ fatal checks; mirrored with a
// local abort on failure (message dropped since a fatal abort follows), matching
// the established pattern in tensor_util.rs / embeddingxb.rs.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}
macro_rules! et_check {
    ($cond:expr) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// PORT-NOTE: `CTYPE_WEIGHT` (Byte/Char) is widened to `float` in the dequant math
// (`static_cast<float>(w_data[j])`). `WeightToF32` reproduces that promotion.
trait WeightToF32: Copy {
    fn to_f32(self) -> f32;
}
impl WeightToF32 for u8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}
impl WeightToF32 for i8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

// PORT-NOTE: `CTYPE_PARAMS` (scale/zero-point element type) is Float/Half/BFloat16;
// the dequant math widens each to `float`. `ParamsF32` reproduces
// `static_cast<float>(scale)` / `static_cast<float>(zp)`. `zp`'s default `0.0` is
// produced via `from_f32(0.0)`.
trait ParamsF32: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}
impl ParamsF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(v: f32) -> Self {
        v
    }
}
impl ParamsF32 for crate::runtime::core::portable_type::Half {
    fn to_f32(self) -> f32 {
        crate::runtime::core::portable_type::Half::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v)
    }
}
impl ParamsF32 for crate::runtime::core::portable_type::BFloat16 {
    fn to_f32(self) -> f32 {
        crate::runtime::core::portable_type::BFloat16::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v)
    }
}

// PORT-NOTE: `CTYPE_OUT` (output element type) is Float/Half/BFloat16; the final
// `static_cast<CTYPE_OUT>(float_result)` narrows the float dequant result.
trait OutFromF32: Copy {
    fn from_f32(v: f32) -> Self;
}
impl OutFromF32 for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
}
impl OutFromF32 for crate::runtime::core::portable_type::Half {
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v)
    }
}
impl OutFromF32 for crate::runtime::core::portable_type::BFloat16 {
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v)
    }
}

/// Asserts that the parameters are valid.
// [spec:et:def:op-embedding.torch.executor.native.check-embedding-byte-args-fn]
// [spec:et:sem:op-embedding.torch.executor.native.check-embedding-byte-args-fn]
#[allow(clippy::too_many_arguments)]
fn check_embedding_byte_args(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &Tensor,
) {
    et_check_msg!(weight.dim() == 2, "weight must be 2D but got() dims");

    et_check_msg!(
        weight_scales.dim() == 1 || weight_scales.dim() == 2,
        "weight_scales must be 1D or 2D but got() dims"
    );

    et_check_msg!(
        weight_scales.size(0) == weight.size(0),
        "Number of scales must be == weight.size(0)"
    );

    if weight_scales.dim() == 2 {
        let num_groups = weight_scales.size(1);
        et_check_msg!(
            weight.size(1) % num_groups == 0,
            "Number of groups must divide weight.size(1)"
        );
    }

    et_check_msg!(
        weight.scalar_type() == ScalarType::Byte || weight.scalar_type() == ScalarType::Char,
        "weight.scalar_type() is not supported:"
    );

    et_check_msg!(
        out.scalar_type() == ScalarType::Float
            || out.scalar_type() == ScalarType::Half
            || out.scalar_type() == ScalarType::BFloat16,
        "out.scalar_type() is not supported:"
    );

    et_check_msg!(
        weight_scales.scalar_type() == ScalarType::Float
            || weight_scales.scalar_type() == ScalarType::Half
            || weight_scales.scalar_type() == ScalarType::BFloat16,
        "weight_scales.scalar_type() is not supported:"
    );

    if let Some(zp) = opt_weight_zero_points {
        et_check_msg!(
            zp.dim() == weight_scales.dim(),
            "weight_zero_points's rank match that of weight_scales."
        );

        et_check_msg!(
            zp.scalar_type() == out.scalar_type(),
            "weight zero points scalar type does not match out.scalar_type()"
        );

        let mut i: i32 = 0;
        while i < weight_scales.dim() as i32 {
            et_check_msg!(
                zp.size(i as isize) == weight_scales.size(i as isize),
                "Dimension size misatch at dim"
            );
            i += 1;
        }
    }

    et_check_msg!(
        indices.scalar_type() == ScalarType::Long,
        "indices.scalar_type() is not Long only Long is supported:"
    );

    et_check_msg!(
        weight_quant_min <= weight_quant_max,
        "weight quant min is greater than weight quant max"
    );

    if let Some(od) = out_dtype {
        et_check_msg!(
            out.scalar_type() == od,
            "output_dtype must match the dtype of the out tensor"
        );
    }
}

/// Retrieves the embeddings specified by indices, dequantizes them, and stores
/// them in out
// [spec:et:def:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]
// [spec:et:sem:op-embedding.torch.executor.native.embedding-byte-per-channel-fn]
#[allow(non_camel_case_types)]
fn embedding_byte_per_channel<CTYPE_WEIGHT, CTYPE_PARAMS, CTYPE_OUT>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    indices: &Tensor,
    out: &Tensor,
) where
    CTYPE_WEIGHT: WeightToF32,
    CTYPE_PARAMS: ParamsF32,
    CTYPE_OUT: OutFromF32,
{
    // An embedding layer nn.Embedding(num_embeddings, embedding_dim) has a
    // weight of shape (num_embeddings, embedding_dim).
    let embedding_dim = weight.size(1);

    let mut num_groups_per_channel: i32 = 1;
    if weight_scales.dim() == 2 {
        num_groups_per_channel = weight_scales.size(1) as i32;
    }
    let group_size: i32 = weight.size(1) as i32 / num_groups_per_channel;

    let mut out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
    let indices_ptr: *const i64 = indices.const_data_ptr::<i64>();

    let scales: *const CTYPE_PARAMS = weight_scales.const_data_ptr::<CTYPE_PARAMS>();
    let mut zero_points: *const CTYPE_PARAMS = core::ptr::null();
    if let Some(zp) = opt_weight_zero_points {
        zero_points = zp.const_data_ptr::<CTYPE_PARAMS>();
    }

    for i in 0..indices.numel() {
        let index: i64 = unsafe { *indices_ptr.offset(i) };

        // Check if index is out of bounds for both weight and weight_scales
        et_check_msg!(
            index >= 0 && index < weight.size(0) as i64,
            "Index out of bounds for weight"
        );

        et_check_msg!(
            index >= 0 && index < weight_scales.size(0) as i64,
            "Index out of bounds for weight_scales"
        );

        // If using groupwise embedding
        let qparams_index: i32 = index as i32 * num_groups_per_channel;
        let mut zp: CTYPE_PARAMS = CTYPE_PARAMS::from_f32(0.0);
        let scale_ptr: *const CTYPE_PARAMS = unsafe { scales.offset(qparams_index as isize) };
        let mut zero_points_ptr: *const CTYPE_PARAMS = core::ptr::null();
        if opt_weight_zero_points.is_some() {
            zero_points_ptr = unsafe { zero_points.offset(qparams_index as isize) };
        }

        let w_data: *const CTYPE_WEIGHT = unsafe {
            weight
                .const_data_ptr::<CTYPE_WEIGHT>()
                .offset(embedding_dim * index as isize)
        };

        for j in 0..embedding_dim as i32 {
            let group_id: i32 = j / group_size;
            let scale: CTYPE_PARAMS = unsafe { *scale_ptr.offset(group_id as isize) };
            if opt_weight_zero_points.is_some() {
                zp = unsafe { *zero_points_ptr.offset(group_id as isize) };
            }
            unsafe {
                *out_data.offset(j as isize) = CTYPE_OUT::from_f32(
                    ((*w_data.offset(j as isize)).to_f32() - zp.to_f32()) * scale.to_f32(),
                );
            }
        }
        out_data = unsafe { out_data.offset(embedding_dim as isize) };
    }
}

// [spec:et:def:op-embedding.torch.executor.native.resize-out-tensor-fn]
// [spec:et:sem:op-embedding.torch.executor.native.resize-out-tensor-fn]
fn resize_out_tensor(weight: &Tensor, indices: &Tensor, out: &Tensor) {
    let mut expected_output_size: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut i: usize = 0;
    while (i as isize) < indices.dim() {
        expected_output_size[i] = indices.size(i as isize) as TensorSizesType;
        i += 1;
    }
    let embedding_dim: usize = weight.size(1) as usize;
    expected_output_size[(out.dim() - 1) as usize] = embedding_dim as TensorSizesType;

    let output_size: ArrayRef<TensorSizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize);

    let err: Error = resize_tensor(out, output_size);
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in quantized_embedding_byte_out"
    );
}

/// Retrieves the embeddings specified by indices, dequantizes them, and stores
/// them in out. The weight is quantized per channel, with a scale and zero_point
/// for each embedding.
///
/// Corresponds as the out variant to torch.ops.quantized.embedding_byte
///
/// NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
/// metadata that is passed around which can be useful for pattern matching.
// [spec:et:def:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn]
// [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_byte_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let w_type: ScalarType = weight.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    resize_out_tensor(weight, indices, out);

    // TODO (jakeszwe): improve these to account for the size of out in relation
    // to weight and indices accounting for a possible batch dimension
    check_embedding_byte_args(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        Some(out_type),
        out,
    );

    let name = "quantized_decomposed::embedding_byte.out";
    crate::et_switch_two_types!(Byte, Char, w_type, ctx, name, CTYPE_W, {
        crate::et_switch_three_types!(Float, Half, BFloat16, out_type, ctx, name, CTYPE_OUT, {
            embedding_byte_per_channel::<CTYPE_W, CTYPE_OUT, CTYPE_OUT>(
                weight,
                weight_scales,
                opt_weight_zero_points,
                indices,
                out,
            );
        });
    });

    out
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_byte_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let mut context = default_context();
    let res = quantized_embedding_byte_out(
        &mut context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

// [spec:et:def:op-embedding.torch.executor.native.quantized-embedding-byte-dtype-out-fn]
// [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-dtype-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_byte_dtype_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    resize_out_tensor(weight, indices, out);

    // TODO (jakeszwe): improve these to account for the size of out in relation
    // to weight and indices accounting for a possible batch dimension
    check_embedding_byte_args(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
    );

    let weight_type: ScalarType = weight.scalar_type();
    let params_type: ScalarType = weight_scales.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    let name = "quantized_decomposed::embedding_byte.dtype_out";
    crate::et_switch_two_types!(Byte, Char, weight_type, ctx, name, CTYPE_W, {
        crate::et_switch_three_types!(Float, Half, BFloat16, params_type, ctx, name, CTYPE_P, {
            crate::et_switch_three_types!(Float, Half, BFloat16, out_type, ctx, name, CTYPE_OUT, {
                embedding_byte_per_channel::<CTYPE_W, CTYPE_P, CTYPE_OUT>(
                    weight,
                    weight_scales,
                    opt_weight_zero_points,
                    indices,
                    out,
                );
            });
        });
    });

    out
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_byte_dtype_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let mut context = default_context();
    let res = quantized_embedding_byte_dtype_out(
        &mut context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::op_embedding::embedding_out;
    use crate::kernels::quantized::cpu::op_dequantize::dequantize_per_tensor_out;
    use crate::kernels::quantized::cpu::op_quantize::quantize_per_tensor_out;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::testing_util::tensor_util::tensors_are_close;
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::portable_type::{BFloat16, Half};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

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
    macro_rules! assert_tensor_close_with_tol {
        ($t1:expr, $t2:expr, $rtol:expr, $atol:expr) => {
            assert!(
                tensors_are_close(&$t1, &$t2, $rtol, Some($atol)),
                "tensors are not close within tolerance"
            )
        };
    }

    // Reduced-precision element constructor from a double.
    trait FromF64Ctor: Copy {
        fn ctor(v: f64) -> Self;
    }
    impl FromF64Ctor for Half {
        fn ctor(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64Ctor for BFloat16 {
        fn ctor(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    /// A generic smoke test that works for any dtype that supports ones() and
    /// zeros(). Only Byte is instantiated by AllDtypesSupported.
    //
    // quantized_embedding_byte_out runs resize_out_tensor, check_embedding_byte_args,
    // and the embedding_byte_per_channel dequant loop; the exact expected output
    // values pin all three.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    // [spec:et:sem:op-embedding.torch.executor.native.check-embedding-byte-args-fn/test]
    // [spec:et:sem:op-embedding.torch.executor.native.embedding-byte-per-channel-fn/test]
    // [spec:et:sem:op-embedding.torch.executor.native.resize-out-tensor-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    fn test_dtype() {
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let scale: f64 = 0.5;
        let zero_point: f32 = 1.0;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight = tf.make_default(vec![3, 2], vec![3.5, 2.0, 4.0, 1.0, 5.5, 13.2]);
        let weight_scales = tf.full(vec![3], scale as f32, STATIC);
        let weight_zero_points = tf.full(vec![3], zero_point, STATIC);

        let indices = tf_l.make_default(vec![2], vec![0, 2]);

        let out = tf.zeros(vec![2, 2], STATIC);

        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.zeros(vec![3, 2], STATIC);

        // 3.5 / 0.5 + 1 = 8, etc.
        quantize_per_tensor_out(
            &weight,
            scale,
            zero_point as i64,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qweight,
        );

        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        let expected = tf.make_default(vec![2, 2], vec![3.5, 2.0, 5.5, 13.0]);

        assert_tensor_eq!(out, expected);
    }

    #[test]
    fn op_quantized_embedding_test_all_dtypes_supported() {
        test_dtype();
    }

    // Q -> DQ -> FP Embedding should be == to Q -> QEmbedding Bytes
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    fn op_quantized_embedding_test_consitency_with_reference_pattern() {
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let scale: f64 = 0.5;
        let zero_point: f32 = 1.0;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        // Do Q -> QEmbedding Bytes
        let weight = tf.make_default(vec![3, 1], vec![3.5, 5.5, 1.0]);
        let weight_scales = tf.full(vec![3], scale as f32, STATIC);
        let weight_zero_points = tf.full(vec![3], zero_point, STATIC);

        let indices = tf_l.make_default(vec![2], vec![0, 2]);

        let out = tf.zeros(vec![2, 1], STATIC);
        let fp_out = tf.zeros(vec![2, 1], STATIC);

        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.zeros(vec![3, 1], STATIC);
        let mut context = default_context();
        // 3.5 / 0.5 + 1 = 8, 5.5 / 0.5 + 1 = 12, 1 / 0.5 + 1 = 3
        quantize_per_tensor_out(
            &weight,
            scale,
            zero_point as i64,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &qweight,
        );

        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        // Do Q DQ embedding
        dequantize_per_tensor_out(
            &qweight,
            scale,
            zero_point as i64,
            quant_min,
            quant_max,
            ScalarType::Byte,
            None,
            &weight,
        );

        embedding_out(
            &mut context,
            &weight,
            &indices,
            /*padding_idx=*/ 0,
            /*scale_grad_by_freq=*/ false,
            /*sparse=*/ false,
            &fp_out,
        );

        // (8 - 1) * 0.5 = 3.5, (3 - 1) * 0.5 = 1
        let expected = tf.make_default(vec![2, 1], vec![3.5, 1.0]);
        assert_tensor_eq!(out, fp_out);
        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let mut weight_scales = tf.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let mut weight_zero_points = tf.make_default(vec![3], vec![1.0, 5.0, 7.0]);
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(
            vec![3, 4],
            vec![8, 10, 12, 14, 10, 12, 12, 14, 8, 9, 10, 12],
        );

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let mut out = tf.zeros(vec![3, 4], STATIC);
        let mut expected = tf.make_default(
            vec![3, 4],
            vec![3.5, 4.5, 5.5, 6.5, 1.5, 3.0, 4.5, 7.5, 5.0, 7.0, 7.0, 9.0],
        );

        quantized_embedding_byte_out_nocontext(
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
        weight_zero_points = tf.make_default(vec![3, 2], vec![1.0, 5.0, 7.0, 9.0, 11.0, 13.0]);

        out = tf.zeros(vec![3, 4], STATIC);
        expected = tf.make_default(
            vec![3, 4],
            vec![
                3.5, 4.5, 7.0, 9.0, -7.5, -5.0, -9.0, -3.0, 4.5, 7.5, 6.0, 10.0,
            ],
        );

        quantized_embedding_byte_out_nocontext(
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

    // PORT-NOTE: `ET_EXPECT_DEATH` — the group-wise arg mismatch is caught by
    // `check_embedding_byte_args` -> `et_check_msg!` -> `runtime_abort()` ->
    // `libc::abort()`, which terminates the process rather than unwinding, so
    // `#[should_panic]` cannot catch it and running it would kill the whole test
    // binary. Ported and `#[ignore]`d.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding_death1() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.make_default(vec![4], vec![0.5, 1.0, 1.5, 3.3]);
        let weight_zero_points = tf.make_default(vec![4], vec![1.0, 5.0, 7.0, 5.0]);
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(
            vec![3, 4],
            vec![8, 10, 12, 14, 10, 12, 12, 14, 8, 9, 10, 12],
        );

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 4], STATIC);
        quantized_embedding_byte_out_nocontext(
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
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding_death2() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.make_default(vec![2], vec![0.5, 1.0]);
        let weight_zero_points = tf.make_default(vec![2], vec![1.0, 5.0]);
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(
            vec![3, 4],
            vec![8, 10, 12, 14, 10, 12, 12, 14, 8, 9, 10, 12],
        );

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 4], STATIC);
        quantized_embedding_byte_out_nocontext(
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
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding_death3() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.make_default(vec![3, 2], vec![0.5, 1.0, 1.5, 2.5, 3.5, 3.5]);
        let weight_zero_points = tf.make_default(vec![3, 2], vec![1.0, 5.0, 7.0, 9.0, 11.0, 13.0]);
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(vec![3, 3], vec![8, 10, 12, 14, 10, 12, 12, 14, 8]);

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 3], STATIC);
        quantized_embedding_byte_out_nocontext(
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
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding_death4() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.make_default(vec![3, 2], vec![0.5, 1.0, 1.5, 2.5, 3.5, 3.5]);
        let weight_zero_points = tf.make_default(vec![3], vec![1.0, 5.0, 7.0]);
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(vec![3, 3], vec![8, 10, 12, 14, 10, 12, 12, 14, 8]);

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 3], STATIC);
        quantized_embedding_byte_out_nocontext(
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
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_group_wise_quantized_embedding_death5() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.make_default(vec![3, 2], vec![0.5, 1.0, 1.5, 2.5, 3.5, 3.5]);
        let weight_zero_points = tf.make_default(
            vec![3, 3],
            vec![1.0, 5.0, 7.0, 1.0, 5.0, 7.0, 1.0, 5.0, 7.0],
        );
        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(vec![3, 3], vec![8, 10, 12, 14, 10, 12, 12, 14, 8]);

        let indices = tf_l.make_default(vec![3], vec![0, 2, 1]);

        let out = tf.zeros(vec![3, 3], STATIC);
        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }

    // PORT-NOTE: abort-based death test (out-of-bounds index); see death1.
    // `#[ignore]`d.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    #[ignore]
    #[should_panic]
    fn op_quantized_embedding_test_test_out_of_bounds_index() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<f32>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let qweight = tfo.make_default(
            vec![3, 4],
            vec![8, 10, 12, 14, 10, 12, 12, 14, 8, 9, 10, 12],
        );

        let weight_scales = tf.make_default(vec![3, 1], vec![0.5, 1.0, 1.5]);
        let weight_zero_points = tf.make_default(vec![3, 1], vec![1.0, 5.0, 7.0]);

        // out-of-bounds index (3, which is >= weight.size(0))
        let indices = tf_l.make_default(vec![2], vec![1, 3]);

        let out = tf.zeros(vec![2, 4], STATIC);

        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );
    }

    // Runs embedding_byte.out with scales, zero points, and output all in the
    // given reduced-precision dtype.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    fn test_reduced_precision_out<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Ctor,
    {
        let tf = TensorFactory::<T>::new();
        let tfb = TensorFactory::<u8>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.full(vec![3], T::ctor(0.5), STATIC);
        let weight_zero_points = tf.full(vec![3], T::ctor(1.0), STATIC);
        // (q - 1) * 0.5
        let qweight = tfb.make_default(vec![3, 2], vec![8, 5, 9, 3, 12, 27]);
        let indices = tf_l.make_default(vec![2], vec![0, 2]);

        let out = tf.zeros(vec![2, 2], STATIC);
        let expected = tf.make_default(
            vec![2, 2],
            vec![T::ctor(3.5), T::ctor(2.0), T::ctor(5.5), T::ctor(13.0)],
        );

        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_close!(out, expected);
    }

    #[test]
    fn op_quantized_embedding_test_reduced_precision_out() {
        crate::runtime::platform::runtime::runtime_init();
        test_reduced_precision_out::<Half>();
        test_reduced_precision_out::<BFloat16>();
    }

    // embedding_byte.dtype_out with scales and output both bf16.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-dtype-out-fn/test]
    #[test]
    fn op_quantized_embedding_test_b_float16_dtype_out() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<BFloat16>::new();
        let tfb = TensorFactory::<u8>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.full(vec![3], BFloat16::from_f32(0.5), STATIC);
        let weight_zero_points = tf.full(vec![3], BFloat16::from_f32(1.0), STATIC);
        let qweight = tfb.make_default(vec![3, 2], vec![8, 5, 9, 3, 12, 27]);
        let indices = tf_l.make_default(vec![2], vec![0, 2]);

        let out = tf.zeros(vec![2, 2], STATIC);
        let expected = tf.make_default(
            vec![2, 2],
            vec![
                BFloat16::from_f32(3.5),
                BFloat16::from_f32(2.0),
                BFloat16::from_f32(5.5),
                BFloat16::from_f32(13.0),
            ],
        );

        quantized_embedding_byte_dtype_out_nocontext(
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

    // bf16 output for scales not exactly representable, verifying the dequant math
    // is done in fp32 and only the store is rounded to bf16.
    // [spec:et:sem:op-embedding.torch.executor.native.quantized-embedding-byte-out-fn/test]
    #[test]
    fn op_quantized_embedding_test_b_float16_rounding() {
        crate::runtime::platform::runtime::runtime_init();
        let tf = TensorFactory::<BFloat16>::new();
        let tfb = TensorFactory::<u8>::new();
        let tf_l = TensorFactory::<i64>::new();

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let weight_scales = tf.full(vec![3], BFloat16::from_f32(0.1), STATIC);
        let weight_zero_points = tf.full(vec![3], BFloat16::from_f32(0.0), STATIC);
        let qweight = tfb.make_default(vec![3, 2], vec![8, 5, 9, 3, 12, 27]);
        let indices = tf_l.make_default(vec![2], vec![0, 2]);

        let out = tf.zeros(vec![2, 2], STATIC);
        // scale (0.1) rounds to bf16 before the multiply, so reference the bf16
        // scale rather than the exact decimal.
        let expected = tf.make_default(
            vec![2, 2],
            vec![
                BFloat16::from_f32(0.8),
                BFloat16::from_f32(0.5),
                BFloat16::from_f32(1.2),
                BFloat16::from_f32(2.7),
            ],
        );

        quantized_embedding_byte_out_nocontext(
            &qweight,
            &weight_scales,
            &Some(&weight_zero_points),
            quant_min,
            quant_max,
            &indices,
            &out,
        );

        assert_tensor_close_with_tol!(out, expected, 1e-2, 1e-2);
    }
}
