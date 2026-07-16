//! Literal port of kernels/quantized/cpu/embeddingxb.cpp.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::tensor_util::{K_TENSOR_DIMENSION_LIMIT, resize_tensor};
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::{Tensor, TensorSizesType};
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// PORT-NOTE: header inline convenience overloads that omit `ctx` construct a
// fresh default-constructed `KernelRuntimeContext ctx;`. The ported
// `KernelRuntimeContext::new` requires two raw `dyn` pointers (no default
// arguments in Rust); null `dyn` fat pointers are built mirroring the
// established `core::ptr::null_mut::<Concrete>() as *mut dyn Trait` pattern.
fn default_context() -> KernelRuntimeContext<'static> {
    KernelRuntimeContext::new(
        crate::extension::module::module::null_event_tracer(),
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
    )
}

// PORT-NOTE: `ET_CHECK_MSG` / `ET_CHECK` are C++ fatal checks; mirrored with a
// local abort on failure (message dropped since a fatal abort follows),
// matching the established pattern in tensor_util.rs / op_embedding.rs.
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

// PORT-NOTE: `CTYPE_PARAMS` (scale/zero-point element type) is
// Float/Half/BFloat16; the dequant math widens each to `float`. `ToF32`
// reproduces `static_cast<float>(scale)` / `static_cast<float>(zp)`. `zp`'s
// default `0.0` is produced via `FromF32::from_f32(0.0)`.
trait ToF32: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}
impl ToF32 for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(v: f32) -> Self {
        v
    }
}
impl ToF32 for crate::runtime::core::portable_type::Half {
    fn to_f32(self) -> f32 {
        crate::runtime::core::portable_type::Half::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v)
    }
}
impl ToF32 for crate::runtime::core::portable_type::BFloat16 {
    fn to_f32(self) -> f32 {
        crate::runtime::core::portable_type::BFloat16::to_f32(self)
    }
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v)
    }
}

// PORT-NOTE: `CTYPE_OUT` (output element type) is Float/Half/BFloat16; the final
// `static_cast<CTYPE_OUT>(float_result)` narrows the float dequant result.
trait FromF32Out: Copy {
    fn from_f32(v: f32) -> Self;
}
impl FromF32Out for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
}
impl FromF32Out for crate::runtime::core::portable_type::Half {
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::Half::from_f32(v)
    }
}
impl FromF32Out for crate::runtime::core::portable_type::BFloat16 {
    fn from_f32(v: f32) -> Self {
        crate::runtime::core::portable_type::BFloat16::from_f32(v)
    }
}

// PORT-NOTE: `CTYPE_INDICES` is Int (i32) or Long (i64); indices participate in
// `index * num_groups_per_channel` and pointer arithmetic
// (`weight.size(1) * index`), promoting to the wider integer / pointer-index
// type. `AsI64` reproduces that widening.
trait AsI64: Copy {
    fn as_i64(self) -> i64;
}
impl AsI64 for i32 {
    fn as_i64(self) -> i64 {
        self as i64
    }
}
impl AsI64 for i64 {
    fn as_i64(self) -> i64 {
        self
    }
}

// [spec:et:def:embeddingxb.torch.executor.native.weight-value-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.weight-value-fn]
///
/// # Safety
/// `w_data` must point at the packed bytes of the weight row being unpacked, and
/// `index >> (2 or 1)` must be within that row.
unsafe fn weight_value(w_data: *const u8, index: i32, weight_nbit: i32) -> i32 {
    if weight_nbit == 2 {
        let subbyte: i32 = index % 4;
        let index = index >> 2;
        match subbyte {
            0 => return (unsafe { *w_data.offset(index as isize) } & 3) as i32 - 2,
            1 => return ((unsafe { *w_data.offset(index as isize) } & 12) >> 2) as i32 - 2,
            2 => return ((unsafe { *w_data.offset(index as isize) } & 48) >> 4) as i32 - 2,
            3 => return ((unsafe { *w_data.offset(index as isize) } & 192) >> 6) as i32 - 2,
            _ => {}
        }
    } else if weight_nbit == 4 {
        let odd: i32 = index & 1;
        let index = index >> 1;
        if odd != 0 {
            return (unsafe { *w_data.offset(index as isize) } & 0x0F) as i32 - 8;
        } else {
            return ((unsafe { *w_data.offset(index as isize) } >> 4) & 0x0F) as i32 - 8;
        }
    }

    et_check_msg!(false, "invalid weight_nbit");
    // PORT-NOTE: the C++ `ET_CHECK_MSG(false, ...)` never returns (aborts); the
    // switch on `subbyte` (2-bit case) covers 0..=3 so the fallthrough after the
    // match cannot be reached for valid input. This unreachable value only
    // satisfies the return type after the (diverging) abort above.
    #[allow(unreachable_code)]
    0
}

// [spec:et:def:embeddingxb.torch.executor.native.get-embedding-dim-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.get-embedding-dim-fn]
fn get_embedding_dim(packed_dim: i32, weight_nbit: i32) -> i32 {
    et_check_msg!(8 % weight_nbit == 0, "invalid embedding dim");
    let packed_values_per_byte: i32 = 8 / weight_nbit;
    packed_dim * packed_values_per_byte
}

/// Asserts that the parameters are valid.
// [spec:et:def:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.check-embedding-xbit-args-fn]
#[allow(clippy::too_many_arguments)]
fn check_embedding_xbit_args(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &Tensor,
    weight_nbit: i32,
) {
    et_check_msg!(8 % weight_nbit == 0, "nbit must divide 8");

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
            // each 8b uint8 column is packed_values_per_byte columns
            get_embedding_dim(weight.size(1) as i32, weight_nbit) as i64 % num_groups as i64 == 0,
            "Number of groups must divide weight.size(1)"
        );
    }

    et_check_msg!(
        weight.scalar_type() == ScalarType::Byte,
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
        indices.scalar_type() == ScalarType::Long || indices.scalar_type() == ScalarType::Int,
        "indices.scalar_type() is not Long or Int"
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
/// them in out. Weight will always be uint8
// [spec:et:def:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.embedding-xbit-per-channel-fn]
#[allow(non_camel_case_types)]
fn embedding_xbit_per_channel<CTYPE_PARAMS, CTYPE_OUT, CTYPE_INDICES>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    indices: &Tensor,
    out: &Tensor,
    weight_nbit: i32,
) where
    CTYPE_PARAMS: ToF32,
    CTYPE_OUT: FromF32Out,
    CTYPE_INDICES: AsI64,
{
    let embedding_dim: i32 = get_embedding_dim(weight.size(1) as i32, weight_nbit);

    let mut num_groups_per_channel: i32 = 1;
    if weight_scales.dim() == 2 {
        num_groups_per_channel = weight_scales.size(1) as i32;
    }
    let group_size: i32 = embedding_dim / num_groups_per_channel;

    let mut out_data: *mut CTYPE_OUT = out.mutable_data_ptr::<CTYPE_OUT>();
    let indices_ptr: *const CTYPE_INDICES = indices.const_data_ptr::<CTYPE_INDICES>();

    let scales: *const CTYPE_PARAMS = weight_scales.const_data_ptr::<CTYPE_PARAMS>();
    let mut zero_points: *const CTYPE_PARAMS = core::ptr::null();
    if let Some(zp) = opt_weight_zero_points {
        zero_points = zp.const_data_ptr::<CTYPE_PARAMS>();
    }

    for i in 0..indices.numel() {
        let index: CTYPE_INDICES = unsafe { *indices_ptr.offset(i) };
        // If using groupwise embedding
        let qparams_index: i32 = index.as_i64() as i32 * num_groups_per_channel;
        let mut zp: CTYPE_PARAMS = CTYPE_PARAMS::from_f32(0.0);
        let scale_ptr: *const CTYPE_PARAMS = unsafe { scales.offset(qparams_index as isize) };
        let mut zero_points_ptr: *const CTYPE_PARAMS = core::ptr::null();
        if opt_weight_zero_points.is_some() {
            zero_points_ptr = unsafe { zero_points.offset(qparams_index as isize) };
        }

        let w_data: *const u8 = unsafe {
            weight
                .const_data_ptr::<u8>()
                .offset((weight.size(1) as i64 * index.as_i64()) as isize)
        };

        for j in 0..embedding_dim {
            let group_id: i32 = j / group_size;
            let scale: CTYPE_PARAMS = unsafe { *scale_ptr.offset(group_id as isize) };
            if opt_weight_zero_points.is_some() {
                zp = unsafe { *zero_points_ptr.offset(group_id as isize) };
            }
            unsafe {
                *out_data.offset(j as isize) = CTYPE_OUT::from_f32(
                    (weight_value(w_data, j, weight_nbit) as f32 - zp.to_f32()) * scale.to_f32(),
                );
            }
        }
        out_data = unsafe { out_data.offset(embedding_dim as isize) };
    }
}

// [spec:et:def:embeddingxb.torch.executor.native.resize-out-tensor-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.resize-out-tensor-fn]
fn resize_out_tensor(weight: &Tensor, indices: &Tensor, out: &Tensor, weight_nbit: i32) {
    let mut expected_output_size: [TensorSizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let mut i: usize = 0;
    while (i as isize) < indices.dim() {
        expected_output_size[i] = indices.size(i as isize) as TensorSizesType;
        i += 1;
    }
    let embedding_dim: usize = get_embedding_dim(weight.size(1) as i32, weight_nbit) as usize;
    expected_output_size[(out.dim() - 1) as usize] = embedding_dim as TensorSizesType;

    let output_size: ArrayRef<TensorSizesType> =
        ArrayRef::from_raw_parts(expected_output_size.as_ptr(), out.dim() as usize);

    let err: Error = resize_tensor(out, output_size);
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in quantized_embedding_xbit_out"
    );
}

/// Retrieves the embeddings specified by indices, dequantizes them, and stores
/// them in out. The weight is quantized per channel, with a scale and zero_point
/// for each embedding.
///
/// Corresponds as the out variant to torch.ops.quantized.embedding_xbit
///
/// NOTE: quant_min, quant_max, and Dtype are not used in computation, but rather
/// metadata that is passed around which can be useful for pattern matching.
// [spec:et:def:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_xbit_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
    weight_nbit: i32,
) -> &'a Tensor<'b> {
    let out_type: ScalarType = out.scalar_type();

    resize_out_tensor(weight, indices, out, weight_nbit);

    // TODO (jakeszwe): improve these to account for the size of out in relation
    // to weight and indices accounting for a possible batch dimension
    check_embedding_xbit_args(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        Some(out_type),
        out,
        weight_nbit,
    );

    let name = "quantized_decomposed::embedding_xbit.out";
    let indices_type: ScalarType = indices.scalar_type();
    crate::et_switch_three_types!(Float, Half, BFloat16, out_type, ctx, name, CTYPE_OUT, {
        crate::et_switch_two_types!(Int, Long, indices_type, ctx, name, CTYPE_IDX, {
            embedding_xbit_per_channel::<CTYPE_OUT, CTYPE_OUT, CTYPE_IDX>(
                weight,
                weight_scales,
                opt_weight_zero_points,
                indices,
                out,
                weight_nbit,
            );
        });
    });

    out
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_xbit_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out: &'a Tensor<'b>,
    weight_nbit: i32,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let mut context = default_context();
    let res = quantized_embedding_xbit_out(
        &mut context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out,
        weight_nbit,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

// [spec:et:def:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]
// [spec:et:sem:embeddingxb.torch.executor.native.quantized-embedding-xbit-dtype-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_xbit_dtype_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
    weight_nbit: i32,
) -> &'a Tensor<'b> {
    resize_out_tensor(weight, indices, out, weight_nbit);

    // TODO (jakeszwe): improve these to account for the size of out in relation
    // to weight and indices accounting for a possible batch dimension
    check_embedding_xbit_args(
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
        weight_nbit,
    );

    let params_type: ScalarType = weight_scales.scalar_type();
    let out_type: ScalarType = out.scalar_type();

    let name = "quantized_decomposed::embedding_xbit.dtype_out";
    let indices_type: ScalarType = indices.scalar_type();
    crate::et_switch_three_types!(Float, Half, BFloat16, params_type, ctx, name, CTYPE_P, {
        crate::et_switch_three_types!(Float, Half, BFloat16, out_type, ctx, name, CTYPE_OUT, {
            crate::et_switch_two_types!(Int, Long, indices_type, ctx, name, CTYPE_IDX, {
                embedding_xbit_per_channel::<CTYPE_P, CTYPE_OUT, CTYPE_IDX>(
                    weight,
                    weight_scales,
                    opt_weight_zero_points,
                    indices,
                    out,
                    weight_nbit,
                );
            });
        });
    });

    out
}

#[allow(clippy::too_many_arguments)]
pub fn quantized_embedding_xbit_dtype_out_nocontext<'a, 'b>(
    weight: &Tensor,
    weight_scales: &Tensor,
    opt_weight_zero_points: &Option<&Tensor>,
    weight_quant_min: i64,
    weight_quant_max: i64,
    indices: &Tensor,
    out_dtype: Option<ScalarType>,
    out: &'a Tensor<'b>,
    weight_nbit: i32,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let mut context = default_context();
    let res = quantized_embedding_xbit_dtype_out(
        &mut context,
        weight,
        weight_scales,
        opt_weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        out_dtype,
        out,
        weight_nbit,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}
