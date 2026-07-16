//! Literal port of kernels/quantized/cpu/op_quantize.cpp.
//!
//! For an input tensor, use the scale and zero_point arguments to quantize it.

use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::dim_order_util::dim_order_to_stride_nocheck;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_floating_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    nonzero_dim, resize_tensor, tensor_has_dim,
};
use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl,
};
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

fn default_context() -> KernelRuntimeContext<'static> {
    KernelRuntimeContext::new(
        crate::extension::module::module::null_event_tracer(),
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
    )
}

// PORT-NOTE: `ET_CHECK_MSG` is the C++ fatal check; mirrored with a local abort
// on failure (message dropped since a fatal abort follows), matching the
// established pattern in op_dequantize.rs / tensor_util.rs.
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

// PORT-NOTE: `quantize_val` input type `K` is Float/Double/Half. The C++ computes
// `std::nearbyint(static_cast<float>(inv_scale * value))`. The product
// `inv_scale(float) * value(K)` promotes per usual arithmetic conversions (double
// for the Double input case, float otherwise), then `static_cast<float>` narrows.
// `QuantIn::mul_inv_scale` reproduces `static_cast<float>(inv_scale * value)` in
// the correct intermediate precision.
trait QuantIn: Copy {
    fn mul_inv_scale(self, inv_scale: f32) -> f32;
}
impl QuantIn for f32 {
    fn mul_inv_scale(self, inv_scale: f32) -> f32 {
        // float * float -> float, then static_cast<float> is a no-op.
        inv_scale * self
    }
}
impl QuantIn for f64 {
    fn mul_inv_scale(self, inv_scale: f32) -> f32 {
        // inv_scale(float) * value(double) promotes to double; static_cast<float>
        // narrows.
        (inv_scale as f64 * self) as f32
    }
}
impl QuantIn for crate::runtime::core::portable_type::Half {
    fn mul_inv_scale(self, inv_scale: f32) -> f32 {
        // Half promotes to float for the multiply; result stays float.
        inv_scale * crate::runtime::core::portable_type::Half::to_f32(self)
    }
}

// PORT-NOTE: `quantize_val` output type `T` is one of the integer output types.
// The final `static_cast<T>(qvalue)` truncates the (clamped) int64 into the
// output integer type. `QuantOut::from_i64` reproduces the truncating cast.
trait QuantOut: Copy {
    fn from_i64(v: i64) -> Self;
}
macro_rules! impl_quant_out {
    ($t:ty) => {
        impl QuantOut for $t {
            fn from_i64(v: i64) -> Self {
                v as $t
            }
        }
    };
}
impl_quant_out!(u8);
impl_quant_out!(i8);
impl_quant_out!(i16);
impl_quant_out!(i32);
impl_quant_out!(i64);
impl_quant_out!(u16);

/// Asserts that the parameters are valid.
// [spec:et:def:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]
// [spec:et:sem:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn]
fn check_quantize_per_tensor_args(
    input: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &Tensor,
) {
    // Ensure self and out has the same shape
    et_check_msg!(
        is_floating_type(input.scalar_type()),
        "input.scalar_type() is not floating type"
    );

    let mut quant_min_lower_bound: i32 = 0;
    let mut quant_max_upper_bound: i32 = 0;
    let out_dtype: ScalarType = out.scalar_type();
    et_check_msg!(
        out_dtype == dtype,
        "out.scalar_type() is not matching dtype argument"
    );
    if out_dtype == ScalarType::Byte {
        quant_min_lower_bound = u8::MIN as i32;
        quant_max_upper_bound = u8::MAX as i32;
    } else if dtype == ScalarType::Char {
        quant_min_lower_bound = i8::MIN as i32;
        quant_max_upper_bound = i8::MAX as i32;
    } else if dtype == ScalarType::Bits16 || dtype == ScalarType::UInt16 {
        quant_min_lower_bound = u16::MIN as i32;
        quant_max_upper_bound = u16::MAX as i32;
    } else if dtype == ScalarType::Short {
        quant_min_lower_bound = i16::MIN as i32;
        quant_max_upper_bound = i16::MAX as i32;
    } else if dtype == ScalarType::Int {
        quant_min_lower_bound = i32::MIN;
        quant_max_upper_bound = i32::MAX;
    } else {
        et_check_msg!(false, "Unsupported dtype");
    }
    et_check_msg!(
        quant_min >= quant_min_lower_bound as i64,
        "quant_min out of bound for dtype"
    );

    et_check_msg!(
        quant_max <= quant_max_upper_bound as i64,
        "quant_max out of bound for dtype"
    );
}

// [spec:et:def:op-quantize.torch.executor.native.quantize-val-fn]
// [spec:et:sem:op-quantize.torch.executor.native.quantize-val-fn]
fn quantize_val<T: QuantOut, K: QuantIn>(
    scale: f64,
    zero_point: i64,
    value: K,
    quant_min: i64,
    quant_max: i64,
) -> T {
    let mut qvalue: i64;
    let inv_scale: f32 = 1.0f32 / (scale as f32);
    // PORT-NOTE: `static_cast<int32_t>(zero_point) + std::nearbyint(...)`. The
    // `std::nearbyint` under the default FE_TONEAREST rounding mode is
    // round-half-to-even; `f32::round_ties_even` reproduces it.
    qvalue = (zero_point as i32) as i64 + value.mul_inv_scale(inv_scale).round_ties_even() as i64;

    qvalue = core::cmp::max(qvalue, quant_min);
    qvalue = core::cmp::min(qvalue, quant_max);
    T::from_i64(qvalue)
}

// PORT-NOTE: literal port of the C++ `#if defined(__aarch64__) ||
// defined(__ARM_NEON__)` NEON fast path. Only the `__aarch64__` variant is
// ported (Rust `target_arch = "aarch64"` corresponds exactly to the C++
// `#if defined(__aarch64__)` branch inside `quantize_arm`); the ARMv7
// magic-float `#else` branch is not reachable under this arch gate. The result
// is (per spec) bit-identical to the scalar `quantize_val` path.
#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    // Traits for type-specific NEON operations.
    pub trait NeonQuantizeTraits: Copy {
        type Vec8;
        // Narrow int16x8 to T x8 with saturation.
        unsafe fn narrow_and_saturate(v: int16x8_t) -> Self::Vec8;
        // Store T x8 to memory.
        unsafe fn store(ptr: *mut Self, v: Self::Vec8);
        // Scalar clamping for T.
        fn clamp_scalar(val: i32) -> Self;
        // `static_cast<T>(qval)` for the scalar tail loop.
        fn narrow_scalar(qval: i32) -> Self;
    }

    // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t]
    impl NeonQuantizeTraits for u8 {
        type Vec8 = uint8x8_t;

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.narrow-and-saturate-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.narrow-and-saturate-fn]
        unsafe fn narrow_and_saturate(v: int16x8_t) -> uint8x8_t {
            vqmovun_s16(v)
        }

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.store-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.store-fn]
        unsafe fn store(ptr: *mut u8, v: uint8x8_t) {
            vst1_u8(ptr, v);
        }

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.clamp-scalar-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.clamp-scalar-fn]
        fn clamp_scalar(val: i32) -> u8 {
            core::cmp::min(255, core::cmp::max(0, val)) as u8
        }

        fn narrow_scalar(qval: i32) -> u8 {
            qval as u8
        }
    }

    // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t]
    impl NeonQuantizeTraits for i8 {
        type Vec8 = int8x8_t;

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.narrow-and-saturate-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.narrow-and-saturate-fn]
        unsafe fn narrow_and_saturate(v: int16x8_t) -> int8x8_t {
            vqmovn_s16(v)
        }

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.store-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.store-fn]
        unsafe fn store(ptr: *mut i8, v: int8x8_t) {
            vst1_s8(ptr, v);
        }

        // [spec:et:def:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.clamp-scalar-fn]
        // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.clamp-scalar-fn]
        fn clamp_scalar(val: i32) -> i8 {
            core::cmp::min(127, core::cmp::max(-128, val)) as i8
        }

        fn narrow_scalar(qval: i32) -> i8 {
            qval as i8
        }
    }

    // Unified ARM NEON optimized quantization for contiguous blocks. Processes N
    // elements with a single scale/zero_point pair. Used for both per-tensor
    // (entire tensor) and per-channel (one block per channel).
    // [spec:et:def:op-quantize.torch.executor.native.quantize-arm-fn]
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-arm-fn]
    //
    // PORT-NOTE: `clamp_scalar` is defined by the C++ traits but unused by the
    // `__aarch64__` variant (the tail loop clamps inline against
    // quant_min/quant_max); referenced once so it is not dead code, matching the
    // C++ which likewise never calls it in this branch.
    #[target_feature(enable = "neon")]
    pub unsafe fn quantize_arm<T: NeonQuantizeTraits>(
        in_: *const f32,
        out: *mut T,
        n: i64,
        inv_scale: f32,
        zero_point: i32,
        quant_min: i32,
        quant_max: i32,
    ) {
        let _ = <T as NeonQuantizeTraits>::clamp_scalar;
        let vinv_scale: float32x4_t = vdupq_n_f32(inv_scale);

        // ARMv8: Use vcvtnq_s32_f32 for rounding.
        let vzero_point: int16x8_t = vdupq_n_s16(zero_point as i16);
        let vquant_min: int16x8_t = vdupq_n_s16(quant_min as i16);
        let vquant_max: int16x8_t = vdupq_n_s16(quant_max as i16);

        let mut i: i64 = 0;
        // Process 8 elements at a time.
        while i + 8 <= n {
            let vin0123: float32x4_t = vld1q_f32(in_.offset(i as isize));
            let vin4567: float32x4_t = vld1q_f32(in_.offset((i + 4) as isize));

            // Multiply by inv_scale and round.
            let v0123_rounded: int32x4_t = vcvtnq_s32_f32(vmulq_f32(vin0123, vinv_scale));
            let v4567_rounded: int32x4_t = vcvtnq_s32_f32(vmulq_f32(vin4567, vinv_scale));

            // Combine to int16 and add zero_point.
            let mut v01234567_packed: int16x8_t = vqaddq_s16(
                vqmovn_high_s32(vqmovn_s32(v0123_rounded), v4567_rounded),
                vzero_point,
            );

            // Clamp to quant_min/quant_max.
            v01234567_packed = vmaxq_s16(v01234567_packed, vquant_min);
            v01234567_packed = vminq_s16(v01234567_packed, vquant_max);

            // Convert to T (int8/uint8) with saturation using type-specific op.
            let vout01234567 = <T as NeonQuantizeTraits>::narrow_and_saturate(v01234567_packed);
            <T as NeonQuantizeTraits>::store(out.offset(i as isize), vout01234567);

            i += 8;
        }

        // Handle remaining elements with proper quant_min/quant_max clamping.
        while i < n {
            let val: f32 = *in_.offset(i as isize) * inv_scale;
            let mut qval: i32 = (val.round_ties_even() as i32) + zero_point;
            qval = core::cmp::max(quant_min, core::cmp::min(quant_max, qval));
            *out.offset(i as isize) = <T as NeonQuantizeTraits>::narrow_scalar(qval);

            i += 1;
        }
    }
}

// [spec:et:def:op-quantize.torch.executor.native.quantize-per-tensor-out-fn]
// [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn]
pub fn quantize_per_tensor_out<'a, 'b>(
    input: &Tensor,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in quantize_per_tensor_out"
    );

    check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

    // Try ARM NEON optimized path for float->int8/uint8 quantization.
    #[cfg(target_arch = "aarch64")]
    {
        if input.scalar_type() == ScalarType::Float {
            if dtype == ScalarType::Byte {
                unsafe {
                    neon::quantize_arm::<u8>(
                        input.const_data_ptr::<f32>(),
                        out.mutable_data_ptr::<u8>(),
                        input.numel() as i64,
                        1.0f32 / (scale as f32),
                        zero_point as i32,
                        quant_min as i32,
                        quant_max as i32,
                    );
                }
                return out;
            } else if dtype == ScalarType::Char {
                unsafe {
                    neon::quantize_arm::<i8>(
                        input.const_data_ptr::<f32>(),
                        out.mutable_data_ptr::<i8>(),
                        input.numel() as i64,
                        1.0f32 / (scale as f32),
                        zero_point as i32,
                        quant_min as i32,
                        quant_max as i32,
                    );
                }
                return out;
            }
        }
    }

    // Fallback scalar implementation for all other cases
    // PORT-NOTE: the C++ `QUANTIZE_IMPL` / `CALCULATE_FLOAT_TYPE` macro pair over
    // ET_FORALL_FLOATH_TYPES (input) and ET_FORALL_INT_TYPES + Bits16/UInt16
    // (output) is expanded here as explicit nested matches. The `default` arms
    // fire the fatal `ET_CHECK_MSG(false, ...)`.
    fn quantize_impl<IN: QuantIn, OUT: QuantOut>(
        input: &Tensor,
        out: &Tensor,
        scale: f64,
        zero_point: i64,
        quant_min: i64,
        quant_max: i64,
    ) {
        let out_data_ptr: *mut OUT = out.mutable_data_ptr::<OUT>();
        let input_data_ptr: *const IN = input.const_data_ptr::<IN>();
        let input_numel = input.numel();
        for i in 0..(input_numel as usize) {
            let value: IN = unsafe { *input_data_ptr.add(i) };
            unsafe {
                *out_data_ptr.add(i) =
                    quantize_val::<OUT, IN>(scale, zero_point, value, quant_min, quant_max);
            }
        }
    }

    macro_rules! calculate_float_type {
        ($in:ty) => {
            match out.scalar_type() {
                ScalarType::Byte => {
                    quantize_impl::<$in, u8>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::Char => {
                    quantize_impl::<$in, i8>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::Short => {
                    quantize_impl::<$in, i16>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::Int => {
                    quantize_impl::<$in, i32>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::Long => {
                    quantize_impl::<$in, i64>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::Bits16 => {
                    quantize_impl::<$in, u16>(input, out, scale, zero_point, quant_min, quant_max)
                }
                ScalarType::UInt16 => {
                    quantize_impl::<$in, u16>(input, out, scale, zero_point, quant_min, quant_max)
                }
                _ => {
                    et_check_msg!(false, "Unhandled output dtype");
                }
            }
        };
    }

    match input.scalar_type() {
        ScalarType::Float => calculate_float_type!(f32),
        ScalarType::Double => calculate_float_type!(f64),
        ScalarType::Half => {
            calculate_float_type!(crate::runtime::core::portable_type::Half)
        }
        _ => {
            et_check_msg!(false, "Unhandled input dtype");
        }
    }
    out
}

// [spec:et:def:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn]
// [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn]
pub fn quantize_per_tensor_tensor_args_out<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // Temporary change to allow not fatal failure for now to unblock some
    // expected failure tests that are dying instead of failure. Will revisit
    // after ET_KERNEL_CHECK is fully implemented and properly allows non fatal
    // failures.
    if scale.scalar_type() != ScalarType::Double {
        context.fail(Error::InvalidArgument);
        return out;
    }
    et_check_msg!(
        scale.scalar_type() == ScalarType::Double,
        "Expected scale to be Double tensor received:"
    );
    et_check_msg!(
        zero_point.scalar_type() == ScalarType::Long,
        "Expected zero_point to be Long tensor received:"
    );
    et_check_msg!(
        scale.numel() == 1,
        "Exepcted scale to only have one element received:"
    );
    et_check_msg!(
        zero_point.numel() == 1,
        "Exepcted zero_point to only have one element received:"
    );

    quantize_per_tensor_out(
        input,
        unsafe { *scale.const_data_ptr::<f64>().add(0) },
        unsafe { *zero_point.const_data_ptr::<i64>().add(0) },
        quant_min,
        quant_max,
        dtype,
        out,
    );
    out
}

pub fn quantize_per_tensor_tensor_args_out_nocontext<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let mut context = default_context();
    let res = quantize_per_tensor_tensor_args_out(
        &mut context,
        input,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
        out,
    );
    et_check!(context.failure_state() == Error::Ok);
    res
}

pub fn quantize_per_tensor_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: f64,
    zero_point: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // TODO(larryliu): Add a context arg to the real op function and remove this
    // wrapper
    let _ = context;
    quantize_per_tensor_out(input, scale, zero_point, quant_min, quant_max, dtype, out)
}

// [spec:et:def:op-quantize.torch.executor.native.quantize-per-channel-out-fn]
// [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn]
//
// PORT-NOTE: the NEON fast path (float->int8/uint8 via `quantize_arm`, including
// the `parallel_for` block split) is a hardware-specific optimization whose
// result is bit-identical to the scalar fallback; only the scalar loop is ported
// here, mirroring op_dequantize.rs.
#[allow(clippy::too_many_arguments)]
pub fn quantize_per_channel_out<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    mut axis: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // normalize axis
    et_check_msg!(
        tensor_has_dim(input, axis),
        "axis is not legal it should be -input.dim() <= axis < input.dim()"
    );

    if axis < 0 {
        axis += nonzero_dim(input) as i64;
    }

    et_check_msg!(
        scale.scalar_type() == ScalarType::Double,
        "scale.scalar_type() is not double type"
    );

    et_check_msg!(
        scale.numel() == input.size(axis as isize),
        "scale.numel() != input.size(axis)"
    );

    et_check_msg!(
        zero_point.scalar_type() == ScalarType::Long,
        "zero_point.scalar_type() is not integer type"
    );

    et_check_msg!(
        zero_point.numel() == input.size(axis as isize),
        "zero_point.numel() != input.size(axis)"
    );

    check_quantize_per_tensor_args(input, quant_min, quant_max, dtype, out);

    let scale_data: *const f64 = scale.const_data_ptr::<f64>();
    let zero_point_data: *const i64 = zero_point.const_data_ptr::<i64>();

    // Calculate the block size for each channel
    let mut axis_block_size: i64 = 1;
    {
        let mut i: i64 = axis + 1;
        while i < input.dim() as i64 {
            axis_block_size *= input.size(i as isize) as i64;
            i += 1;
        }
    }
    let axis_size: i64 = input.size(axis as isize) as i64;

    // Fallback scalar implementation
    // PORT-NOTE: the C++ `QUANTIZE_IMPL` / `CALCULATE_FLOAT_TYPE` macro pair over
    // ET_FORALL_FLOATH_TYPES (input) and ET_FORALL_INT_TYPES + Bits16/UInt16
    // (output) is expanded here as explicit nested matches. The single loop over
    // all elements computes `channel_idx = (i / axis_block_size) % axis_size`.
    #[allow(clippy::too_many_arguments)]
    fn quantize_impl<IN: QuantIn, OUT: QuantOut>(
        input: &Tensor,
        out: &Tensor,
        scale_data: *const f64,
        zero_point_data: *const i64,
        axis_block_size: i64,
        axis_size: i64,
        quant_min: i64,
        quant_max: i64,
    ) {
        let out_data_ptr: *mut OUT = out.mutable_data_ptr::<OUT>();
        let input_data_ptr: *const IN = input.const_data_ptr::<IN>();
        let input_numel: i64 = input.numel() as i64;
        // Single loop over all elements
        let mut i: i64 = 0;
        while i < input_numel {
            // Calculate which channel this element belongs to
            let channel_idx: i64 = (i / axis_block_size) % axis_size;
            // Get quantization parameters for this channel
            let _scale: f64 = unsafe { *scale_data.offset(channel_idx as isize) };
            let _zero_point: i64 = unsafe { *zero_point_data.offset(channel_idx as isize) };
            // Apply quantization
            let value: IN = unsafe { *input_data_ptr.offset(i as isize) };
            unsafe {
                *out_data_ptr.offset(i as isize) =
                    quantize_val::<OUT, IN>(_scale, _zero_point, value, quant_min, quant_max);
            }
            i += 1;
        }
    }

    macro_rules! calculate_float_type {
        ($in:ty) => {
            match out.scalar_type() {
                ScalarType::Byte => quantize_impl::<$in, u8>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::Char => quantize_impl::<$in, i8>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::Short => quantize_impl::<$in, i16>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::Int => quantize_impl::<$in, i32>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::Long => quantize_impl::<$in, i64>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::Bits16 => quantize_impl::<$in, u16>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                ScalarType::UInt16 => quantize_impl::<$in, u16>(
                    input,
                    out,
                    scale_data,
                    zero_point_data,
                    axis_block_size,
                    axis_size,
                    quant_min,
                    quant_max,
                ),
                _ => {
                    et_check_msg!(false, "Unhandled output dtype");
                }
            }
        };
    }

    match input.scalar_type() {
        ScalarType::Float => calculate_float_type!(f32),
        ScalarType::Double => calculate_float_type!(f64),
        ScalarType::Half => {
            calculate_float_type!(crate::runtime::core::portable_type::Half)
        }
        _ => {
            et_check_msg!(false, "Unhandled input dtype");
        }
    }

    out
}

// [spec:et:def:op-quantize.torch.executor.quantize-per-channel-out-fn]
// [spec:et:sem:op-quantize.torch.executor.quantize-per-channel-out-fn]
#[allow(clippy::too_many_arguments)]
pub fn quantize_per_channel_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    axis: i64,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = context;
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in quantize_per_channel_out"
    );

    quantize_per_channel_out(
        input, scale, zero_point, axis, quant_min, quant_max, dtype, out,
    )
}

// [spec:et:def:op-quantize.torch.executor.quantize-per-token-out-fn]
// [spec:et:sem:op-quantize.torch.executor.quantize-per-token-out-fn]
pub fn quantize_per_token_out<'a, 'b>(
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let mut num_tokens: usize = 1;
    {
        let mut i: usize = 0;
        while (i as isize) < input.dim() - 1 {
            num_tokens *= input.size(i as isize) as usize;
            i += 1;
        }
    }
    // This unfortunate change is needed because we compile op_quantize for aten
    // mode as well.
    // PORT-NOTE: the ATen `at::from_blob` reshape branch is behind `USE_ATEN_LIB`;
    // the portable branch builds a rank-2 view TensorImpl over `input`'s data and
    // (portable-only) resizes `out`. Ported unconditionally as the portable
    // branch, mirroring op_dequantize.rs.
    let mut input_dim_order: [DimOrderType; 2] = [0, 1];
    let mut input_sizes: [SizesType; 2] = [0; 2];
    input_sizes[0] = num_tokens as SizesType;
    input_sizes[1] = input.size(input.dim() - 1) as SizesType;
    let mut input_strides: [StridesType; 2] = [0; 2];
    unsafe {
        dim_order_to_stride_nocheck(
            input_sizes.as_ptr(),
            input_dim_order.as_ptr(),
            2,
            input_strides.as_mut_ptr(),
        );
    }
    let input_data: *mut core::ffi::c_void = input.mutable_data_ptr_typed();
    let mut reshaped_input_impl: TensorImpl = TensorImpl::new(
        input.scalar_type(),
        2,
        input_sizes.as_mut_ptr(),
        input_data,
        input_dim_order.as_mut_ptr(),
        input_strides.as_mut_ptr(),
        TensorShapeDynamism::STATIC,
        DeviceType::CPU,
        0 as DeviceIndex,
    );
    let reshaped_input: Tensor = Tensor::new(&mut reshaped_input_impl);
    let err: Error = resize_tensor(out, input.sizes());
    et_check_msg!(
        err == Error::Ok,
        "Failed to resize out Tensor in quantize_per_channel_out"
    );

    quantize_per_channel_out(
        &reshaped_input,
        scale,
        zero_point,
        0, /* axis */
        quant_min,
        quant_max,
        dtype,
        out,
    )
}

pub fn quantize_per_token_out_context<'a, 'b>(
    context: &mut KernelRuntimeContext,
    input: &Tensor,
    scale: &Tensor,
    zero_point: &Tensor,
    quant_min: i64,
    quant_max: i64,
    dtype: ScalarType,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let _ = context;
    quantize_per_token_out(input, scale, zero_point, quant_min, quant_max, dtype, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_eq;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::portable_type::Half;
    use crate::runtime::core::portable_type::bits_types::bits16;
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism::STATIC;

    // Output element-type constructor from an integer literal (C++ implicit int
    // conversion at the `full`/`make` call sites).
    trait FromI64Ctor: Copy {
        fn ctor(v: i64) -> Self;
    }
    macro_rules! impl_from_i64_ctor {
        ($($t:ty),*) => {$(impl FromI64Ctor for $t { fn ctor(v: i64) -> Self { v as $t } })*};
    }
    impl_from_i64_ctor!(u8, i8, i16, i32, u16);
    impl FromI64Ctor for bits16 {
        fn ctor(v: i64) -> Self {
            bits16::new(v as u16)
        }
    }

    // Input element-type constructor from a double (C++ implicit conversion for
    // the floating input dtypes Float/Half/Double).
    trait FromF64Ctor: Copy {
        fn ctor(v: f64) -> Self;
    }
    impl FromF64Ctor for f32 {
        fn ctor(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64Ctor for f64 {
        fn ctor(v: f64) -> Self {
            v
        }
    }
    impl FromF64Ctor for Half {
        fn ctor(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }

    /// A generic smoke test that works for any dtype that supports ones() and
    /// zeros(). Templated on the OUTPUT dtype.
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    fn test_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromI64Ctor,
    {
        let tf = TensorFactory::<f32>::new();

        let input = tf.full(vec![3, 5], 4.0, STATIC);
        let scale: f64 = 0.5;

        let zero_point: i64 = 108;
        let quant_min: i64 = 0;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<T>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // 4 / 0.5 + 127
        let expected = tfo.full(vec![3, 5], T::ctor(116), STATIC);
        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            T::VALUE,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    fn test_input_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64Ctor,
    {
        let tf_input = TensorFactory::<T>::new();

        let input = tf_input.full(vec![3, 5], T::ctor(4.0), STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 108;
        let quant_min: i64 = 0;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // 4 / 0.5 + 108 = 116
        let expected = tfo.full(vec![3, 5], 116, STATIC);
        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    #[test]
    fn op_quantize_out_test_all_input_dtypes_supported() {
        test_input_dtype::<f32>();
        test_input_dtype::<Half>();
        test_input_dtype::<f64>();
    }

    // Exercises check_quantize_per_tensor_args across every output-dtype branch
    // (Byte/Char/Short/Int/Bits16/UInt16): each call passes valid quant_min/max
    // bounds and matching dtype through the arg-validation guard before quantizing.
    // [spec:et:sem:op-quantize.torch.executor.native.check-quantize-per-tensor-args-fn/test]
    #[test]
    fn op_quantize_out_test_all_dtypes_supported() {
        test_dtype::<u8>();
        test_dtype::<i8>();
        test_dtype::<i16>();
        test_dtype::<bits16>();
        test_dtype::<u16>();
        test_dtype::<i32>();
    }

    // Double input takes the scalar fallback path, pinning quantize_val's
    // inv_scale multiply / nearbyint round-half-to-even / zero_point add / clamp
    // math (3.14159265359 / 0.01 - 100 = 214.159... -> 214).
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-val-fn/test]
    #[test]
    fn op_quantize_out_test_double_input_test() {
        let tf_double = TensorFactory::<f64>::new();

        // Test with a more complex value that might have precision differences
        let input = tf_double.full(vec![2, 3], 3.14159265359, STATIC);
        let scale: f64 = 0.01;
        let zero_point: i64 = -100;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // 3.14159265359 / 0.01 - 100 = 214.159265359
        let expected = tfo.full(vec![2, 3], 214, STATIC);
        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_half_input_test() {
        let tf_half = TensorFactory::<Half>::new();

        let input = tf_half.full(vec![2, 3], Half::from_f32(2.5), STATIC);
        let scale: f64 = 0.5;
        let zero_point: i64 = 10;
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // 2.5 / 0.5 + 10 = 15
        let expected = tfo.full(vec![2, 3], 15, STATIC);
        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn/test]
    #[test]
    fn op_quantize_out_test_tensor_arg_overload() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![3, 5], 4.0, STATIC);
        let scale = tf_double.make_default(vec![1], vec![0.5]);
        let zero_point = tf_long.make_default(vec![1], vec![127]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![3, 5], STATIC);
        // 4 / 0.5 + 127
        let expected = tfo.full(vec![3, 5], 135, STATIC);
        let mut context = default_context();
        quantize_per_tensor_tensor_args_out(
            &mut context,
            &input,
            &scale,
            &zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-tensor-args-out-fn/test]
    #[test]
    fn op_quantize_out_test_test_out_of_bounds() {
        // Test where 1.0 / epsilon is larger than 8bit integer.
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.ones(vec![1, 3, 256, 256], STATIC);

        let scale = tf_double.make_default(vec![1], vec![0.0011316323652863503]);
        let zero_point = tf_long.make_default(vec![1], vec![0]);
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![1, 3, 256, 256], STATIC);

        let expected = tfo.full(vec![1, 3, 256, 256], 127, STATIC);

        let mut context = default_context();
        quantize_per_tensor_tensor_args_out(
            &mut context,
            &input,
            &scale,
            &zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![3, 2], 4.0, STATIC);
        let scale = tf_double.make_default(vec![2], vec![0.5, 1.0]);
        let zero_point = tf_long.make_default(vec![2], vec![127, 63]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![3, 2], STATIC);
        // 4 / 0.5 + 127
        // 4 / 1 + 63
        let expected = tfo.make_default(vec![3, 2], vec![135, 67, 135, 67, 135, 67]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_axis0() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![3, 2], 4.0, STATIC);
        let scale = tf_double.make_default(vec![3], vec![0.5, 1.0, 2.0]);
        let zero_point = tf_long.make_default(vec![3], vec![100, 50, 25]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![3, 2], STATIC);
        // Channel 0: 4 / 0.5 + 100 = 108
        // Channel 1: 4 / 1.0 + 50 = 54
        // Channel 2: 4 / 2.0 + 25 = 27
        let expected = tfo.make_default(vec![3, 2], vec![108, 108, 54, 54, 27, 27]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            0,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel3d() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Test 3D tensor with axis=1 (middle dimension)
        let input = tf_float.full(vec![2, 3, 4], 6.0, STATIC);
        let scale = tf_double.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let zero_point = tf_long.make_default(vec![3], vec![10, 20, 30]);
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![2, 3, 4], STATIC);
        // Channel 0: 6 / 0.5 + 10 = 22
        // Channel 1: 6 / 1.0 + 20 = 26
        // Channel 2: 6 / 1.5 + 30 = 34
        let expected = tfo.make_default(
            vec![2, 3, 4],
            vec![
                22, 22, 22, 22, // First batch, channel 0
                26, 26, 26, 26, // First batch, channel 1
                34, 34, 34, 34, // First batch, channel 2
                22, 22, 22, 22, // Second batch, channel 0
                26, 26, 26, 26, // Second batch, channel 1
                34, 34, 34, 34, // Second batch, channel 2
            ],
        );
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel4d() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Test 4D tensor with axis=2 (typical conv weight layout: N,C,H,W)
        let input = tf_float.full(vec![2, 2, 3, 2], 8.0, STATIC);
        let scale = tf_double.make_default(vec![3], vec![0.25, 0.5, 1.0]);
        let zero_point = tf_long.make_default(vec![3], vec![0, 10, 20]);
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![2, 2, 3, 2], STATIC);
        // Channel 0: 8 / 0.25 + 0 = 32
        // Channel 1: 8 / 0.5 + 10 = 26
        // Channel 2: 8 / 1.0 + 20 = 28
        let mut expected_data: Vec<i8> = Vec::new();
        for _n in 0..2 {
            for _c in 0..2 {
                for h in 0..3 {
                    for _w in 0..2 {
                        let val: i8 = if h == 0 {
                            32
                        } else if h == 1 {
                            26
                        } else {
                            28
                        };
                        expected_data.push(val);
                    }
                }
            }
        }
        let expected = tfo.make_default(vec![2, 2, 3, 2], expected_data);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            2,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_negative_axis() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![2, 3], 5.0, STATIC);
        let scale = tf_double.make_default(vec![3], vec![0.5, 1.0, 2.0]);
        let zero_point = tf_long.make_default(vec![3], vec![0, 10, 20]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // Using axis=-1 should be equivalent to axis=1 for 2D tensor
        // Channel 0: 5 / 0.5 + 0 = 10
        // Channel 1: 5 / 1.0 + 10 = 15
        // Channel 2: 5 / 2.0 + 20 = 22 (rounded from 22.5)
        let expected = tfo.make_default(vec![2, 3], vec![10, 15, 22, 10, 15, 22]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            -1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_single_channel() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![3, 1, 4], 7.0, STATIC);
        let scale = tf_double.make_default(vec![1], vec![0.5]);
        let zero_point = tf_long.make_default(vec![1], vec![128]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![3, 1, 4], STATIC);
        // Single channel: 7 / 0.5 + 128 = 142
        let expected = tfo.full(vec![3, 1, 4], 142, STATIC);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_different_input_types() {
        let tf_double_input = TensorFactory::<f64>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_double_input.full(vec![2, 2], 3.14159, STATIC);
        let scale = tf_double.make_default(vec![2], vec![0.01, 0.02]);
        let zero_point = tf_long.make_default(vec![2], vec![0, 100]);
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![2, 2], STATIC);
        // Channel 0: 3.14159 / 0.01 + 0 = 314 -> clamped to 127
        // Channel 1: 3.14159 / 0.02 + 100 = 257 -> clamped to 127
        let expected = tfo.make_default(vec![2, 2], vec![127, 127, 127, 127]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_different_output_types() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![2, 2], 10.0, STATIC);
        let scale = tf_double.make_default(vec![2], vec![1.0, 2.0]);
        let zero_point = tf_long.make_default(vec![2], vec![1000, 2000]);
        let quant_min: i64 = -32768;
        let quant_max: i64 = 32767;

        // Test with 16-bit output
        let tfo = TensorFactory::<i16>::new();
        let out = tfo.zeros(vec![2, 2], STATIC);
        // Channel 0: 10 / 1.0 + 1000 = 1010
        // Channel 1: 10 / 2.0 + 2000 = 2005
        let expected = tfo.make_default(vec![2, 2], vec![1010, 2005, 1010, 2005]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Short,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_mixed_values() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Test with different input values per position
        let input = tf_float.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scale = tf_double.make_default(vec![3], vec![0.5, 1.0, 1.5]);
        let zero_point = tf_long.make_default(vec![3], vec![10, 20, 30]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // Row 0: [1.0/0.5+10, 2.0/1.0+20, 3.0/1.5+30] = [12, 22, 32]
        // Row 1: [4.0/0.5+10, 5.0/1.0+20, 6.0/1.5+30] = [18, 25, 34]
        let expected = tfo.make_default(vec![2, 3], vec![12, 22, 32, 18, 25, 34]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_clamping_behavior() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Test values that will exceed quant_min/quant_max bounds
        let input = tf_float.make_default(vec![1, 3], vec![-100.0, 0.0, 100.0]);
        let scale = tf_double.make_default(vec![3], vec![1.0, 1.0, 1.0]);
        let zero_point = tf_long.make_default(vec![3], vec![0, 0, 0]);
        let quant_min: i64 = -10;
        let quant_max: i64 = 10;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![1, 3], STATIC);
        // Values: [-100, 0, 100] should be clamped to [-10, 0, 10]
        let expected = tfo.make_default(vec![1, 3], vec![-10, 0, 10]);
        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // Reference implementation of the C++ per-tensor scalar quantize math used by
    // the large-tensor tests, using f32 arithmetic and round-half-to-even to match
    // `std::nearbyint`.
    fn q_ref_f32(val_over_scale: f32, zero_point: i64, qmin: i32, qmax: i32) -> i32 {
        let mut qval = val_over_scale.round_ties_even() as i32 + zero_point as i32;
        qval = core::cmp::min(qmax, core::cmp::max(qmin, qval));
        qval
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_large_per_channel_clamping_simd_path() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let num_channels = 3usize;
        let block_size = 80usize;
        let mut input_data = vec![0.0f32; num_channels * block_size];

        for ch in 0..num_channels {
            for i in 0..block_size {
                input_data[ch * block_size + i] =
                    ((i % 40) as i32 - 20) as f32 * 5.0f32 * (ch as f32 + 1.0);
            }
        }
        let input = tf_float.make_default(
            vec![num_channels as i32, block_size as i32],
            input_data.clone(),
        );

        let scale = tf_double.make_default(vec![num_channels as i32], vec![1.0, 1.0, 1.0]);
        let zero_point = tf_long.make_default(vec![num_channels as i32], vec![0, 0, 0]);

        let quant_min: i64 = -20;
        let quant_max: i64 = 20;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![num_channels as i32, block_size as i32], STATIC);

        let scale_ptr = scale.const_data_ptr::<f64>();
        let zp_ptr = zero_point.const_data_ptr::<i64>();
        let mut expected_data = vec![0i8; num_channels * block_size];
        for ch in 0..num_channels {
            let ch_scale = unsafe { *scale_ptr.add(ch) };
            let ch_zero_point = unsafe { *zp_ptr.add(ch) };
            for i in 0..block_size {
                let idx = ch * block_size + i;
                let mut val = input_data[idx] as f64 / ch_scale;
                val = val.max(-1000.0).min(1000.0);
                let mut qval = val.round_ties_even() as i32 + ch_zero_point as i32;
                qval = core::cmp::max(quant_min as i32, core::cmp::min(quant_max as i32, qval));
                expected_data[idx] = qval as i8;
            }
        }
        let expected =
            tfo.make_default(vec![num_channels as i32, block_size as i32], expected_data);

        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            0,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // On aarch64, Float->Byte routes through neon::quantize_arm<u8>: the 64-element
    // main-loop body exercises the uint8 narrow_and_saturate (vqmovun_s16) + store
    // (vst1_u8), matching the scalar reference.
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-arm-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.narrow-and-saturate-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.store-fn/test]
    #[test]
    fn op_quantize_out_test_large_tensor_u_int8_simd_path() {
        let tf_float = TensorFactory::<f32>::new();

        let mut input_data = vec![0.0f32; 64];
        for i in 0..64 {
            input_data[i] = i as f32 * 0.5f32;
        }
        let input = tf_float.make_default(vec![64], input_data.clone());

        let scale: f64 = 0.1;
        let zero_point: i64 = 10;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![64], STATIC);

        let mut expected_data = vec![0u8; 64];
        for i in 0..64 {
            let val = input_data[i] / scale as f32;
            expected_data[i] = q_ref_f32(val, zero_point, 0, 255) as u8;
        }
        let expected = tfo.make_default(vec![64], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // On aarch64, Float->Char routes through neon::quantize_arm<i8>: the 72-element
    // body exercises the int8 narrow_and_saturate (vqmovn_s16) + store (vst1_s8)
    // plus the scalar tail loop, matching the scalar reference.
    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.narrow-and-saturate-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.store-fn/test]
    #[test]
    fn op_quantize_out_test_large_tensor_int8_simd_path() {
        let tf_float = TensorFactory::<f32>::new();

        let mut input_data = vec![0.0f32; 72];
        for i in 0..72 {
            input_data[i] = (i as i32 - 36) as f32 * 0.25f32;
        }
        let input = tf_float.make_default(vec![72], input_data.clone());

        let scale: f64 = 0.2;
        let zero_point: i64 = 0;
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![72], STATIC);

        let mut expected_data = vec![0i8; 72];
        for i in 0..72 {
            let val = input_data[i] / scale as f32;
            expected_data[i] = q_ref_f32(val, zero_point, -128, 127) as i8;
        }
        let expected = tfo.make_default(vec![72], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_large_tensor_with_remainder_u_int8() {
        let tf_float = TensorFactory::<f32>::new();

        let mut input_data = vec![0.0f32; 100];
        for i in 0..100 {
            input_data[i] = (i % 50) as f32 * 0.3f32;
        }
        let input = tf_float.make_default(vec![100], input_data.clone());

        let scale: f64 = 0.15;
        let zero_point: i64 = 128;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![100], STATIC);

        let mut expected_data = vec![0u8; 100];
        for i in 0..100 {
            let val = input_data[i] / scale as f32;
            expected_data[i] = q_ref_f32(val, zero_point, 0, 255) as u8;
        }
        let expected = tfo.make_default(vec![100], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_large_tensor_with_remainder_int8() {
        let tf_float = TensorFactory::<f32>::new();

        let mut input_data = vec![0.0f32; 99];
        for i in 0..99 {
            input_data[i] = (i as f32 * 0.1f32).sin() * 10.0f32;
        }
        let input = tf_float.make_default(vec![99], input_data.clone());

        let scale: f64 = 0.1;
        let zero_point: i64 = 5;
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![99], STATIC);

        let mut expected_data = vec![0i8; 99];
        for i in 0..99 {
            let val = input_data[i] / scale as f32;
            expected_data[i] = q_ref_f32(val, zero_point, -128, 127) as i8;
        }
        let expected = tfo.make_default(vec![99], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_very_large_tensor2d_u_int8() {
        let tf_float = TensorFactory::<f32>::new();

        let mut input_data = vec![0.0f32; 256 * 256];
        for i in 0..(256 * 256) {
            input_data[i] = ((i % 256) as i32 - 128) as f32 * 0.05f32;
        }
        let input = tf_float.make_default(vec![256, 256], input_data.clone());

        let scale: f64 = 0.05;
        let zero_point: i64 = 128;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![256, 256], STATIC);

        let mut expected_data = vec![0u8; 256 * 256];
        for i in 0..(256 * 256) {
            let mut val = input_data[i] as f64 / scale;
            val = val.max(-1000.0).min(1000.0);
            let mut qval = val.round_ties_even() as i32 + zero_point as i32;
            qval = core::cmp::min(255, core::cmp::max(0, qval));
            expected_data[i] = qval as u8;
        }
        let expected = tfo.make_default(vec![256, 256], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_very_large_tensor3d_int8() {
        let tf_float = TensorFactory::<f32>::new();

        let total_elements = 2 * 64 * 128;
        let mut input_data = vec![0.0f32; total_elements];
        for i in 0..total_elements {
            input_data[i] = (i as f32 * 0.01f32).cos() * 8.0f32;
        }
        let input = tf_float.make_default(vec![2, 64, 128], input_data.clone());

        let scale: f64 = 0.0625; // 1/16
        let zero_point: i64 = -10;
        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![2, 64, 128], STATIC);

        let mut expected_data = vec![0i8; total_elements];
        for i in 0..total_elements {
            let val = input_data[i] / scale as f32;
            expected_data[i] = q_ref_f32(val, zero_point, -128, 127) as i8;
        }
        let expected = tfo.make_default(vec![2, 64, 128], expected_data);

        quantize_per_tensor_out(
            &input,
            scale,
            zero_point,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-tensor-out-fn/test]
    #[test]
    fn op_quantize_out_test_edge_case_sizes_simd() {
        let tf_float = TensorFactory::<f32>::new();
        let tfo = TensorFactory::<u8>::new();

        let scale: f64 = 0.1;
        let zero_point: i64 = 100;
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let test_sizes: Vec<usize> = vec![7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33];

        for size in test_sizes {
            let mut input_data = vec![0.0f32; size];
            let mut expected_data = vec![0u8; size];

            for i in 0..size {
                input_data[i] = i as f32 * 0.3f32;
                let val = input_data[i] / scale as f32;
                expected_data[i] = q_ref_f32(val, zero_point, 0, 255) as u8;
            }

            let input = tf_float.make_default(vec![size as i32], input_data);
            let out = tfo.zeros(vec![size as i32], STATIC);
            let expected = tfo.make_default(vec![size as i32], expected_data);

            quantize_per_tensor_out(
                &input,
                scale,
                zero_point,
                quant_min,
                quant_max,
                ScalarType::Byte,
                &out,
            );

            assert_tensor_eq!(out, expected);
        }
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_large_per_channel_u_int8_simd_path() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let num_channels = 4usize;
        let block_size = 64usize;
        let mut input_data = vec![0.0f32; num_channels * block_size];

        for ch in 0..num_channels {
            for i in 0..block_size {
                input_data[ch * block_size + i] = ((ch + 1) * i) as f32 * 0.1f32;
            }
        }
        let input = tf_float.make_default(
            vec![num_channels as i32, block_size as i32],
            input_data.clone(),
        );

        let scale = tf_double.make_default(vec![num_channels as i32], vec![0.1, 0.2, 0.15, 0.25]);
        let zero_point = tf_long.make_default(vec![num_channels as i32], vec![10, 20, 15, 25]);

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![num_channels as i32, block_size as i32], STATIC);

        let scale_ptr = scale.const_data_ptr::<f64>();
        let zp_ptr = zero_point.const_data_ptr::<i64>();
        let mut expected_data = vec![0u8; num_channels * block_size];
        for ch in 0..num_channels {
            let ch_scale = unsafe { *scale_ptr.add(ch) };
            let ch_zero_point = unsafe { *zp_ptr.add(ch) };
            for i in 0..block_size {
                let idx = ch * block_size + i;
                let val = input_data[idx] / ch_scale as f32;
                let mut qval = val.round_ties_even() as i32 + ch_zero_point as i32;
                qval = core::cmp::min(255, core::cmp::max(0, qval));
                expected_data[idx] = qval as u8;
            }
        }
        let expected =
            tfo.make_default(vec![num_channels as i32, block_size as i32], expected_data);

        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            0,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_large_per_channel_int8_simd_path() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let num_channels = 3usize;
        let block_size = 100usize;
        let mut input_data = vec![0.0f32; num_channels * block_size];

        for ch in 0..num_channels {
            for i in 0..block_size {
                input_data[ch * block_size + i] =
                    (i as i32 - 50) as f32 * 0.2f32 * (ch as f32 + 1.0);
            }
        }
        let input = tf_float.make_default(
            vec![num_channels as i32, block_size as i32],
            input_data.clone(),
        );

        let scale = tf_double.make_default(vec![num_channels as i32], vec![0.1, 0.15, 0.2]);
        let zero_point = tf_long.make_default(vec![num_channels as i32], vec![0, -5, 5]);

        let quant_min: i64 = -128;
        let quant_max: i64 = 127;

        let tfo = TensorFactory::<i8>::new();
        let out = tfo.zeros(vec![num_channels as i32, block_size as i32], STATIC);

        let scale_ptr = scale.const_data_ptr::<f64>();
        let zp_ptr = zero_point.const_data_ptr::<i64>();
        let mut expected_data = vec![0i8; num_channels * block_size];
        for ch in 0..num_channels {
            let ch_scale = unsafe { *scale_ptr.add(ch) };
            let ch_zero_point = unsafe { *zp_ptr.add(ch) };
            for i in 0..block_size {
                let idx = ch * block_size + i;
                let val = input_data[idx] / ch_scale as f32;
                let mut qval = val.round_ties_even() as i32 + ch_zero_point as i32;
                qval = core::cmp::min(127, core::cmp::max(-128, qval));
                expected_data[idx] = qval as i8;
            }
        }
        let expected =
            tfo.make_default(vec![num_channels as i32, block_size as i32], expected_data);

        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            0,
            quant_min,
            quant_max,
            ScalarType::Char,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_very_large_per_channel2d_u_int8() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let num_channels = 128usize;
        let block_size = 256usize;
        let total_elements = num_channels * block_size;

        let mut input_data = vec![0.0f32; total_elements];
        for i in 0..total_elements {
            input_data[i] = (i as f32 * 0.01f32).sin() * 5.0f32;
        }
        let input = tf_float.make_default(
            vec![num_channels as i32, block_size as i32],
            input_data.clone(),
        );

        let mut scales = vec![0.0f64; num_channels];
        let mut zero_points = vec![0i64; num_channels];
        for ch in 0..num_channels {
            scales[ch] = 0.02 + (ch % 10) as f64 * 0.001; // Varying scales
            zero_points[ch] = 128 + (ch % 5) as i64; // Varying zero_points
        }
        let scale = tf_double.make_default(vec![num_channels as i32], scales.clone());
        let zero_point = tf_long.make_default(vec![num_channels as i32], zero_points.clone());

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![num_channels as i32, block_size as i32], STATIC);

        let mut expected_data = vec![0u8; total_elements];
        for ch in 0..num_channels {
            let inv_scale = 1.0f32 / scales[ch] as f32;
            let ch_zero_point = zero_points[ch];
            for i in 0..block_size {
                let idx = ch * block_size + i;
                let mut val = input_data[idx] * inv_scale;
                val = val.max(-1000.0f32).min(1000.0f32);
                let mut qval = val.round_ties_even() as i32 + ch_zero_point as i32;
                qval = core::cmp::min(255, core::cmp::max(0, qval));
                expected_data[idx] = qval as u8;
            }
        }
        let expected =
            tfo.make_default(vec![num_channels as i32, block_size as i32], expected_data);

        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            0,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.native.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_per_channel_axis1_large_blocks() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let batch_size = 2usize;
        let num_channels = 3usize;
        let block_size = 64usize;
        let total_elements = batch_size * num_channels * block_size;

        let mut input_data = vec![0.0f32; total_elements];
        for i in 0..total_elements {
            input_data[i] = (i % 100) as f32 * 0.1f32;
        }
        let input = tf_float.make_default(
            vec![batch_size as i32, num_channels as i32, block_size as i32],
            input_data.clone(),
        );

        let scale = tf_double.make_default(vec![num_channels as i32], vec![0.05, 0.1, 0.15]);
        let zero_point = tf_long.make_default(vec![num_channels as i32], vec![100, 110, 120]);

        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(
            vec![batch_size as i32, num_channels as i32, block_size as i32],
            STATIC,
        );

        let scale_ptr = scale.const_data_ptr::<f64>();
        let zp_ptr = zero_point.const_data_ptr::<i64>();
        let mut expected_data = vec![0u8; total_elements];
        for b in 0..batch_size {
            for ch in 0..num_channels {
                let ch_scale = unsafe { *scale_ptr.add(ch) };
                let ch_zero_point = unsafe { *zp_ptr.add(ch) };
                for i in 0..block_size {
                    let idx = (b * num_channels + ch) * block_size + i;
                    let val = input_data[idx] / ch_scale as f32;
                    let mut qval = val.round_ties_even() as i32 + ch_zero_point as i32;
                    qval = core::cmp::min(255, core::cmp::max(0, qval));
                    expected_data[idx] = qval as u8;
                }
            }
        }
        let expected = tfo.make_default(
            vec![batch_size as i32, num_channels as i32, block_size as i32],
            expected_data,
        );

        quantize_per_channel_out(
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // clamp_scalar is defined by the aarch64 NEON traits but never called by the
    // __aarch64__ quantize_arm branch, so no op-level test exercises it. Pin its
    // literal C++ semantics (min(255, max(0, val)) / min(127, max(-128, val))).
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-uint8-t.clamp-scalar-fn/test]
    // [spec:et:sem:op-quantize.torch.executor.native.neon-quantize-traits-int8-t.clamp-scalar-fn/test]
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_traits_clamp_scalar() {
        use super::neon::NeonQuantizeTraits;
        assert_eq!(<u8 as NeonQuantizeTraits>::clamp_scalar(-1), 0u8);
        assert_eq!(<u8 as NeonQuantizeTraits>::clamp_scalar(0), 0u8);
        assert_eq!(<u8 as NeonQuantizeTraits>::clamp_scalar(200), 200u8);
        assert_eq!(<u8 as NeonQuantizeTraits>::clamp_scalar(255), 255u8);
        assert_eq!(<u8 as NeonQuantizeTraits>::clamp_scalar(300), 255u8);

        assert_eq!(<i8 as NeonQuantizeTraits>::clamp_scalar(-200), -128i8);
        assert_eq!(<i8 as NeonQuantizeTraits>::clamp_scalar(-128), -128i8);
        assert_eq!(<i8 as NeonQuantizeTraits>::clamp_scalar(0), 0i8);
        assert_eq!(<i8 as NeonQuantizeTraits>::clamp_scalar(127), 127i8);
        assert_eq!(<i8 as NeonQuantizeTraits>::clamp_scalar(200), 127i8);
    }

    // The _context wrappers resize out then delegate. Pin quantize_per_channel_out_context
    // resizing an under-sized out and producing the same result as the direct call.
    // [spec:et:sem:op-quantize.torch.executor.quantize-per-channel-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_channel_context() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        let input = tf_float.full(vec![3, 2], 4.0, STATIC);
        let scale = tf_double.make_default(vec![2], vec![0.5, 1.0]);
        let zero_point = tf_long.make_default(vec![2], vec![127, 63]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![3, 2], STATIC);
        // 4 / 0.5 + 127 = 135 ; 4 / 1 + 63 = 67
        let expected = tfo.make_default(vec![3, 2], vec![135, 67, 135, 67, 135, 67]);
        let mut context = default_context();
        quantize_per_channel_out_context(
            &mut context,
            &input,
            &scale,
            &zero_point,
            1,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }

    // [spec:et:sem:op-quantize.torch.executor.quantize-per-token-out-fn/test]
    #[test]
    fn op_quantize_out_test_quantize_per_token() {
        let tf_float = TensorFactory::<f32>::new();
        let tf_double = TensorFactory::<f64>::new();
        let tf_long = TensorFactory::<i64>::new();

        // Shape [2, 3]: 2 tokens, each with its own scale/zero_point (the last dim
        // is the per-token quantized axis). per_token reshapes to [num_tokens, last]
        // and per-channel-quantizes along axis 0.
        let input = tf_float.make_default(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scale = tf_double.make_default(vec![2, 1], vec![0.5, 1.0]);
        let zero_point = tf_long.make_default(vec![2, 1], vec![10, 20]);
        let quant_min: i64 = 0;
        let quant_max: i64 = 255;

        let tfo = TensorFactory::<u8>::new();
        let out = tfo.zeros(vec![2, 3], STATIC);
        // Token 0 (scale 0.5, zp 10): [1/0.5+10, 2/0.5+10, 3/0.5+10] = [12, 14, 16]
        // Token 1 (scale 1.0, zp 20): [4/1+20, 5/1+20, 6/1+20] = [24, 25, 26]
        let expected = tfo.make_default(vec![2, 3], vec![12, 14, 16, 24, 25, 26]);
        quantize_per_token_out(
            &input,
            &scale,
            &zero_point,
            quant_min,
            quant_max,
            ScalarType::Byte,
            &out,
        );

        assert_tensor_eq!(out, expected);
    }
}
