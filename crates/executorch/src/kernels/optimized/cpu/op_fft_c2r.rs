//! Literal port of kernels/optimized/cpu/op_fft_c2r.cpp.
//!
//! DEVIATION: pocketfft -> realfft/rustfft. The C++ issues a single
//! `pocketfft::c2r(out_shape, in_stride, out_stride, axes, false /*inverse*/,
//! ...)` performing the whole N-D complex->real inverse transform (remaining
//! axes c2c inverse, last axis c2r), scaled by `fct`, against the strided
//! tensor buffers. realfft only offers a 1-D inverse real transform over a
//! contiguous buffer, so pocketfft's internal N-D bookkeeping is reproduced
//! explicitly here:
//!   1. inverse c2c FFTs (`rustfft::FftPlanner`, `FftDirection::Inverse`) along
//!      every transformed axis except the last, over the one-sided complex
//!      input,
//!   2. c2r (`realfft::plan_fft_inverse`) along the last transformed axis,
//!      expanding it back to `last_dim_size` real samples,
//!   3. multiply every real output by the normalization factor `fct`.
//! Control flow (checks, resize, last-dim = last_dim_size, ET_SWITCH_FLOAT_TYPES
//! dispatch, `if !fct { return }` bail-out) is kept bug-for-bug; only the
//! pocketfft leaf is substituted.

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use rustfft::{FftDirection, FftPlanner};

use crate::kernels::optimized::cpu::fft_utils::{compute_fct, shape_from_tensor};
use crate::kernels::optimized::cpu::op_fft_r2c::{FftFloat, fft_along_axis};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::util::scalar_type_util::to_complex_type;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::Complex as EtComplex;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn]
// [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn]
pub fn opt_fft_c2r_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: IntArrayRef,
    normalization: i64,
    last_dim_size: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let in_sizes = in_.sizes();
    crate::et_kernel_check!(
        ctx,
        in_.dim() <= K_TENSOR_DIMENSION_LIMIT as isize,
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, !dim.empty(), InvalidArgument, out);
    crate::et_kernel_check!(ctx, last_dim_size >= 1, InvalidArgument, out);

    // Determine the output size
    let mut out_sizes_storage: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let out_sizes_len = in_sizes.size();
    for i in 0..out_sizes_len {
        out_sizes_storage[i] = *in_sizes.at(i);
    }
    // PORT-NOTE (wave-3 fix): the C++ writes `out_sizes[dim.back()]` before the
    // per-dim bounds check below; an out-of-range dim (negative or >= the
    // dimension limit) is UB there but merely a stray stack write in practice,
    // and the function then fails the bounds check with InvalidArgument. Rust
    // indexing would panic instead, so the write is guarded — the bounds check
    // below still rejects the call identically.
    let last_dim = *dim.back() as usize;
    if last_dim < K_TENSOR_DIMENSION_LIMIT {
        out_sizes_storage[last_dim] = last_dim_size as SizesType;
    }

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check_msg!(
        ctx,
        in_.scalar_type() == to_complex_type(out.scalar_type()),
        InvalidArgument,
        out,
        "the input type for _fft_c2r must be the Complex type corresponding to the output type"
    );

    for i in 0..dim.size() {
        let d = *dim.at(i);
        crate::et_kernel_check_msg!(
            ctx,
            d >= 0 && d < in_.dim() as i64,
            InvalidArgument,
            out,
            "dims must be in bounds (got {})",
            d
        );
    }

    crate::et_kernel_check_msg!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::<SizesType>::from_raw_parts(out_sizes_storage.as_ptr(), out_sizes_len)
        ) == Error::Ok,
        InvalidArgument,
        out,
        "Failed to resize output tensor (last dim {}).",
        out_sizes_storage[last_dim]
    );

    // NOTE: as of this writing, upstream PyTorch only supports float/double, so
    // we follow suit.
    match out.scalar_type() {
        ScalarType::Float => c2r_impl::<f32>(ctx, in_, dim, normalization, out),
        ScalarType::Double => c2r_impl::<f64>(ctx, in_, dim, normalization, out),
        _ => {
            crate::et_kernel_check!(ctx, false, InvalidArgument, out);
        }
    }
    out
}

// Body of the ET_SWITCH_FLOAT_TYPES lambda, generic over the output ctype.
fn c2r_impl<T: FftFloat>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: IntArrayRef,
    normalization: i64,
    out: &Tensor,
) {
    // Note the C++ computes fct from the OUTPUT tensor's sizes over `dim`.
    let fct = match compute_fct::<T>(ctx, out, dim, normalization) {
        Some(f) => f,
        // Check failed, just bail out of the lambda.
        None => return,
    };

    // Logical shapes: `in_shape` is the one-sided complex input extent;
    // `out_shape` is the real output extent (last transformed axis expanded to
    // last_dim_size).
    let in_shape = shape_from_tensor(in_);
    let out_shape = shape_from_tensor(out);
    let last_axis = *dim.back() as usize;

    let in_cdata: *const EtComplex<T> = in_.const_data_ptr::<EtComplex<T>>();
    let out_data: *mut T = out.mutable_data_ptr::<T>();

    let in_numel: usize = in_shape.iter().product();

    // Gather the one-sided complex input into a working buffer (rustfft/realfft
    // `Complex<T>`), matching the interleaved etensor::complex<T> layout.
    let mut work: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); in_numel];
    for idx in 0..in_numel {
        let c = unsafe { *in_cdata.add(idx) };
        work[idx] = Complex::new(c.real, c.imag);
    }

    // Step 1: inverse c2c FFTs along every transformed axis except the last,
    // over the one-sided complex data.
    let mut planner = FftPlanner::<T>::new();
    for j in 0..(dim.size() - 1) {
        let axis = *dim.at(j) as usize;
        fft_along_axis(
            &mut planner,
            &mut work,
            &in_shape,
            axis,
            FftDirection::Inverse,
        );
    }

    // Step 2: c2r along the last transformed axis, expanding N/2+1 one-sided
    // complex samples back to `last_dim_size` real samples.
    let n_last_out = out_shape[last_axis]; // == last_dim_size
    // PORT-NOTE (wave-3 fix): pocketfft derives the one-sided input extent from
    // the OUTPUT shape (shape_in[axis] = shape_out[axis]/2 + 1) and reads only
    // that many entries along the axis; the input tensor's extent may be larger
    // (kernels/test/op_fft_c2r_test.cpp drives dim=0 on a {4,3} input with
    // last_dim_size 4, i.e. buffer extent 4 but only 3 entries read). realfft's
    // inverse plan also requires exactly n/2+1 inputs. Read n_last_out/2+1
    // entries while stepping the buffer by its real extent (`in_extent`).
    let n_last_in = n_last_out / 2 + 1;
    let in_extent = in_shape[last_axis];

    let mut in_stride: usize = 1;
    for &s in &in_shape[last_axis + 1..] {
        in_stride *= s;
    }
    let mut out_stride: usize = 1;
    for &s in &out_shape[last_axis + 1..] {
        out_stride *= s;
    }

    let mut real_planner = RealFftPlanner::<T>::new();
    let c2r = real_planner.plan_fft_inverse(n_last_out);
    let mut c2r_in: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); n_last_in];
    let mut c2r_out: Vec<T> = vec![T::zero(); n_last_out];
    let mut c2r_scratch = c2r.make_scratch_vec();

    let in_inner = in_stride;
    let in_outer = (in_numel / in_extent) / in_inner;
    // in_outer == out_outer (extents before last_axis are identical).
    for o in 0..in_outer {
        for i in 0..in_inner {
            let in_base = o * in_extent * in_stride + i;
            for k in 0..n_last_in {
                c2r_in[k] = work[in_base + k * in_stride];
            }
            // realfft flags a non-zero imaginary part at the zero/nyquist bins
            // (FftError::InputValues) but still performs the transform; pocketfft
            // silently used those values. Ignore the flag to match pocketfft.
            let _ = c2r.process_with_scratch(&mut c2r_in, &mut c2r_out, &mut c2r_scratch);
            let out_base = o * n_last_out * out_stride + i;
            for k in 0..n_last_out {
                let scaled = T::scale(c2r_out[k], fct);
                unsafe {
                    *out_data.add(out_base + k * out_stride) = scaled;
                }
            }
        }
    }
}

// Port of kernels/test/op_fft_c2r_test.cpp (OpFftC2rOutTest).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }

    fn test_dtype<T>(norm: i64, dim_v: i64, expect_failure: bool)
    where
        T: FactoryValue + CppTypeToScalarType + FromF64,
        EtComplex<T>: FactoryValue + CppTypeToScalarType,
    {
        let tf_out = TensorFactory::<T>::new();
        let tf_in = TensorFactory::<EtComplex<T>>::new();
        let mk = |re: f64, im: f64| EtComplex::<T> {
            real: T::from_f64(re),
            imag: T::from_f64(im),
        };

        let input_data = vec![
            mk(24.0, 4.0),
            mk(4.0, -8.0),
            mk(0.0, 4.0),
            mk(8.0, -16.0),
            mk(-4.0, 0.0),
            mk(0.0, 32.0),
            mk(12.0, 0.0),
            mk(0.0, 4.0),
            mk(-8.0, 4.0),
            mk(0.0, 8.0),
            mk(-4.0, 8.0),
            mk(8.0, 0.0),
        ];
        let in_ = tf_in.make_default(vec![4, 3], input_data);
        let out = tf_out.full(vec![4, 3], T::from_f64(0.0), TensorShapeDynamism::STATIC);

        let last_dim_size: i64 = if dim_v >= 0 && dim_v < out.dim() as i64 {
            out.size(dim_v as isize) as i64
        } else {
            0
        };
        let dim_data = [dim_v];
        let dim = IntArrayRef::from_raw_parts(dim_data.as_ptr(), 1);

        let mut ctx = context();
        opt_fft_c2r_out(&mut ctx, &in_, dim, norm, last_dim_size, &out);

        if expect_failure {
            assert_ne!(ctx.failure_state(), Error::Ok);
        } else {
            assert_eq!(ctx.failure_state(), Error::Ok);
            let norm_factor: f64 = match norm {
                1 => 2.0,
                2 => 4.0,
                _ => 1.0,
            };
            let expected_data: Vec<T> = [
                52.0, -4.0, -8.0, 44.0, 4.0, -56.0, 20.0, 12.0, -8.0, -20.0, 4.0, 72.0,
            ]
            .iter()
            .map(|&v: &f64| T::from_f64(v / norm_factor))
            .collect();
            assert_tensor_close!(out, tf_out.make_default(vec![4, 3], expected_data));
        }
    }

    fn test_dtype_multiple_axes<T>()
    where
        T: FactoryValue + CppTypeToScalarType + FromF64,
        EtComplex<T>: FactoryValue + CppTypeToScalarType,
    {
        let tf_out = TensorFactory::<T>::new();
        let tf_in = TensorFactory::<EtComplex<T>>::new();
        let mk = |re: f64, im: f64| EtComplex::<T> {
            real: T::from_f64(re),
            imag: T::from_f64(im),
        };

        let input_data = vec![
            mk(16.0, 4.0),
            mk(4.0, -8.0),
            mk(0.0, 4.0),
            mk(8.0, -16.0),
            mk(-4.0, 0.0),
            mk(0.0, 36.0),
            mk(32.0, 0.0),
            mk(0.0, 4.0),
            mk(-8.0, 4.0),
            mk(0.0, 8.0),
            mk(-4.0, 8.0),
            mk(8.0, 0.0),
        ];
        let in_ = tf_in.make_default(vec![4, 3], input_data);
        let out = tf_out.full(vec![4, 4], T::from_f64(0.0), TensorShapeDynamism::STATIC);

        let last_dim_size: i64 = out.size(0) as i64;
        let dim_data = [0i64, 1];
        let dim = IntArrayRef::from_raw_parts(dim_data.as_ptr(), 2);

        let mut ctx = context();
        opt_fft_c2r_out(&mut ctx, &in_, dim, 1, last_dim_size, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);

        let expected_data: Vec<T> = [
            12.0, 12.0, 16.0, 16.0, 1.0, 15.0, -11.0, 3.0, 12.0, 20.0, 0.0, 8.0, -1.0, -15.0, 3.0,
            -27.0,
        ]
        .iter()
        .map(|&v: &f64| T::from_f64(v))
        .collect();
        assert_tensor_close!(out, tf_out.make_default(vec![4, 4], expected_data));
    }

    // [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn/test]
    #[test]
    fn op_fft_c2r_out_test_all_dtypes_supported() {
        test_dtype::<f32>(0, 0, false);
        test_dtype::<f32>(1, 0, false);
        test_dtype::<f32>(2, 0, false);
        test_dtype::<f64>(0, 0, false);
        test_dtype::<f64>(1, 0, false);
        test_dtype::<f64>(2, 0, false);
    }

    // [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn/test]
    #[test]
    fn op_fft_c2r_out_test_multiple_dims() {
        test_dtype_multiple_axes::<f32>();
        test_dtype_multiple_axes::<f64>();
    }

    // [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn/test]
    #[test]
    fn op_fft_c2r_out_test_invalid_norm() {
        test_dtype::<f32>(3, 0, true);
        test_dtype::<f32>(4, 0, true);
        test_dtype::<f32>(-1, 0, true);
        test_dtype::<f32>(9999999, 0, true);
    }

    // [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn/test]
    #[test]
    fn op_fft_c2r_out_test_invalid_dim() {
        test_dtype::<f32>(0, -1, true);
        test_dtype::<f32>(0, 3, true);
        test_dtype::<f32>(0, 9001, true);
    }
}
