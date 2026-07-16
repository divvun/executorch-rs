//! Literal port of kernels/optimized/cpu/op_fft_r2c.cpp.
//!
//! DEVIATION: pocketfft -> realfft/rustfft. The C++ issues a single
//! `pocketfft::r2c(in_shape, in_stride, out_stride, axes, forward, ...)` that
//! performs the whole N-D real->complex transform (last axis r2c, remaining
//! axes c2c) directly against the strided tensor buffers, scaled by `fct`.
//! realfft only offers a 1-D real transform over a contiguous buffer, so the
//! N-D bookkeeping pocketfft did internally is reproduced here explicitly:
//!   1. r2c (`realfft::plan_fft_forward`) along the last transformed axis,
//!      yielding the one-sided spectrum (N/2+1) with torch's DC-at-0 layout,
//!   2. forward c2c FFTs (`rustfft::FftPlanner`) along every remaining axis,
//!   3. multiply every complex output by the normalization factor `fct`.
//! The control flow (checks, resize, one-sided last-dim halving, the
//! ET_SWITCH_FLOAT_TYPES dispatch, the `if !fct { return }` bail-out) is kept
//! bug-for-bug; only the pocketfft leaf is substituted.

use realfft::RealFftPlanner;
use realfft::num_complex::Complex;
use rustfft::{FftDirection, FftPlanner};

use crate::kernels::optimized::cpu::fft_utils::{ComputeFct, compute_fct, shape_from_tensor};
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

// realfft's `FftNum` bound is satisfied by f32/f64. This trait glues the ET
// float ctype to the realfft/rustfft primitive scalar and the ET complex ctype
// so the switch body stays generic without a `where` explosion at the call
// site.
pub trait FftFloat: ComputeFct + realfft::FftNum + ScaleBy {}
impl FftFloat for f32 {}
impl FftFloat for f64 {}

// The C++ passes `*fct` to pocketfft, which MULTIPLIES the transform output by
// it (fct already encodes 1, 1/n, or 1/sqrt(n)).
pub trait ScaleBy: Sized {
    fn scale(v: Self, fct: Self) -> Self;
}
impl ScaleBy for f32 {
    fn scale(v: Self, fct: Self) -> Self {
        v * fct
    }
}
impl ScaleBy for f64 {
    fn scale(v: Self, fct: Self) -> Self {
        v * fct
    }
}

// Forward c2c FFT along logical axis `axis` of a contiguous row-major complex
// buffer with the given `shape`. Mirrors the c2c passes pocketfft folds into
// its single r2c call for the non-last transformed axes.
pub(crate) fn fft_along_axis<T: realfft::FftNum>(
    planner: &mut FftPlanner<T>,
    data: &mut [Complex<T>],
    shape: &[usize],
    axis: usize,
    direction: FftDirection,
) {
    let n = shape[axis];
    if n <= 1 {
        return;
    }
    let fft = planner.plan_fft(n, direction);

    // stride of `axis` in the contiguous row-major buffer.
    let mut stride: usize = 1;
    for &s in &shape[axis + 1..] {
        stride *= s;
    }
    // number of independent lines along `axis`: product of all other extents.
    let total: usize = shape.iter().product();
    let num_lines = total / n;

    let mut line: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); n];
    let mut scratch: Vec<Complex<T>> =
        vec![Complex::new(T::zero(), T::zero()); fft.get_inplace_scratch_len()];

    // Enumerate the base offset of every line: iterate all index tuples with
    // `axis` fixed at 0. `outer` groups indices before `axis`; `inner` after.
    let inner = stride; // product of extents after `axis`
    let outer = num_lines / inner; // product of extents before `axis`
    for o in 0..outer {
        for i in 0..inner {
            let base = o * n * stride + i;
            for k in 0..n {
                line[k] = data[base + k * stride];
            }
            fft.process_with_scratch(&mut line, &mut scratch);
            for k in 0..n {
                data[base + k * stride] = line[k];
            }
        }
    }
}

// [spec:et:def:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn]
// [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn]
pub fn opt_fft_r2c_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: IntArrayRef,
    normalization: i64,
    onesided: bool,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    let in_sizes = in_.sizes();
    crate::et_kernel_check!(
        ctx,
        in_.dim() <= K_TENSOR_DIMENSION_LIMIT as isize,
        InvalidArgument,
        out
    );

    let mut out_sizes_storage: [SizesType; K_TENSOR_DIMENSION_LIMIT] =
        [0; K_TENSOR_DIMENSION_LIMIT];
    let out_sizes_len = in_sizes.size();
    for i in 0..out_sizes_len {
        out_sizes_storage[i] = *in_sizes.at(i);
    }
    crate::et_kernel_check!(ctx, !dim.empty(), InvalidArgument, out);

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check_msg!(
        ctx,
        onesided,
        InvalidArgument,
        out,
        "onesided=False is not supported yet in _fft_r2c"
    );

    crate::et_kernel_check_msg!(
        ctx,
        out.scalar_type() == to_complex_type(in_.scalar_type()),
        InvalidArgument,
        out,
        "the output type for _fft_r2c must be the Complex type corresponding to the input type"
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

    let last_dim = *dim.back() as usize;
    if onesided {
        out_sizes_storage[last_dim] = out_sizes_storage[last_dim] / 2 + 1;
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
    match in_.scalar_type() {
        ScalarType::Float => r2c_impl::<f32>(ctx, in_, dim, normalization, out),
        ScalarType::Double => r2c_impl::<f64>(ctx, in_, dim, normalization, out),
        _ => {
            crate::et_kernel_check!(ctx, false, InvalidArgument, out);
        }
    }
    out
}

// Body of the ET_SWITCH_FLOAT_TYPES lambda, generic over the input ctype.
fn r2c_impl<T: FftFloat>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    dim: IntArrayRef,
    normalization: i64,
    out: &Tensor,
) {
    let fct = match compute_fct::<T>(ctx, in_, dim, normalization) {
        Some(f) => f,
        // Check failed, just bail out of the lambda.
        None => return,
    };

    // Logical shapes: `in_shape` is the real input extent; the complex output
    // extent halves the last transformed axis.
    let in_shape = shape_from_tensor(in_);
    let mut out_shape = in_shape.clone();
    let last_axis = *dim.back() as usize;
    out_shape[last_axis] = in_shape[last_axis] / 2 + 1;

    let in_data: *const T = in_.const_data_ptr::<T>();
    let out_cdata: *mut EtComplex<T> = out.mutable_data_ptr::<EtComplex<T>>();

    let in_numel: usize = in_shape.iter().product();
    let out_numel: usize = out_shape.iter().product();

    // Working complex buffer for the out extent (rustfft/realfft `Complex<T>`).
    let mut work: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); out_numel];

    // Step 1: r2c along the last transformed axis. For every line along
    // `last_axis`, run realfft's forward real transform producing N/2+1
    // one-sided complex samples (DC at index 0, torch layout).
    let n_last_in = in_shape[last_axis];
    let n_last_out = out_shape[last_axis];

    // strides (contiguous row-major) for input and output extents.
    let mut in_stride: usize = 1;
    for &s in &in_shape[last_axis + 1..] {
        in_stride *= s;
    }
    let mut out_stride: usize = 1;
    for &s in &out_shape[last_axis + 1..] {
        out_stride *= s;
    }

    let mut real_planner = RealFftPlanner::<T>::new();
    let r2c = real_planner.plan_fft_forward(n_last_in);
    let mut r2c_in: Vec<T> = vec![T::zero(); n_last_in];
    let mut r2c_out: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); n_last_out];
    let mut r2c_scratch = r2c.make_scratch_vec();

    // Enumerate lines: outer indices before `last_axis`, inner after.
    let in_inner = in_stride;
    let in_outer = (in_numel / n_last_in) / in_inner;
    // in_outer == out_outer (extents before last_axis are identical).
    for o in 0..in_outer {
        for i in 0..in_inner {
            let in_base = o * n_last_in * in_stride + i;
            for k in 0..n_last_in {
                r2c_in[k] = unsafe { *in_data.add(in_base + k * in_stride) };
            }
            // process_with_scratch treats input as scratch (garbage after).
            r2c.process_with_scratch(&mut r2c_in, &mut r2c_out, &mut r2c_scratch)
                .expect("realfft r2c");
            let out_base = o * n_last_out * out_stride + i;
            for k in 0..n_last_out {
                work[out_base + k * out_stride] = r2c_out[k];
            }
        }
    }

    // Step 2: forward c2c FFTs along every remaining transformed axis (all of
    // `dim` except the last).
    let mut planner = FftPlanner::<T>::new();
    for j in 0..(dim.size() - 1) {
        let axis = *dim.at(j) as usize;
        fft_along_axis(
            &mut planner,
            &mut work,
            &out_shape,
            axis,
            FftDirection::Forward,
        );
    }

    // Step 3: apply the normalization factor and scatter into the ET complex
    // output buffer (interleaved re/im, matching etensor::complex<T>).
    for idx in 0..out_numel {
        let v = work[idx];
        unsafe {
            *out_cdata.add(idx) = EtComplex {
                real: T::scale(v.re, fct),
                imag: T::scale(v.im, fct),
            };
        }
    }
}

// Port of kernels/test/op_fft_r2c_test.cpp (OpFftR2cOutTest).
#[cfg(test)]
mod tests {
    use super::*;
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
        fn to_f64(self) -> f64;
        // EXPECT_TENSOR_CLOSE-like absolute tolerance for the dtype. The C++
        // compares pocketfft output bitwise (complex falls into the memcmp
        // branch of tensors_are_close); the realfft/rustfft substitution is
        // numerically but not bitwise identical, so complex elements are
        // compared with a per-dtype tolerance instead.
        fn tol() -> f64;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
        fn to_f64(self) -> f64 {
            self as f64
        }
        fn tol() -> f64 {
            1e-4
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
        fn to_f64(self) -> f64 {
            self
        }
        fn tol() -> f64 {
            1e-10
        }
    }

    fn assert_complex_close<T>(out: &Tensor, expected: &[(f64, f64)])
    where
        T: FromF64,
    {
        assert_eq!(out.numel() as usize, expected.len());
        let p = out.const_data_ptr::<EtComplex<T>>();
        for (i, &(re, im)) in expected.iter().enumerate() {
            let got = unsafe { *p.add(i) };
            assert!(
                (got.real.to_f64() - re).abs() <= T::tol(),
                "element {} real: {} vs {}",
                i,
                got.real.to_f64(),
                re
            );
            assert!(
                (got.imag.to_f64() - im).abs() <= T::tol(),
                "element {} imag: {} vs {}",
                i,
                got.imag.to_f64(),
                im
            );
        }
    }

    fn test_dtype<T>(norm: i64, dim_v: i64, onesided: bool, expect_failure: bool)
    where
        T: FactoryValue + CppTypeToScalarType + FromF64,
        EtComplex<T>: FactoryValue + CppTypeToScalarType,
    {
        let tf = TensorFactory::<T>::new();
        let tf_out = TensorFactory::<EtComplex<T>>::new();
        let mk = |re: f64, im: f64| EtComplex::<T> {
            real: T::from_f64(re),
            imag: T::from_f64(im),
        };

        let in_data: Vec<T> = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
            .iter()
            .map(|&v: &f64| T::from_f64(v))
            .collect();
        let in_ = tf.make_default(vec![2, 4], in_data);
        let out = tf_out.full(vec![2, 3], mk(0.0, 0.0), TensorShapeDynamism::STATIC);

        let dim_data = [dim_v];
        let dim = IntArrayRef::from_raw_parts(dim_data.as_ptr(), 1);

        let mut ctx = context();
        opt_fft_r2c_out(&mut ctx, &in_, dim, norm, onesided, &out);

        if expect_failure {
            assert_ne!(ctx.failure_state(), Error::Ok);
        } else {
            assert_eq!(ctx.failure_state(), Error::Ok);
            let norm_factor: f64 = match norm {
                1 => 2.0,
                2 => 4.0,
                _ => 1.0,
            };
            let expected_data: Vec<(f64, f64)> = [
                (6.0, 0.0),
                (-2.0, 2.0),
                (-2.0, 0.0),
                (6.0, 0.0),
                (-2.0, 2.0),
                (-2.0, 0.0),
            ]
            .iter()
            .map(|&(re, im)| (re / norm_factor, im / norm_factor))
            .collect();
            assert_complex_close::<T>(&out, &expected_data);
        }
    }

    fn test_dtype_multiple_axes<T>()
    where
        T: FactoryValue + CppTypeToScalarType + FromF64,
        EtComplex<T>: FactoryValue + CppTypeToScalarType,
    {
        let tf = TensorFactory::<T>::new();
        let tf_out = TensorFactory::<EtComplex<T>>::new();
        let mk = |re: f64, im: f64| EtComplex::<T> {
            real: T::from_f64(re),
            imag: T::from_f64(im),
        };

        let in_data: Vec<T> = [
            0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 2.0, 3.0, 0.0, 1.0, 1.0, 2.0, 3.0, 0.0,
        ]
        .iter()
        .map(|&v: &f64| T::from_f64(v))
        .collect();
        let in_ = tf.make_default(vec![4, 4], in_data);
        let out = tf_out.full(vec![4, 3], mk(0.0, 0.0), TensorShapeDynamism::STATIC);

        let dim_data = [0i64, 1];
        let dim = IntArrayRef::from_raw_parts(dim_data.as_ptr(), 2);

        let mut ctx = context();
        opt_fft_r2c_out(&mut ctx, &in_, dim, 0, true, &out);
        assert_eq!(ctx.failure_state(), Error::Ok);

        let expected_data: [(f64, f64); 12] = [
            (24.0, 0.0),
            (0.0, -4.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (-4.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 4.0),
            (-8.0, 0.0),
            (0.0, 0.0),
            (-4.0, 8.0),
            (0.0, 0.0),
        ];
        assert_complex_close::<T>(&out, &expected_data);
    }

    // [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn/test]
    #[test]
    fn op_fft_r2c_out_test_all_dtypes_supported() {
        test_dtype::<f32>(0, 1, true, false);
        test_dtype::<f32>(1, 1, true, false);
        test_dtype::<f32>(2, 1, true, false);
        test_dtype::<f64>(0, 1, true, false);
        test_dtype::<f64>(1, 1, true, false);
        test_dtype::<f64>(2, 1, true, false);
    }

    // [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn/test]
    #[test]
    fn op_fft_r2c_out_test_multiple_dims() {
        test_dtype_multiple_axes::<f32>();
        test_dtype_multiple_axes::<f64>();
    }

    // [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn/test]
    #[test]
    fn op_fft_r2c_out_test_invalid_norm() {
        test_dtype::<f32>(3, 1, true, true);
        test_dtype::<f32>(4, 1, true, true);
        test_dtype::<f32>(-1, 1, true, true);
        test_dtype::<f32>(9999999, 1, true, true);
    }

    // [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn/test]
    #[test]
    fn op_fft_r2c_out_test_invalid_dim() {
        test_dtype::<f32>(0, -1, true, true);
        test_dtype::<f32>(0, 3, true, true);
        test_dtype::<f32>(0, 9001, true, true);
    }

    // TODO(from the C++ test): support two-sided output.
    // [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn/test]
    #[test]
    fn op_fft_r2c_out_test_two_sided_is_not_supported() {
        test_dtype::<f64>(0, 1, false, true);
    }
}
