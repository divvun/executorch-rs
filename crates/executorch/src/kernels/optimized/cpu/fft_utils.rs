//! Literal port of kernels/optimized/cpu/fft_utils.h.
//!
//! DEVIATION: pocketfft -> realfft/rustfft. The C++ helpers `stride_from_tensor`
//! / `tensor_cdata` exist purely to feed pocketfft's strided N-D transform (byte
//! strides + a `std::complex<T>*` view over the tensor buffer). The realfft /
//! rustfft substitution operates on plain contiguous `[T]` / `[Complex<T>]`
//! slices, so those two pocketfft-shaped helpers have no analogue here; the
//! op-level packing (op_fft_r2c.rs / op_fft_c2r.rs) does the gather/scatter
//! against the tensor buffer directly. `shape_from_tensor` and the
//! normalization (`fft_norm_mode` / `compute_fct`) bookkeeping port literally
//! because they are Tensor-level (not pocketfft-level) and the ops depend on
//! them verbatim.

use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:fft-utils.torch.executor.native.shape-from-tensor-fn]
// [spec:et:sem:fft-utils.torch.executor.native.shape-from-tensor-fn]
// Build a pocketfft `shape_t` (a `Vec<usize>`) from a tensor's sizes: copy the
// tensor's `sizes()` element-for-element into a length vector. Here it stands in
// as the logical N-D extent used to drive the realfft/rustfft transforms.
pub fn shape_from_tensor(t: &Tensor) -> Vec<usize> {
    let sizes = t.sizes();
    let mut shape = Vec::with_capacity(sizes.size());
    for i in 0..sizes.size() {
        shape.push(*sizes.at(i) as usize);
    }
    shape
}

// [spec:et:def:fft-utils.torch.executor.native.fft-norm-mode]
// Normalization selector shared with ATen/native/SpectralOpsUtils.h. The
// `normalization` int64 argument to the ops is one of these discriminants.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum FftNormMode {
    /// No normalization.
    None = 0,
    /// Divide by sqrt(signal_size).
    ByRootN = 1,
    /// Divide by signal_size.
    ByN = 2,
}

// PORT-NOTE: the C++ `static_cast<fft_norm_mode>(normalization)` casts the raw
// int64 to the enum, then `switch`es over the three named values, falling
// through to the ET_KERNEL_CHECK failure for anything else. Rust cannot cast an
// arbitrary int64 to a `#[repr(i64)]` enum without UB, so the discriminant is
// matched directly and non-{0,1,2} values fall through exactly like an
// unhandled `switch` case (returning `None` / failing the check at the call
// site).

// [spec:et:def:fft-utils.torch.executor.native.compute-fct-fn]
// [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn]
// Compute the scalar normalization factor for a transform over a signal of
// `size` samples, given the `normalization` mode:
//   none      -> 1
//   by_n      -> 1 / size
//   by_root_n -> 1 / sqrt(size)
// For any other `normalization` value, record InvalidArgument on `ctx` and
// return None (the C++ `ET_KERNEL_CHECK_MSG(..., std::nullopt, ...)` bail-out).
pub trait ComputeFct: Copy {
    fn one() -> Self;
    fn from_i64(v: i64) -> Self;
    fn div(a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
}

impl ComputeFct for f32 {
    fn one() -> Self {
        1.0
    }
    fn from_i64(v: i64) -> Self {
        v as f32
    }
    fn div(a: Self, b: Self) -> Self {
        a / b
    }
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
}

impl ComputeFct for f64 {
    fn one() -> Self {
        1.0
    }
    fn from_i64(v: i64) -> Self {
        v as f64
    }
    fn div(a: Self, b: Self) -> Self {
        a / b
    }
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
}

pub fn compute_fct_size<T: ComputeFct>(
    ctx: &mut KernelRuntimeContext,
    size: i64,
    normalization: i64,
) -> Option<T> {
    let one = T::one();
    match normalization {
        // fft_norm_mode::none
        0 => Some(one),
        // fft_norm_mode::by_n
        2 => Some(T::div(one, T::from_i64(size))),
        // fft_norm_mode::by_root_n
        1 => Some(T::div(one, T::from_i64(size).sqrt())),
        _ => {
            crate::et_kernel_check_msg!(
                ctx,
                false,
                InvalidArgument,
                None,
                "Unsupported normalization type: {}",
                normalization
            );
            #[allow(unreachable_code)]
            None
        }
    }
}

// [spec:et:def:fft-utils.torch.executor.native.compute-fct-fn]
// [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn]
// Tensor overload: if normalization is none, return 1 immediately. Otherwise
// multiply the sizes of `t` at each axis in `dim` to get the signal size `n`,
// then defer to `compute_fct_size`.
pub fn compute_fct<T: ComputeFct>(
    ctx: &mut KernelRuntimeContext,
    t: &Tensor,
    dim: IntArrayRef,
    normalization: i64,
) -> Option<T> {
    // fft_norm_mode::none
    if normalization == 0 {
        return Some(T::one());
    }
    let sizes = t.sizes();
    let mut n: i64 = 1;
    for i in 0..dim.size() {
        let idx = *dim.at(i);
        n *= *sizes.at(idx as usize) as i64;
    }
    compute_fct_size::<T>(ctx, n, normalization)
}

// SUBSUMED (rust/PORTING.md optimized-kernels DEVIATION): `stride_from_tensor` / `tensor_cdata` fed pocketfft's strided N-D transform; the realfft/rustfft port drives contiguous per-axis transforms and needs neither.
// [spec:et:def:fft-utils.torch.executor.native.stride-from-tensor-fn]
// [spec:et:sem:fft-utils.torch.executor.native.stride-from-tensor-fn]
// [spec:et:def:fft-utils.torch.executor.native.tensor-cdata-fn]
// [spec:et:sem:fft-utils.torch.executor.native.tensor-cdata-fn]

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    // [spec:et:sem:fft-utils.torch.executor.native.shape-from-tensor-fn/test]
    #[test]
    fn fft_utils_shape_from_tensor() {
        let tf = TensorFactory::<f32>::new();
        let t = tf.zeros_default(vec![2, 3, 4]);
        assert_eq!(shape_from_tensor(&t), vec![2usize, 3, 4]);

        let vec1d = tf.make_default(vec![5], vec![0.0; 5]);
        assert_eq!(shape_from_tensor(&vec1d), vec![5usize]);

        // 0-D (scalar) tensor: empty sizes -> empty shape.
        let scalar = tf.make_default(vec![], vec![0.0]);
        assert_eq!(shape_from_tensor(&scalar), Vec::<usize>::new());
    }

    // [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn/test]
    #[test]
    fn fft_utils_compute_fct_size_modes() {
        let mut ctx = context();
        // none -> 1, by_n (2) -> 1/size, by_root_n (1) -> 1/sqrt(size).
        assert_eq!(compute_fct_size::<f32>(&mut ctx, 10, 0), Some(1.0));
        assert_eq!(compute_fct_size::<f32>(&mut ctx, 10, 2), Some(0.1));
        assert_eq!(compute_fct_size::<f32>(&mut ctx, 16, 1), Some(0.25));
        assert_eq!(compute_fct_size::<f64>(&mut ctx, 10, 0), Some(1.0));
        assert_eq!(compute_fct_size::<f64>(&mut ctx, 8, 2), Some(0.125));
        let got = compute_fct_size::<f64>(&mut ctx, 2, 1).unwrap();
        assert!((got - 1.0 / libm::sqrt(2.0)).abs() <= 1e-12, "got {got}");
        assert_eq!(ctx.failure_state(), Error::Ok);
    }

    // [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn/test]
    #[test]
    fn fft_utils_compute_fct_invalid_normalization() {
        // Out-of-range normalization: InvalidArgument recorded on ctx, None
        // returned (the ET_KERNEL_CHECK_MSG bail-out).
        let mut ctx = context();
        assert_eq!(compute_fct_size::<f32>(&mut ctx, 4, 3), None);
        assert_eq!(ctx.failure_state(), Error::InvalidArgument);

        let mut ctx = context();
        assert_eq!(compute_fct_size::<f64>(&mut ctx, 4, -1), None);
        assert_eq!(ctx.failure_state(), Error::InvalidArgument);
    }

    // [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn/test]
    #[test]
    fn fft_utils_compute_fct_tensor_overload() {
        let tf = TensorFactory::<f32>::new();
        let t = tf.zeros_default(vec![3, 4, 5]);
        let dims: [i64; 2] = [1, 2];
        let dim = crate::runtime::core::array_ref::make_array_ref_from_raw_parts(
            dims.as_ptr(),
            dims.len(),
        );

        let mut ctx = context();
        // by_n over dims {1,2}: n = 4*5 = 20 -> 1/20.
        assert_eq!(compute_fct::<f32>(&mut ctx, &t, dim, 2), Some(0.05));
        // by_root_n -> 1/sqrt(20).
        let got = compute_fct::<f32>(&mut ctx, &t, dim, 1).unwrap();
        assert!((got - 1.0 / libm::sqrtf(20.0)).abs() <= 1e-7, "got {got}");
        // none returns 1 immediately, before dim is consulted (empty dim ok).
        let empty = IntArrayRef::new();
        assert_eq!(compute_fct::<f32>(&mut ctx, &t, empty, 0), Some(1.0));
        assert_eq!(ctx.failure_state(), Error::Ok);

        // Invalid normalization flows through to the size-overload failure.
        let mut ctx = context();
        assert_eq!(compute_fct::<f64>(&mut ctx, &t, dim, 5), None);
        assert_eq!(ctx.failure_state(), Error::InvalidArgument);
    }
}
