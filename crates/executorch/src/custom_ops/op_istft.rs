//! Fused inverse-STFT kernel for the Vocos vocoder — Rust translation of
//! `divvun-speech-rs/wrapper/custom_ops/op_istft.cpp`.
//!
//! Registered as `tts::istft.out`. The exported Vocos `.pte` computes real and
//! imaginary spectrogram tensors (`mag*cos(phase)`, `mag*sin(phase)`) and hands
//! them to this op, which performs a per-frame inverse real FFT, applies a Hann
//! window, overlap-adds, normalizes by the window overlap (COLA), and trims the
//! synthesis padding — matching `torch.istft(..., center=True)`.
//!
//! Schema (mirrors the C++ registration):
//! ```text
//! tts::istft.out(Tensor real, Tensor imag, int n_fft, int hop_length,
//!                int win_length, *, Tensor(a!) out) -> Tensor(a!)
//! ```
//!
//! The inverse real FFT uses `realfft` (a real-valued wrapper over `rustfft`,
//! with runtime SSE/AVX/NEON dispatch), matching the SIMD approach of the C++
//! kernel's pffft. Any `n_fft` is supported (rustfft is mixed-radix); Vocos
//! uses 1024.

use realfft::RealFftPlanner;

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::SizesType;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

/// Operator name as it appears in the exported Vocos `.pte`.
pub(crate) const ISTFT_NAME: &core::ffi::CStr = c"tts::istft.out";

/// Fill `window` with a periodic Hann window: `0.5 * (1 - cos(2*pi*i/length))`.
fn make_hann_window(window: &mut [f32]) {
    let length = window.len() as f64;
    for (i, w) in window.iter_mut().enumerate() {
        *w = (0.5 * (1.0 - (2.0 * core::f64::consts::PI * i as f64 / length).cos())) as f32;
    }
}

/// Minimum frame count (the `chunks` range unit) at which the per-frame FFT
/// stage is worth dispatching across the pool. Parallelize only once the total
/// spectral work (`work_size == n_freqs * n_frames`) exceeds ~32k values; below
/// that the dispatch overhead dominates. Expressed in frames so it matches the
/// range `chunks` iterates over. Returns `usize::MAX` (never parallelize) for a
/// degenerate zero-frame input.
fn par_frame_threshold(work_size: usize, n_frames: usize) -> usize {
    const PAR_MIN_VALUES: usize = 32 * 1024;
    if n_frames == 0 {
        return usize::MAX;
    }
    let n_freqs = work_size / n_frames; // == n_fft/2 + 1
    PAR_MIN_VALUES.div_ceil(n_freqs.max(1))
}

/// ISTFT: `(real, imag) -> audio`.
///
/// `real`/`imag` are `[B, F, T]` float tensors with `F == n_fft/2 + 1`. `out` is
/// resized to `[B, (T-1)*hop_length + n_fft]` and filled with the reconstructed
/// waveform. Returns `out` for signature parity with the C++ kernel.
pub fn istft_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    real: &Tensor,
    imag: &Tensor,
    n_fft: i64,
    hop_length: i64,
    win_length: i64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    // `win_length` is accepted for schema parity with torch.istft / the C++
    // kernel, but (as in the C++ kernel) the synthesis window spans the full
    // n_fft, so it is unused.
    let _ = win_length;

    crate::et_kernel_check!(ctx, real.dim() == 3, InvalidArgument, out);
    crate::et_kernel_check!(ctx, imag.dim() == 3, InvalidArgument, out);
    crate::et_kernel_check!(
        ctx,
        real.scalar_type() == ScalarType::Float,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(
        ctx,
        imag.scalar_type() == ScalarType::Float,
        InvalidArgument,
        out
    );
    crate::et_kernel_check!(ctx, n_fft >= 2, InvalidArgument, out);

    let batch = real.size(0) as i64;
    let n_freqs = real.size(1) as i64;
    let n_frames = real.size(2) as i64;

    crate::et_kernel_check!(ctx, n_freqs == n_fft / 2 + 1, InvalidArgument, out);

    // Output length matches torch.istft with center=True.
    let audio_len = (n_frames - 1) * hop_length + n_fft;

    let out_sizes: [SizesType; 2] = [batch as SizesType, audio_len as SizesType];
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, ArrayRef::from_raw_parts(out_sizes.as_ptr(), 2)) == Error::Ok,
        InvalidArgument,
        out
    );

    let n_fft = n_fft as usize;
    let n_freqs = n_freqs as usize;
    let n_frames = n_frames as usize;
    let audio_len = audio_len as usize;
    let hop_length = hop_length as usize;
    let half_n = n_fft / 2;

    let spectrum_len = batch as usize * n_freqs * n_frames;
    let real_data =
        unsafe { core::slice::from_raw_parts(real.const_data_ptr::<f32>(), spectrum_len) };
    let imag_data =
        unsafe { core::slice::from_raw_parts(imag.const_data_ptr::<f32>(), spectrum_len) };
    let out_data = unsafe {
        core::slice::from_raw_parts_mut(out.mutable_data_ptr::<f32>(), batch as usize * audio_len)
    };

    // Plan the inverse real FFT once. The per-frame inverse FFTs are
    // independent, so they parallelize; the plan (`Arc<dyn ComplexToReal>`) is
    // `Sync + Send`, so parallel chunks share it by reference.
    let c2r = RealFftPlanner::<f32>::new().plan_fft_inverse(n_fft);

    let mut window = vec![0.0f32; n_fft];
    make_hann_window(&mut window);
    // rustfft's inverse is unnormalized; divide by n_fft.
    let inv_n = 1.0f32 / n_fft as f32;

    for b in 0..batch as usize {
        let batch_out = &mut out_data[b * audio_len..(b + 1) * audio_len];
        batch_out.fill(0.0);

        // Accumulated squared window for the COLA normalization.
        let mut window_sum = vec![0.0f32; audio_len];

        let base_bf = b * n_freqs * n_frames;

        // Stage 1 (parallelizable): per-frame inverse FFT into `windowed`, a
        // `n_frames * n_fft` buffer holding each frame already multiplied by
        // `inv_n * window[i]`. This is exactly the `frame_out[i] * inv_n * w`
        // term the overlap-add adds, precomputed independently per frame — so
        // the OLA sum below is bit-identical regardless of parallelism. The
        // work is dispatched across the shared XNNPACK pool in contiguous frame
        // chunks (each chunk owns a disjoint `windowed` slice); below the
        // `par_frame_threshold` frame count it runs serially.
        let mut windowed = vec![0.0f32; n_frames * n_fft];
        let work_size = n_freqs * n_frames;

        let windowed_base = windowed.as_mut_ptr() as usize;
        let fft_ok = std::sync::atomic::AtomicBool::new(true);
        // The realfft plan (`Arc<dyn ComplexToReal>`) is `Sync + Send`, so chunks
        // share it by reference.
        let c2r_ref: &dyn realfft::ComplexToReal<f32> = c2r.as_ref();
        let window_ref = &window;
        let ok_ref = &fft_ok;
        crate::custom_ops::parallel::chunks(n_frames, par_frame_threshold(work_size, n_frames), {
            move |frame_start, frames| {
                // SAFETY: `chunks` hands out disjoint frame ranges, so the
                // `windowed` sub-slice for `[frame_start, frame_start+frames)`
                // never aliases another chunk's.
                let off = frame_start * n_fft;
                let len = frames * n_fft;
                let chunk = unsafe {
                    core::slice::from_raw_parts_mut((windowed_base as *mut f32).add(off), len)
                };
                if !istft_frames(
                    c2r_ref,
                    real_data,
                    imag_data,
                    base_bf,
                    n_freqs,
                    n_frames,
                    n_fft,
                    half_n,
                    window_ref,
                    inv_n,
                    frame_start,
                    frame_start + frames,
                    chunk,
                ) {
                    ok_ref.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });

        if !fft_ok.load(std::sync::atomic::Ordering::Relaxed) {
            ctx.fail(Error::Internal);
            return out;
        }

        // Stage 2 (serial): windowed overlap-add + COLA window accumulation.
        // Cheap (adds only); the overlapping writes make it awkward to
        // parallelize safely and it is not the hot path.
        for t in 0..n_frames {
            let frame = &windowed[t * n_fft..(t + 1) * n_fft];
            let start = t * hop_length;
            let mut i = 0;
            while i < n_fft && start + i < audio_len {
                let w = window[i];
                batch_out[start + i] += frame[i];
                window_sum[start + i] += w * w;
                i += 1;
            }
        }

        // Normalize by the window overlap (COLA condition).
        for i in 0..audio_len {
            if window_sum[i] > 1e-8 {
                batch_out[i] /= window_sum[i];
            }
        }

        // torch.istft(center=True) discards n_fft/2 samples from each end of the
        // raw OLA buffer — synthesis padding where the divide above amplifies
        // near-zero window accumulation into clicks.
        let cold_trim = half_n.min(audio_len);
        for x in batch_out[..cold_trim].iter_mut() {
            *x = 0.0;
        }
        let tail_start = audio_len.saturating_sub(half_n);
        for x in batch_out[tail_start..].iter_mut() {
            *x = 0.0;
        }
    }

    out
}

/// Inverse-FFT a contiguous range of frames `[frame_start, frame_end)` into
/// `windowed_chunk` (length `(frame_end - frame_start) * n_fft`). Each frame is
/// scaled by `inv_n * window[i]` in place, so the caller only has to overlap-add
/// the results. Returns `false` if realfft rejects any frame's input.
///
/// Every frame is independent (its own half-spectrum in, its own `n_fft` samples
/// out), so this runs identically whether called once serially or on disjoint
/// frame ranges across threads.
#[allow(clippy::too_many_arguments)]
fn istft_frames(
    c2r: &dyn realfft::ComplexToReal<f32>,
    real_data: &[f32],
    imag_data: &[f32],
    base_bf: usize,
    n_freqs: usize,
    n_frames: usize,
    n_fft: usize,
    half_n: usize,
    window: &[f32],
    inv_n: f32,
    frame_start: usize,
    frame_end: usize,
    windowed_chunk: &mut [f32],
) -> bool {
    let mut spectrum = c2r.make_input_vec();
    let mut frame_out = c2r.make_output_vec();
    let mut scratch = c2r.make_scratch_vec();

    for (local_t, t) in (frame_start..frame_end).enumerate() {
        for f in 0..n_freqs {
            let idx = base_bf + f * n_frames + t;
            spectrum[f].re = real_data[idx];
            spectrum[f].im = imag_data[idx];
        }
        // The DC bin (and Nyquist bin, for even n_fft) is self-conjugate: its
        // imaginary part must be zero for a real signal. Zero it so realfft does
        // not flag `InputValues`; matches the C++ pffft packing, which used the
        // real part only for those bins.
        spectrum[0].im = 0.0;
        if n_fft % 2 == 0 {
            spectrum[half_n].im = 0.0;
        }

        // `process_with_scratch` uses `spectrum` as scratch (garbage afterwards);
        // it is refilled at the top of each iteration.
        if c2r
            .process_with_scratch(&mut spectrum, &mut frame_out, &mut scratch)
            .is_err()
        {
            return false;
        }

        // Pre-apply `inv_n * window[i]` so the serial OLA can add directly. Same
        // left-associative product as the original `frame_out[i] * inv_n * w`.
        let dst = &mut windowed_chunk[local_t * n_fft..(local_t + 1) * n_fft];
        for i in 0..n_fft {
            dst[i] = frame_out[i] * inv_n * window[i];
        }
    }

    true
}

/// `OpFunction` shim: unpacks the EValue stack and calls [`istft_out`]. The
/// runtime passes the out tensor as the last stack entry; an optional argument
/// may sit between the scalars and `out`, so the stack is 6 or 7 entries.
pub(crate) fn istft_wrapper(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let n = stack.size();
    if !(6..=7).contains(&n) {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let out_idx = if n == 7 { 6 } else { 5 };
    unsafe {
        let real = (*(*stack.index(0))).to_tensor();
        let imag = (*(*stack.index(1))).to_tensor();
        let n_fft = (*(*stack.index(2))).to_int();
        let hop_length = (*(*stack.index(3))).to_int();
        let win_length = (*(*stack.index(4))).to_int();
        let out = (*(*stack.index(out_idx))).to_tensor();
        istft_out(ctx, real, imag, n_fft, hop_length, win_length, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;

    fn context() -> KernelRuntimeContext<'static> {
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    /// Independent closed-form reference: reconstruct each frame with a naive
    /// inverse real FFT (Hermitian symmetry, no library), then window / OLA /
    /// COLA-normalize / cold-trim exactly like [`istft_out`]. Batch size 1,
    /// spectra laid out `[F, T]`. Works for any even `n_fft`.
    fn istft_reference(
        real: &[f32],
        imag: &[f32],
        n_fft: usize,
        hop: usize,
        n_frames: usize,
    ) -> Vec<f32> {
        let pi = core::f64::consts::PI;
        let half_n = n_fft / 2;
        let audio_len = (n_frames - 1) * hop + n_fft;

        let mut window = vec![0.0f32; n_fft];
        make_hann_window(&mut window);

        let mut out = vec![0.0f32; audio_len];
        let mut window_sum = vec![0.0f32; audio_len];
        for t in 0..n_frames {
            let mut frame = vec![0.0f32; n_fft];
            for (i, fv) in frame.iter_mut().enumerate() {
                // DC term (imag == 0).
                let mut s = real[t] as f64;
                // Nyquist term: e^{j*pi*i} == cos(pi*i), imag == 0.
                s += real[half_n * n_frames + t] as f64 * (pi * i as f64).cos();
                for k in 1..half_n {
                    let theta = 2.0 * pi * k as f64 * i as f64 / n_fft as f64;
                    let rr = real[k * n_frames + t] as f64;
                    let ii = imag[k * n_frames + t] as f64;
                    s += 2.0 * (rr * theta.cos() - ii * theta.sin());
                }
                *fv = (s / n_fft as f64) as f32;
            }
            let start = t * hop;
            let mut i = 0;
            while i < n_fft && start + i < audio_len {
                out[start + i] += frame[i] * window[i];
                window_sum[start + i] += window[i] * window[i];
                i += 1;
            }
        }
        for i in 0..audio_len {
            if window_sum[i] > 1e-8 {
                out[i] /= window_sum[i];
            }
        }
        for x in out[..half_n.min(audio_len)].iter_mut() {
            *x = 0.0;
        }
        let tail = audio_len.saturating_sub(half_n);
        for x in out[tail..].iter_mut() {
            *x = 0.0;
        }
        out
    }

    /// Small deterministic LCG producing values in `[-1, 1)`.
    fn pseudo_random_fill(buf: &mut [f32], seed: &mut u32) {
        for v in buf.iter_mut() {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((*seed >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0;
        }
    }

    /// Run `istft_out` for a single-batch `[1, F, T]` input, returning the audio.
    fn run_istft(
        real: &[f32],
        imag: &[f32],
        n_fft: usize,
        hop: usize,
        n_frames: usize,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let audio_len = (n_frames - 1) * hop + n_fft;

        let tf = TensorFactory::<f32>::new();
        let real_t = tf.make_default(vec![1, n_freqs as i32, n_frames as i32], real.to_vec());
        let imag_t = tf.make_default(vec![1, n_freqs as i32, n_frames as i32], imag.to_vec());
        let out_t = tf.zeros(
            vec![1, audio_len as i32],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let mut ctx = context();
        let result = istft_out(
            &mut ctx,
            &real_t,
            &imag_t,
            n_fft as i64,
            hop as i64,
            n_fft as i64,
            &out_t,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        assert_eq!(result.dim(), 2);
        assert_eq!(result.size(1) as usize, audio_len);
        unsafe { core::slice::from_raw_parts(result.const_data_ptr::<f32>(), audio_len).to_vec() }
    }

    fn assert_close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1e-3,
                "sample {i}: got {}, expected {}",
                got[i],
                expected[i]
            );
        }
    }

    #[test]
    fn istft_matches_closed_form_reference() {
        let n_fft = 1024usize; // realistic Vocos size
        let hop = 256usize;
        let n_freqs = n_fft / 2 + 1;
        let n_frames = 6usize;

        let mut real = vec![0.0f32; n_freqs * n_frames];
        let mut imag = vec![0.0f32; n_freqs * n_frames];
        let mut seed = 0x1234_5678u32;
        pseudo_random_fill(&mut real, &mut seed);
        pseudo_random_fill(&mut imag, &mut seed);

        let expected = istft_reference(&real, &imag, n_fft, hop, n_frames);
        let got = run_istft(&real, &imag, n_fft, hop, n_frames);
        assert_close(&got, &expected);
    }

    #[test]
    fn istft_supports_non_power_of_two_n_fft() {
        // rustfft is mixed-radix, so non-power-of-two sizes now work (the old
        // hand-rolled radix-2 required a power of two). 24 = 2^3 * 3.
        let n_fft = 24usize;
        let hop = 6usize;
        let n_freqs = n_fft / 2 + 1;
        let n_frames = 5usize;

        let mut real = vec![0.0f32; n_freqs * n_frames];
        let mut imag = vec![0.0f32; n_freqs * n_frames];
        let mut seed = 0x2222_3333u32;
        pseudo_random_fill(&mut real, &mut seed);
        pseudo_random_fill(&mut imag, &mut seed);

        let expected = istft_reference(&real, &imag, n_fft, hop, n_frames);
        let got = run_istft(&real, &imag, n_fft, hop, n_frames);
        assert_close(&got, &expected);
    }

    #[test]
    fn istft_output_shape_and_cold_trim() {
        let n_fft = 16usize;
        let hop = 4usize;
        let n_freqs = n_fft / 2 + 1;
        let n_frames = 3usize;
        let audio_len = (n_frames - 1) * hop + n_fft;

        let real = vec![1.0f32; n_freqs * n_frames];
        let imag = vec![0.5f32; n_freqs * n_frames];
        let got = run_istft(&real, &imag, n_fft, hop, n_frames);

        assert_eq!(got.len(), audio_len);
        for &x in &got[..n_fft / 2] {
            assert_eq!(x, 0.0);
        }
        for &x in &got[audio_len - n_fft / 2..] {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn istft_wrapper_unpacks_stack() {
        let n_fft = 32usize;
        let hop = 8usize;
        let n_freqs = n_fft / 2 + 1;
        let n_frames = 4usize;
        let audio_len = (n_frames - 1) * hop + n_fft;

        let mut real = vec![0.0f32; n_freqs * n_frames];
        let mut imag = vec![0.0f32; n_freqs * n_frames];
        let mut seed = 0x0bad_c0deu32;
        pseudo_random_fill(&mut real, &mut seed);
        pseudo_random_fill(&mut imag, &mut seed);

        let expected = istft_reference(&real, &imag, n_fft, hop, n_frames);

        let tf = TensorFactory::<f32>::new();
        let real_t = tf.make_default(vec![1, n_freqs as i32, n_frames as i32], real);
        let imag_t = tf.make_default(vec![1, n_freqs as i32, n_frames as i32], imag);
        let out_t = tf.zeros(
            vec![1, audio_len as i32],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        // Build the 6-entry EValue stack: real, imag, n_fft, hop, win, out.
        let mut evalues: Vec<EValue> = vec![
            EValue::from_tensor(real_t),
            EValue::from_tensor(imag_t),
            EValue::from_int(n_fft as i64),
            EValue::from_int(hop as i64),
            EValue::from_int(n_fft as i64),
            EValue::from_tensor(out_t),
        ];
        let mut ptrs: Vec<*mut EValue> = evalues.iter_mut().map(|e| e as *mut EValue).collect();

        let mut ctx = context();
        istft_wrapper(
            &mut ctx,
            Span::from_raw_parts(ptrs.as_mut_ptr(), ptrs.len()),
        );
        assert_eq!(ctx.failure_state(), Error::Ok);

        let out = evalues[5].to_tensor();
        assert_eq!(out.size(1) as usize, audio_len);
        let got = unsafe { core::slice::from_raw_parts(out.const_data_ptr::<f32>(), audio_len) };
        assert_close(got, &expected);
    }

    /// Realistic Vocos-size input (`n_fft = 1024`, hop 256) with enough frames
    /// that `n_freqs * n_frames` crosses the 32k parallel threshold, exercising
    /// the multi-threaded per-frame FFT path. Compared against the independent
    /// closed-form reference. The parallel FFTs write disjoint frame slices and
    /// the overlap-add stays serial, so the reconstruction matches the serial
    /// path.
    #[test]
    fn istft_parallel_path_matches_reference() {
        let n_fft = 1024usize;
        let hop = 256usize;
        let n_freqs = n_fft / 2 + 1; // 513
        let n_frames = 80usize; // 513 * 80 = 41_040 > 32k -> parallel path

        let mut real = vec![0.0f32; n_freqs * n_frames];
        let mut imag = vec![0.0f32; n_freqs * n_frames];
        let mut seed = 0x0f2e_31a7u32;
        pseudo_random_fill(&mut real, &mut seed);
        pseudo_random_fill(&mut imag, &mut seed);

        let expected = istft_reference(&real, &imag, n_fft, hop, n_frames);
        let got = run_istft(&real, &imag, n_fft, hop, n_frames);
        assert_eq!(got.len(), expected.len());
        assert!(
            got.iter().all(|x| !x.is_nan()),
            "no NaNs in parallel output"
        );
        assert_close(&got, &expected);
    }
}
