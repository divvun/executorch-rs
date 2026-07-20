//! Coarse-grained data parallelism for the divvun-speech host kernels
//! (`op_layer_norm`, `op_istft`).
//!
//! These kernels run host-side between XNNPACK delegate calls and each iterate
//! over an independent outer dimension (LayerNorm rows / ISTFT frames). Spawning
//! fresh OS threads per invocation is far too heavyweight — the kernels are
//! called ~12-24 times per synthesis, so the spawn/join churn dwarfs the work.
//! Instead we reuse the process-wide XNNPACK `pthreadpool` (already warm from the
//! surrounding delegates) via `pthreadpool_parallelize_1d_tile_1d`, which is a
//! no-spawn dispatch onto persistent workers.
//!
//! When the `xnnpack` feature is off there is no pool, so this degrades to a
//! plain serial loop.

/// Run `f(start, count)` over contiguous chunks that together cover
/// `0..range`, in parallel across the shared pool when `range >= threshold` and
/// more than one worker is available. `f` is invoked concurrently on disjoint
/// `[start, start + count)` sub-ranges, so it must only write data private to
/// its own chunk. Below `threshold`, or with a single worker, runs serially as a
/// single `f(0, range)` call (bit-identical to the non-parallel path).
#[inline]
pub(crate) fn chunks<F: Fn(usize, usize) + Sync>(range: usize, threshold: usize, f: F) {
    if range == 0 {
        return;
    }

    #[cfg(feature = "xnnpack")]
    {
        if range >= threshold && !force_serial() {
            let workers = crate::backends::xnnpack::runtime::sys::pthreadpool_threads().min(range);
            if workers > 1 {
                crate::backends::xnnpack::runtime::sys::parallelize_chunks(
                    range,
                    workers,
                    |s, c| f(s, c),
                );
                return;
            }
        }
    }
    #[cfg(not(feature = "xnnpack"))]
    let _ = threshold;

    f(0, range);
}

/// Diagnostic escape hatch: `DIVVUN_KERNEL_SERIAL=1` forces the serial path.
/// Read once and cached.
#[cfg(feature = "xnnpack")]
fn force_serial() -> bool {
    use core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = false, 2 = true
    match CACHE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let v = std::env::var("DIVVUN_KERNEL_SERIAL")
                .map(|s| s == "1" || s == "true")
                .unwrap_or(false);
            CACHE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
            v
        }
    }
}
