//! Custom LayerNorm kernel for the Vocos vocoder — Rust translation of
//! `divvun-speech-rs/wrapper/custom_ops/op_layer_norm.cpp`.
//!
//! Registered as `tts::layer_norm.out`. XNNPACK does not support LayerNorm, so
//! the exported graph falls back to this op for the "gaps" between XNNPACK
//! subgraphs. It normalizes over the last `len(normalized_shape)` dimensions:
//! `y = (x - mean) / sqrt(var + eps) * weight + bias`.
//!
//! Schema (mirrors the C++ registration):
//! ```text
//! tts::layer_norm.out(Tensor input, int[] normalized_shape, Tensor? weight,
//!                     Tensor? bias, float eps, *, Tensor(a!) out) -> Tensor(a!)
//! ```

use crate::runtime::core::array_ref::IntArrayRef;
use crate::runtime::core::error::Error;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::exec_aten::util::tensor_util::resize_tensor;
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

/// Operator name as it appears in the exported Vocos `.pte`.
pub(crate) const LAYER_NORM_NAME: &core::ffi::CStr = c"tts::layer_norm.out";

/// LayerNorm: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
///
/// `input` is `[*, normalized_shape]`; normalization runs over the last
/// `normalized_shape.size()` dimensions. `out` is resized to match `input`.
/// Returns `out` for signature parity with the C++ kernel.
pub fn layer_norm_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    input: &Tensor,
    normalized_shape: IntArrayRef,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        input.scalar_type() == ScalarType::Float,
        InvalidArgument,
        out
    );

    // Normalization size = product of normalized_shape.
    let mut norm_size: i64 = 1;
    for i in 0..normalized_shape.size() {
        norm_size *= *normalized_shape.at(i);
    }
    // Guard against a zero-sized normalized dim (integer div-by-zero would panic
    // in Rust, unlike the C++ UB).
    crate::et_kernel_check!(ctx, norm_size > 0, InvalidArgument, out);

    // Number of independent normalizations (everything before the normalized dims).
    let num_instances = input.numel() as i64 / norm_size;

    // Resize output to match input.
    crate::et_kernel_check!(
        ctx,
        resize_tensor(out, input.sizes()) == Error::Ok,
        InvalidArgument,
        out
    );

    let numel = input.numel() as usize;
    let in_data = unsafe { core::slice::from_raw_parts(input.const_data_ptr::<f32>(), numel) };
    let out_data = unsafe { core::slice::from_raw_parts_mut(out.mutable_data_ptr::<f32>(), numel) };

    let norm_size = norm_size as usize;
    let gamma: Option<&[f32]> = weight
        .map(|w| unsafe { core::slice::from_raw_parts(w.const_data_ptr::<f32>(), norm_size) });
    let beta: Option<&[f32]> =
        bias.map(|b| unsafe { core::slice::from_raw_parts(b.const_data_ptr::<f32>(), norm_size) });

    let eps_f = eps as f32;
    let num_instances = num_instances as usize;

    // Each of the `num_instances` rows normalizes over its own `norm_size`
    // window and is fully independent, so the row loop parallelizes cleanly over
    // contiguous row chunks (each chunk writes a disjoint output slice). Below
    // ~32k total elements the dispatch overhead is not worth it, so
    // `parallel::chunks` runs a single serial pass — bit-identical to the
    // pre-parallel path (parallelism only reorders whole rows, never the
    // arithmetic within a row). The threshold is expressed in rows so it tracks
    // the actual work (`num_instances * norm_size == numel`).
    const PAR_MIN_ELEMS: usize = 32 * 1024;
    let row_threshold = PAR_MIN_ELEMS.div_ceil(norm_size.max(1));

    let in_base = in_data.as_ptr() as usize;
    let out_base = out_data.as_mut_ptr() as usize;
    crate::custom_ops::parallel::chunks(num_instances, row_threshold, |row_start, rows| {
        // SAFETY: `chunks` hands out disjoint `[row_start, row_start+rows)`
        // ranges, so the sub-slices below never alias across concurrent calls.
        // Reconstructing them from the base addresses (rather than moving the
        // borrows) is what lets the closure be `Sync`.
        let off = row_start * norm_size;
        let len = rows * norm_size;
        let in_chunk =
            unsafe { core::slice::from_raw_parts((in_base as *const f32).add(off), len) };
        let out_chunk =
            unsafe { core::slice::from_raw_parts_mut((out_base as *mut f32).add(off), len) };
        layer_norm_rows(in_chunk, out_chunk, norm_size, gamma, beta, eps_f);
    });

    out
}

/// Normalize a contiguous block of rows in place: for each `norm_size`-element
/// row of `in_data`, write `(x - mean) / sqrt(var + eps) * gamma + beta` to the
/// matching row of `out_data`. `in_data`/`out_data` hold the same number of
/// whole rows. Keeps the exact 3-pass f32 mean/var/affine math so results are
/// bit-identical to the serial path (and to the unit-test reference).
fn layer_norm_rows(
    in_data: &[f32],
    out_data: &mut [f32],
    norm_size: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps_f: f32,
) {
    let norm_size_f = norm_size as f32;
    for (x, y) in in_data
        .chunks_exact(norm_size)
        .zip(out_data.chunks_exact_mut(norm_size))
    {
        // Pass 1: mean.
        let mut sum = 0.0f32;
        for &v in x {
            sum += v;
        }
        let mean = sum / norm_size_f;

        // Pass 2: variance.
        let mut var_sum = 0.0f32;
        for &v in x {
            let diff = v - mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0f32 / (var_sum / norm_size_f + eps_f).sqrt();

        // Pass 3: normalize + scale + shift.
        match (gamma, beta) {
            (Some(g), Some(b)) => {
                for j in 0..norm_size {
                    y[j] = g[j] * ((x[j] - mean) * inv_std) + b[j];
                }
            }
            (Some(g), None) => {
                for j in 0..norm_size {
                    y[j] = g[j] * ((x[j] - mean) * inv_std);
                }
            }
            (None, Some(b)) => {
                for j in 0..norm_size {
                    y[j] = (x[j] - mean) * inv_std + b[j];
                }
            }
            (None, None) => {
                for j in 0..norm_size {
                    y[j] = (x[j] - mean) * inv_std;
                }
            }
        }
    }
}

/// `OpFunction` shim: unpacks the EValue stack and calls [`layer_norm_out`].
/// Stack layout: `input, normalized_shape, weight?, bias?, eps, [opt], out`
/// (6 or 7 entries; the out tensor is always the last entry). `weight`/`bias`
/// may be `None`.
pub(crate) fn layer_norm_wrapper(ctx: &mut KernelRuntimeContext, stack: Span<*mut EValue>) {
    let n = stack.size();
    if !(6..=7).contains(&n) {
        ctx.fail(Error::InvalidArgument);
        return;
    }
    let out_idx = n - 1;
    unsafe {
        let input = (*(*stack.index(0))).to_tensor();
        let normalized_shape = (*(*stack.index(1))).to_int_list();
        let weight_ev = &*(*stack.index(2));
        let bias_ev = &*(*stack.index(3));
        let weight = if weight_ev.is_none() {
            None
        } else {
            Some(weight_ev.to_tensor())
        };
        let bias = if bias_ev.is_none() {
            None
        } else {
            Some(bias_ev.to_tensor())
        };
        let eps = (*(*stack.index(4))).to_double();
        let out = (*(*stack.index(out_idx))).to_tensor();
        layer_norm_out(ctx, input, normalized_shape, weight, bias, eps, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::array_ref::ArrayRef;
    use crate::runtime::core::evalue::BoxedEvalueList;
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

    /// Independent reference: normalize each instance over the last `norm_size`
    /// elements, matching [`layer_norm_out`]'s f32 mean/var/affine math.
    fn layer_norm_reference(
        input: &[f32],
        norm_size: usize,
        weight: Option<&[f32]>,
        bias: Option<&[f32]>,
        eps: f32,
    ) -> Vec<f32> {
        let n = norm_size as f32;
        let mut out = vec![0.0f32; input.len()];
        for (inst_in, inst_out) in input.chunks(norm_size).zip(out.chunks_mut(norm_size)) {
            let mean: f32 = inst_in.iter().sum::<f32>() / n;
            let var: f32 = inst_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv_std = 1.0f32 / (var + eps).sqrt();
            for j in 0..norm_size {
                let normalized = (inst_in[j] - mean) * inv_std;
                let g = weight.map_or(1.0, |w| w[j]);
                let b = bias.map_or(0.0, |b| b[j]);
                inst_out[j] = g * normalized + b;
            }
        }
        out
    }

    fn assert_close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1e-4,
                "element {i}: got {}, expected {}",
                got[i],
                expected[i]
            );
        }
    }

    /// Deterministic pseudo-random `[B, T, C]` input plus weight/bias of length C.
    fn make_inputs(b: usize, t: usize, c: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut seed = 0x51ce_1234u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / 16_777_216.0) * 4.0 - 2.0
        };
        let input: Vec<f32> = (0..b * t * c).map(|_| next()).collect();
        let weight: Vec<f32> = (0..c).map(|_| next()).collect();
        let bias: Vec<f32> = (0..c).map(|_| next()).collect();
        (input, weight, bias)
    }

    #[test]
    fn layer_norm_matches_reference_all_affine_variants() {
        let (b, t, c) = (2usize, 3usize, 8usize);
        let (input, weight, bias) = make_inputs(b, t, c);
        let eps = 1e-5f32;

        for &(use_w, use_b) in &[(true, true), (true, false), (false, true), (false, false)] {
            let w = if use_w { Some(weight.as_slice()) } else { None };
            let bi = if use_b { Some(bias.as_slice()) } else { None };
            let expected = layer_norm_reference(&input, c, w, bi, eps);

            let tf = TensorFactory::<f32>::new();
            let input_t = tf.make_default(vec![b as i32, t as i32, c as i32], input.clone());
            let weight_t = tf.make_default(vec![c as i32], weight.clone());
            let bias_t = tf.make_default(vec![c as i32], bias.clone());
            let out_t = tf.zeros(
                vec![b as i32, t as i32, c as i32],
                TensorShapeDynamism::DYNAMIC_BOUND,
            );

            let ns_vec: Vec<i64> = vec![c as i64];
            let normalized_shape = ArrayRef::from_raw_parts(ns_vec.as_ptr(), ns_vec.len());

            let mut ctx = context();
            let result = layer_norm_out(
                &mut ctx,
                &input_t,
                normalized_shape,
                if use_w { Some(&weight_t) } else { None },
                if use_b { Some(&bias_t) } else { None },
                eps as f64,
                &out_t,
            );
            assert_eq!(ctx.failure_state(), Error::Ok);
            assert_eq!(result.numel() as usize, b * t * c);
            let got =
                unsafe { core::slice::from_raw_parts(result.const_data_ptr::<f32>(), b * t * c) };
            assert_close(got, &expected);
        }
    }

    #[test]
    fn layer_norm_wrapper_unpacks_stack() {
        let (b, t, c) = (2usize, 2usize, 8usize);
        let (input, weight, bias) = make_inputs(b, t, c);
        let eps = 1e-5f32;
        let expected = layer_norm_reference(&input, c, Some(&weight), Some(&bias), eps);

        let tf = TensorFactory::<f32>::new();
        let input_t = tf.make_default(vec![b as i32, t as i32, c as i32], input.clone());
        let weight_t = tf.make_default(vec![c as i32], weight.clone());
        let bias_t = tf.make_default(vec![c as i32], bias.clone());
        let out_t = tf.zeros(
            vec![b as i32, t as i32, c as i32],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        // Build the `int[] normalized_shape` EValue: an IntList wrapping [C].
        let ns_vec: Vec<i64> = vec![c as i64];
        let mut ns_int_evalues: Vec<EValue> = ns_vec.iter().map(|&v| EValue::from_int(v)).collect();
        let mut ns_wrapped: Vec<*mut EValue> = ns_int_evalues
            .iter_mut()
            .map(|e| e as *mut EValue)
            .collect();
        let mut ns_unwrapped: Vec<i64> = vec![0; ns_vec.len()];
        let mut boxed = BoxedEvalueList::<i64>::new(
            ns_wrapped.as_mut_ptr(),
            ns_unwrapped.as_mut_ptr(),
            ns_vec.len() as i32,
        );

        // Stack: input, normalized_shape, weight, bias, eps, out.
        let mut evalues: Vec<EValue> = vec![
            EValue::from_tensor(input_t),
            EValue::from_int_list(&mut boxed as *mut BoxedEvalueList<i64>),
            EValue::from_tensor(weight_t),
            EValue::from_tensor(bias_t),
            EValue::from_double(eps as f64),
            EValue::from_tensor(out_t),
        ];
        let mut ptrs: Vec<*mut EValue> = evalues.iter_mut().map(|e| e as *mut EValue).collect();

        let mut ctx = context();
        layer_norm_wrapper(
            &mut ctx,
            Span::from_raw_parts(ptrs.as_mut_ptr(), ptrs.len()),
        );
        assert_eq!(ctx.failure_state(), Error::Ok);

        let out = evalues[5].to_tensor();
        let got = unsafe { core::slice::from_raw_parts(out.const_data_ptr::<f32>(), b * t * c) };
        assert_close(got, &expected);
    }

    /// Large `[1, T, 384]` input that crosses the parallel threshold (T*C well
    /// over 32k), exercising the multi-threaded row-chunk path. The per-row math
    /// is unchanged by threading, so the result must be bit-identical to the
    /// single-threaded reference (exact equality, not just within tolerance).
    #[test]
    fn layer_norm_parallel_path_matches_reference() {
        let (b, t, c) = (1usize, 700usize, 384usize); // 700*384 = 268_800 > 64k
        let (input, weight, bias) = make_inputs(b, t, c);
        let eps = 1e-5f32;
        let expected = layer_norm_reference(&input, c, Some(&weight), Some(&bias), eps);

        let tf = TensorFactory::<f32>::new();
        let input_t = tf.make_default(vec![b as i32, t as i32, c as i32], input.clone());
        let weight_t = tf.make_default(vec![c as i32], weight.clone());
        let bias_t = tf.make_default(vec![c as i32], bias.clone());
        let out_t = tf.zeros(
            vec![b as i32, t as i32, c as i32],
            TensorShapeDynamism::DYNAMIC_BOUND,
        );

        let ns_vec: Vec<i64> = vec![c as i64];
        let normalized_shape = ArrayRef::from_raw_parts(ns_vec.as_ptr(), ns_vec.len());

        let mut ctx = context();
        let result = layer_norm_out(
            &mut ctx,
            &input_t,
            normalized_shape,
            Some(&weight_t),
            Some(&bias_t),
            eps as f64,
            &out_t,
        );
        assert_eq!(ctx.failure_state(), Error::Ok);
        let got = unsafe { core::slice::from_raw_parts(result.const_data_ptr::<f32>(), b * t * c) };
        // Bit-identical: parallelism only reorders whole rows, never the
        // arithmetic within a row.
        assert_eq!(got, expected.as_slice());
    }
}
