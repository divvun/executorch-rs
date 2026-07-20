//! divvun-speech custom operators.
//!
//! NOT part of the literal ExecuTorch port. These are the Rust translations of
//! the custom kernels in `divvun-speech-rs/wrapper/custom_ops/`, needed because
//! the exported TTS `.pte` models call operators that are not in upstream
//! ExecuTorch. They live here (rather than in `kernels/`, which mirrors the C++
//! source tree 1:1) so the port's literal-translation invariant is not muddied.
//!
//! Register them with the runtime before loading a method that uses them, via
//! [`register_custom_ops`], otherwise method loading fails with
//! `Error::OperatorMissing`.

pub mod op_istft;
pub mod op_layer_norm;
pub(crate) mod parallel;

use crate::runtime::core::error::Error;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::operator_registry::{Kernel, OpFunction, register_kernels};

/// Register all divvun-speech custom ops (`tts::istft.out`, `tts::layer_norm.out`)
/// with the global operator registry. Call once before loading a method that
/// uses them. Mirrors `tts_register_custom_ops()` in the C++ wrapper.
#[must_use]
pub fn register_custom_ops() -> Error {
    let mut kernels = [
        Kernel::new_fallback(
            op_istft::ISTFT_NAME.as_ptr(),
            op_istft::istft_wrapper as OpFunction,
        ),
        Kernel::new_fallback(
            op_layer_norm::LAYER_NORM_NAME.as_ptr(),
            op_layer_norm::layer_norm_wrapper as OpFunction,
        ),
    ];
    register_kernels(Span::from_raw_parts(kernels.as_mut_ptr(), kernels.len()))
}
