//! Literal Rust port of the ExecuTorch runtime.
//!
//! Module tree mirrors the C++ source tree 1:1; each module is the
//! wave-2 translation of the same-named .cpp/.h pair. See
//! rust/PORTING.md for the translation conventions.

pub mod backends;
// NOT part of the literal port: divvun-speech custom kernels (e.g. the Vocos
// `tts::istft.out` op) that the exported TTS models call but upstream
// ExecuTorch does not provide. See custom_ops/mod.rs.
pub mod custom_ops;
// PORT-NOTE: devtools/ is out of wave-2 scope; only the minimal
// `bundled_program` surface referenced by extension/module/bundled_module.rs is
// stubbed, and only when the `bundled-program` feature (which gates those call
// sites) is enabled.
#[cfg(feature = "bundled-program")]
pub mod devtools;
pub mod extension;
pub mod kernels;
pub mod runtime;
pub mod schema;
