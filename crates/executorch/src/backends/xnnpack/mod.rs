pub mod runtime;

/// Register the XNNPACK delegate in the global backend registry. Call once
/// before loading an XNNPACK-delegated `.pte`. Idempotent. Convenience
/// re-export of [`runtime::XNNPACKBackend::register`].
#[cfg(feature = "xnnpack")]
pub use runtime::XNNPACKBackend::register;
