//! Literal port of runtime/platform/types.h.
//!
//! Public types used by the ExecuTorch Platform Abstraction Layer.

/// Platform timestamp in system ticks.
// [spec:et:def:types.et-timestamp-t]
#[allow(non_camel_case_types)]
pub type et_timestamp_t = u64;
