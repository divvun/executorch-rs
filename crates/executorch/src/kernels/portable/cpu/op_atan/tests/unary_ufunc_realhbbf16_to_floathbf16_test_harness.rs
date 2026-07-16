//! Shared gtest harness for the unary-ufunc (RealHBBF16 -> FloatHBF16) op test
//! suites, port of kernels/test/UnaryUfuncRealHBBF16ToFloatHBF16Test.{h,cpp}.
//!
//! PORT-NOTE: op_atan.rs pulls this file in via `#[path]` from its `#[cfg(test)]
//! mod tests`. The harness body itself lives in the pattern module
//! (`pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness`) so the acos /
//! acosh / atan / ... suites share one definition; this file re-exports it so
//! the `harness::<fn>` call sites resolve unchanged.
pub use crate::kernels::portable::cpu::pattern::unary_ufunc_realhbbf16_to_floathbf16::test_harness::*;
