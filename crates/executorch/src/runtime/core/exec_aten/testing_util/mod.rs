//! Port of runtime/core/exec_aten/testing_util/.
//!
//! Test-support library (TensorFactory + tensor comparison helpers/macros). Ships
//! as a regular `pub` module, mirroring how the C++ builds it as a library rather
//! than gating it behind test-only compilation.

pub mod tensor_factory;
pub mod tensor_util;
