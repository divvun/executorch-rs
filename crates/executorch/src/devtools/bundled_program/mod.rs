//! Literal port of devtools/bundled_program/bundled_program.{h,cpp} +
//! the `bundled_program_flatbuffer` schema accessors.
//!
//! WAVE-2 SCOPE: devtools/bundled_program is out of scope. This module provides
//! only the type/function surface that the ported
//! `extension/module/bundled_module.rs` links against (behind the
//! `bundled-program` feature): `schema::get_bundled_program` and the
//! `load_bundled_input` / `verify_method_outputs` helpers. Bodies are
//! `unimplemented!()` placeholders.
//!
//! PORT-NOTE: type-surface-only stub — see scope note above.

use crate::runtime::core::error::Error;
use crate::runtime::executor::method::Method;

// [spec:et:def:bundled-program.executorch.bundled-program.load-bundled-input-fn]
// [spec:et:sem:bundled-program.executorch.bundled-program.load-bundled-input-fn]
#[must_use]
pub fn load_bundled_input(
    _method: &mut Method<'_>,
    _bundled_program_ptr: *const core::ffi::c_void,
    _testset_idx: usize,
) -> Error {
    unimplemented!()
}

// [spec:et:def:bundled-program.executorch.bundled-program.verify-method-outputs-fn]
// [spec:et:sem:bundled-program.executorch.bundled-program.verify-method-outputs-fn]
#[must_use]
pub fn verify_method_outputs(
    _method: &mut Method<'_>,
    _bundled_program_ptr: *const core::ffi::c_void,
    _testset_idx: usize,
    _rtol: f64,
    _atol: f64,
) -> Error {
    unimplemented!()
}

pub mod schema {
    //! `bundled_program_flatbuffer` generated-schema accessor surface. Only the
    //! `get_bundled_program` -> `BundledProgram` -> `Program` (`data`/`size`)
    //! path used by `bundled_module.rs::program_data_loader` is stubbed.

    pub struct Program {
        _private: (),
    }

    impl Program {
        #[must_use]
        pub fn data(&self) -> *const u8 {
            unimplemented!()
        }

        #[must_use]
        pub fn size(&self) -> usize {
            unimplemented!()
        }
    }

    pub struct BundledProgram {
        _private: (),
    }

    impl BundledProgram {
        #[must_use]
        pub fn program(&self) -> Program {
            unimplemented!()
        }
    }

    #[must_use]
    pub fn get_bundled_program(_bundled_program_ptr: *const core::ffi::c_void) -> BundledProgram {
        unimplemented!()
    }
}
