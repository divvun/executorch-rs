//! Literal port of extension/flat_tensor/flat_tensor_data_map.cpp +
//! flat_tensor_data_map.h.
//!
//! WAVE-2 SCOPE: the FlatTensor flatbuffer parsing and `NamedDataMap`
//! implementation have not yet been ported. This module currently provides only
//! the `FlatTensorDataMap` type surface that the already-ported
//! `extension/module/module.rs` links against, mirroring the public signatures
//! in flat_tensor_data_map.h. Bodies are `unimplemented!()` placeholders.
//!
//! PORT-NOTE: type-surface-only stub — see scope note above. `empty()` is a
//! port-local helper used by module.rs's `core::mem::replace`; it has no C++
//! counterpart.

use crate::runtime::core::data_loader::DataLoader;
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::Result;
use crate::runtime::core::tensor_layout::TensorLayout;

pub struct FlatTensorDataMap {
    _private: (),
}

impl FlatTensorDataMap {
    #[must_use]
    pub fn load(_loader: *mut dyn DataLoader) -> Result<FlatTensorDataMap> {
        unimplemented!()
    }

    #[must_use]
    pub fn empty() -> FlatTensorDataMap {
        unimplemented!()
    }
}

impl NamedDataMap for FlatTensorDataMap {
    #[must_use]
    fn get_tensor_layout(&self, _key: &str) -> Result<TensorLayout> {
        unimplemented!()
    }

    #[must_use]
    fn get_data(&self, _key: &str) -> Result<FreeableBuffer> {
        unimplemented!()
    }

    #[must_use]
    fn load_data_into(&self, _key: &str, _buffer: *mut core::ffi::c_void, _size: usize) -> Error {
        unimplemented!()
    }

    #[must_use]
    fn get_num_keys(&self) -> Result<u32> {
        unimplemented!()
    }

    #[must_use]
    fn get_key(&self, _index: u32) -> Result<*const core::ffi::c_char> {
        unimplemented!()
    }
}

// Literal port of backends/xnnpack/test/runtime/test_xnn_data_separation.cpp.
//
// PORT-NOTE: BLOCKED ON STUBS + FIXTURES. `FlatTensorDataMap::load` and every
// `NamedDataMap` accessor above are Wave-2 `unimplemented!()` stubs (the
// FlatTensor flatbuffer parsing is not ported yet), and the C++
// `DataSeparationTest` fixture requires the `.pte`/`.ptd` model fixtures pointed
// at by `ET_MODULE_LINEAR_XNN_PROGRAM_PATH` / `ET_MODULE_LINEAR_XNN_DATA_PATH`
// plus the `ManagedMemoryManager` test helper (runtime/executor/test/), which is
// also not ported. Both tests are therefore `#[ignore]`d and skip early when the
// fixture env vars are unset. When `FlatTensorDataMap` (and `ManagedMemoryManager`)
// land, unblock these — the C++ bodies transcribe directly onto the ported
// `FileDataLoader` / `Program` / `Method` APIs.
//
// This suite is NOT xnnpack-gated (FlatTensorDataMap has no XNNPACK dependency),
// so it lives in the default test build.
#[cfg(test)]
mod tests {
    // PORT-NOTE: no facet — the `FlatTensorDataMap` accessors this exercises are
    // Wave-2 `unimplemented!()` stubs with no spec `def`/`sem` rules to bind a
    // `/test` marker to. Add facets when the module gains its spec rules.
    #[test]
    #[ignore]
    fn data_separation_test_external_data() {
        crate::runtime::platform::runtime::runtime_init();
        if std::env::var("ET_MODULE_LINEAR_XNN_DATA_PATH").is_err() {
            eprintln!(
                "skipping data_separation_test_external_data: \
                 ET_MODULE_LINEAR_XNN_DATA_PATH unset"
            );
            return;
        }
        // Blocked on the FlatTensorDataMap Wave-2 stub (get_num_keys / get_key /
        // get_data all `unimplemented!()`); see module-level PORT-NOTE.
    }

    // PORT-NOTE: no facet — see the sibling test's note; blocked on the same stub.
    #[test]
    #[ignore]
    fn data_separation_test_e2e() {
        crate::runtime::platform::runtime::runtime_init();
        if std::env::var("ET_MODULE_LINEAR_XNN_PROGRAM_PATH").is_err() {
            eprintln!(
                "skipping data_separation_test_e2e: \
                 ET_MODULE_LINEAR_XNN_PROGRAM_PATH unset"
            );
            return;
        }
        // Blocked on the FlatTensorDataMap Wave-2 stub and the unported
        // ManagedMemoryManager test helper; see module-level PORT-NOTE.
    }
}
