//! Literal port of runtime/executor/merged_data_map.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::{Result, ResultExt};
use crate::runtime::core::tensor_layout::TensorLayout;

/// A NamedDataMap implementation that wraps other NamedDataMaps.
// [spec:et:def:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn]
// [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn]
//
// PORT-NOTE: `first_`/`second_` are borrowed, non-owning `const NamedDataMap*`
// pointers; both maps must outlive the MergedDataMap. They are stored as raw
// `*const dyn NamedDataMap` to preserve pointer identity and the base-pointer
// polymorphism the C++ relied on.
//
// PORT-NOTE: the type is move-constructible (defaulted) but neither copyable
// nor assignable (copy ctor, copy-assign, and move-assign are all `= delete`d).
// See `[spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.operator-fn]`.
// In Rust there is no assignment-through-reference path that rebinds the two
// borrowed maps in place: the struct is constructed once via `load` and
// thereafter only moved, never overwritten. No `Clone`/`Copy` is derived.
pub struct MergedDataMap {
    first_: *const dyn NamedDataMap,
    second_: *const dyn NamedDataMap,
}

impl MergedDataMap {
    /// Creates a new NamedDataMap that wraps two other data maps.
    ///
    /// @param[in] first The first NamedDataMap to merge.
    /// @param[in] second The second NamedDataMap to merge.
    /// Note: the data maps must outlive the MergedDataMap instance.
    pub fn load(
        first: *const dyn NamedDataMap,
        second: *const dyn NamedDataMap,
    ) -> Result<MergedDataMap> {
        crate::et_check_or_return_error!(
            !crate::runtime::executor::merged_data_map::is_null(first)
                && !crate::runtime::executor::merged_data_map::is_null(second),
            InvalidArgument,
            "Input data map is null."
        );

        // Check for duplicate keys.
        let mut k: u32 = 0;
        while k < *ResultExt::get(&unsafe { (*first).get_num_keys() }) {
            let key = *ResultExt::get(&unsafe { (*first).get_key(k) });
            // PORT-NOTE: C++ passes the `const char*` returned by `get_key`
            // (implicitly a NUL-terminated string) as the `std::string_view`
            // argument to `get_tensor_layout`. Here we reconstruct a `&str`
            // from that NUL-terminated pointer to mirror the string_view.
            let key_str = unsafe { core::ffi::CStr::from_ptr(key).to_str().unwrap_or("") };
            let error = ResultExt::error(&unsafe { (*second).get_tensor_layout(key_str) });
            // TODO(lfq): add API to check if key exists.
            crate::et_check_or_return_error!(
                error == Error::NotFound || error == Error::NotImplemented,
                InvalidArgument,
                "Duplicate key {:?}.",
                key
            );
            k += 1;
        }
        Ok(MergedDataMap::new(first, second))
    }

    // [spec:et:def:merged-data-map.executorch.et-runtime-namespace.internal.operator-fn]
    // [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.operator-fn]
    //
    // PORT-NOTE: private two-pointer constructor (`MergedDataMap(first, second)`).
    // The C++ copy ctor, copy-assign, and move-assign are all `= delete`d; move-
    // construction is defaulted. No assignment path is provided in Rust.
    fn new(first: *const dyn NamedDataMap, second: *const dyn NamedDataMap) -> Self {
        MergedDataMap {
            first_: first,
            second_: second,
        }
    }
}

impl NamedDataMap for MergedDataMap {
    /// Retrieve the tensor_layout for the specified key.
    ///
    /// @param[in] key The name of the tensor to get metadata on.
    ///
    /// @return Error::NotFound if the key is not present.
    fn get_tensor_layout(&self, key: &str) -> Result<TensorLayout> {
        let layout = unsafe { (*self.first_).get_tensor_layout(key) };
        if ResultExt::ok(&layout) {
            // PORT-NOTE: C++ `return layout.get();` copies the (trivially
            // copyable) TensorLayout out of the Result. `TensorLayout` derives
            // no `Clone` in the Rust port, so we move the owned value out of
            // `layout` rather than copy it — same observable result.
            return layout;
        }
        if ResultExt::error(&layout) != Error::NotFound {
            return Err(ResultExt::error(&layout));
        }
        unsafe { (*self.second_).get_tensor_layout(key) }
    }

    /// Retrieve read-only data for the specified key.
    ///
    /// @param[in] key The name of the tensor to get data on.
    ///
    /// @return error if the key is not present or data cannot be loaded.
    fn get_data(&self, key: &str) -> Result<FreeableBuffer> {
        let data = unsafe { (*self.first_).get_data(key) };
        if ResultExt::error(&data) != Error::NotFound {
            return data;
        }
        unsafe { (*self.second_).get_data(key) }
    }

    /// Loads the data of the specified tensor into the provided buffer.
    /// Not used in the MergedDataMap.
    ///
    /// @param[in] key The name of the tensor to get the data of.
    /// @param[in] buffer The buffer to load data into. Must point to at least
    /// `size` bytes of memory.
    /// @param[in] size The number of bytes to load.
    ///
    /// @returns an Error indicating if the load was successful.
    fn load_data_into(&self, _key: &str, _buffer: *mut core::ffi::c_void, _size: usize) -> Error {
        Error::NotImplemented
    }

    /// @returns The number of keys in the map.
    fn get_num_keys(&self) -> Result<u32> {
        Ok(*ResultExt::get(&unsafe { (*self.first_).get_num_keys() })
            + *ResultExt::get(&unsafe { (*self.second_).get_num_keys() }))
    }

    /// @returns The key at the specified index, error if index out of bounds.
    fn get_key(&self, index: u32) -> Result<*const core::ffi::c_char> {
        let total_num_keys: u32 = *ResultExt::get(&self.get_num_keys());
        crate::et_check_or_return_error!(
            index < total_num_keys,
            InvalidArgument,
            "Index {} out of range of size {}",
            index,
            total_num_keys
        );

        if index < *ResultExt::get(&unsafe { (*self.first_).get_num_keys() }) {
            unsafe { (*self.first_).get_key(index) }
        } else {
            unsafe {
                (*self.second_).get_key(index - *ResultExt::get(&(*self.first_).get_num_keys()))
            }
        }
    }
}

// PORT-NOTE: helper mirroring the C++ `ptr != nullptr` test for `*const dyn`
// fat pointers, whose data component determines null-ness.
fn is_null(ptr: *const dyn NamedDataMap) -> bool {
    ptr.is_null()
}

// Literal port of runtime/executor/test/merged_data_map_test.cpp.
//
// PORT-NOTE: the fixture loads two `FlatTensorDataMap`s from `.pte` data files
// (`ET_MODULE_ADD_MUL_DATA_PATH`, `ET_MODULE_LINEAR_DATA_PATH`). Both the
// fixtures and `FlatTensorDataMap::load` (a Wave-2 type-surface-only stub that
// `unimplemented!()`s) are absent here, so every fixture-dependent test skips
// early when the env vars are unset and PORT-NOTEs the dependency. Only
// `LoadNullDataMap`, which needs neither, runs.
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    // A null `*const dyn NamedDataMap`: a fat pointer needs a concrete null to
    // cast from. Uses `MergedDataMap` itself as the concrete type only to spell
    // the null; it is never dereferenced.
    fn null_map() -> *const dyn NamedDataMap {
        core::ptr::null::<MergedDataMap>() as *const dyn NamedDataMap
    }

    // [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn/test]
    #[test]
    fn merged_data_map_test_load_null_data_map() {
        setup();
        let merged_map = MergedDataMap::load(null_map(), null_map());
        assert_eq!(ResultExt::error(&merged_map), Error::InvalidArgument);
    }

    // PORT-NOTE: fixture + `FlatTensorDataMap::load` stub dependency (see module
    // note). Skips early when `ET_MODULE_ADD_MUL_DATA_PATH` /
    // `ET_MODULE_LINEAR_DATA_PATH` are unset.
    // [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn/test]
    #[test]
    fn merged_data_map_test_load_multiple_data_maps() {
        setup();
        if std::env::var("ET_MODULE_ADD_MUL_DATA_PATH").is_err()
            || std::env::var("ET_MODULE_LINEAR_DATA_PATH").is_err()
        {
            eprintln!(
                "skipping merged_data_map_test_load_multiple_data_maps: \
                 ET_MODULE_ADD_MUL_DATA_PATH / ET_MODULE_LINEAR_DATA_PATH unset"
            );
            return;
        }
        // Requires FlatTensorDataMap::load (unimplemented stub); cannot run.
        eprintln!(
            "skipping merged_data_map_test_load_multiple_data_maps: \
             FlatTensorDataMap::load is an unimplemented Wave-2 stub"
        );
    }

    // PORT-NOTE: fixture + `FlatTensorDataMap::load` stub dependency (see module
    // note).
    // [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn/test]
    #[test]
    fn merged_data_map_test_load_duplicate_data_maps_fail() {
        setup();
        if std::env::var("ET_MODULE_ADD_MUL_DATA_PATH").is_err()
            || std::env::var("ET_MODULE_LINEAR_DATA_PATH").is_err()
        {
            eprintln!(
                "skipping merged_data_map_test_load_duplicate_data_maps_fail: \
                 ET_MODULE_ADD_MUL_DATA_PATH / ET_MODULE_LINEAR_DATA_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping merged_data_map_test_load_duplicate_data_maps_fail: \
             FlatTensorDataMap::load is an unimplemented Wave-2 stub"
        );
    }

    // PORT-NOTE: fixture + `FlatTensorDataMap::load` stub dependency; also
    // exercises `get_num_keys`, `load_data_into`, `get_tensor_layout`, `get_data`
    // over the merged and constituent maps (see the C++ `compare_ndm_api_calls`).
    // [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn/test]
    #[test]
    fn merged_data_map_test_check_data_map_contents() {
        setup();
        if std::env::var("ET_MODULE_ADD_MUL_DATA_PATH").is_err()
            || std::env::var("ET_MODULE_LINEAR_DATA_PATH").is_err()
        {
            eprintln!(
                "skipping merged_data_map_test_check_data_map_contents: \
                 ET_MODULE_ADD_MUL_DATA_PATH / ET_MODULE_LINEAR_DATA_PATH unset"
            );
            return;
        }
        eprintln!(
            "skipping merged_data_map_test_check_data_map_contents: \
             FlatTensorDataMap::load is an unimplemented Wave-2 stub"
        );
    }
}
