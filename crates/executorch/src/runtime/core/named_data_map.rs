//! Literal port of runtime/core/named_data_map.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;
use crate::runtime::core::tensor_layout::TensorLayout;

/// Interface to access and retrieve data via name.
/// See executorch/extension/flat_tensor/ for an example.
// [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map]
pub trait NamedDataMap {
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.named-data-map-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.named-data-map-fn]
    //
    // PORT-NOTE: `virtual ~NamedDataMap() = default` — the trait is object-safe
    // and normal `Drop` runs on the concrete type behind a boxed/dyn reference;
    // no explicit action is required here.

    /// Get tensor_layout by key.
    ///
    /// `key`: The name of the tensor.
    /// @return Result containing TensorLayout.
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]
    #[must_use]
    fn get_tensor_layout(&self, key: &str) -> Result<TensorLayout>;

    /// Get data by key.
    ///
    /// `key`: Name of the data.
    /// @return Result containing a FreeableBuffer.
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn]
    #[must_use]
    fn get_data(&self, key: &str) -> Result<FreeableBuffer>;

    /// Loads data corresponding to the key into the provided buffer.
    ///
    /// `key`: The name of the data.
    /// `buffer`: The buffer to load the data into. Must point to at least
    /// `size` bytes of memory.
    /// `size`: The number of bytes to load. Use `get_tensor_layout` to
    /// retrieve the size of the data for a given key.
    /// @returns an Error indicating if the load was successful.
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn]
    #[must_use]
    fn load_data_into(&self, key: &str, buffer: *mut core::ffi::c_void, size: usize) -> Error;

    /// Get the number of keys in the NamedDataMap.
    ///
    /// @return Result containing the number of keys.
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn]
    #[must_use]
    fn get_num_keys(&self) -> Result<u32>;

    /// Get the key at the given index.
    ///
    /// `index`: The index of the key to retrieve.
    /// @return Result containing the key at the given index. Note: the returned
    /// pointer is only valid for the lifetime of the DataMap.
    // [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn]
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn]
    #[must_use]
    fn get_key(&self, index: u32) -> Result<*const core::ffi::c_char>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::portable_type::scalar_type::ScalarType;
    use crate::runtime::core::span::Span;
    use crate::runtime::platform::platform::pal_init;

    // A minimal concrete NamedDataMap used to exercise the trait contract
    // (dynamic dispatch + object safety) in-file. The pure-virtual interface
    // itself has no C++ test; the sem rules describe the contract every
    // implementation must honor, which this mock follows.
    struct MockDataMap {
        keys: Vec<&'static str>,
        bytes: Vec<u8>,
        // Backing storage for the single layout's sizes/dim_order spans.
        sizes: Vec<i32>,
        dim_order: Vec<u8>,
    }

    impl MockDataMap {
        fn new() -> Self {
            MockDataMap {
                keys: vec!["alpha", "beta"],
                bytes: vec![10u8, 20, 30, 40],
                sizes: vec![2, 2],
                dim_order: vec![0, 1],
            }
        }
    }

    impl NamedDataMap for MockDataMap {
        fn get_tensor_layout(&self, key: &str) -> Result<TensorLayout> {
            if !self.keys.contains(&key) {
                return Err(Error::NotFound);
            }
            let sizes = Span::from_raw_parts(self.sizes.as_ptr() as *mut i32, self.sizes.len());
            let dim_order =
                Span::from_raw_parts(self.dim_order.as_ptr() as *mut u8, self.dim_order.len());
            TensorLayout::create(sizes, dim_order, ScalarType::Byte)
        }

        fn get_data(&self, key: &str) -> Result<FreeableBuffer> {
            if !self.keys.contains(&key) {
                return Err(Error::NotFound);
            }
            Ok(FreeableBuffer::from_pointer(
                self.bytes.as_ptr() as *const core::ffi::c_void,
                self.bytes.len(),
                None,
                core::ptr::null_mut(),
            ))
        }

        fn load_data_into(&self, key: &str, buffer: *mut core::ffi::c_void, size: usize) -> Error {
            if !self.keys.contains(&key) {
                return Error::NotFound;
            }
            if size != self.bytes.len() {
                return Error::InvalidArgument;
            }
            unsafe {
                core::ptr::copy_nonoverlapping(self.bytes.as_ptr(), buffer as *mut u8, size);
            }
            Error::Ok
        }

        fn get_num_keys(&self) -> Result<u32> {
            Ok(self.keys.len() as u32)
        }

        fn get_key(&self, index: u32) -> Result<*const core::ffi::c_char> {
            match self.keys.get(index as usize) {
                Some(k) => Ok(k.as_ptr() as *const core::ffi::c_char),
                None => Err(Error::InvalidArgument),
            }
        }
    }

    // Exercises the trait through a `&dyn NamedDataMap` base reference: the
    // `named-data-map-fn` rule maps the virtual destructor to trait object
    // safety / `Drop` on the concrete type behind a dyn reference. Constructing
    // and dropping a boxed dyn object proves object safety and that the concrete
    // type's Drop runs.
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.named-data-map-fn/test]
    #[test]
    fn named_data_map_object_safe_dyn() {
        pal_init();
        let map: Box<dyn NamedDataMap> = Box::new(MockDataMap::new());
        let dyn_ref: &dyn NamedDataMap = map.as_ref();
        assert_eq!(dyn_ref.get_num_keys().unwrap(), 2);
        drop(map);
    }

    // get_num_keys: returns N such that valid get_key indices are [0, N).
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn/test]
    #[test]
    fn named_data_map_get_num_keys() {
        pal_init();
        let map = MockDataMap::new();
        let dyn_ref: &dyn NamedDataMap = &map;
        assert_eq!(dyn_ref.get_num_keys().unwrap(), 2);
    }

    // get_key: valid index returns a NUL-usable key pointer that is a valid
    // by-key lookup argument; out-of-range index returns a non-Ok Error.
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn/test]
    #[test]
    fn named_data_map_get_key() {
        pal_init();
        let map = MockDataMap::new();
        let dyn_ref: &dyn NamedDataMap = &map;
        let n = dyn_ref.get_num_keys().unwrap();
        for i in 0..n {
            let ptr = dyn_ref.get_key(i);
            assert!(ptr.is_ok());
            assert!(!ptr.unwrap().is_null());
        }
        // Out-of-range index -> non-Ok Error in the Result.
        assert!(dyn_ref.get_key(n).is_err());
    }

    // get_tensor_layout: present key yields a layout; absent key yields a
    // non-Ok Error. nbytes is derivable for sizing load_data_into.
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn/test]
    #[test]
    fn named_data_map_get_tensor_layout() {
        pal_init();
        let map = MockDataMap::new();
        let dyn_ref: &dyn NamedDataMap = &map;
        let layout = dyn_ref.get_tensor_layout("alpha").unwrap();
        assert_eq!(layout.scalar_type(), ScalarType::Byte);
        assert_eq!(layout.nbytes(), 4);
        assert!(dyn_ref.get_tensor_layout("missing").is_err());
    }

    // get_data: present key returns a FreeableBuffer referencing the bytes with
    // the right size; absent key returns a non-Ok Error.
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn/test]
    #[test]
    fn named_data_map_get_data() {
        pal_init();
        let map = MockDataMap::new();
        let dyn_ref: &dyn NamedDataMap = &map;
        let buf = dyn_ref.get_data("beta").unwrap();
        assert_eq!(buf.size(), 4);
        let first = unsafe { *(buf.data() as *const u8) };
        assert_eq!(first, 10);
        assert!(dyn_ref.get_data("missing").is_err());
    }

    // load_data_into: eager copy into caller memory; Error::Ok on success, a
    // non-Ok Error on missing key or size mismatch (returned directly, not
    // wrapped in Result).
    // [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn/test]
    #[test]
    fn named_data_map_load_data_into() {
        pal_init();
        let map = MockDataMap::new();
        let dyn_ref: &dyn NamedDataMap = &map;
        let mut dst = [0u8; 4];
        let err = dyn_ref.load_data_into(
            "alpha",
            dst.as_mut_ptr() as *mut core::ffi::c_void,
            dst.len(),
        );
        assert_eq!(err, Error::Ok);
        assert_eq!(dst, [10u8, 20, 30, 40]);

        // Missing key -> non-Ok Error, no wrapping in Result.
        assert_ne!(
            dyn_ref.load_data_into("missing", dst.as_mut_ptr() as *mut core::ffi::c_void, 4),
            Error::Ok
        );
    }
}
