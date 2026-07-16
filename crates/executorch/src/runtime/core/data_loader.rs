//! Literal port of runtime/core/data_loader.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;

/// Loads from a data source.
///
/// See //executorch/extension/data_loader for common implementations.
// [spec:et:def:data-loader.executorch.runtime.data-loader]
pub trait DataLoader {
    // [spec:et:def:data-loader.executorch.runtime.data-loader.data-loader-fn]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.data-loader-fn]
    //
    // PORT-NOTE: `virtual ~DataLoader() = default` carries no drop logic; the
    // base trait requires none and ownership/cleanup is left to the concrete
    // implementor's `Drop`.

    /// Loads data from the underlying data source.
    ///
    /// NOTE: This must be thread-safe. If this call modifies common state, the
    /// implementation must do its own locking.
    ///
    /// `offset`: The byte offset in the data source to start loading from.
    /// `size`: The number of bytes to load.
    /// `segment_info`: Information about the segment being loaded.
    ///
    /// @returns a `FreeableBuffer` that owns the loaded data.
    // [spec:et:def:data-loader.executorch.runtime.data-loader.load-fn]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn]
    #[must_use]
    fn load(
        &self,
        offset: usize,
        size: usize,
        segment_info: &SegmentInfo,
    ) -> Result<FreeableBuffer>;

    /// Loads data from the underlying data source into the provided buffer.
    ///
    /// NOTE: This must be thread-safe. If this call modifies common state, the
    /// implementation must do its own locking.
    ///
    /// `offset`: The byte offset in the data source to start loading from.
    /// `size`: The number of bytes to load.
    /// `segment_info`: Information about the segment being loaded.
    /// `buffer`: The buffer to load data into. Must point to at least `size`
    /// bytes of memory.
    ///
    /// @returns an Error indicating if the load was successful.
    // [spec:et:def:data-loader.executorch.runtime.data-loader.load-into-fn]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-into-fn]
    #[must_use]
    fn load_into(
        &self,
        offset: usize,
        size: usize,
        segment_info: &SegmentInfo,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
        // Using a stub implementation here instead of pure virtual to expand the
        // data_loader interface in a backwards compatible way.
        let _ = buffer;
        let _ = offset;
        let _ = size;
        let _ = segment_info;
        crate::et_log!(Error, "load_into() not implemented for this data loader.");
        Error::NotImplemented
    }

    /// Returns the length of the underlying data source, typically the file size.
    // [spec:et:def:data-loader.executorch.runtime.data-loader.size-fn]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.size-fn]
    #[must_use]
    fn size(&self) -> Result<usize>;
}

/// Describes the content of the segment.
// [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info]
#[derive(Clone, Copy)]
pub struct SegmentInfo {
    /// Type of the segment.
    pub segment_type: Type,

    /// Index of the segment within the segment list. Undefined for program
    /// segments.
    pub segment_index: usize,

    /// An optional, null-terminated string describing the segment. For
    /// `Backend` segments, this is the backend ID. Null for other segment
    /// types.
    pub descriptor: *const core::ffi::c_char,
}

/// Represents the purpose of the segment.
// [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info.type]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Type {
    /// Data for the actual program.
    Program,
    /// Holds constant tensor data.
    Constant,
    /// Data used for initializing a backend.
    Backend,
    /// Data used for initializing mutable tensors.
    Mutable,
    /// Data used for initializing external tensors.
    External,
}

impl SegmentInfo {
    // [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn]
    //
    // PORT-NOTE: the C++ `explicit SegmentInfo(Type, size_t = 0, const char* =
    // nullptr)` has default arguments; Rust has none, so callers pass all three
    // (0 and null()) explicitly to match. The separate defaulted default
    // constructor (`SegmentInfo() = default`, all fields uninitialized) is not
    // reproduced — Rust cannot leave fields uninitialized.
    pub fn new(
        segment_type_: Type,
        segment_index_: usize,
        descriptor_: *const core::ffi::c_char,
    ) -> Self {
        SegmentInfo {
            segment_type: segment_type_,
            segment_index: segment_index_,
            descriptor: descriptor_,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // SegmentInfo::new stores its arguments verbatim into the corresponding
    // fields (no validation, no copy of the descriptor string). Mirrors the
    // C++ explicit ctor's default arguments (index = 0, descriptor = nullptr)
    // by passing them explicitly.
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn/test]
    #[test]
    fn segment_info_stores_fields_verbatim() {
        // Defaults: index 0, null descriptor.
        let program = SegmentInfo::new(Type::Program, 0, core::ptr::null());
        assert_eq!(program.segment_type, Type::Program);
        assert_eq!(program.segment_index, 0);
        assert!(program.descriptor.is_null());

        // Descriptor pointer stored as-is, not copied.
        let backend_id = c"xnnpack";
        let backend = SegmentInfo::new(Type::Backend, 7, backend_id.as_ptr());
        assert_eq!(backend.segment_type, Type::Backend);
        assert_eq!(backend.segment_index, 7);
        assert_eq!(backend.descriptor, backend_id.as_ptr());
    }

    // `virtual ~DataLoader() = default` maps to Drop on the concrete
    // implementor behind a `Box<dyn DataLoader>`: the trait is object-safe,
    // usable through the base pointer, and dropping through the base runs the
    // concrete type's cleanup exactly once.
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.data-loader-fn/test]
    #[test]
    fn data_loader_drop_through_dyn_runs_concrete_drop() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        struct TrackedLoader;
        impl Drop for TrackedLoader {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        impl DataLoader for TrackedLoader {
            fn load(
                &self,
                _offset: usize,
                size: usize,
                _segment_info: &SegmentInfo,
            ) -> Result<FreeableBuffer> {
                Ok(FreeableBuffer::from_pointer(
                    core::ptr::null(),
                    size,
                    None,
                    core::ptr::null_mut(),
                ))
            }
            fn size(&self) -> Result<usize> {
                Ok(64)
            }
        }

        let loader: Box<dyn DataLoader> = Box::new(TrackedLoader);
        assert_eq!(loader.size().unwrap(), 64);
        drop(loader);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    }
}
