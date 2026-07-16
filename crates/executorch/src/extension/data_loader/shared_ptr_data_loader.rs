//! Literal port of extension/data_loader/shared_ptr_data_loader.h.

extern crate alloc;

use alloc::sync::Arc;

use crate::runtime::core::data_loader::{DataLoader, SegmentInfo};
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;

/// A DataLoader that wraps a pre-allocated buffer and shares ownership to it.
/// The FreeableBuffers that it returns do not actually free any data.
///
/// This can be used to wrap data that was allocated elsewhere.
///
/// PORT-NOTE: the C++ field is `std::shared_ptr<void> data_`. Per the wave-2
/// group convention `shared_ptr` maps to `Arc`. The shared handle keeps the
/// buffer alive; `data_ptr_` caches the base pointer (`data_.get()`) so `load`
/// can index into it exactly as the C++ does.
// [spec:et:def:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader]
pub struct SharedPtrDataLoader {
    data_: Arc<dyn core::any::Any + Send + Sync>,
    data_ptr_: *const core::ffi::c_void,
    size_: usize,
}

impl SharedPtrDataLoader {
    // [spec:et:def:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader.shared-ptr-data-loader-fn]
    // [spec:et:sem:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader.shared-ptr-data-loader-fn]
    pub fn new(
        data: Arc<dyn core::any::Any + Send + Sync>,
        data_ptr: *const core::ffi::c_void,
        size: usize,
    ) -> Self {
        SharedPtrDataLoader {
            data_: data,
            data_ptr_: data_ptr,
            size_: size,
        }
    }

    /// Accessor for the shared handle, mirroring `const std::shared_ptr<void>
    /// data_` being held for the loader's lifetime.
    #[allow(dead_code)]
    fn data(&self) -> &Arc<dyn core::any::Any + Send + Sync> {
        &self.data_
    }
}

impl DataLoader for SharedPtrDataLoader {
    fn load(
        &self,
        offset: usize,
        size: usize,
        _segment_info: &SegmentInfo,
    ) -> Result<FreeableBuffer> {
        let total_size: usize;
        let overflow = match offset.checked_add(size) {
            Some(sum) => {
                total_size = sum;
                false
            }
            None => {
                total_size = 0;
                true
            }
        };
        crate::et_check_or_return_error!(
            !overflow && total_size <= self.size_,
            InvalidArgument,
            "offset {} + size {} > size_ {}, or overflow detected",
            offset,
            size,
            self.size_
        );
        Ok(FreeableBuffer::from_pointer(
            unsafe { (self.data_ptr_ as *const u8).add(offset) } as *const core::ffi::c_void,
            size,
            /*free_fn=*/ None,
            core::ptr::null_mut(),
        ))
    }

    fn size(&self) -> Result<usize> {
        Ok(self.size_)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::data_loader::{SegmentInfo, Type};
    use crate::runtime::core::error::Error;
    use crate::runtime::core::result::ResultExt;
    use alloc::vec;
    use alloc::vec::Vec;

    // class SharedPtrDataLoaderTest : public ::testing::Test { void SetUp() ... }
    fn setup() {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        crate::runtime::platform::runtime::runtime_init();
    }

    fn program_segment() -> SegmentInfo {
        SegmentInfo::new(Type::Program, 0, core::ptr::null())
    }

    // The C++ test holds a `std::shared_ptr<uint8_t[]>`. Here the shared handle
    // is an `Arc<Vec<u8>>`; `data_ptr` caches its base pointer, matching the C++
    // `data.get()` passed to the loader ctor.
    fn make_loader(data: Vec<u8>) -> (SharedPtrDataLoader, Arc<Vec<u8>>) {
        let arc: Arc<Vec<u8>> = Arc::new(data);
        let data_ptr = arc.as_ptr() as *const core::ffi::c_void;
        let size = arc.len();
        let loader = SharedPtrDataLoader::new(arc.clone(), data_ptr, size);
        (loader, arc)
    }

    // [spec:et:sem:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader.shared-ptr-data-loader-fn/test]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn/test]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.size-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn shared_ptr_data_loader_test_in_bounds_loads_succeed() {
        setup();
        // Create some heterogeneous data.
        const SIZE: usize = 256;
        let mut buf = vec![0u8; SIZE];
        for i in 0..SIZE {
            buf[i] = i as u8;
        }
        let (sbdl, data) = make_loader(buf);

        // size() should succeed and reflect the total size.
        let size = sbdl.size();
        assert!(ResultExt::ok(&size));
        assert_eq!(*ResultExt::get(&size), SIZE);

        // Load the first bytes of the data.
        {
            let fb = sbdl.load(0, 8, &program_segment());
            assert!(ResultExt::ok(&fb));
            let mut fb = fb.unwrap();
            assert_eq!(fb.size(), 8);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    b"\x00\x01\x02\x03\x04\x05\x06\x07".as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });

            // Freeing should be a no-op but should still clear out the data/size.
            fb.free();
            assert_eq!(fb.size(), 0);
            assert_eq!(fb.data(), core::ptr::null());

            // Safe to call multiple times.
            fb.free();
        }

        // Load the last few bytes of the data, a different size than the first time.
        {
            let fb = sbdl.load(SIZE - 3, 3, &program_segment());
            assert!(ResultExt::ok(&fb));
            let fb = fb.unwrap();
            assert_eq!(fb.size(), 3);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    b"\xfd\xfe\xff".as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Loading all of the data succeeds.
        {
            let fb = sbdl.load(0, SIZE, &program_segment());
            assert!(ResultExt::ok(&fb));
            let fb = fb.unwrap();
            assert_eq!(fb.size(), SIZE);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    data.as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Loading zero-sized data succeeds, even at the end of the data.
        {
            let fb = sbdl.load(SIZE, 0, &program_segment());
            assert!(ResultExt::ok(&fb));
            let fb = fb.unwrap();
            assert_eq!(fb.size(), 0);
        }
    }

    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn/test]
    #[test]
    fn shared_ptr_data_loader_test_out_of_bounds_load_fails() {
        setup();
        // Wrap some data in a loader.
        const SIZE: usize = 256;
        let (sbdl, _data) = make_loader(vec![0u8; SIZE]);

        // Loading beyond the end of the data should fail.
        {
            let fb = sbdl.load(0, SIZE + 1, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }

        // Loading zero bytes still fails if it's past the end of the data.
        {
            let fb = sbdl.load(SIZE + 1, 0, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }
    }
}
