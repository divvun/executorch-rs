//! Literal port of extension/data_loader/buffer_data_loader.h.

use crate::runtime::core::data_loader::{DataLoader, SegmentInfo};
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::{Result, ResultExt};

/// A DataLoader that wraps a pre-allocated buffer. The FreeableBuffers
/// that it returns do not actually free any data.
///
/// This can be used to wrap data that is directly embedded into the firmware
/// image, or to wrap data that was allocated elsewhere.
// [spec:et:def:buffer-data-loader.executorch.extension.buffer-data-loader]
pub struct BufferDataLoader {
    data_: *const u8, // uint8 is easier to index into.
    size_: usize,
}

impl BufferDataLoader {
    // [spec:et:def:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn]
    // [spec:et:sem:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn]
    pub fn new(data: *const core::ffi::c_void, size: usize) -> Self {
        BufferDataLoader {
            data_: data as *const u8,
            size_: size,
        }
    }
}

impl DataLoader for BufferDataLoader {
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
            unsafe { self.data_.add(offset) } as *const core::ffi::c_void,
            size,
            /*free_fn=*/ None,
            core::ptr::null_mut(),
        ))
    }

    fn size(&self) -> Result<usize> {
        Ok(self.size_)
    }

    fn load_into(
        &self,
        offset: usize,
        size: usize,
        segment_info: &SegmentInfo,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
        crate::et_check_or_return_error!(
            !buffer.is_null(),
            InvalidArgument,
            "Destination buffer cannot be null"
        );

        let result = self.load(offset, size, segment_info);
        if !ResultExt::ok(&result) {
            return ResultExt::error(&result);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                ResultExt::get(&result).data() as *const u8,
                buffer as *mut u8,
                size,
            );
        }
        Error::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::data_loader::Type;
    use crate::runtime::core::result::ResultExt;

    // class BufferDataLoaderTest : public ::testing::Test { void SetUp() ... }
    fn setup() {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        crate::runtime::platform::runtime::runtime_init();
    }

    fn program_segment() -> SegmentInfo {
        SegmentInfo::new(Type::Program, 0, core::ptr::null())
    }

    // [spec:et:sem:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn/test]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn/test]
    // [spec:et:sem:data-loader.executorch.runtime.data-loader.size-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn buffer_data_loader_test_in_bounds_loads_succeed() {
        setup();
        // Create some heterogeneous data.
        let mut data = [0u8; 256];
        for i in 0..data.len() {
            data[i] = i as u8;
        }

        // Wrap it in a loader.
        let edl = BufferDataLoader::new(data.as_ptr() as *const core::ffi::c_void, data.len());

        // size() should succeed and reflect the total size.
        let size = edl.size();
        assert!(ResultExt::ok(&size));
        assert_eq!(*ResultExt::get(&size), data.len());

        // Load the first bytes of the data.
        {
            let fb = edl.load(0, 8, &program_segment());
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
            let fb = edl.load(data.len() - 3, 3, &program_segment());
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
            let fb = edl.load(0, data.len(), &program_segment());
            assert!(ResultExt::ok(&fb));
            let fb = fb.unwrap();
            assert_eq!(fb.size(), data.len());
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
            let fb = edl.load(data.len(), 0, &program_segment());
            assert!(ResultExt::ok(&fb));
            let fb = fb.unwrap();
            assert_eq!(fb.size(), 0);
        }
    }

    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn/test]
    #[test]
    fn buffer_data_loader_test_out_of_bounds_load_fails() {
        setup();
        // Wrap some data in a loader.
        let data = [0u8; 256];
        let edl = BufferDataLoader::new(data.as_ptr() as *const core::ffi::c_void, data.len());

        // Loading beyond the end of the data should fail.
        {
            let fb = edl.load(0, data.len() + 1, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }

        // Loading zero bytes still fails if it's past the end of the data.
        {
            let fb = edl.load(data.len() + 1, 0, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }
    }

    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn/test]
    #[test]
    fn buffer_data_loader_test_overflow_load_fails() {
        setup();
        // Wrap some data in a loader.
        let data = [0u8; 256];
        let edl = BufferDataLoader::new(data.as_ptr() as *const core::ffi::c_void, data.len());

        // Loading with offset + size that would overflow should fail.
        // Use a small valid offset but a size that causes overflow.
        // If overflow wasn't checked, 1 + SIZE_MAX would wrap to 0, which is <= 256.
        {
            let fb = edl.load(1, usize::MAX, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }

        // Another overflow case: offset within bounds, size causes overflow.
        // 128 + (SIZE_MAX - 127) wraps to 0.
        {
            let fb = edl.load(128, usize::MAX - 127, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }
    }

    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-into-fn/test]
    #[test]
    fn buffer_data_loader_test_load_into_null_dst_fails() {
        setup();
        // Wrap some data in a loader.
        let data = [0u8; 256];
        let edl = BufferDataLoader::new(data.as_ptr() as *const core::ffi::c_void, data.len());

        // Loading beyond the end of the data should fail.
        {
            let fb = edl.load_into(0, 1, &program_segment(), core::ptr::null_mut());
            assert_ne!(fb, Error::Ok);
        }

        // Loading zero bytes still fails if dst is null.
        {
            let fb = edl.load_into(0, 0, &program_segment(), core::ptr::null_mut());
            assert_ne!(fb, Error::Ok);
        }
    }

    // [spec:et:sem:data-loader.executorch.runtime.data-loader.load-into-fn/test]
    #[test]
    fn buffer_data_loader_test_in_bounds_load_into_succeeds() {
        setup();
        // Wrap some data in a loader.
        let mut data = [0u8; 256];
        data[0] = 1;
        let mut buffer = [0u8; 256];
        buffer[0] = 0;
        let edl = BufferDataLoader::new(data.as_ptr() as *const core::ffi::c_void, data.len());

        {
            // Buffer contains 0 before load_into.
            assert_eq!(buffer[0], 0);
            let fb = edl.load_into(
                0,
                1,
                &program_segment(),
                buffer.as_mut_ptr() as *mut core::ffi::c_void,
            );
            assert_eq!(fb, Error::Ok);
            // Buffer contains 1 after load_into.
            assert_eq!(buffer[0], 1);
            // Data is unaltered.
            assert_eq!(data[0], 1);
        }
    }
}
