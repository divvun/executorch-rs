//! Literal port of extension/data_loader/mmap_data_loader.{h,cpp}.

extern crate std;

use crate::extension::data_loader::mman::{
    fcntl_rdadvise_apple, get_file_stat, get_mmap_offset, get_os_page_size,
    madvise_pages_willneed_sequential,
};
use crate::runtime::core::data_loader::{DataLoader, SegmentInfo};
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;

// Platform mmap primitives and constants. On unix they come from `libc`; on
// Windows from the `mman_windows` shim.
#[cfg(windows)]
use crate::extension::data_loader::mman_windows::{MAP_FAILED, MAP_PRIVATE, MAP_SHARED, PROT_READ};
#[cfg(unix)]
use libc::{MAP_FAILED, MAP_PRIVATE, MAP_SHARED, PROT_READ};

#[cfg(unix)]
unsafe fn platform_mmap(
    len: usize,
    prot: i32,
    flags: i32,
    fildes: libc::c_int,
    off: libc::off_t,
) -> *mut core::ffi::c_void {
    unsafe { libc::mmap(core::ptr::null_mut(), len, prot, flags, fildes, off) }
}
#[cfg(unix)]
unsafe fn platform_munmap(addr: *mut core::ffi::c_void, len: usize) -> libc::c_int {
    unsafe { libc::munmap(addr, len) }
}
#[cfg(unix)]
unsafe fn platform_mlock(addr: *const core::ffi::c_void, len: usize) -> libc::c_int {
    unsafe { libc::mlock(addr, len) }
}

#[cfg(windows)]
unsafe fn platform_mmap(
    len: usize,
    prot: i32,
    flags: i32,
    fildes: libc::c_int,
    off: u64,
) -> *mut core::ffi::c_void {
    crate::extension::data_loader::mman_windows::mmap(
        core::ptr::null_mut(),
        len,
        prot,
        flags,
        fildes,
        off,
    )
}
#[cfg(windows)]
unsafe fn platform_munmap(addr: *mut core::ffi::c_void, len: usize) -> libc::c_int {
    crate::extension::data_loader::mman_windows::munmap(addr, len) as libc::c_int
}
#[cfg(windows)]
unsafe fn platform_mlock(addr: *const core::ffi::c_void, len: usize) -> libc::c_int {
    crate::extension::data_loader::mman_windows::mlock(addr, len) as libc::c_int
}

// [spec:et:def:mmap-data-loader.executorch.extension.range]
struct Range {
    // Address or offset.
    start: usize,
    // Size in bytes.
    size: usize,
}

/// Given an address region, returns the start offset and byte size of the set of
/// pages that completely covers the region.
// [spec:et:def:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]
// [spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]
fn get_overlapping_pages(offset: usize, size: usize, page_size: usize) -> Range {
    let page_mask: usize = !(page_size - 1);
    // The address of the page that starts at or before the beginning of the
    // region.
    let start: usize = offset & page_mask;
    // The address of the page that starts after the end of the region.
    let end: usize = (offset.wrapping_add(size).wrapping_add(!page_mask)) & page_mask;
    Range {
        start,
        size: (end - start),
    }
}

/// Describes how and whether to lock loaded pages with `mlock()`.
// [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.mlock-config]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MlockConfig {
    /// Do not call `mlock()` on loaded pages.
    NoMlock,
    /// Call `mlock()` on loaded pages, failing if it fails.
    UseMlock,
    /// Call `mlock()` on loaded pages, ignoring errors if it fails.
    UseMlockIgnoreErrors,
    /// Use madvise(MADV_WILLNEED | MADV_SEQUENTIAL) instead of mlock.
    UseMadvise,
}

/// A DataLoader that loads segments from a file, allocating the memory
/// with `malloc()`.
///
/// Note that this will keep the file open for the duration of its lifetime, to
/// avoid the overhead of opening it again for every load() call.
// [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader]
pub struct MmapDataLoader {
    file_name_: *const core::ffi::c_char, // String data is owned by the instance.
    file_size_: usize,
    page_size_: usize,
    fd_: libc::c_int, // Owned by the instance.
    mlock_config_: MlockConfig,
}

impl MmapDataLoader {
    /// Creates a new MmapDataLoader that wraps the named file. Fails if
    /// the file can't be opened for reading or if its size can't be found.
    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
    pub fn from(
        file_name: *const core::ffi::c_char,
        mlock_config: MlockConfig,
    ) -> Result<MmapDataLoader> {
        // Cache the page size.
        let page_size: libc::c_long = get_os_page_size();
        if page_size < 0 {
            crate::et_log!(
                Error,
                "Could not get page size: {} ({})",
                errno_str(),
                errno()
            );
            return Err(Error::AccessFailed);
        }
        if (page_size & !(page_size - 1)) != page_size {
            crate::et_log!(Error, "Page size 0x{} is not a power of 2", page_size);
            return Err(Error::InvalidState);
        }

        // Use open() instead of fopen() because mmap() needs a file descriptor.
        let fd = unsafe { libc::open(file_name, libc::O_RDONLY) };
        if fd < 0 {
            crate::et_log!(
                Error,
                "Failed to open {}: {} ({})",
                c_str(file_name),
                errno_str(),
                errno()
            );
            return Err(Error::AccessFailed);
        }

        // Cache the file size.
        let mut file_size: usize = 0;
        let err = get_file_stat(fd, &mut file_size);
        if err < 0 {
            crate::et_log!(
                Error,
                "Could not get length of {}: {} ({})",
                c_str(file_name),
                errno_str(),
                errno()
            );
            unsafe {
                libc::close(fd);
            }
            return Err(Error::AccessFailed);
        }

        // Copy the filename so we can print better debug messages if reads fail.
        let file_name_copy = unsafe { libc::strdup(file_name) };
        if file_name_copy.is_null() {
            crate::et_log!(Error, "strdup({}) failed", c_str(file_name));
            unsafe {
                libc::close(fd);
            }
            return Err(Error::MemoryAllocationFailed);
        }

        Ok(MmapDataLoader {
            file_name_: file_name_copy,
            file_size_: file_size,
            page_size_: page_size as usize,
            fd_: fd,
            mlock_config_: mlock_config,
        })
    }

    /// `from(file_name)` with the default mlock config.
    ///
    /// PORT-NOTE: the C++ `from`/`From` declare `MlockConfig mlock_config =
    /// MlockConfig::UseMlock`. Rust has no default arguments, so the default is
    /// exposed via this convenience wrapper and the deprecated `From` below.
    pub fn from_default(file_name: *const core::ffi::c_char) -> Result<MmapDataLoader> {
        Self::from(file_name, MlockConfig::UseMlock)
    }

    /// DEPRECATED: Use the lowercase `from()` instead.
    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
    #[allow(non_snake_case)]
    pub fn From(file_name: *const core::ffi::c_char) -> Result<MmapDataLoader> {
        Self::from(file_name, MlockConfig::UseMlock)
    }

    /// Validates that file read range is within bounds.
    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]
    fn validate_input(&self, offset: usize, size: usize) -> Error {
        crate::et_check_or_return_error!(
            // Probably had its value moved to another instance.
            self.fd_ >= 0,
            InvalidState,
            "Uninitialized"
        );
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
            !overflow && total_size <= self.file_size_,
            InvalidArgument,
            "File {}: offset {} + size {} > file_size_ {}, or overflow detected",
            c_str(self.file_name_),
            offset,
            size,
            self.file_size_
        );
        Error::Ok
    }
}

// Not safely copyable.
// [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.operator-fn]
// [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.operator-fn]
//
// PORT-NOTE: copy ctor / copy-assign / move-assign are `= delete`d in C++. The
// Rust type derives neither `Clone` nor `Copy`; ownership moves only.

// [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.mmap-data-loader-fn]
// [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.mmap-data-loader-fn]
//
// PORT-NOTE: the C++ move constructor copies the five fields then resets the
// source to moved-from sentinels (null name, size/page 0, fd -1, NoMlock) so
// its destructor is a no-op. Rust move semantics transfer ownership and a value
// moved out of is never dropped, so no explicit reset is needed.

impl Drop for MmapDataLoader {
    fn drop(&mut self) {
        // file_name_ can be nullptr if this instance was moved from, but freeing
        // a null pointer is safe.
        unsafe {
            libc::free(self.file_name_ as *mut core::ffi::c_void);
        }
        // fd_ can be -1 if this instance was moved from, but closing a negative
        // fd is safe (though it will return an error).
        if self.fd_ != -1 {
            unsafe {
                libc::close(self.fd_);
            }
        }
    }
}

/// FreeableBuffer::FreeFn-compatible callback.
///
/// `context` is actually the OS page size as a uintptr_t.
// [spec:et:def:mmap-data-loader.executorch.extension.munmap-segment-fn]
// [spec:et:sem:mmap-data-loader.executorch.extension.munmap-segment-fn]
unsafe extern "C" fn munmap_segment(
    context: *mut core::ffi::c_void,
    data: *mut core::ffi::c_void,
    size: usize,
) {
    let page_size: usize = context as usize;

    let range = get_overlapping_pages(data as usize, size, page_size);
    let ret = unsafe { platform_munmap(range.start as *mut core::ffi::c_void, range.size) };
    if ret < 0 {
        // Let the user know that something went wrong, but there's nothing we
        // can do about it.
        crate::et_log!(
            Error,
            "munmap(0x{:x}, {}) failed: {} ({}) (ignored)",
            range.start,
            range.size,
            errno_str(),
            errno()
        );
    }
}

impl DataLoader for MmapDataLoader {
    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn]
    fn load(
        &self,
        offset: usize,
        size: usize,
        _segment_info: &SegmentInfo,
    ) -> Result<FreeableBuffer> {
        // Ensure read range is valid.
        let validation_err = self.validate_input(offset, size);
        if validation_err != Error::Ok {
            return Err(validation_err);
        }

        // mmap() will fail if the size is zero.
        if size == 0 {
            return Ok(FreeableBuffer::from_pointer(
                core::ptr::null(),
                0,
                /*free_fn=*/ None,
                core::ptr::null_mut(),
            ));
        }

        // Find the range of pages that covers the requested region.
        let range = get_overlapping_pages(offset, size, self.page_size_);

        let mut map_size = range.size;
        if range.start + map_size > self.file_size_ {
            // Clamp to the end of the file.
            //
            // The Windows implementation of mmap uses CreateFileMapping which
            // returns error STATUS_SECTION_TOO_BIG (0xc0000040) if we try to map
            // past the end of the last page of a file mapped in as read-only.
            map_size = self.file_size_ - range.start;
        }

        // Map the pages read-only. Use shared mappings so that other processes
        // can also map the same pages and share the same memory.
        let map_offset = get_mmap_offset(range.start);

        let pages = unsafe { platform_mmap(map_size, PROT_READ, MAP_SHARED, self.fd_, map_offset) };
        crate::et_check_or_return_error!(
            pages != MAP_FAILED,
            AccessFailed,
            "Failed to map {}: mmap(..., size={}, ..., fd={}, offset=0x{:x})",
            c_str(self.file_name_),
            range.size,
            self.fd_,
            range.start
        );

        if self.mlock_config_ == MlockConfig::UseMlock
            || self.mlock_config_ == MlockConfig::UseMlockIgnoreErrors
        {
            let err = unsafe { platform_mlock(pages, size) };
            if err < 0 {
                if self.mlock_config_ == MlockConfig::UseMlockIgnoreErrors {
                    crate::et_log!(
                        Debug,
                        "Ignoring mlock error for file {} (off=0x{}): mlock({:p}, {}) failed: {} ({})",
                        c_str(self.file_name_),
                        offset,
                        pages,
                        size,
                        errno_str(),
                        errno()
                    );
                } else {
                    crate::et_log!(
                        Error,
                        "File {} (off=0x{}): mlock({:p}, {}) failed: {} ({})",
                        c_str(self.file_name_),
                        offset,
                        pages,
                        size,
                        errno_str(),
                        errno()
                    );
                    unsafe {
                        platform_munmap(pages, size);
                    }
                    return Err(Error::NotSupported);
                }
            }
            // No need to keep track of this. munmap() will unlock as a side
            // effect.
        }

        if self.mlock_config_ == MlockConfig::UseMadvise {
            madvise_pages_willneed_sequential(pages, map_size);
            fcntl_rdadvise_apple(self.fd_, self.file_size_);
        }

        // The requested data is at an offset into the mapped pages.
        let data = unsafe { (pages as *const u8).add(offset - range.start) };

        Ok(FreeableBuffer::from_pointer(
            // The callback knows to unmap the whole pages that encompass this
            // region.
            data as *const core::ffi::c_void,
            size,
            Some(munmap_segment),
            // Pass the cached OS page size to the callback so it doesn't need to
            // query it again.
            self.page_size_ as *mut core::ffi::c_void,
        ))
    }

    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn]
    fn size(&self) -> Result<usize> {
        crate::et_check_or_return_error!(
            // Probably had its value moved to another instance.
            self.fd_ >= 0,
            InvalidState,
            "Uninitialized"
        );
        Ok(self.file_size_)
    }

    // [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn]
    fn load_into(
        &self,
        offset: usize,
        size: usize,
        _segment_info: &SegmentInfo,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
        crate::et_check_or_return_error!(!buffer.is_null(), InvalidArgument, "Buffer is null");

        // Ensure read range is valid.
        let err = self.validate_input(offset, size);
        if err != Error::Ok {
            return err;
        }

        // Nothing to copy.
        if size == 0 {
            return Error::Ok;
        }

        // Find the range of pages that covers the requested region.
        let range = get_overlapping_pages(offset, size, self.page_size_);

        let mut map_size = range.size;
        if range.start + map_size > self.file_size_ {
            // Clamp to the end of the file.
            //
            // The Windows implementation of mmap uses CreateFileMapping which
            // returns error STATUS_SECTION_TOO_BIG (0xc0000040) if we try to map
            // past the end of the last page of a file mapped in as read-only.
            map_size = self.file_size_ - range.start;
        }

        // Map the pages read-only. MAP_PRIVATE vs. MAP_SHARED doesn't matter
        // since the data is read-only, but use PRIVATE just to further avoid
        // accidentally modifying the file.
        let map_offset = get_mmap_offset(range.start);

        let pages =
            unsafe { platform_mmap(map_size, PROT_READ, MAP_PRIVATE, self.fd_, map_offset) };
        crate::et_check_or_return_error!(
            pages != MAP_FAILED,
            AccessFailed,
            "Failed to map {}: mmap(..., size={}, ..., fd={}, offset=0x{:x})",
            c_str(self.file_name_),
            range.size,
            self.fd_,
            range.start
        );

        // Offset into mapped region.
        let map_delta: usize = offset - range.start;

        // Copy data into caller's buffer.
        unsafe {
            core::ptr::copy_nonoverlapping(
                (pages as *const u8).add(map_delta),
                buffer as *mut u8,
                size,
            );
        }

        // Unmap mapped region.
        unsafe {
            platform_munmap(pages, map_size);
        }

        Error::Ok
    }
}

// PORT-NOTE: helpers backing the C++ `%s`/`strerror(errno)`/`errno` log
// substitutions (see file_data_loader.rs for the same shape).
fn c_str(ptr: *const core::ffi::c_char) -> std::string::String {
    if ptr.is_null() {
        return std::string::String::from("(null)");
    }
    unsafe { core::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn errno() -> libc::c_int {
    unsafe { *errno_location() }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__errno_location() }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__error() }
}

fn errno_str() -> std::string::String {
    let msg = unsafe { libc::strerror(errno()) };
    if msg.is_null() {
        std::string::String::new()
    } else {
        unsafe { core::ffi::CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::data_loader::testing::TempFile;
    use crate::runtime::core::data_loader::{SegmentInfo, Type};
    use crate::runtime::core::result::ResultExt;
    use std::vec;

    // class MmapDataLoaderTest : public ::testing::Test { void SetUp() ... }
    // Returns the page size, asserting it is positive and a power of 2.
    fn setup() -> usize {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        crate::runtime::platform::runtime::runtime_init();

        // Get the page size and ensure it's a power of 2.
        let page_size = get_os_page_size();
        assert!(page_size > 0);
        assert_eq!(page_size & !(page_size - 1), page_size);
        page_size as usize
    }

    fn program_segment() -> SegmentInfo {
        SegmentInfo::new(Type::Program, 0, core::ptr::null())
    }

    // void MmapDataLoaderTest::test_in_bounds_loads_succeed(mlock_config)
    fn test_in_bounds_loads_succeed(page_size_: usize, mlock_config: MlockConfig) {
        // Create a file containing multiple pages' worth of data, where each
        // 4-byte word has a different value.
        let contents_size = 8 * page_size_;
        let mut contents = vec![0u8; contents_size];
        // NOTE (bug-for-bug): the C++ loop condition is `i > contents_size /
        // sizeof(uint32_t)`, which is false on the first iteration, so the file
        // is left all-zeros. Preserved literally.
        {
            let mut i: usize = 0;
            while i > contents_size / core::mem::size_of::<u32>() {
                let words = contents.as_mut_ptr() as *mut u32;
                unsafe {
                    *words.add(i) = i as u32;
                }
                i += 1;
            }
        }
        let tf = TempFile::new(&contents);

        // Wrap it in a loader.
        let mdl = MmapDataLoader::from(tf.path_c().as_ptr(), mlock_config);
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // size() should succeed and reflect the total size.
        let total_size = mdl.size();
        assert_eq!(ResultExt::error(&total_size), Error::Ok);
        assert_eq!(*ResultExt::get(&total_size), contents_size);

        //
        // Aligned offsets and sizes
        //

        // Load the first page of the file.
        {
            let fb = mdl.load(0, page_size_, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let mut fb = fb.unwrap();
            assert_eq!(fb.size(), page_size_);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[0..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });

            // Freeing should unmap the pages and clear out the segment.
            fb.free();
            assert_eq!(fb.size(), 0);
            assert_eq!(fb.data(), core::ptr::null());

            // Safe to call multiple times.
            fb.free();
        }

        // Load the last couple pages of the data.
        {
            let size = page_size_ * 2;
            let offset = contents_size - size;
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Loading all of the data succeeds.
        {
            let fb = mdl.load(0, contents_size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), contents_size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[0..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Loading two overlapping segments succeeds.
        {
            let offset1 = 0;
            let size1 = page_size_ * 3;
            let fb1 = mdl.load(offset1, size1, &program_segment());
            assert_eq!(ResultExt::error(&fb1), Error::Ok);
            let fb1 = fb1.unwrap();
            assert_eq!(fb1.size(), size1);

            let offset2 = (offset1 + size1) - page_size_;
            let size2 = page_size_ * 3;
            let fb2 = mdl.load(offset2, size2, &program_segment());
            assert_eq!(ResultExt::error(&fb2), Error::Ok);
            let fb2 = fb2.unwrap();
            assert_eq!(fb2.size(), size2);

            // The contents of both segments look good.
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb1.data(),
                    contents[offset1..].as_ptr() as *const core::ffi::c_void,
                    fb1.size(),
                )
            });
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb2.data(),
                    contents[offset2..].as_ptr() as *const core::ffi::c_void,
                    fb2.size(),
                )
            });
        }

        // Loading zero-sized data succeeds, even at the end of the data.
        {
            let fb = mdl.load(contents_size, 0, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), 0);
        }

        //
        // Aligned offsets, unaligned sizes
        //

        // Load a single, partial page.
        {
            let offset = page_size_;
            let size = page_size_ / 2;
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Load a whole number of pages and a partial page.
        {
            let offset = page_size_;
            let size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        //
        // Unaligned offsets and sizes
        //

        // Load a single, partial page with an offset that is not a multiple of
        // the page size.
        {
            let offset = 128; // Small power of 2
            assert!(offset < page_size_);
            let size = page_size_ / 2;
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Load multiple pages from a non-page-aligned but power-of-two offset.
        {
            let offset = page_size_ + 128; // Small power of 2
            let size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }

        // Load multiple pages from an offset that is not a power of 2.
        {
            let offset = page_size_ * 2 + 3; // Not a power of 2
            let size = page_size_ * 3 + page_size_ / 2 + 1; // Odd size
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    // from() calls get_file_stat (size() reflects it) and get_os_page_size;
    // load() calls get_mmap_offset; fb.free() unmaps via munmap_segment.
    // [spec:et:sem:mman.get-file-stat-fn/test]
    // [spec:et:sem:mman.get-os-page-size-fn/test]
    // [spec:et:sem:mman.get-mmap-offset-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.munmap-segment-fn/test]
    #[test]
    fn mmap_data_loader_test_in_bounds_loads_succeed_no_mlock() {
        // There's no portable way to test that mlock() is not called, but
        // exercise the path to make sure the code still behaves correctly.
        let page_size_ = setup();
        test_in_bounds_loads_succeed(page_size_, MlockConfig::NoMlock);
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    #[test]
    fn mmap_data_loader_test_in_bounds_loads_succeed_use_mlock() {
        // There's no portable way to test that mlock() is actually called, but
        // exercise the path to make sure the code still behaves correctly.
        let page_size_ = setup();
        test_in_bounds_loads_succeed(page_size_, MlockConfig::UseMlock);
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    #[test]
    fn mmap_data_loader_test_in_bounds_loads_succeed_use_mlock_ignore_errors() {
        // There's no portable way to inject an mlock() error, but exercise the
        // path to make sure the code still behaves correctly.
        let page_size_ = setup();
        test_in_bounds_loads_succeed(page_size_, MlockConfig::UseMlockIgnoreErrors);
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // The UseMadvise path in load() calls madvise_pages_willneed_sequential and
    // fcntl_rdadvise_apple; the loads still succeed with correct contents.
    // [spec:et:sem:mman.madvise-pages-willneed-sequential-fn/test]
    // [spec:et:sem:mman.fcntl-rdadvise-apple-fn/test]
    #[test]
    fn mmap_data_loader_test_in_bounds_loads_succeed_use_madvise() {
        // There's no portable way to verify madvise() is called, but exercise
        // the path to make sure the code still behaves correctly.
        let page_size_ = setup();
        test_in_bounds_loads_succeed(page_size_, MlockConfig::UseMadvise);
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn/test]
    #[test]
    fn mmap_data_loader_test_final_page_of_uneven_file_succeeds() {
        let page_size_ = setup();
        // Create a file whose length is not an even multiple of a page.
        // Each 4-byte word in the file has a different value.
        const K_NUM_WHOLE_PAGES: usize = 3;
        let contents_size = K_NUM_WHOLE_PAGES * page_size_ + page_size_ / 2;
        let mut contents = vec![0u8; contents_size];
        // See test_in_bounds_loads_succeed: this C++ loop never executes.
        {
            let mut i: usize = 0;
            while i > contents_size / core::mem::size_of::<u32>() {
                let words = contents.as_mut_ptr() as *mut u32;
                unsafe {
                    *words.add(i) = i as u32;
                }
                i += 1;
            }
        }
        let tf = TempFile::new(&contents);

        // Wrap it in a loader.
        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // size() should succeed and reflect the total size.
        let total_size = mdl.size();
        assert_eq!(ResultExt::error(&total_size), Error::Ok);
        assert_eq!(*ResultExt::get(&total_size), contents_size);

        // Read the final page of the file, whose size is smaller than a whole page.
        {
            let offset = K_NUM_WHOLE_PAGES * page_size_;
            let size = contents_size - offset;

            // Demonstrate that this is not a whole page.
            assert!(size > 0);
            assert_ne!(size % page_size_, 0);

            // Load and validate the final partial page.
            let fb = mdl.load(offset, size, &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert_eq!(fb.size(), size);
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents[offset..].as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn/test]
    #[test]
    fn mmap_data_loader_test_out_of_bounds_load_fails() {
        let page_size_ = setup();
        // Create a multi-page file; contents don't matter.
        let contents_size = 8 * page_size_;
        let mut contents = vec![0u8; contents_size];
        for b in contents.iter_mut() {
            *b = 0x55;
        }
        let tf = TempFile::new(&contents);

        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // Loading beyond the end of the data should fail.
        {
            let fb = mdl.load(0, contents_size + 1, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }

        // Loading zero bytes still fails if it's past the end of the data, even
        // if it's aligned.
        {
            let offset = contents_size + page_size_;
            assert_eq!(offset % page_size_, 0);

            let fb = mdl.load(offset, 0, &program_segment());
            assert_ne!(ResultExt::error(&fb), Error::Ok);
        }
    }

    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn/test]
    #[test]
    fn mmap_data_loader_test_from_missing_file_fails() {
        setup();
        // Wrapping a file that doesn't exist should fail.
        let path =
            std::ffi::CString::new("/tmp/FILE_DOES_NOT_EXIST_EXECUTORCH_MMAP_LOADER_TEST").unwrap();
        let mdl = MmapDataLoader::from_default(path.as_ptr());
        assert_ne!(ResultExt::error(&mdl), Error::Ok);
    }

    // Tests that the move ctor works.
    //
    // PORT-NOTE: the C++ move ctor leaves the source reporting
    // `Error::InvalidState`; in Rust a moved-from value cannot be named, so only
    // the "new loader works" half is ported.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.mmap-data-loader-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn/test]
    //
    // The deleted copy-assign (`operator=`) collapses onto the move-only loader
    // in Rust (no `Copy`/`Clone`): the move transfers unique ownership of the
    // fd/name copy, and the single surviving owner still works.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.operator-fn/test]
    #[test]
    fn mmap_data_loader_test_move_ctor() {
        setup();
        // Create a loader.
        let contents = b"FILE_CONTENTS";
        let tf = TempFile::new(contents);
        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();
        assert_eq!(mdl.size().get(), &contents.len());

        // Move it into another instance.
        let mdl2 = mdl;

        // New loader should point to the file.
        assert_eq!(mdl2.size().get(), &contents.len());
        let fb = mdl2.load(0, contents.len(), &program_segment());
        assert_eq!(ResultExt::error(&fb), Error::Ok);
        let fb = fb.unwrap();
        assert_eq!(fb.size(), contents.len());
        assert_eq!(0, unsafe {
            libc::memcmp(
                fb.data(),
                contents.as_ptr() as *const core::ffi::c_void,
                fb.size(),
            )
        });
    }

    // Test that the deprecated From method (capital 'F') still works.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn/test]
    #[test]
    fn mmap_data_loader_test_deprecated_from() {
        let page_size_ = setup();
        // Create a file containing multiple pages' worth of data.
        let contents_size = 8 * page_size_;
        let mut contents = vec![0u8; contents_size];
        // See test_in_bounds_loads_succeed: this C++ loop never executes.
        {
            let mut i: usize = 0;
            while i > contents_size / core::mem::size_of::<u32>() {
                let words = contents.as_mut_ptr() as *mut u32;
                unsafe {
                    *words.add(i) = i as u32;
                }
                i += 1;
            }
        }
        let tf = TempFile::new(&contents);

        // Wrap it in a loader.
        #[allow(deprecated)]
        let mdl = MmapDataLoader::From(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // size() should succeed and reflect the total size.
        let total_size = mdl.size();
        assert_eq!(ResultExt::error(&total_size), Error::Ok);
        assert_eq!(*ResultExt::get(&total_size), contents_size);
    }

    // Tests that load_into copies bytes correctly.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn/test]
    #[test]
    fn mmap_data_loader_test_load_into_copies_correctly() {
        setup();
        // Create a test string.
        let test_text = b"FILE_CONTENTS";
        let text_size = test_text.len();
        let tf = TempFile::new(test_text);

        // Wrap it in a loader.
        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // Destination buffer.
        let mut dst = vec![0u8; text_size];

        // Call load_into()
        let err = mdl.load_into(
            0,
            text_size,
            &program_segment(),
            dst.as_mut_ptr() as *mut core::ffi::c_void,
        );
        assert_eq!(err, Error::Ok);

        // Verify memory copied correctly.
        assert_eq!(0, unsafe {
            libc::memcmp(
                dst.as_ptr() as *const core::ffi::c_void,
                test_text.as_ptr() as *const core::ffi::c_void,
                text_size,
            )
        });
    }

    // Tests that load_into copies offset slice correctly.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn/test]
    #[test]
    fn mmap_data_loader_test_load_into_copies_offset_correctly() {
        setup();
        // Create a test string.
        let contents = b"ABCDEFGH";
        let tf = TempFile::new(contents);

        // Wrap it in a loader.
        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // Copying 3 bytes starting at offset 2 = "CDE"
        let offset = 2;
        let size = 3;
        let mut dst = [0u8; 3];

        // Call load_into()
        let err = mdl.load_into(
            offset,
            size,
            &program_segment(),
            dst.as_mut_ptr() as *mut core::ffi::c_void,
        );
        assert_eq!(err, Error::Ok);

        // Verify memory copied correctly.
        assert_eq!(0, unsafe {
            libc::memcmp(
                dst.as_ptr() as *const core::ffi::c_void,
                contents[offset..].as_ptr() as *const core::ffi::c_void,
                size,
            )
        });
    }

    // Tests that the loader can handle files requiring 64-bit file systems.
    //
    // PORT-NOTE: on 64-bit unix `off_t` is 8 bytes, so the C++ guard
    // `if (sizeof(off_t) <= 8) return;` makes this test a no-op (early return)
    // on all non-Windows 64-bit targets. Preserved literally: the body creates a
    // 3GB sparse file only when `off_t` exceeds 8 bytes.
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn/test]
    // [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn/test]
    #[test]
    fn mmap_data_loader_test_large_file_offset_support() {
        setup();
        // We run some 32 bit tests on Linux so we need to skip this test.
        #[cfg(not(windows))]
        {
            if core::mem::size_of::<libc::off_t>() <= 8 {
                return;
            }
        }

        // Create a sparse file with a marker at an offset beyond 2GB (32-bit
        // limit). We use 3GB to ensure we're testing 64-bit offset handling.
        let large_offset: usize = 3usize * 1024 * 1024 * 1024; // 3GB
        let test_marker = b"TEST_MARKER_AT_LARGE_OFFSET";

        // Use TempFile sparse file API to create a 3GB+ file
        let tf = TempFile::new_sparse(large_offset, test_marker, large_offset + test_marker.len());

        // Now try to load the data using MmapDataLoader.
        let mdl = MmapDataLoader::from_default(tf.path_c().as_ptr());
        assert_eq!(ResultExt::error(&mdl), Error::Ok);
        let mdl = mdl.unwrap();

        // Verify the file size is reported correctly (should be > 3GB).
        let file_size = mdl.size();
        assert_eq!(ResultExt::error(&file_size), Error::Ok);
        assert!(*ResultExt::get(&file_size) > large_offset);
        assert_eq!(
            *ResultExt::get(&file_size),
            large_offset + test_marker.len()
        );

        // Try to load the marker data from the large offset.
        let fb = mdl.load(large_offset, test_marker.len(), &program_segment());
        assert_eq!(ResultExt::error(&fb), Error::Ok);
        let fb = fb.unwrap();

        assert_eq!(fb.size(), test_marker.len());
        assert_eq!(0, unsafe {
            libc::memcmp(
                fb.data(),
                test_marker.as_ptr() as *const core::ffi::c_void,
                test_marker.len(),
            )
        });

        // Test load_into as well.
        let mut buffer = vec![0u8; test_marker.len()];
        let err = mdl.load_into(
            large_offset,
            test_marker.len(),
            &program_segment(),
            buffer.as_mut_ptr() as *mut core::ffi::c_void,
        );
        assert_eq!(err, Error::Ok);

        assert_eq!(0, unsafe {
            libc::memcmp(
                buffer.as_ptr() as *const core::ffi::c_void,
                test_marker.as_ptr() as *const core::ffi::c_void,
                test_marker.len(),
            )
        });
    }
}
