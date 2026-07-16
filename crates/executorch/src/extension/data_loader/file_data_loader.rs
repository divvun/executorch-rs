//! Literal port of extension/data_loader/file_data_loader.{h,cpp}.

extern crate alloc;
extern crate std;

use alloc::alloc::Layout;

use crate::runtime::core::data_loader::{DataLoader, SegmentInfo};
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::Result;

// Some platforms (e.g. Xtensa) do not support pread() that we use to read the
// file at different offsets simultaneously from multiple threads not affecting
// each other. We list them below and use a workaround for them.
//
// PORT-NOTE: the C++ macro `ET_HAVE_PREAD` is 0 on `__xtensa__` / `__hexagon__`
// and 1 elsewhere. Those bare-metal DSP targets are not `unix` targets this
// crate builds for, so ET_HAVE_PREAD is modeled as a `const bool = true`; the
// no-pread branch is retained verbatim behind `if !ET_HAVE_PREAD` so it stays a
// literal translation even though it is currently dead.
const ET_HAVE_PREAD: bool = true;

// Mirrors `alignof(std::max_align_t)`: the strictest alignment any scalar may
// require, used as the default alignment for FileDataLoader::from.
#[repr(C)]
struct MaxAlign {
    _a: u128,
    _b: f64,
}

// [spec:et:def:file-data-loader.executorch.extension.et-aligned-alloc-fn]
// [spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]
fn et_aligned_alloc(size: usize, alignment: usize) -> *mut core::ffi::c_void {
    // Use the nothrow form so allocation failure returns nullptr instead of
    // throwing std::bad_alloc. ExecuTorch is built exception-free and callers
    // (e.g. FileDataLoader::load) check for nullptr and return
    // Error::MemoryAllocationFailed; a throw here would unwind with no landing
    // pad and abort the process.
    //
    // PORT-NOTE: `::operator new(size, alignment, std::nothrow)` becomes an
    // explicit `Layout::from_size_align(size, alignment)` alloc that maps
    // failure to null. `alignment` is already validated as a power of two by
    // the caller.
    match Layout::from_size_align(size, alignment) {
        Ok(layout) => {
            // C++ `operator new` never returns null for a zero-size request;
            // Rust `alloc` is UB for a zero-size layout, so guard it and return
            // a non-null, suitably aligned dangling pointer.
            if layout.size() == 0 {
                return alignment as *mut core::ffi::c_void;
            }
            unsafe { alloc::alloc::alloc(layout) as *mut core::ffi::c_void }
        }
        Err(_) => core::ptr::null_mut(),
    }
}

// [spec:et:def:file-data-loader.executorch.extension.et-aligned-free-fn]
// [spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn]
//
// PORT-NOTE: Rust's `dealloc` needs the full `Layout` (size and alignment),
// whereas C++ `::operator delete(ptr, alignment)` recovers the size from the
// allocator. `size` is threaded through explicitly by every caller (the free
// callback receives it via `FreeableBuffer`; the filename copy records its own
// length). Passing a null `ptr` is a no-op, matching the C++.
fn et_aligned_free(ptr: *mut core::ffi::c_void, size: usize, alignment: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, alignment) {
        if layout.size() == 0 {
            // Matched the dangling pointer returned by the zero-size alloc path.
            return;
        }
        unsafe {
            alloc::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

/// FreeableBuffer::FreeFn-compatible callback.
///
/// `data` is the original buffer pointer.
/// `context` is the original alignment.
///
/// `size` is the original allocation size (needed to recover the Rust Layout).
// [spec:et:def:file-data-loader.executorch.extension.free-segment-fn]
// [spec:et:sem:file-data-loader.executorch.extension.free-segment-fn]
unsafe extern "C" fn free_segment(
    context: *mut core::ffi::c_void,
    data: *mut core::ffi::c_void,
    size: usize,
) {
    et_aligned_free(data, size, context as usize);
}

/// Returns true if the value is an integer power of 2.
// [spec:et:def:file-data-loader.executorch.extension.is-power-of-2-fn]
// [spec:et:sem:file-data-loader.executorch.extension.is-power-of-2-fn]
fn is_power_of_2(value: usize) -> bool {
    value > 0 && (value & !(value.wrapping_sub(1))) == value
}

/// A DataLoader that loads segments from a file, allocating the memory
/// with `malloc()`.
///
/// Note that this will keep the file open for the duration of its lifetime, to
/// avoid the overhead of opening it again for every load() call.
// [spec:et:def:file-data-loader.executorch.extension.file-data-loader]
pub struct FileDataLoader {
    file_name_: *const core::ffi::c_char, // Owned by the instance.
    file_name_len_: usize,                // Length recorded to free the aligned copy.
    file_size_: usize,
    alignment_: usize,
    fd_: libc::c_int, // Owned by the instance.
}

impl FileDataLoader {
    /// Creates a new FileDataLoader that wraps the named file.
    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.from-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]
    pub fn from(file_name: *const core::ffi::c_char, alignment: usize) -> Result<FileDataLoader> {
        crate::et_check_or_return_error!(
            is_power_of_2(alignment),
            InvalidArgument,
            "Alignment {} is not a power of 2",
            alignment
        );

        crate::et_check_or_return_error!(
            !file_name.is_null(),
            InvalidArgument,
            "File name cannot be empty."
        );

        // Use open() instead of fopen() to avoid the layer of buffering that
        // fopen() does. We will be reading large portions of the file in one
        // shot, so buffering does not help.
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
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        let err = unsafe { libc::fstat(fd, &mut st) };
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
        let file_size: usize = st.st_size as usize;
        // Copy the filename so we can print better debug messages if reads fail.
        let file_name_len = unsafe { libc::strlen(file_name) } + 1;
        let file_name_copy = et_aligned_alloc(file_name_len, alignment) as *mut core::ffi::c_char;

        if file_name_copy.is_null() {
            crate::et_log!(Error, "strdup({}) failed", c_str(file_name));
            unsafe {
                libc::close(fd);
            }
            return Err(Error::MemoryAllocationFailed);
        }
        unsafe {
            libc::strcpy(file_name_copy, file_name);
        }

        Ok(FileDataLoader {
            file_name_: file_name_copy,
            file_name_len_: file_name_len,
            file_size_: file_size,
            alignment_: alignment,
            fd_: fd,
        })
    }

    /// Default alignment used by `from` when the caller does not specify one.
    ///
    /// PORT-NOTE: the C++ `from`/`From` declare `size_t alignment =
    /// alignof(std::max_align_t)`. Rust has no default arguments, so the default
    /// is exposed as this constant and the convenience wrappers below.
    pub const DEFAULT_ALIGNMENT: usize = core::mem::align_of::<MaxAlign>();

    /// `from(file_name)` with the default alignment.
    pub fn from_default(file_name: *const core::ffi::c_char) -> Result<FileDataLoader> {
        Self::from(file_name, Self::DEFAULT_ALIGNMENT)
    }

    /// DEPRECATED: Use the lowercase `from()` instead.
    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.from-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]
    #[allow(non_snake_case)]
    pub fn From(file_name: *const core::ffi::c_char, alignment: usize) -> Result<FileDataLoader> {
        Self::from(file_name, alignment)
    }
}

// Not safely copyable.
// [spec:et:def:file-data-loader.executorch.extension.file-data-loader.operator-fn]
// [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.operator-fn]
//
// PORT-NOTE: copy ctor / copy-assign / move-assign are `= delete`d in C++.
// `FileDataLoader` owns a unique fd and heap filename copy; it derives neither
// `Clone` nor `Copy`, so it can only be moved, transferring ownership. Rust's
// move leaves the source logically invalid; the explicit moved-from sentinels
// (null name / fd == -1) that the C++ move ctor writes are unnecessary because
// a Rust value moved out of is never dropped.

impl Drop for FileDataLoader {
    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.file-data-loader-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.file-data-loader-fn]
    fn drop(&mut self) {
        // file_name_ can be nullptr if this instance was moved from, but freeing
        // a null pointer is safe.
        et_aligned_free(
            self.file_name_ as *mut core::ffi::c_void,
            self.file_name_len_,
            self.alignment_,
        );
        // fd_ can be -1 if this instance was moved from, but closing a negative
        // fd is safe (though it will return an error).
        if self.fd_ == -1 {
            return;
        }
        unsafe {
            libc::close(self.fd_);
        }
    }
}

impl DataLoader for FileDataLoader {
    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.load-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn]
    fn load(
        &self,
        offset: usize,
        size: usize,
        segment_info: &SegmentInfo,
    ) -> Result<FreeableBuffer> {
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

        // Don't bother allocating/freeing for empty segments.
        if size == 0 {
            return Ok(FreeableBuffer::from_pointer(
                core::ptr::null(),
                0,
                /*free_fn=*/ None,
                core::ptr::null_mut(),
            ));
        }

        // Allocate memory for the FreeableBuffer.
        let aligned_buffer = et_aligned_alloc(size, self.alignment_);
        if aligned_buffer.is_null() {
            crate::et_log!(
                Error,
                "Reading from {} at offset {}: et_aligned_alloc({}, {}) failed",
                c_str(self.file_name_),
                offset,
                size,
                self.alignment_
            );
            return Err(Error::MemoryAllocationFailed);
        }

        let err = self.load_into(offset, size, segment_info, aligned_buffer);
        if err != Error::Ok {
            et_aligned_free(aligned_buffer, size, self.alignment_);
            return Err(err);
        }

        // Pass the alignment as context to free_segment.
        Ok(FreeableBuffer::from_pointer(
            aligned_buffer,
            size,
            Some(free_segment),
            self.alignment_ as *mut core::ffi::c_void,
        ))
    }

    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.size-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn]
    fn size(&self) -> Result<usize> {
        crate::et_check_or_return_error!(
            // Probably had its value moved to another instance.
            self.fd_ >= 0,
            InvalidState,
            "Uninitialized"
        );
        Ok(self.file_size_)
    }

    // [spec:et:def:file-data-loader.executorch.extension.file-data-loader.load-into-fn]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-into-fn]
    fn load_into(
        &self,
        offset: usize,
        size: usize,
        _segment_info: &SegmentInfo,
        buffer: *mut core::ffi::c_void,
    ) -> Error {
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
        crate::et_check_or_return_error!(
            !buffer.is_null(),
            InvalidArgument,
            "Provided buffer cannot be null"
        );

        // Read the data into the aligned address.
        let mut needed = size;
        let mut buf = buffer as *mut u8;
        let mut offset = offset;

        // Make a duplicate fd if pread() is not available and we have to seek().
        // Cannot use the standard dup() or fcntl() calls because the returned
        // duplicate will share the underlying file record and affect the
        // original fd when seeking on multiple threads simultaneously.
        let dup_fd = if ET_HAVE_PREAD {
            self.fd_
        } else {
            unsafe { libc::open(self.file_name_, libc::O_RDONLY) }
        };

        while needed > 0 {
            // Reads on macOS will fail with EINVAL if size > INT32_MAX.
            let chunk_size: usize = core::cmp::min(needed, i32::MAX as usize);
            let nread: isize = if ET_HAVE_PREAD {
                unsafe {
                    libc::pread(
                        dup_fd,
                        buf as *mut core::ffi::c_void,
                        chunk_size,
                        offset as libc::off_t,
                    )
                }
            } else if unsafe { libc::lseek(dup_fd, offset as libc::off_t, libc::SEEK_SET) }
                == (-1 as libc::off_t)
            {
                -1
            } else {
                unsafe { libc::read(dup_fd, buf as *mut core::ffi::c_void, chunk_size) }
            };
            if nread < 0 && errno() == libc::EINTR {
                // Interrupted by a signal; zero bytes read.
                continue;
            }
            if nread <= 0 {
                // nread == 0 means EOF, which we shouldn't see if we were able
                // to read the full amount. nread < 0 means an error occurred.
                crate::et_log!(
                    Error,
                    "Reading from {}: failed to read {} bytes at offset {}: {}",
                    c_str(self.file_name_),
                    size,
                    offset,
                    if nread == 0 {
                        std::string::String::from("EOF")
                    } else {
                        errno_str()
                    }
                );
                if !ET_HAVE_PREAD {
                    unsafe {
                        libc::close(dup_fd);
                    }
                }
                return Error::AccessFailed;
            }
            needed -= nread as usize;
            buf = unsafe { buf.add(nread as usize) };
            offset += nread as usize;
        }
        if !ET_HAVE_PREAD {
            unsafe {
                libc::close(dup_fd);
            }
        }
        Error::Ok
    }
}

// PORT-NOTE: helpers backing the C++ `%s`/`strerror(errno)`/`errno` log
// substitutions. `c_str` renders a C string pointer for `{}`; `errno` /
// `errno_str` mirror the thread-local `errno` and `strerror(errno)`.
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
    use std::ffi::CString;

    fn setup() {
        // Since these tests cause ET_LOG to be called, the PAL must be
        // initialized first.
        crate::runtime::platform::runtime::runtime_init();
    }

    fn program_segment() -> SegmentInfo {
        SegmentInfo::new(Type::Program, 0, core::ptr::null())
    }

    // The `INSTANTIATE_TEST_SUITE_P(VariedSegments, ...)` values interpreted as
    // "alignment".
    fn alignments() -> std::vec::Vec<usize> {
        std::vec![
            1,
            4,
            core::mem::align_of::<MaxAlign>(),
            2 * core::mem::align_of::<MaxAlign>(),
            128,
            1024,
        ]
    }

    fn is_aligned(ptr: *const core::ffi::c_void, alignment: usize) -> bool {
        (ptr as usize) % alignment == 0
    }

    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    // load() allocates the buffer with et_aligned_alloc (is_aligned asserts),
    // reads it via load_into (byte comparisons), and fb.free() releases it via
    // free_segment -> et_aligned_free.
    // [spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.free-segment-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-into-fn/test]
    #[test]
    fn file_data_loader_test_in_bounds_loads_succeed() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            // Wrap it in a loader.
            let fdl = FileDataLoader::from(tf.path_c().as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::Ok);
            let fdl = fdl.unwrap();

            // size() should succeed and reflect the total size.
            let size = fdl.size();
            assert_eq!(ResultExt::error(&size), Error::Ok);
            assert_eq!(*ResultExt::get(&size), data.len());

            // Load the first bytes of the data.
            {
                let fb = fdl.load(0, 8, &program_segment());
                assert_eq!(ResultExt::error(&fb), Error::Ok);
                let mut fb = fb.unwrap();
                assert!(is_aligned(fb.data(), alignment));
                assert_eq!(fb.size(), 8);
                assert_eq!(0, unsafe {
                    libc::memcmp(
                        fb.data(),
                        b"\x00\x01\x02\x03\x04\x05\x06\x07".as_ptr() as *const core::ffi::c_void,
                        fb.size(),
                    )
                });

                // Freeing should release the buffer and clear out the segment.
                fb.free();
                assert_eq!(fb.size(), 0);
                assert_eq!(fb.data(), core::ptr::null());

                // Safe to call multiple times.
                fb.free();
            }

            // Load the last few bytes of the data, a different size than the first.
            {
                let fb = fdl.load(data.len() - 3, 3, &program_segment());
                assert_eq!(ResultExt::error(&fb), Error::Ok);
                let fb = fb.unwrap();
                assert!(is_aligned(fb.data(), alignment));
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
                let fb = fdl.load(0, data.len(), &program_segment());
                assert_eq!(ResultExt::error(&fb), Error::Ok);
                let fb = fb.unwrap();
                assert!(is_aligned(fb.data(), alignment));
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
                let fb = fdl.load(data.len(), 0, &program_segment());
                assert_eq!(ResultExt::error(&fb), Error::Ok);
                let fb = fb.unwrap();
                assert_eq!(fb.size(), 0);
            }
        }
    }

    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn/test]
    #[test]
    fn file_data_loader_test_out_of_bounds_load_fails() {
        setup();
        for alignment in alignments() {
            // Create a temp file; contents don't matter.
            let data = [0u8; 256];
            let tf = TempFile::new(&data);

            let fdl = FileDataLoader::from(tf.path_c().as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::Ok);
            let fdl = fdl.unwrap();

            // Loading beyond the end of the data should fail.
            {
                let fb = fdl.load(0, data.len() + 1, &program_segment());
                assert_ne!(ResultExt::error(&fb), Error::Ok);
            }

            // Loading zero bytes still fails if it's past the end of the data.
            {
                let fb = fdl.load(data.len() + 1, 0, &program_segment());
                assert_ne!(ResultExt::error(&fb), Error::Ok);
            }
        }
    }

    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn/test]
    #[test]
    fn file_data_loader_test_from_missing_file_fails() {
        setup();
        for _alignment in alignments() {
            // Wrapping a file that doesn't exist should fail.
            let path =
                CString::new("/tmp/FILE_DOES_NOT_EXIST_EXECUTORCH_MMAP_LOADER_TEST").unwrap();
            let fdl = FileDataLoader::from_default(path.as_ptr());
            assert_ne!(ResultExt::error(&fdl), Error::Ok);
        }
    }

    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn/test]
    #[test]
    fn file_data_loader_test_from_empty_file_path_fails() {
        setup();
        for _alignment in alignments() {
            // Nullptr should fail
            let fdl = FileDataLoader::from_default(core::ptr::null());
            assert_ne!(ResultExt::error(&fdl), Error::Ok);
        }
    }

    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn/test]
    // from() rejects the non-power-of-2 alignments via is_power_of_2.
    // [spec:et:sem:file-data-loader.executorch.extension.is-power-of-2-fn/test]
    #[test]
    fn file_data_loader_test_bad_alignment_fails() {
        setup();
        for _alignment in alignments() {
            // Create a temp file; contents don't matter.
            let data = [0u8; 256];
            let tf = TempFile::new(&data);

            // Creating a loader with default alignment works fine.
            {
                let fdl = FileDataLoader::from_default(tf.path_c().as_ptr());
                assert_eq!(ResultExt::error(&fdl), Error::Ok);
            }

            // Bad alignments fail.
            let bad_alignments: [usize; 4] = [0, 3, 5, 17];
            for bad_alignment in bad_alignments {
                let fdl = FileDataLoader::from(tf.path_c().as_ptr(), bad_alignment);
                assert_eq!(ResultExt::error(&fdl), Error::InvalidArgument);
            }
        }
    }

    // Tests that the move ctor works.
    //
    // PORT-NOTE: the C++ move ctor leaves the source in a moved-from state
    // (`fd_ == -1`) that still reports `Error::InvalidState`. In Rust a value
    // moved out of cannot be named again (compile error), so the "old loader is
    // invalid" assertions have no literal analogue; only the "new loader works"
    // half is ported.
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.file-data-loader-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn/test]
    //
    // The deleted copy-assign (`operator=`) collapses onto the move-only loader
    // in Rust (no `Copy`/`Clone`): `let fdl2 = fdl;` transfers unique ownership
    // of the fd/name, and the single surviving owner still works.
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.operator-fn/test]
    #[test]
    fn file_data_loader_test_move_ctor() {
        setup();
        for alignment in alignments() {
            // Create a loader.
            let contents = b"FILE_CONTENTS";
            let tf = TempFile::new(contents);
            let fdl = FileDataLoader::from(tf.path_c().as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::Ok);
            let fdl = fdl.unwrap();
            assert_eq!(fdl.size().get(), &contents.len());

            // Move it into another instance.
            let fdl2 = fdl;

            // New loader should point to the file.
            assert_eq!(fdl2.size().get(), &contents.len());
            let fb = fdl2.load(0, contents.len(), &program_segment());
            assert_eq!(ResultExt::error(&fb), Error::Ok);
            let fb = fb.unwrap();
            assert!(is_aligned(fb.data(), alignment));
            assert_eq!(fb.size(), contents.len());
            assert_eq!(0, unsafe {
                libc::memcmp(
                    fb.data(),
                    contents.as_ptr() as *const core::ffi::c_void,
                    fb.size(),
                )
            });
        }
    }

    // Test that the deprecated From method (capital 'F') still works.
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn/test]
    // [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn/test]
    #[test]
    fn file_data_loader_test_deprecated_from() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            // Wrap it in a loader.
            #[allow(deprecated)]
            let fdl = FileDataLoader::From(tf.path_c().as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::Ok);
            let fdl = fdl.unwrap();

            // size() should succeed and reflect the total size.
            let size = fdl.size();
            assert_eq!(ResultExt::error(&size), Error::Ok);
            assert_eq!(*ResultExt::get(&size), data.len());
        }
    }
}
