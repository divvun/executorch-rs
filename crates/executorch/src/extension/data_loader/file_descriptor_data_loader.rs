//! Literal port of extension/data_loader/file_descriptor_data_loader.{h,cpp}.

extern crate alloc;
extern crate std;

use alloc::alloc::Layout;
use alloc::vec::Vec;

use crate::runtime::core::data_loader::{DataLoader, SegmentInfo};
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::result::{Result, ResultExt};

// Mirrors the C++ `static constexpr char kFdFilesystemPrefix[] = "fd:///";`,
// which is a NUL-terminated char array. The trailing `\0` is kept so that
// `K_FD_FILESYSTEM_PREFIX.len() - 1` equals `strlen(kFdFilesystemPrefix)` (6),
// matching the C++ `strncmp`/`strlen` byte counts exactly.
const K_FD_FILESYSTEM_PREFIX: &[u8] = b"fd:///\0";

/// Returns true if the value is an integer power of 2.
// [spec:et:def:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn]
fn is_power_of_2(value: usize) -> bool {
    value > 0 && (value & !(value.wrapping_sub(1))) == value
}

// [spec:et:def:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]
//
// PORT-NOTE: the C++ uses the *throwing* `::operator new(size, alignment)` here
// (no `std::nothrow`), so under a throwing global operator new an allocation
// failure would throw `std::bad_alloc`. The caller still checks the result for
// null. The Rust port allocates via `Layout::from_size_align`, mapping failure
// to null (which the caller maps to `MemoryAllocationFailed`).
fn et_aligned_alloc(size: usize, alignment: usize) -> *mut core::ffi::c_void {
    match Layout::from_size_align(size, alignment) {
        Ok(layout) => {
            if layout.size() == 0 {
                return alignment as *mut core::ffi::c_void;
            }
            unsafe { alloc::alloc::alloc(layout) as *mut core::ffi::c_void }
        }
        Err(_) => core::ptr::null_mut(),
    }
}

// [spec:et:def:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]
//
// PORT-NOTE: Rust's `dealloc` needs the full `Layout`; `size` is threaded in via
// the `FreeableBuffer` free callback. Freeing a null `ptr` is a no-op.
fn et_aligned_free(ptr: *mut core::ffi::c_void, size: usize, alignment: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(size, alignment) {
        if layout.size() == 0 {
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
// [spec:et:def:file-descriptor-data-loader.executorch.extension.free-segment-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.free-segment-fn]
unsafe extern "C" fn free_segment(
    context: *mut core::ffi::c_void,
    data: *mut core::ffi::c_void,
    size: usize,
) {
    et_aligned_free(data, size, context as usize);
}

// [spec:et:def:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn]
fn get_fd_from_uri(file_descriptor_uri: *const core::ffi::c_char) -> Result<libc::c_int> {
    // check if the uri starts with the prefix "fd://"
    crate::et_check_or_return_error!(
        unsafe {
            libc::strncmp(
                file_descriptor_uri,
                K_FD_FILESYSTEM_PREFIX.as_ptr() as *const core::ffi::c_char,
                K_FD_FILESYSTEM_PREFIX.len() - 1,
            )
        } == 0,
        InvalidArgument,
        "File descriptor uri ({}) does not start with {}",
        c_str(file_descriptor_uri),
        std::str::from_utf8(&K_FD_FILESYSTEM_PREFIX[..K_FD_FILESYSTEM_PREFIX.len() - 1]).unwrap()
    );

    // strip "fd:///" from the uri
    let prefix_len = K_FD_FILESYSTEM_PREFIX.len() - 1;
    let fd_len = unsafe { libc::strlen(file_descriptor_uri) } - prefix_len;
    // C++ builds a stack VLA `fd_without_prefix[fd_len + 1]`, memcpy's the
    // remainder, and NUL-terminates. A heap Vec stands in for the VLA.
    let mut fd_without_prefix: Vec<core::ffi::c_char> = Vec::with_capacity(fd_len + 1);
    unsafe {
        core::ptr::copy_nonoverlapping(
            file_descriptor_uri.add(prefix_len),
            fd_without_prefix.as_mut_ptr(),
            fd_len,
        );
        fd_without_prefix.set_len(fd_len);
        fd_without_prefix.push(0);
    }

    // check if remaining fd string is a valid integer
    let fd = unsafe { libc::atoi(fd_without_prefix.as_ptr()) };
    Ok(fd)
}

/// A DataLoader that loads segments from a file descriptor, allocating the
/// memory with `malloc()`. This data loader is used when ET is running in a
/// process that does not have access to the filesystem, and the caller is able
/// to open the file and pass the file descriptor.
///
/// Note that this will keep the file open for the duration of its lifetime, to
/// avoid the overhead of opening it again for every load() call.
// [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader]
pub struct FileDescriptorDataLoader {
    file_descriptor_uri_: *const core::ffi::c_char, // Owned by the instance.
    file_size_: usize,
    alignment_: usize,
    fd_: libc::c_int, // Owned by the instance.
}

impl FileDescriptorDataLoader {
    /// Creates a new FileDescriptorDataLoader that wraps the named file
    /// descriptor, and the ownership of the file descriptor is passed.
    // [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn]
    pub fn from_file_descriptor_uri(
        file_descriptor_uri: *const core::ffi::c_char,
        alignment: usize,
    ) -> Result<FileDescriptorDataLoader> {
        crate::et_check_or_return_error!(
            is_power_of_2(alignment),
            InvalidArgument,
            "Alignment {} is not a power of 2",
            alignment
        );

        let parsed_fd = get_fd_from_uri(file_descriptor_uri);
        if !ResultExt::ok(&parsed_fd) {
            return Err(ResultExt::error(&parsed_fd));
        }

        let fd = *ResultExt::get(&parsed_fd);

        // Cache the file size.
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        let err = unsafe { libc::fstat(fd, &mut st) };
        if err < 0 {
            crate::et_log!(
                Error,
                "Could not get length of {}: {} ({})",
                c_str(file_descriptor_uri),
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
        let file_descriptor_uri_copy = unsafe { libc::strdup(file_descriptor_uri) };
        if file_descriptor_uri_copy.is_null() {
            crate::et_log!(Error, "strdup({}) failed", c_str(file_descriptor_uri));
            unsafe {
                libc::close(fd);
            }
            return Err(Error::MemoryAllocationFailed);
        }

        Ok(FileDescriptorDataLoader {
            file_descriptor_uri_: file_descriptor_uri_copy,
            file_size_: file_size,
            alignment_: alignment,
            fd_: fd,
        })
    }
}

// Not safely copyable.
// [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.operator-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.operator-fn]
//
// PORT-NOTE: copy ctor / copy-assign / move-assign are `= delete`d in C++. The
// Rust type derives neither `Clone` nor `Copy`; only moves transfer ownership.

// [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn]
// [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn]
//
// PORT-NOTE: the C++ move constructor copies the four fields then resets the
// source to moved-from sentinels (null uri, size 0, `{}` alignment, fd -1) so
// its destructor is a no-op. Rust move semantics transfer ownership and a value
// moved out of is never dropped, so no explicit reset is needed.

impl Drop for FileDescriptorDataLoader {
    fn drop(&mut self) {
        // file_descriptor_uri_ can be nullptr if this instance was moved from,
        // but freeing a null pointer is safe.
        unsafe {
            libc::free(self.file_descriptor_uri_ as *mut core::ffi::c_void);
        }
        // fd_ can be -1 if this instance was moved from, but closing a negative
        // fd is safe (though it will return an error).
        unsafe {
            libc::close(self.fd_);
        }
    }
}

impl DataLoader for FileDescriptorDataLoader {
    // [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]
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
            c_str(self.file_descriptor_uri_),
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
                c_str(self.file_descriptor_uri_),
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

    // [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn]
    fn size(&self) -> Result<usize> {
        crate::et_check_or_return_error!(
            // Probably had its value moved to another instance.
            self.fd_ >= 0,
            InvalidState,
            "Uninitialized"
        );
        Ok(self.file_size_)
    }

    // [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn]
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
            c_str(self.file_descriptor_uri_),
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

        while needed > 0 {
            // Reads on macOS will fail with EINVAL if size > INT32_MAX.
            let chunk_size: usize = core::cmp::min(needed, i32::MAX as usize);
            let nread: isize = unsafe {
                platform_pread(self.fd_, buf as *mut core::ffi::c_void, chunk_size, offset)
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
                    c_str(self.file_descriptor_uri_),
                    size,
                    offset,
                    if nread == 0 {
                        std::string::String::from("EOF")
                    } else {
                        errno_str()
                    }
                );
                return Error::AccessFailed;
            }
            needed -= nread as usize;
            buf = unsafe { buf.add(nread as usize) };
            offset += nread as usize;
        }
        Error::Ok
    }
}

// Positioned read normalized to the POSIX `pread` shape. On unix this is
// `libc::pread`; on Windows it forwards to the `compat_unistd` Win64 shim
// (`ReadFile`/`GetOverlappedResult`), mirroring the C++ `compat_unistd.h`.
#[cfg(unix)]
unsafe fn platform_pread(
    fd: libc::c_int,
    buf: *mut core::ffi::c_void,
    count: usize,
    offset: usize,
) -> isize {
    unsafe { libc::pread(fd, buf, count, offset as libc::off_t) }
}
#[cfg(windows)]
unsafe fn platform_pread(
    fd: libc::c_int,
    buf: *mut core::ffi::c_void,
    count: usize,
    offset: usize,
) -> isize {
    crate::runtime::platform::compat_unistd::pread(fd, buf, count, offset)
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

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__error() }
}

#[cfg(windows)]
unsafe fn errno_location() -> *mut libc::c_int {
    // MSVCRT exposes the thread-local errno via `_errno()`; the `libc` crate
    // does not re-export it on Windows, so declare the accessor locally
    // (mirrors mman_windows.rs / compat_unistd.rs).
    unsafe extern "C" {
        fn _errno() -> *mut libc::c_int;
    }
    unsafe { _errno() }
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

    fn alignments() -> Vec<usize> {
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

    // `::open(tf.path().c_str(), O_RDONLY)`
    fn open_rdonly(tf: &TempFile) -> libc::c_int {
        let path = tf.path_c();
        unsafe { libc::open(path.as_ptr(), libc::O_RDONLY) }
    }

    fn fd_uri(fd: libc::c_int) -> CString {
        CString::new(std::format!("fd:///{}", fd)).unwrap()
    }

    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    // load() allocates via et_aligned_alloc (is_aligned asserts), reads via
    // load_into (byte comparisons), and fb.free() releases via free_segment ->
    // et_aligned_free.
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.free-segment-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_in_bounds_file_descriptor_loads_succeed() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            let fd = open_rdonly(&tf);

            // Wrap it in a loader.
            let fdl =
                FileDescriptorDataLoader::from_file_descriptor_uri(fd_uri(fd).as_ptr(), alignment);
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

            // Load the last few bytes of the data, a different size.
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

    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_file_descriptor_load_prefix_fail() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            let fd = open_rdonly(&tf);

            // Wrap it in a loader, without the "fd:///" prefix.
            let uri = CString::new(std::format!("{}", fd)).unwrap();
            let fdl = FileDescriptorDataLoader::from_file_descriptor_uri(uri.as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::InvalidArgument);
        }
    }

    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn/test]
    // [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_in_bounds_loads_succeed() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            let fd = open_rdonly(&tf);

            // Wrap it in a loader.
            let fdl =
                FileDescriptorDataLoader::from_file_descriptor_uri(fd_uri(fd).as_ptr(), alignment);
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

                fb.free();
                assert_eq!(fb.size(), 0);
                assert_eq!(fb.data(), core::ptr::null());
                fb.free();
            }

            // Load the last few bytes of the data.
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

    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_out_of_bounds_load_fails() {
        setup();
        for alignment in alignments() {
            // Create a temp file; contents don't matter.
            let data = [0u8; 256];
            let tf = TempFile::new(&data);

            let fd = open_rdonly(&tf);

            // Wrap it in a loader.
            let fdl =
                FileDescriptorDataLoader::from_file_descriptor_uri(fd_uri(fd).as_ptr(), alignment);
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

    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn/test]
    // from_file_descriptor_uri() rejects non-power-of-2 alignments via is_power_of_2.
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_bad_alignment_fails() {
        setup();
        for alignment in alignments() {
            // Create a temp file; contents don't matter.
            let data = [0u8; 256];
            let tf = TempFile::new(&data);

            // Creating a loader with default alignment works fine.
            {
                let fd = open_rdonly(&tf);
                let fdl = FileDescriptorDataLoader::from_file_descriptor_uri(
                    fd_uri(fd).as_ptr(),
                    alignment,
                );
                assert_eq!(ResultExt::error(&fdl), Error::Ok);
            }

            // Bad alignments fail.
            let bad_alignments: [usize; 4] = [0, 3, 5, 17];
            for bad_alignment in bad_alignments {
                let fd = open_rdonly(&tf);
                let fdl = FileDescriptorDataLoader::from_file_descriptor_uri(
                    fd_uri(fd).as_ptr(),
                    bad_alignment,
                );
                assert_eq!(ResultExt::error(&fdl), Error::InvalidArgument);
            }
        }
    }

    // Tests that the move ctor works.
    //
    // PORT-NOTE: as in file_data_loader.rs, the C++ move ctor leaves the source
    // reporting `Error::InvalidState`; in Rust a moved-from value cannot be
    // named, so only the "new loader works" half is ported.
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn/test]
    //
    // The deleted copy-assign (`operator=`) collapses onto the move-only loader
    // in Rust (no `Copy`/`Clone`): the move transfers unique ownership of the
    // fd/uri copy, and the single surviving owner still works.
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.operator-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_move_ctor() {
        setup();
        for alignment in alignments() {
            // Create a loader.
            let contents = b"FILE_CONTENTS";
            let tf = TempFile::new(contents);
            let fd = open_rdonly(&tf);

            let fdl =
                FileDescriptorDataLoader::from_file_descriptor_uri(fd_uri(fd).as_ptr(), alignment);
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
    //
    // PORT-NOTE: the C++ test named `DEPRECATEDFrom` actually calls the current
    // `fromFileDescriptorUri` (there is no deprecated capital-`From` overload on
    // this loader), so the port calls `from_file_descriptor_uri` directly.
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn/test]
    // [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn/test]
    #[test]
    fn file_descriptor_data_loader_test_deprecated_from() {
        setup();
        for alignment in alignments() {
            // Write some heterogeneous data to a file.
            let mut data = [0u8; 256];
            for i in 0..data.len() {
                data[i] = i as u8;
            }
            let tf = TempFile::new(&data);

            let fd = open_rdonly(&tf);

            // Wrap it in a loader.
            let fdl =
                FileDescriptorDataLoader::from_file_descriptor_uri(fd_uri(fd).as_ptr(), alignment);
            assert_eq!(ResultExt::error(&fdl), Error::Ok);
            let fdl = fdl.unwrap();

            // size() should succeed and reflect the total size.
            let size = fdl.size();
            assert_eq!(ResultExt::error(&size), Error::Ok);
            assert_eq!(*ResultExt::get(&size), data.len());
        }
    }
}

// Mirrors `alignof(std::max_align_t)`, used by tests for the parameterized
// alignment values.
#[cfg(test)]
#[repr(C)]
struct MaxAlign {
    _a: u128,
    _b: f64,
}
