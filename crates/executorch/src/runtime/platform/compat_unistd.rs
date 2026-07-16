//! Literal port of runtime/platform/compat_unistd.h.
//!
//! unistd.h related macros for POSIX/Windows compatibility.
//!
//! On non-Windows (`!_WIN64`) builds this shim does not exist; the system
//! `<unistd.h>` `pread` is used instead. The Win64 positioned-read shim below is
//! gated to Windows targets, mirroring the C++ `#if !defined(_WIN64)` guard.
//!
//! PORT-NOTE(wave-3): no tests — the `pread` shim only exists (and only
//! compiles) on 64-bit Windows, where it drives `ReadFile`/`GetOverlappedResult`.
//! Coverage of `compat-unistd.pread-fn` is a recorded platform gap.

// [spec:et:def:compat-unistd.pread-fn]
// [spec:et:sem:compat-unistd.pread-fn]
//
// PORT-NOTE: This is the `_WIN64`-only shim. It calls Win32 APIs
// (`ReadFile`/`GetOverlappedResult`/`_get_osfhandle`) via `windows-sys`-style
// externs. Because the executorch crate does not depend on a Windows API crate,
// the raw `extern "C"` declarations are provided inline; this compiles only when
// targeting 64-bit Windows.
#[cfg(all(windows, target_pointer_width = "64"))]
pub use win64::pread;

#[cfg(all(windows, target_pointer_width = "64"))]
mod win64 {
    use core::ffi::{c_int, c_void};

    type DWORD = u32;
    type BOOL = c_int;
    type HANDLE = *mut c_void;

    const ERROR_IO_PENDING: DWORD = 997;
    const ERROR_HANDLE_EOF: DWORD = 38;
    const EIO: c_int = 5;
    const TRUE: BOOL = 1;

    #[repr(C)]
    struct OVERLAPPED {
        internal: usize,
        internal_high: usize,
        offset: DWORD,
        offset_high: DWORD,
        h_event: HANDLE,
    }

    unsafe extern "C" {
        fn ReadFile(
            hFile: HANDLE,
            lpBuffer: *mut c_void,
            nNumberOfBytesToRead: DWORD,
            lpNumberOfBytesRead: *mut DWORD,
            lpOverlapped: *mut OVERLAPPED,
        ) -> BOOL;
        fn GetLastError() -> DWORD;
        fn GetOverlappedResult(
            hFile: HANDLE,
            lpOverlapped: *mut OVERLAPPED,
            lpNumberOfBytesTransferred: *mut DWORD,
            bWait: BOOL,
        ) -> BOOL;
        fn _get_osfhandle(fd: c_int) -> isize;
        fn _errno() -> *mut c_int;
    }

    #[allow(non_snake_case)]
    pub fn pread(fd: c_int, buf: *mut c_void, nbytes: usize, offset: usize) -> isize {
        // The offset for ReadFile.
        let mut overlapped: OVERLAPPED = unsafe { core::mem::zeroed() };
        overlapped.offset = offset as DWORD;
        overlapped.offset_high = (offset >> 32) as DWORD;

        let mut result: BOOL; // The result of ReadFile.
        let mut bytes_read: DWORD = 0; // The number of bytes read.
        let file: HANDLE = unsafe { _get_osfhandle(fd) as HANDLE };

        unsafe {
            result = ReadFile(file, buf, nbytes as DWORD, &mut bytes_read, &mut overlapped);
            let mut error = GetLastError();
            if result == 0 {
                if error == ERROR_IO_PENDING {
                    result = GetOverlappedResult(file, &mut overlapped, &mut bytes_read, TRUE);
                    if result == 0 {
                        error = GetLastError();
                    }
                }
            }
            if result == 0 {
                // Translate error into errno.
                match error {
                    ERROR_HANDLE_EOF => {
                        *_errno() = 0;
                    }
                    _ => {
                        *_errno() = EIO;
                    }
                }
                return -1;
            }
            bytes_read as isize
        }
    }
}
