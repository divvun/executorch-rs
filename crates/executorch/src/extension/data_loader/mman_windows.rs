//! Literal port of extension/data_loader/mman_windows.cpp / mman_windows.h.
//!
//! Adapted from: https://code.google.com/archive/p/mman-win32/
//!
//! A light implementation of the mmap functions for Windows, wrapping the
//! Win32 memory-mapping API. This module is entirely `#[cfg(windows)]`.
//!
//! PORT-NOTE: the ExecuTorch runtime crate depends only on `libc`, which does
//! not expose the Win32 memory APIs used here. The required `windows.h`
//! declarations (`VirtualLock`, `CreateFileMapping`, etc.), the handle/DWORD
//! typedefs, and the `PROT_*`/`MAP_*`/`PAGE_*`/`FILE_MAP_*` constants are
//! therefore declared locally in this module, mirroring `mman_windows.h` and
//! the Win32 SDK signatures.
//!
//! PORT-NOTE(wave-3): no tests — the entire module is `#[cfg(windows)]` and
//! every function drives Win32 APIs, so nothing here can genuinely execute on
//! a unix host. Test coverage for all `mman-windows.*` symbols is a recorded
//! platform gap until a Windows CI leg runs the crate's tests.
#![cfg(windows)]

use core::ffi::c_void;

#[allow(non_camel_case_types)]
type DWORD = u32;
#[allow(non_camel_case_types)]
type BOOL = i32;
#[allow(non_camel_case_types)]
type HANDLE = *mut c_void;
#[allow(non_camel_case_types)]
type HRESULT = i32;
#[allow(non_camel_case_types)]
type LPVOID = *mut c_void;
#[allow(non_camel_case_types)]
type LPCVOID = *const c_void;
#[allow(non_camel_case_types)]
type SIZE_T = usize;

// mman_windows.h constants.
pub const PROT_NONE: i32 = 0;
pub const PROT_READ: i32 = 1;
pub const PROT_WRITE: i32 = 2;
pub const PROT_EXEC: i32 = 4;

pub const MAP_FILE: i32 = 0;
pub const MAP_SHARED: i32 = 1;
pub const MAP_PRIVATE: i32 = 2;
pub const MAP_TYPE: i32 = 0xf;
pub const MAP_FIXED: i32 = 0x10;
pub const MAP_ANONYMOUS: i32 = 0x20;
pub const MAP_ANON: i32 = MAP_ANONYMOUS;

pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void; // (void*)-1

pub const MS_ASYNC: i32 = 1;
pub const MS_SYNC: i32 = 2;
pub const MS_INVALIDATE: i32 = 4;

// Win32 flags used by this shim.
const FILE_MAP_READ: DWORD = 0x0004;
const FILE_MAP_WRITE: DWORD = 0x0002;
const FILE_MAP_EXECUTE: DWORD = 0x0020;

const PAGE_READONLY: DWORD = 0x02;
const PAGE_READWRITE: DWORD = 0x04;
const PAGE_EXECUTE_READ: DWORD = 0x20;
const PAGE_EXECUTE_READWRITE: DWORD = 0x40;

const STATUS_SECTION_TOO_BIG: DWORD = 0xC0000040;
const ERROR_ARITHMETIC_OVERFLOW: DWORD = 534;

const INVALID_HANDLE_VALUE: HANDLE = usize::MAX as HANDLE; // (HANDLE)-1

const S_OK: HRESULT = 0;

const EPERM: i32 = 1;
const EINVAL: i32 = 22;
const EBADF: i32 = 9;

// HRESULT_FROM_WIN32(x): ((HRESULT)(x) <= 0 ? (HRESULT)(x)
//   : (HRESULT)(((x) & 0x0000FFFF) | (FACILITY_WIN32 << 16) | 0x80000000))
const FACILITY_WIN32: u32 = 7;
#[inline]
fn hresult_from_win32(x: DWORD) -> HRESULT {
    let hr = x as i32;
    if hr <= 0 {
        hr
    } else {
        (((x & 0x0000FFFF) | (FACILITY_WIN32 << 16) | 0x8000_0000) as u32) as i32
    }
}

#[inline]
fn failed(hr: HRESULT) -> bool {
    hr < 0
}

#[inline]
fn succeeded(hr: HRESULT) -> bool {
    hr >= 0
}

#[repr(C)]
struct SYSTEM_INFO {
    w_processor_architecture: u16,
    w_reserved: u16,
    dw_page_size: DWORD,
    lp_minimum_application_address: LPVOID,
    lp_maximum_application_address: LPVOID,
    dw_active_processor_mask: usize,
    dw_number_of_processors: DWORD,
    dw_processor_type: DWORD,
    dw_allocation_granularity: DWORD,
    w_processor_level: u16,
    w_processor_revision: u16,
}

unsafe extern "system" {
    fn GetLastError() -> DWORD;
    fn GetCurrentProcess() -> HANDLE;
    fn GetSystemInfo(lp_system_info: *mut SYSTEM_INFO);
    fn GetProcessWorkingSetSize(
        h_process: HANDLE,
        lp_minimum_working_set_size: *mut SIZE_T,
        lp_maximum_working_set_size: *mut SIZE_T,
    ) -> BOOL;
    fn SetProcessWorkingSetSize(
        h_process: HANDLE,
        dw_minimum_working_set_size: SIZE_T,
        dw_maximum_working_set_size: SIZE_T,
    ) -> BOOL;
    fn VirtualLock(lp_address: LPVOID, dw_size: SIZE_T) -> BOOL;
    fn VirtualUnlock(lp_address: LPVOID, dw_size: SIZE_T) -> BOOL;
    fn VirtualProtect(
        lp_address: LPVOID,
        dw_size: SIZE_T,
        fl_new_protect: DWORD,
        lpfl_old_protect: *mut DWORD,
    ) -> BOOL;
    fn CreateFileMappingA(
        h_file: HANDLE,
        lp_file_mapping_attributes: LPVOID,
        fl_protect: DWORD,
        dw_maximum_size_high: DWORD,
        dw_maximum_size_low: DWORD,
        lp_name: *const i8,
    ) -> HANDLE;
    fn MapViewOfFile(
        h_file_mapping_object: HANDLE,
        dw_desired_access: DWORD,
        dw_file_offset_high: DWORD,
        dw_file_offset_low: DWORD,
        dw_number_of_bytes_to_map: SIZE_T,
    ) -> LPVOID;
    fn UnmapViewOfFile(lp_base_address: LPCVOID) -> BOOL;
    fn FlushViewOfFile(lp_base_address: LPCVOID, dw_number_of_bytes_to_flush: SIZE_T) -> BOOL;
    fn CloseHandle(h_object: HANDLE) -> BOOL;
}

unsafe extern "C" {
    fn _get_osfhandle(fd: i32) -> isize;
}

// PORT-NOTE: `errno` is accessed via the MSVCRT `_errno()` accessor so writes
// are visible to callers checking `errno`, mirroring the C++ `errno = ...`.
unsafe extern "C" {
    fn _errno() -> *mut i32;
}
#[inline]
fn set_errno(value: i32) {
    unsafe {
        *_errno() = value;
    }
}

// [spec:et:def:mman-windows.get-os-page-size-fn]
// [spec:et:sem:mman.get-os-page-size-fn]
#[allow(non_snake_case)]
pub fn get_os_page_size() -> libc::c_long {
    let mut si: SYSTEM_INFO = unsafe { core::mem::zeroed() };
    unsafe {
        GetSystemInfo(&mut si);
    }
    let pagesize: libc::c_long = if si.dw_allocation_granularity > si.dw_page_size {
        si.dw_allocation_granularity as libc::c_long
    } else {
        si.dw_page_size as libc::c_long
    };
    pagesize
}

// [spec:et:def:mman-windows.try-grow-process-memory-working-set-fn]
// [spec:et:sem:mman-windows.try-grow-process-memory-working-set-fn]
fn try_grow_process_memory_working_set(dw_size_required: DWORD) -> HRESULT {
    // Get current working set
    let mut min_working_set_initial: SIZE_T = 0;
    let mut max_working_set: SIZE_T = 0;
    if unsafe {
        GetProcessWorkingSetSize(
            GetCurrentProcess(),
            &mut min_working_set_initial,
            &mut max_working_set,
        )
    } == 0
    {
        return unsafe { GetLastError() } as HRESULT;
    }

    // Calculate new sizes
    let min_working_set: SIZE_T = min_working_set_initial.wrapping_add(dw_size_required as SIZE_T);
    if min_working_set < min_working_set_initial {
        return hresult_from_win32(ERROR_ARITHMETIC_OVERFLOW);
    }

    if max_working_set < min_working_set {
        max_working_set = min_working_set;
    }

    // Grow working set
    if unsafe { SetProcessWorkingSetSize(GetCurrentProcess(), min_working_set, max_working_set) }
        == 0
    {
        return unsafe { GetLastError() } as HRESULT;
    }
    S_OK
}

// [spec:et:def:mman-windows.virtual-lock-fn]
// [spec:et:sem:mman-windows.virtual-lock-fn]
fn virtual_lock(p_mem: *mut c_void, dw_size: DWORD) -> HRESULT {
    if unsafe { VirtualLock(p_mem, dw_size as SIZE_T) } == 0 {
        return unsafe { GetLastError() } as HRESULT;
    }
    S_OK
}

// [spec:et:def:mman-windows.virtual-lock-allowing-working-set-growth-fn]
// [spec:et:sem:mman-windows.virtual-lock-allowing-working-set-growth-fn]
fn virtual_lock_allowing_working_set_growth(p_mem: *mut c_void, dw_size: DWORD) -> HRESULT {
    let hr = virtual_lock(p_mem, dw_size);

    if hr == hresult_from_win32(STATUS_SECTION_TOO_BIG) {
        // Attempt to grow the process working set and try again
        let grow = try_grow_process_memory_working_set(dw_size);
        if failed(grow) {
            return grow;
        }
        let retry = virtual_lock(p_mem, dw_size);
        if failed(retry) {
            return retry;
        }
    }

    hr
}

// [spec:et:def:mman-windows.map-mman-error-fn]
// [spec:et:sem:mman-windows.map-mman-error-fn]
#[allow(unused_variables)]
fn map_mman_error(err: DWORD, deferr: i32) -> i32 {
    if err == 0 {
        return 0;
    }
    // TODO: implement
    err as i32
}

// [spec:et:def:mman-windows.map-mmap-prot-page-fn]
// [spec:et:sem:mman-windows.map-mmap-prot-page-fn]
fn map_mmap_prot_page(prot: i32) -> DWORD {
    let mut protect: DWORD = 0;

    if prot == PROT_NONE {
        return protect;
    }
    if (prot & PROT_EXEC) != 0 {
        protect = if (prot & PROT_WRITE) != 0 {
            PAGE_EXECUTE_READWRITE
        } else {
            PAGE_EXECUTE_READ
        };
    } else {
        protect = if (prot & PROT_WRITE) != 0 {
            PAGE_READWRITE
        } else {
            PAGE_READONLY
        };
    }
    protect
}

// [spec:et:def:mman-windows.map-mmap-prot-file-fn]
// [spec:et:sem:mman-windows.map-mmap-prot-file-fn]
fn map_mmap_prot_file(prot: i32) -> DWORD {
    let mut desired_access: DWORD = 0;

    if prot == PROT_NONE {
        return desired_access;
    }
    if (prot & PROT_READ) != 0 {
        desired_access |= FILE_MAP_READ;
    }
    if (prot & PROT_WRITE) != 0 {
        desired_access |= FILE_MAP_WRITE;
    }
    if (prot & PROT_EXEC) != 0 {
        desired_access |= FILE_MAP_EXECUTE;
    }
    desired_access
}

// [spec:et:def:mman-windows.mmap-fn]
// [spec:et:sem:mman-windows.mmap-fn]
#[allow(non_snake_case)]
pub fn mmap(
    addr: *mut c_void,
    len: usize,
    prot: i32,
    flags: i32,
    fildes: i32,
    off: u64,
) -> *mut c_void {
    let _ = addr;
    let fm: HANDLE;
    let h: HANDLE;
    let mut map: *mut c_void = MAP_FAILED;

    set_errno(0);

    if len == 0
        // Unsupported flag combinations
        || (flags & MAP_FIXED) != 0
        // Unsupported protection combinations
        || prot == PROT_EXEC
    {
        set_errno(EINVAL);
        return MAP_FAILED;
    }

    if off > u64::MAX - (len as u64) {
        set_errno(EINVAL);
        return MAP_FAILED;
    }

    let max_size: u64 = off + (len as u64);

    let dw_file_offset_low: DWORD = (off & 0xFFFFFFFF) as DWORD;
    let dw_file_offset_high: DWORD = ((off >> 32) & 0xFFFFFFFF) as DWORD;
    let protect: DWORD = map_mmap_prot_page(prot);
    let desired_access: DWORD = map_mmap_prot_file(prot);

    let dw_max_size_low: DWORD = (max_size & 0xFFFFFFFF) as DWORD;
    let dw_max_size_high: DWORD = ((max_size >> 32) & 0xFFFFFFFF) as DWORD;

    h = if (flags & MAP_ANONYMOUS) == 0 {
        unsafe { _get_osfhandle(fildes) as HANDLE }
    } else {
        INVALID_HANDLE_VALUE
    };

    if (flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE {
        set_errno(EBADF);
        return MAP_FAILED;
    }

    fm = unsafe {
        CreateFileMappingA(
            h,
            core::ptr::null_mut(),
            protect,
            dw_max_size_high,
            dw_max_size_low,
            core::ptr::null(),
        )
    };

    if fm.is_null() {
        set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));
        return MAP_FAILED;
    }

    map = unsafe {
        MapViewOfFile(
            fm,
            desired_access,
            dw_file_offset_high,
            dw_file_offset_low,
            len,
        )
    };

    unsafe {
        CloseHandle(fm);
    }

    if map.is_null() {
        set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));
        return MAP_FAILED;
    }

    map
}

// [spec:et:def:mman-windows.munmap-fn]
// [spec:et:sem:mman-windows.munmap-fn]
pub fn munmap(addr: *mut c_void, len: usize) -> i32 {
    let _ = len;
    if unsafe { UnmapViewOfFile(addr) } != 0 {
        return 0;
    }

    set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));

    -1
}

// [spec:et:def:mman-windows.mprotect-fn]
// [spec:et:sem:mman-windows.mprotect-fn]
pub fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32 {
    let new_protect: DWORD = map_mmap_prot_page(prot);
    let mut old_protect: DWORD = 0;

    if unsafe { VirtualProtect(addr, len, new_protect, &mut old_protect) } != 0 {
        return 0;
    }

    set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));

    -1
}

// [spec:et:def:mman-windows.msync-fn]
// [spec:et:sem:mman-windows.msync-fn]
pub fn msync(addr: *mut c_void, len: usize, flags: i32) -> i32 {
    let _ = flags;
    if unsafe { FlushViewOfFile(addr, len) } != 0 {
        return 0;
    }

    set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));

    -1
}

// [spec:et:def:mman-windows.mlock-fn]
// [spec:et:sem:mman-windows.mlock-fn]
pub fn mlock(addr: *const c_void, len: usize) -> i32 {
    let hr = virtual_lock_allowing_working_set_growth(addr as LPVOID, len as DWORD);
    if succeeded(hr) {
        return 0;
    }

    set_errno(map_mman_error(hr as DWORD, EPERM));

    -1
}

// [spec:et:def:mman-windows.munlock-fn]
// [spec:et:sem:mman-windows.munlock-fn]
pub fn munlock(addr: *const c_void, len: usize) -> i32 {
    if unsafe { VirtualUnlock(addr as LPVOID, len as SIZE_T) } != 0 {
        return 0;
    }

    set_errno(map_mman_error(unsafe { GetLastError() }, EPERM));

    -1
}
