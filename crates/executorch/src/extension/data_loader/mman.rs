//! Literal port of extension/data_loader/mman.h.
//!
//! This file ensures that mman.h compatible functions are defined for windows
//! and posix environments. The POSIX shapes live under `#[cfg(unix)]`; the
//! Windows shapes under `#[cfg(windows)]` forward to the `mman_windows` shim.

/// Platform mmap offset integer type. POSIX `off_t` on unix, `u64` on Windows.
#[cfg(unix)]
pub type MmapOffset = libc::off_t;
#[cfg(windows)]
pub type MmapOffset = u64;

// ---------------------------------------------------------------------------
// POSIX (non-_WIN32) branch.
// ---------------------------------------------------------------------------
#[cfg(unix)]
// [spec:et:def:mman.get-os-page-size-fn]
// [spec:et:sem:mman.get-os-page-size-fn]
pub fn get_os_page_size() -> libc::c_long {
    // PORT-NOTE: the C++ POSIX variant declares the return type `size_t`, but
    // callers (MmapDataLoader::from) treat it as `long` and test `< 0`. The
    // `long` (`c_long`) return type is used here so the negative check the
    // callers rely on is expressible; `sysconf` itself returns `long`.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) }
}

/// Platform-specific file stat function.
// [spec:et:def:mman.get-file-stat-fn]
// [spec:et:sem:mman.get-file-stat-fn]
#[cfg(unix)]
pub fn get_file_stat(fd: libc::c_int, out_size: &mut usize) -> libc::c_int {
    let mut st: libc::stat = unsafe { core::mem::zeroed() };
    let err = unsafe { libc::fstat(fd, &mut st) };
    if err >= 0 {
        *out_size = st.st_size as usize;
    }
    err
}

/// Platform-specific mmap offset type conversion.
// [spec:et:def:mman.get-mmap-offset-fn]
// [spec:et:sem:mman.get-mmap-offset-fn]
#[cfg(unix)]
pub fn get_mmap_offset(offset: usize) -> MmapOffset {
    offset as libc::off_t
}

/// Hint the kernel to prefetch pages eagerly and to optimize for sequential
/// reads. Intended to reduce page-fault stutter during model initialization
/// when the caller does not want to mlock the pages into RAM.
///
/// MADV_WILLNEED / MADV_SEQUENTIAL are absent on some POSIX libcs (e.g. the
/// Hexagon DSP toolchain).
// [spec:et:def:mman.madvise-pages-willneed-sequential-fn]
// [spec:et:sem:mman.madvise-pages-willneed-sequential-fn]
#[cfg(unix)]
pub fn madvise_pages_willneed_sequential(addr: *mut core::ffi::c_void, len: usize) {
    // PORT-NOTE: the C++ guards each call with `#ifdef MADV_WILLNEED` /
    // `#ifdef MADV_SEQUENTIAL` because those constants are missing on some
    // libcs. `libc` exposes both on the unix targets it supports, so both
    // calls are always emitted here.
    unsafe {
        libc::madvise(addr, len, libc::MADV_WILLNEED);
    }
    unsafe {
        libc::madvise(addr, len, libc::MADV_SEQUENTIAL);
    }
}

/// On Apple platforms, schedule kernel read-ahead on the file descriptor itself
/// via fcntl(F_RDADVISE). This is more aggressive than madvise for cold starts:
/// it brings pages into the unified buffer cache so first-touch faults are
/// serviced from RAM instead of storage. No-op on non-Apple POSIX platforms.
// [spec:et:def:mman.fcntl-rdadvise-apple-fn]
// [spec:et:sem:mman.fcntl-rdadvise-apple-fn]
#[cfg(all(unix, target_vendor = "apple"))]
pub fn fcntl_rdadvise_apple(fd: libc::c_int, file_size: usize) {
    let mut advice: libc::radvisory = unsafe { core::mem::zeroed() };
    advice.ra_offset = 0;
    advice.ra_count = file_size as libc::c_int;
    unsafe {
        libc::fcntl(fd, libc::F_RDADVISE, &mut advice);
    }
}

#[cfg(all(unix, not(target_vendor = "apple")))]
pub fn fcntl_rdadvise_apple(fd: libc::c_int, file_size: usize) {
    let _ = fd;
    let _ = file_size;
}

// ---------------------------------------------------------------------------
// Windows (_WIN32) branch.
// ---------------------------------------------------------------------------
#[cfg(windows)]
pub fn get_os_page_size() -> libc::c_long {
    crate::extension::data_loader::mman_windows::get_os_page_size()
}

#[cfg(windows)]
pub fn get_file_stat(fd: libc::c_int, out_size: &mut usize) -> libc::c_int {
    let mut st: libc::stat64 = unsafe { core::mem::zeroed() };
    let err = unsafe { libc::fstat64(fd, &mut st) };
    if err >= 0 {
        *out_size = st.st_size as usize;
    }
    err
}

#[cfg(windows)]
pub fn get_mmap_offset(offset: usize) -> MmapOffset {
    offset as u64
}

#[cfg(windows)]
pub fn madvise_pages_willneed_sequential(addr: *mut core::ffi::c_void, len: usize) {
    let _ = addr;
    let _ = len;
}

#[cfg(windows)]
pub fn fcntl_rdadvise_apple(fd: libc::c_int, file_size: usize) {
    let _ = fd;
    let _ = file_size;
}
