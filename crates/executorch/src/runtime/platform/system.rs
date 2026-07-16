//! Literal port of runtime/platform/system.h.
//!
//! Platform abstraction layer to allow individual host OS to override
//! symbols in ExecuTorch. PAL functions are defined as C functions so an
//! implementer can use C in lieu of C++.

use core::ffi::{c_char, c_void};

pub const DYNAMIC_LIBRARY_NOT_SUPPORTED: &core::ffi::CStr = c"NOT_SUPPORTED";
pub const DYNAMIC_LIBRARY_NOT_FOUND: &core::ffi::CStr = c"NOT_FOUND";

/// Return shared library.
///
/// @param[in] addr Address to the symbol we are looking for in shared libraries.
/// @retval The path to the shared library containing the symbol.
// [spec:et:def:system.et-pal-get-shared-library-name-fn]
// [spec:et:sem:system.et-pal-get-shared-library-name-fn]
//
// PORT-NOTE: the C++ `ET_USE_LIBDL` compile-time switch is mapped to the
// `et_use_libdl` cfg flag (off by default, matching the C++ default which
// leaves `ET_USE_LIBDL` undefined). The libdl branch calls `dladdr` from libc.
#[unsafe(no_mangle)]
pub extern "C" fn et_pal_get_shared_library_name(addr: *const c_void) -> *const c_char {
    #[cfg(et_use_libdl)]
    {
        let mut info: libc::Dl_info = unsafe { core::mem::zeroed() };
        if unsafe { libc::dladdr(addr, &mut info) } != 0 && !info.dli_fname.is_null() {
            return info.dli_fname;
        } else {
            return DYNAMIC_LIBRARY_NOT_FOUND.as_ptr();
        }
    }
    #[allow(unreachable_code)]
    {
        let _ = addr;
        DYNAMIC_LIBRARY_NOT_SUPPORTED.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default (non-libdl) configuration: the lookup block is compiled out, `addr`
    // is discarded, and the function returns the constant string
    // DYNAMIC_LIBRARY_NOT_SUPPORTED ("NOT_SUPPORTED"). No allocation occurs; the
    // returned pointer is the static literal, independent of the address passed
    // (including null).
    // [spec:et:sem:system.et-pal-get-shared-library-name-fn/test]
    #[test]
    #[cfg(not(et_use_libdl))]
    fn get_shared_library_name_returns_not_supported() {
        let dummy: u32 = 0;
        let ret = et_pal_get_shared_library_name(&dummy as *const u32 as *const c_void);
        assert_eq!(ret, DYNAMIC_LIBRARY_NOT_SUPPORTED.as_ptr());
        let s = unsafe { core::ffi::CStr::from_ptr(ret) };
        assert_eq!(s, c"NOT_SUPPORTED");

        // A null address yields the same static literal.
        let ret_null = et_pal_get_shared_library_name(core::ptr::null());
        assert_eq!(ret_null, DYNAMIC_LIBRARY_NOT_SUPPORTED.as_ptr());
    }
}
