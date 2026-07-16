//! Literal port of backends/xnnpack/runtime/XNNStatus.h.

// PORT-NOTE: The C++ header includes <xnnpack.h> and takes `enum xnn_status` as
// input. The XNNPACK submodule is not checked out in this tree, so the
// `xnn_status` type is defined in `super::sys` (feature-gated) mirroring the
// XNNPACK C ABI. This function operates purely on the integer value of the
// enum, so it is available only behind the `xnnpack` feature to match the
// header's dependency on `xnn_status`.
#[cfg(feature = "xnnpack")]
use super::sys::xnn_status;

const K_OFFSET: [u16; 7] = [0, 19, 44, 73, 98, 131, 163];

const K_DATA: &[u8] = b"xnn_status_success\0\
xnn_status_uninitialized\0\
xnn_status_invalid_parameter\0\
xnn_status_invalid_state\0\
xnn_status_unsupported_parameter\0\
xnn_status_unsupported_hardware\0\
xnn_status_out_of_memory\0";

// [spec:et:def:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn]
// [spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn]
//
// PORT-NOTE: The C++ returns `const char*` into the interned `kData` buffer.
// Here we return a `&'static core::ffi::CStr` pointing into `K_DATA` at the
// same offset, preserving the "no allocation, no copy, caller does not own"
// contract. The C++ `assert(type <= xnn_status_out_of_memory)` is a debug-only
// bounds check; the Rust `K_OFFSET[type]` index (and the `CStr` construction)
// panics on out-of-range input, matching "treat out-of-range as a programming
// error".
#[cfg(feature = "xnnpack")]
#[inline]
pub fn xnn_status_to_string(type_: xnn_status) -> &'static core::ffi::CStr {
    let type_ = type_.0 as usize;
    let start = K_OFFSET[type_] as usize;
    // The bytes from `start` up to and including the terminating NUL form the
    // name string; find the terminator so `CStr::from_bytes_with_nul` sees a
    // single well-formed NUL-terminated slice.
    let mut end = start;
    while K_DATA[end] != 0 {
        end += 1;
    }
    core::ffi::CStr::from_bytes_with_nul(&K_DATA[start..=end]).unwrap()
}

#[cfg(all(test, feature = "xnnpack"))]
mod tests {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::xnn_status;

    // Each in-range status maps to its interned name, matching the C++
    // `&kData[kOffset[type]]` lookup exactly (offsets 0,19,44,73,98,131,163).
    // [spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn/test]
    #[test]
    fn xnn_status_to_string_maps_each_code() {
        let cases: [(xnn_status, &str); 7] = [
            (xnn_status::SUCCESS, "xnn_status_success"),
            (xnn_status::UNINITIALIZED, "xnn_status_uninitialized"),
            (
                xnn_status::INVALID_PARAMETER,
                "xnn_status_invalid_parameter",
            ),
            (xnn_status::INVALID_STATE, "xnn_status_invalid_state"),
            (
                xnn_status::UNSUPPORTED_PARAMETER,
                "xnn_status_unsupported_parameter",
            ),
            (
                xnn_status::UNSUPPORTED_HARDWARE,
                "xnn_status_unsupported_hardware",
            ),
            (xnn_status::OUT_OF_MEMORY, "xnn_status_out_of_memory"),
        ];
        for (status, expected) in cases {
            let s = xnn_status_to_string(status);
            assert_eq!(s.to_str().unwrap(), expected);
        }
    }

    // Distinct codes return distinct pointers into the interned buffer, as the
    // C++ returns `&kData[kOffset[type]]` at distinct offsets.
    // [spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn/test]
    #[test]
    fn xnn_status_to_string_returns_distinct_pointers() {
        let a = xnn_status_to_string(xnn_status::SUCCESS);
        let b = xnn_status_to_string(xnn_status::OUT_OF_MEMORY);
        assert_ne!(a.as_ptr(), b.as_ptr());
        // Successive calls with the same code return the same interned pointer
        // (no allocation, no copy).
        let a2 = xnn_status_to_string(xnn_status::SUCCESS);
        assert_eq!(a.as_ptr(), a2.as_ptr());
    }

    // Out-of-range status is a programming error: the C++ `assert` (debug) and
    // the Rust `K_OFFSET[type]` index both reject it. Mirrors an ET_CHECK death.
    // [spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn/test]
    #[test]
    #[should_panic]
    fn xnn_status_to_string_out_of_range_panics() {
        // 7 (REINITIALIZATION_REQUIRED) is > xnn_status_out_of_memory (6), so it
        // is out of the interned range and must panic on the K_OFFSET index.
        let _ = xnn_status_to_string(xnn_status(7));
    }
}
