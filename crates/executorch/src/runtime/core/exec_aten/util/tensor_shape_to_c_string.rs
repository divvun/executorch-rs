//! Literal port of runtime/core/exec_aten/util/tensor_shape_to_c_string.cpp
//! (and the constants from tensor_shape_to_c_string.h).

use crate::runtime::core::span::Span;

// PORT-NOTE: `kTensorDimensionLimit` is defined in
// runtime/core/exec_aten/util/tensor_dimension_limit.h, which has no ported
// `tensor_dimension_limit.rs` target yet. The constant is inlined here with the
// same value (16) until that module lands. Unresolved cross-module reference.
const K_TENSOR_DIMENSION_LIMIT: usize = 16;

/// Maximum size of a string returned by `tensor_shape_to_c_string`, for stack
/// allocation.
// [spec:et:def:tensor-shape-to-c-string constant (header)]
pub const K_TENSOR_SHAPE_STRING_SIZE_LIMIT: usize = 1 /* opening parenthesis */
    + 10 * K_TENSOR_DIMENSION_LIMIT /* maximum digits we will print */
    + 2 * K_TENSOR_DIMENSION_LIMIT /* comma and space after each item */
    + 1 /* padding for temporary NUL terminator */;

pub mod internal {
    pub const K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT: usize = i32::MAX as usize;
}

/// Trait abstracting over the two `SizesType` element widths (`i32` / `i64`)
/// that the templated impl accepts, mirroring the C++ template parameter.
pub trait SizesType: Copy {
    fn lt_zero(self) -> bool;
    fn as_usize(self) -> usize;
    fn as_u32(self) -> u32;
}

impl SizesType for i32 {
    fn lt_zero(self) -> bool {
        self < 0
    }
    fn as_usize(self) -> usize {
        self as usize
    }
    fn as_u32(self) -> u32 {
        self as u32
    }
}

impl SizesType for i64 {
    fn lt_zero(self) -> bool {
        self < 0
    }
    fn as_usize(self) -> usize {
        self as usize
    }
    fn as_u32(self) -> u32 {
        self as u32
    }
}

// [spec:et:def:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn]
// [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn]
//
// PORT-NOTE: C++ returns `std::array<char, N>`; `char` maps to
// `core::ffi::c_char`. The routine builds the string in a `[u8; N]` cursor
// buffer and converts to `[c_char; N]` on return.
fn tensor_shape_to_c_string_impl<S: SizesType>(
    shape: Span<S>,
) -> [core::ffi::c_char; K_TENSOR_SHAPE_STRING_SIZE_LIMIT] {
    let mut out: [u8; K_TENSOR_SHAPE_STRING_SIZE_LIMIT] = [0; K_TENSOR_SHAPE_STRING_SIZE_LIMIT];
    if shape.size() > K_TENSOR_DIMENSION_LIMIT {
        const K_LIMIT_EXCEEDED_ERROR: &[u8] = b"(ERR: tensor ndim exceeds limit)";
        const _: () = assert!(K_LIMIT_EXCEEDED_ERROR.len() + 1 <= K_TENSOR_SHAPE_STRING_SIZE_LIMIT);
        out[..K_LIMIT_EXCEEDED_ERROR.len()].copy_from_slice(K_LIMIT_EXCEEDED_ERROR);
        out[K_LIMIT_EXCEEDED_ERROR.len()] = 0;
        return out.map(|b| b as core::ffi::c_char);
    }

    // PORT-NOTE: The C++ writes through a raw cursor `p` and, for an empty
    // shape, executes `*(p - 2)`/`*(p - 1)` one byte BEFORE the buffer (an
    // out-of-bounds write) — the sem rule flags this as a degenerate/
    // unsupported input and directs a Rust port to treat empty shape explicitly
    // rather than reproduce the OOB write. This port yields "()" for the empty
    // shape. `p` is modeled as a byte index into `out`.
    let mut p: usize = 0;
    out[p] = b'(';
    p += 1;
    for i in 0..shape.size() {
        let elem = unsafe { *shape.index(i) };
        if elem.lt_zero() || elem.as_usize() > internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT {
            const _: () = assert!(internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT > 99999);
            const ERR: &[u8] = b"ERR, ";
            out[p..p + ERR.len()].copy_from_slice(ERR);
            p += ERR.len();
        } else {
            // snprintf(p, remaining, "%u, ", (uint32_t)elem): unsigned decimal
            // value followed by ", "; advance p by the count written (excluding
            // NUL).
            let rendered = format_u32_comma(elem.as_u32());
            let n = rendered.len();
            out[p..p + n].copy_from_slice(&rendered);
            p += n;
        }
    }
    // Overwrite the trailing ", " of the last element with ')' and NUL.
    // Only reached with a non-empty shape (see PORT-NOTE above); for the empty
    // shape the buffer already holds "(\0..." which reads as "(", so emit "()".
    if shape.size() == 0 {
        out[p] = b')';
        out[p + 1] = 0;
    } else {
        out[p - 2] = b')';
        out[p - 1] = 0;
    }
    out.map(|b| b as core::ffi::c_char)
}

/// Render `"%u, "` for a `u32` into a small byte buffer, returning the written
/// bytes (excluding NUL), mirroring `snprintf(..., "%u, ", elem)`.
fn format_u32_comma(value: u32) -> heapless_buf::Buf {
    let mut buf = heapless_buf::Buf::new();
    // up to 10 digits for u32
    let mut digits = [0u8; 10];
    let mut n = 0usize;
    let mut v = value;
    if v == 0 {
        digits[0] = b'0';
        n = 1;
    } else {
        while v > 0 {
            digits[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
    }
    for k in (0..n).rev() {
        buf.push(digits[k]);
    }
    buf.push(b',');
    buf.push(b' ');
    buf
}

mod heapless_buf {
    /// Minimal fixed-capacity byte buffer (max "4294967295, " = 12 bytes).
    pub struct Buf {
        data: [u8; 12],
        len: usize,
    }

    impl Buf {
        pub fn new() -> Self {
            Buf {
                data: [0; 12],
                len: 0,
            }
        }
        pub fn push(&mut self, b: u8) {
            self.data[self.len] = b;
            self.len += 1;
        }
        pub fn len(&self) -> usize {
            self.len
        }
    }

    impl core::ops::Deref for Buf {
        type Target = [u8];
        fn deref(&self) -> &[u8] {
            &self.data[..self.len]
        }
    }
}

/// Convert a shape to a NUL-terminated C string with limited size. If elements
/// of the shape are larger than `kMaximumPrintableTensorShapeElement`, those
/// elements will be rendered as ERR instead.
///
/// PORT-NOTE: C++ declares two same-named overloads (`Span<const int32_t>` and
/// `Span<const int64_t>`) to serve both ExecuTorch (`SizesType = int32_t`) and
/// ATen (`int64_t`) tensor sizes. Rust cannot overload by name; this `i32`
/// entry keeps the canonical `tensor_shape_to_c_string` name (the only variant
/// the portable runtime instantiates), and the `i64` variant is exposed as
/// `tensor_shape_to_c_string_i64`.
// [spec:et:def:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
// [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
pub fn tensor_shape_to_c_string(
    shape: Span<i32>,
) -> [core::ffi::c_char; K_TENSOR_SHAPE_STRING_SIZE_LIMIT] {
    tensor_shape_to_c_string_impl(shape)
}

// [spec:et:def:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
// [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn]
pub fn tensor_shape_to_c_string_i64(
    shape: Span<i64>,
) -> [core::ffi::c_char; K_TENSOR_SHAPE_STRING_SIZE_LIMIT] {
    tensor_shape_to_c_string_impl(shape)
}

// Literal port of runtime/core/exec_aten/util/test/tensor_shape_to_c_string_test.cpp.
#[cfg(test)]
mod tests {
    use super::*;

    // Reads the NUL-terminated C string held in the returned buffer, mirroring
    // gtest's `EXPECT_STREQ(str.data(), ...)`.
    fn c_str(buf: &[core::ffi::c_char; K_TENSOR_SHAPE_STRING_SIZE_LIMIT]) -> String {
        let mut s = String::new();
        for &b in buf.iter() {
            if b == 0 {
                break;
            }
            s.push(b as u8 as char);
        }
        s
    }

    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn/test]
    // also verifies tensor_shape_to_c_string_impl: the public entry point is a
    // thin wrapper that calls the generic *_impl over Span<i32>, so these
    // assertions exercise the impl's digit-rendering and trailing-", "->")" cursor
    // logic directly.
    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-impl-fn/test]
    #[test]
    fn tensor_shape_to_c_string_test_basic() {
        let mut sizes: [i32; 3] = [123, 456, 789];
        let str = tensor_shape_to_c_string(Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len()));
        assert_eq!(c_str(&str), "(123, 456, 789)");

        let mut one_size: [i32; 1] = [1234567890];
        let str =
            tensor_shape_to_c_string(Span::from_raw_parts(one_size.as_mut_ptr(), one_size.len()));
        assert_eq!(c_str(&str), "(1234567890)");
    }

    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn/test]
    #[test]
    fn tensor_shape_to_c_string_test_negative_items() {
        let mut sizes: [i32; 4] = [-1, -3, -2, 4];
        let str = tensor_shape_to_c_string(Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len()));
        assert_eq!(c_str(&str), "(ERR, ERR, ERR, 4)");

        let mut one_size: [i32; 1] = [-1234567890];
        let str =
            tensor_shape_to_c_string(Span::from_raw_parts(one_size.as_mut_ptr(), one_size.len()));
        // SizesType (i32) is signed, so the negative element renders as "(ERR)".
        assert_eq!(c_str(&str), "(ERR)");
    }

    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn/test]
    #[test]
    fn tensor_shape_to_c_string_test_maximum_element() {
        let mut sizes: [i32; 3] = [123, i32::MAX, 789];
        let str = tensor_shape_to_c_string(Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len()));
        // Mirrors the C++ ostringstream: "(" then each `elem, ` then the trailing
        // ", " overwritten so the last ", " becomes ")".
        let mut expected = String::from("(");
        for elem in sizes.iter() {
            expected.push_str(&format!("{}, ", elem));
        }
        expected.pop(); // drop trailing space
        expected.pop(); // drop trailing comma
        expected.push(')');
        assert_eq!(c_str(&str), expected);
    }

    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn/test]
    #[test]
    fn tensor_shape_to_c_string_test_maximum_length() {
        let mut sizes: [i32; K_TENSOR_DIMENSION_LIMIT] =
            [internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT as i32; K_TENSOR_DIMENSION_LIMIT];
        let str = tensor_shape_to_c_string(Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len()));

        let mut expected = format!("({}", internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT);
        for _ in 0..(K_TENSOR_DIMENSION_LIMIT - 1) {
            expected.push_str(&format!(
                ", {}",
                internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT
            ));
        }
        expected.push(')');
        assert_eq!(expected, c_str(&str));
    }

    // [spec:et:sem:tensor-shape-to-c-string.executorch.runtime.tensor-shape-to-c-string-fn/test]
    #[test]
    fn tensor_shape_to_c_string_test_exceeds_dimension_limit() {
        let mut sizes: [i32; K_TENSOR_DIMENSION_LIMIT + 1] =
            [internal::K_MAXIMUM_PRINTABLE_TENSOR_SHAPE_ELEMENT as i32;
                K_TENSOR_DIMENSION_LIMIT + 1];
        let str = tensor_shape_to_c_string(Span::from_raw_parts(sizes.as_mut_ptr(), sizes.len()));
        assert_eq!(c_str(&str), "(ERR: tensor ndim exceeds limit)");
    }
}
