//! Literal port of runtime/core/tag.cpp + runtime/core/tag.h.

// EXECUTORCH_FORALL_TAGS declaration order:
//   None, Tensor, String, Double, Int, Bool,
//   ListBool, ListDouble, ListInt, ListTensor, ListScalar, ListOptionalTensor

/// The dynamic type of an EValue.
// [spec:et:def:tag.executorch.runtime.tag]
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    None,
    Tensor,
    String,
    Double,
    Int,
    Bool,
    ListBool,
    ListDouble,
    ListInt,
    ListTensor,
    ListScalar,
    ListOptionalTensor,
}

// PORT-NOTE: `ET_ENABLE_ENUM_STRINGS` is on by default; this port compiles the
// enabled branch. The disabled branch (snprintf of the integer index) is not
// emitted, matching a default build.
pub const ET_ENABLE_ENUM_STRINGS: bool = true;

/// Inline `tag_to_string(Tag)` from the header: returns a static `const char*`
/// name for the tag. Only present when `ET_ENABLE_ENUM_STRINGS` is set.
pub fn tag_to_string_static(tag: Tag) -> &'static str {
    match tag {
        Tag::None => "None",
        Tag::Tensor => "Tensor",
        Tag::String => "String",
        Tag::Double => "Double",
        Tag::Int => "Int",
        Tag::Bool => "Bool",
        Tag::ListBool => "ListBool",
        Tag::ListDouble => "ListDouble",
        Tag::ListInt => "ListInt",
        Tag::ListTensor => "ListTensor",
        Tag::ListScalar => "ListScalar",
        Tag::ListOptionalTensor => "ListOptionalTensor",
        // PORT-NOTE: C++ `default:` returns "Unknown" for out-of-range bit
        // patterns; a Rust `#[repr(u32)]` enum only holds declared
        // discriminants so the match is exhaustive.
    }
}

/// Convert a tag value to a string representation. If ET_ENABLE_ENUM_STRINGS is
/// set (it is on by default), this will return a string name (for example,
/// "Tensor"). Otherwise, it will return a string representation of the index
/// value ("1").
///
/// If the user buffer is not large enough to hold the string representation, the
/// string will be truncated.
///
/// The return value is the number of characters written, or in the case of
/// truncation, the number of characters that would be written if the buffer was
/// large enough.
// [spec:et:def:tag.executorch.runtime.tag-to-string-fn]
// [spec:et:sem:tag.executorch.runtime.tag-to-string-fn]
pub fn tag_to_string(tag: Tag, buffer: &mut [u8], buffer_size: usize) -> usize {
    if ET_ENABLE_ENUM_STRINGS {
        let name_str: &str = match tag {
            Tag::None => "None",
            Tag::Tensor => "Tensor",
            Tag::String => "String",
            Tag::Double => "Double",
            Tag::Int => "Int",
            Tag::Bool => "Bool",
            Tag::ListBool => "ListBool",
            Tag::ListDouble => "ListDouble",
            Tag::ListInt => "ListInt",
            Tag::ListTensor => "ListTensor",
            Tag::ListScalar => "ListScalar",
            Tag::ListOptionalTensor => "ListOptionalTensor",
            // PORT-NOTE: unreachable `default: name_str = "Unknown"` per above.
        };

        et_snprintf_s(buffer, buffer_size, name_str)
    } else {
        // Disabled branch: snprintf(buffer, buffer_size, "%d", tag as int).
        et_snprintf_s(buffer, buffer_size, &(tag_index(tag) as i32).to_string())
    }
}

/// Underlying integer index of the enumerator (declaration order), matching the
/// C++ `static_cast<int>(tag)`.
fn tag_index(tag: Tag) -> u32 {
    tag as u32
}

/// Mirrors `snprintf(buffer, buffer_size, "%s", s)`: writes `s` (null-truncated
/// to fit `buffer_size` including the terminator) into `buffer`, and returns the
/// untruncated length of `s` excluding the terminator.
fn et_snprintf_s(buffer: &mut [u8], buffer_size: usize, s: &str) -> usize {
    let src = s.as_bytes();
    let full_len = src.len();
    let cap = buffer_size.min(buffer.len());
    if cap > 0 {
        // Reserve one byte for the null terminator, like snprintf.
        let to_copy = full_len.min(cap - 1);
        buffer[..to_copy].copy_from_slice(&src[..to_copy]);
        buffer[to_copy] = 0;
    }
    full_len
}

/// The size of the buffer needed to hold the longest tag string, including the
/// null terminator. This value is expected to be updated manually, but it
/// checked in test_tag.cpp.
pub const K_TAG_NAME_BUFFER_SIZE: usize = 19;

// Literal port of runtime/core/test/tag_test.cpp.
//
// The C++ suite is entirely gated on `#if ET_ENABLE_ENUM_STRINGS`, which is on
// by default. `ET_ENABLE_ENUM_STRINGS` is `true` in this port, so the enabled
// branch is exercised.
#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors the C-string comparison the C++ tests perform via `EXPECT_STREQ`:
    // reads `buffer` up to the first NUL and compares to `expected`.
    fn assert_cstr_eq(buffer: &[u8], expected: &str) {
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        assert_eq!(&buffer[..end], expected.as_bytes());
    }

    // `EXECUTORCH_FORALL_TAGS` declaration order, used by `TagNameBufferSize`.
    const ALL_TAGS: [Tag; 12] = [
        Tag::None,
        Tag::Tensor,
        Tag::String,
        Tag::Double,
        Tag::Int,
        Tag::Bool,
        Tag::ListBool,
        Tag::ListDouble,
        Tag::ListInt,
        Tag::ListTensor,
        Tag::ListScalar,
        Tag::ListOptionalTensor,
    ];

    // [spec:et:sem:tag.executorch.runtime.tag-to-string-fn/test]
    #[test]
    fn tag_to_string_tag_values() {
        let mut name = [0u8; 16];
        let n = name.len();

        tag_to_string(Tag::Tensor, &mut name, n);
        assert_cstr_eq(&name, "Tensor");

        tag_to_string(Tag::Int, &mut name, n);
        assert_cstr_eq(&name, "Int");

        tag_to_string(Tag::Double, &mut name, n);
        assert_cstr_eq(&name, "Double");

        tag_to_string(Tag::Bool, &mut name, n);
        assert_cstr_eq(&name, "Bool");
    }

    // The C++ `PrintTag` test exercises the header's inline `tag_to_string(Tag)`
    // overload returning a `const char*`; that maps to `tag_to_string_static`.
    // [spec:et:sem:tag.executorch.runtime.tag-to-string-fn/test]
    #[test]
    fn tag_to_string_print_tag() {
        assert_eq!(tag_to_string_static(Tag::Tensor), "Tensor");
        assert_eq!(tag_to_string_static(Tag::Int), "Int");
        assert_eq!(tag_to_string_static(Tag::Double), "Double");
        assert_eq!(tag_to_string_static(Tag::Bool), "Bool");
    }

    // [spec:et:sem:tag.executorch.runtime.tag-to-string-fn/test]
    #[test]
    fn tag_to_string_tag_name_buffer_size() {
        let mut name = [0u8; K_TAG_NAME_BUFFER_SIZE];
        let n = name.len();
        let mut longest: usize = 0;

        for &tag in ALL_TAGS.iter() {
            let tag_len = tag_to_string(tag, &mut name, n);
            assert!(
                tag_len < K_TAG_NAME_BUFFER_SIZE,
                "kTagNameBufferSize is too small to hold {:?}",
                tag
            );
            longest = longest.max(tag_len);
        }

        assert_eq!(
            longest + 1,
            K_TAG_NAME_BUFFER_SIZE,
            "kTagNameBufferSize has incorrect value, expected {}",
            longest + 1
        );
    }

    // [spec:et:sem:tag.executorch.runtime.tag-to-string-fn/test]
    #[test]
    fn tag_to_string_fits_exact() {
        let mut name = [0u8; 4];
        let n = name.len();
        let ret = tag_to_string(Tag::Int, &mut name, n);
        assert_eq!(3, ret);
        assert_cstr_eq(&name, "Int");
    }

    // [spec:et:sem:tag.executorch.runtime.tag-to-string-fn/test]
    #[test]
    fn tag_to_string_truncate() {
        let mut name = [b'-'; 6];
        let n = name.len();
        let ret = tag_to_string(Tag::Double, &mut name, n);
        assert_eq!(6, ret);
        assert!(name[name.len() - 1] == 0);
        assert_cstr_eq(&name, "Doubl");
    }
}
