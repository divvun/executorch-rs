//! Literal port of schema/extended_header.cpp + schema/extended_header.h.

use crate::runtime::core::error::{Error, Result};

/// An extended, ExecuTorch-specific header that may be embedded in the
/// serialized Program data header.
///
/// For details see //executorch/docs/source/pte-file-format.md
// [spec:et:def:extended-header.executorch.runtime.extended-header]
pub struct ExtendedHeader {
    /// The size in bytes of the Program flatbuffer data, starting from offset
    /// zero.
    pub program_size: u64,

    /// The offset in bytes of the first segment, if present. Zero if no segment
    /// is present.
    pub segment_base_offset: u64,

    /// The size of all the segment data, in bytes. Zero if:
    /// - no segment is present
    /// - the segment_data_size field doesn't exist in the header - the case for
    ///   older PTE files.
    pub segment_data_size: u64,
}

impl ExtendedHeader {
    /// To find the header, callers should provide at least this many bytes of the
    /// head of the serialized Program data. Keep this in sync with NUM_HEAD_BYTES
    /// in //executorch/exir/_serialize/program.py
    pub const K_NUM_HEAD_BYTES: usize = 64;

    /// The offset into the Program serialized program data where the extended
    /// header should begin.
    pub const K_HEADER_OFFSET: usize = 8;

    /// The magic bytes that identify the header.
    pub const K_MAGIC_SIZE: usize = 4;
    pub const K_MAGIC: [u8; Self::K_MAGIC_SIZE] = [b'e', b'h', b'0', b'0'];

    /// Look for and parse an ExtendedHeader in the provided data.
    ///
    /// @param[in] data The contents of the beginning of the serialized binary
    ///     Program data, starting at offset 0 (i.e., the head of the file).
    /// @param[in] size Length of `data` in bytes. Must be >= kNumHeadBytes or this
    ///     call will fail.
    ///
    /// @returns an ExtendedHeader if the header was found and is valid. Returns an
    ///     error if size was too short, if the header was not found, or if the
    ///     header appeared to be corrupt.
    // [spec:et:def:extended-header.executorch.runtime.extended-header.parse-fn]
    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn]
    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    pub fn parse(data: *const core::ffi::c_void, size: usize) -> Result<ExtendedHeader> {
        if size < ExtendedHeader::K_NUM_HEAD_BYTES {
            crate::et_log!(
                Error,
                "Extended header data size {} < minimum {}",
                size,
                ExtendedHeader::K_NUM_HEAD_BYTES
            );
            return Err(Error::InvalidArgument);
        }
        let header: *const u8 = unsafe { (data as *const u8).add(K_HEADER_OFFSET) };

        // Check magic bytes.
        if unsafe {
            libc::memcmp(
                header as *const core::ffi::c_void,
                ExtendedHeader::K_MAGIC.as_ptr() as *const core::ffi::c_void,
                ExtendedHeader::K_MAGIC_SIZE,
            )
        } != 0
        {
            return Err(Error::NotFound);
        }

        // Check header length.
        let header_length: u32 = get_uint32_le(unsafe { header.add(K_HEADER_LENGTH_OFFSET) });
        if (header_length as usize) < K_MINIMUM_HEADER_LENGTH {
            crate::et_log!(
                Error,
                "Extended header length {} < {}",
                header_length,
                K_MINIMUM_HEADER_LENGTH
            );
            return Err(Error::InvalidProgram);
        }

        let mut segment_data_size: u64 = 0;
        if (header_length as usize) >= K_HEADER_LENGTH_WITH_SEGMENT_DATA_SIZE {
            segment_data_size =
                get_uint64_le(unsafe { header.add(K_HEADER_SEGMENT_DATA_SIZE_OFFSET) });
        }

        // The header is present and apparently valid.
        Ok(ExtendedHeader {
            program_size: get_uint64_le(unsafe { header.add(K_HEADER_PROGRAM_SIZE_OFFSET) }),
            segment_base_offset: get_uint64_le(unsafe {
                header.add(K_HEADER_SEGMENT_BASE_OFFSET_OFFSET)
            }),
            segment_data_size,
        })
    }
}

/// The expected location of the header length field relative to the beginning
/// of the header.
const K_HEADER_LENGTH_OFFSET: usize = ExtendedHeader::K_MAGIC_SIZE;

/// The expected location of the program_size field relative to the beginning of
/// the header.
const K_HEADER_PROGRAM_SIZE_OFFSET: usize = K_HEADER_LENGTH_OFFSET + core::mem::size_of::<u32>();

/// The expected location of the segment_base_offset field relative to the
/// beginning of the header.
const K_HEADER_SEGMENT_BASE_OFFSET_OFFSET: usize =
    K_HEADER_PROGRAM_SIZE_OFFSET + core::mem::size_of::<u64>();

/// The size of the header that covers the fields known of by this version of
/// the code. It's ok for a header to be larger as long as the fields stay in
/// the same place, but this code will ignore any new fields.
const K_MINIMUM_HEADER_LENGTH: usize =
    K_HEADER_SEGMENT_BASE_OFFSET_OFFSET + core::mem::size_of::<u64>();

/// The expected location of the segment_data_size field relative to the
/// beginning of the header.
const K_HEADER_SEGMENT_DATA_SIZE_OFFSET: usize =
    K_HEADER_SEGMENT_BASE_OFFSET_OFFSET + core::mem::size_of::<u64>();

/// The expected length of the header, including the segment_data_size field.
const K_HEADER_LENGTH_WITH_SEGMENT_DATA_SIZE: usize =
    K_HEADER_SEGMENT_DATA_SIZE_OFFSET + core::mem::size_of::<u64>();

/// Re-exported so call sites can reach `kHeaderOffset` through the module.
const K_HEADER_OFFSET: usize = ExtendedHeader::K_HEADER_OFFSET;

/// Interprets the 4 bytes at `data` as a little-endian uint32_t.
// [spec:et:def:extended-header.executorch.runtime.get-u-int32-le-fn]
// [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn]
// [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn/test]
fn get_uint32_le(data: *const u8) -> u32 {
    unsafe {
        (*data.add(0) as u32)
            | ((*data.add(1) as u32) << 8)
            | ((*data.add(2) as u32) << 16)
            | ((*data.add(3) as u32) << 24)
    }
}

/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
// [spec:et:def:extended-header.executorch.runtime.get-u-int64-le-fn]
// [spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn]
// [spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn/test]
fn get_uint64_le(data: *const u8) -> u64 {
    unsafe {
        (*data.add(0) as u64)
            | ((*data.add(1) as u64) << 8)
            | ((*data.add(2) as u64) << 16)
            | ((*data.add(3) as u64) << 24)
            | ((*data.add(4) as u64) << 32)
            | ((*data.add(5) as u64) << 40)
            | ((*data.add(6) as u64) << 48)
            | ((*data.add(7) as u64) << 56)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The offset to the header's length field, which is in the 4 bytes after the
    // magic.
    const K_HEADER_LENGTH_OFFSET: usize =
        ExtendedHeader::K_HEADER_OFFSET + ExtendedHeader::K_MAGIC_SIZE;

    // An example, valid extended header.
    //
    // This data is intentionally fragile. If the header layout or magic changes,
    // this test data must change too. The layout of the header is a contract, not
    // an implementation detail.

    // The minimum header.
    #[rustfmt::skip]
    const EXAMPLE_HEADER_DATA: [u8; 24] = [
        // Magic bytes
        b'e', b'h', b'0', b'0',
        // uint32_t header size (little endian)
        0x18, 0x00, 0x00, 0x00,
        // uint64_t program size
        0x71, 0x61, 0x51, 0x41, 0x31, 0x21, 0x11, 0x01,
        // uint64_t segment base offset
        0x72, 0x62, 0x52, 0x42, 0x32, 0x22, 0x12, 0x02,
    ];

    // Contains segment data size.
    #[rustfmt::skip]
    const EXAMPLE_HEADER_DATA_EXTENDED: [u8; 32] = [
        // Magic bytes
        b'e', b'h', b'0', b'0',
        // uint32_t header size (little endian)
        0x20, 0x00, 0x00, 0x00,
        // uint64_t program size
        0x71, 0x61, 0x51, 0x41, 0x31, 0x21, 0x11, 0x01,
        // uint64_t segment base offset
        0x72, 0x62, 0x52, 0x42, 0x32, 0x22, 0x12, 0x02,
        // uint64_t segment data size
        0x73, 0x63, 0x53, 0x43, 0x33, 0x23, 0x13, 0x03,
    ];

    // The program_size field encoded in EXAMPLE_HEADER_DATA. Each byte is unique
    // within the header data.
    const EXAMPLE_PROGRAM_SIZE: u64 = 0x0111213141516171;

    // The segment_base_offset field encoded in EXAMPLE_HEADER_DATA. Each byte is
    // unique within the header data.
    const EXAMPLE_SEGMENT_BASE_OFFSET: u64 = 0x0212223242526272;

    // The segment_data_size field encoded in EXAMPLE_HEADER_DATA. Each byte is
    // unique within the header data.
    const EXAMPLE_SEGMENT_DATA_SIZE: u64 = 0x0313233343536373;

    // ExtendedHeader is a literal port of a C++ struct with no Debug, so tests
    // extract the Err arm without requiring `T: Debug` (as `unwrap_err` would).
    fn err_of(result: Result<ExtendedHeader>) -> Error {
        match result {
            Ok(_) => panic!("expected an error, got Ok"),
            Err(e) => e,
        }
    }

    // Returns fake serialized Program head data that contains the example header
    // at the expected offset.
    fn create_example_program_head(example: &[u8]) -> Vec<u8> {
        // Allocate memory representing the head of the serialized Program.
        // Write non-zeros into it to make it more obvious if we read outside the
        // header.
        let mut ret = vec![0x55u8; ExtendedHeader::K_NUM_HEAD_BYTES];
        // Copy the example header into the right offset.
        ret[ExtendedHeader::K_HEADER_OFFSET..ExtendedHeader::K_HEADER_OFFSET + example.len()]
            .copy_from_slice(example);
        ret
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    // [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn/test]
    // [spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn/test]
    // The unique per-byte field values make success proof that both LE readers
    // look at the right bytes and assemble them in the right order.
    #[test]
    fn valid_header_parses_correctly() {
        let program = create_example_program_head(&EXAMPLE_HEADER_DATA);

        let header =
            ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());

        // The header should be present.
        let header = header.expect("header should parse");

        // Expect this header has size 24.
        assert_eq!(program[K_HEADER_LENGTH_OFFSET], 0x18);

        // Since each byte of these fields is unique, success demonstrates that the
        // endian-to-int conversion is correct and looks at the expected bytes of
        // the header.
        assert_eq!(header.program_size, EXAMPLE_PROGRAM_SIZE);
        assert_eq!(header.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET);
        assert_eq!(header.segment_data_size, 0);
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    // [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn/test]
    // [spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn/test]
    #[test]
    fn valid_header_parses_correctly_extended_example() {
        let program = create_example_program_head(&EXAMPLE_HEADER_DATA_EXTENDED);

        let header =
            ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());

        // The header should be present.
        let header = header.expect("header should parse");

        // Expect this header has size 32.
        assert_eq!(program[K_HEADER_LENGTH_OFFSET], 0x20);

        // Since each byte of these fields is unique, success demonstrates that the
        // endian-to-int conversion is correct and looks at the expected bytes of
        // the header.
        assert_eq!(header.program_size, EXAMPLE_PROGRAM_SIZE);
        assert_eq!(header.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET);
        assert_eq!(header.segment_data_size, EXAMPLE_SEGMENT_DATA_SIZE);
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    #[test]
    fn short_data_fails() {
        // Mirrors `ExtendedHeaderTest::SetUp()`'s `runtime_init()`; this test
        // hits an ET_LOG path, which requires an initialized PAL.
        crate::runtime::platform::runtime::runtime_init();

        let program = create_example_program_head(&EXAMPLE_HEADER_DATA);

        // Try parsing a smaller-than-required part of the data.
        assert!(program.len() >= ExtendedHeader::K_NUM_HEAD_BYTES);
        let header = ExtendedHeader::parse(
            program.as_ptr() as *const core::ffi::c_void,
            ExtendedHeader::K_NUM_HEAD_BYTES - 1,
        );

        // Should have been rejected.
        assert_eq!(err_of(header), Error::InvalidArgument);
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    #[test]
    fn missing_header_not_found() {
        // Program head data without the extended header magic bytes.
        let program = vec![0x55u8; ExtendedHeader::K_NUM_HEAD_BYTES];

        // The header should not be found.
        let header =
            ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());
        assert_eq!(err_of(header), Error::NotFound);
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    #[test]
    fn bad_magic_treated_as_missing() {
        // Get a valid header.
        let mut program = create_example_program_head(&EXAMPLE_HEADER_DATA);

        // Should be present.
        {
            let header =
                ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());
            assert!(header.is_ok());
        }

        // Change a character in the magic.
        program[ExtendedHeader::K_HEADER_OFFSET] = b'x';

        // No longer present.
        {
            let header =
                ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());
            assert_eq!(err_of(header), Error::NotFound);
        }
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    // [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn/test]
    // Overwriting the LE header-length byte and observing InvalidProgram pins the
    // uint32 reader onto the right bytes.
    #[test]
    fn shorter_header_length_fails() {
        // Mirrors `ExtendedHeaderTest::SetUp()`'s `runtime_init()`; this test
        // hits an ET_LOG path, which requires an initialized PAL.
        crate::runtime::platform::runtime::runtime_init();

        // Get a valid header.
        let mut program = create_example_program_head(&EXAMPLE_HEADER_DATA);

        // Should be present.
        {
            let header =
                ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());
            assert!(header.is_ok());
        }

        // Make the header length smaller.
        // First demonstrate that we're looking in the right place.
        assert_eq!(program[K_HEADER_LENGTH_OFFSET], 0x18);
        program[K_HEADER_LENGTH_OFFSET] = 0x10;

        // Program now considered invalid.
        {
            let header =
                ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len());
            assert_eq!(err_of(header), Error::InvalidProgram);
        }
    }

    // [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn/test]
    #[test]
    fn longer_header_length_succeeds() {
        // Get a valid header.
        let mut program = create_example_program_head(&EXAMPLE_HEADER_DATA);

        // Make the header length larger.
        // First demonstrate that we're looking in the right place.
        assert_eq!(program[K_HEADER_LENGTH_OFFSET], 0x18);
        program[K_HEADER_LENGTH_OFFSET] = 0x20;

        // Should still be present and contain the expected values.
        {
            let header =
                ExtendedHeader::parse(program.as_ptr() as *const core::ffi::c_void, program.len())
                    .expect("header should parse");
            assert_eq!(header.program_size, EXAMPLE_PROGRAM_SIZE);
            assert_eq!(header.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET);
        }
    }
}
