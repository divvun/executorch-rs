//! Literal port of backends/xnnpack/runtime/XNNHeader.cpp +
//! backends/xnnpack/runtime/XNNHeader.h.

use crate::runtime::core::error::Error;
use crate::runtime::core::result::Result;

/// Interprets the 8 bytes at `data` as a little-endian uint64_t.
// [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn]
// [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn]
//
// PORT-NOTE: The C++ takes a raw `const uint8_t*` with no bounds check. Here
// `data` is a `&[u8]` slice of at least 8 bytes; the caller (Parse) provides
// the exact sub-slice, and the byte indexing panics rather than reading OOB.
fn get_uint64_le(data: &[u8]) -> u64 {
    (data[0] as u64)
        | ((data[1] as u64) << 8)
        | ((data[2] as u64) << 16)
        | ((data[3] as u64) << 24)
        | ((data[4] as u64) << 32)
        | ((data[5] as u64) << 40)
        | ((data[6] as u64) << 48)
        | ((data[7] as u64) << 56)
}

/// Interprets the 4 bytes at `data` as a little-endian uint32_t.
// [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn]
// [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn]
fn get_uint32_le(data: &[u8]) -> u32 {
    (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16) | ((data[3] as u32) << 24)
}

/// An extended XNNPACK-header that is embeded before the flatbuffer payload
// [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.xnn-header]
#[derive(Clone, Copy, Debug)]
pub struct XNNHeader {
    /// The offset in bytes to the beginning of the flatbuffer data.
    pub flatbuffer_offset: u32,
    /// The size in bytes of the flatbuffer data.
    pub flatbuffer_size: u32,
    /// The offset in bytes to the beginning of the constant data.
    pub constant_data_offset: u32,
    /// The size in bytes of the constant data.
    pub constant_data_size: u64,
}

impl XNNHeader {
    /// The minimum size of the XNNHeader. The caller should provide at least
    /// this many bytes of the head of the serialized XNNPACK Data
    pub const K_MIN_SIZE: usize = 30;

    /// The magic offset. This offset is the same as the offset for flatbuffer
    /// header so we will be able to check if the header is is either the
    /// flatbuffer head or the wrapper header we introduce here
    pub const K_MAGIC_OFFSET: usize = 4;

    /// The size in bytes of the magic bytes that identify the header.
    pub const K_MAGIC_SIZE: usize = 4;
    /// The magic bytes that identify the header.
    pub const K_MAGIC: [u8; Self::K_MAGIC_SIZE] = [b'X', b'H', b'0', b'0'];

    /// The size in bytes of the header length. We store 2 bytes for the header
    /// length
    pub const K_HEADER_LENGTH_SIZE: usize = 2;

    /// The expected location of the header length field relative to the
    /// beginning of the header.
    pub const K_HEADER_LENGTH_OFFSET: usize = XNNHeader::K_MAGIC_OFFSET + XNNHeader::K_MAGIC_SIZE;

    /// The expected location of the flatbuffer data offset field relative to
    /// the beginning of the header.
    pub const K_FLATBUFFER_DATA_OFFSET_OFFSET: usize =
        Self::K_HEADER_LENGTH_OFFSET + core::mem::size_of::<u16>();

    /// The expected location of the flatbuffer data size field relative to the
    /// beginning of the header.
    pub const K_FLATBUFFER_DATA_SIZE_OFFSET: usize =
        Self::K_FLATBUFFER_DATA_OFFSET_OFFSET + core::mem::size_of::<u32>();

    /// The expected location of the constant data offset field relative to the
    /// beginning of the header.
    pub const K_CONSTANT_DATA_OFFSET_OFFSET: usize =
        Self::K_FLATBUFFER_DATA_SIZE_OFFSET + core::mem::size_of::<u32>();

    /// The expected location of the constant data size field relative to the
    /// beginning of the header.
    pub const K_CONSTANT_DATA_SIZE_OFFSET: usize =
        Self::K_CONSTANT_DATA_OFFSET_OFFSET + core::mem::size_of::<u32>();

    /// Look for and parse an ExtendedHeader in the provided data.
    // [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn]
    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn]
    //
    // PORT-NOTE: The C++ takes `const void* data, size_t size`; here the same
    // is expressed as a `&[u8]` slice whose length is `size`. The LE readers
    // receive exact sub-slices from that buffer.
    pub fn parse(data: &[u8]) -> Result<XNNHeader> {
        let header_data = data;
        let size = data.len();

        if size < XNNHeader::K_MIN_SIZE {
            return Err(Error::InvalidArgument);
        }

        let magic_start = &header_data
            [XNNHeader::K_MAGIC_OFFSET..XNNHeader::K_MAGIC_OFFSET + XNNHeader::K_MAGIC_SIZE];
        if magic_start != XNNHeader::K_MAGIC {
            return Err(Error::NotFound);
        }

        let flatbuffer_offset =
            get_uint32_le(&header_data[XNNHeader::K_FLATBUFFER_DATA_OFFSET_OFFSET..]);

        let flatbuffer_size =
            get_uint32_le(&header_data[XNNHeader::K_FLATBUFFER_DATA_SIZE_OFFSET..]);

        let constant_data_offset =
            get_uint32_le(&header_data[XNNHeader::K_CONSTANT_DATA_OFFSET_OFFSET..]);

        let constant_data_size =
            get_uint64_le(&header_data[XNNHeader::K_CONSTANT_DATA_SIZE_OFFSET..]);

        // Validate min flatbuffer size.
        const K_MIN_FLATBUFFER_SIZE: usize = core::mem::size_of::<u32>() + 4; // root offset + identifier
        crate::et_check_or_return_error!(
            (flatbuffer_size as usize) >= K_MIN_FLATBUFFER_SIZE,
            InvalidArgument,
            "flatbuffer_size {} is too small (minimum {})",
            flatbuffer_size,
            K_MIN_FLATBUFFER_SIZE
        );

        // Validate that flatbuffer region does not overflow or exceed the
        // buffer.
        crate::et_check_or_return_error!(
            (flatbuffer_offset as usize) <= size
                && (flatbuffer_size as usize) <= size - flatbuffer_offset as usize,
            InvalidArgument,
            "flatbuffer_offset: {} and flatbuffer_size: {} are invalid for buffer of size: {}",
            flatbuffer_offset,
            flatbuffer_size,
            size
        );
        // Validate that constant data region does not overflow or exceed the
        // buffer.
        crate::et_check_or_return_error!(
            (constant_data_offset as usize) <= size
                && constant_data_size <= (size as u64) - constant_data_offset as u64,
            InvalidArgument,
            "constant_data_offset: {} and constant_data_size: {} are invalid for buffer of size: {}",
            constant_data_offset,
            constant_data_size,
            size
        );

        // Validate that constant data region does not overlap with flatbuffer
        // region. flatbuffer should come before constant data.
        crate::et_check_or_return_error!(
            constant_data_offset >= flatbuffer_offset
                && constant_data_offset - flatbuffer_offset >= flatbuffer_size,
            InvalidArgument,
            "constant_data_offset: {} and flatbuffer_offset: {} with flatbuffer_size: {} are overlapping.",
            constant_data_offset,
            flatbuffer_offset,
            flatbuffer_size
        );

        Ok(XNNHeader {
            flatbuffer_offset,
            flatbuffer_size,
            constant_data_offset,
            constant_data_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a header prefix laid out exactly like the C++/Python contract
    // (see backends/xnnpack/test/serialization/test_xnnheader.py):
    //   [0..4)   zeros
    //   [4..8)   magic "XH00"
    //   [8..10)  header length (LE u16)
    //   [10..14) flatbuffer offset (LE u32)
    //   [14..18) flatbuffer size   (LE u32)
    //   [18..22) constant data offset (LE u32)
    //   [22..30) constant data size   (LE u64)
    fn make_header(
        flatbuffer_offset: u32,
        flatbuffer_size: u32,
        constant_data_offset: u32,
        constant_data_size: u64,
    ) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0, 0, 0, 0]);
        v.extend_from_slice(&XNNHeader::K_MAGIC);
        v.extend_from_slice(&0x001Eu16.to_le_bytes());
        v.extend_from_slice(&flatbuffer_offset.to_le_bytes());
        v.extend_from_slice(&flatbuffer_size.to_le_bytes());
        v.extend_from_slice(&constant_data_offset.to_le_bytes());
        v.extend_from_slice(&constant_data_size.to_le_bytes());
        assert_eq!(v.len(), XNNHeader::K_MIN_SIZE);
        v
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn/test]
    #[test]
    fn get_uint32_le_decodes_little_endian() {
        // Bytes 0x44 0x33 0x22 0x11 -> 0x11223344 (from the Python contract).
        assert_eq!(get_uint32_le(&[0x44, 0x33, 0x22, 0x11]), 0x1122_3344);
        assert_eq!(get_uint32_le(&[0x00, 0x00, 0x00, 0x00]), 0);
        assert_eq!(get_uint32_le(&[0xFF, 0xFF, 0xFF, 0xFF]), u32::MAX);
        // Trailing bytes past index 3 are ignored (matches raw-pointer read).
        assert_eq!(get_uint32_le(&[0x01, 0x00, 0x00, 0x00, 0xAB]), 1);
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn/test]
    #[test]
    fn get_uint64_le_decodes_little_endian() {
        // Bytes CC BB AA 99 CC BB AA 99 -> 0x99AABBCC99AABBCC (Python contract).
        assert_eq!(
            get_uint64_le(&[0xCC, 0xBB, 0xAA, 0x99, 0xCC, 0xBB, 0xAA, 0x99]),
            0x99AA_BBCC_99AA_BBCC
        );
        assert_eq!(get_uint64_le(&[0, 0, 0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(
            get_uint64_le(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            u64::MAX
        );
    }

    // Exercises the whole success path: the field offsets, both LE readers, and
    // all four validation checks passing.
    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    // also verifies get_uint32_le / get_uint64_le through Parse.
    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn/test]
    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn/test]
    #[test]
    fn parse_valid_header_reads_all_fields() {
        // A self-consistent header: flatbuffer [30, 38), constant data [38, 46),
        // all within a 100-byte buffer, non-overlapping, flatbuffer first.
        let mut buf = make_header(30, 8, 38, 8);
        buf.resize(100, 0);

        let header = XNNHeader::parse(&buf).expect("valid header parses");
        assert_eq!(header.flatbuffer_offset, 30);
        assert_eq!(header.flatbuffer_size, 8);
        assert_eq!(header.constant_data_offset, 38);
        assert_eq!(header.constant_data_size, 8);
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    #[test]
    fn parse_too_small_returns_invalid_argument() {
        let buf = vec![0u8; XNNHeader::K_MIN_SIZE - 1];
        assert!(matches!(
            XNNHeader::parse(&buf),
            Err(Error::InvalidArgument)
        ));
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    #[test]
    fn parse_wrong_magic_returns_not_found() {
        let mut buf = make_header(30, 8, 38, 8);
        buf.resize(100, 0);
        // Corrupt the magic ("XH00" -> "YT01"), mirroring the Python test.
        buf[XNNHeader::K_MAGIC_OFFSET..XNNHeader::K_MAGIC_OFFSET + XNNHeader::K_MAGIC_SIZE]
            .copy_from_slice(b"YT01");
        assert!(matches!(XNNHeader::parse(&buf), Err(Error::NotFound)));
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    #[test]
    fn parse_flatbuffer_too_small_returns_invalid_argument() {
        crate::runtime::platform::runtime::runtime_init();
        // flatbuffer_size 7 < kMinFlatbufferSize (8).
        let mut buf = make_header(30, 7, 38, 8);
        buf.resize(100, 0);
        assert!(matches!(
            XNNHeader::parse(&buf),
            Err(Error::InvalidArgument)
        ));
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    #[test]
    fn parse_flatbuffer_exceeds_buffer_returns_invalid_argument() {
        crate::runtime::platform::runtime::runtime_init();
        // flatbuffer_offset 30 + size 80 = 110 > buffer size 100.
        let mut buf = make_header(30, 80, 120, 8);
        buf.resize(100, 0);
        assert!(matches!(
            XNNHeader::parse(&buf),
            Err(Error::InvalidArgument)
        ));
    }

    // [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn/test]
    #[test]
    fn parse_overlapping_regions_returns_invalid_argument() {
        crate::runtime::platform::runtime::runtime_init();
        // constant_data_offset 32 - flatbuffer_offset 30 = 2 < flatbuffer_size 8.
        let mut buf = make_header(30, 8, 32, 8);
        buf.resize(100, 0);
        assert!(matches!(
            XNNHeader::parse(&buf),
            Err(Error::InvalidArgument)
        ));
    }
}
