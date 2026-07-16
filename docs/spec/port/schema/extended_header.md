# schema/extended_header.cpp, schema/extended_header.h

> [spec:et:def:extended-header.executorch.runtime.extended-header]
> struct ExtendedHeader {
>   static constexpr size_t kNumHeadBytes = 64;
>   static constexpr size_t kHeaderOffset = 8;
>   static constexpr size_t kMagicSize = 4;
>   static constexpr char kMagic[kMagicSize] = {'e', 'h', '0', '0'};
>   uint64_t program_size;
>   uint64_t segment_base_offset;
>   uint64_t segment_data_size;
> }

> [spec:et:def:extended-header.executorch.runtime.extended-header.parse-fn]
> Result<ExtendedHeader> ExtendedHeader::Parse( const void* data, size_t size)

> [spec:et:sem:extended-header.executorch.runtime.extended-header.parse-fn]
> Static factory that looks for and parses an `ExtendedHeader` at a fixed
> offset within the head of the serialized Program data. `data` points at
> offset 0 of the file (the head), and `size` is the number of bytes
> available at `data`.
>
> Relevant fixed constants (all offsets are relative to the start of the
> header, not to `data`):
> - `kNumHeadBytes` = 64
> - `kHeaderOffset` = 8 (offset of the header within `data`)
> - `kMagicSize` = 4, `kMagic` = the bytes {'e','h','0','0'}
> - `kHeaderLengthOffset` = `kMagicSize` = 4
> - `kHeaderProgramSizeOffset` = 4 + 4 = 8
> - `kHeaderSegmentBaseOffsetOffset` = 8 + 8 = 16
> - `kMinimumHeaderLength` = 16 + 8 = 24
> - `kHeaderSegmentDataSizeOffset` = 16 + 8 = 24
> - `kHeaderLengthWithSegmentDataSize` = 24 + 8 = 32
>
> Steps:
> 1. Size check: if `size < kNumHeadBytes` (64), log an Error-level message
>    ("Extended header data size <size> < minimum 64") and return
>    `Error::InvalidArgument`. No further access is performed.
> 2. Compute the header pointer as `data + kHeaderOffset`, i.e. the header
>    begins 8 bytes into `data`. All subsequent field offsets are relative
>    to this header pointer. Note that because `size >= 64` is guaranteed
>    here and the largest field read ends at header offset 32 (absolute
>    byte 40 within `data`), all reads below are in bounds.
> 3. Magic check: compare the 4 bytes at header offset 0 against `kMagic`
>    ({'e','h','0','0'}) byte-for-byte (equivalent to `memcmp` over
>    `kMagicSize`=4 bytes). If they differ, return `Error::NotFound` (the
>    header is simply absent; this is not treated as corruption).
> 4. Header length: read a little-endian `uint32_t` at header offset
>    `kHeaderLengthOffset` (4) via `[spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn]`.
>    If this value `header_length < kMinimumHeaderLength` (24), log an
>    Error-level message ("Extended header length <header_length> < 24")
>    and return `Error::InvalidProgram` (the header is present but
>    corrupt/too short). `header_length` is otherwise not used to bound
>    reads except in step 5.
> 5. Optional segment_data_size: initialize `segment_data_size` = 0. If
>    `header_length >= kHeaderLengthWithSegmentDataSize` (32), overwrite it
>    with a little-endian `uint64_t` read at header offset
>    `kHeaderSegmentDataSizeOffset` (24) via
>    `[spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn]`.
>    Otherwise `segment_data_size` stays 0 (older PTE files that predate
>    this field).
> 6. Success: construct and return (wrapped in a successful `Result`) an
>    `ExtendedHeader` with:
>    - `program_size` = little-endian `uint64_t` at header offset
>      `kHeaderProgramSizeOffset` (8),
>    - `segment_base_offset` = little-endian `uint64_t` at header offset
>      `kHeaderSegmentBaseOffsetOffset` (16),
>    - `segment_data_size` = the value from step 5.
>
> The function never mutates `data`. It reads at most 32 bytes starting at
> the header pointer (absolute bytes 8..40 of `data`). Fields defined by a
> header longer than 32 bytes are ignored, and a header may legally be
> larger than 32 bytes as long as these fields keep their positions.

> [spec:et:def:extended-header.executorch.runtime.get-u-int32-le-fn]
> uint32_t GetUInt32LE(const uint8_t* data)

> [spec:et:sem:extended-header.executorch.runtime.get-u-int32-le-fn]
> Interprets the 4 bytes at `data` as a little-endian `uint32_t`. `data`
> must point to at least 4 readable bytes; no bounds checking is performed.
> Returns `data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)`,
> where each byte is zero-extended to `uint32_t` before shifting (so no
> sign extension and no overflow: byte 3 shifted left by 24 fits in the
> 32-bit result). Byte 0 is the least-significant byte. Endianness of the
> host is irrelevant — the byte order is fixed to little-endian by the
> explicit shifts. No allocation, no mutation of `data`.

> [spec:et:def:extended-header.executorch.runtime.get-u-int64-le-fn]
> uint64_t GetUInt64LE(const uint8_t* data)

> [spec:et:sem:extended-header.executorch.runtime.get-u-int64-le-fn]
> Interprets the 8 bytes at `data` as a little-endian `uint64_t`. `data`
> must point to at least 8 readable bytes; no bounds checking is performed.
> Returns the OR of each byte `data[i]` zero-extended to `uint64_t` and
> shifted left by `8*i` for `i` in 0..=7:
> `data[0] | (data[1]<<8) | (data[2]<<16) | (data[3]<<24) |
> (data[4]<<32) | (data[5]<<40) | (data[6]<<48) | (data[7]<<56)`. Each byte
> is widened to 64 bits before shifting (no sign extension, no overflow).
> Byte 0 is the least-significant byte. Byte order is fixed to
> little-endian by the explicit shifts, independent of host endianness. No
> allocation, no mutation of `data`.

