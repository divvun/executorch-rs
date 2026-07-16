# backends/xnnpack/runtime/XNNHeader.cpp, backends/xnnpack/runtime/XNNHeader.h

> [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn]
> uint32_t GetUInt32LE(const uint8_t* data)

> [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn]
> Reads 4 bytes starting at `data` and assembles them into a `uint32_t`
> interpreting them as little-endian (least-significant byte first),
> regardless of host endianness.
>
> Behavior: reads `data[0..4]` and returns
> `data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)`, where each
> byte is widened to `uint32_t` before shifting so no shift overflows.
> The caller must guarantee at least 4 readable bytes at `data`; there is
> no bounds check. No allocation, no error return.

> [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn]
> uint64_t GetUInt64LE(const uint8_t* data)

> [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn]
> Reads 8 bytes starting at `data` and assembles them into a `uint64_t`
> interpreting them as little-endian (least-significant byte first),
> regardless of host endianness.
>
> Behavior: reads `data[0..8]` and returns
> `data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24) |
> (data[4] << 32) | (data[5] << 40) | (data[6] << 48) | (data[7] << 56)`,
> where each byte is widened to `uint64_t` before shifting so no shift
> overflows. The caller must guarantee at least 8 readable bytes at `data`;
> there is no bounds check. No allocation, no error return.

> [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.xnn-header]
> struct XNNHeader {
>   static constexpr size_t kMinSize = 30;
>   static constexpr size_t kMagicOffset = 4;
>   static constexpr size_t kMagicSize = 4;
>   static constexpr char kMagic[kMagicSize] = {'X', 'H', '0', '0'};
>   static constexpr size_t kHeaderLengthSize = 2;
>   static constexpr size_t kHeaderLengthOffset = XNNHeader::kMagicOffset + XNNHeader::kMagicSize;
>   static constexpr size_t kFlatbufferDataOffsetOffset = kHeaderLengthOffset + sizeof(uint16_t);
>   static constexpr size_t kFlatbufferDataSizeOffset = kFlatbufferDataOffsetOffset + sizeof(uint32_t);
>   static constexpr size_t kConstantDataOffsetOffset = kFlatbufferDataSizeOffset + sizeof(uint32_t);
>   static constexpr size_t kConstantDataSizeOffset = kConstantDataOffsetOffset + sizeof(uint32_t);
>   uint32_t flatbuffer_offset;
>   uint32_t flatbuffer_size;
>   uint32_t constant_data_offset;
>   uint64_t constant_data_size;
> }

> [spec:et:def:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn]
> Result<XNNHeader> XNNHeader::Parse(const void* data, size_t size)

> [spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.xnn-header.parse-fn]
> Parses the fixed-layout XNNPACK extended header that precedes the
> flatbuffer payload in serialized XNNPACK delegate data. Returns a
> `Result<XNNHeader>`: either a fully-populated, validated `XNNHeader`
> value, or an `Error` code. Does not allocate; reads only from the caller-
> provided buffer `[data, data+size)`.
>
> Field layout (all multi-byte fields little-endian; offsets are the
> constants from the struct def, measured from the start of `data`):
> - bytes [0, 4): reserved / flatbuffer-compatible prefix (not read here).
> - bytes [kMagicOffset=4, 8): the 4 magic bytes.
> - bytes [kHeaderLengthOffset=8, 10): a 2-byte header-length field
>   (`kHeaderLengthSize=2`); read/validated by the caller, not consumed here.
> - bytes [kFlatbufferDataOffsetOffset=10, 14): uint32 flatbuffer_offset.
> - bytes [kFlatbufferDataSizeOffset=14, 18): uint32 flatbuffer_size.
> - bytes [kConstantDataOffsetOffset=18, 22): uint32 constant_data_offset.
> - bytes [kConstantDataSizeOffset=22, 30): uint64 constant_data_size.
> `kMinSize` is 30 (the total of the above through the end of the 8-byte
> constant_data_size field).
>
> Steps:
> 1. Reinterpret `data` as a `const uint8_t*` (`header_data`).
> 2. Size check: if `size < XNNHeader::kMinSize` (30), return
>    `Error::InvalidArgument` immediately.
> 3. Magic check: compare the 4 bytes at `header_data + kMagicOffset` (4)
>    against `kMagic = {'X','H','0','0'}` via byte-wise equality
>    (`std::memcmp`). If they differ, return `Error::NotFound` (this is how
>    a plain flatbuffer without the extended header is detected).
> 4. Read the four numeric fields, little-endian, using the LE helpers:
>    - `flatbuffer_offset` = uint32 at offset kFlatbufferDataOffsetOffset
>      (10) per `[spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int32-le-fn]`.
>    - `flatbuffer_size` = uint32 at offset kFlatbufferDataSizeOffset (14).
>    - `constant_data_offset` = uint32 at offset kConstantDataOffsetOffset (18).
>    - `constant_data_size` = uint64 at offset kConstantDataSizeOffset (22)
>      per `[spec:et:sem:xnn-header.executorch.backends.xnnpack.delegate.get-u-int64-le-fn]`.
> 5. Minimum-flatbuffer-size check: let `kMinFlatbufferSize = sizeof(uint32_t)
>    + 4 = 8` (a flatbuffer root offset plus 4-byte file identifier). If
>    `flatbuffer_size < 8`, return `Error::InvalidArgument`.
> 6. Flatbuffer bounds check (overflow-safe): require
>    `flatbuffer_offset <= size` AND `flatbuffer_size <= size -
>    flatbuffer_offset`. The subtraction is only evaluated when the first
>    conjunct holds, so it never underflows. If the condition fails, return
>    `Error::InvalidArgument`.
> 7. Constant-data bounds check (overflow-safe): require
>    `constant_data_offset <= size` AND `constant_data_size <= size -
>    constant_data_offset`. Note `constant_data_size` is 64-bit and `size`
>    is `size_t`; the comparison is done in the wider domain. If it fails,
>    return `Error::InvalidArgument`.
> 8. Non-overlap / ordering check: the flatbuffer region must come entirely
>    before the constant-data region. Require `constant_data_offset >=
>    flatbuffer_offset` AND `constant_data_offset - flatbuffer_offset >=
>    flatbuffer_size` (the subtraction is safe because the first conjunct
>    holds). If it fails, return `Error::InvalidArgument`.
> 9. On success, construct and return `XNNHeader{flatbuffer_offset,
>    flatbuffer_size, constant_data_offset, constant_data_size}`.
>
> Each `ET_CHECK_OR_RETURN_ERROR` in steps 5–8 logs a descriptive message
> (with the offending field values) before returning the error; the return
> value on any failure is a `Result` holding `Error::InvalidArgument`,
> except the magic mismatch which yields `Error::NotFound` and the initial
> short-buffer case which yields `Error::InvalidArgument`.

