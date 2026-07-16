# extension/data_loader/buffer_data_loader.h

> [spec:et:def:buffer-data-loader.executorch.extension.buffer-data-loader]
> class BufferDataLoader final : public executorch::runtime::DataLoader {
>   ET_NODISCARD executorch::runtime::Result<executorch::runtime::FreeableBuffer> load( size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segm...;
>   ET_NODISCARD executorch::runtime::Result<size_t> size();
>   ET_NODISCARD executorch::runtime::Error load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer);
>   const uint8_t* const data_;
>   const size_t size_;
> }

> [spec:et:def:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn]
> BufferDataLoader(const void* data, size_t size)

> [spec:et:sem:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn]
> Constructor for `BufferDataLoader`, which wraps a caller-owned, pre-allocated
> byte buffer as a `DataLoader`. It borrows the buffer; it never takes ownership,
> copies, or frees the data.
>
> Store the two fields:
> - `data_`: the `const void* data` argument reinterpreted as `const uint8_t*`
>   (a raw byte pointer, chosen because byte indexing is convenient). No
>   dereference or validation of the pointer occurs here; a null `data` with
>   `size == 0` is a valid empty loader.
> - `size_`: the `size_t size` argument stored verbatim as the total number of
>   bytes the buffer is considered to contain.
>
> There is no error path and no return value. The stored buffer backs the three
> `DataLoader` operations: `load` returns a non-owning `FreeableBuffer` pointing
> at `data_ + offset` (with a null free function, so freeing is a no-op) after
> bounds-checking `offset + size` against `size_` for overflow and range;
> `size()` returns `size_`; `load_into` null-checks the destination then
> `memcpy`s the requested range out of the buffer.

