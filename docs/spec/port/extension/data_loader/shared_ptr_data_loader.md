# extension/data_loader/shared_ptr_data_loader.h

> [spec:et:def:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader]
> class SharedPtrDataLoader final : public executorch::runtime::DataLoader {
>   ET_NODISCARD executorch::runtime::Result<executorch::runtime::FreeableBuffer> load( size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segm...;
>   ET_NODISCARD executorch::runtime::Result<size_t> size();
>   const std::shared_ptr<void> data_;
>   const size_t size_;
> }

> [spec:et:def:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader.shared-ptr-data-loader-fn]
> SharedPtrDataLoader(std::shared_ptr<void> data, size_t size)

> [spec:et:sem:shared-ptr-data-loader.executorch.extension.shared-ptr-data-loader.shared-ptr-data-loader-fn]
> Constructor for `SharedPtrDataLoader`. Wraps a pre-allocated buffer that
> was allocated elsewhere and shares ownership of it.
>
> Steps:
> 1. Store the shared buffer pointer: `data_ = data` (a
>    `std::shared_ptr<void>`; the shared-ownership refcount is incremented,
>    keeping the buffer alive for the loader's lifetime).
> 2. Store the buffer length: `size_ = size`.
>
> No validation, copying, or allocation is performed. The `load()` method
> returns `FreeableBuffer`s that point into this buffer with a null free
> callback (they never free the data); the buffer is released only when the
> last `shared_ptr` owner is dropped. In Rust this maps to holding an
> `Arc`-like shared handle plus a length.

