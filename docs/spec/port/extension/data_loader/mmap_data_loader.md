# extension/data_loader/mmap_data_loader.cpp, extension/data_loader/mmap_data_loader.h

> [spec:et:def:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]
> Range get_overlapping_pages(uintptr_t offset, size_t size, size_t page_size)

> [spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]
> Given a byte region `[offset, offset+size)` and a `page_size` (which is
> assumed to be a power of two), returns a `Range { start, size }`
> describing the smallest page-aligned span that fully covers the region.
>
> Steps:
> 1. Compute `page_mask = ~(page_size - 1)` — the mask that clears the
>    low bits within a page (e.g. for page_size 4096, clears the low 12
>    bits).
> 2. `start = offset & page_mask` — the address/offset of the page that
>    starts at or before the beginning of the region (rounds `offset` down
>    to a page boundary).
> 3. `end = (offset + size + ~page_mask) & page_mask` — rounds the region
>    end (`offset + size`) up to the next page boundary. `~page_mask` equals
>    `page_size - 1`, so this adds `page_size - 1` before masking down,
>    i.e. a ceiling-to-page operation.
> 4. Return `Range{ start, static_cast<size_t>(end - start) }`. The size is
>    a whole multiple of `page_size` (0 only when the input `size` is 0 and
>    `offset` is already page-aligned).
>
> No overflow guarding is performed on `offset + size + (page_size - 1)`;
> callers pass values already validated against the file size.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader]
> class MmapDataLoader final : public executorch::runtime::DataLoader {
>   enum class MlockConfig { /// Do not call `mlock()` on loaded pages. NoMlock, /// Call `mlock()` on loaded pages, failing if it fails. UseMlock, /// Call `mlo...;
>   ET_NODISCARD executorch::runtime::Result<executorch::runtime::FreeableBuffer> load( size_t offset, size_t size, const DataLoader::SegmentInfo& segment_info);
>   const override;
>   ET_NODISCARD executorch::runtime::Result<size_t> size();
>   const override;
>   ET_NODISCARD executorch::runtime::Error load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer);
>   const override;
>   ET_NODISCARD executorch::runtime::Error validate_input( size_t offset, size_t size) const;
>   const char* const file_name_;
>   const size_t file_size_;
>   const size_t page_size_;
>   const int fd_;
>   const MlockConfig mlock_config_;
> }

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
> Result<MmapDataLoader> MmapDataLoader::from( const char* file_name, MmapDataLoader::MlockConfig mlock_config)

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.from-fn]
> Factory that opens `file_name` read-only and constructs an
> `MmapDataLoader` caching the fd, file size, page size, and mlock policy.
> Returns `Result<MmapDataLoader>` — an error code on any failure, otherwise
> the constructed loader.
>
> Steps:
> 1. Query the OS page size via `get_os_page_size()` (per
>    `[spec:et:sem:mman.get-os-page-size-fn]` on Windows / `sysconf(_SC_PAGESIZE)`
>    on POSIX), typed as `long`.
>    - If it is negative, log an Error and return `Error::AccessFailed`.
>    - If it is not a power of two (checked via
>      `(page_size & ~(page_size - 1)) != page_size`), log an Error and return
>      `Error::InvalidState`.
> 2. Open the file: `fd = ::open(file_name, O_RDONLY)`. If `fd < 0`, log an
>    Error (including `strerror(errno)`) and return `Error::AccessFailed`.
> 3. Query the file size via `get_file_stat(fd, &file_size)` (per
>    `[spec:et:sem:mman.get-file-stat-fn]`; `fstat`/`_fstat64`). If it returns
>    < 0, log an Error, `::close(fd)`, and return `Error::AccessFailed`.
> 4. Duplicate the filename string with `::strdup(file_name)` so debug
>    messages survive. If it returns null (allocation failure), log an Error,
>    `::close(fd)`, and return `Error::MemoryAllocationFailed`.
> 5. Construct and return `MmapDataLoader(fd, file_size, file_name_copy,
>    static_cast<size_t>(page_size), mlock_config)`. The instance takes
>    ownership of both `fd` (closed in the destructor) and the strdup'd string
>    (freed in the destructor).
>
> `mlock_config` defaults to `MlockConfig::UseMlock` at the call site. The
> file stays open for the loader's lifetime.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn]
> Result<FreeableBuffer> MmapDataLoader::load( size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segment_info) const

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn]
> Maps the file region `[offset, offset+size)` and returns a `FreeableBuffer`
> pointing into the mapping. The buffer's free callback unmaps the covering
> pages. `segment_info` is unused. Returns `Result<FreeableBuffer>`.
>
> Steps:
> 1. Validate the range with `validate_input(offset, size)` per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]`.
>    If it returns anything other than `Error::Ok`, return that error.
> 2. If `size == 0`, return an empty `FreeableBuffer(nullptr, 0,
>    /*free_fn=*/nullptr)` (mmap rejects zero-length maps).
> 3. Compute the covering page range `range = get_overlapping_pages(offset,
>    size, page_size_)` per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]`.
> 4. Let `map_size = range.size`. If `range.start + map_size > file_size_`,
>    clamp `map_size = file_size_ - range.start` (avoids mapping past the end
>    of the last page — the Windows `CreateFileMapping` shim errors with
>    `STATUS_SECTION_TOO_BIG` otherwise).
> 5. Convert the page start to the platform mmap offset type via
>    `get_mmap_offset(range.start)` (per `[spec:et:sem:mman.get-mmap-offset-fn]`).
> 6. Map read-only, shared: `pages = ::mmap(nullptr, map_size, PROT_READ,
>    MAP_SHARED, fd_, map_offset)`. If `pages == MAP_FAILED`, return
>    `Error::AccessFailed` (via ET_CHECK_OR_RETURN_ERROR, which logs and
>    returns the error).
> 7. If `mlock_config_` is `UseMlock` or `UseMlockIgnoreErrors`, call
>    `::mlock(pages, size)` (note: `size`, the requested length, not
>    `map_size`). On failure (`err < 0`):
>    - `UseMlockIgnoreErrors`: log a Debug message and continue.
>    - `UseMlock`: log an Error, `::munmap(pages, size)` (unmaps using `size`,
>      not `map_size`), and return `Error::NotSupported`.
>    munmap later releases the lock as a side effect, so no lock bookkeeping
>    is kept.
> 8. If `mlock_config_` is `UseMadvise`, call
>    `madvise_pages_willneed_sequential(pages, map_size)` and
>    `fcntl_rdadvise_apple(fd_, file_size_)` (per
>    `[spec:et:sem:mman.madvise-pages-willneed-sequential-fn]` and
>    `[spec:et:sem:mman.fcntl-rdadvise-apple-fn]`); these are best-effort
>    prefetch hints (no-ops where unsupported). `NoMlock` does none of the
>    above.
> 9. Compute the data pointer inside the mapping: `data = (const uint8_t*)pages
>    + offset - range.start` (the requested region begins at `offset -
>    range.start` bytes into the page-aligned mapping).
> 10. Return `FreeableBuffer(data, size, MunmapSegment,
>    /*free_fn_context=*/reinterpret_cast<void*>((uintptr_t)page_size_))`.
>    The page size is smuggled through the context pointer so the free
>    callback (`MunmapSegment`, per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.munmap-segment-fn]`)
>    can recompute the covering pages and unmap them.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn]
> Error MmapDataLoader::load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer) const

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-into-fn]
> Copies the file region `[offset, offset+size)` into a caller-provided
> `buffer`. `segment_info` is unused. Returns `Error::Ok` on success.
>
> Steps:
> 1. Check `buffer != nullptr`; if null, return `Error::InvalidArgument`
>    (ET_CHECK_OR_RETURN_ERROR logs and returns).
> 2. Validate the range with `validate_input(offset, size)` per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]`.
>    If it returns anything other than `Error::Ok`, return that error.
> 3. If `size == 0`, return `Error::Ok` (nothing to copy).
> 4. Compute the covering page range `range = get_overlapping_pages(offset,
>    size, page_size_)` per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]`.
> 5. Let `map_size = range.size`; clamp to `file_size_ - range.start` if
>    `range.start + map_size > file_size_` (same end-of-file clamp as load).
> 6. Convert `range.start` via `get_mmap_offset` per
>    `[spec:et:sem:mman.get-mmap-offset-fn]`.
> 7. Map read-only, private: `pages = ::mmap(nullptr, map_size, PROT_READ,
>    MAP_PRIVATE, fd_, map_offset)`. (PRIVATE vs SHARED is immaterial for a
>    read-only map, chosen to avoid any accidental write-through.) If
>    `pages == MAP_FAILED`, return `Error::AccessFailed`.
> 8. Compute `map_delta = offset - range.start` (offset of the requested data
>    within the mapping).
> 9. `std::memcpy(buffer, (uint8_t*)pages + map_delta, size)`.
> 10. `::munmap(pages, map_size)` (return value ignored).
> 11. Return `Error::Ok`.
>
> No mlock/madvise is performed in this path regardless of `mlock_config_`.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.mlock-config]
> enum class MlockConfig {
>   NoMlock;
>   UseMlock;
>   UseMlockIgnoreErrors;
>   UseMadvise;
> }

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.mmap-data-loader-fn]
> MmapDataLoader(MmapDataLoader&& rhs) noexcept

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.mmap-data-loader-fn]
> Move constructor (`noexcept`). Transfers ownership of the fd and strdup'd
> filename from `rhs` into the new instance, then neutralizes `rhs` so its
> destructor is a no-op.
>
> Steps:
> 1. Copy all five fields from `rhs` into the new instance: `file_name_`,
>    `file_size_`, `page_size_`, `fd_`, `mlock_config_`.
> 2. Reset `rhs` to a "moved-from" empty state (const fields are overwritten
>    via `const_cast`):
>    - `rhs.file_name_ = nullptr`
>    - `rhs.file_size_ = 0`
>    - `rhs.page_size_ = 0`
>    - `rhs.fd_ = -1`
>    - `rhs.mlock_config_ = MlockConfig::NoMlock`
>
> The moved-from state is what makes the destructor safe: it frees a null
> filename (no-op) and skips `::close` because `fd_ == -1`. Required so that
> `MmapDataLoader` is movable and can be wrapped in `Result`. In Rust this is
> the natural move semantics; the explicit reset corresponds to leaving the
> source with `fd = -1` / no owned resources.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn]
> Result<size_t> MmapDataLoader::size() const

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.size-fn]
> Returns the total size in bytes of the wrapped file.
>
> Steps:
> 1. Check `fd_ >= 0`; if not (instance was moved-from / uninitialized),
>    return `Error::InvalidState` (ET_CHECK_OR_RETURN_ERROR logs "Uninitialized"
>    and returns).
> 2. Return `file_size_` (the size cached at construction) as
>    `Result<size_t>`.

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]
> Error MmapDataLoader::validate_input(size_t offset, size_t size) const

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.validate-input-fn]
> Validates that the read range `[offset, offset+size)` lies within the file
> and that the loader is initialized. Returns `Error::Ok` when valid.
>
> Steps:
> 1. Check `fd_ >= 0`; if not, return `Error::InvalidState` (moved-from /
>    uninitialized; ET_CHECK_OR_RETURN_ERROR logs "Uninitialized").
> 2. Compute `total_size = offset + size` using `c10::add_overflows(offset,
>    size, &total_size)`, which returns true if the addition overflows
>    `size_t`.
> 3. Require `!overflow && total_size <= file_size_`. If that fails, return
>    `Error::InvalidArgument` (logs the file name, offset, size, and
>    file_size_).
> 4. Otherwise return `Error::Ok`.

> [spec:et:def:mmap-data-loader.executorch.extension.munmap-segment-fn]
> void MunmapSegment(void* context, void* data, size_t size)

> [spec:et:sem:mmap-data-loader.executorch.extension.munmap-segment-fn]
> `FreeableBuffer::FreeFn`-compatible callback used to release a mapping
> produced by
> `[spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.load-fn]`.
> `context` is the OS page size reinterpreted from a `uintptr_t`; `data` is
> the (page-interior) data pointer; `size` is the requested region size.
>
> Steps:
> 1. Recover `page_size = reinterpret_cast<uintptr_t>(context)`.
> 2. Recompute the covering page range from the DATA pointer:
>    `range = get_overlapping_pages((uintptr_t)data, size, page_size)` per
>    `[spec:et:sem:mmap-data-loader.executorch.extension.get-overlapping-pages-fn]`.
>    Because `data` and `size` describe the same region that was mapped,
>    `range.start` recovers the page-aligned base address of the mapping and
>    `range.size` its length.
> 3. `ret = ::munmap((void*)range.start, range.size)`.
> 4. If `ret < 0`, log an Error (munmap failed, ignored) including
>    `strerror(errno)`; there is no way to recover, so the failure is
>    swallowed. Return void.
>
> Note the mapping in `load` may have been clamped to the file end
> (`map_size < range.size`), but this callback unmaps `range.size`; munmap
> tolerates unmapping a length that rounds up to whole pages of the original
> mapping.

> [spec:et:def:mmap-data-loader.executorch.extension.range]
> struct Range {
>   uintptr_t start;
>   size_t size;
> }

> [spec:et:def:mmap-data-loader.executorch.extension.mmap-data-loader.operator-fn]
> MmapDataLoader& operator=(const MmapDataLoader&) = delete

> [spec:et:sem:mmap-data-loader.executorch.extension.mmap-data-loader.operator-fn]
> Copy-assignment operator, explicitly deleted (`= delete`). An
> `MmapDataLoader` owns an fd and a heap string, so copying would double-free
> / double-close; the type is not copyable. Any attempt to copy-assign is a
> compile-time error. (The copy constructor and move-assignment operator are
> likewise deleted.) In Rust this corresponds to a type that is neither
> `Copy` nor `Clone`; ownership moves only.

