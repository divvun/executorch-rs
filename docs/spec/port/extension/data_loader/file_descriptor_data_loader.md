# extension/data_loader/file_descriptor_data_loader.cpp, extension/data_loader/file_descriptor_data_loader.h

> [spec:et:def:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]
> inline void* et_aligned_alloc(size_t size, std::align_val_t alignment)

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]
> Allocates `size` bytes with the given `alignment`, used for load buffers.
>
> Implementation: `return ::operator new(size, alignment)` — the aligned form of
> global operator new. Note: unlike the `FileDataLoader` sibling
> (`[spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]`),
> this variant does NOT use the `std::nothrow` form, so under a throwing global
> operator new an allocation failure would throw `std::bad_alloc` rather than
> return `nullptr`. The caller
> (`[spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]`)
> nonetheless checks the result for `nullptr` and maps that to
> `Error::MemoryAllocationFailed`.
>
> A Rust port allocates with `Layout::from_size_align(size, alignment)` (alignment
> already validated as a power of two) and maps allocation failure to
> `MemoryAllocationFailed`. The returned pointer is released with
> `[spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]`
> using the same alignment.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]
> inline void et_aligned_free(void* ptr, std::align_val_t alignment)

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]
> Frees a pointer previously returned by
> `[spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]`,
> passing the same `alignment`.
>
> Implementation: `return ::operator delete(ptr, alignment)` — the aligned form
> of global operator delete matching the aligned operator new. Passing a null
> `ptr` is safe (no-op), relied upon by `FreeSegment`.
>
> A Rust port deallocates with the same `Layout` (size and alignment) used to
> allocate.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader]
> class FileDescriptorDataLoader final : public executorch::runtime::DataLoader {
>   ET_NODISCARD executorch::runtime::Result<executorch::runtime::FreeableBuffer> load( size_t offset, size_t size, const DataLoader::SegmentInfo& segment_info);
>   const override;
>   ET_NODISCARD executorch::runtime::Result<size_t> size();
>   const override;
>   ET_NODISCARD executorch::runtime::Error load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer);
>   const override;
>   const char* const file_descriptor_uri_;
>   const size_t file_size_;
>   const std::align_val_t alignment_;
>   const int fd_;
> }

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn]
> FileDescriptorDataLoader(FileDescriptorDataLoader&& rhs) noexcept

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn]
> Move constructor `FileDescriptorDataLoader(FileDescriptorDataLoader&& rhs)
> noexcept`. Transfers ownership of the fd and the heap URI copy from `rhs` to
> the new instance and leaves `rhs` in an inert, safe-to-destroy state. Exists so
> the type is compatible with `Result`.
>
> Steps:
> 1. Copy each field from `rhs` into the new instance: `file_descriptor_uri_`,
>    `file_size_`, `alignment_`, `fd_`.
> 2. Reset the source's fields (via `const_cast`, since they are declared
>    `const`) to their moved-from sentinels: `rhs.file_descriptor_uri_ = nullptr`,
>    `rhs.file_size_ = 0`, `rhs.alignment_ = {}` (default-constructed
>    `std::align_val_t`), `rhs.fd_ = -1`.
>
> After the move, the source's destructor
> (`std::free(nullptr)` is safe and `::close(-1)` merely errors harmlessly) does
> nothing meaningful, and `size()`/`load()`/`load_into()` on the source would
> return `Error::InvalidState` because `fd_ == -1`. The copy constructor and both
> assignment operators are deleted, so move-construction is the only transfer.
>
> A Rust port expresses this as ordinary move semantics: ownership of the fd and
> owned URI string transfers, and the source is left with `fd_ == -1` / null name
> so its `Drop` is a no-op.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn]
> Result<FileDescriptorDataLoader>

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn]
> Factory `fromFileDescriptorUri(file_descriptor_uri, alignment)` (alignment
> defaults to `alignof(std::max_align_t)`). Parses an fd out of a `fd:///<n>`
> URI, takes ownership of that fd, caches the file size, copies the URI string,
> and returns a constructed loader inside a `Result`, or an error.
>
> Steps, in order:
> 1. `ET_CHECK_OR_RETURN_ERROR(is_power_of_2(alignment), ...)`: if `alignment` is
>    not a power of two (per
>    `[spec:et:sem:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn]`),
>    return `Error::InvalidArgument`.
> 2. Parse the fd via
>    `[spec:et:sem:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn]`.
>    If that returns an error (URI missing the `fd:///` prefix), propagate that
>    error unchanged.
> 3. `fstat(fd, &st)` to get the size. If it fails (`< 0`), log, `::close(fd)`,
>    and return `Error::AccessFailed`.
> 4. Set `file_size = st.st_size` (cast to `size_t`).
> 5. Duplicate the URI string with `::strdup(file_descriptor_uri)`. If it returns
>    `nullptr`, log, `::close(fd)`, and return `Error::MemoryAllocationFailed`.
> 6. Return `FileDescriptorDataLoader(fd, file_size, alignment,
>    file_descriptor_uri_copy)` via the private constructor; the instance now
>    owns `fd` and the `strdup`'d string (freed with `std::free` in the
>    destructor).
>
> Note the fd itself is never opened here — the caller already opened it and the
> loader takes ownership of that descriptor via the URI. On the stat/strdup
> failure paths the fd is closed before returning. The alignment argument is
> stored but this loader has no filesystem-path fallback, so it always uses
> `pread`.
>
> A Rust port returns `Result<FileDescriptorDataLoader, Error>`, validating
> alignment, parsing the fd from the URI, stat-ing for size, storing an owned
> copy of the URI, and closing the fd on the failure paths.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]
> Result<FreeableBuffer> FileDescriptorDataLoader::load( size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segment_info) const

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]
> `load(offset, size, segment_info)`: allocate an aligned buffer, read
> `[offset, offset+size)` from the fd into it, and return an owning
> `FreeableBuffer`. `segment_info` is unused. Returns `Result<FreeableBuffer>`.
>
> Steps, in order:
> 1. `ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Uninitialized")`: if
>    `fd_ == -1` (moved-from), return `Error::InvalidState`.
> 2. Bounds/overflow: `total_size = offset + size` via `c10::add_overflows`; if
>    overflow OR `total_size > file_size_`, return `Error::InvalidArgument`.
> 3. Empty-segment fast path: if `size == 0`, return `FreeableBuffer(nullptr, 0,
>    nullptr)` without allocating or reading.
> 4. Allocate an aligned buffer of `size` bytes via
>    `[spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-alloc-fn]`
>    with `alignment_`. If `nullptr`, log and return
>    `Error::MemoryAllocationFailed`.
> 5. Read via
>    `[spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn]`.
>    If it returns anything other than `Error::Ok`, free the aligned buffer and
>    return that error.
> 6. On success return `FreeableBuffer(aligned_buffer, size, FreeSegment, ctx)`
>    where `ctx` is `alignment_` reinterpreted as `void*` (through `uintptr_t`);
>    `FreeSegment`
>    (`[spec:et:sem:file-descriptor-data-loader.executorch.extension.free-segment-fn]`)
>    recovers the alignment from that context to free the buffer later. The
>    returned buffer owns its memory.
>
> A Rust port returns an owning buffer whose Drop frees the aligned allocation;
> the empty case returns an empty non-owning buffer.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn]
> ET_NODISCARD Error FileDescriptorDataLoader::load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer) const

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-into-fn]
> `load_into(offset, size, segment_info, buffer)`: read `[offset, offset+size)`
> from the fd into a caller-provided `buffer`. `segment_info` is unused. Returns
> `Error`.
>
> Validation, in order:
> 1. `fd_ >= 0` else `Error::InvalidState` ("Uninitialized").
> 2. Overflow/bounds: `total_size = offset + size` via `c10::add_overflows`; if
>    overflow OR `total_size > file_size_`, return `Error::InvalidArgument`.
> 3. `buffer != nullptr` else `Error::InvalidArgument`.
>
> Read loop (always uses `pread`; this loader has no no-`pread` fallback):
> - Track `needed = size` remaining and a write cursor `buf = (uint8_t*)buffer`;
>   `offset` advances as bytes are consumed.
> - While `needed > 0`:
>   - `chunk_size = min(needed, INT32_MAX)` — macOS reads fail with `EINVAL` past
>     `INT32_MAX`.
>   - `nread = ::pread(fd_, buf, chunk_size, offset)`.
>   - If `nread < 0 && errno == EINTR` (interrupted by signal, zero bytes read):
>     `continue` and retry the same chunk.
>   - If `nread <= 0` (`0` = premature EOF, unexpected given the bounds check;
>     `<0` = error): log and return `Error::AccessFailed`.
>   - Otherwise: `needed -= nread`, `buf += nread`, `offset += nread`, loop
>     again — short reads are handled by continuing until `needed` is 0.
> - After the loop return `Error::Ok`.
>
> The buffer is written in place; only the status is returned. A Rust port uses
> positional reads (`pread`/`read_at`) on the owned fd, looping over short reads
> and `EINTR`, capping each read at `i32::MAX`.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn]
> Result<size_t> FileDescriptorDataLoader::size() const

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.size-fn]
> `size()`: return the cached total file size. Returns `Result<size_t>`.
>
> Steps:
> 1. `ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Uninitialized")`: if
>    `fd_ == -1` (moved-from), return `Error::InvalidState`.
> 2. Otherwise return `file_size_`, the size captured by
>    `[spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.from-file-descriptor-uri-fn]`
>    at construction. The fd is not re-stat'd.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.free-segment-fn]
> void FreeSegment(void* context, void* data, ET_UNUSED size_t size)

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.free-segment-fn]
> `FreeSegment(context, data, size)`: a `FreeableBuffer::FreeFn`-compatible
> callback that releases a buffer allocated by
> `[spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.load-fn]`.
> `size` is unused.
>
> It reconstructs the alignment from `context` (set by `load` to the alignment
> value cast to `void*` through `uintptr_t`): `alignment =
> static_cast<std::align_val_t>(reinterpret_cast<uintptr_t>(context))`, then
> calls `et_aligned_free(data, alignment)` per
> `[spec:et:sem:file-descriptor-data-loader.executorch.extension.et-aligned-free-fn]`.
>
> A Rust port carries the alignment alongside the pointer and deallocates with
> the matching `Layout`.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn]
> static Result<int> getFDFromUri(const char* file_descriptor_uri)

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.get-fd-from-uri-fn]
> `getFDFromUri(file_descriptor_uri)`: parse an integer file descriptor out of a
> URI of the form `fd:///<n>`. Returns `Result<int>`.
>
> The prefix constant is `kFdFilesystemPrefix = "fd:///"` (6 characters).
>
> Steps:
> 1. `ET_CHECK_OR_RETURN_ERROR(strncmp(uri, "fd:///", 6) == 0, InvalidArgument,
>    ...)`: if the URI does not begin with the exact `fd:///` prefix, return
>    `Error::InvalidArgument`.
> 2. Compute `fd_len = strlen(uri) - strlen("fd:///")`, i.e. the length of the
>    remainder after the prefix.
> 3. Copy that remainder into a local NUL-terminated C string
>    `fd_without_prefix[fd_len + 1]` (a stack VLA): `memcpy` the `fd_len` bytes
>    starting at `uri + 6`, then set the trailing byte to `'\0'`.
> 4. Convert with `::atoi(fd_without_prefix)` and return the resulting `int`.
>
> Note: `atoi` performs no validation — a non-numeric remainder yields `0`, and a
> value out of `int` range is undefined behavior; the only rejected input is a
> missing/incorrect prefix. An empty remainder (`fd:///`) parses to `0`.
>
> A Rust port checks the `fd:///` prefix (returning `InvalidArgument` if absent)
> and parses the remainder as an integer; to match C semantics exactly, an
> unparseable remainder should be treated as `0` rather than an error (though a
> stricter parse is a reasonable, safer deviation).

> [spec:et:def:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn]
> static bool is_power_of_2(size_t value)

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.is-power-of-2-fn]
> `is_power_of_2(value)`: returns `true` iff `value` is a positive integer power
> of two, used to validate the `alignment` argument. Identical to
> `[spec:et:sem:file-data-loader.executorch.extension.is-power-of-2-fn]`.
>
> Returns `value > 0 && (value & ~(value - 1)) == value`. The `value > 0` guard
> rejects zero; `value & ~(value - 1)` isolates the lowest set bit and equals
> `value` exactly when `value` has a single set bit.
>
> A Rust port can use `value != 0 && value.is_power_of_two()`.

> [spec:et:def:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.operator-fn]
> FileDescriptorDataLoader& operator=(const FileDescriptorDataLoader&) = delete

> [spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.operator-fn]
> Copy-assignment operator, explicitly deleted (`= delete`).
> `FileDescriptorDataLoader` owns a unique file descriptor and a `strdup`'d URI
> copy, so it is not safely copyable; the copy constructor is likewise deleted
> and move-assignment (`operator=(FileDescriptorDataLoader&&)`) is also deleted.
> Only move-construction is permitted (for `Result` compatibility, per
> `[spec:et:sem:file-descriptor-data-loader.executorch.extension.file-descriptor-data-loader.file-descriptor-data-loader-fn]`).
>
> There is no runtime behavior: any copy-assignment is a compile-time error. A
> Rust port expresses this by not implementing `Clone` and by moving the value;
> ownership of the fd and URI transfers on move.

