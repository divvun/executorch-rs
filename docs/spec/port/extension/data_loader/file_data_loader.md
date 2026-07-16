# extension/data_loader/file_data_loader.cpp, extension/data_loader/file_data_loader.h

> [spec:et:def:file-data-loader.executorch.extension.et-aligned-alloc-fn]
> inline void* et_aligned_alloc(size_t size, std::align_val_t alignment)

> [spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]
> Allocates `size` bytes with the given `alignment`, returning the raw pointer
> or `nullptr` on failure. It is the aligned-allocation primitive used for both
> load buffers and the filename copy.
>
> Implementation: `return ::operator new(size, alignment, std::nothrow)`. The
> nothrow, aligned form of global operator new is used deliberately: ExecuTorch
> is built without exceptions, so on allocation failure this must return
> `nullptr` (callers such as `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn]`
> and `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]`
> check for null and return `Error::MemoryAllocationFailed`); a throwing `new`
> would unwind with no landing pad and abort the process.
>
> The returned pointer must later be released with the matching aligned free
> `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn]` using
> the same alignment.
>
> A Rust port allocates with an explicit `Layout::from_size_align(size,
> alignment)` (alignment already validated as a power of two by the caller) and
> maps allocation failure to `None`/`MemoryAllocationFailed` rather than
> panicking.

> [spec:et:def:file-data-loader.executorch.extension.et-aligned-free-fn]
> inline void et_aligned_free(void* ptr, std::align_val_t alignment)

> [spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn]
> Frees a pointer previously returned by
> `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]`,
> passing the same `alignment` that was used to allocate it.
>
> Implementation: `return ::operator delete(ptr, alignment)` — the aligned form
> of global operator delete matching the aligned operator new. Passing a null
> `ptr` is safe (a no-op), which is relied upon by the destructor and by
> `FreeSegment`.
>
> A Rust port deallocates with the same `Layout` (size and alignment) that was
> used to allocate; note the size must be recovered by the caller since operator
> delete's alignment-only overload does not carry it here.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader]
> class FileDataLoader final : public executorch::runtime::DataLoader {
>   ET_NODISCARD executorch::runtime::Result<executorch::runtime::FreeableBuffer> load( size_t offset, size_t size, const DataLoader::SegmentInfo& segment_info);
>   const override;
>   ET_NODISCARD executorch::runtime::Result<size_t> size();
>   const override;
>   ET_NODISCARD executorch::runtime::Error load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer);
>   const override;
>   const char* const file_name_;
>   const size_t file_size_;
>   const std::align_val_t alignment_;
>   const int fd_;
> }

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.file-data-loader-fn]
> FileDataLoader::~FileDataLoader()

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.file-data-loader-fn]
> Destructor `~FileDataLoader()`. Releases the two resources the instance owns:
> the heap copy of the filename and the open file descriptor.
>
> Steps, in order:
> 1. Free the filename copy: `et_aligned_free(const_cast<char*>(file_name_),
>    alignment_)` per
>    `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn]`.
>    `file_name_` may be `nullptr` if this instance was moved-from (the move
>    constructor nulls the source's fields); freeing null is safe.
> 2. If `fd_ == -1` (moved-from / uninitialized), return immediately without
>    closing.
> 3. Otherwise call `::close(fd_)` to close the file descriptor; its return
>    value is ignored.
>
> A Rust port models this as `Drop`: deallocate the owned filename buffer with
> its recorded alignment, and close the fd only if it is a valid (non-`-1`)
> descriptor. Move semantics leave the source with a null name and `fd_ == -1`
> so its Drop is a no-op.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.from-fn]
> Result<FileDataLoader> FileDataLoader::from( const char* file_name, size_t alignment)

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]
> Factory `FileDataLoader::from(file_name, alignment)` (alignment defaults to
> `alignof(std::max_align_t)`). Opens the named file, caches its size, copies the
> name, and returns a constructed `FileDataLoader` by value inside a `Result`, or
> an error.
>
> Steps, in order:
> 1. `ET_CHECK_OR_RETURN_ERROR(is_power_of_2(alignment), ...)`: if `alignment` is
>    not a power of two (per
>    `[spec:et:sem:file-data-loader.executorch.extension.is-power-of-2-fn]`),
>    log and return `Error::InvalidArgument`.
> 2. `ET_CHECK_OR_RETURN_ERROR(file_name != nullptr, ...)`: if `file_name` is
>    null, return `Error::InvalidArgument`.
> 3. Open the file read-only: `fd = ::open(file_name, O_RDONLY)`. `open` is used
>    instead of `fopen` to skip stdio buffering. If `fd < 0`, log the errno /
>    strerror and return `Error::AccessFailed`.
> 4. `fstat` the fd into `struct stat st`. If it fails (`< 0`), log, `::close(fd)`,
>    and return `Error::AccessFailed`.
> 5. Set `file_size = st.st_size` (cast to `size_t`).
> 6. Copy the filename into an aligned buffer of `strlen(file_name) + 1` bytes
>    (including the NUL terminator) via
>    `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]`
>    with `std::align_val_t(alignment)`. If allocation returns `nullptr`, log,
>    `::close(fd)`, and return `Error::MemoryAllocationFailed`.
> 7. `::strcpy` the filename (with NUL) into the aligned copy.
> 8. Return `FileDataLoader(fd, file_size, alignment, file_name_copy)` via the
>    private constructor; the instance now owns `fd` and `file_name_copy`.
>
> On any early-return error path after the fd is opened, the fd is closed before
> returning so no descriptor leaks. The deprecated `From()` wrapper simply
> forwards to `from()` with the same arguments.
>
> A Rust port returns `Result<FileDataLoader, Error>`, validating alignment and
> non-null name first, opening the file O_RDONLY, stat-ing for size, and storing
> an owned copy of the path; it must close the fd on the stat/alloc failure paths.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.load-fn]
> Result<FreeableBuffer> FileDataLoader::load( size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segment_info) const

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn]
> `load(offset, size, segment_info)`: allocate an aligned buffer, read
> `[offset, offset+size)` of the file into it, and return an owning
> `FreeableBuffer`. `segment_info` is unused. Returns `Result<FreeableBuffer>`.
>
> Steps, in order:
> 1. `ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Uninitialized")`: if the
>    fd is `-1` (moved-from), return `Error::InvalidState`.
> 2. Bounds/overflow check: compute `total_size = offset + size` with
>    `c10::add_overflows` detecting unsigned wraparound. If it overflowed OR
>    `total_size > file_size_`, return `Error::InvalidArgument`.
> 3. Empty-segment fast path: if `size == 0`, return `FreeableBuffer(nullptr, 0,
>    /*free_fn=*/nullptr)` without allocating or reading.
> 4. Allocate an aligned buffer of `size` bytes via
>    `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-alloc-fn]`
>    with `alignment_`. If it returns `nullptr`, log and return
>    `Error::MemoryAllocationFailed`.
> 5. Read into the buffer via
>    `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-into-fn]`
>    with the same `offset`, `size`, `segment_info`, and the aligned buffer. If
>    it returns anything other than `Error::Ok`, free the aligned buffer
>    (`et_aligned_free`) and return that error.
> 6. On success return `FreeableBuffer(aligned_buffer, size, FreeSegment, ctx)`,
>    where `ctx` is `alignment_` reinterpreted as a `void*` (cast through
>    `uintptr_t`). `FreeSegment`
>    (`[spec:et:sem:file-data-loader.executorch.extension.free-segment-fn]`) will
>    later recover the alignment from that context and free the buffer. The
>    returned buffer owns its memory.
>
> A Rust port returns an owning buffer type whose Drop frees the aligned
> allocation; the empty case returns an empty non-owning buffer.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.load-into-fn]
> ET_NODISCARD Error FileDataLoader::load_into( size_t offset, size_t size, ET_UNUSED const SegmentInfo& segment_info, void* buffer) const

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-into-fn]
> `load_into(offset, size, segment_info, buffer)`: read `[offset, offset+size)`
> of the file into a caller-provided `buffer`. `segment_info` is unused. Returns
> `Error`.
>
> Validation, in order:
> 1. `fd_ >= 0` else `Error::InvalidState` ("Uninitialized").
> 2. Overflow/bounds: `total_size = offset + size` via `c10::add_overflows`; if
>    overflow OR `total_size > file_size_`, return `Error::InvalidArgument`.
> 3. `buffer != nullptr` else `Error::InvalidArgument`.
>
> Read loop:
> - Track `needed = size` remaining bytes and a running `buf = (uint8_t*)buffer`
>   write cursor; `offset` advances as bytes are consumed.
> - Choose the read path at compile time via `ET_HAVE_PREAD`, which is 0 on
>   `__xtensa__` / `__hexagon__` (no thread-safe `pread`) and 1 elsewhere. When
>   `pread` is unavailable, open a *separate* fd on `file_name_` with `O_RDONLY`
>   (`dup_fd`) rather than `dup()`/`fcntl()`, because a dup'd fd would share the
>   file offset and race with concurrent seeks on other threads. When `pread` is
>   available, `dup_fd == fd_`.
> - While `needed > 0`:
>   - `chunk_size = min(needed, INT32_MAX)`. This cap exists because macOS reads
>     fail with `EINVAL` when the count exceeds `INT32_MAX`.
>   - Read: with pread, `nread = ::pread(dup_fd, buf, chunk_size, offset)`.
>     Without pread, first `::lseek(dup_fd, offset, SEEK_SET)`; if that returns
>     `(off_t)-1`, set `nread = -1`, else `nread = ::read(dup_fd, buf,
>     chunk_size)`.
>   - If `nread < 0 && errno == EINTR` (interrupted by a signal, zero bytes
>     read): `continue` and retry the same chunk.
>   - If `nread <= 0` (`0` = premature EOF, which shouldn't happen given the
>     bounds check; `<0` = error): log the failure; if `!ET_HAVE_PREAD` close
>     `dup_fd`; return `Error::AccessFailed`.
>   - Otherwise (partial or full read): `needed -= nread`, `buf += nread`,
>     `offset += nread`, and loop again — short reads are handled by continuing
>     until `needed` reaches 0.
> - After the loop, if `!ET_HAVE_PREAD` close `dup_fd`, then return `Error::Ok`.
>
> The buffer is written in place; nothing is resized or returned other than the
> status. A Rust port uses positional reads (`pread`/`read_at`) where available,
> loops over short reads and `EINTR`, caps each read at `i32::MAX`, and on the
> no-pread targets opens a fresh fd per call and `seek`s before each `read`,
> closing that fd on every exit path.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.size-fn]
> Result<size_t> FileDataLoader::size() const

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn]
> `size()`: return the cached total file size. Returns `Result<size_t>`.
>
> Steps:
> 1. `ET_CHECK_OR_RETURN_ERROR(fd_ >= 0, InvalidState, "Uninitialized")`: if the
>    fd is `-1` (moved-from), return `Error::InvalidState`.
> 2. Otherwise return `file_size_`, the size captured by
>    `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]`
>    at construction. The file is not re-stat'd; the value is the size at open
>    time.

> [spec:et:def:file-data-loader.executorch.extension.free-segment-fn]
> void FreeSegment(void* context, void* data, ET_UNUSED size_t size)

> [spec:et:sem:file-data-loader.executorch.extension.free-segment-fn]
> `FreeSegment(context, data, size)`: a `FreeableBuffer::FreeFn`-compatible
> callback that releases a buffer allocated by
> `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-fn]`.
> `size` is unused.
>
> It reconstructs the alignment from `context` (which `load` set to the alignment
> value cast to a `void*` via `uintptr_t`): `alignment =
> static_cast<std::align_val_t>(reinterpret_cast<uintptr_t>(context))`, then
> calls `et_aligned_free(data, alignment)` per
> `[spec:et:sem:file-data-loader.executorch.extension.et-aligned-free-fn]`.
>
> A Rust port carries the alignment alongside the pointer (e.g. in the owning
> buffer's Drop) and deallocates with the matching `Layout`.

> [spec:et:def:file-data-loader.executorch.extension.is-power-of-2-fn]
> static bool is_power_of_2(size_t value)

> [spec:et:sem:file-data-loader.executorch.extension.is-power-of-2-fn]
> `is_power_of_2(value)`: returns `true` iff `value` is a positive integer power
> of two, used to validate the `alignment` argument.
>
> Returns `value > 0 && (value & ~(value - 1)) == value`. The `value > 0` guard
> rejects zero (and makes the `value - 1` well-defined for unsigned wraparound).
> `value & ~(value - 1)` isolates the lowest set bit; it equals `value` exactly
> when `value` has a single set bit, i.e. is a power of two.
>
> A Rust port can use `value != 0 && value.is_power_of_two()` for the same
> result.

> [spec:et:def:file-data-loader.executorch.extension.file-data-loader.operator-fn]
> FileDataLoader& operator=(const FileDataLoader&) = delete

> [spec:et:sem:file-data-loader.executorch.extension.file-data-loader.operator-fn]
> Copy-assignment operator, explicitly deleted (`= delete`). `FileDataLoader`
> owns a unique file descriptor and a heap filename copy, so it is not safely
> copyable; the copy constructor is likewise deleted and move-assignment
> (`operator=(FileDataLoader&&)`) is also deleted. Only move-construction is
> permitted (so the type is compatible with `Result`).
>
> There is no runtime behavior to implement: any attempt to copy-assign is a
> compile-time error. A Rust port expresses this by not deriving/implementing
> `Clone` and by moving (not copying) the value; ownership of the fd and name
> transfers on move, and the private move constructor invalidates the source.

