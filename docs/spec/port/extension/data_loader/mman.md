# extension/data_loader/mman.h

> [spec:et:def:mman.fcntl-rdadvise-apple-fn]
> ET_INLINE void fcntl_rdadvise_apple(int fd, size_t file_size)

> [spec:et:sem:mman.fcntl-rdadvise-apple-fn]
> Hints the OS to eagerly read-ahead the whole file into the buffer cache, as a
> best-effort cold-start optimization. It never reports success or failure and
> has no return value; any error from the underlying call is ignored.
>
> Behavior is platform-conditional at compile time:
> - On Apple platforms (`__APPLE__` defined): build a `struct radvisory` advice
>   with `ra_offset = 0` and `ra_count = static_cast<int>(file_size)` (the whole
>   file starting at offset 0; note `file_size` is narrowed from `size_t` to
>   `int`, so files larger than `INT_MAX` are truncated in the request), then
>   call `::fcntl(fd, F_RDADVISE, &advice)`, ignoring the return value.
> - On the non-Apple POSIX branch of this header: a no-op; both `fd` and
>   `file_size` are explicitly discarded (`(void)` cast) and nothing is called.
> - The `_WIN32` branch of the same header provides an identically-named no-op
>   that discards both arguments.
>
> A Rust port should model this as: on macOS, issue the `F_RDADVISE` fcntl with
> offset 0 and count `file_size` (clamped to `i32`), ignoring errors; on every
> other target, do nothing.

> [spec:et:def:mman.get-file-stat-fn]
> ET_INLINE int get_file_stat(int fd, size_t* out_size)

> [spec:et:sem:mman.get-file-stat-fn]
> Retrieves the size in bytes of the file behind an open file descriptor.
>
> POSIX (non-`_WIN32`) branch:
> - Declare a local `struct stat st`.
> - Call `err = ::fstat(fd, &st)`.
> - If `err >= 0` (success), write the file size to the caller's out-param:
>   `*out_size = static_cast<size_t>(st.st_size)`. On failure (`err < 0`),
>   `*out_size` is left untouched.
> - Return `err` verbatim (0 or positive on success, negative on failure with
>   `errno` set by `fstat`).
>
> Windows (`_WIN32`) branch: identical structure but uses `struct _stat64` and
> `::_fstat64(fd, &st)` so 64-bit file sizes are reported correctly; same
> `err >= 0` guard, same `*out_size` assignment, same return of `err`.
>
> A Rust port should return the file size and an error status; the size out-param
> is only meaningful when the call succeeds.

> [spec:et:def:mman.get-mmap-offset-fn]
> ET_INLINE off_t get_mmap_offset(size_t offset)

> [spec:et:sem:mman.get-mmap-offset-fn]
> Converts a byte `offset` (a `size_t`) into the platform's native `mmap` offset
> type, purely a type cast with no bounds checking or side effects.
>
> POSIX (non-`_WIN32`) branch: return `static_cast<off_t>(offset)`. `off_t` is
> the signed offset type used by `mmap`/`pread`; on a 32-bit `off_t` build a
> value above its max would wrap, but no check is performed here.
>
> Windows (`_WIN32`) branch: same value cast to `uint64_t` instead of `off_t`
> (`static_cast<uint64_t>(offset)`), matching the Win32 mmap-shim offset type.
>
> A Rust port models this as a widening/identity conversion of the offset to the
> platform mmap offset integer type.

> [spec:et:def:mman.get-os-page-size-fn]
> ET_INLINE long get_os_page_size()

> [spec:et:sem:mman.get-os-page-size-fn]
> Returns the OS memory page/allocation granularity in bytes, used to page-align
> mmap offsets and lengths. No arguments, no error path.
>
> POSIX (non-`_WIN32`) branch: return `sysconf(_SC_PAGESIZE)` (this variant's
> declared return type is `size_t`). This is the virtual-memory page size.
>
> Windows (`_WIN32`) branch (the signature captured by the `def` rule,
> returning `long`): call `GetSystemInfo(&si)` and return the larger of
> `si.dwAllocationGranularity` and `si.dwPageSize` — i.e.
> `dwAllocationGranularity > dwPageSize ? dwAllocationGranularity : dwPageSize`.
> Windows requires mmap base offsets to be aligned to the allocation
> granularity, which is typically larger than the page size, so the max of the
> two is the safe alignment.
>
> A Rust port should return the platform page size on Unix and, on Windows, the
> max of allocation granularity and page size.

> [spec:et:def:mman.madvise-pages-willneed-sequential-fn]
> ET_INLINE void madvise_pages_willneed_sequential(void* addr, size_t len)

> [spec:et:sem:mman.madvise-pages-willneed-sequential-fn]
> Advises the kernel about an mmap'd region `[addr, addr+len)` to reduce
> page-fault stutter during model init: prefetch its pages and expect sequential
> access. Best-effort with no return value; all errors ignored.
>
> POSIX (non-`_WIN32`) branch, each guarded by a compile-time `#ifdef` because
> the advice constants are missing on some libcs (e.g. the Hexagon DSP toolchain):
> - If `MADV_WILLNEED` is defined, call `::madvise(addr, len, MADV_WILLNEED)`
>   (schedule read-ahead of the pages).
> - If `MADV_SEQUENTIAL` is defined, call `::madvise(addr, len, MADV_SEQUENTIAL)`
>   (hint sequential access so the kernel reads ahead more aggressively and
>   releases pages behind the cursor).
> - If a constant is not defined, its call is omitted entirely; if neither is
>   defined the function is a pure no-op.
>
> Windows (`_WIN32`) branch: an unconditional no-op; `addr` and `len` are
> discarded via `(void)` casts (no madvise equivalent in the mman_windows shim).
>
> A Rust port models this as: on Unix, issue `madvise` with WILLNEED and then
> SEQUENTIAL where the constants exist, ignoring errors; elsewhere, do nothing.

