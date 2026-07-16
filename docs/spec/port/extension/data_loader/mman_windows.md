# extension/data_loader/mman_windows.cpp

> [spec:et:def:mman-windows.map-mman-error-fn]
> static int __map_mman_error(const DWORD err, const int deferr)

> [spec:et:sem:mman-windows.map-mman-error-fn]
> Translates a Windows error code into a POSIX `errno`-style value.
>
> Steps:
> 1. If `err == 0`, return 0 (no error).
> 2. Otherwise, return `err` unchanged (cast from `DWORD` to `int`).
>
> The `deferr` parameter is accepted but currently ignored (the intended
> per-code mapping table is not implemented — see anomalies). Callers pass
> `EPERM` as `deferr` throughout this file, but that default is never used;
> the raw Windows `DWORD`/`HRESULT` value is returned as-is for any nonzero
> input. Rust ports should preserve this passthrough behavior: nonzero
> Windows codes are stored directly into the `errno` slot by callers.

> [spec:et:def:mman-windows.map-mmap-prot-file-fn]
> static DWORD __map_mmap_prot_file(const int prot)

> [spec:et:sem:mman-windows.map-mmap-prot-file-fn]
> Maps a POSIX `mmap` protection bitmask (`prot`) to a Windows
> `MapViewOfFile` desired-access flag set (`DWORD`).
>
> The POSIX bits are: `PROT_NONE`=0, `PROT_READ`=1, `PROT_WRITE`=2,
> `PROT_EXEC`=4. The Windows flags are `FILE_MAP_READ`, `FILE_MAP_WRITE`,
> and `FILE_MAP_EXECUTE` (0x0020).
>
> Steps:
> 1. Start with `desiredAccess = 0`.
> 2. If `prot == PROT_NONE` (exactly 0), return 0 immediately.
> 3. If `prot & PROT_READ` is set, OR in `FILE_MAP_READ`.
> 4. If `prot & PROT_WRITE` is set, OR in `FILE_MAP_WRITE`.
> 5. If `prot & PROT_EXEC` is set, OR in `FILE_MAP_EXECUTE`.
> 6. Return the accumulated `desiredAccess`.
>
> Bits are additive/independent; any combination of read/write/exec is
> representable. This is the file-view access mask, distinct from the page
> protection constant produced by
> `[spec:et:sem:mman-windows.map-mmap-prot-page-fn]`.

> [spec:et:def:mman-windows.map-mmap-prot-page-fn]
> static DWORD __map_mmap_prot_page(const int prot)

> [spec:et:sem:mman-windows.map-mmap-prot-page-fn]
> Maps a POSIX `mmap` protection bitmask (`prot`) to a single Windows page
> protection constant (`DWORD`) suitable for `CreateFileMapping` /
> `VirtualProtect`.
>
> The POSIX bits are: `PROT_NONE`=0, `PROT_READ`=1, `PROT_WRITE`=2,
> `PROT_EXEC`=4.
>
> Steps:
> 1. Start with `protect = 0`.
> 2. If `prot == PROT_NONE` (exactly 0), return 0 immediately.
> 3. If `prot & PROT_EXEC` is set:
>    - If `prot & PROT_WRITE` is also set, `protect = PAGE_EXECUTE_READWRITE`.
>    - Otherwise `protect = PAGE_EXECUTE_READ`.
> 4. Else (no exec bit):
>    - If `prot & PROT_WRITE` is set, `protect = PAGE_READWRITE`.
>    - Otherwise `protect = PAGE_READONLY`.
> 5. Return `protect`.
>
> Note: the read bit (`PROT_READ`) is not inspected; the result is always at
> least read-capable when nonzero. Unlike the file-access mask in
> `[spec:et:sem:mman-windows.map-mmap-prot-file-fn]`, this yields exactly one
> of the mutually-exclusive `PAGE_*` constants rather than an OR of flags.

> [spec:et:def:mman-windows.mlock-fn]
> int mlock(const void* addr, size_t len)

> [spec:et:sem:mman-windows.mlock-fn]
> POSIX-compatible `mlock` shim: pins the pages in `[addr, addr+len)` into
> the process working set (physical RAM).
>
> Steps:
> 1. Call `virtual_lock_allowing_working_set_growth((LPVOID)addr, len)` per
>    `[spec:et:sem:mman-windows.virtual-lock-allowing-working-set-growth-fn]`,
>    which attempts `VirtualLock` and, if it fails specifically because the
>    working set is too small, grows the working set and retries.
> 2. If the returned `HRESULT` indicates success (`SUCCEEDED(hr)`, i.e. the
>    high bit is clear), return 0.
> 3. Otherwise set `errno` to `__map_mman_error(hr, EPERM)` (per
>    `[spec:et:sem:mman-windows.map-mman-error-fn]`, which returns the raw
>    `hr` for any nonzero value) and return -1.
>
> Note `len` is a `size_t` but is passed to a `DWORD`-typed parameter,
> truncating to 32 bits on the way in.

> [spec:et:def:mman-windows.mmap-fn]
> void* mmap( void* addr, size_t len, int prot, int flags, int fildes, uint64_t off)

> [spec:et:sem:mman-windows.mmap-fn]
> POSIX-compatible `mmap` shim over the Windows file-mapping API. Maps `len`
> bytes of file descriptor `fildes` (or anonymous memory) starting at file
> offset `off`, with protection `prot` and flags `flags`. The `addr` hint is
> ignored. Returns a pointer to the mapped view, or `MAP_FAILED`
> (`(void*)-1`) on error with `errno` set.
>
> Steps:
> 1. Set `errno = 0`.
> 2. Argument validation — if ANY of the following hold, set `errno=EINVAL`
>    and return `MAP_FAILED`:
>    - `len == 0`; or
>    - `flags & MAP_FIXED` (0x10) is set (fixed mappings unsupported); or
>    - `prot == PROT_EXEC` exactly (execute-only unsupported).
> 3. Overflow check: if `off > UINT64_MAX - len`, set `errno=EINVAL` and
>    return `MAP_FAILED`.
> 4. Compute `maxSize = off + len` (as `uint64_t`; this is the total number
>    of bytes from the start of the file that the mapping object must span).
> 5. Split `off` into `dwFileOffsetLow = off & 0xFFFFFFFF` and
>    `dwFileOffsetHigh = (off >> 32) & 0xFFFFFFFF`. Split `maxSize` likewise
>    into `dwMaxSizeLow` / `dwMaxSizeHigh`.
> 6. Compute `protect = __map_mmap_prot_page(prot)` (page protection, per
>    `[spec:et:sem:mman-windows.map-mmap-prot-page-fn]`) and
>    `desiredAccess = __map_mmap_prot_file(prot)` (view access mask, per
>    `[spec:et:sem:mman-windows.map-mmap-prot-file-fn]`).
> 7. Determine the file handle `h`: if `MAP_ANONYMOUS` (0x20) is NOT set,
>    `h = (HANDLE)_get_osfhandle(fildes)`; otherwise `h =
>    INVALID_HANDLE_VALUE` (anonymous mapping backed by the page file).
> 8. If not anonymous and `h == INVALID_HANDLE_VALUE`, set `errno=EBADF` and
>    return `MAP_FAILED`.
> 9. Create the file-mapping object:
>    `fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow,
>    NULL)`. If `fm == NULL`, set `errno = __map_mman_error(GetLastError(),
>    EPERM)` and return `MAP_FAILED`.
> 10. Map the view:
>    `map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh,
>    dwFileOffsetLow, len)`.
> 11. Close the mapping handle `fm` (the view keeps the mapping alive), via
>    `CloseHandle(fm)`, regardless of whether the view succeeded.
> 12. If `map == NULL`, set `errno = __map_mman_error(GetLastError(), EPERM)`
>    and return `MAP_FAILED`.
> 13. Return `map`.

> [spec:et:def:mman-windows.mprotect-fn]
> int mprotect(void* addr, size_t len, int prot)

> [spec:et:sem:mman-windows.mprotect-fn]
> POSIX-compatible `mprotect` shim: changes the page protection of the
> region `[addr, addr+len)`.
>
> Steps:
> 1. Compute `newProtect = __map_mmap_prot_page(prot)` per
>    `[spec:et:sem:mman-windows.map-mmap-prot-page-fn]`.
> 2. Declare `oldProtect = 0` (required out-param for the Windows call; its
>    returned value is discarded).
> 3. Call `VirtualProtect(addr, len, newProtect, &oldProtect)`.
> 4. On success (nonzero return), return 0.
> 5. On failure, set `errno = __map_mman_error(GetLastError(), EPERM)` and
>    return -1.

> [spec:et:def:mman-windows.msync-fn]
> int msync(void* addr, size_t len, int flags)

> [spec:et:sem:mman-windows.msync-fn]
> POSIX-compatible `msync` shim: flushes dirty pages of the mapped region
> `[addr, addr+len)` back to the backing file.
>
> Steps:
> 1. Call `FlushViewOfFile(addr, len)`.
> 2. On success (nonzero return), return 0.
> 3. On failure, set `errno = __map_mman_error(GetLastError(), EPERM)` and
>    return -1.
>
> The `flags` argument (`MS_ASYNC`, `MS_SYNC`, `MS_INVALIDATE`) is accepted
> but ignored; `FlushViewOfFile` performs the flush unconditionally and does
> not distinguish async vs. sync semantics.

> [spec:et:def:mman-windows.munlock-fn]
> int munlock(const void* addr, size_t len)

> [spec:et:sem:mman-windows.munlock-fn]
> POSIX-compatible `munlock` shim: unpins pages in `[addr, addr+len)` from
> the working set, allowing them to be paged out again.
>
> Steps:
> 1. Call `VirtualUnlock((LPVOID)addr, len)`.
> 2. On success (nonzero return), return 0.
> 3. On failure, set `errno = __map_mman_error(GetLastError(), EPERM)` and
>    return -1.

> [spec:et:def:mman-windows.munmap-fn]
> int munmap(void* addr, size_t len)

> [spec:et:sem:mman-windows.munmap-fn]
> POSIX-compatible `munmap` shim: unmaps a previously mapped view.
>
> Steps:
> 1. Call `UnmapViewOfFile(addr)`.
> 2. On success (nonzero return), return 0.
> 3. On failure, set `errno = __map_mman_error(GetLastError(), EPERM)` and
>    return -1.
>
> The `len` argument is accepted but ignored; `UnmapViewOfFile` unmaps the
> entire view identified by its base address `addr` (which must be a value
> previously returned by `[spec:et:sem:mman-windows.mmap-fn]`).

> [spec:et:def:mman-windows.try-grow-process-memory-working-set-fn]
> HRESULT try_grow_process_memory_working_set(DWORD dwSizeRequired)

> [spec:et:sem:mman-windows.try-grow-process-memory-working-set-fn]
> Grows the current process's minimum working-set size by `dwSizeRequired`
> bytes so a subsequent `VirtualLock` of that many bytes can succeed. Returns
> an `HRESULT` (`S_OK` on success).
>
> Steps:
> 1. Query the current limits:
>    `GetProcessWorkingSetSize(GetCurrentProcess(), &minWorkingSetInitial,
>    &maxWorkingSet)`. If it fails (returns 0/false), return `GetLastError()`
>    as the HRESULT.
> 2. Compute `minWorkingSet = minWorkingSetInitial + dwSizeRequired`.
> 3. Overflow guard: if `minWorkingSet < minWorkingSetInitial` (the addition
>    wrapped), return `HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW)`.
> 4. If `maxWorkingSet < minWorkingSet`, raise the maximum to equal the new
>    minimum (`maxWorkingSet = minWorkingSet`).
> 5. Apply the new limits:
>    `SetProcessWorkingSetSize(GetCurrentProcess(), minWorkingSet,
>    maxWorkingSet)`. If it fails, return `GetLastError()`.
> 6. Return `S_OK`.

> [spec:et:def:mman-windows.virtual-lock-allowing-working-set-growth-fn]
> HRESULT virtual_lock_allowing_working_set_growth(void* pMem, DWORD dwSize)

> [spec:et:sem:mman-windows.virtual-lock-allowing-working-set-growth-fn]
> Locks `dwSize` bytes at `pMem` into RAM, transparently growing the process
> working set if the lock initially fails because the working set is too
> small. Returns an `HRESULT`.
>
> Steps:
> 1. `hr = virtual_lock(pMem, dwSize)` per
>    `[spec:et:sem:mman-windows.virtual-lock-fn]`.
> 2. If `hr == HRESULT_FROM_WIN32(STATUS_SECTION_TOO_BIG)` (the code
>    `0xC0000040` wrapped through `HRESULT_FROM_WIN32`, meaning the lock
>    exceeded the current working-set limit):
>    a. Call `try_grow_process_memory_working_set(dwSize)` per
>       `[spec:et:sem:mman-windows.try-grow-process-memory-working-set-fn]`;
>       if that returns a failure HRESULT (high bit set), return it
>       immediately (via the `RETURN_IF_FAILED` macro).
>    b. Retry `virtual_lock(pMem, dwSize)`; if it fails, return its HRESULT
>       immediately.
>    c. If both succeed, fall through.
> 3. Return `hr`. (Note: after a successful retry, `hr` still holds the
>    original `HRESULT_FROM_WIN32(STATUS_SECTION_TOO_BIG)` failure value — see
>    anomalies; only the `RETURN_IF_FAILED` early-returns above and the
>    non-`STATUS_SECTION_TOO_BIG` path yield the "real" result.)

> [spec:et:def:mman-windows.virtual-lock-fn]
> HRESULT virtual_lock(void* pMem, DWORD dwSize)

> [spec:et:sem:mman-windows.virtual-lock-fn]
> Thin wrapper over the Windows `VirtualLock` API that locks `dwSize` bytes
> at `pMem` into physical memory. Returns an `HRESULT`.
>
> Steps:
> 1. Call `VirtualLock(pMem, dwSize)`.
> 2. On failure (returns 0/false), return `GetLastError()` as the HRESULT.
> 3. On success, return `S_OK`.

