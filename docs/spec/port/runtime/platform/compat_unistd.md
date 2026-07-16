# runtime/platform/compat_unistd.h

> [spec:et:def:compat-unistd.pread-fn]
> inline ssize_t pread(int fd, void* buf, size_t nbytes, size_t offset)

> [spec:et:sem:compat-unistd.pread-fn]
> Windows-only (`_WIN64`) shim providing a POSIX-`pread`-compatible positioned
> read. On non-Windows (`!_WIN64`) builds this function does not exist; the system
> `<unistd.h>` `pread` is used instead. Reads up to `nbytes` bytes from file
> descriptor `fd` starting at absolute byte `offset`, into `buf`, without moving
> the descriptor's current file position.
>
> Steps:
> 1. Zero-initialize a Win32 `OVERLAPPED` structure and set its offset fields from
>    `offset`: `Offset = (DWORD)(offset & 0xFFFFFFFF)` (low 32 bits) and
>    `OffsetHigh = (DWORD)(offset >> 32)` (high 32 bits). `offset` is a `size_t`
>    interpreted as an unsigned 64-bit file position.
> 2. Resolve the OS file `HANDLE` from `fd` via `_get_osfhandle(fd)`.
> 3. Call `ReadFile(file, buf, nbytes, &bytes_read, &overlapped)`. `nbytes` is
>    passed as the `DWORD` byte count; `bytes_read` receives the count actually
>    read.
> 4. Capture `GetLastError()`. If `ReadFile` returned false (failure) and the error
>    is `ERROR_IO_PENDING`, wait for completion via
>    `GetOverlappedResult(file, &overlapped, &bytes_read, TRUE)` (blocking); if that
>    also fails, refresh the error from `GetLastError()`.
> 5. On overall failure (result still false): translate the error into `errno` and
>    return `-1`. If the error is `ERROR_HANDLE_EOF`, set `errno = 0` (clean
>    end-of-file, distinguished from a real error); for any other error set
>    `errno = EIO`.
> 6. On success: return `bytes_read` as an `ssize_t`.
>
> Return type is `ssize_t`: the nonnegative number of bytes read on success (may be
> less than `nbytes`, including 0 at end of file), or `-1` on error. The descriptor
> position is unaffected because the read is positioned via `OVERLAPPED`. No
> argument validation is performed on `fd`, `buf`, or `nbytes` beyond what
> `_get_osfhandle`/`ReadFile` themselves enforce.

