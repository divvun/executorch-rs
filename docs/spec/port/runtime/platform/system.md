# runtime/platform/system.h

> [spec:et:def:system.et-pal-get-shared-library-name-fn]
> inline const char* et_pal_get_shared_library_name(const void* addr)

> [spec:et:sem:system.et-pal-get-shared-library-name-fn]
> Inline C-ABI helper that resolves the filesystem path of the shared library
> containing a given code/data address. Returns a `const char*` string.
>
> Two compile-time configurations exist, selected by the `ET_USE_LIBDL` macro:
>
> When `ET_USE_LIBDL` is defined (dynamic-linking debugging enabled on UNIX-like
> OSes, requires `<dlfcn.h>` / libdl):
> 1. Declare a `Dl_info info` struct and call `dladdr(addr, &info)`.
> 2. If `dladdr` returns nonzero (success) AND `info.dli_fname` is non-null,
>    return `info.dli_fname` — the pathname of the shared object that contains
>    `addr`.
> 3. Otherwise return the constant string `DYNAMIC_LIBRARY_NOT_FOUND` (the
>    literal `"NOT_FOUND"`).
>
> When `ET_USE_LIBDL` is NOT defined (default): the lookup block above is
> compiled out entirely; the `addr` argument is explicitly discarded (`(void)addr`)
> and the function returns the constant string `DYNAMIC_LIBRARY_NOT_SUPPORTED`
> (the literal `"NOT_SUPPORTED"`).
>
> No memory is allocated; all returned strings are either static string literals
> or storage owned by the dynamic linker. No validation of `addr` beyond what
> `dladdr` performs; passing a null or invalid `addr` yields
> `DYNAMIC_LIBRARY_NOT_FOUND` in the libdl configuration.

