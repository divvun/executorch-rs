# backends/xnnpack/runtime/XNNStatus.h

> [spec:et:def:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn]
> inline const char* xnn_status_to_string(enum xnn_status type)

> [spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn]
> Maps an `xnn_status` enum value to a stable, null-terminated C string
> literal naming that status, for use in log/error messages.
>
> Backing data. Two module-level constant tables are used:
> - `kOffset`: a 7-element `uint16_t` array `{0, 19, 44, 73, 98, 131, 163}`,
>   giving the byte offset of each status name within `kData`.
> - `kData`: a single concatenated buffer of 7 null-terminated names, in
>   this exact order (each entry followed by a `\0`):
>   index 0 → `"xnn_status_success"`,
>   index 1 → `"xnn_status_uninitialized"`,
>   index 2 → `"xnn_status_invalid_parameter"`,
>   index 3 → `"xnn_status_invalid_state"`,
>   index 4 → `"xnn_status_unsupported_parameter"`,
>   index 5 → `"xnn_status_unsupported_hardware"`,
>   index 6 → `"xnn_status_out_of_memory"`.
>   The `kOffset` values are exactly the cumulative byte positions of each
>   name given those lengths-plus-terminator (0, then 0+19, 44, 73, 98,
>   131, 163).
>
> Behavior:
> 1. Treats the input `xnn_status` `type` as its integer enum value, which
>    is assumed to lie in the closed range [0, 6] (0 = `xnn_status_success`
>    ... 6 = `xnn_status_out_of_memory`). In the C++ this precondition is
>    guarded by `assert(type <= xnn_status_out_of_memory)`, an assertion
>    active only in debug builds; there is no runtime bounds check in
>    release builds. A Rust port should treat out-of-range input as a
>    programming error (e.g. debug-assert / panic), matching the assert.
> 2. Returns a pointer to the byte at `kData[kOffset[type]]`, i.e. the
>    start of the null-terminated name string for that status index.
>
> Returns: a pointer/reference to the interned static string; the caller
> does not own it and must not free it. No allocation, no copying, no error
> return.

