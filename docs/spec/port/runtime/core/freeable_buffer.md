# runtime/core/freeable_buffer.h

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer]
> class FreeableBuffer final {
>   struct PointerData { const void* data_; FreeFn free_fn_; };
>   struct UInt64Data { // A pointer value cast to uint64_t. uint64_t data_; FreeUInt64Fn free_fn_; };
>   std::variant<PointerData, UInt64Data> data_;
>   void* free_fn_context_;
>   size_t size_;
> }

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-fn]
> const void* data() const

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn]
> Returns the buffer's data pointer as `const void*`. `ET_CHECK_MSG` asserts
> that `data_` currently holds the `PointerData` alternative of the variant;
> if the buffer is instead backed by a `UInt64Data` (uint64 pointer value),
> this aborts with the message directing callers to `data_uint64_type()`.
> When the check passes, return `std::get<PointerData>(data_).data_`, which is
> `nullptr` if the buffer has been freed (or was never populated). No
> mutation. For a non-aborting variant see
> `[spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn]`.

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn]
> Result<const void*> data_safe() const

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-safe-fn]
> Non-aborting variant of `data()`
> (`[spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-fn]`).
> `ET_CHECK_OR_RETURN_ERROR` checks that `data_` holds the `PointerData`
> alternative; if it instead holds `UInt64Data`, return
> `Error::InvalidType` (wrapped in the `Result<const void*>`) with a message
> directing the caller to `data_uint64_type()`. Otherwise return
> `std::get<PointerData>(data_).data_` as a successful `Result<const void*>`
> (which is `nullptr` if the data has been freed). No mutation.

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn]
> Result<uint64_t> data_uint64_type() const

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.data-uint64-type-fn]
> Returns the buffer's data address as a `uint64_t`, for buffers backed by the
> `UInt64Data` alternative (memory on a different core whose pointer value may
> be wider than the local `void*`). `ET_CHECK_OR_RETURN_ERROR` checks that
> `data_` holds `UInt64Data`; if it instead holds `PointerData`, return
> `Error::InvalidType` (in the `Result<uint64_t>`) with a message directing
> the caller to `data()`. Otherwise return `std::get<UInt64Data>(data_).data_`
> as a successful result (which is `0` if the data has been freed). No
> mutation.

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.free-fn]
> void Free()

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.free-fn]
> Frees the buffer's data if not already freed; idempotent and safe to call
> multiple times (also invoked by the destructor).
>
> Case A — `data_` holds `PointerData`: take a reference `ptr_data`. If
> `ptr_data.data_ != nullptr` AND `ptr_data.free_fn_ != nullptr`, invoke
> `free_fn_(free_fn_context_, const_cast<void*>(ptr_data.data_), size_)`
> exactly once. Then set `ptr_data.data_ = nullptr` and `size_ = 0`. (No
> truncation concern: `free_fn_` here always came from the `void*` ctor.)
>
> Case B — `data_` holds `UInt64Data`: take a reference `int64_data`. If
> `int64_data.data_ != 0` AND `int64_data.free_fn_ != nullptr`, invoke
> `free_fn_(free_fn_context_, int64_data.data_, size_)` exactly once. Then set
> `int64_data.data_ = 0` and `size_ = 0`.
>
> After either case the buffer is in the freed state (null/zero data, size 0),
> so a subsequent `Free()` performs no callback. The free function must be
> thread-safe per the ctor contract. Returns void. The variant tag (which
> alternative is active) is not changed.

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn]
> FreeableBuffer(FreeableBuffer&& rhs) noexcept

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn]
> Move constructor (`noexcept`). Transfers ownership of the data from `rhs` to
> the new object, leaving `rhs` in an empty/freed state.
>
> Step 1: copy `rhs`'s fields into the new object: `data_ = rhs.data_`
> (including which variant alternative is active and its contents),
> `free_fn_context_ = rhs.free_fn_context_`, `size_ = rhs.size_`.
>
> Step 2: reset `rhs`'s data to a null-but-same-variant value: if `rhs.data_`
> holds `PointerData`, set `rhs.data_ = PointerData{nullptr, nullptr}`;
> otherwise set `rhs.data_ = UInt64Data{0, nullptr}`.
>
> Step 3: set `rhs.free_fn_context_ = nullptr` and `rhs.size_ = 0`.
>
> After this, only the new object will invoke the free function; `rhs`'s
> destructor / `Free()` becomes a no-op. Copy construction and both
> assignment operators are deleted
> (`[spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.operator-fn]`).

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.pointer-data]
> struct PointerData {
>   const void* data_;
>   FreeFn free_fn_;
> }

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.size-fn]
> size_t size() const

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.size-fn]
> Returns the stored `size_` (bytes), regardless of which variant alternative
> backs the data. Returns 0 if the data has been freed (Free sets `size_ = 0`)
> or if the buffer was default-constructed. No mutation.

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.u-int64-data]
> struct UInt64Data {
>   uint64_t data_;
>   FreeUInt64Fn free_fn_;
> }

> [spec:et:def:freeable-buffer.executorch.runtime.freeable-buffer.operator-fn]
> FreeableBuffer& operator=(FreeableBuffer&& rhs) noexcept = delete

> [spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.operator-fn]
> Move-assignment operator is explicitly `= delete`d (as are the copy
> constructor and copy-assignment operator). FreeableBuffer therefore supports
> only move construction
> (`[spec:et:sem:freeable-buffer.executorch.runtime.freeable-buffer.freeable-buffer-fn]`)
> and destruction; it cannot be assigned. A Rust port models this as a
> move-only, non-reassignable owner (no `Clone`, no assignment that would
> require dropping/re-owning the underlying buffer). There is no runtime
> behavior to implement — any attempt to move-assign is a compile error.

