# runtime/backend/backend_options_map.h

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map]
> class LoadBackendOptionsMap final {
>   struct EntryView { const char* backend_id = nullptr; Span<const BackendOption> options; };
>   static constexpr size_t kMaxBackends = 8;
>   static constexpr size_t kMaxBackendIdLength = 64;
>   struct Entry { char backend_id[kMaxBackendIdLength]; Span<BackendOption> options; };
>   Entry entries_[kMaxBackends];
>   size_t size_;
> }

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry]
> struct Entry {
>   char backend_id[kMaxBackendIdLength];
>   Span<BackendOption> options;
> }

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry-at-fn]
> EntryView entry_at(size_t index) const

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.entry-at-fn]
> Const enumeration accessor returning the entry at `index` as an `EntryView` value.
>
> Steps:
> 1. `ET_DCHECK_MSG(index < size_, ...)`: in debug builds, aborts if `index >= size_` (out of the populated range). In release builds `ET_DCHECK` is a no-op, so an out-of-range index yields undefined behavior — callers must bound `index` via `size()`.
> 2. Construct and return `EntryView{ entries_[index].backend_id, Span<const BackendOption>(entries_[index].options.data(), entries_[index].options.size()) }`: the entry's backend_id C-string pointer and a const view (same data pointer and size) over that entry's stored option span.
>
> The returned view is non-owning and valid only until the map is next mutated or destroyed. Entries are stored in insertion order (see `[spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]`), so index order equals insertion order minus any in-place replacements.

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.entry-view]
> struct EntryView {
>   const char* backend_id = nullptr;
>   Span<const BackendOption> options;
> }

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn]
> Span<const BackendOption> get_options(const char* backend_id) const

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.get-options-fn]
> Const lookup of the option span associated with `backend_id`.
>
> Steps:
> 1. If `backend_id == nullptr`, return an empty span `Span<const BackendOption>(nullptr, 0)`.
> 2. Scan entries `i` from 0 to `size_-1`; compare `entries_[i].backend_id` to `backend_id` with `std::strcmp`. First exact match wins: return `Span<const BackendOption>(entries_[i].options.data(), entries_[i].options.size())` (a const view over that entry's option span).
> 3. If no entry matches, return an empty span `Span<const BackendOption>(nullptr, 0)`.
>
> Non-owning: the returned span points at externally-owned option storage that must outlive the map.

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn]
> bool has_options(const char* backend_id) const

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.has-options-fn]
> Const predicate: whether an entry for `backend_id` exists.
>
> Steps:
> 1. If `backend_id == nullptr`, return `false`.
> 2. Scan entries `i` from 0 to `size_-1`; if `std::strcmp(entries_[i].backend_id, backend_id) == 0` for any, return `true`.
> 3. Otherwise return `false`.
>
> Note this returns `true` even if the matched entry's option span is empty (presence is keyed on the backend_id, not on having options).

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.load-backend-options-map-fn]
> LoadBackendOptionsMap() : size_(0)

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.load-backend-options-map-fn]
> Default constructor. Creates an empty map: initializes `size_ = 0` and, for each of the `kMaxBackends` (8) fixed-storage entries, sets `entries_[i].backend_id[0] = '\0'` so every backend_id slot reads as an empty C-string. The `entries_[i].options` spans are left default-constructed (empty). No allocation occurs; all storage is inline/fixed-capacity.

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn]
> Error set_options(const char* backend_id, Span<BackendOption> options)

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-fn]
> Public entry point to associate `options` with `backend_id`. Returns `Error`.
>
> Steps:
> 1. Validate `backend_id`: if it is `nullptr` or an empty string (`backend_id[0] == '\0'`), return `Error::InvalidArgument` without modifying the map.
> 2. Otherwise delegate to `set_options_impl(backend_id, options)` per `[spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]` and return its result.
>
> The `options` span is stored by reference (non-owning); the caller must keep the underlying option storage alive for the map's lifetime. (A separate templated overload `set_options(Builder&)` — not a spec rule here — forwards `builder.backend_id()` and `builder.view()` directly to `set_options_impl`, bypassing the null/empty check.)

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]
> Error set_options_impl(const char* backend_id, Span<BackendOption> options)

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.set-options-impl-fn]
> Private core that inserts or updates an entry. Assumes `backend_id` is already validated non-null/non-empty. Returns `Error`.
>
> Steps:
> 1. Update-in-place: scan existing entries `i` from 0 to `size_-1`; if `std::strcmp(entries_[i].backend_id, backend_id) == 0`, overwrite `entries_[i].options = options` and return `Error::Ok` (does not change `size_` or the stored id). First match wins.
> 2. Capacity check: if `size_ >= kMaxBackends` (8), return `Error::InvalidArgument` (map full, cannot add new backend).
> 3. Length check: compute `id_len = std::strlen(backend_id)`; if `id_len >= kMaxBackendIdLength` (64), return `Error::InvalidArgument` (id too long to store with a null terminator in the 64-byte buffer).
> 4. Insert new entry at index `size_`: `std::memcpy` the `id_len` bytes of `backend_id` into `entries_[size_].backend_id`, write the null terminator at `entries_[size_].backend_id[id_len]`, set `entries_[size_].options = options`, then increment `size_`. Return `Error::Ok`.
>
> Insertion appends at the end, preserving prior order; updates never move entries.

> [spec:et:def:backend-options-map.executorch.runtime.load-backend-options-map.size-fn]
> size_t size() const

> [spec:et:sem:backend-options-map.executorch.runtime.load-backend-options-map.size-fn]
> Const accessor. Returns `size_`, the number of backends currently registered (entries with configured options). Ranges from 0 to `kMaxBackends` (8). No side effects.

