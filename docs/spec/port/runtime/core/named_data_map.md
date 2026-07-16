# runtime/core/named_data_map.h

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map]
> class NamedDataMap {
>   ET_NODISCARD virtual Result<const TensorLayout>;
>   ET_NODISCARD virtual Result<FreeableBuffer>;
>   ET_NODISCARD virtual Result<uint32_t>;
>   ET_NODISCARD virtual Result<const char*>;
> }

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn]
> get_data( std::string_view key) const = 0

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn]
> Pure-virtual interface method (`= 0`); this rule specifies the contract every
> implementation must honor, not a concrete body.
>
> Given a `key` (`std::string_view`), returns a `Result<FreeableBuffer>` that,
> on success, owns/references the raw bytes stored under that key. On success
> the returned `FreeableBuffer` provides access to the data pointer and its
> size; the caller may free it via the buffer's own API. On failure (key not
> present, backing storage unavailable, etc.) returns a non-Ok Error in the
> Result (implementation-defined which Error, e.g. Error::NotFound). The
> returned data corresponds to the same key used by
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]`.
> Marked ET_NODISCARD, so callers must inspect the Result.

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn]
> get_key(uint32_t index) const = 0

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn]
> Pure-virtual interface method (`= 0`); this rule specifies the contract, not
> a concrete body.
>
> Returns a `Result<const char*>` holding the key name at the given `index`
> (0-based), where `index` must be in `[0, get_num_keys())` (see
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn]`).
> On success the returned pointer is a NUL-terminated C string that remains
> valid only for the lifetime of the NamedDataMap (the caller must not free it
> or use it after the map is destroyed). On an out-of-range `index` or other
> error, returns a non-Ok Error in the Result. The iteration order of keys is
> implementation-defined but stable for a given map instance, and the key
> returned here is a valid argument to the by-key lookups
> (`get_data`/`get_tensor_layout`/`load_data_into`). ET_NODISCARD.

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn]
> get_num_keys() const = 0

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-num-keys-fn]
> Pure-virtual interface method (`= 0`); this rule specifies the contract, not
> a concrete body.
>
> Returns a `Result<uint32_t>` giving the number of distinct keys held by the
> map. On success this is the count `N` such that valid indices for
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-key-fn]`
> are `[0, N)`. May be 0 for an empty map. Returns a non-Ok Error in the
> Result only if the count cannot be determined. ET_NODISCARD.

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]
> get_tensor_layout( std::string_view key) const = 0

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]
> Pure-virtual interface method (`= 0`); this rule specifies the contract, not
> a concrete body.
>
> Given a `key` (`std::string_view`), returns a `Result<const TensorLayout>`
> describing the shape/dtype/dim-order metadata of the tensor stored under that
> key, without loading the tensor bytes. On success the caller can derive the
> byte size of the data (used to size the buffer passed to
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn]`).
> On failure (key not present, no layout metadata, etc.) returns a non-Ok
> Error in the Result. ET_NODISCARD.

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn]
> ET_NODISCARD virtual Error

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.load-data-into-fn]
> Pure-virtual interface method (`= 0`); this rule specifies the contract, not
> a concrete body.
>
> Copies the data stored under `key` into the caller-provided `buffer`,
> reading exactly `size` bytes. `buffer` must point to at least `size` bytes of
> writable memory; the caller obtains the correct `size` from the layout via
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-tensor-layout-fn]`.
> Returns Error::Ok on success, or a non-Ok Error (e.g. key not found, size
> mismatch, or I/O failure) on which the buffer contents are unspecified.
> Unlike
> `[spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.get-data-fn]`,
> this performs an eager copy into caller-owned memory rather than returning a
> FreeableBuffer. Returns `Error` directly (not wrapped in Result).
> ET_NODISCARD.

> [spec:et:def:named-data-map.executorch.et-runtime-namespace.named-data-map.named-data-map-fn]
> virtual ~NamedDataMap() = default

> [spec:et:sem:named-data-map.executorch.et-runtime-namespace.named-data-map.named-data-map-fn]
> Virtual destructor of the abstract `NamedDataMap` interface, declared
> `= default`. It performs no work of its own; being virtual, it guarantees
> that destroying a derived NamedDataMap through a `NamedDataMap*` base pointer
> invokes the derived class's destructor (and thus releases any resources the
> concrete implementation holds). In a Rust port this corresponds to the
> interface trait being object-safe with normal `Drop` running on the concrete
> type behind a boxed/dyn reference; no explicit action is required here.

