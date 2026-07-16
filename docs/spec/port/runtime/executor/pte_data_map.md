# runtime/executor/pte_data_map.cpp, runtime/executor/pte_data_map.h

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map]
> class PteDataMap final : public NamedDataMap {
>   ET_NODISCARD Result<const TensorLayout>;
>   ET_NODISCARD Result<FreeableBuffer>;
>   ET_NODISCARD Result<uint32_t>;
>   ET_NODISCARD Result<const char*>;
>   DataLoader* loader_;
>   size_t segment_base_offset_;
>   const flatbuffers::FlatbufferNamedData* named_data_;
>   const flatbuffers::FlatbufferDataSegment* segments_;
> }

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]
> Result<PteDataMap> PteDataMap::create( DataLoader* loader, size_t segment_base_offset, const flatbuffers::FlatbufferNamedData* named_data, const flatbuffers::FlatbufferDataSegment* segments)

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]
> Static factory. Validates the three flatbuffer/loader inputs are present and
> constructs a `PteDataMap`. Called from
> `[spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn]` when
> the program has a `named_data` table.
>
> Steps:
> 1. ET_CHECK_OR_RETURN_ERROR: if NOT (`loader != nullptr && named_data !=
>    nullptr && segments != nullptr`), log and return `Error::InvalidArgument`
>    (message notes the program most likely has no named_data segments).
> 2. Otherwise construct via the private constructor
>    (`[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn]`)
>    with `(loader, segment_base_offset, named_data, segments)` and return it
>    wrapped in `Result`.
>
> Does not copy or load any data; only stores the pointers/offset for later
> lookups. `segment_base_offset` is passed through unvalidated.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn]
> get_data(std::string_view key) const override

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn]
> `NamedDataMap::get_data` override declaration. Retrieves read-only data for a
> named-data key by looking it up in the PTE's `named_data` table and loading the
> referenced segment. The behavior is defined by the out-of-class definition
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]`.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-key-fn]
> get_key(uint32_t index) const override

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-key-fn]
> `NamedDataMap::get_key(uint32_t index)` override declaration. Returns the name
> of the named-data entry at `index`. Behavior is defined by the out-of-class
> definition
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn]`.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-num-keys-fn]
> get_num_keys() const override

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-num-keys-fn]
> `NamedDataMap::get_num_keys()` override declaration. Returns the number of
> named-data keys. Behavior is defined by the out-of-class definition
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn]`.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-tensor-layout-fn]
> get_tensor_layout( ET_UNUSED std::string_view key) const override

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-tensor-layout-fn]
> `NamedDataMap::get_tensor_layout(std::string_view key)` override, defined
> inline in the header. Unconditionally returns `Error::NotImplemented`; the
> `key` argument is ignored (marked ET_UNUSED). PteDataMap only handles opaque
> blobs and carries no tensor-specific layout metadata, so no key lookup or
> validation is performed.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.load-data-into-fn]
> ET_NODISCARD Error load_data_into( ET_UNUSED std::string_view key, ET_UNUSED void* buffer, ET_UNUSED size_t size) const override

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.load-data-into-fn]
> `NamedDataMap::load_data_into(std::string_view key, void* buffer, size_t size)`
> override, defined inline in the header. Unconditionally returns
> `Error::NotImplemented`; all three arguments (`key`, `buffer`, `size`) are
> ignored (marked ET_UNUSED). PteDataMap does not support loading into a
> caller-provided buffer; callers must use
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]`
> instead.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn]
> PteDataMap( DataLoader* loader, size_t segment_base_offset, const flatbuffers::FlatbufferNamedData* named_data, const flatbuffers::FlatbufferDataSegment* segments) : loader_(loader), segment_base_offset_(segment_base_offset), named_data_...

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn]
> Private constructor. Stores the four inputs directly with no validation or
> data access (validation is done by
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]`
> before this is reached).
>
> Member initialization:
> 1. `loader_` = `loader` (borrowed; must outlive the map).
> 2. `segment_base_offset_` = `segment_base_offset`.
> 3. `named_data_` = `named_data` (borrowed flatbuffer vector pointer).
> 4. `segments_` = `segments` (borrowed flatbuffer vector pointer).
>
> The map owns none of these; all point into loader-managed / Program-owned
> memory that must outlive the PteDataMap.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn]
> ET_NODISCARD Result<const char*> PteDataMap::get_key(uint32_t index) const

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn]
> Out-of-class definition of `get_key(uint32_t index)`. Returns the NUL-
> terminated key string of the `named_data_` entry at `index`.
>
> Steps:
> 1. ET_CHECK_OR_RETURN_ERROR: if NOT (`index < named_data_->size()`), log and
>    return `Error::InvalidArgument`.
> 2. `item = named_data_->Get(index)`. ET_CHECK_OR_RETURN_ERROR: if NOT (`item !=
>    nullptr && item->key() != nullptr`), log and return `Error::InvalidArgument`.
> 3. Return `item->key()->c_str()` — a NUL-terminated pointer into the flatbuffer
>    buffer, valid for the lifetime of the underlying Program/data (the map does
>    not own it).

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]
> ET_NODISCARD

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]
> Out-of-class definition of the `get_data(std::string_view key)` override.
> Linear-searches `named_data_` for an entry whose key equals `key`, then loads
> that entry's segment via the loader and returns it as a `FreeableBuffer`.
>
> Steps:
> 1. Iterate `i` from 0 to `named_data_->size() - 1` in order:
>    a. `named_data_item = named_data_->Get(i)`. ET_CHECK_OR_RETURN_ERROR: if
>       NOT (`named_data_item != nullptr && named_data_item->key() != nullptr`),
>       log and return `Error::InvalidArgument`.
>    b. `named_data_key = named_data_item->key()`. Compare by exact byte length
>       then bytes: match when `named_data_key->size() == key.size()` AND
>       `memcmp(named_data_key->data(), key.data(), key.size()) == 0`. This is a
>       length-first byte comparison (not NUL-terminated), so embedded NULs are
>       significant and no C-string semantics apply.
>    c. On a match:
>       - `segment_index = named_data_item->segment_index()` (`size_t`).
>       - ET_CHECK_OR_RETURN_ERROR: if NOT (`segment_index < segments_->size()`),
>         log and return `Error::InvalidArgument`.
>       - `segment_offset = segments_->Get(segment_index)->offset()`;
>         `segment_size = segments_->Get(segment_index)->size()`.
>       - Return `loader_->load(segment_base_offset_ + segment_offset,
>         segment_size, SegmentInfo(Type::Constant))` — the loaded buffer, or the
>         loader's error. The absolute file offset is `segment_base_offset_ +
>         segment_offset` (no explicit overflow guard here, unlike
>         `[spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn]`).
> 2. If the scan completes with no matching key, return `Error::NotFound`.
>
> The first matching key wins. Returns a fresh `FreeableBuffer` owning the loaded
> segment bytes; the caller takes ownership.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn]
> ET_NODISCARD Result<uint32_t> PteDataMap::get_num_keys() const

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn]
> Out-of-class definition of `get_num_keys()`. Returns `named_data_->size()` (the
> flatbuffer vector length, a `uint32_t`) wrapped in `Result`. Always succeeds;
> no validation or error path. `named_data_` is guaranteed non-null because
> `[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]`
> rejects a null `named_data` at construction.

> [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.operator-fn]
> PteDataMap& operator=(PteDataMap&& rhs) noexcept = delete

> [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.operator-fn]
> Deleted move-assignment operator (`= delete`). `PteDataMap` is move-
> constructible (to be compatible with `Result<PteDataMap>`) but neither move-
> assignable nor copy-assignable nor copy-constructible. Any attempt to move-
> assign is a compile-time error; there is no runtime behavior to reimplement. In
> a Rust port, model as a type that is movable but exposes no assignment-through-
> reference operation.

