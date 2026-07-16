# runtime/executor/merged_data_map.h

> [spec:et:def:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn]
> class MergedDataMap final : public NamedDataMap

> [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.merged-data-map-fn]
> `MergedDataMap` is a `NamedDataMap` implementation that presents two
> underlying `NamedDataMap`s (`first_`, `second_`) as one merged, read-only
> namespace. It stores two borrowed non-owning pointers; both maps must outlive
> the MergedDataMap. It is move-constructible (defaulted) but neither copyable
> nor assignable (copy ctor, copy-assign, and move-assign are all deleted).
>
> Construction is gated by the static factory `load(first, second) ->
> Result<MergedDataMap>`, which is the only public path (the two-pointer
> constructor is private):
> 1. Validate both inputs are non-null: `ET_CHECK_OR_RETURN_ERROR(first !=
>    nullptr && second != nullptr, InvalidArgument, "Input data map is null.")`.
>    On failure, return `Error::InvalidArgument` (no instance is created).
> 2. Reject overlapping key namespaces: iterate `k` over
>    `[0, first->get_num_keys().get())`; for each, fetch `key =
>    first->get_key(k).get()` and probe the second map via
>    `second->get_tensor_layout(key).error()`. The key is considered absent from
>    `second` only if that error is `Error::NotFound` or `Error::NotImplemented`;
>    any other error value (including `Error::Ok`, meaning the key resolved in
>    both maps) triggers `ET_CHECK_OR_RETURN_ERROR(..., InvalidArgument,
>    "Duplicate key %s.", key)` and returns `Error::InvalidArgument`. Note the
>    scan calls into `first`/`second` accessors with `.get()` on Results, so a
>    failing accessor there is a fatal (unchecked) access, matching the C++.
> 3. On success construct and return `MergedDataMap(first, second)`, storing
>    `first_ = first`, `second_ = second`.
>
> As a NamedDataMap, the merged view resolves each operation first-then-second:
> - `get_tensor_layout(key)`: return `first_`'s layout if it is `ok()`; if
>   `first_` returns a non-`NotFound` error, propagate that error; otherwise
>   (NotFound in first) return `second_->get_tensor_layout(key)` verbatim.
> - `get_data(key)`: return `first_`'s result unless its error is exactly
>   `Error::NotFound`, in which case return `second_->get_data(key)`. (Any other
>   error from `first_`, and any `ok` result, is returned as-is.)
> - `load_data_into(key, buffer, size)`: unsupported; always returns
>   `Error::NotImplemented` (args unused).
> - `get_num_keys()`: returns `first_->get_num_keys().get() +
>   second_->get_num_keys().get()` (simple sum, no dedup — duplicates were
>   already rejected at load time).
> - `get_key(index)`: bounds-check `index < total` where `total =
>   get_num_keys().get()`, else return `Error::InvalidArgument` with message
>   "Index N out of range of size M". For in-range indices, if `index <
>   first_->get_num_keys().get()` delegate to `first_->get_key(index)`, otherwise
>   delegate to `second_->get_key(index - first_->get_num_keys().get())`. This
>   orders all of `first_`'s keys ahead of `second_`'s.

> [spec:et:def:merged-data-map.executorch.et-runtime-namespace.internal.operator-fn]
> MergedDataMap& operator=(MergedDataMap&& rhs) noexcept = delete

> [spec:et:sem:merged-data-map.executorch.et-runtime-namespace.internal.operator-fn]
> Move-assignment operator `operator=(MergedDataMap&&)` is explicitly `= delete`d
> (as are copy-construct and copy-assign). A MergedDataMap therefore cannot be
> reassigned; it can only be created via `load(...)` and moved (move-construction
> is defaulted). There is no runtime behavior to implement: any attempt to
> move-assign is a compile-time error in C++. In Rust, model this by not
> providing an assignment path that rebinds the two borrowed maps in place — the
> type is constructed once and thereafter only moved, never overwritten.

