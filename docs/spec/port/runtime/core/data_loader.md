# runtime/core/data_loader.h

> [spec:et:def:data-loader.executorch.runtime.data-loader]
> class DataLoader {
>   struct SegmentInfo { /** * Represents the purpose of the segment. */ enum class Type { /** * Data for the actual program. */ Program, /** * Holds constant te...;
>   ET_NODISCARD virtual Result<FreeableBuffer>;
>   ET_NODISCARD virtual Result<size_t>;
> }

> [spec:et:def:data-loader.executorch.runtime.data-loader.data-loader-fn]
> virtual ~DataLoader() = default

> [spec:et:sem:data-loader.executorch.runtime.data-loader.data-loader-fn]
> Virtual destructor of the abstract `DataLoader` base class, defaulted
> (`= default`). It performs no work of its own; its sole purpose is to be
> `virtual` so that deleting a derived `DataLoader` through a base-class
> pointer runs the concrete implementation's destructor. `DataLoader` holds
> no data members, so there is nothing to release here. In a Rust port this
> corresponds to the base trait carrying no drop logic; ownership/cleanup is
> entirely the responsibility of the concrete implementor.

> [spec:et:def:data-loader.executorch.runtime.data-loader.load-fn]
> load(size_t offset, size_t size, const SegmentInfo& segment_info) const = 0

> [spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn]
> Pure virtual (`= 0`), `const`, `ET_NODISCARD` method; the base class
> provides no body — every concrete `DataLoader` must implement it. Contract:
> load `size` bytes starting at byte `offset` in the underlying data source,
> allocating and returning a `FreeableBuffer` that owns the loaded bytes.
>
> Parameters:
> - `offset`: byte offset in the data source at which to start reading.
> - `size`: number of bytes to read.
> - `segment_info`: a `SegmentInfo` describing the purpose of this segment
>   (see `[spec:et:sem:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn]`);
>   implementations may use it to choose allocation strategy (e.g. which
>   `DeviceAllocator` or memory region to use), but the returned data must be
>   the requested byte range regardless.
>
> Return value: `Result<FreeableBuffer>`. On success, holds a `FreeableBuffer`
> whose `data()` points to at least `size` bytes and whose `size()` equals
> `size`; the buffer owns the memory and frees it on `Free()`/destruction. On
> failure, holds an `Error` code (e.g. `Error::AccessFailed`,
> `Error::InvalidArgument`, `Error::MemoryAllocationFailed`) chosen by the
> implementation. A `size` of 0 is permitted and yields an empty
> `FreeableBuffer`.
>
> Threading: the contract requires this call to be thread-safe. If the
> implementation mutates shared state, it must perform its own locking; the
> interface provides no synchronization.

> [spec:et:def:data-loader.executorch.runtime.data-loader.load-into-fn]
> ET_NODISCARD virtual Error load_into( size_t offset, size_t size, const SegmentInfo& segment_info, void* buffer) const

> [spec:et:sem:data-loader.executorch.runtime.data-loader.load-into-fn]
> Non-pure virtual, `const`, `ET_NODISCARD` method with a default base-class
> body (a stub, deliberately not pure-virtual so the interface can be extended
> in a backwards-compatible way — existing loaders need not override it).
> Contract when overridden: read `size` bytes starting at byte `offset` from
> the data source and copy them into the caller-supplied `buffer`, which must
> point to at least `size` writable bytes. Unlike
> `[spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn]`, this
> writes into memory the caller owns rather than allocating a `FreeableBuffer`.
>
> Parameters: `offset`, `size`, and `segment_info` have the same meaning as in
> `load`; `buffer` is the destination pointer.
>
> Default base-class behavior (used when a subclass does not override it):
> - Ignores all four parameters (they are explicitly cast to void).
> - Logs at Error level: "load_into() not implemented for this data loader."
> - Returns `Error::NotImplemented`.
>
> Overriding implementations return `Error::Ok` on success, or an appropriate
> `Error` on failure. This call must be thread-safe; implementations that
> mutate shared state must lock internally.

> [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info]
> struct SegmentInfo {
>   enum class Type { /** * Data for the actual program. */ Program, /** * Holds constant tensor data. */ Constant, /** * Data used for initializing a backend. *...;
>   Type segment_type;
>   size_t segment_index;
>   const char* descriptor;
> }

> [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn]
> explicit SegmentInfo( Type segment_type_, size_t segment_index_ = 0, const char* descriptor_ = nullptr) : segment_type(segment_type_), segment_index(segment_index_), descriptor(descriptor_)

> [spec:et:sem:data-loader.executorch.runtime.data-loader.segment-info.segment-info-fn]
> `explicit` constructor of the nested `SegmentInfo` struct. It simply stores
> its arguments into the corresponding fields; it performs no validation, no
> allocation, and no copying of the descriptor string (the pointer is stored
> as-is, so the caller retains ownership and must keep the pointed-to string
> alive for as long as the `SegmentInfo` is used).
>
> Parameters and field assignments:
> - `segment_type_` (required, of type `SegmentInfo::Type`) -> `segment_type`.
> - `segment_index_` (defaults to 0) -> `segment_index`. This is the index of
>   the segment within the segment list; it is undefined/unused for `Program`
>   segments.
> - `descriptor_` (defaults to `nullptr`) -> `descriptor`. An optional,
>   null-terminated string describing the segment. For `Backend` segments this
>   is the backend ID; it is null for all other segment types.
>
> There is also a separate defaulted default constructor (`SegmentInfo() =
> default`) that leaves all fields uninitialized; only the explicit
> three-argument form documented here initializes the fields.

> [spec:et:def:data-loader.executorch.runtime.data-loader.segment-info.type]
> enum class Type {
>   Program;
>   Constant;
>   Backend;
>   Mutable;
>   External;
> }

> [spec:et:def:data-loader.executorch.runtime.data-loader.size-fn]
> size() const = 0

> [spec:et:sem:data-loader.executorch.runtime.data-loader.size-fn]
> Pure virtual (`= 0`), `const`, `ET_NODISCARD` method; no base-class body.
> Contract: return the total length in bytes of the underlying data source
> (typically the backing file size). This is the upper bound for valid
> `offset + size` ranges passed to
> `[spec:et:sem:data-loader.executorch.runtime.data-loader.load-fn]`.
>
> Return value: `Result<size_t>`. On success, holds the source length in
> bytes. On failure (e.g. the size cannot be determined), holds an `Error`
> code chosen by the implementation. Takes no arguments and does not mutate
> the loader.

