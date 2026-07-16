# runtime/core/exec_aten/util/dim_order_util.h

> [spec:et:def:dim-order-util.executorch.runtime.dim-order-to-stride-fn]
> ET_NODISCARD inline Error dim_order_to_stride( const SizesType* sizes, const DimOrderType* dim_order, const size_t dims, StridesType* strides)

> [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn]
> Template function (over `SizesType`, `DimOrderType`, `StridesType`) that
> converts a size array plus a dim-order permutation into a stride array,
> with validation. `ET_NODISCARD`: the returned `Error` must be consumed.
> Arguments: `sizes` (length `dims`), `dim_order` (length `dims`), `dims`
> (`size_t`), and out-param `strides` (length `dims`, filled in).
>
> Steps:
> - If `dims == 0` (scalar tensor), return `Error::Ok` immediately without
>   touching any array.
> - Validate the dim order via
>   `[spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn]`
>   (`ET_CHECK_OR_RETURN_ERROR`): if `dim_order[0..dims)` is not a
>   permutation of `[0, dims)`, log "Invalid dim order: values must be a
>   permutation of [0, %zu)" and return `Error::InvalidArgument`.
> - Otherwise delegate to
>   `[spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn]`
>   with the same `sizes`, `dim_order`, `dims`, `strides`, then return
>   `Error::Ok`.
>
> On the error path `strides` is left unmodified. This is the safe wrapper
> around the nocheck variant.

> [spec:et:def:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn]
> inline void dim_order_to_stride_nocheck( const SizesType* sizes, const DimOrderType* dim_order, const size_t dims, StridesType* strides)

> [spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-nocheck-fn]
> Template function (over `SizesType`, `DimOrderType`, `StridesType`) that
> converts sizes plus a dim-order permutation into strides WITHOUT validating
> the dim order (the caller must have already validated it). Arguments:
> `sizes` (length `dims`), `dim_order` (length `dims`), `dims` (`size_t`),
> and out-param `strides` (length `dims`).
>
> `dim_order` lists dimension indices from slowest-varying (outermost,
> largest stride) to fastest-varying (innermost, stride 1). The written
> strides are indexed by original dimension: `strides[dim_order[k]]` is the
> stride of the k-th entry in the memory-layout order. Example: sizes
> `[2,3,4,5]`, `dim_order [0,2,3,1]` yields `strides [60,1,15,3]`.
>
> Steps:
> - If `dims == 0`, return immediately (nothing written).
> - Set the fastest-moving dimension's stride to 1:
>   `strides[dim_order[dims-1]] = 1`.
> - Iterate `i` from `dims-2` down to `0` (descending, using a signed
>   `int32_t` counter). For each `i`, let `nxt = dim_order[i+1]` and `cur =
>   dim_order[i]`:
>   - If `sizes[nxt] == 0`, set `strides[cur] = strides[nxt]` (a zero-size
>     dimension does not scale the running stride; the stride is carried
>     through unchanged).
>   - Otherwise set `strides[cur] = strides[nxt] * sizes[nxt]`.
>
> Products accumulate in `StridesType` arithmetic; overflow is not checked
> here. The zero-size special case ensures strides remain well-defined for
> empty tensors.

> [spec:et:def:dim-order-util.executorch.runtime.internal.sorter]
> struct Sorter

> [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.partition-fn]
> int32_t

> [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.partition-fn]
> Private helper of the `Sorter<ValueType>` struct implementing the
> partition step of quicksort. Arguments: `arr` (array being sorted), `low`
> and `high` (`int32_t` inclusive bounds of the current subrange), and
> `pivot` (a `ValueType` value, chosen by the caller as `arr[high]`).
>
> Uses a Lomuto-style two-pointer scan with the struct's `operator>` as the
> comparison (which for the `StrideDimOrder` value type orders by descending
> stride; see
> `[spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]`):
> - Initialize `i = low` and `j = low`.
> - While `i <= high`:
>   - If `arr[i] > pivot` (element sorts before the pivot under `operator>`),
>     just advance `i` (`i++`), leaving it in the "greater" partition.
>   - Else, swap `arr[i]` with `arr[j]` via
>     `[spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.swap-fn]`,
>     then advance both `i` and `j` (`swap(arr, i++, j++)`).
> - Return `j - 1`, the final index of the pivot after partitioning.
>
> After this runs, all elements that compare `> pivot` precede index `j-1`
> and the rest (including the pivot) follow, giving a descending-by-`operator>`
> arrangement.

> [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn]
> void quick_sort(ValueType arr[], int32_t low, int32_t high)

> [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn]
> Public method of the `Sorter<ValueType>` struct implementing recursive
> quicksort over `arr[low..high]` (inclusive bounds, `int32_t`). Sorts in
> place according to the value type's `operator>` (for `StrideDimOrder`, that
> means descending stride; see
> `[spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]`).
>
> Steps:
> - If `low < high`:
>   - Choose the pivot as `arr[high]`.
>   - Call
>     `[spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.partition-fn]`
>     with `(arr, low, high, pivot)` to partition the subrange, obtaining the
>     pivot's final position `pos`.
>   - Recurse on the left subrange `quick_sort(arr, low, pos - 1)` and the
>     right subrange `quick_sort(arr, pos + 1, high)`.
> - If `low >= high` (empty or single-element subrange), do nothing.
>
> This algorithm is not stable, but that does not matter for the caller
> because stride keys used by
> `[spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn]`
> determine ordering. The port may use any equivalent in-place descending
> sort by stride.

> [spec:et:def:dim-order-util.executorch.runtime.internal.sorter.swap-fn]
> void swap(ValueType arr[], int32_t pos1, int32_t pos2) noexcept

> [spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.swap-fn]
> Private `noexcept` helper of the `Sorter<ValueType>` struct that exchanges
> two array elements in place. Arguments: `arr`, and `pos1`, `pos2`
> (`int32_t` indices). Copies `arr[pos1]` into a temporary, assigns
> `arr[pos1] = arr[pos2]`, then `arr[pos2] = temp`. No bounds checking; if
> `pos1 == pos2` the element is unchanged. Returns nothing.

> [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order]
> struct StrideDimOrder {
>   StridesType stride;
>   DimOrderType dim_order;
> }

> [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]
> bool operator>(const StrideDimOrder& other) const

> [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]
> `operator>` on `StrideDimOrder`, used as the sort comparator. Despite its
> name, it is intentionally inverted to produce a descending-by-stride sort:
> `a > b` is defined as `a.stride < b.stride`. That is, it returns `true`
> when this object's `stride` is strictly less than `other.stride`. Only the
> `stride` field participates; the `dim_order` field is ignored by the
> comparison. This makes the quicksort in
> `[spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn]`
> arrange entries from largest stride to smallest.

> [spec:et:def:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn]
> StrideDimOrder(StridesType stride_, DimOrderType dim_order_)

> [spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn]
> Two-argument constructor of `StrideDimOrder<StridesType, DimOrderType>`. It
> stores its arguments directly into the struct's fields: member `stride` is
> initialized from `stride_` and member `dim_order` from `dim_order_`. No
> validation or transformation. (A defaulted zero-argument constructor also
> exists per the `def` shape.)

> [spec:et:def:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]
> bool is_channels_last_dim_order( const DimOrderType* dim_order, const size_t dims)

> [spec:et:sem:dim-order-util.executorch.runtime.is-channels-last-dim-order-fn]
> Template function (over `DimOrderType`) returning `true` iff the given
> `dim_order` array of length `dims` describes a channels-last memory layout.
> Channels-last is only defined for 4-D (NCHW logical order) and 5-D (NCHWD)
> tensors. Arguments: `dim_order` (pointer, length `dims`) and `dims`
> (`size_t`).
>
> Steps:
> - If `dims != 4` and `dims != 5`, return `false`.
> - The channels dimension is index `1`. Require the last entry to be the
>   channels dim: if `dim_order[dims-1] != 1`, return `false`.
> - Require the first entry to be the batch dim: if `dim_order[0] != 0`,
>   return `false`.
> - For `d` from `1` up to but not including `dims-1` (i.e. the interior
>   spatial dims), require `dim_order[d] == d + 1`; if any mismatches, return
>   `false`.
> - Otherwise return `true`.
>
> Concretely this accepts exactly `[0, 2, 3, 1]` for `dims == 4` and
> `[0, 2, 3, 4, 1]` for `dims == 5`.

> [spec:et:def:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]
> inline bool is_contiguous_dim_order( const DimOrderType* dim_order, const size_t dims)

> [spec:et:sem:dim-order-util.executorch.runtime.is-contiguous-dim-order-fn]
> Template function (over `DimOrderType`) returning `true` iff the given
> `dim_order` equals the contiguous (row-major) identity order
> `{0, 1, 2, ..., dims-1}`. Arguments: `dim_order` (pointer, length `dims`)
> and `dims` (`size_t`).
>
> Iterate `i` from `0` to `dims-1` in ascending order; if any
> `dim_order[i] != i` (comparing after casting `i` to `DimOrderType`), return
> `false` immediately. If all match, return `true`. For `dims == 0` the loop
> does not run and it returns `true` (a scalar tensor is contiguous).

> [spec:et:def:dim-order-util.executorch.runtime.stride-to-dim-order-fn]
> ET_NODISCARD inline Error stride_to_dim_order( const StridesType* strides, const size_t dims, DimOrderType* dim_order)

> [spec:et:sem:dim-order-util.executorch.runtime.stride-to-dim-order-fn]
> Template function (over `DimOrderType`, `StridesType`) that recovers a
> dim-order permutation from a stride array by sorting dimension indices by
> descending stride. `ET_NODISCARD`: the returned `Error` must be consumed.
> Arguments: `strides` (length `dims`), `dims` (`size_t`), out-param
> `dim_order` (length `dims`). Example: sizes `[3,5,2]`, strides `[5,1,15]`
> yield `dim_order [2,0,1]`.
>
> Steps:
> - Let `kMaxNumOfDimensions = 16`.
> - `ET_CHECK_OR_RETURN_ERROR`: if `dim_order == nullptr`, return
>   `Error::MemoryAllocationFailed` ("Need memory to get dim_order.").
> - `ET_CHECK_OR_RETURN_ERROR`: if `dims > 16`, return `Error::NotSupported`
>   ("dims %zu exceeds maximum allowed %zu").
> - Build a fixed-size stack array of 16 `StrideDimOrder` records; for `i` in
>   `[0, dims)` set `array[i].dim_order = i` and `array[i].stride =
>   strides[i]` (constructing each per
>   `[spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.stride-dim-order-fn]`).
> - Sort `array[0..dims)` in place by descending stride using
>   `[spec:et:sem:dim-order-util.executorch.runtime.internal.sorter.quick-sort-fn]`
>   (comparator per
>   `[spec:et:sem:dim-order-util.executorch.runtime.internal.stride-dim-order.operator-fn]`),
>   called as `quick_sort(array, 0, dims - 1)`.
> - Write results out: for `i` in `[0, dims)` set `dim_order[i] =
>   array[i].dim_order`.
> - Return `Error::Ok`.
>
> On either check failure `dim_order` is left unmodified. When multiple
> dimensions share the same stride, their relative order in the output
> follows the (non-stable) quicksort and is unspecified beyond being a valid
> permutation. For `dims == 0` the loops do not run and it returns
> `Error::Ok` with `dim_order` untouched (given a non-null pointer).

> [spec:et:def:dim-order-util.executorch.runtime.validate-dim-order-fn]
> bool validate_dim_order(const DimOrderType* dim_order, const size_t dims)

> [spec:et:sem:dim-order-util.executorch.runtime.validate-dim-order-fn]
> Template function (over `DimOrderType`) in an anonymous namespace returning
> `true` iff `dim_order[0..dims)` is a valid permutation of `[0, dims)`.
> Arguments: `dim_order` (pointer, length `dims`) and `dims` (`size_t`). Uses
> a 16-bit "seen" bitmask, which assumes `kTensorDimensionLimit <= 16` (a
> compile-time `static_assert` enforces this).
>
> Steps:
> - If `dims > kTensorDimensionLimit`, return `false`.
> - Initialize `seen = 0` (`uint16_t`).
> - For each `i` in `[0, dims)` ascending:
>   - If `dim_order[i] >= dims` (out of range for a permutation of `[0,
>     dims)`), return `false`.
>   - Compute `mask = 1u << dim_order[i]`. If `seen & mask` is nonzero (this
>     value already appeared, i.e. a duplicate), return `false`.
>   - Set `seen |= mask`.
> - If the loop completes, return `true`.
>
> Thus it rejects any value `>= dims` and any repeated value; it accepts any
> ordering that is a bijection onto `[0, dims)`. For `dims == 0` it returns
> `true` (vacuously valid). This is the validity check consumed by
> `[spec:et:sem:dim-order-util.executorch.runtime.dim-order-to-stride-fn]`.

