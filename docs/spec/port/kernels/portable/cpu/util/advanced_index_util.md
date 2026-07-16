# kernels/portable/cpu/util/advanced_index_util.cpp

> [spec:et:def:advanced-index-util.torch.executor.check-index-args-fn]
> bool check_index_args(const Tensor& in, TensorOptList indices, Tensor& out)

> [spec:et:sem:advanced-index-util.torch.executor.check-index-args-fn]
> Performs preliminary validation of the arguments to an advanced-indexing
> op (`in[indices...] -> out`). It does NOT validate that integer index values
> are within bounds for `in` (that happens later, in
> `[spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn]`). Steps,
> in order, short-circuiting to return false on the first failure:
> 1. ET_LOG_AND_RETURN_IF_FALSE(`tensors_have_same_dtype(in, out)`): `in` and
>    `out` must have identical scalar_type; otherwise logs and returns false.
> 2. ET_LOG_AND_RETURN_IF_FALSE(`check_indices_dtypes(indices)`) per
>    `[spec:et:sem:advanced-index-util.torch.executor.check-indices-dtypes-fn]`:
>    every non-null index tensor must be Long/Int/Byte/Bool.
> 3. ET_CHECK_OR_RETURN_FALSE(`(ssize_t)indices.size() <= in.dim()`): the number
>    of index slots may not exceed the input's dimensionality ("Indexing too
>    many dimensions"); returns false otherwise.
> 4. ET_LOG_AND_RETURN_IF_FALSE(`check_mask_indices(in, indices)`) per
>    `[spec:et:sem:advanced-index-util.torch.executor.check-mask-indices-fn]`.
> 5. If all checks pass, return true.

> [spec:et:def:advanced-index-util.torch.executor.check-indices-dtypes-fn]
> bool check_indices_dtypes(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.check-indices-dtypes-fn]
> Validates the dtype of every non-null index tensor. Iterate `i` over
> `indices` in order (`0 .. indices.size()-1`); for each slot that
> `has_value()`, take the contained tensor and read its `scalar_type()`.
> ET_CHECK_OR_RETURN_FALSE that the type is one of Long (int64), Int (int32),
> Byte (uint8), or Bool; if a non-null index has any other dtype, log
> "Index tensors should be Long, Int, Byte or Bool; got <n>" and return false.
> Null (nullopt) slots are skipped. Return true if all non-null indices pass.

> [spec:et:def:advanced-index-util.torch.executor.check-mask-indices-fn]
> bool check_mask_indices(const Tensor& in, TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.check-mask-indices-fn]
> Validates that every boolean/byte mask index's shape aligns with the input
> dimensions it consumes. Walk a cursor `in_i = 0` over the input dimensions
> while iterating slots `i` over `indices` in order:
> - If slot `i` is null: advance `in_i += 1` (a null index consumes exactly one
>   input dim).
> - If slot `i` is a non-mask (integer) index (see
>   `[spec:et:sem:advanced-index-util.torch.executor.is-mask-index-fn]`):
>   advance `in_i += 1`.
> - If slot `i` is a mask index (Bool or Byte):
>   - ET_CHECK_OR_RETURN_FALSE(`index.dim() > 0`) — a zero-dimensional mask is
>     not allowed; return false with "Zero-dimensional mask index not allowed".
>   - For each `j` in `[0, index.dim())`: require `index.size(j) == in.size(in_i + j)`.
>     If any dimension mismatches, (optionally log the mask vs. input sub-shape)
>     and return false.
>   - Advance `in_i += index.dim()` (a mask consumes as many input dims as it
>     has).
> Return true if all masks match. Note: `in_i` is not bounds-checked against
> `in.dim()` here; the caller is expected to have already limited the number of
> indexed dimensions.

> [spec:et:def:advanced-index-util.torch.executor.compute-dim-map-fn]
> void compute_dim_map( const Tensor& in, TensorOptList indices, int32_t* dim_map, bool adjacent)

> [spec:et:sem:advanced-index-util.torch.executor.compute-dim-map-fn]
> Fills `dim_map` (length `in.dim()`) mapping each NON-indexed input dimension
> to its corresponding output dimension, and setting each indexed input
> dimension to -1. First compute:
> - `broadcast_ndim = get_indices_broadcast_ndim(indices)` per
>   `[spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn]`.
> - `start = get_num_leading_null_indices(indices)` per
>   `[spec:et:sem:advanced-index-util.torch.executor.get-num-leading-null-indices-fn]`.
> - `num_indexed_dims = get_num_indexed_dims(indices)` per
>   `[spec:et:sem:advanced-index-util.torch.executor.get-num-indexed-dims-fn]`.
> - `num_null_indices = get_num_null_indices(indices)` per
>   `[spec:et:sem:advanced-index-util.torch.executor.get-num-null-indices-fn]`.
>
> `adjacent == true` (all non-null indices form a single contiguous block, so
> the broadcast dims sit where the leading nulls end):
> - For `i` in `[0, start)`: `dim_map[i] = i` (leading null dims map straight
>   through to the same output position).
> - For `i` in `[start, start + num_indexed_dims)`: `dim_map[i] = -1` (the
>   indexed dims).
> - For `i` in `[start + num_indexed_dims, in.dim())`:
>   `dim_map[i] = i - num_indexed_dims + broadcast_ndim` (trailing dims shift by
>   the difference between broadcast width and the collapsed indexed width).
>
> `adjacent == false` (indexed dims are non-contiguous, so all broadcast dims
> are placed at the front of the output): initialize `in_i = 0`,
> `out_i = broadcast_ndim`, then iterate slots `i` over `indices`:
> - null slot: `dim_map[in_i++] = out_i++` (map this input dim to the next
>   output dim after the broadcast prefix).
> - mask index: for each of its `index.dim()` consumed input dims,
>   `dim_map[in_i++] = -1`.
> - integer index: `dim_map[in_i++] = -1`.
> Then for the trailing input dims `i` in
> `[num_indexed_dims + num_null_indices, in.dim())`:
> `dim_map[i] = i - num_indexed_dims + broadcast_ndim`.
> No return value; writes only into `dim_map`.

> [spec:et:def:advanced-index-util.torch.executor.compute-index-map-fn]
> void compute_index_map( const Tensor& in, TensorOptList indices, int32_t* ix_map)

> [spec:et:sem:advanced-index-util.torch.executor.compute-index-map-fn]
> Fills `ix_map` (length `in.dim()`) mapping each INDEXED input dimension to the
> slot number `i` in `indices` that indexes it; non-indexed input dims are -1.
> Steps:
> 1. Initialize `ix_map[i] = -1` for all `i` in `[0, in.dim())`.
> 2. Walk `in_i = 0` over input dims while iterating slots `i` over `indices`:
>    - non-null slot that is a mask index: for each of its `index.dim()`
>      consumed input dims, `ix_map[in_i++] = i` (all those input dims point at
>      the same mask slot `i`).
>    - non-null slot that is an integer index: `ix_map[in_i++] = i`.
>    - null slot: `in_i++` (leaves the pre-initialized -1 in place for that dim).
> No return value; writes only into `ix_map`.

> [spec:et:def:advanced-index-util.torch.executor.count-index-blocks-fn]
> size_t count_index_blocks(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.count-index-blocks-fn]
> Counts the number of maximal contiguous runs ("blocks") of non-null indices
> in the list. Iterate slots `i` over `indices` in order, tracking a boolean
> `in_block` (initially false) and `block_count` (initially 0):
> - When a slot has a value and `in_block` is false: set `in_block = true` and
>   increment `block_count` (start of a new block).
> - When a slot has a value and `in_block` is already true: do nothing (still in
>   the same block).
> - When a slot is null: set `in_block = false` (block ended).
> Return `block_count`. Consequences: 0 means the list is empty or all-null; 1
> means all non-null indices are contiguous (nulls only allowed at the ends);
> >1 means there are null indices interleaved between non-null indices. This
> block count is what callers use to decide the `adjacent` flag elsewhere.

> [spec:et:def:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]
> size_t _count_trues_in_mask_index(const Tensor& index)

> [spec:et:sem:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]
> Counts the number of set (nonzero) elements in a mask index tensor. This is a
> template `_count_trues_in_mask_index<CTYPE_IX>`: obtain the typed data pointer
> `index.const_data_ptr<CTYPE_IX>()`, then sum 1 for every flat element `i` in
> `[0, index.numel())` whose value is truthy (nonzero for uint8, true for bool).
> Return the count as `size_t`. The public dispatcher
> `count_trues_in_mask_index(index)` selects the instantiation by dtype: Bool ->
> `bool`, otherwise (Byte) -> `uint8_t`. Iteration is over the raw contiguous
> element buffer in flat order.

> [spec:et:def:advanced-index-util.torch.executor.get-in-coord-fn]
> bool get_in_coord( const Tensor& in, TensorOptList indices, size_t start, size_t broadcast_ndim, int32_t* dim_map, int32_t* ix_map, size_t* out_coord, size_t* in_coord)

> [spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn]
> Given one fully-delinearized output coordinate `out_coord`, computes the
> corresponding input coordinate `in_coord` (one entry per input dim). `start`
> is the output offset at which the broadcast index dims begin (leading-null
> count for adjacent, or 0 for non-adjacent since broadcast dims lead),
> `broadcast_ndim` is the width of the broadcasted index shape, and `dim_map` /
> `ix_map` come from
> `[spec:et:sem:advanced-index-util.torch.executor.compute-dim-map-fn]` /
> `[spec:et:sem:advanced-index-util.torch.executor.compute-index-map-fn]`.
> Iterate input dim `i` from 0 to `in.dim()-1`:
> - If `dim_map[i] >= 0` (a passthrough/non-indexed dim): copy
>   `in_coord[i] = out_coord[dim_map[i]]`.
> - Otherwise (`dim_map[i] == -1`, an indexed dim): the index tensor is
>   `indices[ix_map[i]].value()`. Build the index-space coordinate `ix_coord` of
>   length `broadcast_ndim` by copying `ix_coord[j] = out_coord[j + start]` for
>   `j` in `[0, broadcast_ndim)`.
>   - Mask index: let `query_ix = ix_coord[broadcast_ndim - 1]` (masks broadcast
>     over the last broadcast dim). Call `query_mask_index(index, query_ix,
>     query_result)` per
>     `[spec:et:sem:advanced-index-util.torch.executor.query-mask-index-fn]` to
>     get the multi-dim coordinate of the `query_ix`-th true element. Write it
>     into `in_coord[i + j] = query_result[j]` for `j` in `[0, index.dim())`,
>     then skip ahead by advancing the loop counter `i += index.dim() - 1` (the
>     mask consumed `index.dim()` input dims).
>   - Integer index: `index_val = query_integral_index(index, ix_coord, broadcast_ndim)`
>     per `[spec:et:sem:advanced-index-util.torch.executor.query-integral-index-fn]`.
>     If `index_val < 0`, wrap it: `index_val += in.size(i)`. Then
>     ET_CHECK_OR_RETURN_FALSE(`0 <= index_val < in.size(i)`) — an out-of-bounds
>     index returns false with an "out of bounds" message. On success store
>     `in_coord[i] = (size_t)index_val`.
> Return true if all coordinates resolved without an out-of-bounds failure.

> [spec:et:def:advanced-index-util.torch.executor.get-in-ix-fn]
> std::pair<size_t, bool> get_in_ix( const Tensor& in, TensorOptList indices, Tensor& out, size_t out_ix, size_t start, size_t broadcast_ndim, int32_t* dim_map, int32_t* ix_map)

> [spec:et:sem:advanced-index-util.torch.executor.get-in-ix-fn]
> Maps a single flat output element index `out_ix` to the flat input element
> index it reads from. Steps:
> 1. Delinearize `out_ix` into a per-dim coordinate `out_coord` using `out`'s
>    shape (`delinearize_index(out_ix, out, out_coord, kTensorDimensionLimit)`).
> 2. Compute `in_coord` via `get_in_coord(in, indices, start, broadcast_ndim,
>    dim_map, ix_map, out_coord, in_coord)` per
>    `[spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn]`. If it
>    returns false (out-of-bounds integer index), return `{0, false}`.
> 3. Otherwise linearize `in_coord` into a flat input offset via
>    `coordinateToIndex(in, in_coord)` (row-major using `in`'s strides/sizes) and
>    return `{that offset, true}`.
> The bool in the returned pair signals success; on failure the offset is a
> meaningless 0 that callers must not use.

> [spec:et:def:advanced-index-util.torch.executor.get-index-out-target-size-fn]
> bool get_index_out_target_size( const Tensor& in, TensorOptList indices, bool adjacent, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:advanced-index-util.torch.executor.get-index-out-target-size-fn]
> Computes the expected output shape (`out_sizes`, `out_ndim`) for advanced
> indexing. Steps:
> 1. `get_indices_broadcast_shape(indices, broadcast_sizes, &broadcast_ndim)`
>    per `[spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn]`;
>    if it returns false, return false.
> 2. `num_null_indices = get_num_null_indices(indices)`,
>    `num_indexed_dims = get_num_indexed_dims(indices)`.
> 3. ET_CHECK_OR_RETURN_FALSE(`num_null_indices + num_indexed_dims <= in.dim()`)
>    ("Indexing too many dimensions"); false otherwise.
> 4. ET_CHECK_OR_RETURN_FALSE(`in.dim() + broadcast_ndim - num_indexed_dims <=
>    kTensorDimensionLimit`) ("Out tensor would exceed number of allowed
>    dimensions"); false otherwise.
> 5. Set `*out_ndim = in.dim() + broadcast_ndim - num_indexed_dims` (indexed
>    dims collapse into the single broadcasted index shape).
> 6. Fill `out_sizes` depending on `adjacent`:
>    - `adjacent == true` (single contiguous block of non-null indices, so the
>      broadcast shape stays in place): let `start = get_num_leading_null_indices(indices)`.
>      - `out_sizes[i] = in.size(i)` for `i` in `[0, start)`.
>      - `out_sizes[i + start] = broadcast_sizes[i]` for `i` in `[0, broadcast_ndim)`.
>      - For `i` in `[num_indexed_dims + start, in.dim())`:
>        `out_sizes[i + broadcast_ndim - num_indexed_dims] = in.size(i)`.
>    - `adjacent == false` (indexed dims scattered, broadcast shape goes to the
>      front):
>      - `out_sizes[i] = broadcast_sizes[i]` for `i` in `[0, broadcast_ndim)`.
>      - Walk `in_i = 0`, `out_i = broadcast_ndim` over slots `i` of `indices`:
>        for a null slot, `out_sizes[out_i++] = in.size(in_i++)`; for a mask
>        index, advance `in_i += index.dim()`; for an integer index, advance
>        `in_i += 1` (indexed/consumed dims produce no output size here).
>      - For trailing input dims `i` in
>        `[num_indexed_dims + num_null_indices, in.dim())`:
>        `out_sizes[i + broadcast_ndim - num_indexed_dims] = in.size(i)`.
> 7. Return true.

> [spec:et:def:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn]
> size_t get_indices_broadcast_ndim(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-ndim-fn]
> Returns the number of dimensions in the broadcasted shape of all index
> tensors. Track `ndim = 0` and iterate slots `i` over `indices`; for each
> non-null index:
> - mask index: it contributes a 1-D index space, so if `ndim == 0` set
>   `ndim = 1` (never lowers an existing larger ndim).
> - integer index: if `index.dim() > ndim`, raise `ndim = index.dim()`.
> Null slots are ignored. Return `ndim` (0 if there are no non-null indices).
> This equals the max rank among integer indices, floored to 1 if any mask
> index is present.

> [spec:et:def:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn]
> bool get_indices_broadcast_shape( TensorOptList indices, Tensor::SizesType* ix_sizes, size_t* ix_ndim)

> [spec:et:sem:advanced-index-util.torch.executor.get-indices-broadcast-shape-fn]
> Computes the NumPy/torch-style broadcast shape across all non-null index
> tensors, writing it (in forward order) into `ix_sizes` and its rank into
> `*ix_ndim`. Internally it accumulates the shape reversed (rightmost/trailing
> dim first) in `rev_ix_sizes`, tracking `curr_ndim = 0`. Iterate slots `i` over
> `indices`; for each non-null index:
> - Mask index: its broadcast extent is a single dimension of length
>   `len = count_trues_in_mask_index(index)` (number of set elements) per
>   `[spec:et:sem:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]`.
>   Merge it into `rev_ix_sizes[0]`: if `curr_ndim == 0` set `curr_ndim = 1` and
>   `rev_ix_sizes[0] = len`; else if the running trailing size is 1 overwrite it
>   with `len`; else if `len != 1` and it differs from the running size,
>   ET_CHECK_OR_RETURN_FALSE fails ("Broadcast of mask index failed.").
> - Integer index: for each `j` in `[0, index.dim())`, take the reversed size
>   `rev_j_size = index.size(index.dim()-j-1)` and merge it at position `j`: if
>   `j >= curr_ndim` grow `curr_ndim = j+1` and set `rev_ix_sizes[j] = rev_j_size`;
>   else if `rev_ix_sizes[j] == 1` overwrite with `rev_j_size`; else if
>   `rev_j_size != 1` and it differs, ET_CHECK_OR_RETURN_FALSE fails
>   ("Broadcast of index failed.").
> After processing all indices, un-reverse: `ix_sizes[i] = rev_ix_sizes[curr_ndim - i - 1]`
> for `i` in `[0, curr_ndim)`, set `*ix_ndim = curr_ndim`, and return true.
> Standard broadcasting rules apply: a dim of size 1 broadcasts to the other's
> size; mismatched non-1 sizes are an error.

> [spec:et:def:advanced-index-util.torch.executor.get-num-indexed-dims-fn]
> size_t get_num_indexed_dims(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.get-num-indexed-dims-fn]
> Returns how many INPUT dimensions are consumed by non-null indices. Iterate
> slots `i` over `indices`, accumulating `num_indexed_dims = 0`; for each
> non-null index add `index.dim()` if it is a mask index (a mask consumes as
> many input dims as its rank), otherwise add 1 (an integer index consumes a
> single input dim). Null slots contribute 0. Return the total.

> [spec:et:def:advanced-index-util.torch.executor.get-num-leading-null-indices-fn]
> size_t get_num_leading_null_indices(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.get-num-leading-null-indices-fn]
> Returns the count of consecutive null indices at the START of the list. Start
> `start = 0` and increment it while `indices[start]` has no value, stopping at
> the first non-null slot. Return `start`. Note: there is no bounds guard, so if
> every slot is null this walks off the end of `indices` (undefined behavior);
> callers only invoke this when at least one non-null index exists (e.g. the
> `adjacent` path where a block is known to be present).

> [spec:et:def:advanced-index-util.torch.executor.get-num-null-indices-fn]
> size_t get_num_null_indices(TensorOptList indices)

> [spec:et:sem:advanced-index-util.torch.executor.get-num-null-indices-fn]
> Returns the total number of null (nullopt) slots in `indices`. Iterate all
> slots `i` and increment a counter for each slot where `!indices[i].has_value()`.
> Return the count.

> [spec:et:def:advanced-index-util.torch.executor.is-mask-index-fn]
> bool is_mask_index(const Tensor& index)

> [spec:et:sem:advanced-index-util.torch.executor.is-mask-index-fn]
> Returns true iff `index` is a boolean-style mask index, i.e. its
> `scalar_type()` is Bool or Byte (uint8); returns false for any other dtype
> (Int, Long treated as integer/positional indices).

> [spec:et:def:advanced-index-util.torch.executor.query-integral-index-fn]
> int64_t query_integral_index( const Tensor& index, size_t* ix_coord, size_t broadcast_ndim)

> [spec:et:sem:advanced-index-util.torch.executor.query-integral-index-fn]
> Reads a signed index value out of an integer (Int/Long) index tensor at a
> given broadcast coordinate. Steps:
> 1. Compute the flat offset into the index tensor by applying broadcasting:
>    `flat_ix = linearize_access_indexes({ix_coord, broadcast_ndim}, broadcast_ndim, index)`.
>    This treats `ix_coord` (length `broadcast_ndim`) as a coordinate in the
>    broadcasted index space and maps it to `index`'s own flat storage,
>    respecting broadcasting (size-1 dims contribute stride 0) and any leading
>    dims the index lacks.
> 2. Read the element: if `index.scalar_type()` is Int, load `int32_t` at
>    `flat_ix` and widen to `int64_t`; otherwise (Long) load `int64_t` at
>    `flat_ix` directly.
> 3. Return the raw (possibly negative) index value; negative-wrapping and
>    bounds checking are done by the caller in
>    `[spec:et:sem:advanced-index-util.torch.executor.get-in-coord-fn]`.

> [spec:et:def:advanced-index-util.torch.executor.query-mask-index-fn]
> void _query_mask_index(const Tensor& index, size_t query_idx, size_t* res)

> [spec:et:sem:advanced-index-util.torch.executor.query-mask-index-fn]
> Finds the multi-dimensional coordinate of the `query_idx`-th set element of a
> mask index and writes it into `res`. This is a template
> `_query_mask_index<CTYPE_IX>`:
> 1. Get the typed data pointer `index.const_data_ptr<CTYPE_IX>()`.
> 2. Broadcasting rule for masks: compute `num_true = count of set elements`
>    (per `[spec:et:sem:advanced-index-util.torch.executor.count-trues-in-mask-index-fn]`);
>    if `num_true == 1`, force `query_idx = 0` (a single-true mask broadcasts to
>    every query).
> 3. Scan the flat buffer `[0, index.numel())` in order counting set elements;
>    when the running count of set elements equals `query_idx`, take that flat
>    position as `flat_ix` and stop. (If `query_idx >= num_true`, no element
>    matches and `flat_ix` remains 0.)
> 4. Delinearize `flat_ix` against `index`'s shape into `res`
>    (`delinearize_index(flat_ix, index, res, kTensorDimensionLimit)`), producing
>    the per-dim coordinate of that true element.
> The public dispatcher `query_mask_index(index, query_idx, res)` picks the
> instantiation by dtype: Bool -> `bool`, otherwise (Byte) -> `uint8_t`. No
> return value; writes only into `res`.

