# kernels/portable/cpu/util/broadcast_util.cpp, kernels/portable/cpu/util/broadcast_util.h

> [spec:et:def:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]
> inline void apply_binary_elementwise_fn( const Op& compute_fun, const Tensor& a, const Tensor& b, const Tensor& out)

> [spec:et:sem:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]
> Template function parameterized on `CTYPE_A`, `CTYPE_B`, `CTYPE_OUT`, and the
> callable type `Op`. Applies a binary elementwise operation over broadcast-
> aligned inputs `a` and `b`, writing to `out`. It performs no dtype dispatch,
> no shape validation, and no output resizing — the caller is responsible for
> having already resized `out` to the broadcast target shape (see
> `[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`)
> and for choosing the concrete C types.
>
> Steps:
> 1. Obtain typed read pointers `data_a = a.const_data_ptr<CTYPE_A>()` and
>    `data_b = b.const_data_ptr<CTYPE_B>()`, and the typed write pointer
>    `data_out = out.mutable_data_ptr<CTYPE_OUT>()`.
> 2. Iterate over the tuples `(out_index, a_index, b_index)` produced by
>    `BroadcastIndexesRange<2>(out, a, b)` — see
>    `[spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]`.
>    That range visits every element of `out` in flat (row-major over `out`'s
>    logical shape) order exactly once, and for each `out` element yields the
>    corresponding flat element offsets into `a` and `b` after applying
>    broadcasting: a size-1 or missing (leading) dimension in an input
>    contributes stride 0 so the same input element is re-read across the
>    broadcasted extent.
> 3. For each tuple compute `data_out[out_index] =
>    compute_fun(data_a[a_index], data_b[b_index])`. The two input values are
>    read as `CTYPE_A`/`CTYPE_B`, passed positionally (a first, b second) to
>    `compute_fun`, and its result is stored as `CTYPE_OUT` (implicit C++
>    conversion of the callable's return value to `CTYPE_OUT`).
>
> There is no accumulation and no explicit iteration counter beyond the range;
> if `out` is empty (numel 0) the range yields nothing and the function is a
> no-op. `out` is written in place; the function returns void and reports no
> error (validity of pointers, dtypes, and broadcastability is the caller's
> precondition).

> [spec:et:def:broadcast-util.torch.executor.apply-ternary-elementwise-fn-fn]
> inline void apply_ternary_elementwise_fn( const Op& compute_fun, const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& out)

> [spec:et:sem:broadcast-util.torch.executor.apply-ternary-elementwise-fn-fn]
> Template function parameterized on `CTYPE_A`, `CTYPE_B`, `CTYPE_C`,
> `CTYPE_OUT`, and the callable type `Op`. Identical in structure to
> `[spec:et:sem:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]`
> but for three inputs. It performs no dtype dispatch, no shape validation, and
> no output resizing; the caller must have resized `out` to the three-way
> broadcast target shape (see
> `[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`).
>
> Steps:
> 1. Obtain typed read pointers `data_a`, `data_b`, `data_c` via
>    `const_data_ptr<CTYPE_A>()` / `<CTYPE_B>()` / `<CTYPE_C>()` and the write
>    pointer `data_out = out.mutable_data_ptr<CTYPE_OUT>()`.
> 2. Iterate over the tuples `(out_index, a_index, b_index, c_index)` from
>    `BroadcastIndexesRange<3>(out, a, b, c)` — see
>    `[spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]`.
>    The range visits every `out` element once in `out` flat order, yielding for
>    each the broadcast-mapped flat offsets into `a`, `b`, and `c` (size-1 /
>    missing dims contribute stride 0).
> 3. For each tuple compute `data_out[out_index] =
>    compute_fun(data_a[a_index], data_b[b_index], data_c[c_index])`, passing the
>    three inputs positionally (a, b, c) and storing the result as `CTYPE_OUT`.
>
> Empty `out` yields a no-op. `out` is written in place; returns void; no error
> reporting (preconditions are the caller's responsibility).

> [spec:et:def:broadcast-util.torch.executor.broadcast-tensor-fn]
> Tensor broadcast_tensor( const Tensor& broadcast_from, const Tensor& broadcast_to)

> [spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn]
> DEPRECATED helper. Broadcasts `broadcast_from` to match `broadcast_to`'s
> shape and returns a newly heap-allocated tensor holding the materialized
> (repeated) data. The returned tensor owns dynamically allocated memory and
> must later be released with
> `[spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn]`.
>
> Let `broadcast_to_shape = broadcast_to.sizes()`,
> `broadcast_from_shape = broadcast_from.sizes()`,
> `broadcast_to_dim_order = broadcast_to.dim_order()`,
> `broadcast_to_strides = broadcast_to.strides()`.
>
> Validation (each check aborts the program via ET_CHECK_MSG on failure; there
> is no error return):
> 1. `broadcast_from` must be non-empty: require
>    `broadcast_from.numel() != 0 || !broadcast_from.sizes().empty()` (i.e.
>    reject a truly-empty tensor; a 0-dim scalar has empty sizes but numel 1 and
>    passes). Message "Input tensor must be non-empty".
> 2. `broadcast_to` must have non-empty sizes: require
>    `!broadcast_to.sizes().empty()`. Message "Input tensor must be non-empty".
> 3. Require `broadcast_to_shape.size() >= broadcast_from_shape.size()`
>    (broadcast_to must be at least as high-dimensional).
> 4. Require `tensor_is_broadcastable_to(broadcast_from, broadcast_to)` is true
>    (see `[spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn]`).
>
> Construction:
> 5. Allocate the output tensor via `make_tensor` (see
>    `[spec:et:sem:broadcast-util.torch.executor.make-tensor-fn]`) with
>    `broadcast_to_shape`, `broadcast_to_dim_order`, `broadcast_to_strides`, and
>    dtype `broadcast_from.scalar_type()` — the result has `broadcast_to`'s shape
>    and layout but `broadcast_from`'s dtype.
> 6. Compute a per-dimension `repeats` array of length `ndim = broadcast_to.dim()`
>    (as int64_t). Initialize `repeats[i] = broadcast_to_shape[i]` for every `i`.
>    Then, aligning the two shapes at their trailing dimensions (iterate
>    `i = to_size-1`, `j = from_size-1` while `j >= 0`, decrementing both): if
>    `broadcast_to_shape[i] == broadcast_from_shape[j]` set `repeats[i] = 1`
>    (no repeat needed along a matching dim). Leading dims of `broadcast_to` that
>    have no counterpart in `broadcast_from` keep `repeats[i] = broadcast_to_shape[i]`.
> 7. Call `repeat_tensor(broadcast_from, repeats, out)` — see
>    `[spec:et:sem:repeat-util.torch.executor.repeat-tensor-fn]` — which fills
>    `out`'s data by tiling `broadcast_from` according to `repeats`. Its return
>    value must equal `Error::Ok` (ET_CHECK aborts otherwise).
> 8. Free the temporary `repeats` buffer and return `out`.
>
> The result is returned by value (a Tensor referencing the heap-allocated impl
> and buffers). Note the header/source comments flag known memory leaks in
> `make_tensor`'s dim_order buffer.

> [spec:et:def:broadcast-util.torch.executor.executorch.aten.tensor-broadcast-tensor-fn]
> ET_DEPRECATED executorch::aten::Tensor broadcast_tensor(

> [spec:et:sem:broadcast-util.torch.executor.executorch.aten.tensor-broadcast-tensor-fn]
> This is the public (header) declaration of the deprecated `broadcast_tensor`
> function; its runtime behavior is fully specified by
> `[spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn]`. It takes
> `broadcast_from` and `broadcast_to` tensors and returns a new heap-allocated
> `executorch::aten::Tensor` with `broadcast_to`'s shape and layout, holding
> `broadcast_from`'s data repeated to fill that shape (dtype = broadcast_from's
> dtype). The returned tensor owns dynamically allocated memory and must be
> released with `free_broadcast_tensor` (see
> `[spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn]`).
> Marked ET_DEPRECATED: prefer index remapping via `delinearize_index()` and
> `linearize_access_indexes()` to avoid allocation.

> [spec:et:def:broadcast-util.torch.executor.free-broadcast-tensor-fn]
> void free_broadcast_tensor(const Tensor& broadcast_tensor)

> [spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn]
> DEPRECATED. Releases all dynamically allocated memory owned by a tensor that
> was previously produced by `broadcast_tensor` (see
> `[spec:et:sem:broadcast-util.torch.executor.broadcast-tensor-fn]` /
> `[spec:et:sem:broadcast-util.torch.executor.make-tensor-fn]`). It must only be
> called on such a tensor.
>
> In order, it calls `free()` (the C library free) on:
> 1. `broadcast_tensor.const_data_ptr()` — the element data buffer.
> 2. `broadcast_tensor.sizes().data()` — the sizes array buffer.
> 3. `broadcast_tensor.dim_order().data()` — the dim-order array buffer.
> 4. `broadcast_tensor.strides().data()` — the strides array buffer.
> 5. `broadcast_tensor.unsafeGetTensorImpl()` — the TensorImpl object itself.
>
> Each pointer is cast to `void*` before freeing. Returns void. It does not null
> out the tensor's fields, so the tensor must not be used afterward. In a Rust
> port this corresponds to dropping/deallocating the owned buffers and the impl
> struct; if the ported `make_tensor` uses ordinary owned allocations, this
> becomes a no-op / Drop.

> [spec:et:def:broadcast-util.torch.executor.get-broadcast-target-size-fn]
> ET_NODISCARD Error get_broadcast_target_size( const executorch::aten::ArrayRef<Tensor::SizesType> a_size, const executorch::aten::ArrayRef<Tensor::SizesType> b_size, Tensor::SizesType* out_sizes, const size_t out_sizes_len, size_t* out_dim)

> [spec:et:sem:broadcast-util.torch.executor.get-broadcast-target-size-fn]
> Computes the shape two tensors would broadcast to, without allocating an
> actual tensor. Writes the result into the caller-provided `out_sizes` buffer
> (capacity `out_sizes_len`) and the resulting rank into `*out_dim`. Returns an
> `Error`. This is the shape-array overload; the Tensor overload simply forwards
> `a.sizes()` and `b.sizes()`.
>
> Steps:
> 1. Broadcastability check: if `!tensors_are_broadcastable_between(a_size,
>    b_size)` (see
>    `[spec:et:sem:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn]`),
>    log an Error (only when logging is enabled) reporting both shapes and
>    return `Error::InvalidArgument`. Neither `out_sizes` nor `*out_dim` is
>    written in this case.
> 2. Capacity check: let `a_dim = a_size.size()`, `b_dim = b_size.size()`.
>    Require `a_dim <= out_sizes_len && b_dim <= out_sizes_len`
>    (ET_CHECK_OR_RETURN_ERROR); on failure return `Error::InvalidArgument`
>    (checking the larger input rank against the buffer capacity).
> 3. Set `*out_dim = max(a_dim, b_dim)`.
> 4. Fill `out_sizes[0 .. *out_dim)` by aligning the two shapes at their
>    trailing dimensions. Iterate with `a_idx = a_dim-1`, `b_idx = b_dim-1`,
>    `expected_target_idx = *out_dim-1`, decrementing all three while
>    `expected_target_idx >= 0`. For each position:
>    - If both `a_idx >= 0` and `b_idx >= 0` (both inputs have this dim):
>      `out_sizes[expected_target_idx] = (b_size[b_idx] == 1) ? a_size[a_idx]
>      : b_size[b_idx]`. Note this takes `a`'s size when `b`'s is 1, otherwise
>      always takes `b`'s size — so when `a`'s size is 1 and `b`'s is not 1 it
>      correctly yields `b`'s size, and when both equal it yields `b`'s (equal)
>      size. (This branch relies on broadcastability already being verified in
>      step 1.)
>    - Otherwise (one input has run out of dims): take whichever input still has
>      this dim: `a_idx >= 0 ? a_size[a_idx] : b_size[b_idx]`.
> 5. Return `Error::Ok`.
>
> The trailing-aligned result matches PyTorch broadcasting: each output dim is
> the max of the two aligned input dims (with size-1 treated as broadcastable),
> and leading dims come from whichever tensor is higher-rank.

> [spec:et:def:broadcast-util.torch.executor.linearize-access-indexes-fn]
> size_t linearize_access_indexes( ArrayRef<size_t> indexes_broadcast_to, ssize_t broadcast_to_ndim, executorch::aten::ArrayRef<Tensor::SizesType> broadcast_from_shape, executorch::aten::ArrayRef<Tensor::StridesType> broadcast_from_strides)

> [spec:et:sem:broadcast-util.torch.executor.linearize-access-indexes-fn]
> Given a multi-dimensional access index into the broadcast_to tensor, compute
> the corresponding flat (linear) element offset into the broadcast_from tensor,
> accounting for broadcasting. Used to read the correct source element without
> materializing a broadcasted copy. This is the shape/strides overload; the
> Tensor overload forwards `broadcast_from.sizes()` and
> `broadcast_from.strides()`.
>
> Inputs: `indexes_broadcast_to` is the per-dimension index vector into
> broadcast_to (length == broadcast_to_ndim); `broadcast_to_ndim` is
> broadcast_to's rank; `broadcast_from_shape` and `broadcast_from_strides` are
> broadcast_from's sizes and strides.
>
> Steps:
> 1. Compute `num_skip_dims = broadcast_to_ndim - broadcast_from_shape.size()`
>    (the count of leading broadcast_to dims that broadcast_from does not have).
> 2. Take the trailing slice of the index vector aligned to broadcast_from:
>    `indexes_broadcast_from = indexes_broadcast_to.slice(num_skip_dims,
>    broadcast_to_ndim - num_skip_dims)` — i.e. drop the first `num_skip_dims`
>    entries, keeping the last `broadcast_from_shape.size()` entries.
> 3. Assert (ET_CHECK) `indexes_broadcast_from.size() ==
>    broadcast_from_shape.size()`.
> 4. Initialize `linear_index = 0`. For each `i` in `0 ..
>    indexes_broadcast_from.size()`:
>    - If `indexes_broadcast_from[i] >= broadcast_from_shape[i]` (the requested
>      index is out of range for this source dim, which happens exactly when this
>      dim was broadcast, so `broadcast_from_shape[i] == 1` and the target index
>      exceeds 0): assert (ET_CHECK_MSG) that `broadcast_from_shape[i] == 1`,
>      then `continue` — contribute nothing (stride-0 broadcast: always read
>      element 0 along this dim).
>    - Otherwise add `indexes_broadcast_from[i] * broadcast_from_strides[i]` to
>      `linear_index`.
> 5. Return `linear_index`.
>
> No allocation, no output mutation; returns a size_t flat index. The strides
> are broadcast_from's own strides (in elements), so the result indexes
> broadcast_from's underlying data buffer directly.

> [spec:et:def:broadcast-util.torch.executor.make-tensor-fn]
> Tensor make_tensor( const ArrayRef<Tensor::SizesType>& sizes, const ArrayRef<Tensor::DimOrderType>& dim_order, const ArrayRef<Tensor::StridesType>& strides, const ScalarType& dtype)

> [spec:et:sem:broadcast-util.torch.executor.make-tensor-fn]
> File-internal (anonymous-namespace) helper used by `broadcast_tensor`. Builds
> and returns a heap-allocated, fully owning Tensor from the given `sizes`,
> `dim_order`, `strides`, and `dtype`, with an uninitialized data buffer sized
> to hold the tensor. All buffers and the impl are allocated with C `malloc`
> and copied by `memcpy`; the caller must later free them via
> `free_broadcast_tensor` (see
> `[spec:et:sem:broadcast-util.torch.executor.free-broadcast-tensor-fn]`).
>
> Steps:
> 1. Let `dim = sizes.size()`.
> 2. Allocate `dim * sizeof(SizesType)` bytes; ET_CHECK_MSG non-null ("Failed to
>    malloc for size bytes"); memcpy `sizes.data()` into it. This is the sizes
>    buffer.
> 3. Allocate `dim * sizeof(DimOrderType)` bytes; ET_CHECK_MSG non-null ("Failed
>    to malloc for dim order bytes"); memcpy `dim_order.data()` into it. (Source
>    comments note this buffer is leaked in some paths — TODO T147221312.)
> 4. Allocate `dim * sizeof(StridesType)` bytes; ET_CHECK_MSG non-null ("Failed
>    to malloc for strides bytes"); memcpy `strides.data()` into it.
> 5. Allocate `sizeof(TensorImpl)` bytes; ET_CHECK_MSG non-null ("Failed to
>    malloc for data TensorImpl"). Placement-new a `TensorImpl` in it with
>    arguments `(dtype, dim, sizes_ptr, /*data=*/nullptr, dim_order_ptr,
>    strides_ptr)` — the data pointer starts null.
> 6. Allocate `tensor_impl->nbytes()` bytes for the element data; ET_CHECK_MSG
>    non-null ("Failed to malloc for data buffer"); call
>    `tensor_impl->set_data(data_ptr)`. The data buffer is left uninitialized
>    (garbage) — the caller (`broadcast_tensor` via `repeat_tensor`) fills it.
> 7. Return `Tensor{tensor_impl}` wrapping the impl.
>
> On any allocation failure the ET_CHECK_MSG aborts the program. No error return.
> Every ET_CHECK_MSG failure aborts (there is no graceful error path). In a Rust
> port these become owned Vec/Box allocations rather than raw malloc.

> [spec:et:def:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
> ET_NODISCARD inline Error resize_to_broadcast_target_size( const Tensor& a, const Tensor& b, const Tensor& c, Tensor& out)

> [spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]
> This annotation is on the three-input (a, b, c) inline overload declared in
> the header. It computes the shape that three tensors broadcast to and resizes
> `out` to it. Returns an `Error`. (There is also a two-input inline overload
> in the same header, not separately annotated, that does the same with just a
> and b.)
>
> Three-input steps:
> 1. Declare a stack buffer `interim_output_size[kTensorDimensionLimit]` and
>    `interim_output_dim = 0`.
> 2. Compute the broadcast size of `a` and `b` via `get_broadcast_target_size(a,
>    b, interim_output_size, kTensorDimensionLimit, &interim_output_dim)` (see
>    `[spec:et:sem:broadcast-util.torch.executor.get-broadcast-target-size-fn]`).
>    On non-Ok error, return it immediately (ET_CHECK_OK_OR_RETURN_ERROR with
>    message "Failed to get broadcast target size").
> 3. Declare `expected_output_size[kTensorDimensionLimit]` and
>    `expected_output_dim = 0`.
> 4. Compute the broadcast size of the interim shape `{interim_output_size,
>    interim_output_dim}` against `c.sizes()` via `get_broadcast_target_size`,
>    writing into `expected_output_size`/`expected_output_dim`. On non-Ok error,
>    return it immediately.
> 5. Return `resize_tensor(out, {expected_output_size, expected_output_dim})`,
>    which resizes `out` to the computed broadcast shape (subject to `out`'s
>    dynamic-shape capacity) and returns its Error.
>
> The two-input overload is identical but performs a single
> `get_broadcast_target_size(a, b, ...)` and resizes `out` to that. Broadcasting
> of three tensors is left-associative: (a broadcast b) then broadcast c;
> because broadcasting is associative/commutative on shapes this equals the full
> three-way broadcast. No data is written — only `out`'s shape metadata changes.

> [spec:et:def:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn]
> bool tensor_is_broadcastable_to( const executorch::aten::ArrayRef<Tensor::SizesType> broadcast_from_shape, const executorch::aten::ArrayRef<Tensor::SizesType> broadcast_to_shape)

> [spec:et:sem:broadcast-util.torch.executor.tensor-is-broadcastable-to-fn]
> Returns a bool: whether `broadcast_from_shape` can be (directionally)
> broadcast onto `broadcast_to_shape`. This is the shape-array overload; the
> Tensor overload forwards `broadcast_from.sizes()` and `broadcast_to.sizes()`.
> Unlike `tensors_are_broadcastable_between`, this is asymmetric — only
> `broadcast_from` may have size-1 dims expanded, and it may not be
> higher-rank than the target.
>
> Steps:
> 1. If `broadcast_to_shape.size() < broadcast_from_shape.size()` return false
>    (target must be at least as high-rank as the source).
> 2. Initialize `feasible_bcast = true`. Align the two shapes at their trailing
>    dimensions: iterate `i = to_size-1`, `j = from_size-1` while `j >= 0`,
>    decrementing both. Let `to_s = broadcast_to_shape[i]`,
>    `from_s = broadcast_from_shape[j]`. Set `feasible_bcast &= (to_s == from_s
>    || from_s == 1)`. If `feasible_bcast` became false, return false
>    immediately. (Only a size-1 source dim may differ from the target; a size-1
>    target dim does NOT permit a differing non-1 source dim — asymmetric.)
> 3. After the loop return `feasible_bcast` (true). Leading target dims with no
>    source counterpart are always acceptable (the source is treated as size-1
>    there).
>
> Pure function; no allocation, no mutation.

> [spec:et:def:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn]
> bool tensors_are_broadcastable_between( const executorch::aten::ArrayRef<Tensor::SizesType> a_shape, const executorch::aten::ArrayRef<Tensor::SizesType> b_shape)

> [spec:et:sem:broadcast-util.torch.executor.tensors-are-broadcastable-between-fn]
> Returns a bool: whether two tensor shapes can both be broadcast to a common
> shape (the symmetric PyTorch broadcastability test). This is the shape-array
> overload; the Tensor overload forwards `a.sizes()` and `b.sizes()`.
>
> Steps:
> 1. Let `a_dim = a_shape.size()`, `b_dim = b_shape.size()`. Ranks may differ
>    and either may be 0 (0-dim scalars are broadcastable with anything — the
>    function does not reject them).
> 2. Align the two shapes at their trailing dimensions: iterate
>    `a_index = a_dim-1`, `b_index = b_dim-1` while both `a_index >= 0 &&
>    b_index >= 0`, decrementing both. For each aligned pair, the dims are
>    compatible if `a_shape[a_index] == b_shape[b_index]` OR
>    `a_shape[a_index] == 1` OR `b_shape[b_index] == 1`; if compatible,
>    `continue`. Otherwise return false immediately.
> 3. When either index runs out (the loop ends), the remaining leading dims of
>    the higher-rank tensor are always acceptable (implicit size-1 on the other
>    side). Return true.
>
> Symmetric (order of a and b does not matter). Pure function; no allocation, no
> mutation.

