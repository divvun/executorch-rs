# kernels/portable/cpu/util/copy_ops_util.cpp, kernels/portable/cpu/util/copy_ops_util.h

> [spec:et:def:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn]
> size_t as_strided_copy_compute_storage_nbytes( IntArrayRef sizes, IntArrayRef strides, size_t itemsize_bytes)

> [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn]
> Computes the number of bytes of underlying storage required to hold a
> strided view described by `sizes` and `strides`, given a per-element size
> of `itemsize_bytes`. The result is `itemsize_bytes` times one plus the flat
> element offset of the last logical element (the element addressed by index
> `sizes[i]-1` in every dimension).
>
> Algorithm:
> - Initialize an accumulator `size = 1` (measured in elements).
> - Iterate `i` over `[0, sizes.size())` in ascending order:
>   - If `sizes[i] == 0`, the view contains no elements; return `0`
>     immediately (short-circuits, ignoring all remaining dimensions).
>   - Otherwise add `strides[i] * (sizes[i] - 1)` to `size`. Arithmetic is on
>     the signed 64-bit `int64_t` element values of the stride/size arrays,
>     assigned into the `size_t` accumulator; strides are assumed non-negative
>     (the caller `[spec:et:sem:copy-ops-util.torch.executor.check-as-strided-copy-args-fn]`
>     rejects negative strides before this is used).
> - Return `size * itemsize_bytes`.
>
> `sizes` and `strides` are assumed to have equal length (guaranteed by the
> caller). Empty `sizes` (rank-0) yields `1 * itemsize_bytes`.

> [spec:et:def:copy-ops-util.torch.executor.as-strided-copy-fn]
> void _as_strided_copy( CTYPE* input_data, CTYPE* output_data, Tensor& out, ArrayRef<int64_t> size, ArrayRef<int64_t> stride, int64_t dim)

> [spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-fn]
> Recursively copies elements from `input_data` (a strided view over the input
> storage) into `output_data` (contiguous over `out`), one dimension at a time,
> for the dimension index `dim`. Template-typed by the element C type `CTYPE`.
>
> Let `stride_dim = stride[dim]`.
>
> Base case — `dim == size.size() - 1` (last dimension):
> - Let `num_elements = size[dim]`.
> - If `stride_dim == 1`: copy `num_elements * sizeof(CTYPE)` bytes from
>   `input_data` to `output_data` via a raw byte copy (memcpy), i.e. a
>   contiguous block copy.
> - Else: for `i` in `[0, num_elements)` ascending, set
>   `output_data[i] = *input_data`, then advance `input_data` by `stride_dim`
>   elements. Output is written contiguously (stride 1); input is read with
>   stride `stride_dim`.
> - Return.
>
> Recursive case (`dim` not the last dimension):
> - Compute `trailing_dims = getTrailingDims(out, dim)`, the product of the
>   sizes of `out` for all dimensions strictly after `dim` (the number of
>   contiguous output elements spanned by advancing one step along `dim`).
> - For each of `size[dim]` iterations (index unused): recurse with
>   `_as_strided_copy(input_data, output_data, out, size, stride, dim+1)`, then
>   advance `input_data` by `stride_dim` elements and `output_data` by
>   `trailing_dims` elements.
>
> Net effect: output is filled in contiguous row-major (dim-order-default)
> order, while the input is read at flat offset
> `sum_over_d(index_d * stride[d])`. Caller `as_strided_copy` handles the
> rank-0 case (empty `size`) by copying a single element `out_data[0] =
> in_data[0]` and does not invoke this function; the input pointer passed in
> has already been advanced by `offset` elements.

> [spec:et:def:copy-ops-util.torch.executor.check-as-strided-copy-args-fn]
> bool check_as_strided_copy_args( const Tensor& in, ArrayRef<int64_t> size, ArrayRef<int64_t> stride, optional<int64_t> storage_offset, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-as-strided-copy-args-fn]
> Validates arguments for `as_strided_copy`. Returns `true` if all checks pass,
> otherwise logs and returns `false` (each check below is
> `ET_LOG_AND_RETURN_IF_FALSE` / `ET_CHECK_OR_RETURN_FALSE`: on failure it logs
> the message and returns `false`; the caller kernel turns this into an
> `Error::InvalidArgument`). Checks, in order:
> 1. `in` and `out` have the same dtype (`tensors_have_same_dtype`).
> 2. `size.size() == stride.size()` (shape and strides must have equal length).
> 3. Every stride value `val` in `stride` satisfies `val >= 0` (negative strides
>    are not supported).
> 4. Let `offset = storage_offset.value()` if `storage_offset` has a value, else
>    `0`. Require `offset >= 0` (no negative storage offset).
> 5. Bounds check: compute `storage_size_bytes =
>    as_strided_copy_compute_storage_nbytes(size, stride, in.element_size())`
>    per `[spec:et:sem:copy-ops-util.torch.executor.as-strided-copy-compute-storage-nbytes-fn]`
>    and `storage_offset_bytes = offset * in.element_size()`. If
>    `storage_size_bytes == 0` (empty view), return `true` immediately (no
>    bounds check). Otherwise require
>    `storage_size_bytes + storage_offset_bytes <= in.nbytes()`.
> Return `true` if all pass.

> [spec:et:def:copy-ops-util.torch.executor.check-cat-args-fn]
> bool check_cat_args( executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-cat-args-fn]
> Validates arguments for `cat` (concatenation of `tensors` along `dim` into
> `out`). Returns `true` on success; each check logs and returns `false` on
> failure. Steps:
> 1. Require `tensors.size() > 0` (non-empty list).
> 2. Find `ref_i`: scan `i` ascending and set `ref_i` to the index of the first
>    tensor with `numel() > 0`; if none is found `ref_i` stays `0`.
> 3. For each `i` in `[0, tensors.size())` ascending:
>    - `canCast(tensors[i].scalar_type(), out.scalar_type())` must be true —
>      every input dtype must be castable to the output dtype (PyTorch type
>      promotion rules).
>    - `tensors_have_same_dim_order(tensors[i], out)` must hold.
>    - If `tensors[i].numel() == 0`, skip the remaining shape checks for this
>      tensor (empty tensors have no shape constraint).
>    - `tensor_is_rank(tensors[ref_i], tensors[i].dim())`: input `i` must have
>      the same rank as the reference tensor.
>    - For each dimension `d` in `[0, tensors[i].dim())`, if `d != dim`, require
>      `tensors_have_same_size_at_dims(tensors[i], d, tensors[ref_i], d)` — all
>      non-concatenating dimensions must match the reference.
> 4. Range check on `dim`: require `tensors[ref_i].numel() == 0 ||
>    tensors[ref_i].dim() > dim`, and require `dim >= 0`.
> Return `true`.
>
> Note: `dim` is expected to already be a non-negative normalized dimension
> (the check rejects negative `dim`).

> [spec:et:def:copy-ops-util.torch.executor.check-diagonal-copy-args-fn]
> bool check_diagonal_copy_args( const Tensor& in, int64_t dim1, int64_t dim2, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-diagonal-copy-args-fn]
> Validates arguments for `diagonal_copy` over dimensions `dim1` and `dim2`.
> Returns `true` on success; each check logs and returns `false` on failure.
> Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_has_rank_greater_or_equal_to(in, 2)` — input must be at least 2-D.
> 3. `tensor_has_dim(in, dim1)` — `dim1` in range `[-in.dim(), in.dim())`.
> 4. `tensor_has_dim(in, dim2)` — `dim2` in range `[-in.dim(), in.dim())`.
> 5. Normalize: if `dim1 < 0`, `dim1 += nonzero_dim(in)`; if `dim2 < 0`,
>    `dim2 += nonzero_dim(in)` (`nonzero_dim(in)` is `in.dim()` for non-scalar
>    tensors, i.e. treats a 0-dim tensor as rank 1).
> 6. Require normalized `dim1 != dim2`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-expand-copy-args-fn]
> bool check_expand_copy_args( const Tensor& input, ArrayRef<int64_t> expand_sizes, bool implicit, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-expand-copy-args-fn]
> Validates arguments for `expand_copy`. `out` is unused for validation.
> Returns `true` on success; each check logs and returns `false` on failure.
> Steps:
> 1. Require `implicit == false` — the implicit-expand form is not implemented.
> 2. Require `expand_sizes.size() >= input.sizes().size()` — the number of
>    requested sizes must be at least the input rank.
> 3. Require `expand_sizes.size() <= kTensorDimensionLimit` — cannot exceed the
>    configured maximum tensor rank.
> 4. `tensors_have_same_dtype(input, out)`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-permute-copy-args-fn]
> bool check_permute_copy_args(const Tensor& in, IntArrayRef dims, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-permute-copy-args-fn]
> Validates arguments for `permute_copy` given a permutation `dims`. Returns
> `true` on success; each check logs and returns `false` on failure. Steps:
> 1. `tensor_is_rank(in, dims.size())` — `dims` length must equal input rank.
> 2. `tensors_have_same_dtype(in, out)`.
> 3. Build a boolean presence array `dim_exist` of length `kTensorDimensionLimit`
>    initialized to all-false. For each `i` in `[0, dims.size())` ascending:
>    - `tensor_has_dim(in, dims[i])` — `dims[i]` in `[-in.dim(), in.dim())`.
>    - Normalize: `dim = dims[i] >= 0 ? dims[i] : in.dim() + dims[i]`.
>    - Internal check `dim < kTensorDimensionLimit`.
>    - Require `dim_exist[dim] == false` (no duplicate dimensions), then set
>      `dim_exist[dim] = true`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-pixel-shuffle-args-fn]
> bool check_pixel_shuffle_args( const Tensor& in, int64_t upscale_factor, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-pixel-shuffle-args-fn]
> Validates arguments for `pixel_shuffle` with the given `upscale_factor`.
> Returns `true` on success; each check logs and returns `false` on failure.
> Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_has_rank_greater_or_equal_to(in, 3)`.
> 3. `tensor_has_rank_greater_or_equal_to(out, 3)`.
> 4. `upscale_factor > 0`.
> 5. `in.size(in.dim() - 3) % (upscale_factor * upscale_factor) == 0` — the
>    channel dimension (third from last) must be divisible by the square of the
>    upscale factor.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn]
> bool check_pixel_unshuffle_args( const Tensor& in, int64_t downscale_factor, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-pixel-unshuffle-args-fn]
> Validates arguments for `pixel_unshuffle` with the given `downscale_factor`.
> Returns `true` on success; each check logs and returns `false` on failure.
> Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_has_rank_greater_or_equal_to(in, 3)`.
> 3. `tensor_has_rank_greater_or_equal_to(out, 3)`.
> 4. `downscale_factor > 0`.
> 5. `in.size(in.dim() - 1) % downscale_factor == 0` — width (last dim)
>    divisible by the downscale factor.
> 6. `in.size(in.dim() - 2) % downscale_factor == 0` — height (second-to-last
>    dim) divisible by the downscale factor.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-select-copy-out-args-fn]
> bool check_select_copy_out_args( const Tensor& in, int64_t dim, int64_t index, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-select-copy-out-args-fn]
> Validates arguments for `select_copy` (selecting index `index` along `dim`).
> Returns `true` on success; each check logs and returns `false` on failure.
> Steps:
> 1. `tensor_has_rank_greater_or_equal_to(in, 1)` — input at least 1-D.
> 2. `tensor_has_dim(in, dim)` — `dim` in `[-in.dim(), in.dim())`.
> 3. `tensor_dim_has_index(in, dim, index)` — `index` is a valid index along
>    `dim` (in `[-in.size(dim), in.size(dim))`).
> 4. `tensors_have_same_dtype(in, out)`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-split-copy-args-fn]
> bool check_split_copy_args( const Tensor& input, int64_t split_size, int64_t dim, TensorList out)

> [spec:et:sem:copy-ops-util.torch.executor.check-split-copy-args-fn]
> Validates arguments for `split_copy` (splitting `input` along `dim` into
> equal `split_size` chunks producing the `out` tensor list). Returns `true` on
> success; each check logs and returns `false` on failure. Steps:
> 1. `input.dim() > 0`.
> 2. `dim >= 0 && dim < input.dim()` (non-negative, in range).
> 3. Let `dim_size = input.size(dim)`.
> 4. `split_size >= 0`.
> 5. `split_size > 0 || dim_size == 0` — a zero `split_size` is only allowed
>    when the split dimension is empty.
> 6. Determine expected number of outputs and the size of the final chunk
>    (`remainder`):
>    - If `split_size >= dim_size` (also covers `split_size == 0`, avoiding
>      division by zero): require `out.size() == 1` and set
>      `remainder = dim_size`.
>    - Else: `expected_out_len = (dim_size + split_size - 1) / split_size`
>      (ceil division); require `out.size() == expected_out_len`; set
>      `remainder = dim_size % split_size`, and if that is `0` set
>      `remainder = split_size`.
> 7. For each output `i` in `[0, out.size())` ascending:
>    - `out[i].scalar_type() == out[0].scalar_type()` — all outputs share the
>      dtype of the first output.
>    - `out[i].dim() == input.dim()` — same rank as input.
>    - For each dimension `d` in `[0, out[i].dim())`:
>      - If `d == dim`: for all outputs except the last (`i < out.size()-1`)
>        require `out[i].size(d) == split_size`; for the last output require
>        `out[i].size(d) == remainder`.
>      - Else: require `tensors_have_same_size_at_dims(out[i], d, input, d)`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-split-with-sizes-copy-args-fn]
> bool check_split_with_sizes_copy_args( const Tensor& in, executorch::aten::ArrayRef<int64_t> split_sizes, int64_t dim, TensorList out)

> [spec:et:sem:copy-ops-util.torch.executor.check-split-with-sizes-copy-args-fn]
> Validates arguments for `split_with_sizes_copy` (splitting `in` along `dim`
> into chunks of the given `split_sizes`). Returns `true` on success; each
> check logs and returns `false` on failure. Steps:
> 1. `tensor_has_rank_greater_or_equal_to(in, 1)`.
> 2. `tensor_has_dim(in, dim)`.
> 3. `split_sizes.size() == out.size()` — one output per split size.
> 4. Accumulate `sum` over `split_sizes` in ascending order: each
>    `split_sizes[i] >= 0`, adding into a signed 64-bit `sum`.
> 5. `sum == in.size(dim)` — the split sizes must partition the split dimension
>    exactly.
> Return `true`. (This function does not validate individual output shapes.)

> [spec:et:def:copy-ops-util.torch.executor.check-squeeze-copy-dim-args-fn]
> bool check_squeeze_copy_dim_args( const Tensor in, int64_t dim, const Tensor out)

> [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dim-args-fn]
> Validates arguments for `squeeze_copy` along a single `dim`. Returns `true`
> on success; each check logs and returns `false` on failure. Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_has_dim(in, dim)` — `dim` in `[-in.dim(), in.dim())`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-squeeze-copy-dims-args-fn]
> bool check_squeeze_copy_dims_args( const Tensor in, const executorch::aten::ArrayRef<int64_t> dims, const Tensor out)

> [spec:et:sem:copy-ops-util.torch.executor.check-squeeze-copy-dims-args-fn]
> Validates arguments for `squeeze_copy` along a set of `dims`. Returns `true`
> on success; each check logs and returns `false` on failure. Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. For each `i` in `[0, dims.size())` ascending:
>    - Normalize `dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i]`
>      (`nonzero_dim(in)` treats a 0-dim tensor as rank 1).
>    - `tensor_has_dim(in, dim)` — the (still possibly-negative-input) `dim`
>      must be a valid dimension.
>    - Inner loop over each `j` in `[0, dims.size())` with `j != i`: normalize
>      `dim_temp` the same way and require `dim != dim_temp` — no dimension may
>      appear twice in `dims`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-stack-args-fn]
> bool check_stack_args( executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-stack-args-fn]
> Validates arguments for `stack` (stacking `tensors` along a new dimension
> `dim` into `out`). Returns `true` on success; each check logs and returns
> `false` on failure. Steps:
> 1. Require `tensors.size() > 0`.
> 2. For each `i` in `[0, tensors.size())` ascending:
>    - `canCast(tensors[i].scalar_type(), out.scalar_type())` — every input
>      dtype must be castable to the output dtype.
>    - `tensor_is_rank(tensors[i], tensors[0].dim())` — same rank as the first
>      input.
>    - For each dimension `d` in `[0, tensors[i].dim())`, require
>      `tensors_have_same_size_at_dims(tensors[i], d, tensors[0], d)` — all
>      inputs must have identical shape (stack, unlike cat, requires full shape
>      match on every dimension).
> 3. Range check: `dim >= 0 && dim < tensors[0].dim() + 1` — the new-dimension
>    insertion position (a rank of `ndim_of_inputs + 1` in the output).
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-to-copy-args-fn]
> bool check_to_copy_args( const Tensor& input, bool non_blocking, std::optional<executorch::aten::MemoryFormat> memory_format, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-to-copy-args-fn]
> Validates arguments for `_to_copy`. `input` and `out` are unused for
> validation. Returns `true` on success; each check logs and returns `false`
> on failure. Steps:
> 1. `non_blocking == false` — only blocking (synchronous) data transfer is
>    supported.
> 2. `!memory_format.has_value() || memory_format.value() ==
>    MemoryFormat::Contiguous` — memory format must be unset or Contiguous.
> Return `true`. (No dtype constraint; casting between any input/output dtype
> is permitted.)

> [spec:et:def:copy-ops-util.torch.executor.check-to-dim-order-copy-args-fn]
> bool check__to_dim_order_copy_args( const Tensor& input, bool non_blocking, executorch::aten::OptionalArrayRef<int64_t> dim_order, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-to-dim-order-copy-args-fn]
> Validates arguments for `_to_dim_order_copy`. Returns `true` on success; each
> check logs and returns `false` on failure. Steps:
> 1. `non_blocking == false` — only blocking data transfer is supported.
> 2. If `dim_order` has a value (`dim_order_ref = dim_order.value()`):
>    - `dim_order_ref.size() == input.dim()` — dim order length equals input
>      rank.
>    - The dim order must be either channels-last
>      (`is_channels_last_dim_order(data, size)`) or contiguous
>      (`is_contiguous_dim_order(data, size)`); no other dim orders are
>      supported.
>    - `out.dim_order().size() == dim_order_ref.size()`, and for each `i` in
>      `[0, dim_order_ref.size())` the out tensor's dim order element must equal
>      `dim_order_ref[i]` (out must already carry exactly the requested dim
>      order).
> 3. Else (`dim_order` unset — preserve the input's dim order):
>    - `out.dim_order().size() == input.dim_order().size()`, and for each `i`
>      the out tensor's dim order element must equal the input's dim order
>      element `i`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-tril-args-fn]
> bool check_tril_args(const Tensor& in, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-tril-args-fn]
> Validates arguments for `tril` (lower-triangular). Returns `true` on success;
> each check logs and returns `false` on failure. Steps:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensor_has_rank_greater_or_equal_to(in, 2)` — at least 2-D.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-unbind-copy-args-fn]
> bool check_unbind_copy_args(const Tensor& in, int64_t dim, TensorList out)

> [spec:et:sem:copy-ops-util.torch.executor.check-unbind-copy-args-fn]
> Validates arguments for `unbind_copy` (removing `dim`, producing one output
> per slice). Returns `true` on success; each check logs and returns `false` on
> failure. Steps:
> 1. `in.dim() > 0` — input must have at least one dimension.
> 2. `dim_is_valid(dim, in.dim())` — `dim` in `[-in.dim(), in.dim())`.
> 3. Let `dim_size = in.size(dim)`. Require
>    `dim_size == out.size()` — the output list length must equal the size of
>    the unbind dimension.
> 4. For each output `i` in `[0, out.size())` ascending:
>    - `out[i].scalar_type() == out[0].scalar_type()` — all outputs share the
>      first output's dtype.
>    - `out[i].dim() == in.dim() - 1` — each output has one fewer dimension than
>      the input.
>    - Shape check: iterate `d` over `[0, in.dim())` with a separate output
>      counter `out_d` starting at 0; for each `d != dim` require
>      `out[i].size(out_d) == in.size(d)` and then increment `out_d`. (The
>      unbind dimension is skipped; remaining dimensions must match in order.)
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-unfold-copy-args-fn]
> bool check_unfold_copy_args( const Tensor& self, int64_t dim, int64_t size, int64_t step)

> [spec:et:sem:copy-ops-util.torch.executor.check-unfold-copy-args-fn]
> Validates arguments for `unfold_copy` (sliding window of `size` with `step`
> along `dim`). Returns `true` on success; each check logs and returns `false`
> on failure. Steps:
> 1. Normalize: if `dim < 0`, `dim += nonzero_dim(self)` (treats 0-dim as rank
>    1).
> 2. `tensor_has_dim(self, dim)` — normalized `dim` must be valid.
> 3. `size >= 0` — window size non-negative.
> 4. `size <= self.size(dim)` — window may not exceed the dimension's size.
> 5. `step > 0` — stride strictly positive.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-unsqueeze-copy-args-fn]
> bool check_unsqueeze_copy_args( const Tensor input, int64_t dim, const Tensor out)

> [spec:et:sem:copy-ops-util.torch.executor.check-unsqueeze-copy-args-fn]
> Validates arguments for `unsqueeze_copy` (inserting a size-1 dimension at
> `dim`). Returns `true` on success; each check logs and returns `false` on
> failure. Steps:
> 1. `dim >= 0` — only non-negative insertion positions are accepted.
> 2. `tensors_have_same_dtype(input, out)`.
> 3. `tensor_has_dim(out, dim)` — `dim` is valid in the (larger) output.
> 4. `input.dim() == out.dim() - 1` — output has exactly one more dimension.
> 5. For each dimension `d` in `[0, out.dim())`, with `dim_normalized = dim`
>    (and, defensively, `dim_normalized += out.dim()` if it were negative —
>    unreachable here since step 1 already required `dim >= 0`):
>    - If `d < dim_normalized`: require `input.size(d) == out.size(d)`.
>    - Else if `d > dim_normalized`: require `input.size(d-1) == out.size(d)`.
>    - Else (`d == dim_normalized`): require `out.size(d) == 1`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.check-view-copy-args-fn]
> bool check_view_copy_args( const Tensor& self, executorch::aten::ArrayRef<int64_t> size_int64_t, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.check-view-copy-args-fn]
> Validates arguments for `view_copy`. `size_int64_t` is the requested view
> shape (may contain a single `-1` for an inferred dimension). Returns `true`
> on success; each check logs and returns `false` on failure. Steps:
> 1. `size_int64_t.size() == out.sizes().size()` — requested rank equals out
>    rank.
> 2. `self.numel() == out.numel()` — total element count is preserved.
> 3. `tensors_have_same_dtype(self, out)`.
> 4. Iterate `i` over `[0, size_int64_t.size())` ascending, tracking a
>    `size_inferred` flag (initially false):
>    - If `size_int64_t[i] == -1`: require `!size_inferred` (at most one
>      inferred dimension), then set `size_inferred = true`.
>    - Require `((int64_t)out.sizes()[i] == size_int64_t[i]) ||
>      (size_int64_t[i] == -1)` — each out dimension must match the requested
>      size unless the requested size is the inferred `-1`.
> Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.get-cat-out-target-size-fn]
> void get_cat_out_target_size( executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-cat-out-target-size-fn]
> Computes the output shape of `cat` into `out_sizes`/`out_ndim`. Assumes args
> already validated by
> `[spec:et:sem:copy-ops-util.torch.executor.check-cat-args-fn]`. Steps:
> 1. Determine the reference tensor and the concatenated-dimension size in a
>    single ascending scan `i` over `tensors`:
>    - Maintain `cat_dim_size = 0`; if `tensors[i].numel() > 0`, add
>      `tensors[i].size(dim)` to `cat_dim_size`.
>    - Maintain `ref_i = 0`; set `ref_i = i` whenever
>      `tensors[i].dim() != 1 || tensors[i].numel() != 0`. (This deliberately
>      skips 1-D empty tensors, which are wildcards, and ends up pointing at the
>      last non-(1-D-empty) tensor.)
> 2. Set `*out_ndim = tensors[ref_i].dim()`.
> 3. For each dimension `d` in `[0, *out_ndim)`: set
>    `out_sizes[d] = cat_dim_size` if `d == dim`, otherwise
>    `out_sizes[d] = tensors[ref_i].size(d)`.

> [spec:et:def:copy-ops-util.torch.executor.get-diagonal-copy-out-target-size-fn]
> void get_diagonal_copy_out_target_size( const Tensor& in, int64_t offset, int64_t dim1, int64_t dim2, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-diagonal-copy-out-target-size-fn]
> Computes the output shape of `diagonal_copy` into `out_sizes`/`out_ndim`.
> Steps:
> 1. `*out_ndim = in.dim() - 1` (the diagonal collapses two dims into one that
>    is appended at the end).
> 2. Normalize: if `dim1 < 0`, `dim1 += nonzero_dim(in)`; if `dim2 < 0`,
>    `dim2 += nonzero_dim(in)`.
> 3. Compute `diagonal_size` (length of the diagonal), where `offset` shifts the
>    diagonal:
>    - If `offset >= 0`: if `in.size(dim2) <= offset` then `diagonal_size = 0`,
>      else `diagonal_size = min(in.size(dim1), in.size(dim2) - offset)`.
>    - Else (`offset < 0`): if `in.size(dim1) <= -offset` then
>      `diagonal_size = 0`, else `diagonal_size =
>      min(in.size(dim1) + offset, in.size(dim2))`.
>    (Minimums computed over `size_t`.)
> 4. Copy the surviving dimensions in order using a `shift` counter (starting
>    0): iterate `d` over `[0, in.dim())`; if `d == dim1 || d == dim2` increment
>    `shift`, else set `out_sizes[d - shift] = in.size(d)`.
> 5. Set the last output dimension `out_sizes[in.dim() - 2] = diagonal_size`.

> [spec:et:def:copy-ops-util.torch.executor.get-expand-copy-out-target-size-fn]
> bool get_expand_copy_out_target_size( executorch::aten::ArrayRef<executorch::aten::SizesType> self_sizes, executorch::aten::ArrayRef<int64_t> expand_sizes, executorch::aten::SizesType* output_sizes, size_t* output_rank)

> [spec:et:sem:copy-ops-util.torch.executor.get-expand-copy-out-target-size-fn]
> Computes the output shape of `expand_copy` into `output_sizes`/`output_rank`.
> `self_sizes` is the input shape; `expand_sizes` the requested (possibly longer)
> shape, where `-1` means "keep the corresponding input size". Returns `true`
> on success; a mismatch check logs and returns `false`. Steps:
> 1. Initialize `*output_rank = 0`. Let `j = expand_sizes.size()`.
> 2. Right-align the input against the expand shape: for `i = self_sizes.size()`
>    counting down while `i > 0 && j > 0`, decrement both `i` and `j` each
>    iteration and:
>    - Set `output_sizes[j] = expand_sizes[j]`.
>    - If `expand_sizes[j] == -1`, override with `output_sizes[j] =
>      self_sizes[i]` (keep the existing size).
>    - Else if `self_sizes[i] != 1`, require `expand_sizes[j] == self_sizes[i]`
>      — a non-singleton input dimension can only "expand" to its own size.
>      (Singleton input dimensions may expand to any size.)
> 3. Handle the remaining leading expand dimensions: while `j > 0`, decrement
>    `j`, set `output_sizes[j] = expand_sizes[j]`, and require
>    `expand_sizes[j] >= 0` (leading, newly-created dimensions cannot be `-1`).
> 4. Set `*output_rank = expand_sizes.size()`. Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn]
> void get_permute_copy_out_target_size( const Tensor& in, IntArrayRef dims, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn]
> Computes the output shape of `permute_copy` into `out_sizes`/`out_ndim`.
> Steps:
> 1. `*out_ndim = in.dim()`.
> 2. For each `i` in `[0, in.dim())` ascending, set `out_sizes[i] =
>    in.size(d)` where `d = dims[i] >= 0 ? dims[i] : dims[i] + in.dim()` (the
>    normalized source dimension). The output's `i`-th dimension is the input's
>    `dims[i]`-th dimension.

> [spec:et:def:copy-ops-util.torch.executor.get-pixel-shuffle-out-target-size-fn]
> bool get_pixel_shuffle_out_target_size( const Tensor& in, int64_t upscale_factor, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-pixel-shuffle-out-target-size-fn]
> Computes the output shape of `pixel_shuffle` into `out_sizes`/`out_ndim`.
> Returns `true` on success; an overflow-guard check logs and returns `false`.
> Steps:
> 1. Require `upscale_factor < 32768` (prevents signed overflow when squaring).
> 2. `*out_ndim = in.dim()`. Let `f = upscale_factor` (cast to `SizesType`).
> 3. Copy leading dimensions: for `i` from 0 while `i < in.dim() - 3`, set
>    `out_sizes[i] = in.size(i)`.
> 4. The trailing three dimensions are (channel, height, width). With `i` now at
>    `in.dim() - 3`:
>    - `out_sizes[i] = in.size(i) / (f * f)` (channel divided by factor²), then
>      `i++`.
>    - `out_sizes[i] = in.size(i) * f` (height times factor), then `i++`.
>    - `out_sizes[i] = in.size(i) * f` (width times factor).
> 5. Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn]
> void get_pixel_unshuffle_out_target_size( const Tensor& in, int64_t downscale_factor, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-pixel-unshuffle-out-target-size-fn]
> Computes the output shape of `pixel_unshuffle` into `out_sizes`/`out_ndim`
> (the inverse of pixel_shuffle). Steps:
> 1. `*out_ndim = in.dim()`. Let `f = downscale_factor` (cast to `SizesType`).
> 2. Copy leading dimensions: for `i` from 0 while `i < in.dim() - 3`, set
>    `out_sizes[i] = in.size(i)`.
> 3. Trailing three dimensions (channel, height, width). With `i` at
>    `in.dim() - 3`:
>    - `out_sizes[i] = in.size(i) * (f * f)` (channel times factor²), then
>      `i++`.
>    - `out_sizes[i] = in.size(i) / f` (height divided by factor), then `i++`.
>    - `out_sizes[i] = in.size(i) / f` (width divided by factor).
> No overflow guard and no return value (void).

> [spec:et:def:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn]
> void get_select_copy_out_target_size( const Tensor& in, int64_t dim, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-select-copy-out-target-size-fn]
> Computes the output shape of `select_copy` into `out_sizes`/`out_ndim` (the
> selected dimension `dim` is removed). Steps:
> 1. `*out_ndim = in.dim() - 1`.
> 2. For each `d` in `[0, in.dim() - 1)` ascending: if `d < dim` set
>    `out_sizes[d] = in.size(d)`, else set `out_sizes[d] = in.size(d + 1)`
>    (dimensions after the removed one shift down by one). `dim` is assumed a
>    non-negative normalized dimension.

> [spec:et:def:copy-ops-util.torch.executor.get-split-with-sizes-copy-out-target-size-fn]
> void get_split_with_sizes_copy_out_target_size( const Tensor& in, int64_t split_size, int64_t dim, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-split-with-sizes-copy-out-target-size-fn]
> Computes the shape of one split-with-sizes output chunk into
> `out_sizes`/`out_ndim`. The chunk has the input shape except the split
> dimension is replaced by `split_size` (this chunk's size along `dim`). Steps:
> 1. `*out_ndim = in.dim()`.
> 2. For each `d` in `[0, in.dim())` ascending, set `out_sizes[d] = in.size(d)`.
> 3. Overwrite `out_sizes[dim] = split_size`.

> [spec:et:def:copy-ops-util.torch.executor.get-squeeze-copy-dim-out-target-size-fn]
> void get_squeeze_copy_dim_out_target_size( const Tensor in, int64_t dim, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dim-out-target-size-fn]
> Computes the output shape of `squeeze_copy` along a single `dim` into
> `out_sizes`/`out_ndim`. `dim` is assumed non-negative/normalized. Steps:
> 1. If `in.dim() == 0`: set `*out_ndim = 0` and return (0-dim input stays
>    0-dim).
> 2. Otherwise: if `in.size(dim) == 1` set `*out_ndim = in.dim() - 1`, else set
>    `*out_ndim = in.dim()` (the dimension is removed only if it is a
>    singleton).
> 3. Copy sizes with an output counter `out_d` starting at 0: for each `in_d`
>    in `[0, in.dim())`, if `in_d != dim || in.size(in_d) != 1` then set
>    `out_sizes[out_d] = in.size(in_d)` and increment `out_d`. (The target
>    dimension is dropped only when its size is 1.)

> [spec:et:def:copy-ops-util.torch.executor.get-squeeze-copy-dims-out-target-size-fn]
> void get_squeeze_copy_dims_out_target_size( const Tensor in, const executorch::aten::ArrayRef<int64_t> dims, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-squeeze-copy-dims-out-target-size-fn]
> Computes the output shape of `squeeze_copy` along a set of `dims` into
> `out_sizes`/`out_ndim`. Steps:
> 1. If `in.dim() == 0`: set `*out_ndim = 0` and return.
> 2. Count `dims_to_remove`: for each `i` in `[0, dims.size())`, normalize
>    `dim = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i]`; if
>    `in.size(dim) == 1` increment `dims_to_remove`. Set
>    `*out_ndim = in.dim() - dims_to_remove`. (Non-singleton dims in `dims` are
>    not removed.)
> 3. Copy sizes with output counter `out_d` starting at 0: for each `in_d` in
>    `[0, in.dim())`, determine `in_d_in_dims` by scanning `dims` (normalizing
>    each as above and testing equality with `in_d`). Keep the dimension unless
>    it is both in `dims` and has size 1: if `!in_d_in_dims || in.size(in_d) !=
>    1`, set `out_sizes[out_d] = in.size(in_d)` and increment `out_d`.

> [spec:et:def:copy-ops-util.torch.executor.get-stack-out-target-size-fn]
> void get_stack_out_target_size( executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-stack-out-target-size-fn]
> Computes the output shape of `stack` into `out_sizes`/`out_ndim` (a new
> dimension of size `tensors.size()` is inserted at position `dim`). Steps:
> 1. `*out_ndim = tensors[0].dim() + 1`.
> 2. For each output dimension `d` in `[0, *out_ndim)` (with `d_ = (int64_t)d`):
>    - If `d_ < dim`: `out_sizes[d_] = tensors[0].size(d_)`.
>    - Else if `d_ == dim`: `out_sizes[d_] = tensors.size()` (the number of
>      stacked tensors).
>    - Else (`d_ > dim`): `out_sizes[d_] = tensors[0].size(d_ - 1)` (dimensions
>      after the inserted axis shift up by one).

> [spec:et:def:copy-ops-util.torch.executor.get-unfold-copy-out-target-size-fn]
> void get_unfold_copy_out_target_size( const Tensor& self, int64_t dim, int64_t size, int64_t step, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:copy-ops-util.torch.executor.get-unfold-copy-out-target-size-fn]
> Computes the output shape of `unfold_copy` into `out_sizes`/`out_ndim`. The
> output has one more dimension than the input (a trailing window dimension of
> size `size`). `dim` is assumed non-negative/normalized. Steps:
> 1. For each `i` in `[0, self.dim())` ascending, set `out_sizes[i] =
>    self.size(i)`.
> 2. Replace the unfold dimension: `out_sizes[dim] =
>    (self.size(dim) - size + step) / step` — the number of windows (integer
>    division).
> 3. Append the window dimension: `out_sizes[self.dim()] = size`.
> 4. `*out_ndim = self.dim() + 1`.

> [spec:et:def:copy-ops-util.torch.executor.get-view-as-real-copy-out-target-size-fn]
> void get_view_as_real_copy_out_target_size( const Tensor& self, executorch::aten::SizesType* out_sizes)

> [spec:et:sem:copy-ops-util.torch.executor.get-view-as-real-copy-out-target-size-fn]
> Computes the output shape of `view_as_real_copy` into `out_sizes` (this
> variant does not report a rank; the caller knows the rank is `self.dim()+1`).
> A trailing dimension of size 2 (real, imaginary) is appended. Steps:
> 1. For each `i` in `[0, self.dim())` ascending, set `out_sizes[i] =
>    self.size(i)`.
> 2. Set `out_sizes[self.dim()] = 2`.

> [spec:et:def:copy-ops-util.torch.executor.get-view-copy-target-size-fn]
> bool get_view_copy_target_size( const Tensor input, executorch::aten::ArrayRef<int64_t> size_int64_t, int64_t dim, executorch::aten::SizesType* out_sizes)

> [spec:et:sem:copy-ops-util.torch.executor.get-view-copy-target-size-fn]
> Computes the output shape of `view_copy` into `out_sizes`, resolving a single
> inferred (`-1`) dimension. `dim` is the requested output rank. Returns `true`
> on success; checks log and return `false` on failure. Steps:
> 1. Initialize `out_numels_without_minus_1 = 1` and `minus_1_dim = -1`.
> 2. Require `size_int64_t.size() == dim` (requested size list length equals the
>    output rank).
> 3. For each `i` in `[0, dim)` ascending:
>    - If `size_int64_t[i] != -1`: set `out_sizes[i] = size_int64_t[i]`
>      (cast to `SizesType`) and multiply `out_numels_without_minus_1` by
>      `size_int64_t[i]`.
>    - Else: require `minus_1_dim == -1` (at most one inferred dimension), then
>      record `minus_1_dim = i`.
> 4. If `minus_1_dim >= 0`, set `out_sizes[minus_1_dim] = input.numel() /
>    out_numels_without_minus_1` (the inferred size divides the remaining
>    element count).
> 5. Return `true`.

> [spec:et:def:copy-ops-util.torch.executor.to-dim-order-copy-impl-fn]
> void _to_dim_order_copy_impl(const Tensor& self, Tensor& out)

> [spec:et:sem:copy-ops-util.torch.executor.to-dim-order-copy-impl-fn]
> Copies and casts every element of `self` into `out` while respecting each
> tensor's own dim order (strides). Template-typed by input element type
> `SELF_CTYPE` and output element type `OUT_CTYPE`. Steps:
> 1. Obtain raw data pointers `self_data` (as `SELF_CTYPE*`) and `out_data`
>    (as `OUT_CTYPE*`).
> 2. Iterate flat element indices by driving a
>    `BroadcastIndexesRange<2, support_noncontiguous_input_tensors=true>`
>    constructed as `(/*dummy output*/ self, self, out)` — the input `self` is
>    reused as the shape reference (dummy output) and `self`, `out` are the two
>    "inputs"; enabling `support_noncontiguous_input_tensors` makes the range
>    honor each tensor's strides/dim order rather than assuming contiguity.
>    See `[spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]`
>    for how the strided flat indices are advanced. Each iteration yields
>    `(unused_index, self_data_index, out_data_index)`.
> 3. For each yielded pair, write `out_data[out_data_index] =
>    static_cast<OUT_CTYPE>(self_data[self_data_index])` — a plain C++ static
>    cast performing the dtype conversion (e.g. float↔int truncation toward
>    zero, integer wraparound, float widening/narrowing per C++ conversion
>    rules). `self` and `out` are assumed to have the same logical shape and
>    element count; no resizing is done here.
