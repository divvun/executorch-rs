# kernels/portable/cpu/util/repeat_util.cpp

> [spec:et:def:repeat-util.torch.executor.check-repeat-args-fn]
> bool check_repeat_args( Tensor self, executorch::aten::ArrayRef<int64_t> repeats, Tensor& out)

> [spec:et:sem:repeat-util.torch.executor.check-repeat-args-fn]
> Validates the arguments of `repeat_tensor` (note: `self` is passed by value).
> Returns `true` only if every check below passes; on the first failing check it
> logs a message and returns `false` (via `ET_CHECK_OR_RETURN_FALSE` /
> `ET_LOG_AND_RETURN_IF_FALSE`), performing no output mutation.
>
> Checks, in order:
> 1. `repeats.size() >= self.dim()`: the repeats list must have at least as many
>    entries as `self` has dimensions.
> 2. Every element of `repeats` is non-negative (`repeat >= 0`); a single
>    negative entry fails ("Trying to create tensor with negative dimension").
> 3. `out.dim() == repeats.size()`: the output rank must exactly equal the number
>    of repeat entries.
> 4. `out.dim() <= kTensorDimensionLimit` (only `out` is checked, since `out.dim()`
>    is >= `self.dim()`).
> 5. `out` and `self` have the same dtype (`tensors_have_same_dtype`).
> 6. Shape consistency: build `reformat_self_size` of length `out.dim()` by
>    left-padding `self`'s sizes with 1s. Concretely, for `i` in
>    `[0, out.dim() - self.dim())` set `reformat_self_size[i] = 1`, and for the
>    trailing `self.dim()` positions copy `self`'s sizes aligned to the right:
>    `reformat_self_size[out.dim()-1-i] = self.size(self.dim()-1-i)` for `i` in
>    `[0, self.dim())`. Then for every output dimension `i` in `[0, repeats.size())`
>    require `reformat_self_size[i] * repeats[i] == out.size(i)`; any mismatch fails.
>
> A zero-dim `self` (`self.dim() == 0`) skips the copy loop in step 6's second half
> so all `reformat_self_size` entries are 1, and each `out.size(i)` must equal
> `repeats[i]`.

> [spec:et:def:repeat-util.torch.executor.compute-access-offset-fn]
> size_t compute_access_offset( const size_t* indices, const size_t* strides, size_t num_entries)

> [spec:et:sem:repeat-util.torch.executor.compute-access-offset-fn]
> Computes a linear byte offset into a tensor given per-dimension `indices` and
> per-dimension `strides` (both arrays of length `num_entries`). Returns
> `sum over i of indices[i] * strides[i]`, iterating `i` from `num_entries - 1`
> down to `0` (iteration order does not affect the sum). `strides` here are byte
> strides, so the result is a byte offset. Both arrays are assumed to have length
> `num_entries`.

> [spec:et:def:repeat-util.torch.executor.repeat-internal-fn]
> void repeat_internal( const Tensor& self, Tensor& out, size_t in_offset, size_t out_offset, const size_t* strides)

> [spec:et:sem:repeat-util.torch.executor.repeat-internal-fn]
> Replicates one contiguous innermost-row of `self` across the output tensor
> `out`, starting the source read at byte offset `in_offset` in `self` and the
> destination writes anchored at byte offset `out_offset` in `out`. `strides` is
> the array of byte strides for the output tensor (length at least `self_dim`).
>
> Steps:
> 1. `src = self.const_data_ptr<char>() + in_offset`;
>    `dest = out.mutable_data_ptr<char>() + out_offset`.
> 2. Treat a zero-dim `self` as one-dim with size `{1}`: `self_dim = max(self.dim(), 1)`
>    and `self_size` is `self.sizes()` (or the single-element `{1}` for zero-dim).
> 3. `num_bytes = self_size[self_dim-1] * out.element_size()` — the byte length of
>    the innermost row. If `num_bytes == 0` return immediately (nothing to copy).
> 4. Set up an n-D `slots` counter of length `self_dim`, all zeroed, and an
>    increment array `incr[i] = self_size[i]`. `start = out.dim() - self_dim` is the
>    offset that aligns `self`'s dimensions to the trailing dimensions of `out`.
> 5. Iterate: while `slots[0] != out.size(start)`:
>    a. compute `offset = compute_access_offset(slots, strides, self_dim)` per
>       `[spec:et:sem:repeat-util.torch.executor.compute-access-offset-fn]` and
>       `memcpy(dest + offset, src, num_bytes)` — copies the whole innermost row.
>    b. advance the counter: add `incr[self_dim-1]` to `slots[self_dim-1]`; then
>       while the current `slots[index] == out.size(start+index)`, if `index==0`
>       break, else zero `slots[index]`, decrement `index`, and add `incr[index]`
>       to `slots[index]`. Reset `index = self_dim-1` each outer iteration.
> This visits every valid tiled position of the `self` block along the trailing
> `self_dim` dimensions of `out`, stepping the innermost dimension by whole rows
> (`incr` = self size in that dim) so copies land on repeat boundaries.

> [spec:et:def:repeat-util.torch.executor.repeat-tensor-fn]
> Error repeat_tensor( const Tensor& self, executorch::aten::ArrayRef<int64_t> repeats, Tensor& out)

> [spec:et:sem:repeat-util.torch.executor.repeat-tensor-fn]
> Repeats `self` along each dimension per `repeats`, writing into the
> pre-resized `out`. Returns `Error` (`Ok` on success).
>
> Steps:
> 1. Validate args via `check_repeat_args(self, repeats, out)` per
>    `[spec:et:sem:repeat-util.torch.executor.check-repeat-args-fn]`; on failure
>    return `Error::InvalidArgument`.
> 2. If `out.numel() == 0`, return `Error::Ok` (nothing to repeat).
> 3. `element_size = out.element_size()`.
> 4. Special case `out.numel() == 1`: `memcpy` `element_size` bytes from `self`'s
>    data to `out`'s data and return `Error::Ok` (handles zero-dim/scalar cleanly).
> 5. Treat zero-dim `self` as one-dim size `{1}`: `self_dim = max(self.dim(), 1)`,
>    `self_size` = `self.sizes()` (or `{1}`).
> 6. Compute the output's per-dimension byte `strides` for the trailing `self_dim`
>    dimensions: with `start = out.dim() - self_dim` and `accum_offset` starting at
>    `element_size`, for `i` from `self_dim-1` down to `0`: `strides[i] = accum_offset`,
>    then `accum_offset *= out.size(start+i)`. After the loop `accum_offset` equals
>    the byte size of one full copy of the (right-aligned) `self` block within `out`.
> 7. Iterate over every innermost row of `self` (an n-D `slots` counter of length
>    `self_dim`, `limits[i] = self_size[i]`, `in_incr = self_size[self_dim-1]*element_size`,
>    then `limits[self_dim-1] = 1` so the innermost dim is handled as a whole-row
>    memcpy). Starting `in_offset = 0`: while `slots[0] != limits[0]`:
>    a. `out_offset = compute_access_offset(slots, strides, self_dim)`.
>    b. call `repeat_internal(self, out, in_offset, out_offset, strides)` per
>       `[spec:et:sem:repeat-util.torch.executor.repeat-internal-fn]` to tile that
>       row across the trailing `self_dim` output dimensions.
>    c. increment `slots` (odometer style: `slots[index]++`; while
>       `slots[index]==limits[index]`, if `index==0` break else zero it, decrement
>       `index`, `slots[index]++`; reset `index=self_dim-1`), and advance
>       `in_offset += in_incr`.
>    This copies the entire `self` tensor tiled across its own `self_dim` trailing
>    output dimensions.
> 8. Handle the remaining leading `out.dim() - self_dim` dimensions by whole-block
>    memcpy: `src = out.const_data_ptr<char>()` (origin), `dest = out data + accum_offset`.
>    For `i` from `start-1` down to `0`: repeat the already-filled block
>    `repeats[i]-1` more times (`for j in [0, repeats[i]-1): memcpy(dest, src, accum_offset); dest += accum_offset;`),
>    then grow `accum_offset *= out.size(i)`.
> 9. Return `Error::Ok`.
