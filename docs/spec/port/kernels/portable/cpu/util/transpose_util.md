# kernels/portable/cpu/util/transpose_util.h

> [spec:et:def:transpose-util.torch.executor.check-t-copy-args-fn]
> inline bool check_t_copy_args(const Tensor& in, Tensor& out)

> [spec:et:sem:transpose-util.torch.executor.check-t-copy-args-fn]
> Validates arguments for `t_copy` (the 0/1/2-D transpose, `Tensor.t()`).
> Returns `true` if all checks pass; on first failure logs and returns `false`.
> Checks: (1) `tensors_have_same_dtype(in, out)`; (2)
> `tensor_has_rank_smaller_or_equal_to(in, 2)` — `in.dim() <= 2` (t() only supports
> up to 2 dimensions).

> [spec:et:def:transpose-util.torch.executor.check-transpose-copy-args-fn]
> inline bool check_transpose_copy_args( const Tensor& in, int64_t dim0, int64_t dim1, Tensor& out)

> [spec:et:sem:transpose-util.torch.executor.check-transpose-copy-args-fn]
> Validates arguments for `transpose_copy` (swap `dim0` and `dim1`). Returns
> `true` if all checks pass; on first failure logs and returns `false`. Checks:
> (1) `tensors_have_same_dtype(in, out)`; (2) `tensor_has_dim(in, dim0)`;
> (3) `tensor_has_dim(in, dim1)` — both dims (possibly negative) must be valid
> dimensions of `in`.

> [spec:et:def:transpose-util.torch.executor.get-transpose-out-target-size-fn]
> inline void get_transpose_out_target_size( const Tensor& in, SizesType dim0, SizesType dim1, SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:transpose-util.torch.executor.get-transpose-out-target-size-fn]
> Computes the output shape for a transpose of `in` swapping `dim0` and `dim1`
> (both assumed already non-negative). Sets `*out_ndim = in.dim()`. If
> `in.dim() == 0`, returns immediately (scalar has no sizes to write). Otherwise
> copies `out_sizes[i] = in.size(i)` for all `i`, then swaps: `out_sizes[dim0] = in.size(dim1)`
> and `out_sizes[dim1] = in.size(dim0)`.

> [spec:et:def:transpose-util.torch.executor.increment-index-and-offset-fn]
> inline void increment_index_and_offset( size_t* index, const SizesType* new_sizes, const StridesType* new_strides, const ArrayRef<size_t> non_one_indices, size_t& offset)

> [spec:et:sem:transpose-util.torch.executor.increment-index-and-offset-fn]
> Advances an N-dimensional odometer counter `index` (output-tensor coordinates)
> by one in row-major order, and updates the corresponding linear `offset` into
> the input tensor's memory (using the transposed `new_strides`). Only the
> dimensions listed in `non_one_indices` (dimensions whose output size is `> 1`,
> in increasing output-dimension order) are considered, which skips size-1
> dimensions.
>
> Iterating `j` from `non_one_indices.size()` down to `1`, let `i = non_one_indices[j-1]`
> (i.e. process the highest non-1 output dimension first):
> 1. `index[i]++` and `offset += new_strides[i]`.
> 2. If `index[i] == new_sizes[i]` (carry): `offset -= new_sizes[i] * new_strides[i]`
>    (undo the full wrap for that dim), set `index[i] = 0`, and continue to the next
>    (more significant) non-1 dimension.
> 3. Otherwise (no carry): return.
> A carry cannot occur at the most-significant non-1 index within a valid iteration
> (guaranteed by the caller's element-count bound), so `offset` stays valid. Net
> effect: `offset` walks the input in output-tensor traversal order.

> [spec:et:def:transpose-util.torch.executor.transpose-tensors-fn]
> void transpose_tensors( const Tensor& a, int64_t dim0, int64_t dim1, Tensor& out)

> [spec:et:sem:transpose-util.torch.executor.transpose-tensors-fn]
> Template on element type `T`. Writes into `out` the transpose of `a` with
> dimensions `dim0` and `dim1` swapped. `out` is assumed already correctly sized
> (see `[spec:et:sem:transpose-util.torch.executor.get-transpose-out-target-size-fn]`)
> and `dim0`/`dim1` already normalized to non-negative.
>
> Steps:
> 1. `dim = a.dim()`, `data_a = a.const_data_ptr<T>()`, `data_out = out.mutable_data_ptr<T>()`.
> 2. Build `new_sizes` and `new_strides` describing how to walk `a` in output order.
>    If `dim != 0`: copy `a.strides()` into `new_strides` and `a.sizes()` into
>    `new_sizes` (first `dim` entries), then swap `new_sizes[dim0] <-> new_sizes[dim1]`
>    and `new_strides[dim0] <-> new_strides[dim1]`. So `new_sizes` are the output
>    sizes and `new_strides[k]` is the input stride to advance along output dimension
>    `k`. (For `dim == 0`, a scalar, these arrays are left unused.) `out_index`
>    (output coordinate odometer) is zero-initialized.
> 3. Collect `non_1_dim_indices`: the output dimensions `cur_dim` in increasing order
>    with `new_sizes[cur_dim] != 1`, stored as an `ArrayRef` `indices`.
> 4. Walk every element: `a_offset = 0`; for `out_offset` in `[0, a.numel())`:
>    write `data_out[out_offset] = data_a[a_offset]`, then advance via
>    `increment_index_and_offset(out_index, new_sizes, new_strides, indices, a_offset)`
>    per `[spec:et:sem:transpose-util.torch.executor.increment-index-and-offset-fn]`.
>    Output is written contiguously in output order; each `a_offset` reads the input
>    element at the transposed position. For `a.numel() == 1` (including scalar) a
>    single element is copied. Values are copied bitwise as `T` (no conversion).
