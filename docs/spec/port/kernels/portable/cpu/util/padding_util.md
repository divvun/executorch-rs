# kernels/portable/cpu/util/padding_util.cpp, kernels/portable/cpu/util/padding_util.h

> [spec:et:def:padding-util.torch.executor.check-padding-args-fn]
> bool check_padding_args( int64_t n, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out, bool reflection)

> [spec:et:sem:padding-util.torch.executor.check-padding-args-fn]
> Validates arguments for an n-dimensional pad op (n = 1, 2, or 3). Signature
> `check_padding_args(n, in, padding, out, reflection = false)` where `padding`
> is an `ArrayRef<int64_t>` and `reflection` selects reflection-pad-specific
> constraints. Returns `bool`. Each check uses `ET_LOG_AND_RETURN_IF_FALSE`
> (logs and returns `false` on first failure); returns `true` if all pass.
>
> Checks in order:
> 1. `padding.size() == 2 * n` (a low/high pad pair per padded dim).
> 2. `in.dim() == n + 1 || in.dim() == n + 2` (an optional batch/channel-style
>    leading dim is allowed: e.g. for n=2, rank 3 or 4).
> 3. `in` and `out` have the same dtype.
> 4. For each `i` in `[1, n]` (1-based, counting from the trailing dim inward):
>    - The padded output extent is non-negative:
>      `in.size(in.dim() - i) + padding[2*i - 2] + padding[2*i - 1] >= 0`,
>      where `padding[2*i - 2]` is the "low"/left pad and `padding[2*i - 1]` the
>      "high"/right pad for that dim. (Padding values may be negative, i.e.
>      cropping, so long as the result stays non-negative.)
>    - If `reflection` is true, additionally each pad must be strictly less than
>      the corresponding input extent:
>      `padding[2*i - 2] < in.size(in.dim() - i)` AND
>      `padding[2*i - 1] < in.size(in.dim() - i)` (reflection padding cannot
>      reflect past the tensor edge).
>
> `padding` is indexed innermost-dim-first: `padding[0],padding[1]` are the
> low/high pads of the last dim, `padding[2],padding[3]` for the second-to-last,
> etc. Pure validation predicate; writes no output.

> [spec:et:def:padding-util.torch.executor.get-padding-out-target-size-fn]
> void get_padding_out_target_size( int64_t n, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]
> Computes the target output shape for an n-dimensional pad op. Signature
> `get_padding_out_target_size(n, in, padding, out_sizes, out_ndim)`;
> `out_sizes` is a caller buffer, `out_ndim` an out-param. No return value; no
> validation.
>
> Behavior:
> 1. Set `*out_ndim = in.dim()` (output keeps input rank).
> 2. For each `i` in `[0, in.dim())`: copy through `out_sizes[i] = in.size(i)`.
> 3. For each `i` in `[1, n]` (1-based, from the trailing dim inward): overwrite
>    the padded dims:
>    `out_sizes[in.dim() - i] = in.size(in.dim() - i) + padding[2*i - 2]
>    + padding[2*i - 1]`.
>
> So the leading (non-padded) dims are unchanged, and each of the trailing `n`
> dims grows (or shrinks, if pads are negative) by low-pad + high-pad. Padding
> is indexed innermost-first, matching
> `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`.

> [spec:et:def:padding-util.torch.executor.pad1d-fn]
> void pad1d( const PaddingIx& padding_ix, const Tensor& in, Tensor& out, executorch::aten::ArrayRef<int64_t> padding)

> [spec:et:sem:padding-util.torch.executor.pad1d-fn]
> Templated 1-D padding kernel `pad1d<CTYPE, PaddingIx>(padding_ix, in, out,
> padding)`. `padding_ix` is a callable `(out_index, in_size, pad_low) ->
> in_index` selecting where each output element reads from — pass
> `[spec:et:sem:padding-util.torch.executor.reflection-ix-fn]` for reflection
> pad or `[spec:et:sem:padding-util.torch.executor.replication-ix-fn]` for
> replication pad. No return value; writes into `out` in place.
>
> Setup (raw contiguous `CTYPE` buffers `in_data`/`out_data`):
> - `dim = in.dim() - 1` (the last/width dim).
> - `outer = getLeadingDims(out, dim)` — product of all dims before `dim`,
>   i.e. the number of independent rows.
> - `in_width = in.size(dim)`, `out_width = out.size(dim)`,
>   `pad_left = padding[0]`.
>
> Algorithm:
> For each `i` in `[0, outer)`:
> - `out_i_base = i * out_width`, `in_i_base = i * in_width`.
> - For each output column `w` in `[0, out_width)`:
>   - `in_w_idx = padding_ix(w, in_width, pad_left)`.
>   - Assert `0 <= in_w_idx < in_width` via `ET_CHECK` (a hard runtime abort if
>     violated — indicates invalid padding that should have been caught by
>     `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`).
>   - `out_data[out_i_base + w] = in_data[in_i_base + in_w_idx]` (pure copy; no
>     arithmetic, so dtype is irrelevant to the value).
>
> Iteration is row-major (outer rows, then width). Only `padding[0]` (the low/
> left pad) is used directly; the high pad is implicit in `out_width`.

> [spec:et:def:padding-util.torch.executor.pad2d-fn]
> void pad2d( const PaddingIx& padding_ix, const Tensor& in, Tensor& out, executorch::aten::ArrayRef<int64_t> padding)

> [spec:et:sem:padding-util.torch.executor.pad2d-fn]
> Templated 2-D padding kernel `pad2d<CTYPE, PaddingIx>(padding_ix, in, out,
> padding)`, analogous to `[spec:et:sem:padding-util.torch.executor.pad1d-fn]`
> but over height and width. `padding_ix` maps `(out_index, in_size, pad_low)
> -> in_index`. No return value; writes into `out` in place.
>
> Setup:
> - `dim = in.dim() - 2` (the height dim; width is `dim + 1`).
> - `outer = getLeadingDims(out, dim)` — number of independent (h,w) planes.
> - `in_height = in.size(dim)`, `in_width = in.size(dim+1)`,
>   `out_height = out.size(dim)`, `out_width = out.size(dim+1)`.
> - `pad_left = padding[0]` (width low pad), `pad_top = padding[2]` (height low
>   pad).
>
> Algorithm (row-major):
> For each `i` in `[0, outer)`:
> - `out_i_base = i * out_height * out_width`,
>   `in_i_base = i * in_height * in_width`.
> - For each output row `h` in `[0, out_height)`:
>   - `out_h_base = out_i_base + h * out_width`.
>   - `in_h_idx = padding_ix(h, in_height, pad_top)`; `ET_CHECK` that
>     `0 <= in_h_idx < in_height`.
>   - `in_h_base = in_i_base + in_h_idx * in_width`.
>   - For each output column `w` in `[0, out_width)`:
>     - `in_w_idx = padding_ix(w, in_width, pad_left)`; `ET_CHECK` that
>       `0 <= in_w_idx < in_width`.
>     - `out_data[out_h_base + w] = in_data[in_h_base + in_w_idx]` (pure copy).
>
> Uses only the low pads `padding[0]` and `padding[2]`; high pads are implicit
> in the output extents.

> [spec:et:def:padding-util.torch.executor.pad3d-fn]
> void pad3d( const PaddingIx& padding_ix, const Tensor& in, Tensor& out, executorch::aten::ArrayRef<int64_t> padding)

> [spec:et:sem:padding-util.torch.executor.pad3d-fn]
> Templated 3-D padding kernel `pad3d<CTYPE, PaddingIx>(padding_ix, in, out,
> padding)`, extending `[spec:et:sem:padding-util.torch.executor.pad2d-fn]` with
> a depth dimension. `padding_ix` maps `(out_index, in_size, pad_low) ->
> in_index`. No return value; writes into `out` in place.
>
> Setup:
> - `dim = in.dim() - 3` (depth dim; height is `dim+1`, width `dim+2`).
> - `outer = getLeadingDims(out, dim)` — number of independent (d,h,w) volumes.
> - `in_depth = in.size(dim)`, `in_height = in.size(dim+1)`,
>   `in_width = in.size(dim+2)`; likewise `out_depth`, `out_height`,
>   `out_width`.
> - `pad_left = padding[0]` (width low), `pad_top = padding[2]` (height low),
>   `pad_front = padding[4]` (depth low).
>
> Algorithm (row-major, depth outer):
> For each `i` in `[0, outer)`:
> - `out_i_base = i * out_depth * out_height * out_width`,
>   `in_i_base = i * in_depth * in_height * in_width`.
> - For each output depth `d` in `[0, out_depth)`:
>   - `out_d_base = out_i_base + d * out_height * out_width`.
>   - `in_d_idx = padding_ix(d, in_depth, pad_front)`; `ET_CHECK`
>     `0 <= in_d_idx < in_depth`.
>   - `in_d_base = in_i_base + in_d_idx * in_height * in_width`.
>   - For each output height `h` in `[0, out_height)`:
>     - `out_h_base = out_d_base + h * out_width`.
>     - `in_h_idx = padding_ix(h, in_height, pad_top)`; `ET_CHECK`
>       `0 <= in_h_idx < in_height`.
>     - `in_h_base = in_d_base + in_h_idx * in_width`.
>     - For each output width `w` in `[0, out_width)`:
>       - `in_w_idx = padding_ix(w, in_width, pad_left)`; `ET_CHECK`
>         `0 <= in_w_idx < in_width`.
>       - `out_data[out_h_base + w] = in_data[in_h_base + in_w_idx]` (pure copy).
>
> Uses the low pads `padding[0]`, `padding[2]`, `padding[4]`; high pads are
> implicit in the output extents.

> [spec:et:def:padding-util.torch.executor.reflection-ix-fn]
> inline int64_t reflection_ix(int64_t j, int64_t size, int64_t pad)

> [spec:et:sem:padding-util.torch.executor.reflection-ix-fn]
> `reflection_ix(j, size, pad)`: maps an output index `j` (in `[0, size + pad +
> high_pad)`) to the source input index for reflection padding, where `size` is
> the input extent along the dim and `pad` is the low/left pad. Returns
> `int64_t`.
>
> Piecewise:
> - If `j < pad` (in the low-pad region): return `pad - j` (reflect across the
>   first input element; the boundary element itself at input index 0 is not
>   duplicated — index 1 maps to the position just inside).
> - Else if `pad <= j < size + pad` (inside the copied input region): return
>   `j - pad`.
> - Else (`j >= size + pad`, in the high-pad region): return
>   `2*size + pad - j - 2` (reflect across the last input element).
>
> This is the mirror-without-edge-repeat reflection used by reflection-pad ops.
> The caller (`pad1d`/`pad2d`/`pad3d`) asserts the result lies in
> `[0, size)`; that holds only when `pad < size` for both low and high pads,
> which `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`
> enforces when `reflection` is set.

> [spec:et:def:padding-util.torch.executor.replication-ix-fn]
> inline int64_t replication_ix(int64_t j, int64_t size, int64_t pad)

> [spec:et:sem:padding-util.torch.executor.replication-ix-fn]
> `replication_ix(j, size, pad)`: maps an output index `j` to the source input
> index for replication (edge-clamp) padding, where `size` is the input extent
> and `pad` the low/left pad. Returns `int64_t`.
>
> Piecewise:
> - If `j < pad` (low-pad region): return `0` (clamp to the first input
>   element).
> - Else if `pad <= j < size + pad` (inside the copied input region): return
>   `j - pad`.
> - Else (`j >= size + pad`, high-pad region): return `size - 1` (clamp to the
>   last input element).
>
> This produces the edge-value-repeat behavior of replication-pad ops. For any
> `size >= 1` the result is always in `[0, size)`, so the `ET_CHECK` bounds
> assertion in the pad kernels always holds.

