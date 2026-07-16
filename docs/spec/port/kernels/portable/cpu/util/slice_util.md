# kernels/portable/cpu/util/slice_util.cpp

> [spec:et:def:slice-util.torch.executor.adjust-slice-indices-fn]
> int64_t adjust_slice_indices( int64_t dim_length, int64_t* start, int64_t* end, int64_t step)

> [spec:et:sem:slice-util.torch.executor.adjust-slice-indices-fn]
> Clamps a Python-style `[start, end)` slice against a dimension of length
> `dim_length` and returns the resulting number of output values. `start` and
> `end` are in/out pointers mutated in place to their normalized C++ values;
> `step` is the (positive) stride.
>
> Steps:
> 1. Convert negatives to Python offsets: if `*start < 0`, `*start += dim_length`;
>    if `*end < 0`, `*end += dim_length`.
> 2. Floor at 0: if still `< 0`, set to `0` (start/end before the beginning).
> 3. Cap at `dim_length`: if `> dim_length`, set to `dim_length` (start/end past
>    the end; note the cap is `dim_length`, not `dim_length-1`).
> 4. If the interval is empty or out of range — `*start >= dim_length` or
>    `*end <= 0` or `*start >= *end` — return `num_values = 0`.
> 5. Otherwise return `num_values = (*end - 1 - *start) / step + 1` (integer
>    division; the count of indices `start, start+step, ...` that are `< end`).
> `start` and `end` retain their normalized/clamped values on return.

> [spec:et:def:slice-util.torch.executor.check-narrow-copy-args-fn]
> bool check_narrow_copy_args( const Tensor& in, int64_t dim, int64_t start, int64_t length, Tensor& out)

> [spec:et:sem:slice-util.torch.executor.check-narrow-copy-args-fn]
> Validates arguments for `narrow_copy` (slice of `length` elements starting at
> `start` along `dim`). Returns `true` if all checks pass; on the first failure
> logs and returns `false` (no output mutation). `start` is normalized to a local
> copy only (does not affect the caller).
>
> Checks, in order:
> 1. `in.dim() > 0` (input must not be a scalar).
> 2. `tensors_have_same_dtype(in, out)`.
> 3. `tensor_has_dim(in, dim)` (dim is a valid, possibly-negative, dimension of `in`).
> 4. `length >= 0` (message "length must be non-negative").
> 5. `start >= -in.size(dim)` and `start <= in.size(dim)`.
> 6. Normalize the local `start`: if `start < 0`, `start += in.size(dim)`.
> 7. `start + length <= in.size(dim)` (the slice fits within the dimension).

> [spec:et:def:slice-util.torch.executor.check-slice-copy-args-fn]
> bool check_slice_copy_args( const Tensor& in, int64_t dim, int64_t step, Tensor& out)

> [spec:et:sem:slice-util.torch.executor.check-slice-copy-args-fn]
> Validates arguments for `slice_copy` (strided slice along `dim` with `step`).
> Returns `true` if all checks pass; on first failure logs and returns `false`.
>
> Checks, in order:
> 1. `in.dim() > 0`.
> 2. `tensors_have_same_dtype(in, out)`.
> 3. `tensor_has_dim(in, dim)`.
> 4. `step > 0` (message "slice step must be greater than zero").

> [spec:et:def:slice-util.torch.executor.check-slice-scatter-args-fn]
> bool check_slice_scatter_args( const Tensor& input, const Tensor& src, int64_t dim, int64_t num_values, int64_t step, Tensor output)

> [spec:et:sem:slice-util.torch.executor.check-slice-scatter-args-fn]
> Validates arguments for `slice_scatter` (writing `src` into a slice of `input`,
> producing `output`; note `output` is passed by value). Returns `true` if all
> checks pass; on first failure logs and returns `false`.
>
> Checks, in order:
> 1. `input.dim() > 0`.
> 2. `dim_is_valid(dim, input.dim())` — `dim` must be a valid dimension of `input`.
> 3. `tensors_have_same_shape_and_dtype(input, output)` — output matches input
>    exactly.
> 4. `tensors_have_same_rank(input, src)` — `input.dim() == src.dim()`.
> 5. `step > 0` (message "slice step must be greater than zero").
> 6. For each dimension `d` in `[0, input.dim())`:
>    - if `d != dim`: `tensors_have_same_size_at_dims(input, d, src, d)` (src matches
>      input in every non-slice dimension).
>    - if `d == dim`: `src.size(dim) == num_values` (src's slice dimension equals the
>      number of values being scattered).

> [spec:et:def:slice-util.torch.executor.compute-slice-fn]
> void compute_slice( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, int64_t start, int64_t length, int64_t step, Tensor& out)

> [spec:et:sem:slice-util.torch.executor.compute-slice-fn]
> Performs the actual strided copy of a slice from `in` into `out`: takes
> `length` values along `dim` starting at `start` with stride `step`. Uses the
> `KernelRuntimeContext` `ctx` for error reporting (returns void; on a failed
> `ET_KERNEL_CHECK_MSG` it sets `Error::InvalidArgument` on `ctx` and returns
> without writing).
>
> Steps:
> 1. If `length <= 0`, return immediately (empty slice, no copy).
> 2. `ET_KERNEL_CHECK_MSG`: `dim < in.dim()` else InvalidArgument
>    ("Requested dim is larger than input tensor dim"). `dim` is assumed already
>    non-negative here.
> 3. `dim_length = in.size(dim)`.
> 4. `ET_KERNEL_CHECK_MSG`: `start >= 0 && length >= 0 && step >= 0` else
>    InvalidArgument ("Input args should be >= 0.").
> 5. `requested_slice = start + (length - 1) * step`;
>    `ET_KERNEL_CHECK_MSG`: `(uint64_t)requested_slice < (uint64_t)dim_length` else
>    InvalidArgument ("Requested slice is larger than the dim size") — the last
>    selected index must be in range.
> 6. `leading_dims = getLeadingDims(in, dim)`, `trailing_dims = getTrailingDims(in, dim)`.
>    If `trailing_dims == 0`, return.
> 7. `length_per_step = trailing_dims * in.element_size()` bytes per copied row.
> 8. `ET_KERNEL_CHECK_MSG`: `out.nbytes() >= length * leading_dims * length_per_step`
>    else InvalidArgument ("out.nbytes() is smaller than the expected slice size.").
> 9. Copy each of the `leading_dims` outer slices: for slice `i`, the source starts
>    at `in data + (i * dim_length + start) * length_per_step`, and for each of the
>    `length` output positions `j`: `memcpy` `length_per_step` bytes, advance source
>    by `step * length_per_step` and destination by `length_per_step`. Output for
>    slice `i` begins at `out data + i * length * length_per_step`.
> 10. Multithreading: when `leading_dims >= 8` AND total elements
>     `leading_dims * length * trailing_dims >= GRAIN_SIZE` (32768), the outer loop
>     over `leading_dims` is run via `parallel_for` with grain size 8 per
>     `[spec:et:sem:thread-parallel-interface...]`; each task processes a contiguous
>     range of leading indices with the exact per-slice logic above (each writing to
>     its own `dest + i * length * length_per_step`, so results are independent of
>     partitioning). Otherwise a single-threaded loop produces identical output.

> [spec:et:def:slice-util.torch.executor.get-narrow-copy-out-target-size-fn]
> void get_narrow_copy_out_target_size( const Tensor& in, int64_t dim, int64_t length, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:slice-util.torch.executor.get-narrow-copy-out-target-size-fn]
> Computes the output shape for `narrow_copy`. Sets `*out_ndim = in.dim()`; copies
> `out_sizes[d] = in.size(d)` for all `d` in `[0, in.dim())`; then overrides
> `out_sizes[dim] = length`. I.e. same shape as `in` with dimension `dim` shrunk to
> `length`. `dim` is assumed already non-negative and valid.

> [spec:et:def:slice-util.torch.executor.get-slice-copy-out-target-size-fn]
> void get_slice_copy_out_target_size( const Tensor& in, int64_t dim, int64_t length, executorch::aten::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:slice-util.torch.executor.get-slice-copy-out-target-size-fn]
> Computes the output shape for `slice_copy`. Delegates directly to
> `get_narrow_copy_out_target_size(in, dim, length, out_sizes, out_ndim)` per
> `[spec:et:sem:slice-util.torch.executor.get-narrow-copy-out-target-size-fn]`;
> here the `length` argument is the number of sliced values (`num_values`), so the
> result is `in`'s shape with dimension `dim` set to that value.
