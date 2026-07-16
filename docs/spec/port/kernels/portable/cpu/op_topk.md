# kernels/portable/cpu/op_topk.cpp

> [spec:et:def:op-topk.torch.executor.native.allocate-temp-memory-fn]
> void* allocate_temp_memory(KernelRuntimeContext& ctx, size_t size)

> [spec:et:sem:op-topk.torch.executor.native.allocate-temp-memory-fn]
> Allocates `size` bytes of scratch memory from the kernel runtime context.
> Calls `ctx.allocate_temp(size)` and returns the resulting pointer on success
> (`temp_mem_res.ok()`), or `nullptr` if allocation failed. Does not throw; the
> caller checks for null and reports MemoryAllocationFailed.

> [spec:et:def:op-topk.torch.executor.native.check-topk-args-fn]
> bool check_topk_args( const Tensor& in, int64_t k, int64_t dim, Tensor& values, Tensor& indices)

> [spec:et:sem:op-topk.torch.executor.native.check-topk-args-fn]
> Validates arguments for `topk_values`, returning true if all pass and false
> (after logging) on the first failure. Checks in order:
>
> - `tensors_have_same_dtype(in, values)`: `in` and `values` must share dtype.
> - `indices.scalar_type() == ScalarType::Long`: indices tensor must be int64.
> - `tensor_has_dim(in, dim)`: `dim` (possibly negative) must be a valid
>   dimension of `in`.
> - Then normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)`.
> - `k >= 0 && k <= nonempty_size(in, dim)`: `k` must be in range of the
>   (non-empty) size along `dim`; failure logs an out-of-range message.
>
> Returns true only if every check passes.

> [spec:et:def:op-topk.torch.executor.native.float-less-than-fn]
> bool float_less_than(T x, T y)

> [spec:et:sem:op-topk.torch.executor.native.float-less-than-fn]
> NaN-aware "less than" comparator `float_less_than<T>(x, y)` used to order topk
> elements.
>
> - If `T` is an integral type, returns plain `x < y`.
> - Otherwise (floating types) returns `(!isnan(x) && isnan(y)) || x < y`, using
>   `utils::isnan_override`. This treats a NaN `y` as greater than any non-NaN
>   `x` (so NaNs sort to the "large" end), and for non-NaN operands falls back to
>   ordinary `x < y`. Note: when both are NaN, or `x` is NaN, the first clause is
>   false and `x < y` is false, so the result is false (NaN not-less-than
>   anything).

> [spec:et:def:op-topk.torch.executor.native.get-topk-target-size-fn]
> bool get_topk_target_size( const Tensor& in, int64_t k, int64_t dim, Tensor::SizesType* target_size, size_t* target_dim)

> [spec:et:sem:op-topk.torch.executor.native.get-topk-target-size-fn]
> Computes the output shape for topk. Sets `*target_dim = in.dim()` and fills
> `target_size[i]` for each dimension `i`: `k` if `i == dim`, otherwise
> `in.size(i)`. In other words the output shape equals `in`'s shape with the size
> along `dim` replaced by `k`. Always returns true. (`dim` is expected to already
> be normalized to non-negative by the caller.)

> [spec:et:def:op-topk.torch.executor.native.perform-topk-fn]
> void perform_topk( const Tensor& in, int64_t k, int64_t dim, bool largest, bool sorted, Tensor& values, Tensor& indices, elem_t* queue)

> [spec:et:sem:op-topk.torch.executor.native.perform-topk-fn]
> Core templated routine `perform_topk<CTYPE, elem_t=pair<CTYPE,int64_t>>` that
> selects the top-`k` elements along `dim` and writes them to `values`/`indices`
> using a caller-provided scratch `queue` of at least `in.size(dim)` `elem_t`
> pairs. `dim` is assumed already normalized. Step by step:
>
> - Grab typed pointers `in_data` (CTYPE), `values_data` (CTYPE), `indices_data`
>   (int64).
> - If `in.dim() == 0` (scalar): set `values_data[0] = in_data[0]`,
>   `indices_data[0] = 0`, return.
> - If `k == 0`: return (nothing to write).
> - Compute traversal geometry: `outer_size = getLeadingDims(in, dim)` (product
>   of sizes before `dim`), `dim_size = in.size(dim)`, `dim_stride =
>   in.strides()[dim]`, `outer_stride_in = dim_size * dim_stride`,
>   `outer_stride_out = k * dim_stride`.
> - `use_partial_sort = (k * 64 <= dim_size)` (partial sort when k is small
>   relative to the axis length).
> - For each `outer_idx` in `[0, outer_size)`: `outer_in = outer_idx *
>   outer_stride_in`, `outer_out = outer_idx * outer_stride_out`. For each
>   `inner_idx` in `[0, dim_stride)`: `base_in = outer_in + inner_idx`, `base_out
>   = outer_out + inner_idx`.
>   - Populate the queue: for each `i` in `[0, dim_size)` set
>     `queue[i] = {in_data[base_in + i*dim_stride], i}` (value, original index).
>   - Build comparators from `float_less_than`
>     (`[spec:et:sem:op-topk.torch.executor.native.float-less-than-fn]`):
>     `elem_greater(x,y) = float_less_than(y.first, x.first)`,
>     `elem_less(x,y) = float_less_than(x.first, y.first)`; `cmp = largest ?
>     elem_greater : elem_less`.
>   - Select: if `use_partial_sort`, `std::partial_sort(queue, queue+k,
>     queue+dim_size, cmp)` (fully sorts the first k). Otherwise
>     `std::nth_element(queue, queue+k-1, queue+dim_size, cmp)` to partition, and
>     if `sorted` additionally `std::sort(queue, queue+k-1, cmp)` (note: sorts
>     only the first `k-1`, leaving element `k-1` in its nth_element position).
>   - Write outputs: for each `i` in `[0, k)`, `out_ix = base_out + i*dim_stride`,
>     `values_data[out_ix] = queue[i].first`, `indices_data[out_ix] =
>     queue[i].second`.
>
> Ties/NaN ordering follow `float_less_than`; stability of equal elements is not
> guaranteed (nth_element/partial_sort are not stable). Returns void.

> [spec:et:def:op-topk.torch.executor.native.topk-values-fn]
> std::tuple<Tensor&, Tensor&> topk_values( KernelRuntimeContext& ctx, const Tensor& in, int64_t k, int64_t dim, bool largest, bool sorted, Tensor& values, Tensor& indices)

> [spec:et:sem:op-topk.torch.executor.native.topk-values-fn]
> Selects the top-`k` elements of `in` along `dim` into `values`/`indices`.
> Implements `topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool
> sorted=True, *, Tensor(a!) values, Tensor(a!) indices)`. Returns
> `std::tuple<Tensor&, Tensor&>({values, indices})`. Step by step:
>
> - `out = tuple(values, indices)` (this exact tuple is returned on every path,
>   including error paths where the tensors are left unchanged).
> - ET_KERNEL_CHECK `check_topk_args(in, k, dim, values, indices)` per
>   `[spec:et:sem:op-topk.torch.executor.native.check-topk-args-fn]`; on failure
>   sets Error::InvalidArgument and returns `out`.
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(in)`.
> - Compute `target_size`/`target_dim` via `get_topk_target_size` per
>   `[spec:et:sem:op-topk.torch.executor.native.get-topk-target-size-fn]` (in's
>   shape with `dim` sized `k`). Resize `values` to it (on failure
>   Error::InvalidArgument), then resize `indices` to it (on failure
>   Error::InvalidArgument).
> - Early return: if `in.numel() == 0`, or (`k == 0` and `in.dim() > 0`), return
>   `out` (already-resized, empty result) without allocating.
> - Dtype dispatch: switch `in.scalar_type()` over REALHBF16 = {Byte, Char,
>   Short, Int, Long, Half, Float, Double, BFloat16} (note: NO Bool) as CTYPE.
>   `elem_t = pair<CTYPE, int64_t>`. Allocate scratch of `nonempty_size(in, dim)
>   * sizeof(elem_t)` bytes via `allocate_temp_memory` per
>   `[spec:et:sem:op-topk.torch.executor.native.allocate-temp-memory-fn]`; if it
>   returns null, leave `temp_mem_allocated` false and return from the switch
>   body. Otherwise set `temp_mem_allocated = true` and call
>   `perform_topk<CTYPE>(in, k, dim, largest, sorted, values, indices, queue)`
>   per `[spec:et:sem:op-topk.torch.executor.native.perform-topk-fn]`.
> - After the switch, ET_KERNEL_CHECK `temp_mem_allocated`; if false sets
>   Error::MemoryAllocationFailed and returns `out`.
> - Returns `out` (values = the top-k values, indices = their original positions
>   along `dim`).

