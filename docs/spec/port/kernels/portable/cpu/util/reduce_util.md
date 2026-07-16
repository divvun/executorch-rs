# kernels/portable/cpu/util/reduce_util.cpp, kernels/portable/cpu/util/reduce_util.h

> [spec:et:def:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]
> void apply_on_flat_and_dim_ix_with_stride_and_base( const Fn& fn, const size_t stride, const size_t base, const size_t start, const size_t end)

> [spec:et:sem:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]
> Iterates `i` from `start` to `end` inclusive (`for i in start..=end`), and for
> each `i` invokes `fn(base + i * stride, i)`. `fn` receives two arguments: the
> flat element index `base + i * stride`, and the per-dimension iteration index
> `i`. All arithmetic is on `size_t` (unsigned pointer-sized). Callers guarantee
> `start <= end + 1`; if `start > end` the loop body never runs (no calls). No
> return value; the side effects are entirely inside `fn`.

> [spec:et:def:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn]
> void apply_on_flat_ix_with_dim_mask_and_base( const Fn& fn, const Tensor& in, const bool* dim_mask, const size_t base, const size_t start, const size_t end)

> [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn]
> Applies `fn(curr_index)` to each flat index in `in` that belongs to the set of
> reduced dimensions selected by `dim_mask`, for the reduction whose first
> element lives at flat index `base`, but only for iteration counts in the
> inclusive window `[start, end]`.
>
> `dim_mask` is a `bool[in.dim()]` where `dim_mask[d]` is true iff dimension `d`
> is being reduced over. Setup:
> - Find `inner_dim`: start at `in.dim() - 1` and decrement while
>   `dim_mask[inner_dim]` is false, i.e. the largest (innermost) dimension index
>   that is in the reduce set. (Callers guarantee at least one masked dim.)
> - `dim_index` is a per-dimension counter array of length `in.dim()`
>   (capacity `kTensorDimensionLimit`), all initialized to 0. Only entries for
>   masked dims are meaningfully advanced.
> - `strides = in.strides()`.
> - `curr_index = base` (the running flat index into `in`).
> - `apply_fun_counter = 0` (counts how many masked elements have been visited).
>
> Main loop (`while true`):
> 1. If `apply_fun_counter >= start && apply_fun_counter <= end`, call
>    `fn(curr_index)`.
> 2. Increment `apply_fun_counter`. If it is now `> end`, return.
> 3. Advance the innermost masked dim: `dim_index[inner_dim] += 1` and
>    `curr_index += strides[inner_dim]`.
> 4. Carry-over when `dim_index[inner_dim] == in.size(inner_dim)`: set
>    `curr_dim = inner_dim`, then `while dim_index[curr_dim] == in.size(curr_dim)`:
>    - If `curr_dim == 0`, return (finished the outermost dim).
>    - Reset `dim_index[curr_dim] = 0` and subtract the wrapped span from
>      `curr_index`: `curr_index -= strides[curr_dim - 1]` (this equals
>      `in.size(curr_dim) * strides[curr_dim]`, since row-major-contiguous-like
>      strides give `strides[curr_dim-1] == in.size(curr_dim) * strides[curr_dim]`;
>      this relies on default/channels-last dim order guaranteed by
>      `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]`).
>    - Decrement `curr_dim`, then keep decrementing while `curr_dim >= 0` and
>      `dim_mask[curr_dim]` is false (skip non-reduced dims). If `curr_dim < 0`
>      after this, return (reduced over all needed elements).
>    - Otherwise `dim_index[curr_dim] += 1` and `curr_index += strides[curr_dim]`,
>      then re-test the `while` carry condition.
>
> The net effect is a nested odometer iterating only over the masked dimensions
> in row-major (outermost-slowest) order, emitting each visited flat index into
> `fn`, gated to the `[start, end]` counter window. All arithmetic on `size_t`
> for `curr_index`/counters and `int64_t` for dim indices.

> [spec:et:def:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn]
> void apply_on_flat_ix_with_stride_and_base( const Fn& fn, const size_t stride, const size_t base, const size_t start, const size_t end)

> [spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn]
> Iterates `i` from `start` to `end` inclusive (`for i in start..=end`), and for
> each `i` invokes `fn(base + i * stride)`. `fn` receives one argument: the flat
> element index `base + i * stride`. All arithmetic is on `size_t`. If
> `start > end` the loop body never runs. No return value. This is the single-arg
> variant of
> `[spec:et:sem:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]`
> (it drops the `dim_ix` argument).

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-fn]
> void apply_over_dim( const Fn& fn, const executorch::aten::Tensor& in, const std::optional<int64_t>& dim, const size_t out_ix, const int64_t start = 0, const int64_t end = -1)

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn]
> Iterates over the elements of `in` that reduce into the output element at flat
> index `out_ix` when reducing over the single dimension `dim`, invoking
> `fn(in_ix, dim_ix)` for each, where `in_ix` is the flat index of the element in
> `in` and `dim_ix` is its index along `dim`. Default args are `start = 0`,
> `end = -1`.
>
> Validation:
> - If `dim` has a value: if `in.dim() != 0`, `ET_CHECK_VALID_DIM(dim, in.dim())`
>   (aborts if `dim` is not in `[-in.dim(), in.dim())`); else (0-D input),
>   `ET_CHECK(dim == 0 || dim == -1)`.
> - `ET_CHECK_MSG(out_ix < get_out_numel(in, dim), ...)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-out-numel-fn]`; aborts on failure.
> - If `in.numel() == 0`, return immediately (nothing to iterate).
>
> Compute the reduction span length `iter_length = get_reduced_dim_product(in, dim)`
> per `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`, then
> normalize the requested `[start, end]` window against it:
> - `normalized_start = ET_NORMALIZE_IX(start, iter_length)` (adds `iter_length`
>   if negative).
> - `normalized_end = ET_NORMALIZE_IX(end, iter_length)` (so the default `-1`
>   maps to `iter_length - 1`).
> - `ustart = max(normalized_start, 0)`, `uend = min(normalized_end, iter_length - 1)`.
>
> Dispatch:
> - If `dim` has no value (reduce whole tensor): call
>   `apply_on_flat_and_dim_ix_with_stride_and_base(fn, stride=1, base=0, ustart, uend)`
>   per `[spec:et:sem:reduce-util.torch.executor.apply-on-flat-and-dim-ix-with-stride-and-base-fn]`,
>   then return.
> - Otherwise compute `base = get_init_index(in, dim, out_ix)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-init-index-fn]` and
>   `d = ET_NORMALIZE_IX(dim, in.dim())`. If `in.dim() == 0`, call `fn(base, ustart)`
>   once; else call `apply_on_flat_and_dim_ix_with_stride_and_base(fn,
>   in.strides()[d], base, ustart, uend)`.
> No return value.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-fn]
> void apply_over_dim_list( const Fn& fn, const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, const size_t out_ix, const int64_t start = 0, const int64_t end = -1)

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-fn]
> Convenience wrapper: constructs an `ApplyOverDimListPlan plan(in, dim_list,
> start, end)` per
> `[spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]`
> then calls `plan.execute(fn, out_ix)` per
> `[spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]`.
> This applies `fn(in_ix)` (single flat-index argument) to each element of `in`
> that reduces into output flat index `out_ix` when reducing over the set of
> dimensions `dim_list`. Default args `start = 0`, `end = -1`. No return value.
> Prefer building the plan once and reusing it when iterating many `out_ix`.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan]
> class ApplyOverDimListPlan {
>   size_t ustart_;
>   size_t uend_;
>   enum class ExecutionMode { // Empty input, no work to do. NothingToDo, // Iterate over the entire tensor with // apply_on_flat_ix_with_stride_and_base. NoDim...;
>   ExecutionMode mode_;
>   size_t out_numel_;
>   std::optional<executorch::aten::ArrayRef<int64_t>> dim_list_;
>   std::array<bool, kTensorDimensionLimit> is_in_dim_list_;
>   const executorch::aten::Tensor& in_;
> }

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]
> ApplyOverDimListPlan( const executorch::aten::Tensor& in, // If set, lifetime must last until execute() returns. const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, const int64_t start = 0, const int64_t end = -1) : dim_l...

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]
> Constructor: precomputes the execution strategy for repeatedly applying a
> function over `dim_list` for varying `out_ix`. Stores `dim_list_ = dim_list`
> and `in_ = in` (input tensor held by reference; if `dim_list` is set, its
> backing storage must outlive `execute()` calls). Default args `start = 0`,
> `end = -1`.
>
> Steps:
> - `ET_CHECK(check_dim_list_is_valid(in, dim_list))` per
>   `[spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn]`; aborts
>   on failure.
> - `out_numel_ = get_out_numel(in, dim_list)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-out-numel-fn]`.
> - If `in.numel() == 0`: set `mode_ = NothingToDo` and return.
> - `iter_length = get_reduced_dim_product(in, dim_list)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`.
> - `ustart_ = max(ET_NORMALIZE_IX(start, iter_length), 0)`;
>   `uend_ = min(ET_NORMALIZE_IX(end, iter_length), iter_length - 1)` (default
>   `end = -1` → `iter_length - 1`).
> - If `dim_list` has no value, or `dim_list.size() == 0`, or `in.dim() == 0`:
>   set `mode_ = NoDimMaskOrZeroDimension` and return (reduce whole tensor).
> - `dim_list_ = dim_list.value()`. If `dim_list_.size() == 1`: set
>   `mode_ = OnlyOneDim` and return.
> - Otherwise fill `is_in_dim_list_` (a `bool[kTensorDimensionLimit]`) with false,
>   then for each `d` in `dim_list`: `non_neg_d = d < 0 ? d + in.dim() : d`, set
>   `is_in_dim_list_[non_neg_d] = true`. Set `mode_ = NormalDimMask`.
>
> See `[spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.execution-mode]`
> for the mode enum.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]
> void execute(const Fn& fn, const size_t out_ix) const

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]
> Applies `fn(in_ix)` to every element of `in_` that reduces into output flat
> index `out_ix`, using the strategy chosen at construction.
>
> - `ET_CHECK_MSG(out_ix < out_numel_, ...)`; aborts on failure.
> - Dispatch on `mode_`:
>   - `NothingToDo`: return (empty input).
>   - `NoDimMaskOrZeroDimension`: call
>     `apply_on_flat_ix_with_stride_and_base(fn, stride=1, base=0, ustart_, uend_)`
>     per `[spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-stride-and-base-fn]`
>     (iterates the whole flattened tensor window).
>   - `OnlyOneDim`: call
>     `apply_on_flat_and_dim_ix_with_stride_and_base` with an adapter lambda
>     `(in_ix, dim_ix) -> fn(in_ix)` (dropping `dim_ix`), stride
>     `in_.strides()[ET_NORMALIZE_IX(dim_list_[0], in_.dim())]`, base
>     `get_init_index(in_, dim_list_, out_ix)` per
>     `[spec:et:sem:reduce-util.torch.executor.get-init-index-fn]`, and window
>     `ustart_, uend_`.
>   - `NormalDimMask`: call `apply_on_flat_ix_with_dim_mask_and_base(fn, in_,
>     is_in_dim_list_.data(), get_init_index(in_, dim_list_, out_ix), ustart_,
>     uend_)` per
>     `[spec:et:sem:reduce-util.torch.executor.apply-on-flat-ix-with-dim-mask-and-base-fn]`.
> No return value. `const` method.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.execution-mode]
> enum class ExecutionMode {
>   NothingToDo;
>   NoDimMaskOrZeroDimension;
>   OnlyOneDim;
>   NormalDimMask;
> }

> [spec:et:def:reduce-util.torch.executor.check-amin-amax-args-fn]
> bool check_amin_amax_args( const Tensor& in, ArrayRef<int64_t> dim_list, bool keepdim, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.check-amin-amax-args-fn]
> Validates arguments for the `amin`/`amax` reduction ops. Returns `true` if all
> checks pass, else logs and returns `false` at the first failure
> (`ET_LOG_AND_RETURN_IF_FALSE`). Only compiled when `USE_ATEN_LIB` is not
> defined.
>
> - `check_reduction_args(in, dim_list, keepdim, {} /*no dtype*/, out)` per
>   `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]` (note
>   `dim_list` is a non-optional `ArrayRef<int64_t>`, implicitly wrapped into the
>   optional).
> - `in.scalar_type() == out.scalar_type()` (output dtype must equal input dtype).

> [spec:et:def:reduce-util.torch.executor.check-argmin-argmax-args-fn]
> bool check_argmin_argmax_args( const Tensor& in, optional<int64_t> dim, bool keepdim, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.check-argmin-argmax-args-fn]
> Validates arguments for the `argmin`/`argmax` ops. Returns `true` if all checks
> pass, else logs and returns `false` at the first failure. Only compiled when
> `USE_ATEN_LIB` is not defined.
>
> - `check_reduction_args_single_dim(in, dim, keepdim, {} /*no dtype*/, out)` per
>   `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn]`
>   (uses default `allow_empty_dim = false`).
> - `out.scalar_type() == ScalarType::Long` (argmin/argmax always produce int64
>   indices).

> [spec:et:def:reduce-util.torch.executor.check-dim-in-dim-list-fn]
> bool check_dim_in_dim_list( const size_t dim, const size_t max_dim, const executorch::aten::ArrayRef<int64_t>& dim_list)

> [spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn]
> Returns `true` iff the non-negative dimension index `dim` (already normalized,
> in `[0, max_dim)`) is a member of `dim_list`. Iterates each entry `d` of
> `dim_list`, normalizes it with `_normalize_non_neg_d(d, max_dim)` per
> `[spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn]` (so negative
> `d` becomes `d + max_dim`), and returns `true` on the first match `dim ==
> non_neg_dim`. Returns `false` if no entry matches (including when `dim_list` is
> empty). No aborts or logging; pure predicate. All values are `size_t`.

> [spec:et:def:reduce-util.torch.executor.check-dim-list-is-valid-fn]
> ET_NODISCARD bool check_dim_list_is_valid( const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list)

> [spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn]
> `ET_NODISCARD` predicate: returns `true` iff `dim_list` is a valid set of
> reduction dims for `in`. If `dim_list` has no value or is empty, returns `true`
> (means "reduce all dims", trivially valid).
>
> Otherwise allocates a local `bool dim_exist[kTensorDimensionLimit]`
> zero-initialized (memset false), and for each `d` in `dim_list`:
> - If `in.dim() == 0`: `ET_LOG_AND_RETURN_IF_FALSE(d == 0 || d == -1)` (only
>   0/-1 accepted for scalar input).
> - Else: `ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(d, in.dim()))` (i.e. `d` in
>   `[-in.dim(), in.dim())`).
> - `non_neg_d = _normalize_non_neg_d(d, in.dim())` per
>   `[spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn]`.
> - `ET_LOG_AND_RETURN_IF_FALSE(non_neg_d < kTensorDimensionLimit)`.
> - `ET_CHECK_OR_RETURN_FALSE(dim_exist[non_neg_d] == false, "dim ... appears
>   multiple times ...")` — rejects duplicate dims; on failure logs the message
>   and returns `false`.
> - Mark `dim_exist[non_neg_d] = true`.
>
> Returns `true` after all entries validate. Each `ET_LOG_AND_RETURN_IF_FALSE` /
> `ET_CHECK_OR_RETURN_FALSE` logs and returns `false` at the first failing check.

> [spec:et:def:reduce-util.torch.executor.check-mean-dim-args-fn]
> bool check_mean_dim_args( const Tensor& in, optional<ArrayRef<int64_t>> dim_list, bool keepdim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.check-mean-dim-args-fn]
> Validates arguments for `mean.dim`. Returns `true` if all checks pass, else
> logs and returns `false` at the first failure. Only compiled when `USE_ATEN_LIB`
> is not defined.
>
> - `check_reduction_args(in, dim_list, keepdim, dtype, out)` per
>   `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]`.
> - If `dtype` has a value: logs it at Info level, then requires
>   `torch::executor::isFloatingType(dtype.value())` (mean output dtype must be
>   floating), and `out.scalar_type() == dtype.value()`.
> - Else (no `dtype`): requires `tensor_is_floating_type(in)` and
>   `tensor_is_floating_type(out)` (both input and output must be floating).

> [spec:et:def:reduce-util.torch.executor.check-min-max-args-fn]
> bool check_min_max_args( const Tensor& in, int64_t dim, bool keepdim, Tensor& max, Tensor& max_indices)

> [spec:et:sem:reduce-util.torch.executor.check-min-max-args-fn]
> Validates arguments for the `min.dim`/`max.dim` ops, which produce a values
> tensor (`max`) and an indices tensor (`max_indices`). Returns `true` if all
> checks pass, else logs and returns `false` at the first failure. Only compiled
> when `USE_ATEN_LIB` is not defined.
>
> - `check_reduction_args_single_dim(in, dim, keepdim, {} /*no dtype*/, max)` per
>   `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn]`
>   (default `allow_empty_dim = false`).
> - `tensors_have_same_dtype(in, max)` (values tensor shares input dtype).
> - `tensors_have_same_shape(max, max_indices)` (values and indices same shape).
> - `tensor_is_default_or_channels_last_dim_order(max_indices)`.
> - `max_indices.scalar_type() == ScalarType::Long`.

> [spec:et:def:reduce-util.torch.executor.check-prod-out-args-fn]
> bool check_prod_out_args( const Tensor& in, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.check-prod-out-args-fn]
> Validates the output dtype for the `prod` op. Returns `true` if it passes, else
> logs and returns `false`. Only compiled when `USE_ATEN_LIB` is not defined.
>
> - If `dtype` has a value: require `dtype.value() == out.scalar_type()`.
> - Else if `in` is an integral type including bool
>   (`isIntegralType(in.scalar_type(), includeBool=true)`): require
>   `out.scalar_type() == ScalarType::Long` (integer/bool products are promoted to
>   int64).
> - Else (floating/complex input): require `out.scalar_type() == in.scalar_type()`
>   (output keeps input dtype).

> [spec:et:def:reduce-util.torch.executor.check-reduction-args-fn]
> bool check_reduction_args( const Tensor& in, const optional<ArrayRef<int64_t>>& dim_list, bool keepdim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]
> General validation shared by dim-list reduction ops. Returns `true` if all
> checks pass, else logs and returns `false` at the first failure. Only compiled
> when `USE_ATEN_LIB` is not defined.
>
> - If `dtype` has a value: require `dtype.value() == out.scalar_type()`.
> - `check_dim_list_is_valid(in, dim_list)` per
>   `[spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn]`.
> - `tensor_is_default_or_channels_last_dim_order(in)`.
> - `tensor_is_default_or_channels_last_dim_order(out)`.
>
> Note `keepdim` is accepted but not checked here.

> [spec:et:def:reduce-util.torch.executor.check-reduction-args-single-dim-fn]
> bool check_reduction_args_single_dim( const Tensor& in, optional<int64_t> dim, bool keepdim, optional<ScalarType> dtype, Tensor& out, bool allow_empty_dim)

> [spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn]
> Validation shared by reduction ops taking a single optional `dim`. Returns
> `true` if all checks pass, else logs and returns `false` at the first failure.
> Only compiled when `USE_ATEN_LIB` is not defined. `allow_empty_dim` defaults to
> `false`.
>
> - If `dtype` has a value: require `dtype.value() == out.scalar_type()`.
> - If `in.dim() == 0` (scalar input): if `dim` has a value require
>   `dim.value() == 0 || dim.value() == -1`; then return `true` immediately
>   (skips the dim-order checks below).
> - If `dim` has a value (and `in.dim() != 0`):
>   - `dim_is_valid(dim.value(), in.dim())` (`dim` in `[-in.dim(), in.dim())`).
>   - If `!allow_empty_dim`: `tensor_has_non_empty_dim(in, dim.value())` (the
>     reduced dimension must have size > 0).
> - `tensor_is_default_or_channels_last_dim_order(in)`.
> - `tensor_is_default_or_channels_last_dim_order(out)`.
>
> `keepdim` is accepted but not checked.

> [spec:et:def:reduce-util.torch.executor.compute-reduced-out-dim-fn]
> inline ssize_t compute_reduced_out_dim( const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, bool keepdim)

> [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-dim-fn]
> Returns the number of dimensions the reduction output tensor will have, for the
> `dim_list` overload. Pure inline arithmetic, no side effects, returns `ssize_t`.
> - If `keepdim` is true: returns `in.dim()` (reduced dims collapse to size 1 but
>   are kept).
> - Else if `dim_list` has a value AND `dim_list.size() != 0` AND `in.dim() != 0`:
>   returns `in.dim() - dim_list.size()` (each reduced dim removed).
> - Else: returns `0` (full reduction to a scalar, or 0-D input).
> This does not deduplicate `dim_list`; callers are expected to have validated it
> via `[spec:et:sem:reduce-util.torch.executor.check-dim-list-is-valid-fn]`.

> [spec:et:def:reduce-util.torch.executor.compute-reduced-out-size-fn]
> size_t compute_reduced_out_size( const Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, bool keepdim, executorch::aten::SizesType* sizes_arr)

> [spec:et:sem:reduce-util.torch.executor.compute-reduced-out-size-fn]
> Writes the output shape of a reduction into the caller-provided `sizes_arr`
> (must hold at least `kTensorDimensionLimit` entries) and returns the number of
> output dims written (`size_t`). This is the `dim_list` overload.
>
> Let `in_dim = static_cast<size_t>(in.dim())`, `out_dim = in_dim` initially.
> - If `dim_list` has a value AND `dim_list.size() != 0`:
>   - If `keepdim`: for each `i` in `[0, in_dim)`, set `sizes_arr[i] = 1` if
>     `check_dim_in_dim_list(i, in_dim, reduce_dims)` per
>     `[spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn]`, else
>     `sizes_arr[i] = in.size(i)`. `out_dim` stays `in_dim`.
>   - Else (drop reduced dims): walk `in_i` from 0 to `in_dim-1`, and for each
>     input dim NOT in `dim_list`, copy `sizes_arr[out_i++] = in.size(in_i)`.
>     Set `out_dim = out_i` (the count of retained dims).
> - Else (no dim_list / empty → full reduction):
>   - If `keepdim`: set all `sizes_arr[i] = 1` for `i` in `[0, in_dim)`; `out_dim`
>     stays `in_dim`.
>   - Else: `out_dim = 0` (scalar output; nothing written to `sizes_arr`).
> Returns `out_dim`. Reduced dims are matched with duplicate-tolerant membership
> (`check_dim_in_dim_list`), so a validated `dim_list` yields correct counts.

> [spec:et:def:reduce-util.torch.executor.get-init-index-fn]
> size_t get_init_index( const Tensor& in, const std::optional<int64_t>& dim, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.get-init-index-fn]
> Returns the flat index into `in` of the first element that maps to output flat
> index `out_ix` when reducing over the single dimension `dim`. This is the
> single-`dim` overload.
>
> - If `dim` has no value: return `0` (full reduction; first element is index 0).
> - Validate `dim`: if `in.dim() == 0`, `ET_CHECK(dim == 0 || dim == -1)`; else
>   `ET_CHECK_VALID_DIM(dim, in.dim())`. Aborts on failure.
> - `non_neg_dim = _normalize_non_neg_d(dim, in.dim())`.
> - Reconstruct the flat index by decomposing `out_ix` across the non-reduced
>   dims: `init_ix = 0`; `mutable_out_ix = out_ix`; `strides = in.strides()`.
>   Iterate `d` from `in.dim() - 1` down to `0`; for each `d != non_neg_dim`:
>   `init_ix += (mutable_out_ix % in.size(d)) * strides[d]`; then
>   `mutable_out_ix /= in.size(d)`. (The reduced dim `non_neg_dim` is skipped, so
>   its coordinate is left at 0 — the "first" element along that dim.)
> - Return `init_ix` (`size_t`).
>
> The overload taking `dim_list` (an `optional<ArrayRef<int64_t>>`) is analogous:
> returns 0 when `dim_list` is null/empty, and otherwise skips every dim `d` for
> which `check_dim_in_dim_list(d, in.dim(), dim_list)` is true per
> `[spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn]`, with no
> `ET_CHECK` validation.

> [spec:et:def:reduce-util.torch.executor.get-out-numel-fn]
> size_t get_out_numel(const Tensor& in, const std::optional<int64_t>& dim)

> [spec:et:sem:reduce-util.torch.executor.get-out-numel-fn]
> Returns the number of elements in the output of reducing `in` over the single
> dimension `dim` (`size_t`). This is the single-`dim` overload.
>
> - Start `out_numel = 1`.
> - If `dim` has no value: return `1` (full reduction to a scalar).
> - Validate `dim`: if `in.dim() == 0`, `ET_CHECK(dim == 0 || dim == -1)`; else
>   `ET_CHECK_VALID_DIM(dim, in.dim())`. Aborts on failure.
> - `non_neg_dim = _normalize_non_neg_d(dim, in.dim())`.
> - For each `d` in `[0, in.dim())` with `d != non_neg_dim`: multiply
>   `out_numel *= in.size(d)`.
> - Return `out_numel` (product of all non-reduced dim sizes; equals 1 when `in`
>   is 0-D since the loop body never runs).
>
> The `dim_list` overload (`optional<ArrayRef<int64_t>>`) returns 1 when
> `dim_list` is null/empty, otherwise the product of `in.size(d)` over all `d`
> not in the dim list (membership via `check_dim_in_dim_list` per
> `[spec:et:sem:reduce-util.torch.executor.check-dim-in-dim-list-fn]`), without
> any `ET_CHECK` validation.

> [spec:et:def:reduce-util.torch.executor.get-reduced-dim-product-fn]
> size_t get_reduced_dim_product( const Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list)

> [spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]
> Returns the product of the sizes of all reduced dimensions (`size_t`), i.e. how
> many input elements collapse into one output element. This is the `dim_list`
> overload (`optional<ArrayRef<int64_t>>`).
>
> - If `in.dim() == 0`: return `1`.
> - If `dim_list` is null or empty: return `in.numel()` (full reduction).
> - Otherwise: `dim_product = 1`; for each `d` in `dim_list`, compute
>   `non_neg_d = _normalize_non_neg_d(d, in.dim())` per
>   `[spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn]` and multiply
>   `dim_product *= in.size(non_neg_d)`. Return `dim_product`. No validation or
>   dedup (assumes a valid dim_list).
>
> The single-`dim` overload (`optional<int64_t>`): returns 1 for 0-D input,
> `in.numel()` when `dim` has no value, else `in.size(_normalize_non_neg_d(dim,
> in.dim()))`.

> [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-fn]
> std::tuple<CTYPE_OUT, long> map_reduce_over_dim( const MapOp& map_fun, const ReduceOp& reduce_fun, const executorch::aten::Tensor& in, const std::optional<int64_t>& dim, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn]
> Reduces `in` over the single dimension `dim` for output element `out_ix`, first
> mapping each input element with `map_fun` (`CTYPE_IN -> CTYPE_OUT`) then folding
> with `reduce_fun` (`CTYPE_OUT val, long ix, CTYPE_OUT acc_val, long acc_ix ->
> (CTYPE_OUT, long)`). Returns the final `(acc_val, acc_ix)` tuple. Templated on
> `CTYPE_IN, CTYPE_OUT`.
>
> Validation:
> - If `dim` has a value: `ET_CHECK_VALID_DIM(dim, in.dim())` when `in.dim() != 0`,
>   else `ET_CHECK(dim == 0 || dim == -1)`.
> - `ET_CHECK_MSG(out_ix < get_out_numel(in, dim), ...)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-out-numel-fn]`.
> - `ET_CHECK_MSG(in.numel() > 0, "Input tensor must be nonempty")` — aborts on
>   empty input.
>
> Algorithm:
> - `init_index = get_init_index(in, dim, out_ix)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-init-index-fn]`.
> - `in_data = in.const_data_ptr<CTYPE_IN>()`.
> - Seed the accumulator with the first element:
>   `acc_val = map_fun(in_data[init_index])`, `acc_ix = 0`.
> - If `in.numel() == 1`: return `(acc_val, acc_ix)` immediately.
> - Otherwise call `apply_over_dim` per
>   `[spec:et:sem:reduce-util.torch.executor.apply-over-dim-fn]` with the same
>   `in, dim, out_ix` and window `start = 1, end = -1` (i.e. skip the seed element
>   at dim-index 0). The applied lambda receives `(in_ix, dim_ix)` and updates
>   `(acc_val, acc_ix) = reduce_fun(map_fun(in_data[in_ix]), dim_ix, acc_val,
>   acc_ix)`.
> - Return the final `(acc_val, acc_ix)`.

> [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-fn]
> CTYPE_OUT map_reduce_over_dim_list( const MapOp& map_fun, const ReduceOp& reduce_fun, const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-fn]
> Convenience wrapper: constructs `MapReduceOverDimListPlan plan(in, dim_list)`
> per
> `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]`
> and returns `plan.execute<CTYPE_IN, CTYPE_OUT>(map_fun, reduce_fun, out_ix)`
> per
> `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]`.
> Reduces `in` over the set `dim_list` for output element `out_ix`, mapping each
> element with `map_fun` (`CTYPE_IN -> CTYPE_OUT`) then folding with `reduce_fun`
> (`CTYPE_OUT v, CTYPE_OUT acc -> CTYPE_OUT`). Returns the final `CTYPE_OUT`
> accumulator. Prefer building the plan once when iterating many `out_ix`.

> [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan]
> class MapReduceOverDimListPlan {
>   ApplyOverDimListPlan plan_;
> }

> [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]
> CTYPE_OUT execute( const MapOp& map_fun, const ReduceOp& reduce_fun, const size_t out_ix) const

> [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]
> Computes the map-reduce for one output element `out_ix` using the precomputed
> `plan_`. Templated on `CTYPE_IN, CTYPE_OUT, MapOp, ReduceOp`. `map_fun` maps
> `CTYPE_IN -> CTYPE_OUT`; `reduce_fun` folds `(CTYPE_OUT v, CTYPE_OUT acc) ->
> CTYPE_OUT`. `const` method. Returns the final `CTYPE_OUT` accumulator.
>
> - `ET_CHECK_MSG(plan_.get_input_tensor().numel() > 0, "Input tensor must be
>   nonempty")`.
> - `init_index = get_init_index(plan_.get_input_tensor(), plan_.get_dim_list(),
>   out_ix)` per `[spec:et:sem:reduce-util.torch.executor.get-init-index-fn]`.
> - `in_data = plan_.get_input_tensor().const_data_ptr<CTYPE_IN>()`.
> - Seed `acc_val = map_fun(in_data[init_index])`.
> - If `plan_.get_input_tensor().numel() == 1`: return `acc_val` immediately.
> - Otherwise `plan_.execute(lambda(in_ix){ acc_val = reduce_fun(map_fun(
>   in_data[in_ix]), acc_val); }, out_ix)` per
>   `[spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.execute-fn]`.
>   Because `plan_` was built with `start = 1` (see the constructor), the seed
>   element at reduction-index 0 is skipped, so it is folded exactly once.
> - Return `acc_val`.

> [spec:et:def:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]
> MapReduceOverDimListPlan( const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list) : plan_(in, dim_list, 1, -1)

> [spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]
> Constructor. Builds the inner `plan_` as `ApplyOverDimListPlan(in, dim_list,
> start=1, end=-1)` per
> `[spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.apply-over-dim-list-plan-fn]`
> — the `start=1` means the plan's iteration skips reduction-index 0, which
> `execute()` uses as the accumulator seed. Then
> `ET_CHECK_MSG(in.numel() > 0, "Input tensor must be nonempty")`, aborting on
> empty input.

> [spec:et:def:reduce-util.torch.executor.normalize-non-neg-d-fn]
> inline size_t _normalize_non_neg_d(ssize_t d, ssize_t in_dim)

> [spec:et:sem:reduce-util.torch.executor.normalize-non-neg-d-fn]
> Normalizes a possibly-negative dimension index `d` into a non-negative index,
> given the input rank `in_dim`. Returns `size_t`.
> - If `in_dim == 0` and (`d == 0` or `d == -1`): return `0` (0-D tensor special
>   case — PyTorch accepts dim 0 or -1 on a scalar).
> - Else if `d < 0`: return `d + in_dim`.
> - Else: return `d`.
> No bounds checking or aborts; callers validate the range separately. Pure
> function.

> [spec:et:def:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]
> [[nodiscard]] bool parallel_for_each_reduce_over_dim_list_output_index( const Tensor& in, std::optional<ArrayRef<int64_t>> dim_list, const Tensor& out, const Func& func)

> [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]
> `[[nodiscard]]` wrapper that runs `func` over the output-index range
> `[0, out.numel())` in parallel, computing an appropriate grain size for the
> `dim_list` reductions. `func` has signature `void func(size_t begin, size_t
> end)` (a chunk of output indices). Returns the `bool` result of
> `executorch::extension::parallel_for` (true on success).
>
> - When compiled with `ET_USE_THREADPOOL`: `reduction_size =
>   get_reduced_dim_product(in, dim_list)` per
>   `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`; then
>   `grain_size = 1` if `reduction_size == 0`, else
>   `max(1, GRAIN_SIZE / reduction_size)` (so each grain does roughly `GRAIN_SIZE`
>   total element-visits). `GRAIN_SIZE` is `executorch::extension::internal::GRAIN_SIZE`.
> - Otherwise (no threadpool): `grain_size = 1`.
> - Returns `executorch::extension::parallel_for(0, out.numel(), grain_size, func)`.

> [spec:et:def:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn]
> [[nodiscard]] bool parallel_for_each_reduce_over_dim_output_index( const Tensor& in, std::optional<int64_t> dim, const Tensor& out, const Func& func)

> [spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn]
> `[[nodiscard]]` wrapper that runs `func` over the output-index range
> `[0, out.numel())` in parallel, computing an appropriate grain size for a
> single-`dim` reduction. Identical in structure to
> `[spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]`
> except `reduction_size = get_reduced_dim_product(in, dim)` uses the single
> `optional<int64_t> dim`. `func` has signature `void func(size_t begin, size_t
> end)`. Returns the `bool` result of `executorch::extension::parallel_for(0,
> out.numel(), grain_size, func)`.
> - With `ET_USE_THREADPOOL`: `grain_size = 1` when `reduction_size == 0`, else
>   `max(1, GRAIN_SIZE / reduction_size)`.
> - Without: `grain_size = 1`.

> [spec:et:def:reduce-util.torch.executor.reduce-over-dim-fn]
> std::tuple<CTYPE, long> reduce_over_dim( const ReduceOp& reduce_fun, const executorch::aten::Tensor& in, const std::optional<int64_t>& dim, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-fn]
> Thin wrapper over
> `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn]` with the
> identity map: calls `map_reduce_over_dim<CTYPE, CTYPE>([](CTYPE v){ return v; },
> reduce_fun, in, dim, out_ix)`. Reduces `in` over single dimension `dim` for
> output element `out_ix` using `reduce_fun` (`CTYPE val, long ix, CTYPE acc_val,
> long acc_ix -> (CTYPE, long)`). Returns the resulting `(CTYPE, long)` tuple.
> Inherits all validation and empty/single-element handling of `map_reduce_over_dim`.

> [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-fn]
> CTYPE reduce_over_dim_list( const ReduceOp& reduce_fun, const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-fn]
> Convenience wrapper: constructs `ReduceOverDimListPlan plan(in, dim_list)` per
> `[spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn]`
> and returns `plan.execute<CTYPE>(reduce_fun, out_ix)` per
> `[spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn]`.
> Reduces `in` over the set `dim_list` for output element `out_ix` using
> `reduce_fun` (`CTYPE v, CTYPE acc -> CTYPE`, no index tracking). Returns the
> `CTYPE` accumulator. Prefer building the plan once when iterating many `out_ix`.

> [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan]
> class ReduceOverDimListPlan {
>   MapReduceOverDimListPlan plan_;
> }

> [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn]
> CTYPE execute(const ReduceOp& reduce_fun, const size_t out_ix)

> [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.execute-fn]
> Computes the reduction for one output element `out_ix` using the precomputed
> `plan_`. Templated on `CTYPE, ReduceOp`. Delegates to
> `plan_.execute<CTYPE, CTYPE>([](CTYPE v){ return v; }, reduce_fun, out_ix)` per
> `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]`
> (identity map). `reduce_fun` folds `(CTYPE v, CTYPE acc) -> CTYPE`. Returns the
> `CTYPE` accumulator. Non-`const` (matches the inner plan's method).

> [spec:et:def:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn]
> ReduceOverDimListPlan( const executorch::aten::Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list) : plan_(in, dim_list)

> [spec:et:sem:reduce-util.torch.executor.reduce-over-dim-list-plan.reduce-over-dim-list-plan-fn]
> Constructor. Builds the inner `plan_` as `MapReduceOverDimListPlan(in,
> dim_list)` per
> `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]`.
> No additional work; all validation (dim-list validity, nonempty input,
> start=1 skip of the seed element) is inherited from that plan.

> [spec:et:def:reduce-util.torch.executor.resize-reduction-out-fn]
> Error resize_reduction_out( const Tensor& in, const std::optional<executorch::aten::ArrayRef<int64_t>>& dim_list, bool keepdim, Tensor& out)

> [spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]
> Resizes the reduction output tensor `out` to the shape implied by reducing `in`
> over `dim_list` with the given `keepdim`. Returns an `Error`. This is the
> `dim_list` overload.
>
> - Allocate a local `SizesType sizes_arr[kTensorDimensionLimit]`.
> - `out_dim = compute_reduced_out_size(in, dim_list, keepdim, sizes_arr)` per
>   `[spec:et:sem:reduce-util.torch.executor.compute-reduced-out-size-fn]`, which
>   fills `sizes_arr` and returns the output rank.
> - Build `out_size = ArrayRef{sizes_arr, out_dim}`.
> - Return `resize_tensor(out, out_size)` (propagates its `Error`; success is
>   `Error::Ok`).
>
> The single-`dim` overload is identical but calls the single-`dim`
> `compute_reduced_out_size`.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.get-dim-list-fn]
> const std::optional<executorch::aten::ArrayRef<int64_t>>& get_dim_list()

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-dim-list-fn]
> Accessor. Returns a const reference to the stored `dim_list_`
> (`std::optional<ArrayRef<int64_t>>`). Note the constructor overwrites
> `dim_list_` with the unwrapped `.value()` when the original had a value and was
> non-empty and the input was non-0-D; so this returns the normalized dim list
> used internally by `execute()`. `const` method, no side effects.

> [spec:et:def:reduce-util.torch.executor.apply-over-dim-list-plan.get-input-tensor-fn]
> const executorch::aten::Tensor& get_input_tensor() const

> [spec:et:sem:reduce-util.torch.executor.apply-over-dim-list-plan.get-input-tensor-fn]
> Accessor. Returns a const reference to the stored input tensor `in_` (the
> tensor bound at construction). `const` method, no side effects.

