# kernels/portable/cpu/util/distance_util.cpp, kernels/portable/cpu/util/distance_util.h

> [spec:et:def:distance-util.torch.executor.check-cdist-args-fn]
> bool check_cdist_args( const Tensor& x1, const Tensor& x2, double p, optional<int64_t> compute_mode, const Tensor& out)

> [spec:et:sem:distance-util.torch.executor.check-cdist-args-fn]
> Validates the operand and output tensors for `cdist` (pairwise distance
> between the rows of two batched 2-D-or-higher tensors `x1` and `x2`). Returns
> `bool`: `true` if all checks pass, `false` on the first failing check.
>
> Each check below is an `ET_CHECK_OR_RETURN_FALSE`/`ET_LOG_AND_RETURN_IF_FALSE`
> guard: if the condition is false it logs (a message, empty for the
> `LOG_AND_RETURN` form) and returns `false` immediately from the function; it
> does not abort. Checks run in this exact order and short-circuit:
> - `tensors_have_same_dtype(x1, x2)`: `x1` and `x2` must have the same
>   `ScalarType`. False otherwise.
> - `tensors_have_same_dtype(x1, out)`: `out` must have the same `ScalarType`
>   as `x1`. False otherwise.
> - `tensor_has_rank_greater_or_equal_to(x1, 2)`: `x1.dim() >= 2`.
> - `tensor_has_rank_greater_or_equal_to(x2, 2)`: `x2.dim() >= 2`.
> - `tensors_have_same_size_at_dims(x1, x1.dim()-1, x2, x2.dim()-1)`: the last
>   (innermost, feature) dimension of `x1` and of `x2` must have equal extent,
>   i.e. `x1.size(x1.dim()-1) == x2.size(x2.dim()-1)`.
> - `p >= 0`: the norm order must be non-negative; on failure logs
>   `"cdist only supports non-negative p values; p = %.6f"` with `p`.
> - If `compute_mode` is present (optional has a value), let `mode =
>   compute_mode.value()`; require `mode >= 0 && mode <= 2` (the three valid
>   modes 0, 1, 2). On failure logs
>   `"possible modes: 0, 1, 2, but was: %<PRId64>"` with `mode`. If
>   `compute_mode` is empty (nullopt), this check is skipped.
> - If every check passes, return `true`.
>
> Note: this function does NOT check `out`'s shape/rank beyond dtype; batch
> dimensions and the last dim of `x1`/`x2` being equal are the only structural
> constraints enforced here.

> [spec:et:def:distance-util.torch.executor.check-pdist-args-fn]
> bool check_pdist_args(const Tensor& in, double p, const Tensor& out)

> [spec:et:sem:distance-util.torch.executor.check-pdist-args-fn]
> Validates the input and output tensors for `pdist` (pairwise distances among
> the rows of a single 2-D matrix). Returns `bool`: `true` if all checks pass,
> `false` on the first failing check. Each check is a
> `ET_CHECK_OR_RETURN_FALSE`/`ET_LOG_AND_RETURN_IF_FALSE` guard that logs and
> returns `false` immediately without aborting; checks run in order and
> short-circuit:
> - `tensors_have_same_dtype(in, out)`: `in` and `out` must have the same
>   `ScalarType`. False otherwise.
> - `tensor_is_rank(in, 2)`: `in` must be exactly rank 2 (`in.dim() == 2`).
>   False otherwise.
> - `p >= 0`: the norm order must be non-negative; on failure logs
>   `"pdist only supports non-negative p values; p = %.6f"` with `p`.
> - If all pass, return `true`.
>
> Note: `out` is only dtype-checked here; its expected shape is computed
> separately by `[spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn]`.

> [spec:et:def:distance-util.torch.executor.get-pdist-out-target-size-fn]
> void get_pdist_out_target_size( const Tensor& in, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn]
> Computes the expected output shape of `pdist(in)` and writes it into the
> caller-provided buffers. Returns void.
>
> - Sets `*out_ndim = 1`: the pdist output is always a 1-D tensor.
> - Let `n = in.size(0)` (the number of rows of the rank-2 input; stored into a
>   `size_t`).
> - Sets `out_sizes[0] = n * (n - 1) / 2`: the number of unordered row pairs
>   `{i, j}` with `i < j` (the count of upper-triangle off-diagonal entries).
>   The multiplication `n * (n - 1)` is exact and then integer-divided by 2.
>   For `n == 0` or `n == 1` this yields 0 (an empty output). Arithmetic is
>   `size_t` (unsigned); with `n == 0`, `n - 1` wraps to `SIZE_MAX` but
>   `0 * (n-1) == 0`, so the result is still 0.
>
> Only `out_sizes[0]` is written (one dimension). Callers must have validated
> rank via `[spec:et:sem:distance-util.torch.executor.check-pdist-args-fn]`
> before calling; this function does not re-check.

> [spec:et:def:distance-util.torch.executor.l0]
> struct L0

> [spec:et:def:distance-util.torch.executor.l0.finish-fn]
> static inline CTYPE finish(const CTYPE& agg, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l0.finish-fn]
> `L0` (Hamming / count-of-nonzero distance) finalizer. Given the accumulated
> aggregate `agg` (the running count of nonzero coordinate differences) and the
> unused second argument (`p`, ignored), returns `agg` unchanged. No root or
> scaling is applied. See `[spec:et:sem:distance-util.torch.executor.pdist-fn]`
> for how map/reduce/finish compose.

> [spec:et:def:distance-util.torch.executor.l0.map-fn]
> static inline CTYPE map(const CTYPE& diff, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l0.map-fn]
> `L0` per-coordinate map. Takes `diff` (the already absolute-valued difference
> of two coordinates, computed by the caller as `std::abs(a - b)`) and an unused
> second argument (`p`, ignored). Returns `1` if `diff != 0`, else `0` — i.e.
> an indicator of whether the coordinate differs. The result has type `CTYPE`
> (0 or 1 as that type). Because `diff` is the absolute difference, this counts
> coordinates that are not exactly equal.

> [spec:et:def:distance-util.torch.executor.l0.reduce-fn]
> static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up)

> [spec:et:sem:distance-util.torch.executor.l0.reduce-fn]
> `L0` reduction. Combines the running aggregate `agg` with the newly mapped
> value `up` by returning `agg + up` (summation). Across all coordinates this
> sums the per-coordinate indicators, giving the count of differing
> coordinates. Addition is in `CTYPE`.

> [spec:et:def:distance-util.torch.executor.l1]
> struct L1

> [spec:et:def:distance-util.torch.executor.l1.finish-fn]
> static inline CTYPE finish(const CTYPE& agg, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l1.finish-fn]
> `L1` (Manhattan / taxicab distance) finalizer. Returns the aggregate `agg`
> (the summed absolute differences) unchanged; the second argument (`p`) is
> ignored. No root is applied.

> [spec:et:def:distance-util.torch.executor.l1.map-fn]
> static inline CTYPE map(const CTYPE& diff, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l1.map-fn]
> `L1` per-coordinate map. Returns `diff` unchanged (`diff` is the caller's
> absolute coordinate difference `std::abs(a - b)`); the second argument (`p`)
> is ignored. Identity map.

> [spec:et:def:distance-util.torch.executor.l1.reduce-fn]
> static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up)

> [spec:et:sem:distance-util.torch.executor.l1.reduce-fn]
> `L1` reduction. Returns `agg + up` (summation) in `CTYPE`. Accumulates the
> absolute differences into the total.

> [spec:et:def:distance-util.torch.executor.l2]
> struct L2

> [spec:et:def:distance-util.torch.executor.l2.finish-fn]
> static inline CTYPE finish(const CTYPE& agg, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l2.finish-fn]
> `L2` (Euclidean distance) finalizer. Returns `std::sqrt(agg)` where `agg` is
> the accumulated sum of squared differences; the second argument (`p`) is
> ignored. `std::sqrt` follows IEEE semantics for the floating-point `CTYPE`:
> `sqrt(0) == 0`, and `sqrt` of a negative value would be NaN (not reachable
> here since squares are non-negative). Applies the square root exactly once
> at the end.

> [spec:et:def:distance-util.torch.executor.l2.map-fn]
> static inline CTYPE map(const CTYPE& diff, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.l2.map-fn]
> `L2` per-coordinate map. Returns `diff * diff` (the square of the absolute
> coordinate difference); the second argument (`p`) is ignored. Since it is
> squared, the prior absolute value is redundant but harmless. Result is
> non-negative in `CTYPE`.

> [spec:et:def:distance-util.torch.executor.l2.reduce-fn]
> static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up)

> [spec:et:sem:distance-util.torch.executor.l2.reduce-fn]
> `L2` reduction. Returns `agg + up` (summation) in `CTYPE`. Accumulates the
> squared differences into the sum of squares.

> [spec:et:def:distance-util.torch.executor.linf]
> struct Linf

> [spec:et:def:distance-util.torch.executor.linf.finish-fn]
> static inline CTYPE finish(const CTYPE& agg, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.linf.finish-fn]
> `Linf` (Chebyshev / maximum-coordinate distance) finalizer. Returns the
> aggregate `agg` (the running maximum of absolute differences) unchanged; the
> second argument (`p`) is ignored.

> [spec:et:def:distance-util.torch.executor.linf.map-fn]
> static inline CTYPE map(const CTYPE& diff, const CTYPE&)

> [spec:et:sem:distance-util.torch.executor.linf.map-fn]
> `Linf` per-coordinate map. Returns `diff` unchanged (the caller's absolute
> coordinate difference); the second argument (`p`) is ignored. Identity map.

> [spec:et:def:distance-util.torch.executor.linf.reduce-fn]
> static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up)

> [spec:et:sem:distance-util.torch.executor.linf.reduce-fn]
> `Linf` reduction. Returns `std::max(agg, up)` — the larger of the running
> aggregate and the newly mapped absolute difference. Across all coordinates
> this yields the maximum absolute difference. Note the aggregate starts at 0
> in `[spec:et:sem:distance-util.torch.executor.pdist-fn]`, and all mapped
> values (absolute differences) are non-negative, so the initial 0 never
> exceeds a real difference. `std::max` returns the first argument on a tie.

> [spec:et:def:distance-util.torch.executor.lp]
> struct Lp

> [spec:et:def:distance-util.torch.executor.lp.finish-fn]
> static inline CTYPE finish(const CTYPE& agg, const CTYPE& p)

> [spec:et:sem:distance-util.torch.executor.lp.finish-fn]
> `Lp` (general p-norm, for `p` not in {0, 1, 2, INF}) finalizer. Returns
> `std::pow(agg, 1.0 / p)` — the `p`-th root of the accumulated sum of
> `|diff|^p`. Here `agg` is `CTYPE` and `p` is passed as `CTYPE` (converted
> from the `double` `p` at the pdist call site). `1.0 / p` is computed in
> `double`. Follows `std::pow` semantics; with the valid non-negative `p`
> guaranteed by the caller (and `p != 0`, since `p == 0` dispatches to `L0`),
> `1.0 / p` is finite.

> [spec:et:def:distance-util.torch.executor.lp.map-fn]
> static inline CTYPE map(const CTYPE& diff, const CTYPE& p)

> [spec:et:sem:distance-util.torch.executor.lp.map-fn]
> `Lp` per-coordinate map. Returns `std::pow(diff, p)` — the absolute
> coordinate difference `diff` raised to the power `p` (`p` as `CTYPE`).
> Follows `std::pow` semantics; `diff` is non-negative (absolute value) so the
> power is well-defined for real `p`.

> [spec:et:def:distance-util.torch.executor.lp.reduce-fn]
> static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up)

> [spec:et:sem:distance-util.torch.executor.lp.reduce-fn]
> `Lp` reduction. Returns `agg + up` (summation) in `CTYPE`. Accumulates the
> `|diff|^p` terms into the sum before the final root in
> `[spec:et:sem:distance-util.torch.executor.lp.finish-fn]`.

> [spec:et:def:distance-util.torch.executor.pdist-fn]
> void pdist(const Tensor& in, Tensor& out, double p)

> [spec:et:sem:distance-util.torch.executor.pdist-fn]
> Templated `pdist<CTYPE, Norm>(in, out, p)` worker: computes the pairwise
> `p`-norm distance between every unordered pair of rows of the rank-2 input
> `in` and writes the results, in upper-triangular row-major order, into the
> 1-D output `out`. `CTYPE` is the element C type; `Norm` is one of the norm
> policy structs (`L0`, `L1`, `L2`, `Linf`, `Lp`) selected by the dispatcher
> below. Returns void.
>
> Data access:
> - `in_data = in.const_data_ptr<CTYPE>()`: read-only pointer to the input's
>   contiguous data (assumes default/contiguous dim order; row `i` starts at
>   offset `i * m`).
> - `out_data = out.mutable_data_ptr<CTYPE>()`: writable pointer to the output.
> - `n = in.size(0)` (number of rows), `m = in.size(1)` (row length / features).
>
> Algorithm — enumerate row pairs `(i, j)` with `i < j` in lexicographic order
> and write one distance per pair into consecutive output slots:
> - `out_ix = 0`.
> - For `i = 0 .. n-1`:
>   - For `j = i+1 .. n-1`:
>     - `row_i = in_data + i*m`, `row_j = in_data + j*m`.
>     - `agg = 0` (the aggregate, type `CTYPE`).
>     - For `k = 0 .. m-1`:
>       - `diff = std::abs(row_i[k] - row_j[k])` (absolute coordinate
>         difference in `CTYPE`).
>       - `agg = Norm::reduce(agg, Norm::map(diff, p_as_CTYPE))` — map the
>         difference then fold it into the aggregate per the selected norm
>         (see the `map`/`reduce` rules for each norm). `p` is passed through as
>         `CTYPE` (`Norm::map`/`finish` ignore it except for `Lp`).
>     - `out_data[out_ix++] = Norm::finish(agg, p_as_CTYPE)` — finalize (e.g.
>       sqrt for L2, p-th root for Lp) and store, then advance the output index.
>
> The total number of writes is `n*(n-1)/2`, matching
> `[spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn]`.
> For `m == 0` (zero-width rows) each `agg` stays 0 and `finish(0, p)` is
> written (0 for L0/L1/L2/Linf; `std::pow(0, 1/p) == 0` for Lp). For `n <= 1`
> no pairs exist and nothing is written. No bounds/dtype validation happens
> here; callers must have passed
> `[spec:et:sem:distance-util.torch.executor.check-pdist-args-fn]` and resized
> `out` first.
>
> Norm selection (the non-annotated `pdist<CTYPE>(in, out, p)` overload that
> wraps this worker) dispatches on the runtime `double p`, comparing `p`
> against exact `double` values: `p == 0.0` -> `L0`; `p == 1.0` -> `L1`;
> `p == 2.0` -> `L2`; `p == INFINITY` -> `Linf`; otherwise -> `Lp` (general
> case). The comparison is exact floating-point equality.

