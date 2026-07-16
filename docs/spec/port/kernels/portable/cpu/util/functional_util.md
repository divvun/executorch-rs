# kernels/portable/cpu/util/functional_util.h

> [spec:et:def:functional-util.torch.executor.apply-unary-map-fn-fn]
> inline void apply_unary_map_fn( const MapOp& map_fun, const CTYPE_IN* const data_in, CTYPE_OUT* const data_out, const int64_t size, const int64_t stride = 1)

> [spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]
> Applies a unary map function `map_fun` elementwise to `size` strided
> elements of `data_in`, writing each result into `data_out` at the same
> strided position. `stride` defaults to 1.
>
> Algorithm: for each logical index `i` in the half-open range `[0, size)`,
> compute `data_out[i * stride] = map_fun(data_in[i * stride])`. Both the
> input read and the output write use the same `i * stride` offset, so input
> and output are expected to have identical strided layouts. `map_fun` takes
> one `CTYPE_IN` value and returns a `CTYPE_OUT` value; the returned value is
> stored directly (any implicit conversion is whatever the map function's
> return type dictates).
>
> The iteration is dispatched through `executorch::extension::parallel_for`
> over the range `[0, size)` with grain size
> `executorch::extension::internal::GRAIN_SIZE`. The work may be split into
> contiguous sub-ranges `[begin, end)` executed on separate threads (or a
> single range when parallelism is unavailable); within each sub-range the
> elements are processed in ascending index order. Because each output slot
> `i * stride` is written by exactly one `i`, the result is independent of how
> the range is partitioned. A Rust port may compute the same result with a
> plain ascending loop over `[0, size)`, optionally parallelized when `size`
> exceeds the grain size.
>
> No bounds checking, dtype dispatch, or resizing is performed here; the
> caller is responsible for having allocated `size * stride`-addressable input
> and output buffers. When `size == 0` no elements are read or written.
> Returns nothing (writes in place into `data_out`).

> [spec:et:def:functional-util.torch.executor.apply-unary-map-reduce-fn-fn]
> inline CTYPE_OUT apply_unary_map_reduce_fn( const MapOp& map_fun, const ReduceOp& reduce_fun, const CTYPE_IN* const data_in, const int64_t size, const int64_t stride = 1)

> [spec:et:sem:functional-util.torch.executor.apply-unary-map-reduce-fn-fn]
> Applies a unary map function `map_fun` to `size` strided elements of
> `data_in` and folds the mapped values together with a binary reduction
> `reduce_fun`, returning the single accumulated `CTYPE_OUT` value. `stride`
> defaults to 1.
>
> Algorithm:
> 1. Seed the accumulator: `acc_val = map_fun(data_in[0])` (the element at
>    offset 0 is always used regardless of `stride`).
> 2. For each `i` in `[1, size)` in strictly ascending order, update
>    `acc_val = reduce_fun(map_fun(data_in[i * stride]), acc_val)`. Note the
>    argument order: the newly mapped element is the first argument and the
>    running accumulator is the second.
> 3. Return `acc_val`.
>
> The map result type is `CTYPE_OUT`; `reduce_fun` takes two `CTYPE_OUT`
> values and returns `CTYPE_OUT`. This is a sequential left fold and is NOT
> parallelized (unlike `[spec:et:sem:functional-util.torch.executor.apply-unary-map-fn-fn]`);
> the reduction order is deterministic and left-to-right, which matters for
> non-associative floating-point reductions.
>
> Precondition: `size >= 1`; `data_in[0]` is unconditionally dereferenced, so
> calling with `size == 0` is undefined (the caller must guarantee at least
> one element). No dtype dispatch, bounds checking, or resizing is performed.

> [spec:et:def:functional-util.torch.executor.apply-unary-reduce-fn-fn]
> inline CTYPE apply_unary_reduce_fn( const ReduceOp& reduce_fun, const CTYPE* const data_in, const int64_t size, const int64_t stride = 1)

> [spec:et:sem:functional-util.torch.executor.apply-unary-reduce-fn-fn]
> Folds `size` strided elements of `data_in` together with a binary reduction
> `reduce_fun` and returns the single accumulated `CTYPE` value. No mapping is
> applied to the elements. `stride` defaults to 1.
>
> Algorithm:
> 1. Seed the accumulator: `acc_val = data_in[0]` (offset 0, regardless of
>    `stride`).
> 2. For each `i` in `[1, size)` in strictly ascending order, update
>    `acc_val = reduce_fun(data_in[i * stride], acc_val)`. The current element
>    is the first argument and the running accumulator is the second.
> 3. Return `acc_val`.
>
> This is the identity-map special case of
> `[spec:et:sem:functional-util.torch.executor.apply-unary-map-reduce-fn-fn]`.
> `reduce_fun` takes two `CTYPE` values and returns `CTYPE`. The fold is
> sequential, deterministic, and left-to-right (not parallelized), which is
> significant for non-associative floating-point reductions.
>
> Precondition: `size >= 1`; `data_in[0]` is unconditionally dereferenced, so
> `size == 0` is undefined behavior and must be prevented by the caller. No
> dtype dispatch, bounds checking, or resizing is performed.

