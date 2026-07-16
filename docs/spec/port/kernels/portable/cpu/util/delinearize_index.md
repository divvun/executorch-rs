# kernels/portable/cpu/util/delinearize_index.cpp

> [spec:et:def:delinearize-index.torch.executor.delinearize-index-fn]
> void delinearize_index( size_t linear_index, executorch::aten::ArrayRef<Tensor::SizesType> shape, size_t* out_indexes, const size_t out_indexes_len)

> [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn]
> Converts a single flattened (row-major / C-contiguous) `linear_index` into
> its per-dimension coordinate indexes for a tensor of the given `shape`, and
> writes them into the caller-provided `out_indexes` buffer.
>
> Arguments:
> - `linear_index`: the flattened offset (unsigned, `size_t`), assumed to be in
>   range `[0, product(shape))`; the function does not itself range-check it.
> - `shape`: an `ArrayRef<Tensor::SizesType>` giving the size of each dimension,
>   ordered from outermost (dim 0) to innermost (last dim). `Tensor::SizesType`
>   is a signed 32-bit integer type, but each entry is treated as a positive
>   extent.
> - `out_indexes`: pointer to a caller-owned array that receives one coordinate
>   per dimension; `out_indexes[d]` is set to the index along dimension `d`.
> - `out_indexes_len`: the capacity (element count) of `out_indexes`.
>
> Precondition check: assert `shape.size() <= out_indexes_len` via `ET_CHECK`.
> On failure this aborts the program (fatal check, not a recoverable Error);
> the Rust port should treat a `shape` longer than the output buffer as a
> programming error (panic / debug-assert). Only the first `shape.size()`
> entries of `out_indexes` are written; any remaining entries up to
> `out_indexes_len` are left untouched.
>
> Algorithm — iterate dimensions from innermost to outermost. For
> `i = 0 .. shape.size()-1`:
> - `dim = shape.size() - 1 - i` (so the first iteration handles the last /
>   innermost dimension and the last iteration handles dim 0).
> - `dim_size = shape[dim]`.
> - `out_indexes[dim] = linear_index % dim_size` (the coordinate along that
>   dimension is the current remainder modulo that dimension's extent).
> - `linear_index /= dim_size` (integer division; carry the quotient into the
>   next, more-significant dimension).
> All arithmetic is unsigned `size_t` modulo/division. A `dim_size` of 0 would
> divide by zero (undefined) and is not guarded against; callers never pass a
> zero-extent dimension here.
>
> This is the exact inverse of the row-major flattening
> `linear_index = sum_d out_indexes[d] * stride[d]` where
> `stride[d] = product(shape[d+1 ..])`. For a rank-0 (empty `shape`) tensor the
> loop body never executes and no output is written; `linear_index` is ignored.
>
> Returns void; the result is delivered through the `out_indexes` buffer.
>
> Overload `delinearize_index(linear_index, const Tensor& t, out_indexes,
> out_indexes_len)`: a thin forwarder that calls the primary overload with
> `t.sizes()` as `shape`; identical behavior for the tensor's shape.
