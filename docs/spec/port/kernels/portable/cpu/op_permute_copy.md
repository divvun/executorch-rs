# kernels/portable/cpu/op_permute_copy.cpp

> [spec:et:def:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn]
> void increment_coordinate_permuted( const Tensor& tensor, size_t* const coordinate, IntArrayRef dims)

> [spec:et:sem:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn]
> Advances an input-space coordinate `coordinate` by one output-space step, iterating the input dims in the order given by the permutation `dims` (so the output is traversed in row-major order while indexing into the input).
>
> Steps: iterate `i` from `dims.size()-1` down to 0: resolve the target input dim `d = dims[i] >= 0 ? dims[i] : dims[i] + tensor.dim()` (negative dims wrap); increment `coordinate[d]`; if `coordinate[d]` now equals `tensor.size(d)`, reset `coordinate[d] = 0` and carry to the next entry of `dims`; otherwise return immediately. `dims[dims.size()-1]` is the fastest-varying output dim. Mutates `coordinate` in place; no return value.

> [spec:et:def:op-permute-copy.torch.executor.native.permute-copy-out-fn]
> Tensor& permute_copy_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef dims, Tensor& out)

> [spec:et:sem:op-permute-copy.torch.executor.native.permute-copy-out-fn]
> Copies `in` into `out` with its dimensions reordered by the permutation `dims` (`out.size(k) == in.size(dims[k])`); returns `out`. Same dtype in and out.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_permute_copy_args(in, dims, out)` per `[spec:et:sem:copy-ops-util.torch.executor.check-permute-copy-args-fn]` (validates `dims` is a valid permutation of `in`'s axes, correct length, dtype match); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else InvalidArgument, return `out`.
> 3. Compute `expected_out_size`/`expected_out_dim` via `get_permute_copy_out_target_size` per `[spec:et:sem:copy-ops-util.torch.executor.get-permute-copy-out-target-size-fn]` (out shape = `in`'s sizes indexed by `dims`).
> 4. ET_KERNEL_CHECK: `resize_tensor(out, {expected_out_size, expected_out_dim})` == Ok; else InvalidArgument, return `out`.
> 5. Initialize input coordinate `in_coord[..]` to all zeros; memoize `in`'s trailing-dim strides via `memoizeTrailingDims(in, trailing_dims_memo)`.
> 6. Dispatch on `out.scalar_type()` over ALL types (ET_SWITCH_ALL_TYPES — every ScalarType including Bool and complex); an unsupported type sets InvalidArgument and returns `out`.
> 7. For each output flat index `i` in [0,out.numel()) (row-major over `out`): `out_data[i] = in_data[ coordinateToIndexWithTrailingDimsMemo(in, in_coord, trailing_dims_memo) ]` (converts the current input coordinate to `in`'s flat offset), then advance `in_coord` via `increment_coordinate_permuted(in, in_coord, dims)` per `[spec:et:sem:op-permute-copy.torch.executor.native.increment-coordinate-permuted-fn]`. This walks `out` in order while gathering the correspondingly-permuted elements from `in`.
> 8. Return `out`.

