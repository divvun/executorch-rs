# kernels/portable/cpu/op_unfold_copy.cpp

> [spec:et:def:op-unfold-copy.torch.executor.native.unfold-copy-out-fn]
> Tensor& unfold_copy_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, int64_t size, int64_t step, Tensor& out)

> [spec:et:sem:op-unfold-copy.torch.executor.native.unfold-copy-out-fn]
> Extracts sliding windows of length `size` with hop `step` along dimension `dim`
> of `self`, appending a new trailing window dimension. Implements
> `unfold_copy(Tensor self, int dimension, int size, int step, *, Tensor(a!) out)`.
> The output has the same shape as `self` except `self.size(dim)` becomes
> `floor((self.size(dim) - size)/step) + 1`, plus a new last dimension of length
> `size`. Step by step:
>
> - ET_KERNEL_CHECK `check_unfold_copy_args(self, dim, size, step)` (see
>   `[spec:et:sem:copy-ops-util...check-unfold-copy-args-fn]`): validates `dim`
>   valid, `size >= 0`, `step >= 1`, and `size <= self.size(dim)`. On failure
>   sets Error::InvalidArgument and returns `out` unchanged.
> - Normalize `dim`: if `dim < 0`, `dim += nonzero_dim(self)`.
> - Compute `expected_output_size`/`expected_out_dim` via
>   `get_unfold_copy_out_target_size` and resize `out`; on failure
>   Error::InvalidArgument.
> - Let `leading_dims = getLeadingDims(self, dim)`, `trailing_dims =
>   getTrailingDims(self, dim)`.
> - Dtype dispatch: in_type and out_type from REALHBBF16 = {Byte, Char, Short,
>   Int, Long, Bool, Half, Float, Double, BFloat16} as CTYPE_IN / CTYPE_OUT.
> - Fill `out` sequentially (`out_ptr` advances by 1 per write). For each
>   `i` in `[0, leading_dims)`: `src = input_ptr + i*self.size(dim)*trailing_dims`.
>   For each window `j` in `[0, out.size(dim))`: `dim_src = src +
>   j*step*trailing_dims`. For each `k` in `[0, trailing_dims)`: for each `l` in
>   `[0, size)`: write `*out_ptr = convert<CTYPE_OUT,CTYPE_IN>(dim_src[k +
>   l*trailing_dims])` then `out_ptr++`. So the innermost (new) dimension of
>   length `size` varies fastest, gathering the `size` consecutive elements of
>   each window along `dim` (spaced by `trailing_dims`).
> - Returns `out`.

