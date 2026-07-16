# kernels/portable/cpu/op_cat.cpp

> [spec:et:def:op-cat.torch.executor.native.cat-out-fn]
> Tensor& cat_out( KernelRuntimeContext& ctx, executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, Tensor& out)

> [spec:et:sem:op-cat.torch.executor.native.cat-out-fn]
> Concatenates the list `tensors` along dimension `dim` into `out`.
>
> Steps:
> 1. Normalize `dim`: if `dim < 0`, set `dim += out.dim()`.
> 2. ET_KERNEL_CHECK: `check_cat_args(tensors, dim, out)` (non-empty inputs match
>    in all dims except `dim`, share dtype-compatibility with `out`, `dim` in
>    range, dim orders consistent). On failure set Error::InvalidArgument on the
>    context and return `out` unchanged.
> 3. Compute `expected_out_size`/`expected_out_dim` via
>    `get_cat_out_target_size(tensors, dim, ...)` (same shape as inputs except
>    size along `dim` = sum of the inputs' sizes along `dim`), then resize `out`;
>    if resize fails set Error::InvalidArgument and return `out` unchanged.
> 4. Special case: if every tensor in the list is a 1D empty tensor (numel == 0
>    and dim == 1), return `out` immediately (aten consistency), skipping the
>    copy.
> 5. Compute copy geometry: `outer = getLeadingDims(out, dim)` (product of dims
>    before `dim`), `dim_stride = getTrailingDims(out, dim)` (product of dims
>    after `dim`), `ninputs = tensors.size()`.
> 6. If `out` is a complex type: require every input's dtype to equal `out`'s
>    dtype (ET_KERNEL_CHECK; on mismatch set Error::InvalidArgument and return
>    `out` unchanged), then dispatch over COMPLEXH. For each outer index i in
>    [0, outer) and each input j in [0, ninputs): if `tensors[j].numel() == 0`
>    stop this dispatch (the lambda `return`s early, ending the whole copy);
>    otherwise `inner = tensors[j].size(dim) * dim_stride`, memcpy `inner`
>    elements from the input slice at `i * inner` to the running `out` pointer,
>    and advance the `out` pointer by `inner`.
> 7. Otherwise (`out` real/half/bool/bfloat16): dispatch output over REALHBBF16.
>    For each outer index i and each input j, dispatch the input's dtype over
>    REALHBBF16; if `tensors[j].numel() == 0` skip that input; else compute
>    `inner = tensors[j].size(dim) * dim_stride` and copy the input slice at
>    `i * inner`: if input and output element sizes are equal use memcpy,
>    otherwise elementwise `static_cast<CTYPE_OUT>` each of the `inner` values.
>    Advance the `out` pointer by `inner`. This concatenates each input's
>    dim-slice contiguously per outer block, promoting element types via cast
>    where dtypes differ.
> 8. Return `out`.

