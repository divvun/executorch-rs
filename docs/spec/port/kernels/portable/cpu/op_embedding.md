# kernels/portable/cpu/op_embedding.cpp

> [spec:et:def:op-embedding.torch.executor.native.embedding-kernel-fn]
> void embedding_kernel( KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& indices, Tensor& out)

> [spec:et:sem:op-embedding.torch.executor.native.embedding-kernel-fn]
> Templated on `CTYPE` (the index element type, either Long or Int). Gathers rows of the 2D `weight`
> lookup table `[num_embeddings, embedding_dim]` selected by the flattened `indices`, writing them
> consecutively into `out`.
> 1. `nbytes_per_entry = weight.size(1) * weight.element_size()` (bytes per embedding row).
> 2. `w_data` = raw byte pointer to `weight`; `out_data` = raw byte pointer to `out`;
>    `indices_ptr` = `CTYPE*` to `indices`; `weight_height = weight.size(0)`;
>    `indices_numel = indices.numel()`.
> 3. For `i` in `[0, indices_numel)`:
>    - ET_KERNEL_CHECK_MSG `indices_ptr[i] < weight_height`, else InvalidArgument (on failure the
>      kernel returns void immediately, leaving `out` partially written); message
>      "indices_ptr[i] >= weight.size(0)".
>    - ET_KERNEL_CHECK_MSG `indices_ptr[i] >= 0`, else InvalidArgument (returns void); message
>      "indices_ptr[i] < 0".
>    - If `w_data != nullptr`: `memcpy(out_data, w_data + nbytes_per_entry * indices_ptr[i],
>      nbytes_per_entry)` (copy the selected row).
>    - Advance `out_data += nbytes_per_entry`.
> Note: raw byte copy independent of the weight's floating/integer dtype; only the index dtype is
> dispatched.

> [spec:et:def:op-embedding.torch.executor.native.embedding-out-fn]
> Tensor& embedding_out( KernelRuntimeContext& ctx, const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, Tensor& out)

> [spec:et:sem:op-embedding.torch.executor.native.embedding-out-fn]
> Entry point for `embedding.out(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
> sparse=False, *, out)`. The `padding_idx`, `scale_grad_by_freq`, and `sparse` arguments are ignored
> (forward-only lookup). Every ET_KERNEL_CHECK failure sets the error on `ctx` and returns `out`
> unchanged.
> 1. ET_KERNEL_CHECK `check_embedding_args(weight, indices, out)` (validates `weight` is 2D, dtypes
>    consistent, etc.), else InvalidArgument.
> 2. ET_KERNEL_CHECK `resize_embedding_output(weight, indices, out) == Error::Ok` (resizes `out` to
>    `indices.sizes()` with an appended `weight.size(1)` dim), else InvalidArgument.
> 3. ET_KERNEL_CHECK_MSG `out.size(out.dim()-1) == weight.size(1)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `tensors_have_same_dim_order(weight, indices, out)`, else InvalidArgument.
> 5. ET_KERNEL_CHECK `tensor_is_default_dim_order(weight)`, else InvalidArgument.
> 6. `ix_type = indices.scalar_type()`; ET_CHECK_MSG `ix_type == Long || ix_type == Int` (this is a
>    hard ET_CHECK / abort, not a graceful context error).
> 7. Dispatch over `ix_type` between Long and Int (`ET_SWITCH_TWO_TYPES(Long, Int, ...)`) and call
>    `embedding_kernel<CTYPE>(ctx, weight, indices, out)` per
>    `[spec:et:sem:op-embedding.torch.executor.native.embedding-kernel-fn]`.
> 8. Return `out`.

