# kernels/portable/cpu/op_alias_copy.cpp

> [spec:et:def:op-alias-copy.torch.executor.native.alias-copy-out-fn]
> Tensor&

> [spec:et:sem:op-alias-copy.torch.executor.native.alias-copy-out-fn]
> Implements `alias_copy.out(in, out)`: a materialized copy of `in` into `out`
> (functional analog of a view/alias). `ctx` unused for control flow.
>
> Steps:
> 1. `resize_tensor(out, in.sizes()) == Error::Ok` (`ET_KERNEL_CHECK_MSG`; on
>    failure sets context Error to `InvalidArgument`, message "Failed to resize
>    output tensor.", returns `out`).
> 2. `tensors_have_same_dtype(in, out)` (`ET_KERNEL_CHECK` → `InvalidArgument`).
> 3. `tensors_have_same_dim_order(in, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`).
>
> If `in.nbytes() > 0`, performs a raw `memcpy` of `in.nbytes()` bytes from
> `in.const_data_ptr()` to `out.mutable_data_ptr()`. The `> 0` guard avoids
> passing a possibly-null pointer to `memcpy` for zero-element tensors (a
> zero-numel tensor may legitimately have a null data pointer). No dtype
> dispatch — byte-for-byte copy. Returns `out`.

