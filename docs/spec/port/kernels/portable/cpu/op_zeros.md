# kernels/portable/cpu/op_zeros.cpp

> [spec:et:def:op-zeros.torch.executor.native.check-sizes-fn]
> bool check_sizes( executorch::aten::ArrayRef<int64_t> size_int64_t, executorch::aten::ArrayRef<int32_t> size_int32_t)

> [spec:et:sem:op-zeros.torch.executor.native.check-sizes-fn]
> Predicate checking that a requested `int64_t` size list matches the resized
> tensor's `int32_t` size list, element for element. Returns bool.
>
> 1. `ET_LOG_AND_RETURN_IF_FALSE(size_int64_t.size() == size_int32_t.size())`:
>    if the two lists differ in length, log and return `false`.
> 2. For each `i` in [0, size_int64_t.size()): compare
>    `(int64_t)size_int32_t[i] == size_int64_t[i]`; on mismatch log and return
>    `false`.
> 3. Return `true`.
> This guards against `int64` requested dimensions being truncated when stored
> as the tensor's `int32` `SizesType`.

> [spec:et:def:op-zeros.torch.executor.native.zeros-out-fn]
> Tensor& zeros_out(KernelRuntimeContext& ctx, IntArrayRef size, Tensor& out)

> [spec:et:sem:op-zeros.torch.executor.native.zeros-out-fn]
> Entry point for `zeros.out(size, *, out)`. Resizes `out` to `size` and fills
> it with zeros. Returns `out`.
>
> 1. `ET_KERNEL_CHECK_MSG(resize_tensor(out, size) == Error::Ok, InvalidArgument,
>    "Failed to resize output tensor.")`: resize `out` to the requested `size`;
>    on failure sets InvalidArgument on `ctx` and returns `out`.
> 2. `ET_KERNEL_CHECK(check_sizes(size, out.sizes()), InvalidArgument)` per
>    `[spec:et:sem:op-zeros.torch.executor.native.check-sizes-fn]`: confirms the
>    resized tensor's sizes match the requested `int64` sizes; on failure sets
>    InvalidArgument and returns `out`.
> 3. Get `out_data = out.mutable_data_ptr()`. If it is non-null, `memset(out_data,
>    0, out.nbytes())` — writes zero bytes across the entire (contiguous)
>    storage, which is the correct all-zeros representation for every supported
>    dtype. A null data pointer (e.g. zero-element tensor) skips the memset.
> 4. Return `out`. Note: the output dtype is whatever `out` already is; this
>    kernel does not restrict or dispatch on dtype.

