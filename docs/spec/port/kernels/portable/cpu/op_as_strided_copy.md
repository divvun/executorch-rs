# kernels/portable/cpu/op_as_strided_copy.cpp

> [spec:et:def:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn]
> Tensor& as_strided_copy_out( KernelRuntimeContext& ctx, const Tensor& in, ArrayRef<int64_t> size, ArrayRef<int64_t> stride, optional<int64_t> storage_offset, Tensor& out)

> [spec:et:sem:op-as-strided-copy.torch.executor.native.as-strided-copy-out-fn]
> Implements `as_strided_copy.out(in, size, stride, storage_offset, out)`:
> materializes a strided view of `in`'s underlying storage — shape `size`,
> strides `stride`, starting at `storage_offset` elements into `in`'s data — into
> a contiguous `out`. `ctx` unused for control flow.
>
> Validation (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns `out`):
> 1. `check_as_strided_copy_args(in, size, stride, storage_offset, out)` —
>    validates `size`/`stride` have equal length, strides non-negative, and that
>    the largest addressed element stays within `in`'s storage.
> 2. `resize_tensor(out, size) == Error::Ok`.
> 3. `tensors_have_same_dim_order(in, out)`.
> 4. `tensor_is_default_dim_order(in)`.
>
> If `in.numel() == 0`, returns `out` immediately.
>
> `offset = storage_offset.has_value() ? storage_offset.value() : 0`.
>
> Dtype dispatch: `ET_SWITCH_ALL_TYPES` on `in.scalar_type()` (all scalar
> types); `out` uses the same CTYPE. Calls `as_strided_copy<CTYPE>(in, size,
> stride, offset, out)`, which, for each element of the logical output shape
> `size`, computes its multi-index, maps it to a flat source offset `offset +
> sum(index[d] * stride[d])` into `in`'s data, and writes that value contiguously
> into `out`.
>
> Returns `out`.

