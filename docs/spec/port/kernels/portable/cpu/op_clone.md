# kernels/portable/cpu/op_clone.cpp

> [spec:et:def:op-clone.torch.executor.native.clone-out-fn]
> Tensor& clone_out( KernelRuntimeContext& context, const Tensor& self, std::optional<executorch::aten::MemoryFormat> memory_format, Tensor& out)

> [spec:et:sem:op-clone.torch.executor.native.clone-out-fn]
> Copies `self` byte-for-byte into `out` (a deep copy). Signature:
> `clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out)`.
> The context is unused for computation.
>
> Steps:
> 1. Resize `out` to `self.sizes()`; if resize does not return Error::Ok set
>    Error::InvalidArgument and return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_shape_and_dtype(self, out)` — `out` must
>    match `self` in both shape and dtype. On failure set Error::InvalidArgument
>    and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(self, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. ET_KERNEL_CHECK: `memory_format` must be either absent (nullopt) or equal to
>    `MemoryFormat::Contiguous`; any other memory format sets
>    Error::InvalidArgument and returns `out` unchanged (only contiguous is
>    supported).
> 5. If `self.nbytes() > 0`, memcpy `self.nbytes()` bytes from `self`'s data
>    pointer into `out`'s data pointer. The zero-byte case is skipped
>    deliberately because a numel-0 tensor may have a null data pointer and
>    passing null to memcpy is invalid in some environments.
> 6. Return `out`.

