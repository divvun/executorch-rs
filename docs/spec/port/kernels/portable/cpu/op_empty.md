# kernels/portable/cpu/op_empty.cpp

> [spec:et:def:op-empty.torch.executor.native.empty-out-fn]
> Tensor& empty_out( KernelRuntimeContext& context, IntArrayRef size, std::optional<executorch::aten::MemoryFormat> memory_format, Tensor& out)

> [spec:et:sem:op-empty.torch.executor.native.empty-out-fn]
> Implements `empty.out(size, *, out)`: resizes `out` to `size` and leaves its contents
> uninitialized (no data is written). `memory_format` is ignored.
> 1. ET_KERNEL_CHECK_MSG `resize_tensor(out, size) == Error::Ok`, else set `Error::InvalidArgument`
>    on `ctx` and return `out` (message "Failed to resize output tensor.").
> 2. Return `out`. The out tensor's element values are not initialized; only its shape is set.

