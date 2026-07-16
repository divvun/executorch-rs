# kernels/portable/cpu/op_detach_copy.cpp

> [spec:et:def:op-detach-copy.torch.executor.native.detach-copy-out-fn]
> Tensor&

> [spec:et:sem:op-detach-copy.torch.executor.native.detach-copy-out-fn]
> Byte-for-byte copy of `self` into `out` (both assumed to have the same dtype and shape). Returns
> `out`. Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK_MSG `resize_tensor(out, self.sizes()) == Error::Ok`, else InvalidArgument
>    with message "Failed to resize output tensor."
> 2. ET_KERNEL_CHECK `tensors_have_same_dim_order(self, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_shape_and_dtype(self, out)`, else InvalidArgument.
> 4. If `self.nbytes() > 0`: `memcpy(out.data, self.data, self.nbytes())`. (The guard is important:
>    a numel-0 tensor may legitimately have a null data pointer, and passing null to memcpy is
>    undefined behavior in some environments even for size 0.)
> 5. Return `out`. (No dtype dispatch: this is a raw byte copy of any dtype.)

