# kernels/portable/cpu/op_relu.cpp

> [spec:et:def:op-relu.torch.executor.native.relu-out-fn]
> Tensor& relu_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-relu.torch.executor.native.relu-out-fn]
> Elementwise ReLU into `out`. Steps:
>
> - Resize `out` to `in.sizes()` via `resize_tensor`; on failure set
>   `Error::InvalidArgument` (message "Failed to resize output tensor.") and
>   return `out`.
> - ET_KERNEL_CHECK: `in` and `out` must have the same shape AND same dtype
>   (`tensors_have_same_shape_and_dtype`); else `Error::InvalidArgument`, return
>   `out`.
> - ET_KERNEL_CHECK: `out` dtype must be realhbf16 (`tensor_is_realhbf16_type`),
>   i.e. REALHBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double,
>   BFloat16} — Bool is excluded; else `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Dispatch on `in.scalar_type()` over REALHBF16. Elementwise over `in.numel()`
>   elements in flat order at the single ctype CTYPE (in and out share dtype):
>   for each `val_in`, if `utils::isnan_override(val_in)` is true (see
>   `[spec:et:sem:math-util.torch.executor.native.utils.isnan-override]`) OR
>   `val_in >= CTYPE(0)`, write `val_in`; otherwise write `CTYPE(0)`. So NaN is
>   propagated unchanged, negatives (and negative zero, since `-0 >= 0`) map to
>   `+0`, non-negatives pass through.
> - Returns `out`.

