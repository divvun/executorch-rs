# kernels/portable/cpu/op_bitwise_not.cpp

> [spec:et:def:op-bitwise-not.torch.executor.native.bitwise-not-out-fn]
> Tensor&

> [spec:et:sem:op-bitwise-not.torch.executor.native.bitwise-not-out-fn]
> Computes the elementwise bitwise NOT of `in` into `out`. For Bool tensors this
> is the logical NOT; for integral tensors this is the C++ `~` operator. The
> context is unused for validation logging only.
>
> Steps:
> 1. Resize `out` to `in.sizes()`; if resize does not return Error::Ok set
>    Error::InvalidArgument (message "Failed to resize output tensor.") and
>    return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dtype(in, out)` — input and output must
>    share the same dtype (no promotion). On failure set Error::InvalidArgument
>    and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Dtype handling over `in.numel()` elements in flat order:
>    - If `in` dtype is Bool: write `!val_in` for each element (logical NOT).
>    - Else if `in` is an integral type excluding Bool ({Byte, Char, Short, Int,
>      Long}, ET_SWITCH_INT_TYPES): write `~val_in` (bitwise complement in that
>      integer type).
>    - Otherwise (floating/other dtype): ET_KERNEL_CHECK_MSG with a false
>      condition — set Error::InvalidArgument (message "Unsupported input dtype
>      <n>") and return `out` unchanged.
> 5. Return `out`.

