# kernels/portable/cpu/op_masked_scatter.cpp

> [spec:et:def:op-masked-scatter.torch.executor.native.masked-scatter-out-fn]
> Tensor& masked_scatter_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mask, const Tensor& src, Tensor& out)

> [spec:et:sem:op-masked-scatter.torch.executor.native.masked-scatter-out-fn]
> Copies elements from `src` into positions of (broadcast) `in` selected by
> `mask`, in order, writing the result to `out`: `out = mask ? src[next] : in`,
> where `next` walks `src` linearly (row-major) advancing once per true mask
> element.
>
> Steps:
> 1. Let `in_type = in.scalar_type()`.
> 2. Require `in` to be a realhbbf16-type tensor (ET_KERNEL_CHECK: on failure sets
>    Error::InvalidArgument and returns `out` unchanged). Accepted `in` dtypes:
>    REALHBBF16 {Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>    BFloat16}.
> 3. Require `mask.scalar_type() == Bool` (ET_KERNEL_CHECK: InvalidArgument).
> 4. Require `src.scalar_type() == in_type` (ET_KERNEL_CHECK: InvalidArgument).
> 5. Require `out.scalar_type() == in_type` (ET_KERNEL_CHECK: InvalidArgument).
> 6. Check `in`, `mask`, `out` share a dim order (ET_KERNEL_CHECK:
>    InvalidArgument).
> 7. Resize `out` to the broadcast of `in` and `mask` shapes per
>    `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
>    (ET_KERNEL_CHECK: InvalidArgument).
> 8. Dispatch on `in_type` over REALHBBF16 (CTYPE). Maintain a source cursor
>    `idx = 0` and `src_numel = src.numel()`, and a flag `src_numel_check =
>    true`. Apply the binary elementwise map over the broadcast of `in` (as
>    CTYPE) and `mask` (as bool), iterating output positions in the util's order:
>    for each position, if `mask_elem` is true and `idx >= src_numel`, set
>    `src_numel_check = false` and leave the output equal to the `in` element
>    (do not advance `idx`); otherwise if `mask_elem` is true, write
>    `src_data[idx]` and post-increment `idx`; if `mask_elem` is false, write the
>    `in` element unchanged.
> 9. After the map, ET_KERNEL_CHECK_MSG on `src_numel_check`: if false (src ran
>    out of elements), set Error::InvalidArgument and return `out` (message
>    "masked_scatter: src doesn't have enough elements").
> 10. Return `out`.
>
> Note: `src` is consumed in its own linear (row-major) order, independent of
> `in`/`mask` broadcasting.

