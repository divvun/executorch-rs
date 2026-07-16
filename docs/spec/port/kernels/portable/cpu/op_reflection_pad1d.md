# kernels/portable/cpu/op_reflection_pad1d.cpp

> [spec:et:def:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn]
> Tensor& reflection_pad1d_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out)

> [spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn]
> `reflection_pad1d.out`: reflection-pads the last dimension of `in` by
> `padding = [pad_left, pad_right]`. Steps:
>
> - ET_KERNEL_CHECK: `check_padding_args(1, in, padding, out, reflection=true)`
>   (see `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`),
>   which requires `padding.size() == 2`, `in.dim() == 2 or 3` (n+1 or n+2),
>   `in`/`out` same dtype, output-length non-negative, and — because reflection —
>   each pad `< in.size` of the corresponding padded dim; on failure
>   `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - ET_KERNEL_CHECK: `in` is default dim order; else `Error::InvalidArgument`,
>   return `out`.
> - Compute `target_sizes`/`target_ndim` via `get_padding_out_target_size(1, in,
>   padding, ...)` (see
>   `[spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]`),
>   which copies `in.sizes()` and enlarges the last dim by `pad_left +
>   pad_right`.
> - Resize `out` to `{target_sizes, target_ndim}`; on failure
>   `Error::InvalidArgument`, return `out`.
> - Dispatch on `in.scalar_type()` over ALL types (every ScalarType) as CTYPE and
>   call `pad1d<CTYPE>(reflection_ix, in, out, padding)` (see
>   `[spec:et:sem:padding-util.torch.executor.pad1d-fn]` for the copy loop and
>   `[spec:et:sem:padding-util.torch.executor.reflection-ix-fn]` for the source
>   index mapping): each output column `w` reads input column
>   `reflection_ix(w, in_width, pad_left)`, i.e. positions inside the padded
>   region mirror across the border without repeating the edge element.
> - Returns `out`.

