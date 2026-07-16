# kernels/portable/cpu/op_reflection_pad3d.cpp

> [spec:et:def:op-reflection-pad3d.torch.executor.native.reflection-pad3d-out-fn]
> Tensor& reflection_pad3d_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out)

> [spec:et:sem:op-reflection-pad3d.torch.executor.native.reflection-pad3d-out-fn]
> `reflection_pad3d.out`: reflection-pads the last three dimensions of `in` by
> `padding = [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]`.
> Same structure as
> `[spec:et:sem:op-reflection-pad1d.torch.executor.native.reflection-pad1d-out-fn]`
> with `n = 3`. Steps:
>
> - ET_KERNEL_CHECK: `check_padding_args(3, in, padding, out, reflection=true)`
>   (see `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`):
>   `padding.size() == 6`, `in.dim() == 4 or 5`, `in`/`out` same dtype,
>   non-negative padded sizes, each pad `<` the corresponding input dim; on
>   failure `Error::InvalidArgument`, return `out`.
> - ET_KERNEL_CHECK: `in`/`out` same dim order; else `Error::InvalidArgument`.
> - ET_KERNEL_CHECK: `in` default dim order; else `Error::InvalidArgument`.
> - `get_padding_out_target_size(3, in, padding, ...)` (see
>   `[spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]`):
>   enlarges the last three dims by their respective front+back / top+bottom /
>   left+right pads.
> - Resize `out`; on failure `Error::InvalidArgument`, return `out`.
> - Dispatch on `in.scalar_type()` over ALL types as CTYPE and call
>   `pad3d<CTYPE>(reflection_ix, in, out, padding)` (see
>   `[spec:et:sem:padding-util.torch.executor.pad3d-fn]` and
>   `[spec:et:sem:padding-util.torch.executor.reflection-ix-fn]`): each output
>   `(d, h, w)` reads input `(reflection_ix(d, in_depth, pad_front),
>   reflection_ix(h, in_height, pad_top), reflection_ix(w, in_width,
>   pad_left))`.
> - Returns `out`.

