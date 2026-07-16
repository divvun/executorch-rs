# kernels/portable/cpu/op_replication_pad3d.cpp

> [spec:et:def:op-replication-pad3d.torch.executor.native.replication-pad3d-out-fn]
> Tensor& replication_pad3d_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out)

> [spec:et:sem:op-replication-pad3d.torch.executor.native.replication-pad3d-out-fn]
> `replication_pad3d.out`: edge-replication-pads the last three dimensions of
> `in` by `padding = [pad_left, pad_right, pad_top, pad_bottom, pad_front,
> pad_back]`. Same structure as
> `[spec:et:sem:op-replication-pad1d.torch.executor.native.replication-pad1d-out-fn]`
> with `n = 3`. Steps:
>
> - ET_KERNEL_CHECK: `check_padding_args(3, in, padding, out)` with
>   `reflection=false` (see
>   `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`):
>   `padding.size() == 6`, `in.dim() == 4 or 5`, `in`/`out` same dtype,
>   non-negative padded sizes; no `pad < in.size` constraint and no dim-order
>   checks; on failure `Error::InvalidArgument`, return `out`.
> - `get_padding_out_target_size(3, in, padding, ...)` (see
>   `[spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]`):
>   enlarges the last three dims by their front+back / top+bottom / left+right
>   pads.
> - Resize `out`; on failure `Error::InvalidArgument`, return `out`.
> - Dispatch on `in.scalar_type()` over ALL types as CTYPE and call
>   `pad3d<CTYPE>(replication_ix, in, out, padding)` (see
>   `[spec:et:sem:padding-util.torch.executor.pad3d-fn]` and
>   `[spec:et:sem:padding-util.torch.executor.replication-ix-fn]`): each output
>   `(d, h, w)` reads input `(replication_ix(d, in_depth, pad_front),
>   replication_ix(h, in_height, pad_top), replication_ix(w, in_width,
>   pad_left))` (edge clamp).
> - Returns `out`.

