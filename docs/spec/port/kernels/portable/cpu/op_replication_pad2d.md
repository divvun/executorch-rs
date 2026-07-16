# kernels/portable/cpu/op_replication_pad2d.cpp

> [spec:et:def:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn]
> Tensor& replication_pad2d_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out)

> [spec:et:sem:op-replication-pad2d.torch.executor.native.replication-pad2d-out-fn]
> `replication_pad2d.out`: edge-replication-pads the last two dimensions of `in`
> by `padding = [pad_left, pad_right, pad_top, pad_bottom]`. Same structure as
> `[spec:et:sem:op-replication-pad1d.torch.executor.native.replication-pad1d-out-fn]`
> with `n = 2`. Steps:
>
> - ET_KERNEL_CHECK: `check_padding_args(2, in, padding, out)` with
>   `reflection=false` (see
>   `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`):
>   `padding.size() == 4`, `in.dim() == 3 or 4`, `in`/`out` same dtype,
>   non-negative padded sizes; no `pad < in.size` constraint and no dim-order
>   checks; on failure `Error::InvalidArgument`, return `out`.
> - `get_padding_out_target_size(2, in, padding, ...)` (see
>   `[spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]`):
>   enlarges the last two dims by their left+right / top+bottom pads.
> - Resize `out`; on failure `Error::InvalidArgument`, return `out`.
> - Dispatch on `in.scalar_type()` over ALL types as CTYPE and call
>   `pad2d<CTYPE>(replication_ix, in, out, padding)` (see
>   `[spec:et:sem:padding-util.torch.executor.pad2d-fn]` and
>   `[spec:et:sem:padding-util.torch.executor.replication-ix-fn]`): each output
>   `(h, w)` reads input `(replication_ix(h, in_height, pad_top),
>   replication_ix(w, in_width, pad_left))` (edge clamp).
> - Returns `out`.

