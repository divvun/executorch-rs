# kernels/portable/cpu/op_replication_pad1d.cpp

> [spec:et:def:op-replication-pad1d.torch.executor.native.replication-pad1d-out-fn]
> Tensor& replication_pad1d_out( KernelRuntimeContext& ctx, const Tensor& in, executorch::aten::ArrayRef<int64_t> padding, Tensor& out)

> [spec:et:sem:op-replication-pad1d.torch.executor.native.replication-pad1d-out-fn]
> `replication_pad1d.out`: edge-replication-pads the last dimension of `in` by
> `padding = [pad_left, pad_right]`. Steps:
>
> - ET_KERNEL_CHECK: `check_padding_args(1, in, padding, out)` with
>   `reflection=false` (see
>   `[spec:et:sem:padding-util.torch.executor.check-padding-args-fn]`):
>   `padding.size() == 2`, `in.dim() == 2 or 3`, `in`/`out` same dtype,
>   non-negative padded output size; the reflection-only `pad < in.size`
>   constraint is NOT applied (replication pads may exceed the input size); on
>   failure `Error::InvalidArgument`, return `out`. (Note: unlike the reflection
>   variant, no separate dim-order checks are performed here.)
> - Compute `target_sizes`/`target_ndim` via `get_padding_out_target_size(1, in,
>   padding, ...)` (see
>   `[spec:et:sem:padding-util.torch.executor.get-padding-out-target-size-fn]`),
>   enlarging the last dim by `pad_left + pad_right`.
> - Resize `out` to `{target_sizes, target_ndim}`; on failure
>   `Error::InvalidArgument`, return `out`.
> - Dispatch on `in.scalar_type()` over ALL types as CTYPE and call
>   `pad1d<CTYPE>(replication_ix, in, out, padding)` (see
>   `[spec:et:sem:padding-util.torch.executor.pad1d-fn]` and
>   `[spec:et:sem:padding-util.torch.executor.replication-ix-fn]`): each output
>   column `w` reads input column `replication_ix(w, in_width, pad_left)`, i.e.
>   the padded region repeats the nearest edge element (clamp).
> - Returns `out`.

