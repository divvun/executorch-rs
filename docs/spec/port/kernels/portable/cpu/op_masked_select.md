# kernels/portable/cpu/op_masked_select.cpp

> [spec:et:def:op-masked-select.torch.executor.native.masked-select-out-fn]
> Tensor& masked_select_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mask, Tensor& out)

> [spec:et:sem:op-masked-select.torch.executor.native.masked-select-out-fn]
> Selects the elements of (broadcast) `in` where the (broadcast) `mask` is true,
> producing a 1-D `out` tensor containing those elements in row-major order over
> the broadcast shape.
>
> Steps:
> 1. Let `in_type = in.scalar_type()`.
> 2. Require `in` to be a realhbbf16-type tensor (ET_KERNEL_CHECK: on failure sets
>    Error::InvalidArgument and returns `out` unchanged). Accepted `in` dtypes:
>    REALHBBF16 {Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>    BFloat16}.
> 3. Require `mask.scalar_type() == Bool` (ET_KERNEL_CHECK: InvalidArgument).
> 4. Require `out.scalar_type() == in_type` (ET_KERNEL_CHECK: InvalidArgument).
> 5. Check `in`, `mask`, `out` share a dim order (ET_KERNEL_CHECK:
>    InvalidArgument).
> 6. Require `in` and `mask` are broadcastable together (ET_KERNEL_CHECK:
>    InvalidArgument).
> 7. Empty shortcut: if `in.numel() == 0` or `mask.numel() == 0`, resize `out` to
>    shape `{0}` (ET_KERNEL_CHECK: InvalidArgument) and return `out`.
> 8. Compute the broadcast target shape of `in` and `mask` into
>    `broadcast_sizes`/`broadcast_ndim` per
>    `[spec:et:sem:broadcast-util.torch.executor.native.get-broadcast-target-size-fn]`
>    (on error, ET_KERNEL_CHECK_MSG fails with InvalidArgument, message "Failed to
>    broadcast input and mask"). Let `broadcast_numel` be the product of the
>    broadcast sizes.
> 9. Count `mask_true_count` = number of true entries across `mask.numel()`
>    elements (row-major). The output element count is
>    `out_numel = mask_true_count * (broadcast_numel / mask.numel())` (accounts
>    for `mask` being broadcast up to the target shape).
> 10. Resize `out` to shape `{out_numel}` (ET_KERNEL_CHECK: InvalidArgument).
> 11. Determine whether `in` is broadcasted (its dim/sizes differ from the
>     broadcast shape) and whether `mask` is broadcasted, similarly.
> 12. Walk `i` from 0 to `broadcast_numel - 1` (row-major over the broadcast
>     shape). If neither is broadcasted, `in_linear_index = mask_linear_index =
>     i`; otherwise delinearize `i` into per-dim indices in the broadcast space,
>     then map to `in`'s linear index (if `in` broadcasted) and `mask`'s linear
>     index (if `mask` broadcasted) via the broadcast access-index linearization.
>     If `mask[mask_linear_index]` is true, `memcpy` `element_size` bytes of
>     `in[in_linear_index]` into the next `out` slot and advance the output
>     cursor.
> 13. Return `out`.

