# kernels/portable/cpu/op__empty_dim_order.cpp

> [spec:et:def:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn]
> bool _check__empty_out_dim_order(OptionalIntArrayRef dim_order, Tensor& out)

> [spec:et:sem:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn]
> Validates that `out`'s dim order is consistent with the requested `dim_order`.
> Reads `out_dim_order = out.dim_order()`. Each check below is
> `ET_LOG_AND_RETURN_IF_FALSE` (logs and returns false on failure).
>
> - If `dim_order` has a value: let `dim_order_ref = dim_order.value()`.
>   1. The requested dim order must be either channels-last
>      (`is_channels_last_dim_order(data, size)`) or contiguous
>      (`is_contiguous_dim_order(data, size)`); otherwise false.
>   2. `out_dim_order.size() == dim_order_ref.size()`; otherwise false.
>   3. For each `i` in `[0, dim_order_ref.size())`, `out_dim_order[i] ==
>      dim_order_ref[i]`; any mismatch → false.
> - If `dim_order` has no value (None): `out` must be contiguous —
>   `is_contiguous_dim_order(out_dim_order.data(), out_dim_order.size())` must
>   hold; otherwise false.
>
> Returns true if all applicable checks pass.

> [spec:et:def:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn]
> Tensor& _empty_dim_order_out( KernelRuntimeContext& context, IntArrayRef size, OptionalIntArrayRef dim_order, Tensor& out)

> [spec:et:sem:op-empty-dim-order.torch.executor.native.empty-dim-order-out-fn]
> Implements `_empty_dim_order.out(size, dim_order=None, out)`: produces an
> uninitialized tensor of shape `size` with the requested `dim_order`, in `out`.
> `context` is otherwise unused for control flow.
>
> Steps:
> 1. Calls `_check__empty_out_dim_order(dim_order, out)` per
>    `[spec:et:sem:op-empty-dim-order.torch.executor.native.check-empty-out-dim-order-fn]`.
>    NOTE: the boolean result is NOT wrapped in an `ET_KERNEL_CHECK` and is
>    discarded — a failing dim-order check only logs; it does not abort or set an
>    error. (Behavioral quirk to preserve.)
> 2. `resize_tensor(out, size) == Error::Ok` (`ET_KERNEL_CHECK_MSG`; on failure
>    sets context Error to `InvalidArgument`, message "Failed to resize output
>    tensor.", returns `out`).
>
> Does not write any element values — `out`'s data is left uninitialized (this
> is the "empty" op). Returns `out`.

