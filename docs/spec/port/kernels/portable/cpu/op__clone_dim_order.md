# kernels/portable/cpu/op__clone_dim_order.cpp

> [spec:et:def:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn]
> bool check_fast_path_conditions( const Tensor& in, OptionalArrayRef<int64_t> dim_order)

> [spec:et:sem:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn]
> Returns true when a plain `memcpy` clone is valid because the output dim order
> is unchanged from the input's.
>
> - If `dim_order` has no value (None), returns true immediately (a missing
>   dim_order means "preserve input dim order", so no reordering is needed).
> - Otherwise returns true iff the requested `dim_order` sequence equals
>   `in.dim_order()` element-for-element and has the same length (compared via
>   `std::equal` over both begin/end ranges, so length mismatch → false). Returns
>   false when they differ.

> [spec:et:def:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn]
> Tensor& _clone_dim_order_out( KernelRuntimeContext& ctx, const Tensor& self, bool non_blocking, OptionalArrayRef<int64_t> dim_order, Tensor& out)

> [spec:et:sem:op-clone-dim-order.torch.executor.native.clone-dim-order-out-fn]
> Implements `_clone_dim_order.out(self, non_blocking=False, dim_order=None,
> out)`: clones `self` into `out`, producing `out` in the layout specified by
> `dim_order` (or preserving `self`'s layout when `dim_order` is None). `ctx` and
> `non_blocking` are otherwise unused for control flow.
>
> Validation, in order (each `ET_KERNEL_CHECK`; on failure sets context Error to
> `InvalidArgument` and returns `out` unmodified):
> 1. `self.scalar_type() == out.scalar_type()` — dtype must match exactly (this
>    op does not convert dtype, unlike `[spec:et:sem:op-to-dim-order-copy.torch.executor.native.to-dim-order-copy-out-fn]`).
> 2. `check__to_dim_order_copy_args(self, non_blocking, dim_order, out)` — the
>    shared to_dim_order_copy argument validation (validates `dim_order` is a
>    valid channels-last or contiguous permutation and that `out`'s dim order
>    matches it / matches `self` when None).
> 3. `resize_tensor(out, self.sizes()) == Error::Ok` — resize `out` to `self`'s
>    shape.
>
> If `self.numel() == 0`, returns `out` immediately (no copy).
>
> Fast path: if `check_fast_path_conditions(self, dim_order)` per
> `[spec:et:sem:op-clone-dim-order.torch.executor.native.check-fast-path-conditions-fn]`
> is true (output layout unchanged), performs a raw `std::memcpy` of
> `self.nbytes()` bytes from `self.const_data_ptr()` to `out.mutable_data_ptr()`.
>
> Slow path (layout changes): dispatches on `self.scalar_type()` over
> `ET_SWITCH_REALHBBF16_TYPES` (accepted set {Byte, Char, Short, Int, Long, Half,
> Float, Double, Bool, BFloat16}) and calls `_to_dim_order_copy_impl<CTYPE,
> CTYPE>(self, out)`, which performs an element-wise copy that reads `self` in
> its layout and writes each element to its position in `out`'s dim order (same
> CTYPE in and out; no value conversion).
>
> Returns `out`. (A second overload without `ctx` constructs a default
> `KernelRuntimeContext` and delegates to this function.)

