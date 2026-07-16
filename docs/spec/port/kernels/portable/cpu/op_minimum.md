# kernels/portable/cpu/op_minimum.cpp

> [spec:et:def:op-minimum.torch.executor.native.minimum-out-fn]
> Tensor& minimum_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-minimum.torch.executor.native.minimum-out-fn]
> Elementwise binary minimum of tensors `a` and `b`, broadcasting, writing into `out`; returns `out`.
>
> Steps:
> 1. Compute `common_type = promoteTypes(a.scalar_type(), b.scalar_type())` (PyTorch type-promotion lattice).
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())` must hold; on failure set `Error::InvalidArgument` on `ctx` and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `a`, `b`, `out` must all have the same dim order (`tensors_have_same_dim_order`); on failure set InvalidArgument and return `out`.
> 4. ET_KERNEL_CHECK: resize `out` to the broadcast shape of `a` and `b` via `resize_to_broadcast_target_size` (see `[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`); if it returns anything other than `Error::Ok`, set InvalidArgument and return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)` — the type in which the elementwise op is evaluated (see `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`; floating-point compute types are kept, integer `common_type` promotes so half/bfloat16 compute in a wider float representation but for `minimum` the compute type equals the promoted real type).
> 6. Dispatch on `compute_type` over the REALB type set (ET_SWITCH_REALB_TYPES = all real dtypes Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16 plus Bool); if `compute_type` is outside this set, ET_SWITCH sets InvalidArgument and returns `out`.
> 7. Apply the binary elementwise functor over broadcasted `a` and `b` per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]`, loading both inputs from the REALHBBF16 dtype set (Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16 — no Bool), converting each to `CTYPE_COMPUTE`, computing `utils::min_override(val_a, val_b)` per `[spec:et:sem:math-util.torch.executor.native.utils.min-override-fn]` (NaN-propagating minimum: if either operand is NaN the result is NaN, otherwise the numerically smaller value), and storing to `out` in the REALHBBF16 dtype set with conversion from `CTYPE_COMPUTE`.
> 8. Return `out`.

