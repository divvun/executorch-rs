# kernels/portable/cpu/op_copy.cpp

> [spec:et:def:op-copy.torch.executor.native.copy-fn]
> Tensor& copy_( KernelRuntimeContext& ctx, Tensor& in, const Tensor& src, bool non_blocking)

> [spec:et:sem:op-copy.torch.executor.native.copy-fn]
> In-place copy of `src` into the mutable tensor `in`, returning `in`. Every failure path
> sets the error on `ctx` and returns `in` unchanged.
> 1. ET_KERNEL_CHECK `non_blocking == false` (only blocking transfers supported), else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensor_is_broadcastable_to(src, in)` (src broadcasts to in's shape), else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, src)`, else InvalidArgument.
> 4. Fast path: if `sizes_match_ignoring_leading_1s(in.sizes(), src.sizes())` AND
>    `src.numel() > 0` AND `in.nbytes() >= src.nbytes()` AND `src` and `in` have the same dtype,
>    then `memcpy(in.data, src.data, src.nbytes())` (raw byte copy, no broadcast).
> 5. Otherwise dispatch over `in.scalar_type()` in the REALHBBF16 set
>    (`{Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}`) via
>    `ET_SWITCH_REALHBBF16_TYPES` and apply a two-input elementwise op that ignores the
>    first operand (the current `in` value) and writes `src`'s value, broadcasting `src`
>    against `in` and writing into `in`, per `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`.
>    Both `in` and `src` are read/written as REALHBBF16; the destination is `in`.
> 6. Return `in`.

> [spec:et:def:op-copy.torch.executor.native.copy-out-fn]
> Tensor& copy_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& src, bool non_blocking, Tensor& out)

> [spec:et:sem:op-copy.torch.executor.native.copy-out-fn]
> Copies `src` into `out`, using `in` for shape/dtype/dim-order reference, returning `out`.
> (This is the functional `copy.out` variant; note `in` provides the target shape and dtype,
> `src` provides the values.) Every failure path sets the error on `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `non_blocking == false`, else InvalidArgument.
> 2. ET_KERNEL_CHECK `tensors_have_same_dtype(in, out)`, else InvalidArgument.
> 3. ET_KERNEL_CHECK `tensor_is_broadcastable_to(src, in)`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `resize_tensor(out, in.sizes()) == Error::Ok`, else InvalidArgument.
> 5. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`, else InvalidArgument.
> 6. Fast path: if `sizes_match_ignoring_leading_1s(out.sizes(), src.sizes())` AND
>    `src.numel() > 0` AND `out.nbytes() >= src.nbytes()` AND `src` and `out` have the same dtype,
>    then `memcpy(out.data, src.data, src.nbytes())`.
> 7. Otherwise dispatch over `in.scalar_type()` in the REALHBBF16 set
>    (`{Byte, Char, Short, Int, Long, Half, Float, Double, Bool, BFloat16}`) via
>    `ET_SWITCH_REALHBBF16_TYPES` and apply a two-input elementwise op that ignores the first
>    operand (the `in` value) and returns `src`'s value, broadcasting `src` against `in` and
>    writing into `out`, per `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`.
>    Both `in` and `src` are treated as REALHBBF16; the destination is `out`.
> 8. Return `out`.

