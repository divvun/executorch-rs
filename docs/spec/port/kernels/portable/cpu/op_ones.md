# kernels/portable/cpu/op_ones.cpp

> [spec:et:def:op-ones.torch.executor.native.ones-out-fn]
> Tensor& ones_out(KernelRuntimeContext& ctx, IntArrayRef size, Tensor& out)

> [spec:et:sem:op-ones.torch.executor.native.ones-out-fn]
> Fills `out` (resized to `size`) with the value 1 in `out`'s dtype; returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `resize_tensor(out, size)` == Ok; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. Dispatch on `out.scalar_type()` over REALHBBF16 (ET_SWITCH_REALHBBF16_TYPES = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 3. For each flat index `i` in [0,out.numel()): `out_data[i] = static_cast<CTYPE>(1)`.
> 4. Return `out`.

