# kernels/portable/cpu/op_neg.cpp

> [spec:et:def:op-neg.torch.executor.native.neg-out-fn]
> Tensor& neg_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-neg.torch.executor.native.neg-out-fn]
> Elementwise negation `-in` into `out` (same shape and dtype as `in`); returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK_MSG: `resize_tensor(out, in.sizes())` == Ok (message "Failed to resize output tensor."); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_shape_and_dtype(in, out)` (out must exactly match `in`'s shape and dtype — no promotion); else InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else InvalidArgument, return `out`.
> 4. Dispatch on `in.scalar_type()` over REALHBF16 (ET_SWITCH_REALHBF16_TYPES = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16 — real types plus Half/BFloat16, no Bool); unsupported dtype → InvalidArgument, return `out`.
> 5. Apply the unary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`: load each `in` element from the REALHBF16 set, compute `-val_in` in `CTYPE`, and store to `out` with SAME_AS_COMMON policy (out dtype equals `in` dtype). For unsigned Byte, negation wraps modulo 256 (two's-complement wraparound); floating negation flips the sign bit (negating NaN yields NaN with flipped sign).
> 6. Return `out`.

