# kernels/portable/cpu/op_mul.cpp

> [spec:et:def:op-mul.torch.executor.native.mul-out-fn]
> Tensor& mul_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-mul.torch.executor.native.mul-out-fn]
> Elementwise product `a * b` with broadcasting into `out`; returns `out`. Supports real, complex, and Bool compute types.
>
> Steps:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: resize `out` to broadcast shape of `a`,`b` via `resize_to_broadcast_target_size` (`[spec:et:sem:broadcast-util.torch.executor.resize-to-broadcast-target-size-fn]`) must return `Error::Ok`; else InvalidArgument, return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)` (`[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]`).
> 6. ET_KERNEL_CHECK: `compute_type` must be a real type (`isRealType`) OR a complex type (`isComplexType`) OR exactly `Bool`; else InvalidArgument, return `out`.
> 7. Complex branch — if `compute_type` is complex: ET_KERNEL_CHECK that `a`, `b`, and `out` all have the identical scalar type (no promotion across differing complex/real inputs allowed here); else InvalidArgument, return `out`. Then dispatch on `out.scalar_type()` over the complex-with-half type set (ET_SWITCH_COMPLEXH_TYPES = ComplexHalf, ComplexFloat, ComplexDouble) and apply `apply_binary_elementwise_fn<CTYPE,CTYPE,CTYPE>` (`[spec:et:sem:broadcast-util.torch.executor.apply-binary-elementwise-fn-fn]`) computing `val_a * val_b` in the native complex type, with broadcasting, writing to `out`.
> 8. Non-complex branch — dispatch on `compute_type` over REALB (ET_SWITCH_REALB_TYPES = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16, Bool). Apply the binary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]`, loading `a` and `b` from the REALHBBF16 dtype set (all reals + Half + BFloat16, no Bool as an input-load set here), converting each to `CTYPE_COMPUTE`, computing `val_a * val_b` in `CTYPE_COMPUTE` (integer multiply wraps per two's-complement; Bool compute multiplies as 0/1), and storing to `out` in the REALHBBF16 store set with conversion.
> 9. Return `out`.

> [spec:et:def:op-mul.torch.executor.native.mul-scalar-out-fn]
> Tensor& mul_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-mul.torch.executor.native.mul-scalar-out-fn]
> Elementwise product of tensor `a` with scalar `b` into `out` (same shape as `a`); returns `out`.
>
> Steps:
> 1. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` — promotes `a`'s dtype against the scalar's category (int/float/bool) without widening to a specific bit-width beyond the promotion rules.
> 2. ET_KERNEL_CHECK: `common_type == out.scalar_type()` exactly (no cast allowed, must match); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, out)`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: `resize_tensor(out, a.sizes())` must return `Error::Ok` (out gets `a`'s shape, no broadcasting); else InvalidArgument, return `out`.
> 5. `compute_type = utils::get_compute_type(common_type)`.
> 6. Dispatch on `compute_type` over REALB (ET_SWITCH_REALB_TYPES = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16, Bool); unsupported dtype sets InvalidArgument, returns `out`.
> 7. Convert the scalar to `CTYPE_COMPUTE` via `utils::scalar_to<CTYPE_COMPUTE>(b)` giving `val_b`. Apply the unary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]`: load each element of `a` from the REALHBBF16 input set, convert to `CTYPE_COMPUTE`, compute `val_a * val_b`, and store to `out` using the SAME_AS_COMMON store policy (out dtype equals `common_type`, so store converts `CTYPE_COMPUTE` to `out`'s dtype).
> 8. Return `out`.

