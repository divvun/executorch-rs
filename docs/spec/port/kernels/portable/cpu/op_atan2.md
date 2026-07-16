# kernels/portable/cpu/op_atan2.cpp

> [spec:et:def:op-atan2.torch.executor.native.get-common-type-fn]
> ScalarType get_common_type(ScalarType a_type, ScalarType b_type)

> [spec:et:sem:op-atan2.torch.executor.native.get-common-type-fn]
> Determines the common ScalarType used to promote the two atan2 inputs.
>
> Given `a_type` and `b_type`:
> - If both are floating-point types (Half, Float, Double, BFloat16), return the
>   PyTorch type-promotion result of the two (`promoteTypes(a_type, b_type)`).
> - Else if only `a_type` is floating-point, return `a_type`.
> - Else if only `b_type` is floating-point, return `b_type`.
> - Otherwise (neither is floating-point, i.e. both integral/bool), return
>   `ScalarType::Float`.
>
> This is a pure function with no side effects; it does not touch the context.

> [spec:et:def:op-atan2.torch.executor.native.atan2-out-fn]
> Tensor& atan2_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-atan2.torch.executor.native.atan2-out-fn]
> Computes elementwise `out = atan2(a, b)` with NumPy/PyTorch broadcasting.
>
> Steps:
> 1. Compute `common_type = get_common_type(a.scalar_type(), b.scalar_type())`
>    per `[spec:et:sem:op-atan2.torch.executor.native.get-common-type-fn]`.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)` must hold (all
>    three tensors share one dim order). On failure set Error::InvalidArgument on
>    the context and return `out` unchanged.
> 3. Resize `out` to the broadcast shape of `a` and `b` via
>    `resize_to_broadcast_target_size(a, b, out)`; if it does not return
>    Error::Ok, set Error::InvalidArgument and return `out` unchanged.
> 4. Compute `compute_type = utils::get_compute_type(common_type)` (the type in
>    which the math is performed; Half/BFloat16 are computed as Float).
> 5. Dispatch over `compute_type` restricted to the float set {Float, Double}
>    (ET_SWITCH_FLOAT_TYPES). If `compute_type` is not one of these, the switch
>    sets Error::InvalidArgument and returns `out` unchanged.
> 6. For each broadcasted output element apply the binary functor
>    `atan2(val_a, val_b)` using `executorch::math::atan2` in the compute type
>    (equivalent to std::atan2, i.e. the angle in radians of the point
>    (val_b, val_a) in range [-pi, pi], honoring IEEE signs of zero and the
>    documented atan2 behavior for inf/NaN inputs). This is performed by the
>    bitensor elementwise helper (see
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`): input `a`
>    is loaded from SupportedTensorDtypes::REALHBBF16, input `b` from
>    REALHBBF16, results are written to `out` as
>    SupportedTensorDtypes::FLOATHBF16 (Half, Float, Double, BFloat16), casting
>    the compute-type result to the output dtype. REALHBBF16 = {Byte, Char,
>    Short, Int, Long, Half, Float, Double, Bool, BFloat16}. Broadcasting,
>    iteration order, and index arithmetic follow the elementwise helper.
> 7. Return `out` (the same tensor reference passed in).

