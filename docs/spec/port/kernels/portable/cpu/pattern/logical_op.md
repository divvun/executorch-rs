# kernels/portable/cpu/pattern/logical_op.h

> [spec:et:def:logical-op.torch.executor.native.internal.logical-tensor-out-fn]
> Tensor& logical_tensor_out( bool (*fn)(bool, bool), KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:logical-op.torch.executor.native.internal.logical-tensor-out-fn]
> Generic tensor-vs-tensor binary logical pattern with NumPy-style broadcasting,
> templated on a compile-time `op_name` string and taking a runtime function
> pointer `fn(bool, bool) -> bool` (e.g. logical_and/or/xor). Every element is
> first coerced to `bool` (nonzero → true), `fn` is applied, and the boolean
> result is written to `out`. Steps:
>
> 1. ET_KERNEL_CHECK `tensors_have_same_dim_order(a, b, out)`: on failure set
>    Error::InvalidArgument on `ctx` and return `out` unchanged.
> 2. ET_KERNEL_CHECK `resize_to_broadcast_target_size(a, b, out) == Error::Ok`
>    per `[spec:et:sem:broadcast-util.torch.executor.native.resize-to-broadcast-target-size-fn]`
>    (resizes `out` to the broadcast shape of `a` and `b`); on failure set
>    Error::InvalidArgument and return `out`.
> 3. Call `utils::apply_bitensor_elementwise_fn<bool, op_name,
>    SupportedTensorDtypes::REALHBBF16>` per
>    `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]`
>    with compute type fixed to `bool`. For each output index it loads the
>    broadcast-corresponding elements of `a` and `b` (both accepted from the
>    REALHBBF16 = {Byte, Char, Short, Int, Long, Half, Float, Double, Bool,
>    BFloat16} set), converts each loaded value to `bool` (any nonzero value,
>    including NaN, becomes true; exact 0/0.0 becomes false), applies
>    `fn(bool_a, bool_b)`, and stores the boolean result into `out` (stored as
>    REALHBBF16; `out` is a Bool tensor).
>
> There is no dtype-set dispatch/switch here (compute type is unconditionally
> `bool`), so no ScalarType is rejected on dtype grounds. Returns `out`; a failed
> ET_KERNEL_CHECK returns `out` unchanged with Error::InvalidArgument on `ctx`.

