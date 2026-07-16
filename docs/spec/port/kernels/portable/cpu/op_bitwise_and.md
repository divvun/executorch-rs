# kernels/portable/cpu/op_bitwise_and.cpp

> [spec:et:def:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn]
> Tensor& bitwise_and_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-scalar-out-fn]
> Computes elementwise `out = a & b` where `b` is a Scalar. Delegates to
> `internal::bitwise_scalar_out<std::bit_and, "bitwise_and.Scalar_out">(ctx, a,
> b, out)`; see `[spec:et:sem:bitwise-op.operator-fn]` for the operator wiring.
>
> Behavior:
> 1. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)` — the
>    PyTorch promotion of the tensor dtype against the scalar's category.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Resize `out` to `a.sizes()`; if resize fails set Error::InvalidArgument and
>    return `out` unchanged.
> 5. `compute_type = utils::get_compute_type(common_type)`, then dispatch over
>    the integer-plus-Bool set {Byte, Char, Short, Int, Long, Bool}
>    (ET_SWITCH_INT_TYPES_AND(Bool, ...)). Other compute types set
>    Error::InvalidArgument and return `out` unchanged.
> 6. Convert `b` to the compute type (`utils::scalar_to<CTYPE_COMPUTE>(b)`); for
>    each element of `a` compute `val_a & val_b` (bitwise AND; for Bool this is
>    logical AND). Inputs are loaded from SupportedTensorDtypes::INTB = {Byte,
>    Char, Short, Int, Long, Bool}, results written to `out` as REALHBBF16 with a
>    cast to the output dtype (unitensor elementwise helper).
> 7. Return `out`.

> [spec:et:def:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn]
> Tensor& bitwise_and_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-bitwise-and.torch.executor.native.bitwise-and-tensor-out-fn]
> Computes elementwise `out = a & b` for two tensors with broadcasting.
> Delegates to `internal::bitwise_tensor_out<std::bit_and,
> "bitwise_and.Tensor_out">(ctx, a, b, out)`.
>
> Behavior:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Resize `out` to the broadcast shape of `a` and `b`
>    (`resize_to_broadcast_target_size`); if it does not return Error::Ok set
>    Error::InvalidArgument and return `out` unchanged.
> 5. `compute_type = utils::get_compute_type(common_type)`, then dispatch over
>    {Byte, Char, Short, Int, Long, Bool} (ET_SWITCH_INT_TYPES_AND(Bool, ...));
>    other compute types set Error::InvalidArgument and return `out` unchanged.
> 6. For each broadcasted output element compute `val_a & val_b` (bitwise AND; on
>    Bool, logical AND) via the `std::bit_and<CTYPE_COMPUTE>` functor. Inputs
>    loaded from SupportedTensorDtypes::INTB = {Byte, Char, Short, Int, Long,
>    Bool}, results written to `out` as REALHBBF16 with a cast (bitensor
>    elementwise helper handles broadcasting and iteration).
> 7. Return `out`.

