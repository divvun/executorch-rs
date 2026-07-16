# kernels/portable/cpu/op_bitwise_xor.cpp

> [spec:et:def:op-bitwise-xor.torch.executor.native.bitwise-xor-scalar-out-fn]
> Tensor& bitwise_xor_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-bitwise-xor.torch.executor.native.bitwise-xor-scalar-out-fn]
> Computes elementwise `out = a ^ b` where `b` is a Scalar. Delegates to
> `internal::bitwise_scalar_out<std::bit_xor, "bitwise_xor.Scalar_out">(ctx, a,
> b, out)`.
>
> Behavior:
> 1. `common_type = utils::promote_type_with_scalar(a.scalar_type(), b)`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Resize `out` to `a.sizes()`; on failure set Error::InvalidArgument and
>    return `out` unchanged.
> 5. `compute_type = utils::get_compute_type(common_type)`; dispatch over {Byte,
>    Char, Short, Int, Long, Bool}; other types set Error::InvalidArgument and
>    return `out` unchanged.
> 6. Convert `b` to the compute type; for each element of `a` compute
>    `val_a ^ val_b` (bitwise XOR; on Bool, logical XOR). Inputs loaded from
>    SupportedTensorDtypes::INTB = {Byte, Char, Short, Int, Long, Bool}, results
>    written to `out` as REALHBBF16 with a cast.
> 7. Return `out`.

> [spec:et:def:op-bitwise-xor.torch.executor.native.bitwise-xor-tensor-out-fn]
> Tensor& bitwise_xor_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-bitwise-xor.torch.executor.native.bitwise-xor-tensor-out-fn]
> Computes elementwise `out = a ^ b` for two tensors with broadcasting.
> Delegates to `internal::bitwise_tensor_out<std::bit_xor,
> "bitwise_xor.Tensor_out">(ctx, a, b, out)`.
>
> Behavior:
> 1. `common_type = promoteTypes(a.scalar_type(), b.scalar_type())`.
> 2. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(a, b, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Resize `out` to the broadcast shape of `a` and `b`; on non-Ok set
>    Error::InvalidArgument and return `out` unchanged.
> 5. `compute_type = utils::get_compute_type(common_type)`; dispatch over {Byte,
>    Char, Short, Int, Long, Bool}; other types set Error::InvalidArgument and
>    return `out` unchanged.
> 6. For each broadcasted output element compute `val_a ^ val_b` (bitwise XOR; on
>    Bool, logical XOR) via `std::bit_xor<CTYPE_COMPUTE>`. Inputs loaded from
>    SupportedTensorDtypes::INTB = {Byte, Char, Short, Int, Long, Bool}, results
>    written to `out` as REALHBBF16 with a cast.
> 7. Return `out`.

