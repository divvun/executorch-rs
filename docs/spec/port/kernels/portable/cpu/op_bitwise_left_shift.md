# kernels/portable/cpu/op_bitwise_left_shift.cpp

> [spec:et:def:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn]
> Tensor& bitwise_left_shift_Tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-out-fn]
> Computes elementwise `out = a << b` for two tensors with broadcasting.
> Delegates to `internal::bitwise_tensor_out<internal::bit_lshift,
> "bitwise_left_shift.Tensor_out">(ctx, a, b, out)`, where the `bit_lshift<T>`
> functor returns `static_cast<T>(lhs << rhs)` (see
> `[spec:et:sem:bitwise-op.operator-fn]`).
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
>    Char, Short, Int, Long, Bool} (ET_SWITCH_INT_TYPES_AND(Bool, ...)); other
>    types set Error::InvalidArgument and return `out` unchanged.
> 6. For each broadcasted output element compute `static_cast<CTYPE_COMPUTE>(
>    val_a << val_b)` (C++ left-shift in the compute type; shift by a
>    negative/out-of-range count is C++ undefined behavior and not specially
>    guarded). Inputs loaded from SupportedTensorDtypes::INTB = {Byte, Char,
>    Short, Int, Long, Bool}, results written to `out` as REALHBBF16 with a cast.
> 7. Return `out`.

> [spec:et:def:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn]
> Tensor& bitwise_left_shift_Tensor_Scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-bitwise-left-shift.torch.executor.native.bitwise-left-shift-tensor-scalar-out-fn]
> Computes elementwise `out = a << b` where `b` is a Scalar. Delegates to
> `internal::bitwise_scalar_out<internal::bit_lshift,
> "bitwise_left_shift.Tensor_Scalar_out">(ctx, a, b, out)`.
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
>    `static_cast<CTYPE_COMPUTE>(val_a << val_b)` (C++ left-shift; negative/
>    over-wide shift counts are C++ undefined behavior and not guarded). Inputs
>    loaded from SupportedTensorDtypes::INTB = {Byte, Char, Short, Int, Long,
>    Bool}, results written to `out` as REALHBBF16 with a cast.
> 7. Return `out`.

