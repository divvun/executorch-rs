# kernels/portable/cpu/op_full_like.cpp

> [spec:et:def:op-full-like.torch.executor.native.full-like-out-fn]
> Tensor& full_like_out( KernelRuntimeContext& ctx, const Tensor& in, const Scalar& fill_value, optional<MemoryFormat> memory_format, Tensor& out)

> [spec:et:sem:op-full-like.torch.executor.native.full-like-out-fn]
> Produces an output tensor with the same shape as `in`, filled with the scalar
> `fill_value`. The contents of `in` are ignored; only its shape and dim order
> matter. Returns `out`.
>
> Steps:
> 1. If `memory_format` has a value, ET_KERNEL_CHECK_MSG that it is either
>    `MemoryFormat::Contiguous` or `MemoryFormat::Preserve`; otherwise fail with
>    Error::InvalidArgument and message "memory_format must be contiguous",
>    returning `out`. A null `memory_format` skips this check.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)` must hold; else
>    Error::InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)` must hold (input must be
>    in default/contiguous dim order); else Error::InvalidArgument, return `out`.
> 4. Resize `out` to `in.sizes()` via `resize_tensor`; on non-Ok fail with
>    ET_KERNEL_CHECK_MSG Error::InvalidArgument and message "Failed to resize
>    output tensor.", returning `out`.
> 5. Read `out_type = out.scalar_type()` and dispatch over it, which must be one
>    of the REALHBBF16 set: {Byte, Char, Short, Int, Long, Half, Float, Double,
>    Bool, BFloat16}; unsupported dtype triggers Error::InvalidArgument.
> 6. Cast `fill_value` to the output ctype with overflow checking via
>    `utils::internal::check_overflow_scalar_cast<CTYPE_OUT>`; ET_KERNEL_CHECK
>    that the resulting optional has a value (empty means the value is not
>    representable → Error::InvalidArgument, return `out`).
> 7. Write the casted value into every element of `out` (contiguous, indices
>    `0 .. out.numel()-1`).
> 8. Return `out`.

