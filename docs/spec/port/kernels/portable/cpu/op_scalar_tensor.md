# kernels/portable/cpu/op_scalar_tensor.cpp

> [spec:et:def:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn]
> Tensor&

> [spec:et:sem:op-scalar-tensor.torch.executor.native.scalar-tensor-out-fn]
> Produces a 0-dimensional (scalar) tensor `out` holding the scalar `s`. Steps:
>
> - Resize `out` to shape `{}` (0-dim, 1 element) via `resize_tensor`; on failure
>   `Error::InvalidArgument`, return `out`.
> - `out_type = out.scalar_type()`. Dispatch on it over REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Half, Float, Double, Bool, BFloat16} as CTYPE.
> - Cast `s` to CTYPE with overflow checking via
>   `utils::internal::check_overflow_scalar_cast<CTYPE>(s)` (see
>   `[spec:et:sem:scalar-utils.torch.executor.native.utils.internal.check-overflow-scalar-cast-fn]`),
>   which returns an optional. ET_KERNEL_CHECK: the optional has a value (the
>   scalar fits in CTYPE without overflow); on failure set `Error::InvalidArgument`
>   and return `out` (the ET_KERNEL_CHECK with empty return-value argument returns
>   from the lambda; the function then returns `out`).
> - Write the casted value into `out.mutable_data_ptr<CTYPE>()[0]`.
> - Returns `out`.

