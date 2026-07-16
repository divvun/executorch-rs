# kernels/portable/cpu/op_bmm.cpp

> [spec:et:def:op-bmm.torch.executor.native.bmm-out-fn]
> Tensor& bmm_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-bmm.torch.executor.native.bmm-out-fn]
> Computes batched matrix multiply `out = in @ mat2`, where `in` has shape
> (B, P, M), `mat2` has shape (B, M, R), and `out` has shape (B, P, R).
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_bmm_args(in, mat2, out)` (both inputs are 3D, share
>    compatible dtype with `out`, batch sizes match, and inner dims are
>    contractible: `in.size(2) == mat2.size(1)`). On failure set
>    Error::InvalidArgument on the context and return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, mat2, out)`; on failure
>    set Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)` (contiguous); on failure
>    set Error::InvalidArgument and return `out` unchanged.
> 4. Compute `output_sizes` via `get_bmm_out_target_size(in, mat2, ...)` =
>    {in.size(0), in.size(1), mat2.size(2)}, then resize `out` to it; if resize
>    fails set Error::InvalidArgument and return `out` unchanged.
> 5. Dispatch on `in.scalar_type()`:
>    - If it is a complex type, dispatch over COMPLEXH (Half/Float/Double complex
>      variants) and call `internal::bmm_out_impl<CTYPE>(in, mat2, out)`.
>    - Otherwise dispatch over REALHBF16 = {Byte, Char, Short, Int, Long, Half,
>      Float, Double, BFloat16} and call `internal::bmm_out_impl<CTYPE>`.
>    A dtype outside the selected set sets Error::InvalidArgument and returns
>    `out` unchanged.
> 6. `bmm_out_impl` performs, for each batch b in [0, B) and each (p, r), the dot
>    product `sum over m of in[b,p,m] * mat2[b,m,r]` accumulated in CTYPE, writing
>    to `out[b,p,r]`.
> 7. Return `out`.

