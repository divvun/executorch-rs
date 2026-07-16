# kernels/portable/cpu/op_mm.cpp

> [spec:et:def:op-mm.torch.executor.native.mm-out-fn]
> Tensor& mm_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-mm.torch.executor.native.mm-out-fn]
> Matrix multiply of 2-D tensors `in` (m×n) and `mat2` (n×p) into `out` (m×p); returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_mm_args(in, mat2, out)` per `[spec:et:sem:matmul-ops-util.torch.executor.check-mm-args-fn]` (requires `in.dim()==2`, `mat2.dim()==2`, inner dims match `in.size(1)==mat2.size(0)`, and all three tensors share the same dtype). On failure set `Error::InvalidArgument` on `ctx` and return `out` unchanged.
> 2. Compute the target output shape into `output_sizes`/`output_ndim` via `get_mm_out_target_size` per `[spec:et:sem:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]` (ndim 2, sizes `{in.size(0), mat2.size(1)}`).
> 3. ET_KERNEL_CHECK: `resize_tensor(out, {output_sizes, output_ndim})` must return `Error::Ok`; else InvalidArgument, return `out`.
> 4. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, mat2, out)`; else InvalidArgument, return `out`.
> 5. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)` (contiguous/default dim order required); else InvalidArgument, return `out`.
> 6. Dispatch on `in.scalar_type()` over the real dtypes plus Half and BFloat16 (ET_SWITCH_REAL_TYPES_AND2(Half, BFloat16) = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16 — no Bool); an unsupported dtype sets InvalidArgument and returns `out`.
> 7. Let `m = in.size(0)`, `n = in.size(1)`, `p = mat2.size(1)`. Call `vec_matmul<CTYPE>(out_data, in_data, mat2_data, m, n, p)`: standard row-major dense matmul, `out[i*p + k] = sum over j in [0,n) of in[i*n + j] * mat2[j*p + k]`, accumulating in `CTYPE` (no wider accumulator; for `m*n==0` or `p==0` the output is empty / zero-length so nothing is written; when `n==0` each output element is the empty-sum 0).
> 8. Return `out`.

