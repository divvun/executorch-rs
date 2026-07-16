# kernels/optimized/cpu/op_mm.cpp

> [spec:et:def:op-mm.torch.executor.native.opt-mm-out-fn]
> Tensor& opt_mm_out( RuntimeContext& ctx, const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-mm.torch.executor.native.opt-mm-out-fn]
> Optimized matrix-multiply out-variant. Validate `in`/`mat2`/`out` with
> `check_mm_args` (returning `out` with InvalidArgument on failure). Compute the
> target output shape via `get_mm_out_target_size` (= `[in.size(0),
> mat2.size(1)]`) and `resize_tensor(out, ...)`; InvalidArgument on failure. If
> `out.numel() == 0`, return `out` immediately (GEMM doesn't tolerate empty
> input). Otherwise, switch over `in.scalar_type()` across the real + Half +
> BFloat16 dtypes and compute `out = in @ mat2` with a single column-major GEMM:
> set `n = in.size(0)`, `k = in.size(1)`, `m = mat2.size(1)`, then call
> `cpublas::gemm(NoTranspose, NoTranspose, m, n, k, 1, mat2_data, m, in_data, k,
> 0, out_data, m)`. GEMM is column-major, so the row-major product is produced
> via the identity `(A @ B).t() = B.t() @ A.t()`: row-major `mat2`/`in` are their
> transposes in GEMM's column-major view, so the untransposed `(mat2, in)` call
> writes the row-major result directly. Return `out`.

