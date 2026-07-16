# kernels/portable/cpu/op_pdist_forward.cpp

> [spec:et:def:op-pdist-forward.torch.executor.native.pdist-forward-out-fn]
> Tensor& _pdist_forward_out( KernelRuntimeContext& ctx, const Tensor& in, double p, Tensor& out)

> [spec:et:sem:op-pdist-forward.torch.executor.native.pdist-forward-out-fn]
> Computes the pairwise p-norm distances between the rows of a 2-D tensor `in` (shape [R, C]), writing the flattened upper-triangular (i<j) distance vector of length R*(R-1)/2 into `out`; returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_pdist_args(in, p, out)` per `[spec:et:sem:distance-util.torch.executor.check-pdist-args-fn]` (validates `in` is 2-D, dtype compatibility, `p >= 0`, etc.); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; else InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(in)`; else InvalidArgument, return `out`.
> 4. Compute `target_sizes`/`target_ndim` via `get_pdist_out_target_size` per `[spec:et:sem:distance-util.torch.executor.get-pdist-out-target-size-fn]` (1-D of length `R*(R-1)/2` where `R = in.size(0)`).
> 5. ET_KERNEL_CHECK: `resize_tensor(out, {target_sizes, target_ndim})` == Ok; else InvalidArgument, return `out`.
> 6. Dispatch on `in.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 7. Invoke `pdist<CTYPE>(in, out, p)` per `[spec:et:sem:distance-util.torch.executor.pdist-fn]`, which selects the norm by `p`: `p==0` → L0 (count of differing coordinates), `p==1` → L1 (sum of |diff|), `p==2` → L2 (Euclidean), `p==INFINITY` → Linf (max |diff|), otherwise → general Lp (`(sum |diff|^p)^(1/p)`). For each row pair (i<j) it maps/reduces over the `C` coordinates and writes the finished distance into `out` in row-major (i outer, j inner) order.
> 8. Return `out`.

