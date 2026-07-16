# kernels/portable/cpu/op_narrow_copy.cpp

> [spec:et:def:op-narrow-copy.torch.executor.native.narrow-copy-out-fn]
> Tensor& narrow_copy_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, int64_t start, int64_t length, Tensor& out)

> [spec:et:sem:op-narrow-copy.torch.executor.native.narrow-copy-out-fn]
> Copies a contiguous slice of `length` elements starting at `start` along dimension `dim` of `in` into `out`; returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_narrow_copy_args(in, dim, start, length, out)` per the slice-util argument check (validates `in`/`out` dtype match, `dim` within `[-in.dim(), in.dim())`, `length >= 0`, and that `start` plus `length` fits within `in.size(dim)` after dim normalization). On failure set `Error::InvalidArgument` on `ctx` and return `out` unchanged.
> 2. Normalize `dim`: if `dim < 0`, set `dim += in.dim()`.
> 3. Compute `target_sizes`/`target_ndim` via `get_narrow_copy_out_target_size` (= `in`'s sizes with the `dim`-th size replaced by `length`).
> 4. ET_KERNEL_CHECK: `resize_tensor(out, {target_sizes, target_ndim})` must return `Error::Ok`; else InvalidArgument, return `out`.
> 5. If `length != 0`, perform the copy via `compute_slice(ctx, in, dim, start, length, /*step=*/1, out)` — copies `length` consecutive index positions along `dim` (step 1) from `in` to `out`, preserving all other dims. If `length == 0`, `out` is left as the zero-length resized tensor (nothing copied).
> 6. Return `out`.

