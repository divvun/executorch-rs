# kernels/portable/cpu/op_cdist_forward.cpp

> [spec:et:def:op-cdist-forward.torch.executor.native.cdist-fn]
> void cdist(const Tensor& x1, const Tensor& x2, Tensor& out, double p)

> [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-fn]
> Templated on `CTYPE` and a `Norm` policy, this computes the pairwise
> p-distance matrices for each batch. It writes into `out.mutable_data_ptr`.
> `x1` has shape (..., P, M), `x2` has shape (..., R, M), `out` has shape
> (..., P, R) where `...` is the broadcast of the two batch-dim sets.
>
> Steps:
> 1. If `out.numel() == 0`, return immediately (empty output).
> 2. If `x1.numel() == 0` (the shared last dimension M is 0), fill the entire
>    `out` buffer with 0 and return.
> 3. Compute batch views: `x1_batch_sizes`, `x2_batch_sizes`, `out_batch_sizes`
>    via `[spec:et:sem:op-cdist-forward.torch.executor.native.get-batch-sizes-fn]`.
>    A tensor "is broadcasted" if its batch sizes differ from `out`'s batch
>    sizes. `any_is_broadcasted = x1_is_broadcasted || x2_is_broadcasted`.
> 4. `out_batch_numel` = product of `out_batch_sizes` (1 if empty). Let
>    P = x1.size(-2), R = x2.size(-2), M = x1.size(-1). Inner block sizes:
>    `x1_inner_size = P*M`, `x2_inner_size = R*M`, `out_inner_size = P*R`.
> 5. For each batch index b in [0, out_batch_numel):
>    - Base offsets `x1_base_ix = b*x1_inner_size`, `x2_base_ix = b*x2_inner_size`,
>      `out_base_ix = b*out_inner_size`.
>    - If `any_is_broadcasted`: delinearize `out_base_ix` into per-dim
>      coordinates over `out` (`delinearize_index`), then for each broadcasted
>      input recompute its base offset by mapping those output coordinates into
>      the input's layout with `linearize_access_indexes(out_base_coord,
>      out.dim(), x1_or_x2)` (broadcasting a size-1 batch dim to index 0).
>    - Then, with `out_ix` starting at 0, for each i in [0, P) let
>      `row_i = x1_data + x1_base_ix + i*M`; for each j in [0, R) let
>      `row_j = x2_data + x2_base_ix + j*M`; accumulate `agg` (initialized 0)
>      over k in [0, M): `diff = std::abs(row_i[k] - row_j[k])`, then
>      `agg = Norm::reduce(agg, Norm::map(diff, p))`. Store
>      `out_data[out_base_ix + out_ix++] = Norm::finish(agg, p)`.
> 6. The `Norm` policy is chosen by the caller from p: p==0 -> L0, p==1 -> L1,
>    p==2 -> L2, p==INFINITY -> Linf, otherwise -> Lp. Their map/reduce/finish
>    semantics are defined in the distance_util rules (e.g.
>    `[spec:et:sem:distance-util.torch.executor.lp.map-fn]`,
>    `[spec:et:sem:distance-util.torch.executor.lp.finish-fn]`): L0 counts
>    nonzero diffs; L1 sums diffs; L2 sums squares then sqrt; Linf takes the max
>    diff; Lp sums diff^p then takes the p-th root.

> [spec:et:def:op-cdist-forward.torch.executor.native.get-batch-sizes-fn]
> inline ArrayRef<Tensor::SizesType> get_batch_sizes(const Tensor& tensor)

> [spec:et:sem:op-cdist-forward.torch.executor.native.get-batch-sizes-fn]
> Returns an ArrayRef view over the tensor's "batch" dimensions, i.e. all sizes
> except the trailing two (which are the matrix rows and columns for cdist). The
> view points at `tensor.sizes().data()` with length `tensor.sizes().size() - 2`.
> No allocation, no copy; the returned ArrayRef borrows the tensor's sizes
> storage. Precondition: the tensor has at least 2 dimensions (callers guarantee
> this).

> [spec:et:def:op-cdist-forward.torch.executor.native.cdist-forward-out-fn]
> Tensor& _cdist_forward_out( KernelRuntimeContext& ctx, const Tensor& x1, const Tensor& x2, double p, optional<int64_t> compute_mode, Tensor& out)

> [spec:et:sem:op-cdist-forward.torch.executor.native.cdist-forward-out-fn]
> Implements `_cdist_forward.out`: batched pairwise p-norm distances between the
> rows of `x1` and `x2`. `compute_mode` is accepted but unused here (validated
> only). The context is unused for computation.
>
> Steps:
> 1. ET_KERNEL_CHECK: `tensors_have_same_dim_order(x1, x2, out)`; on failure set
>    Error::InvalidArgument on the context and return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensor_is_default_dim_order(x1)` (contiguous); on failure
>    set Error::InvalidArgument and return `out` unchanged.
> 3. ET_KERNEL_CHECK: `check_cdist_args(x1, x2, p, compute_mode, out)` (both
>    inputs at least 2D, equal trailing dim M, `p >= 0`, valid compute_mode,
>    floating output dtype, batch-dim broadcast compatibility). On failure set
>    Error::InvalidArgument and return `out` unchanged.
> 4. Compute the broadcast of the two batch-dim sets (all dims except the last
>    two) via `get_broadcast_target_size`; if it does not return Error::Ok set
>    Error::InvalidArgument and return `out` unchanged. Then append two more
>    dims: `target_sizes[-2] = x1.size(-2)` (P) and
>    `target_sizes[-1] = x2.size(-2)` (R), and `target_ndim += 2`.
> 5. Resize `out` to `{target_sizes, target_ndim}`; if resize fails set
>    Error::InvalidArgument and return `out` unchanged.
> 6. Dispatch on `out.scalar_type()` over FLOATHBF16 = {Half, Float, Double,
>    BFloat16}; other dtypes set Error::InvalidArgument and return `out`
>    unchanged. Call the CTYPE-templated `cdist<CTYPE>(x1, x2, out, p)`
>    dispatcher, which selects the Norm policy from `p` and runs the core loop
>    per `[spec:et:sem:op-cdist-forward.torch.executor.native.cdist-fn]`.
> 7. Return `out`.

