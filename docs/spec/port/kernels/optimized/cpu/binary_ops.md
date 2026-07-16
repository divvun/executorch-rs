# kernels/optimized/cpu/binary_ops.cpp, kernels/optimized/cpu/binary_ops.h

> [spec:et:def:binary-ops.torch.executor.elementwise-optimized-path]
> enum class ElementwiseOptimizedPath {
>   kNone;
>   kTreatAs1d;
>   kBroadcast2dBy1d;
>   kBroadcast2dBy1dReverseArguments;
>   kBroadcastNdByNd;
>   kBroadcastNdByNdReverseArguments;
>   kBroadcastLastDim;
>   kBroadcastLastDimReverseArguments;
> }

> [spec:et:def:binary-ops.torch.executor.get-normalized-tensor-size-fn]
> std::array<int32_t, 3> inline get_normalized_tensor_size( const Tensor& a, const int32_t broadcast_dim)

> [spec:et:sem:binary-ops.torch.executor.get-normalized-tensor-size-fn]
> ET_CHECK_MSG that a.dim() > broadcast_dim ("Size of tensor ... must be larger
> than broadcast_dim"). Build a 3-element int32 array [outer, mid, inner]:
> initialize to [1, a.size(broadcast_dim), 1]. Multiply element[0] by a.size(i)
> for every axis i in [0, broadcast_dim). Multiply element[2] by a.size(i) for
> every axis i in (broadcast_dim, a.dim()). Return the array. It collapses `a`'s
> shape into (product-of-leading-dims, broadcast-dim-size, product-of-trailing-
> dims) about `broadcast_dim`.

> [spec:et:def:binary-ops.torch.executor.handle-broadcast-elementwise-fn]
> Tensor& handle_broadcast_elementwise( KernelRuntimeContext& ctx, const Op& vec_fun, const Tensor& a, const Tensor& b, Tensor& out, const ElementwiseOptimizedPath selected_optimized_path, const std::optional<Scalar>& alpha = {})

> [spec:et:sem:binary-ops.torch.executor.handle-broadcast-elementwise-fn]
> Dispatch for the broadcast optimized paths. If selected path is
> kBroadcastLastDim or kBroadcastLastDimReverseArguments, delegate to
> handle_last_dim_broadcast_elementwise<CTYPE> and return its result. Otherwise
> call internal::plan_broadcast_elementwise; if it returns nullopt (a resize
> failure already recorded on ctx) return `out` unchanged. Otherwise run the
> element-wise map over a 3D (outer_size x broadcast_size x inner_size) layout
> where `lhs` is treated as the full 3D block and `rhs` is broadcast along the
> broadcast axis (an unsqueezed 3D: shape [outer,1,inner] indexing), writing
> `vec_fun(lhs_elem, rhs_elem)` into out. Returns `out`. The `alpha` parameter is
> accepted but unused here (only threaded through by callers).

> [spec:et:def:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn]
> Tensor& handle_last_dim_broadcast_elementwise( KernelRuntimeContext& ctx, const Op& vec_fun, const Tensor& a, const Tensor& b, Tensor& out, const ElementwiseOptimizedPath selected_optimized_path)

> [spec:et:sem:binary-ops.torch.executor.handle-last-dim-broadcast-elementwise-fn]
> Handle the last-dim broadcast. Pick lhs/rhs: if the path is
> kBroadcastLastDimReverseArguments then lhs=&b, rhs=&a, else lhs=&a, rhs=&b.
> Resize `out` to lhs->sizes(); ET_KERNEL_CHECK_MSG that the resize succeeded
> (else record InvalidArgument and return out). Compute outer_size =
> getLeadingDims(out, out.dim()-1) and broadcast_size = out.size(out.dim()-1).
> Run broadcasting_map_broadcast_last_dim: for each of outer_size rows, rhs
> contributes one scalar per row (rhs has one element per outer row), lhs
> contributes broadcast_size contiguous elements; out[row*broadcast_size + j] =
> vec_fun(lhs[row*broadcast_size + j], rhs[row]). Returns out.

> [spec:et:def:binary-ops.torch.executor.internal.broadcast-elementwise-plan]
> struct BroadcastElementwisePlan {
>   const Tensor* lhs;
>   const Tensor* rhs;
>   int64_t outer_size;
>   int64_t broadcast_size;
>   int64_t inner_size;
> }

> [spec:et:def:binary-ops.torch.executor.internal.get-broadcast-dim-fn]
> int32_t inline get_broadcast_dim(const Tensor& lhs, const Tensor& rhs)

> [spec:et:sem:binary-ops.torch.executor.internal.get-broadcast-dim-fn]
> Returns the (negative) axis index at which exactly one broadcast dim exists,
> else 0. Compute lhs_begin = first non-1 leading element of lhs.sizes() (via
> arrayref_begin_ignoring_leading_1s), lhs_end = lhs.sizes().end(); same for rhs.
> lhs_size = lhs_end-lhs_begin, rhs_size = rhs_end-rhs_begin. If lhs_size !=
> rhs_size return 0 (mismatched effective rank not handled). Set broadcast_dim=0.
> Decrement lhs_end and rhs_end once (to point at the last element). Walk backward
> while lhs_end != lhs_begin: at each position, if *lhs_end==1 OR *rhs_end==1 then
> this is a broadcast dim — if broadcast_dim was already set (nonzero) return 0
> (more than one broadcast dim unsupported), else set broadcast_dim = lhs_end -
> lhs.sizes().end() (a negative offset from the end). Else if *lhs_end != *rhs_end
> return 0 (unequal non-1 dims). Decrement both pointers. Return broadcast_dim.
> Note the loop stops when lhs_end reaches lhs_begin, so the leading (post-
> ignored-1s) dim is not itself examined for being 1.

> [spec:et:def:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn]
> std::optional<BroadcastElementwisePlan> plan_broadcast_elementwise( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out, const ElementwiseOptimizedPath selected_optimized_path)

> [spec:et:sem:binary-ops.torch.executor.internal.plan-broadcast-elementwise-fn]
> Build a BroadcastElementwisePlan. Choose lhs/rhs: if the path is one of the
> ReverseArguments variants (kBroadcast2dBy1dReverseArguments or
> kBroadcastNdByNdReverseArguments) then lhs=&b, rhs=&a; otherwise ET_DCHECK the
> path is kBroadcast2dBy1d or kBroadcastNdByNd and set lhs=&a, rhs=&b. Resize out
> to lhs->sizes(); ET_KERNEL_CHECK_MSG the resize succeeded, else record
> InvalidArgument and return nullopt. Set outer_size=1. If the path is
> kBroadcastNdByNd or kBroadcastNdByNdReverseArguments: broadcast_dim =
> get_broadcast_dim(*lhs,*rhs); broadcast_dim_lhs = lhs->dim()+broadcast_dim
> (convert negative offset to absolute axis); normalized =
> get_normalized_tensor_size(*lhs, broadcast_dim_lhs); set outer=normalized[0],
> broadcast_size=normalized[1], inner=normalized[2]. Else (2dBy1d):
> broadcast_size = lhs->sizes()[lhs->dim()-2], inner_size =
> lhs->sizes()[lhs->dim()-1] (outer stays 1). Return the plan.

> [spec:et:def:binary-ops.torch.executor.internal.select-broadcast-optimized-path-fn]
> inline ElementwiseOptimizedPath select_broadcast_optimized_path( const Tensor& lhs, const Tensor& rhs)

> [spec:et:sem:binary-ops.torch.executor.internal.select-broadcast-optimized-path-fn]
> Classify a broadcast into a specific optimized path. Compute effective
> (ignoring leading 1s) begin/end of lhs and rhs; lhs_size, rhs_size their
> lengths. If lhs_size==2 && rhs_size==1 && lhs_begin[1]==rhs_begin[0] return
> kBroadcast2dBy1d. If lhs_size==1 && rhs_size==2 && rhs_begin[1]==lhs_begin[0]
> return kBroadcast2dBy1dReverseArguments. Else broadcast_dim =
> get_broadcast_dim(lhs, rhs). If broadcast_dim < -1: if exactly one element of
> [rhs_begin, rhs_end) equals 1 return kBroadcastNdByNd else
> kBroadcastNdByNdReverseArguments. Else if broadcast_dim == -1 (last-dim
> broadcast): if exactly one element of [lhs_begin, lhs_end) equals 1 return
> kBroadcastLastDimReverseArguments else kBroadcastLastDim. Otherwise return
> kNone.

> [spec:et:def:binary-ops.torch.executor.select-optimized-path-fn]
> ElementwiseOptimizedPath inline select_optimized_path( const Tensor& a, const Tensor& b, const Tensor& out)

> [spec:et:sem:binary-ops.torch.executor.select-optimized-path-fn]
> Decide whether an optimized elementwise path applies for a,b -> out. Read the
> three scalar types. If a_type != b_type, or a_type != out_type, or a_type is
> Half or BFloat16, return kNone. Then if a.sizes().equals(b.sizes()) OR
> (a.numel()==b.numel() && (a.numel()==out.numel() ||
> internal::sizes_match_ignoring_leading_1s(a.sizes(), b.sizes()))), return
> kTreatAs1d. Otherwise return internal::select_broadcast_optimized_path(a, b).
