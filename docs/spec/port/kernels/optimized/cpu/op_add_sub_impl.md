# kernels/optimized/cpu/op_add_sub_impl.h

> [spec:et:def:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn]
> Tensor& opt_add_sub_out_impl( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-add-sub-impl.torch.executor.kernels.impl.opt-add-sub-out-impl-fn]
> Shared implementation of the optimized add.out (is_sub=false) and sub.out
> (is_sub=true) kernels, templated on `is_sub` and a compile-time op_name. Read
> a_type, b_type, out_type. selected_optimized_path = select_optimized_path(a, b,
> out).
>
> (1) Complex branch: if any of a_type/b_type/out_type is complex,
> ET_KERNEL_CHECK that a_type==b_type && a_type==out_type &&
> selected_optimized_path==kTreatAs1d (only same-dtype, same-shape complex
> supported), else record InvalidArgument and return out. Then ET_SWITCH over
> the complex-half type set (CTYPE): alpha_val = scalar_to<CTYPE>(alpha); if
> is_sub negate alpha_val; map over out.numel() elements: out[i] = a[i] +
> alpha_val*b[i] (complex arithmetic). Return out.
>
> (2) kTreatAs1d branch: ET_SWITCH_REALB over a_type (CTYPE): extract alpha into
> alpha_val (ET_KERNEL_CHECK the extraction, else InvalidArgument); if is_sub
> negate alpha_val; map over out.numel(): out[i] = a[i] + alpha_val*b[i].
>
> (3) other-non-kNone (broadcast) branch: ET_SWITCH_REALB over out_type (CTYPE):
> extract alpha into alpha_val (ET_KERNEL_CHECK_MSG else InvalidArgument "Failed
> to extract scalar alpha."). Because Scalar has no unary minus, the sign is
> baked into the lambda instead of alpha:
>   - if is_sub: for the ReverseArguments paths (2dBy1dReverse,
>     LastDimReverse, NdByNdReverse) the lambda is (x,y) -> y - alpha_val*x; for
>     the non-reverse paths it is (x,y) -> x - alpha_val*y.
>   - if !is_sub: for the ReverseArguments paths the lambda is (x,y) -> y +
>     alpha_val*x; for the non-reverse paths it is (x,y) -> x + alpha_val*y.
>   Then call handle_broadcast_elementwise<CTYPE>(ctx, lambda, a, b, out,
>   selected_optimized_path, alpha).
>
> (4) kNone (fully general) fallback: common_type = promoteTypes(a_type,b_type);
> compute_type = get_compute_type(common_type). ET_SWITCH_REALB over compute_type
> (CTYPE_COMPUTE): extract alpha into val_alpha (ET_KERNEL_CHECK else
> InvalidArgument); if is_sub negate val_alpha; apply_bitensor_elementwise_fn with
> REALHBBF16 dtypes for a, b and out, closure (val_a,val_b) -> val_a +
> val_alpha*val_b.
>
> Return out.
