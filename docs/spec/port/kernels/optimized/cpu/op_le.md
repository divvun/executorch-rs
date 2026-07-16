# kernels/optimized/cpu/op_le.cpp

> [spec:et:def:op-le.torch.executor.native.opt-le-scalar-out-fn]
> Tensor& opt_le_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-le.torch.executor.native.opt-le-scalar-out-fn]
> Optimized le.Scalar_out (a <= b, b a Scalar). a_type=a.scalar_type();
> b_type=get_scalar_dtype(b); common_type=promote_type_with_scalar(a_type,b);
> out_type. ET_KERNEL_CHECK tensors_have_same_dim_order(a,out). ET_KERNEL_CHECK
> resize_tensor(out,a.sizes())==Ok. op_name="le.Scalar_out".
>
> If a_type==common_type==out_type and not Half/BFloat16: nested ET_SWITCH_REALB
> over a_type (CTYPE) then b_type (CTYPE_B): extract b into b_val
> (ET_EXTRACT_SCALAR, default 0); b_casted=(CTYPE)b_val; map over a.numel():
> out[i] = a[i].le(b_casted), i.e. (a[i] <= b_casted) written into out (same
> CTYPE, boolean 0/1).
> Else fall back to internal::comparison_scalar_out<std::less_equal, op_name>(
> ctx, a, b, out). Return out.

> [spec:et:def:op-le.torch.executor.native.opt-le-tensor-out-fn]
> Tensor& opt_le_tensor_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-le.torch.executor.native.opt-le-tensor-out-fn]
> Optimized le.Tensor_out (a <= b). a_type=a.scalar_type(); out_type.
> ET_KERNEL_CHECK tensors_have_same_dim_order(a,b,out). ET_KERNEL_CHECK
> resize_to_broadcast_target_size(a,b,out)==Ok. op_name="le.Tensor_out".
> selected_optimized_path=select_optimized_path(a,b,out).
>   - kTreatAs1d: ET_SWITCH_REALB over a_type (CTYPE): map2 out[i] = a[i].le(b[i])
>     ((a[i] <= b[i]) as CTYPE).
>   - other non-kNone (broadcast): ET_SWITCH_REALB over out_type (CTYPE): lambda
>     (x,y)->x.le(y); handle_broadcast_elementwise<CTYPE>(ctx, le_lambda, a, b,
>     out, selected_optimized_path).
>   - kNone: internal::comparison_tensor_out<std::less_equal, op_name>(ctx,a,b,
>     out).
> Return out.
