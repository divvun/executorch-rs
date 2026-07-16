# kernels/optimized/cpu/op_sub.cpp

> [spec:et:def:op-sub.torch.executor.native.opt-sub-out-fn]
> Tensor& opt_sub_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-sub.torch.executor.native.opt-sub-out-fn]
> Optimized sub.out (a - alpha*b). Read a_type,b_type,alpha_type =
> get_scalar_dtype(alpha), out_type. ET_KERNEL_CHECK alpha_type != Bool.
> common_type=promoteTypes(a_type,b_type). ET_KERNEL_CHECK
> canCast(common_type,out_type) && canCast(alpha_type,common_type).
> ET_KERNEL_CHECK tensors_have_same_dim_order(a,b,out). ET_KERNEL_CHECK
> resize_to_broadcast_target_size(a,b,out)==Ok. op_name="sub.out".
>
> Single-element fast path (a.numel()==1 || b.numel()==1) when
> a_type==b_type==out_type and not Half/BFloat16: choose (tensor,scalar): if
> a.numel()==1 then tensor=&b, scalar=&a else tensor=&a, scalar=&b (with their
> types). Nested ET_SWITCH_REAL over tensor_type (CTYPE) then scalar_type
> (CTYPE_SCALAR): extract alpha into alpha_val; scalar_val=*scalar_data
> (CTYPE_SCALAR); scalar_casted=(CTYPE)scalar_val; map over out.numel():
>   - if a.numel()==1: out[i] = scalar_casted - alpha_val*tensor[i].
>   - else: out[i] = tensor[i] - (alpha_val*scalar_casted).
> Return out.
>
> Otherwise delegate to
> kernels::impl::opt_add_sub_out_impl<true, op_name>(ctx, a, b, alpha, out).

> [spec:et:def:op-sub.torch.executor.native.opt-sub-scalar-out-fn]
> Tensor& opt_sub_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-sub.torch.executor.native.opt-sub-scalar-out-fn]
> Optimized sub.Scalar_out (a - alpha*b, b a Scalar). a_type=a.scalar_type();
> common_type=promote_type_with_scalar(a_type,b); alpha_type=get_scalar_dtype
> (alpha); out_type. ET_KERNEL_CHECK alpha_type != Bool. ET_KERNEL_CHECK
> common_type==out_type && canCast(alpha_type,common_type). ET_KERNEL_CHECK
> tensors_have_same_dim_order(a,out). ET_KERNEL_CHECK
> resize_tensor(out,a.sizes())==Ok. op_name="sub.Scalar_out".
>
> If a_type==common_type==out_type and not Half/BFloat16: ET_SWITCH_REAL over
> a_type (CTYPE): b_casted=scalar_to<CTYPE>(b); extract alpha into alpha_val; map
> over out.numel(): out[i] = a[i] - (alpha_val*b_casted).
> Else: compute_type=get_compute_type(common_type); ET_SWITCH_REAL over
> compute_type (CTYPE_COMPUTE): val_b=scalar_to<CTYPE_COMPUTE>(b),
> val_alpha=scalar_to<CTYPE_COMPUTE>(alpha), val_alpha_times_b=val_alpha*val_b;
> apply_unitensor_elementwise_fn (a REALHBF16, out SAME_AS_COMMON) closure val_a
> -> val_a - val_alpha_times_b. Return out.
