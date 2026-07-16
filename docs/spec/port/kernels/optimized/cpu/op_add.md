# kernels/optimized/cpu/op_add.cpp

> [spec:et:def:op-add.torch.executor.native.opt-add-out-fn]
> Tensor& opt_add_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-add.torch.executor.native.opt-add-out-fn]
> Optimized add.out (a + alpha*b). Read a_type,b_type,out_type; common_type =
> promoteTypes(a_type,b_type). ET_KERNEL_CHECK canCast(common_type,out_type) &&
> check_alpha_type(get_scalar_dtype(alpha), common_type) (else InvalidArgument).
> ET_KERNEL_CHECK tensors_have_same_dim_order(a,b,out). ET_KERNEL_CHECK
> resize_to_broadcast_target_size(a,b,out)==Ok. op_name="add.out".
>
> Fast path when b is a single element (b.numel()==1):
>   - If any dtype is complex: ET_KERNEL_CHECK a_type==b_type==out_type; ET_SWITCH
>     complex-half (CTYPE): alpha_val=scalar_to<CTYPE>(alpha), b_val=*b_data;
>     map over out.numel(): out[i] = a[i] + (alpha_val*b_val). Return out.
>   - Else if a_type==b_type==out_type and a_type is not Half/BFloat16: nested
>     ET_SWITCH_REALB over a_type (CTYPE) then b_type (CTYPE_B): extract alpha into
>     alpha_val; b_val=*b_data (CTYPE_B); b_casted=(CTYPE)b_val; map over
>     out.numel(): out[i] = a[i] + (alpha_val*b_casted). Return out.
> Else if a.numel()==1: return opt_add_out(ctx, b, a, alpha, out) (swap args;
> add is commutative in a,b for the a+alpha*b form because the scalar-side fast
> path multiplies the single element by alpha — here b becomes the tensor and a
> the scalar).
>
> Otherwise delegate to
> kernels::impl::opt_add_sub_out_impl<false, op_name>(ctx, a, b, alpha, out).

> [spec:et:def:op-add.torch.executor.native.opt-add-scalar-out-fn]
> Tensor& opt_add_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-add.torch.executor.native.opt-add-scalar-out-fn]
> Optimized add.Scalar_out (a + alpha*b, b a Scalar). a_type=a.scalar_type();
> common_type = promote_type_with_scalar(a_type, b); out_type=out.scalar_type().
> ET_KERNEL_CHECK common_type==a_type &&
> check_alpha_type(get_scalar_dtype(alpha), common_type). ET_KERNEL_CHECK
> tensors_have_same_dim_order(a,out). ET_KERNEL_CHECK
> resize_tensor(out,a.sizes())==Ok. op_name="add.Scalar_out".
>
> If a_type==common_type==out_type and a_type not Half/BFloat16: ET_SWITCH_REALB
> over a_type (CTYPE): b_casted=scalar_to<CTYPE>(b); extract alpha into alpha_val;
> map over out.numel(): out[i] = a[i] + (alpha_val*b_casted).
> Else: compute_type=get_compute_type(common_type); ET_SWITCH_REALB over
> compute_type (CTYPE_COMPUTE): val_b=scalar_to<CTYPE_COMPUTE>(b); extract alpha
> into val_alpha; val_alpha_times_b = val_alpha*val_b;
> apply_unitensor_elementwise_fn (a REALHBBF16, out SAME_AS_COMMON) closure
> val_a -> val_a + val_alpha_times_b. Return out.
