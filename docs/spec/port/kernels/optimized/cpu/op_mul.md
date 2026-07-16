# kernels/optimized/cpu/op_mul.cpp

> [spec:et:def:op-mul.torch.executor.native.opt-mul-out-fn]
> Tensor& opt_mul_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-mul.torch.executor.native.opt-mul-out-fn]
> Optimized mul.out (a * b). Read a_type,b_type,out_type;
> common_type=promoteTypes(a_type,b_type). ET_KERNEL_CHECK
> canCast(common_type,out_type). ET_KERNEL_CHECK
> tensors_have_same_dim_order(a,b,out). ET_KERNEL_CHECK
> resize_to_broadcast_target_size(a,b,out)==Ok. op_name="mul.out".
>
> Single-element fast path: if b.numel()==1 and a_type==b_type==out_type and not
> Half/BFloat16: nested ET_SWITCH_REALB over a_type (CTYPE) then b_type (CTYPE_B):
> b_val=*b_data (CTYPE_B); b_casted=(CTYPE)b_val; map over out.numel(): out[i] =
> a[i] * b_casted. Return out. Else if a.numel()==1: return opt_mul_out(ctx, b,
> a, out) (swap args).
>
> selected_optimized_path = select_optimized_path(a,b,out).
>   - kTreatAs1d: if out_type complex, ET_KERNEL_CHECK a_type==b_type==out_type,
>     ET_SWITCH complex-half (CTYPE), map2 out[i]=a[i]*b[i] (complex). Else
>     ET_SWITCH_REALB over out_type (CTYPE), map2 out[i]=a[i]*b[i].
>   - other non-kNone (broadcast): if out_type complex, ET_KERNEL_CHECK
>     a_type==b_type==out_type, ET_SWITCH complex-half (CTYPE) with lambda (x,y)->
>     x*y; else ET_SWITCH_REALB over out_type (CTYPE) with lambda (x,y)->x*y;
>     either way call handle_broadcast_elementwise<CTYPE>(ctx, mul_lambda, a, b,
>     out, selected_optimized_path). (No alpha.)
>   - kNone: if any of a/b/out complex, ET_KERNEL_CHECK a_type==b_type==out_type,
>     ET_SWITCH complex-half (CTYPE), apply_binary_elementwise_fn closure
>     (val_a,val_b)->val_a*val_b. Else compute_type=get_compute_type(common_type),
>     ET_SWITCH_REALB over compute_type (CTYPE_COMPUTE),
>     apply_bitensor_elementwise_fn (a,b,out REALHBBF16) closure
>     (val_a,val_b)->val_a*val_b.
> Return out.

> [spec:et:def:op-mul.torch.executor.native.opt-mul-scalar-out-fn]
> Tensor& opt_mul_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-mul.torch.executor.native.opt-mul-scalar-out-fn]
> Optimized mul.Scalar_out (a * b, b a Scalar). a_type=a.scalar_type();
> common_type=promote_type_with_scalar(a_type,b); out_type. ET_KERNEL_CHECK
> common_type==out_type. ET_KERNEL_CHECK tensors_have_same_dim_order(a,out).
> ET_KERNEL_CHECK resize_tensor(out,a.sizes())==Ok. op_name="mul.Scalar_out".
>
> If a_type==common_type==out_type and not Half/BFloat16: ET_SWITCH_REALB over
> a_type (CTYPE): b_casted=scalar_to<CTYPE>(b); map over out.numel(): out[i] =
> a[i] * b_casted.
> Else: compute_type=get_compute_type(common_type); ET_SWITCH_REALB over
> compute_type (CTYPE_COMPUTE): val_b=scalar_to<CTYPE_COMPUTE>(b);
> apply_unitensor_elementwise_fn (a REALHBBF16, out SAME_AS_COMMON) closure
> val_a -> val_a * val_b. Return out.
