# kernels/optimized/cpu/op_where.cpp

> [spec:et:def:op-where.torch.executor.native.opt-where-out-fn]
> Tensor& opt_where_out( KernelRuntimeContext& ctx, const Tensor& cond, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-where.torch.executor.native.opt-where-out-fn]
> Optimized where.self_out (cond ? a : b). common_type=promoteTypes(a.
> scalar_type(), b.scalar_type()). ET_KERNEL_CHECK common_type==out.scalar_type().
> ET_KERNEL_CHECK tensors_have_same_dim_order(cond,a,b,out). ET_KERNEL_CHECK
> resize_to_broadcast_target_size(a,b,cond,out)==Ok.
> compute_type=get_compute_type(common_type). op_name="where.self_out".
>
> Fast path when a,b,out all share compute_type and cond is Bool: out_numel =
> out.numel(); ET_SWITCH_REALB over compute_type (CTYPE_COMPUTE): grab typed data
> pointers data_a, data_b (CTYPE_COMPUTE), data_cond (bool), data_out. Run a
> parallel_for over [0,out_numel) with GRAIN_SIZE; each chunk [begin,end)
> constructs a BroadcastIndexesRange<3>(out, a, b, cond), advances its begin
> iterator by `begin`, then iterates while the current output index (element [0])
> < end: destructure (out_index, a_index, b_index, cond_index) = *it; set
> data_out[out_index] = data_cond[cond_index] ? data_a[a_index] :
> data_b[b_index].
>
> Mixed-dtype fallback: ET_SWITCH_REALB over compute_type (CTYPE_COMPUTE):
> apply_tritensor_elementwise_fn (a REALHBBF16, b REALHBBF16, cond BOOL_OR_BYTE,
> out SAME_AS_COMMON) closure (val_a,val_b,val_c)-> val_c ? val_a : val_b.
> Return out.
