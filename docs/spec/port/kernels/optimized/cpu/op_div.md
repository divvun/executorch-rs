# kernels/optimized/cpu/op_div.cpp

> [spec:et:def:op-div.torch.executor.native.get-common-type-fn]
> ScalarType get_common_type(ScalarType a_type, ScalarType b_type)

> [spec:et:sem:op-div.torch.executor.native.get-common-type-fn]
> Compute the common dtype for division. If either a_type or b_type is complex,
> return promoteTypes(a_type,b_type). ET_CHECK neither is a QInt/Bits type. If
> both are floating, return promoteTypes(a_type,b_type). Else if a is floating
> return a_type; else if b is floating return b_type; else return Float (integer
> division promotes to Float).

> [spec:et:def:op-div.torch.executor.native.opt-div-out-fn]
> Tensor& opt_div_out( KernelRuntimeContext& ctx, const Tensor& a, const Tensor& b, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.opt-div-out-fn]
> Optimized div.out (a / b). ET_KERNEL_CHECK tensors_have_same_dim_order(a,b,out).
> ET_KERNEL_CHECK resize_to_broadcast_target_size(a,b,out)==Ok. op_name="div.out".
> Read a_type,b_type,out_type.
>
> Complex branch: if a or b is complex, common_type=get_common_type(a_type,
> b_type), ET_SWITCH complex (CTYPE): loop i in [0,out.numel()): out[i] =
> a[i]/b[i] (complex division). Return out.
>
> Single-element fast path: if (a.numel()==1 || b.numel()==1) and
> a_type==b_type==out_type and not Half/BFloat16: choose (tensor,scalar): if
> a.numel()==1 then tensor=&b/scalar=&a else tensor=&a/scalar=&b (with types).
> Nested ET_SWITCH_REALB over tensor_type (CTYPE) then scalar_type (CTYPE_SCALAR):
> scalar_val=*scalar_data; scalar_casted=(CTYPE)scalar_val. If a.numel()==1: map
> out[i] = scalar_casted / tensor[i]. Else: inv = CTYPE(1)/scalar_casted; map
> out[i] = tensor[i] * inv. Return out.
>
> selected_optimized_path=select_optimized_path(a,b,out).
>   - kTreatAs1d: ET_SWITCH_REALB over out_type (CTYPE): map2 out[i]=a[i]/b[i].
>   - other non-kNone (broadcast): ET_SWITCH_REALB over out_type (CTYPE): for the
>     ReverseArguments paths (2dBy1dReverse, LastDimReverse, NdByNdReverse) the
>     lambda is (x,y)->y/x, else (x,y)->x/y; call
>     handle_broadcast_elementwise<CTYPE>(ctx, div_lambda, a, b, out,
>     selected_optimized_path).
>   - kNone: common_type=get_common_type(a.scalar_type(),b.scalar_type());
>     compute_type=get_compute_type(common_type); ET_SWITCH_FLOAT over
>     compute_type (CTYPE_COMPUTE): apply_bitensor_elementwise_fn (out FLOATHBF16;
>     a,b REALHBBF16) closure (val_a,val_b)->val_a/val_b.
> Return out.

> [spec:et:def:op-div.torch.executor.native.opt-div-scalar-out-fn]
> Tensor& opt_div_scalar_out( KernelRuntimeContext& ctx, const Tensor& a, const Scalar& b, Tensor& out)

> [spec:et:sem:op-div.torch.executor.native.opt-div-scalar-out-fn]
> Optimized div.Scalar_out (a / b, b a Scalar). a_type=a.scalar_type();
> b_type=get_scalar_dtype(b); common_type = isFloatingType(a_type) ? a_type :
> Float; out_type. ET_KERNEL_CHECK common_type==out_type. ET_KERNEL_CHECK
> tensors_have_same_dim_order(a,out). ET_KERNEL_CHECK
> resize_tensor(out,a.sizes())==Ok. op_name="div.Scalar_out".
>
> If a_type==common_type==out_type and not Half/BFloat16: nested ET_SWITCH_REAL
> over a_type (CTYPE) then ET_SWITCH_REALB over b_type (CTYPE_B): extract b into
> b_val (ET_EXTRACT_SCALAR); b_casted=(CTYPE)b_val; inv_b=CTYPE(1)/b_casted; map
> over out.numel(): out[i] = a[i] * inv_b.
> Else: compute_type=get_compute_type(common_type); ET_SWITCH_FLOAT over
> compute_type (CTYPE_COMPUTE): val_b=scalar_to<CTYPE_COMPUTE>(b);
> apply_unitensor_elementwise_fn (a REALHBBF16, out SAME_AS_COMMON) closure
> val_a -> val_a / val_b. Return out.
