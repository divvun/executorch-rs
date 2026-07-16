# kernels/optimized/cpu/op_sum.cpp

> [spec:et:def:op-sum.torch.executor.native.opt-sum-dim-out-fn]
> Tensor& opt_sum_dim_out( KernelRuntimeContext& ctx, const Tensor& in, std::optional<ArrayRef<int64_t>> dim_list, bool keepdim, std::optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-sum.torch.executor.native.opt-sum-dim-out-fn]
> Optimized `sum.IntList_out`. Validates args then, for a narrow "fast" shape,
> runs a tight vectorized reduction; otherwise defers to the portable
> `sum_dim_out`. Steps:
> 1. ET_KERNEL_CHECK `check_reduction_args(in, dim_list, keepdim, dtype, out)`
>    (InvalidArgument on fail, return out).
> 2. ET_KERNEL_CHECK `resize_reduction_out(in, dim_list, keepdim, out) ==
>    Error::Ok`.
> 3. ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)`.
> 4. ET_KERNEL_CHECK `tensor_is_default_dim_order(in)`.
> 5. If `in.numel() == 0`: if `out.numel() > 0`, memset out's data buffer to 0
>    over `out.nbytes()`; return out.
> 6. Compute `fast_eligible = dim_list.has_value() && dim_list.size()==1 &&
>    in.scalar_type()==out.scalar_type() && !isComplexType(in.scalar_type()) &&
>    tensor_is_contiguous(in)`.
> 7. If fast_eligible: resolve `d` (negative dim wraps via `+ in.dim()`).
>    Compute `outer_size = product of sizes[0..d]`, `reduce_size = in.size(d)`,
>    `inner_size = product of sizes[d+1..in.dim()]`. Switch over
>    REALHBBF16 dtypes (op name "sum.IntList_out"): get typed in/out pointers;
>    if `inner_size == 1` call `sum_innermost(ip, op, outer_size, reduce_size)`,
>    else `sum_strided(ip, op, outer_size, reduce_size, inner_size)`; set
>    `handled = true`. If handled, return out.
> 8. Fallback: return `sum_dim_out(ctx, in, dim_list, keepdim, dtype, out)`.

> [spec:et:def:op-sum.torch.executor.native.sum-innermost-fn]
> inline void sum_innermost( const CTYPE* in, CTYPE* out, int64_t outer_size, int64_t reduce_size)

> [spec:et:sem:op-sum.torch.executor.native.sum-innermost-fn]
> Contiguous innermost reduction: sum each row of `reduce_size` contiguous
> elements into one scalar. `out[i] = sum over j of in[i*reduce_size + j]`.
> fp32 accumulates in fp32; Half/BFloat16 accumulate in fp32 (each element cast
> to float before adding), then the fp32 sum is cast back to CTYPE on store. The
> C++ vectorizes the inner sum with `Vectorized<float>` (loadu, vector add, then
> `vec_reduce_all` horizontal add) plus a scalar tail; DEVIATION: the Rust port
> collapses this to a plain scalar accumulation loop over all `reduce_size`
> elements (autovectorizes; numerics match to op tolerance). For each `i in
> 0..outer_size`: `sum = 0.0f`; for `j in 0..reduce_size`: `sum += (float)
> row[j]`; `out[i] = (CTYPE) sum`.

> [spec:et:def:op-sum.torch.executor.native.sum-strided-fn]
> inline void sum_strided( const CTYPE* in, CTYPE* out, int64_t outer_size, int64_t reduce_size, int64_t inner_size)

> [spec:et:sem:op-sum.torch.executor.native.sum-strided-fn]
> Non-innermost (strided) single-dim reduction. Layout is
> [outer_size, reduce_size, inner_size]; reduce over the middle axis:
> `out[o*inner_size + j] = sum over k of in[o*reduce_size*inner_size +
> k*inner_size + j]`. `outer_stride = reduce_size * inner_size`. fp32
> accumulates in fp32; Half/BFloat16 accumulate in fp32 and cast back on store.
> The C++ vectorizes across the contiguous inner axis (kVecSize output positions
> per step) with a scalar tail; DEVIATION: the Rust port collapses to a plain
> scalar loop nest. For each `o in 0..outer_size`: for each `j in 0..inner_size`:
> `sum = 0.0f`; for `k in 0..reduce_size`: `sum += (float) in[o*outer_stride +
> k*inner_size + j]`; `out[o*inner_size + j] = (CTYPE) sum`.
