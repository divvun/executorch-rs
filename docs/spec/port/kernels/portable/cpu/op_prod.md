# kernels/portable/cpu/op_prod.cpp

> [spec:et:def:op-prod.torch.executor.native.prod-int-out-fn]
> Tensor& prod_int_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, bool keepdim, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-prod.torch.executor.native.prod-int-out-fn]
> Product of `in` reduced over a single dimension `dim` (with optional `keepdim`), writing per-output-index products into `out`; returns `out`. (`prod.int_out`.)
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_reduction_args_single_dim(in, dim, keepdim, dtype, out, /*allow_empty_dim=*/true)` per `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-single-dim-fn]`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `resize_reduction_out(in, dim, keepdim, out)` == Ok per `[spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]` (out shape = `in` with `dim` removed, or size-1 if `keepdim`); else InvalidArgument, return `out`.
> 3. Dispatch on `in.scalar_type()` (CTYPE_IN) then on `out.scalar_type()` (CTYPE_OUT), both over REALHBBF16 (Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 4. Reduce with `parallel_for_each_reduce_over_dim_output_index` per `[spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-output-index-fn]`: for each output flat index `out_ix`, if `in.numel() > 0` compute the running product over the `dim` slice via `map_reduce_over_dim` (`[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-fn]`) — map each input value to `CTYPE_OUT`, reduce with `acc * outv` starting from `prod = 1`; if `in.numel() == 0` the result stays `1`. Write `out_data[out_ix] = prod`. Accumulation is in `CTYPE_OUT` (integer products wrap two's-complement).
> 5. ET_KERNEL_CHECK_MSG: if the parallel_for reports failure, set `Error::Internal` on `ctx` (message "parallel_for failed") and return `out`.
> 6. Return `out`.

> [spec:et:def:op-prod.torch.executor.native.prod-out-fn]
> Tensor& prod_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-prod.torch.executor.native.prod-out-fn]
> Product of ALL elements of `in` into a scalar (0-dim) `out`; returns `out`. (`prod` full reduction.)
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_prod_out_args(in, dtype, out)` per `[spec:et:sem:reduce-util.torch.executor.check-prod-out-args-fn]`; on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. ET_KERNEL_CHECK: `resize_tensor(out, {})` == Ok (out is 0-dim / scalar); else InvalidArgument, return `out`.
> 3. Dispatch on `in.scalar_type()` (CTYPE_IN) then on `out.scalar_type()` (CTYPE_OUT), both over REALHBBF16 (Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 4. Initialize `data_out[0] = (CTYPE_OUT)1`; for each flat index `i` in [0,in.numel()): `data_out[0] *= (CTYPE_OUT)data_in[i]` (each input element cast to `CTYPE_OUT` before multiplying; accumulation in `CTYPE_OUT`, integer wrap on overflow). An empty input yields the product `1`.
> 5. Return `out`.

