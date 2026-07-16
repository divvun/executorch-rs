# kernels/portable/cpu/op_var_mean.cpp

> [spec:et:def:op-var-mean.torch.executor.native.compute-var-mean-fn]
> void compute_var_mean( KernelRuntimeContext& ctx, const Tensor& in, Tensor& var_out, Tensor& mean_out, optional<ArrayRef<int64_t>> dim_list, const size_t num, const double denominator)

> [spec:et:sem:op-var-mean.torch.executor.native.compute-var-mean-fn]
> Two-pass computation of both variance and mean, templated on `CTYPE_IN` and
> `CTYPE_OUT`. Inputs: `ctx`, `in`, `var_out`, `mean_out` (both resized to the
> reduced shape), optional `dim_list`, `num` (elements reduced per output), and
> `denominator` (divisor for variance). Writes `var_out` and `mean_out`.
>
> 1. Degenerate case: if `num == 0` OR `denominator <= 0`, fill every element of
>    both `var_out` and `mean_out` (indices 0..var_out.numel()-1) with `NAN`.
>    Return.
> 2. Else if `in.numel() > 0`:
>    - Fast path (used only when `dim_list` has a value with exactly one entry,
>      `in.scalar_type() == var_out.scalar_type()`, that dim normalized to
>      `d = dim < 0 ? dim + in.dim() : dim` is the innermost dim
>      (`d == in.dim()-1`, `0 <= d < in.dim()`), and `in` is contiguous):
>      Let `reduce_size = in.size(d)`, `outer_size = in.numel()/reduce_size`,
>      `cnum = (CTYPE_OUT)num`, `cdenom = (CTYPE_OUT)denominator`. For each
>      `i` in [0, outer_size): let `row = in_data + i*reduce_size`.
>      Pass 1 sums `row[0..reduce_size)` into `sum` (CTYPE_OUT), computes
>      `mean = sum/cnum`, writes `mean_data[i] = mean`. Pass 2 accumulates
>      `sum2 += (row[j]-mean)^2` (CTYPE_OUT), writes `var_data[i] =
>      sum2/cdenom`. (Note this fast path divides variance by the CTYPE_OUT-cast
>      denominator, whereas the general path divides by the raw double
>      `denominator`.)
>    - General path (when the fast path is not taken): build
>      `MapReduceOverDimListPlan plan(in, dim_list)` (see
>      `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]`)
>      and run `parallel_for_each_reduce_over_dim_list_output_index(in,
>      dim_list, var_out, lambda)` (see
>      `[spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]`).
>      For each `out_ix`: Pass 1 `sum = plan.execute(map = v -> (CTYPE_OUT)v,
>      reduce = acc+outv, out_ix)`, `mean = sum/(CTYPE_OUT)num`, store
>      `mean_data[out_ix] = mean`. Pass 2 `sum2 = plan.execute(map = v ->
>      ((CTYPE_OUT)v - mean)^2, reduce = acc+outv, out_ix)`, store
>      `var_data[out_ix] = sum2 / denominator`. `ET_KERNEL_CHECK_MSG(success,
>      Internal, ...)`: on parallel-loop failure sets Error::Internal on `ctx`
>      and returns void.
> 3. If `in.numel() == 0` and not degenerate, neither output is written.
> All accumulation is in `CTYPE_OUT`; inputs are cast to `CTYPE_OUT` before use.

> [spec:et:def:op-var-mean.torch.executor.native.var-mean-correction-out-fn]
> std::tuple<Tensor&, Tensor&> var_mean_correction_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, const optional<Scalar>& correction, bool keepdim, Tensor& out0, Tensor& out1)

> [spec:et:sem:op-var-mean.torch.executor.native.var-mean-correction-out-fn]
> Entry point for `var_mean.correction_out`. Arguments: `in`, optional
> `dim_list`, optional `correction` scalar, `keepdim`, and the two outputs
> `out0` (variance) and `out1` (mean). Returns `std::tuple<Tensor&, Tensor&>`
> bound to `(out0, out1)`.
>
> 1. Construct `ret_val = (out0, out1)`.
> 2. `ET_KERNEL_CHECK(check_reduction_args(in, dim_list, keepdim, {}, out0),
>    InvalidArgument)` then the same check against `out1`, per
>    `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]`.
> 3. `ET_KERNEL_CHECK(resize_reduction_out(in, dim_list, keepdim, out0) ==
>    Error::Ok, InvalidArgument)` then the same for `out1`, per
>    `[spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]`.
> Each failed check sets InvalidArgument on `ctx` and returns `ret_val` with
> outputs unchanged.
> 4. Resolve correction: `correction_val = 1.0`; if `correction` has a value,
>    `correction_val = utils::scalar_to<double>(correction.value())`.
> 5. `num = get_reduced_dim_product(in, dim_list)` (see
>    `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`);
>    `denom = num - correction_val` (double).
> 6. Dispatch `CTYPE_IN` over `in.scalar_type()` and nested `CTYPE_OUT` over
>    `out0.scalar_type()` with `ET_SWITCH_FLOATHBF16_TYPES` (Float, Double,
>    Half, BFloat16); call
>    `[spec:et:sem:op-var-mean.torch.executor.native.compute-var-mean-fn]` with
>    `(ctx, in, out0, out1, dim_list, num, denom)`. Here `out0` is the variance
>    output and `out1` the mean output.
> 7. Return `ret_val`.

