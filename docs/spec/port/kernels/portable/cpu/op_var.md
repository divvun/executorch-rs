# kernels/portable/cpu/op_var.cpp

> [spec:et:def:op-var.torch.executor.native.compute-variance-fn]
> void compute_variance( KernelRuntimeContext& ctx, const Tensor& in, Tensor& out, optional<ArrayRef<int64_t>> dim_list, const size_t num, const double denominator)

> [spec:et:sem:op-var.torch.executor.native.compute-variance-fn]
> Core two-pass variance computation, templated on `CTYPE_IN` (input element
> type) and `CTYPE_OUT` (output/accumulation type). Inputs: `ctx`, `in`, `out`
> (already resized to the reduced shape), `dim_list` (optional list of reduced
> dims; absent/empty means reduce all), `num` (count of elements reduced per
> output element), `denominator` (divisor for the sum of squared deviations —
> `num` for biased, `num-1` for unbiased, or `num-correction` for the
> correction variant). Writes results into `out`.
>
> 1. Degenerate case: if `num == 0` OR `denominator <= 0`, fill every element
>    of `out` (indices 0..out.numel()-1) with `NAN`. Return.
> 2. Else if `in.numel() > 0`:
>    - Build a `MapReduceOverDimListPlan plan(in, dim_list)` (see
>      `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.map-reduce-over-dim-list-plan-fn]`).
>    - Run over each output flat index via
>      `parallel_for_each_reduce_over_dim_list_output_index(in, dim_list, out,
>      lambda)` (see
>      `[spec:et:sem:reduce-util.torch.executor.parallel-for-each-reduce-over-dim-list-output-index-fn]`).
>      For each output index `out_ix` in the assigned `[begin, end)` chunk:
>      - Pass 1 — sum: `sum = plan.execute<CTYPE_IN,CTYPE_OUT>(map = v ->
>        (CTYPE_OUT)v, reduce = (outv, acc) -> acc + outv, out_ix)` per
>        `[spec:et:sem:reduce-util.torch.executor.map-reduce-over-dim-list-plan.execute-fn]`.
>        Then `mean = sum / (CTYPE_OUT)num`.
>      - Pass 2 — sum of squared deviations: `sum2 =
>        plan.execute<CTYPE_IN,CTYPE_OUT>(map = v ->
>        ((CTYPE_OUT)v - mean)*((CTYPE_OUT)v - mean), reduce = (outv, acc) ->
>        acc + outv, out_ix)`.
>      - Write `out_data[out_ix] = sum2 / denominator`.
>    - `ET_KERNEL_CHECK_MSG(success, Internal, ...)`: if the parallel loop
>      reports failure, set Error::Internal on `ctx` and return void (with
>      partial results possibly written).
> 3. If `in.numel() == 0` and not degenerate (num>0, denom>0), `out` is left
>    unwritten by this function.
> Accumulation is entirely in `CTYPE_OUT`; each input value is cast to
> `CTYPE_OUT` before use.

> [spec:et:def:op-var.torch.executor.native.var-correction-out-fn]
> Tensor& var_correction_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, const optional<Scalar>& correction, bool keepdim, Tensor& out)

> [spec:et:sem:op-var.torch.executor.native.var-correction-out-fn]
> Entry point for `var.correction_out`. Arguments: `in`, optional `dim_list`,
> optional `correction` scalar, `keepdim`, and `out`. Returns `out`.
>
> 1. `ET_KERNEL_CHECK(check_reduction_args(in, dim_list, keepdim, {}, out),
>    InvalidArgument)` per
>    `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]`
>    (validates dims in range, unique, and out dtype/shape acceptability). On
>    failure sets InvalidArgument on `ctx` and returns `out`. Unlike `var_out`,
>    this variant does NOT additionally require floating in/out here.
> 2. `ET_KERNEL_CHECK(resize_reduction_out(in, dim_list, keepdim, out) ==
>    Error::Ok, InvalidArgument)` — resizes `out` to the reduced shape (see
>    `[spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]`).
> 3. Resolve correction: `correction_val = 1.0` by default; if `correction` has
>    a value, `correction_val = utils::scalar_to<double>(correction.value())`.
> 4. `num = get_reduced_dim_product(in, dim_list)` (elements per output, see
>    `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`);
>    `denom = num - correction_val` (a double).
> 5. Dispatch `CTYPE_IN` over `in.scalar_type()` with `ET_SWITCH_FLOATHBF16_TYPES`
>    (Float, Double, Half, BFloat16) and nested `CTYPE_OUT` over
>    `out.scalar_type()` with the same set; call
>    `[spec:et:sem:op-var.torch.executor.native.compute-variance-fn]` with
>    `(ctx, in, out, dim_list, num, denom)`.
> 6. Return `out`.

> [spec:et:def:op-var.torch.executor.native.var-out-fn]
> Tensor& var_out( KernelRuntimeContext& ctx, const Tensor& in, optional<ArrayRef<int64_t>> dim_list, bool unbiased, bool keepdim, Tensor& out)

> [spec:et:sem:op-var.torch.executor.native.var-out-fn]
> Entry point for `var.out`. Arguments: `in`, optional `dim_list`, `unbiased`,
> `keepdim`, and `out`. Returns `out`.
>
> 1. `ET_KERNEL_CHECK(check_reduction_args(in, dim_list, keepdim, {}, out),
>    InvalidArgument)` per
>    `[spec:et:sem:reduce-util.torch.executor.check-reduction-args-fn]`.
> 2. `ET_KERNEL_CHECK(tensor_is_floating_type(in), InvalidArgument)` and
>    `ET_KERNEL_CHECK(tensor_is_floating_type(out), InvalidArgument)` — both
>    `in` and `out` must be a floating dtype.
> 3. `ET_KERNEL_CHECK(tensors_have_same_dim_order(in, out), InvalidArgument)`.
> 4. `ET_KERNEL_CHECK(tensor_is_default_dim_order(in), InvalidArgument)`.
> 5. `ET_KERNEL_CHECK(resize_reduction_out(in, dim_list, keepdim, out) ==
>    Error::Ok, InvalidArgument)` (see
>    `[spec:et:sem:reduce-util.torch.executor.resize-reduction-out-fn]`).
> Each failed check sets InvalidArgument on `ctx` and returns `out` unchanged.
> 6. `num = get_reduced_dim_product(in, dim_list)` (see
>    `[spec:et:sem:reduce-util.torch.executor.get-reduced-dim-product-fn]`);
>    `denom = unbiased ? num - 1 : num` (as `size_t`; note when `num == 0` and
>    `unbiased`, `denom` wraps but the degenerate `num == 0` path in
>    `compute_variance` short-circuits to NAN before it is used as a divisor).
> 7. Dispatch `CTYPE_IN` over `in.scalar_type()` and nested `CTYPE_OUT` over
>    `out.scalar_type()` with `ET_SWITCH_FLOATHBF16_TYPES` (Float, Double, Half,
>    BFloat16); call
>    `[spec:et:sem:op-var.torch.executor.native.compute-variance-fn]` with
>    `(ctx, in, out, dim_list, num, (double)denom)`.
> 8. Return `out`.

