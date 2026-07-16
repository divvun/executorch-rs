# kernels/quantized/cpu/op_mixed_mm.cpp

> [spec:et:def:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn]
> bool check_quantized_mixed_mm_args( const Tensor& in, const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, Tensor& out)

> [spec:et:sem:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn]
> Returns `true` if all argument constraints for mixed_mm hold; on the first
> failing check it logs and returns `false` (ET_LOG_AND_RETURN_IF_FALSE /
> ET_CHECK_OR_RETURN_FALSE). Never aborts, never touches a context.
>
> Checks, in order:
> 1. `in` is rank 2.
> 2. `weight` is rank 2.
> 3. `weight_scales` is rank 1.
> 4. `out` is rank 2.
> 5. `in.size(1) == weight.size(0)` (shared contraction dim; weight is [n, p], not
>    transposed here).
> 6. `weight_scales.size(0) == weight.size(0)` (one scale per contraction-dim row
>    of weight).
> 7. `in`, `weight_scales`, and `out` all have the same dtype.
> 8. `weight.scalar_type() == Char` (int8), else false ("weight dtype must be
>    int8").
> 9. `in.scalar_type()` is `Float` or `Half`, else false.
> 10. If `opt_weight_zero_points` has a value: it must have the same shape as
>     `weight_scales` and the same dtype as `in`.
> 11. `opt_weight_zero_points` must NOT have a value (non-null zero points not yet
>     implemented) — false with "zero points not supported yet." otherwise.
> 12. Return true.
>
> Note: unlike mixed_linear, the weight scale is applied per contraction row `k`
> (`s[k]`), and `weight` is not stored transposed.

> [spec:et:def:op-mixed-mm.torch.executor.native.quantized-mixed-mm-out-fn]
> Tensor& quantized_mixed_mm_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, Tensor& out)

> [spec:et:sem:op-mixed-mm.torch.executor.native.quantized-mixed-mm-out-fn]
> Computes `out = in @ dequant(weight)` where `weight` is int8 quantized with one
> scale per contraction-dim row. `in` is [m, n], `weight` is [n, p], `out` is
> [m, p]. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK(ctx, check_quantized_mixed_mm_args(in, weight, weight_scales,
>    opt_weight_zero_points, out), InvalidArgument, out): on failure sets
>    `Error::InvalidArgument` on `ctx` and returns `out` unchanged per
>    `[spec:et:sem:op-mixed-mm.torch.executor.native.check-quantized-mixed-mm-args-fn]`.
> 2. Compute output sizes `{in.size(0), weight.size(1)}` (i.e. [m, p]);
>    `resize_tensor(out, {output_sizes, 2})`; if not `Error::Ok`, ET_KERNEL_CHECK
>    sets `InvalidArgument` on `ctx` and returns `out`.
> 3. Dispatch `CTYPE` over {Float, Half} keyed on `in.scalar_type()` (out and
>    weight_scales share this dtype per the arg check).
> 4. Set `m = in.size(0)`, `n = in.size(1)`, `p = weight.size(1)`.
> 5. Call `vec_quantized_matmul_int8<CTYPE>(out_data, in_data, weight_int8_data,
>    weight_scales_data, m, n, p)` per
>    `[spec:et:sem:vec-ops.torch.executor.native.vec-quantized-matmul-int8-fn]`,
>    computing `out[i][j] = sum over k in [0,n) of in[i][k] * CTYPE(weight[k][j])
>    * scale[k]`, accumulating in CTYPE. weight is indexed `weight[k*p + j]`
>    (row-major [n, p]); scale is indexed by the contraction index `k`.
> 6. Return `out`.
>
> A non-context overload exists that builds a local `KernelRuntimeContext`,
> forwards, ET_CHECKs `failure_state() == Error::Ok`, and returns the result.
> Non-null zero points are rejected by the arg check, so dequant uses an implicit
> zero_point of 0.

