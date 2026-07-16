# kernels/quantized/cpu/op_mixed_linear.cpp

> [spec:et:def:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn]
> bool check_quantized_mixed_linear_args( const Tensor& in, const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const std::optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn]
> Returns `true` if all argument constraints for mixed_linear hold; on the first
> failing check it logs and returns `false` (via ET_LOG_AND_RETURN_IF_FALSE /
> ET_CHECK_OR_RETURN_FALSE). It never aborts and never touches a context.
>
> Checks, in order (each returns false on failure):
> 1. `in` is rank 2.
> 2. `weight` is rank 2.
> 3. `weight_scales` is rank 1 or rank 2.
> 4. `out` is rank 2.
> 5. `in.size(1) == weight.size(1)` (shared contraction dim; weight is stored
>    transposed so its dim 1 is the input feature dim).
> 6. `weight_scales.size(0) == weight.size(0)`.
> 7. `in.size(1) == weight.size(1)` (checked again, redundant).
> 8. `in` and `weight_scales` have the same dtype.
> 9. If `dtype` has a value: `out.scalar_type() == dtype.value()`, and
>    `dtype.value()` must be `Float` or `Half` (else returns false with message
>    "dtype must be Float or Half").
> 10. `weight.scalar_type() == Char` (int8), else false ("weight dtype must be
>     int8").
> 11. `in.scalar_type()` is `Float` or `Half`, else false.
> 12. If `opt_weight_zero_points` has a value: it must have the same shape as
>     `weight_scales` and the same dtype as `in`.
> 13. `opt_weight_zero_points` must NOT have a value (non-null zero points are not
>     implemented yet) — returns false with "zero points not supported yet." if
>     it does. (Steps 12 and 13 together mean any provided zero_points are
>     shape/dtype-validated and then still rejected.)
> 14. Return true.

> [spec:et:def:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn]
> Tensor& quantized_mixed_linear_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& weight, const Tensor& weight_scales, const std::optional<Tensor>& opt_weight_zero_points, const std::optional<ScalarType> dtype, Tensor& out)

> [spec:et:sem:op-mixed-linear.torch.executor.native.quantized-mixed-linear-out-fn]
> Computes `out = in @ dequant(weight)^T` where `weight` is int8, quantized
> per-channel (per output row) and optionally per-group along the contraction dim.
> `in` is [m, n], `weight` is [p, n] (transposed layout), `out` is [m, p].
> Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK(ctx, check_quantized_mixed_linear_args(in, weight,
>    weight_scales, opt_weight_zero_points, dtype, out), InvalidArgument, out):
>    on failure sets `Error::InvalidArgument` on `ctx` and returns `out`
>    unchanged (no computation) per
>    `[spec:et:sem:op-mixed-linear.torch.executor.native.check-quantized-mixed-linear-args-fn]`.
> 2. `out_dtype = dtype.has_value() ? dtype.value() : out.scalar_type()`.
> 3. Compute output sizes: `output_sizes = {in.size(0), weight.size(0)}` (i.e.
>    [m, p]); `resize_tensor(out, {output_sizes, 2})`; if not `Error::Ok`,
>    ET_KERNEL_CHECK sets `InvalidArgument` on `ctx` and returns `out`.
> 4. Dispatch: `CTYPE` over {Float, Half} keyed on `in.scalar_type()`; `CTYPE_OUT`
>    over float types plus Half keyed on `out_dtype` (ET_SWITCH_FLOAT_TYPES_AND
>    (Half): Float, Double, Half).
> 5. Set `m = in.size(0)`, `n = in.size(1)`, `p = weight.size(0)`, `g = n`. If
>    `weight_scales.dim() == 2`, override `g = ceil(n / weight_scales.size(1)) =
>    (n + weight_scales.size(1) - 1) / weight_scales.size(1)` (group size along
>    the contraction dim).
> 6. Call `vec_quantized_matmul_transb_int8<CTYPE_OUT, CTYPE>(out_data, in_data,
>    weight_int8_data, weight_scales_data, m, n, p, g)` per
>    `[spec:et:sem:vec-ops.torch.executor.native.vec-quantized-matmul-transb-int8-fn]`,
>    which computes `out[i][j] = sum over k in [0,n) of in[i][k] *
>    CTYPE(weight[j][k]) * scale[j][k/g]`, accumulating in the output ctype and
>    grouping the scale index by `k/g` (the last group may be short). Scales are
>    read as CTYPE (same as input), matching check step 8; the "FIXME: currently
>    ignores dtype" means the accumulator/scale narrow to CTYPE, and only the
>    stored result uses CTYPE_OUT.
> 7. Return `out`.
>
> A non-context overload exists that builds a local `KernelRuntimeContext`,
> forwards, ET_CHECKs `failure_state() == Error::Ok`, and returns the result.
> Non-null zero points are rejected by the arg check, so the dequant here uses an
> implicit zero_point of 0.

