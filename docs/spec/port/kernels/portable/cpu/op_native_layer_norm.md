# kernels/portable/cpu/op_native_layer_norm.cpp

> [spec:et:def:op-native-layer-norm.torch.executor.native.layer-norm-fn]
> void layer_norm( const Tensor& input, IntArrayRef normalized_shape, const optional<Tensor>& weight, const optional<Tensor>& bias, CTYPE eps, Tensor& out, Tensor& mean, Tensor& rstd)

> [spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn]
> Templated worker (on `CTYPE`) that layer-normalizes over the trailing `normalized_shape` dims of `input`, writing `out`, `mean`, `rstd`. No return value.
>
> Setup: `dim = input.dim() - normalized_shape.size()` (the first normalized dim index); `dim_size = input.size(dim)`; `leading = getLeadingDims(input, dim)` (product of dims before `dim` = number of independent normalization groups M); `normalized = getTrailingDims(input, dim) * dim_size` (elements per group N — the product of all dims from `dim` onward).
>
> Steps:
> 1. If `leading == 0`, return immediately (empty outputs).
> 2. Acquire `out_data`, `mean_data` (length `leading`), `rstd_data` (length `leading`).
> 3. Degenerate case — if `normalized == 0`: for each `i` in [0,leading) set `mean_data[i] = 0` and `rstd_data[i] = NAN`, then return.
> 4. Resolve `weight_data`/`bias_data` (nullptr when the optional is absent).
> 5. Delegate to `layer_norm_scalar<CTYPE>(input_data, weight_data, bias_data, out_data, mean_data, rstd_data, M=leading, N=normalized, eps)` per `[spec:et:sem:normalization-ops-util.torch.executor.layer-norm-scalar-fn]`: for each group `i` in [0,M), over its contiguous `N` elements `x`, compute `mean = sum(x)/N` and `variance = sum(x^2)/N - mean^2` (accumulated in float), `std = sqrt(variance + eps)`; write `y[j] = (x[j] - mean)/std * (weight[j] if present else 1) + (bias[j] if present else 0)`; store `mean_data[i] = mean` and `rstd_data[i] = 1.0/std`.

> [spec:et:def:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> native_layer_norm_out( KernelRuntimeContext& ctx, const Tensor& input, IntArrayRef normalized_shape, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps, Tensor& out, T...

> [spec:et:sem:op-native-layer-norm.torch.executor.native.native-layer-norm-out-fn]
> Entry point for `native_layer_norm.out`: validates, resizes outputs, dispatches to `layer_norm<CTYPE>`. Returns tuple `(out, mean_out, rstd_out)`.
>
> Steps:
> 1. Build `ret_val = (out, mean_out, rstd_out)`.
> 2. ET_KERNEL_CHECK: `check_layer_norm_args(input, normalized_shape, weight, bias, out, mean_out, rstd_out)` per `[spec:et:sem:normalization-ops-util.torch.executor.check-layer-norm-args-fn]`; on failure set `Error::InvalidArgument` on `ctx`, return `ret_val`.
> 3. ET_KERNEL_CHECK: `tensor_is_default_dim_order(input)` (only default dim order supported); else InvalidArgument, return `ret_val`.
> 4. ET_KERNEL_CHECK: `tensors_have_same_dim_order(input, out, mean_out, rstd_out)`; else InvalidArgument, return `ret_val`.
> 5. If `weight` present, ET_KERNEL_CHECK same dim order as `input`; if `bias` present, likewise. Failure → InvalidArgument, return `ret_val`.
> 6. Compute `mean_rstd_sizes`/`mean_rstd_ndim` via `get_layer_norm_out_target_size(input, normalized_shape, ...)` per `[spec:et:sem:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn]` (= `input`'s leading dims before the normalized region, each trailing normalized dim collapsed to size 1).
> 7. ET_KERNEL_CHECK: `resize_tensor(out, input.sizes())` == Ok; else InvalidArgument, return `ret_val`.
> 8. ET_KERNEL_CHECK: `resize_tensor(mean_out, {mean_rstd_sizes, mean_rstd_ndim})` == Ok; else InvalidArgument, return `ret_val`.
> 9. ET_KERNEL_CHECK: `resize_tensor(rstd_out, {mean_rstd_sizes, mean_rstd_ndim})` == Ok; else InvalidArgument, return `ret_val`.
> 10. Dispatch on `input.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `ret_val`. Invoke `layer_norm<CTYPE>(input, normalized_shape, weight, bias, eps, out, mean_out, rstd_out)` per `[spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn]` (the `double eps` is narrowed to `CTYPE` at the call).
> 11. Return `ret_val`.

