# kernels/portable/cpu/util/normalization_ops_util.cpp, kernels/portable/cpu/util/normalization_ops_util.h

> [spec:et:def:normalization-ops-util.torch.executor.check-batch-norm-args-fn]
> bool check_batch_norm_args( const Tensor& in, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const std::optional<Tensor>& running_mean, const std::optional<Tensor>& running_var, double momentum, double eps, Tenso...

> [spec:et:sem:normalization-ops-util.torch.executor.check-batch-norm-args-fn]
> Validates argument dtypes and per-channel parameter shapes for batch norm.
> Signature `check_batch_norm_args(in, weight, bias, running_mean, running_var,
> momentum, eps, out, mean_out, var_out)` where `weight`, `bias`,
> `running_mean`, `running_var` are `std::optional<Tensor>`. `momentum` and
> `eps` are accepted but not inspected here. Returns `bool`.
>
> Every check uses `ET_LOG_AND_RETURN_IF_FALSE` (logs and returns `false` on
> the first failed predicate); returns `true` if all pass. Order:
>
> Dtype checks (each optional tensor only checked if present via `has_value()`):
> 1. If `weight` present: `weight` has same dtype as `in`.
> 2. If `bias` present: same dtype as `in`.
> 3. If `running_mean` present: same dtype as `in`.
> 4. If `running_var` present: same dtype as `in`.
> 5. `out` same dtype as `in`.
> 6. `mean_out` same dtype as `in`.
> 7. `var_out` same dtype as `in`.
>
> Channel-dim resolution: `C_dim = (in.dim() >= 1) ? 1 : 0` — the channel
> dimension is index 1 (NCHW-style) unless `in` is 0-dimensional, in which case
> it is 0.
>
> Per-parameter shape checks (each only if the optional tensor is present):
> 8. If `weight` present: `weight` is rank 1 AND
>    `weight.size(0) == in.size(C_dim)`.
> 9. If `bias` present: rank 1 AND `bias.size(0) == in.size(C_dim)`.
> 10. If `running_mean` present: rank 1 AND `size(0) == in.size(C_dim)`.
> 11. If `running_var` present: rank 1 AND `size(0) == in.size(C_dim)`.
>
> Pure validation predicate; writes no output.

> [spec:et:def:normalization-ops-util.torch.executor.check-group-norm-args-fn]
> bool check_group_norm_args( const Tensor& in, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW, int64_t group, Tensor& out, Tensor& mean_out, Tensor& rstd_out)

> [spec:et:sem:normalization-ops-util.torch.executor.check-group-norm-args-fn]
> Validates arguments for group norm. Signature
> `check_group_norm_args(in, weight, bias, N, C, HxW, group, out, mean_out,
> rstd_out)` where `weight`/`bias` are `std::optional<Tensor>` and `N`, `C`,
> `HxW`, `group` are `int64_t`. Returns `bool`.
>
> Checks use two macro forms: `ET_LOG_AND_RETURN_IF_FALSE` (logs and returns
> `false`) and `ET_CHECK_OR_RETURN_FALSE(cond, fmt, ...)` (returns `false` with
> a formatted message when `cond` is false). Both short-circuit-return `false`
> on failure. Order:
> 1. `in.size(0) == N`.
> 2. `in.size(1) == C`.
> 3. `in.numel() == N * C * HxW`.
> 4. `group > 0`.
> 5. `C % group == 0` (channels divisible by number of groups).
> 6. `weight` is absent OR (`weight` is rank 1 AND `weight.size(0) == C`).
> 7. `bias` is absent OR (`bias` is rank 1 AND `bias.size(0) == C`).
> 8. If `weight` present: `weight` has same dtype as `in`.
> 9. If `bias` present: `bias` same dtype as `in`.
> 10. `out` same dtype as `in`.
> 11. `mean_out` same dtype as `in`.
> 12. `rstd_out` same dtype as `in`.
>
> Returns `true` if all pass. Pure validation predicate; writes no output.

> [spec:et:def:normalization-ops-util.torch.executor.check-layer-norm-args-fn]
> bool check_layer_norm_args( const Tensor& in, IntArrayRef normalized_shape, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, Tensor& out, Tensor& mean_out, Tensor& rstd_out)

> [spec:et:sem:normalization-ops-util.torch.executor.check-layer-norm-args-fn]
> Validates arguments for layer norm. Signature
> `check_layer_norm_args(in, normalized_shape, weight, bias, out, mean_out,
> rstd_out)` where `normalized_shape` is an `IntArrayRef` and `weight`/`bias`
> are `std::optional<Tensor>`. Returns `bool`.
>
> Let `ndim = normalized_shape.size()`. Uses `ET_CHECK_OR_RETURN_FALSE(cond,
> ...)` and `ET_LOG_AND_RETURN_IF_FALSE(...)`, both returning `false` on the
> first failure. Order:
> 1. `ndim >= 1` (normalized_shape must be at least 1-dimensional).
> 2. `in.dim() >= ndim` (input rank at least the length of normalized_shape).
> 3. `ndim <= kTensorDimensionLimit`.
> 4. Let `shift = in.dim() - ndim`. For each `d` in `[0, ndim)`:
>    `in.size(d + shift) == normalized_shape[d]` — the rightmost `ndim`
>    dimensions of `in` must exactly equal `normalized_shape`.
> 5. Build `shape` = `normalized_shape` cast element-wise to
>    `executorch::aten::SizesType` (length `ndim`).
> 6. If `weight` present: `weight` has same dtype as `in` AND `weight` has
>    exactly the sizes `shape` (i.e. shape equal to `normalized_shape`).
> 7. If `bias` present: `bias` same dtype as `in` AND `bias` has sizes `shape`.
> 8. `out` same dtype as `in`.
> 9. `mean_out` same dtype as `in`.
> 10. `rstd_out` same dtype as `in`.
>
> Returns `true` if all pass. Pure validation predicate; writes no output.

> [spec:et:def:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn]
> void get_layer_norm_out_target_size( const Tensor& in, IntArrayRef normalized_shape, Tensor::SizesType* mean_rstd_sizes, size_t* mean_rstd_ndim)

> [spec:et:sem:normalization-ops-util.torch.executor.get-layer-norm-out-target-size-fn]
> Computes the target shape for the `mean`/`rstd` auxiliary outputs of layer
> norm. Signature `get_layer_norm_out_target_size(in, normalized_shape,
> mean_rstd_sizes, mean_rstd_ndim)`; `mean_rstd_sizes` is a caller buffer,
> `mean_rstd_ndim` an out-param. No return value; no validation.
>
> Behavior:
> 1. Set `*mean_rstd_ndim = in.dim()` (same rank as input).
> 2. For each `d` in `[0, in.dim())`:
>    - If `d < in.dim() - normalized_shape.size()` (a leading, non-normalized
>      dim): `mean_rstd_sizes[d] = in.size(d)`.
>    - Otherwise (one of the trailing normalized dims): `mean_rstd_sizes[d] = 1`.
>
> Result: the leading (batch-like) dims of `in` are preserved and each of the
> trailing `normalized_shape.size()` dims is collapsed to 1, giving the
> per-row mean/rstd statistics shape. The main `out` tensor keeps `in`'s full
> shape (handled elsewhere).

> [spec:et:def:normalization-ops-util.torch.executor.layer-norm-scalar-fn]
> inline void layer_norm_scalar( const CTYPE* input_data, const CTYPE* weight_data, // nullable const CTYPE* bias_data, // nullable CTYPE* out_data, CTYPE* mean_data, CTYPE* rstd_data, size_t M, size_t N, float eps)

> [spec:et:sem:normalization-ops-util.torch.executor.layer-norm-scalar-fn]
> Scalar layer-norm kernel over `M` rows of `N` elements each, templated on the
> element type `CTYPE`. Signature `layer_norm_scalar<CTYPE>(input_data,
> weight_data, bias_data, out_data, mean_data, rstd_data, M, N, eps)`.
> `weight_data` and `bias_data` are nullable raw pointers (null means the
> gamma/beta term is omitted). All array pointers are contiguous row-major
> buffers. No return value; writes results in place. The caller must handle the
> `M == 0` and `N == 0` edge cases before calling (this function assumes
> `N >= 1`; with `N == 0` it would divide by zero).
>
> For each row `i` in `[0, M)`:
> - Let `x = input_data + i*N` (row input) and `y = out_data + i*N` (row output).
> - Compute the mean and variance in `float` (statistics are always accumulated
>   as `float`, regardless of `CTYPE`):
>   - `sum = std::accumulate(x, x+N, 0.0f)` — sum of the row's elements
>     accumulated in `float`, left to right.
>   - `sq_sum = 0`; then for each `j` in `[0, N)`:
>     `sq_sum += static_cast<float>(x[j]) * x[j]` (sum of squares in `float`).
>   - `mean_value = sum / N`.
>   - `variance = sq_sum / N - mean_value * mean_value` (biased variance via
>     `E[x^2] - E[x]^2`; divisor is `N`, not `N-1`).
>   - `std = std::sqrt(variance + eps)`.
> - Normalize each element `j` in `[0, N)`:
>   - `w = weight_data ? weight_data[j] : static_cast<CTYPE>(1)`.
>   - `b = bias_data ? bias_data[j] : static_cast<CTYPE>(0)`.
>   - `y[j] = (x[j] - mean_value) / std * w + b`. (Note: `mean_value` and `std`
>     are `float`, `x[j]`/`w`/`b` are `CTYPE`; the expression is evaluated in
>     the usual-arithmetic-conversion type and stored back to `CTYPE` in
>     `y[j]`.)
> - Store statistics: `mean_data[i] = mean_value` and `rstd_data[i] = 1.0 / std`
>   (the reciprocal standard deviation), each converted to `CTYPE`.
>
> Iteration is row-major, outer over rows then inner over columns. `eps` is a
> `float` added to the variance before the square root for numerical stability.

