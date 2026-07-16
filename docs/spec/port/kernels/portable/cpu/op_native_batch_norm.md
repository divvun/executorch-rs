# kernels/portable/cpu/op_native_batch_norm.cpp

> [spec:et:def:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-no-stats-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_stats_out( KernelRuntimeContext& ctx, const Tensor& in, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, bool training, double momentum, double eps,...

> [spec:et:sem:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-no-stats-out-fn]
> Batch norm with no precomputed running stats: computes per-channel mean and variance from `in` itself (training-style normalization over N and spatial dims), normalizes, applies optional affine `weight`/`bias`, and also writes the computed per-channel `mean_out` and `invstd_out`. Returns the tuple `(out, mean_out, invstd_out)`. `training` is ignored.
>
> Layout assumption: `in` is NCHW-like — dim 0 is batch `N`, dim 1 is channel `C`, remaining dims are spatial. `mean_out`/`invstd_out` are length-`C` vectors.
>
> Steps:
> 1. Build `ret_val = (out, mean_out, invstd_out)` (all three are the return-on-failure value).
> 2. ET_KERNEL_CHECK: `check_batch_norm_args(in, weight, bias, /*running_mean=*/nullopt, /*running_var=*/nullopt, momentum, eps, out, mean_out, invstd_out)` per `[spec:et:sem:normalization-ops-util.torch.executor.check-batch-norm-args-fn]`; on failure set `Error::InvalidArgument` on `ctx`, return `ret_val`.
> 3. ET_KERNEL_CHECK: `in` must have contiguous dim order (`is_contiguous_dim_order(in.dim_order())`); else InvalidArgument, return `ret_val`.
> 4. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out, mean_out, invstd_out)`; else InvalidArgument, return `ret_val`.
> 5. If `weight` present, ET_KERNEL_CHECK `tensors_have_same_dim_order(in, weight)`; if `bias` present, likewise for `bias`. Failure → InvalidArgument, return `ret_val`.
> 6. ET_KERNEL_CHECK: `in.dim() >= 2`; else InvalidArgument, return `ret_val`.
> 7. Let `N = in.size(0)`, `C = in.size(1)`, `inner = getTrailingDims(in, 1)` (product of all dims after channel), `elements_per_channel = N * inner`.
> 8. ET_KERNEL_CHECK three resizes: `resize_tensor(out, in.sizes())`, `resize_tensor(mean_out, {C})`, `resize_tensor(invstd_out, {C})` must each return `Error::Ok`; else InvalidArgument, return `ret_val`.
> 9. Dispatch on `in.scalar_type()` over FLOATHBF16 (ET_SWITCH_FLOATHBF16_TYPES = Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `ret_val`.
> 10. Zero-initialize `mean_data` and `invstd_data` (each length `C`) via memset. First accumulation pass: for each batch `b` in [0,N), for each channel `c` in [0,C), let `x` point at the `inner`-length contiguous span `in_data + b*C*inner + c*inner`; add `reduce_add(x, inner)` (sum) into `mean_data[c]` and add `vec_powerf(x, inner)` (sum of squares) into `invstd_data[c]`.
> 11. Finalize per channel `c`: `mean = mean_data[c] / elements_per_channel`; `var = invstd_data[c] / elements_per_channel - mean*mean` (i.e. E[x^2]-E[x]^2); `invstd = 1.0 / sqrt(var + eps)`; store `mean_data[c] = mean`, `invstd_data[c] = invstd`.
> 12. Normalization/write pass: iterate `i` in [0,N), `c` in [0,C); `weight_val = weight[c]` if present else 1; `bias_val = bias[c]` if present else 0; for each of the `inner` elements advancing `out_data` and `in_data` together: `*out_data = (*in_data - mean_data[c]) * invstd_data[c] * weight_val + bias_val`.
> 13. Return `ret_val`. Accumulation and arithmetic are performed in `CTYPE` (the input's float dtype); `1.0 / sqrt(...)` uses `std::sqrt`.

> [spec:et:def:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-no-training-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_training_out( KernelRuntimeContext& ctx, const Tensor& in, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, const Tensor& running_mean, const Tensor...

> [spec:et:sem:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-no-training-out-fn]
> Inference batch norm using precomputed `running_mean` and `running_var`: normalizes `in` per channel with these stats, applies optional affine `weight`/`bias`, writes result to `out`, and returns empty `mean_out`/`invstd_out`. Returns tuple `(out, mean_out, invstd_out)`. `momentum` is accepted but unused (no stats update).
>
> Layout: `in` treated as [outer..., C, inner...] where the channel dim is dim 1 when `in.dim() >= 1` else dim 0.
>
> Steps:
> 1. Build `ret_val = (out, mean_out, invstd_out)`.
> 2. ET_KERNEL_CHECK: `resize_tensor(out, in.sizes())` == Ok; else InvalidArgument, return `ret_val`.
> 3. ET_KERNEL_CHECK: `resize_tensor(mean_out, {0})` == Ok (empty output); else InvalidArgument, return `ret_val`.
> 4. ET_KERNEL_CHECK: `resize_tensor(invstd_out, {0})` == Ok (empty output); else InvalidArgument, return `ret_val`.
> 5. ET_KERNEL_CHECK: `check_batch_norm_args(in, weight, bias, running_mean, running_var, momentum, eps, out, mean_out, invstd_out)` per `[spec:et:sem:normalization-ops-util.torch.executor.check-batch-norm-args-fn]`; else InvalidArgument, return `ret_val`.
> 6. ET_KERNEL_CHECK: `in` has contiguous dim order (`is_contiguous_dim_order`); else InvalidArgument, return `ret_val`.
> 7. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out, mean_out, invstd_out)`; else InvalidArgument, return `ret_val`.
> 8. If `weight` present, ET_KERNEL_CHECK same dim order as `in`; if `bias` present, likewise. Failure → InvalidArgument, return `ret_val`.
> 9. `C_dim = (in.dim() >= 1) ? 1 : 0`; `C = in.size(C_dim)`; `outer = getLeadingDims(in, C_dim)` (product of dims before channel); `inner = getTrailingDims(in, C_dim)` (product of dims after channel).
> 10. Dispatch on `in.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `ret_val`.
> 11. For each `i` in [0,outer), for each channel `c` in [0,C): `mean = running_mean[c]`; `var = running_var[c]`; `invstd = 1.0 / sqrt(var + eps)`; `weight_val = weight[c]` if present else 1; `bias_val = bias[c]` if present else 0; then for each of the `inner` elements advancing `out_data`/`in_data` together: `*out_data = (*in_data - mean) * invstd * weight_val + bias_val`. Iteration order is outer→channel→inner, matching the NCHW contiguous memory layout.
> 12. `mean_out` and `invstd_out` remain empty (length 0).
> 13. Return `ret_val`. Arithmetic in `CTYPE`; `std::sqrt` used for invstd.

> [spec:et:def:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_out( KernelRuntimeContext& ctx, const Tensor& in, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, Tensor& running_mean, Tensor& running_var, bool trai...

> [spec:et:sem:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-out-fn]
> `_native_batch_norm_legit` variant that takes mutable `running_mean`/`running_var` and a `training` flag but only supports inference in the portable kernel. Returns tuple `(out, mean_out, invstd_out)`.
>
> Steps:
> 1. Build `ret_val = (out, mean_out, invstd_out)`.
> 2. ET_KERNEL_CHECK_MSG: `training == false` (message "Portable kernels only support inference mode!"); if `training` is true, set `Error::InvalidArgument` on `ctx` and return `ret_val` unchanged.
> 3. Delegate to `_native_batch_norm_legit_no_training_out(ctx, in, weight, bias, running_mean, running_var, momentum, eps, out, mean_out, invstd_out)` per `[spec:et:sem:op-native-batch-norm.torch.executor.native.native-batch-norm-legit-no-training-out-fn]` and return its result. (The `running_mean`/`running_var` tensors are read as inference stats, never updated.)

