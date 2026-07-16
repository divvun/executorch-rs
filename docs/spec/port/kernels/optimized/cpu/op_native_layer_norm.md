# kernels/optimized/cpu/op_native_layer_norm.cpp

> [spec:et:def:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> opt_native_layer_norm_out( KernelRuntimeContext& ctx, const Tensor& input, IntArrayRef normalized_shape, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, double eps, Tensor& out, Tensor& mean_out, Tensor& rstd_out)

> [spec:et:sem:op-native-layer-norm.torch.executor.native.opt-native-layer-norm-out-fn]
> Optimized `native_layer_norm.out`. Same validation/resize contract as the
> portable op; the numeric kernel uses a Welford (RowwiseMoments) path for large
> N and falls back to the portable scalar layer_norm for small N. Steps:
> 1. `(void)ctx`. Build `ret_val = (out, mean_out, rstd_out)`.
> 2. ET_KERNEL_CHECK `check_layer_norm_args(input, normalized_shape, weight,
>    bias, out, mean_out, rstd_out)` (InvalidArgument, return ret_val).
> 3. Compute mean/rstd target size via `get_layer_norm_out_target_size(input,
>    normalized_shape, &mean_rstd_sizes, &mean_rstd_ndim)`.
> 4. ET_KERNEL_CHECK `resize_tensor(out, input.sizes()) == Error::Ok`.
> 5. ET_KERNEL_CHECK `resize_tensor(mean_out, {mean_rstd_sizes, mean_rstd_ndim})
>    == Error::Ok`.
> 6. ET_KERNEL_CHECK `resize_tensor(rstd_out, {...}) == Error::Ok`.
> 7. Switch over FLOATHBF16 dtypes (op name "native_layer_norm.out") on
>    `input.scalar_type()`, calling the templated worker `layer_norm<CTYPE>(
>    input, normalized_shape, weight, bias, eps, out, mean_out, rstd_out)`.
> 8. Return ret_val.
> NOTE (vs portable): unlike the portable op, this optimized variant does NOT
> check `tensor_is_default_dim_order` / `tensors_have_same_dim_order` — ported
> bug-for-bug.

> [spec:et:def:op-native-layer-norm.torch.executor.native.layer-norm-fn]
> void layer_norm( const Tensor& input, IntArrayRef normalized_shape, const optional<Tensor>& weight, const optional<Tensor>& bias, CTYPE eps, Tensor& out, Tensor& mean, Tensor& rstd)

> [spec:et:sem:op-native-layer-norm.torch.executor.native.layer-norm-fn]
> Templated per-row layer norm worker. `dim = input.dim() -
> normalized_shape.size()`; `dim_size = input.size(dim)`; `M =
> getLeadingDims(input, dim)`; `N = getTrailingDims(input, dim) * dim_size`.
> If `M == 0` return. Fetch `out_data`, `mean_data`, `rstd_data` pointers.
> If `N == 0`: for `i in 0..M` set `mean_data[i] = 0`, `rstd_data[i] = NAN`;
> return. Fetch `input_data`; `gamma_data = weight ? weight.data : nullptr`;
> `beta_data = bias ? bias.data : nullptr`; record `gamma_null`, `beta_null`.
> If `N < kSmallNThreshold` (256), delegate to the portable
> `layer_norm_scalar<CTYPE>(input_data, gamma_data, beta_data, out_data,
> mean_data, rstd_data, M, N, eps)` and return.
> Otherwise, for each row `i in 0..M`: `src = input_data + i*N`, `dst = out_data
> + i*N`. Compute `(mean_val, rstd_val) = RowwiseMoments(src, N)` (in acc_t,
> i.e. f32 for 16-bit float CTYPE). Set `rstd_val = 1 / sqrt(rstd_val + eps)`.
> `scale = rstd_val`; `offset = -rstd_val * mean_val`. Then produce outputs:
> if gamma or beta is null, scalar loop for `j in 0..N`: `gamma_v = gamma_null ?
> 1 : gamma_data[j]`; `beta_v = beta_null ? 0 : beta_data[j]`; `dst[j] =
> (src[j]*scale + offset)*gamma_v + beta_v`. Else (both present) apply the same
> elementwise map `(x*scale + offset)*gamma + beta` across the row (C++ uses
> `at::vec::map3`; DEVIATION: scalar loop in the Rust port). Store
> `mean_data[i] = mean_val`, `rstd_data[i] = rstd_val` (cast to CTYPE on store).
