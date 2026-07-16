# kernels/portable/cpu/op_native_group_norm.cpp

> [spec:et:def:op-native-group-norm.torch.executor.native.group-norm-fn]
> void group_norm( const Tensor& input, const optional<Tensor>& weight, const optional<Tensor>& bias, int64_t sN, int64_t sC, int64_t sHxW, int64_t group, double eps, Tensor& out, Tensor& mean, Tensor& rstd)

> [spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn]
> Templated worker (on element type `CTYPE`) that computes group normalization over `input` and fills `out`, `mean`, `rstd`. `input` is logically [N, C, HxW]; channels are split into `group` contiguous groups. Statistics are computed per (batch, group) pair over its `D*HxW` elements, where `D = C/group`. No return value; writes through the output tensors.
>
> Setup: `N=sN`, `C=sC`, `HxW=sHxW`, `G=group`; `leading = N*G`; `D = C/G`; `inner_size = D*HxW`.
>
> Steps:
> 1. If `leading == 0` (empty batch or zero groups), return immediately (outputs untouched).
> 2. Acquire `out_data`, `mean_data`, `rstd_data` pointers (`mean`/`rstd` have `leading` elements).
> 3. Degenerate spatial case — if `inner_size == 0`: for each `i` in [0,leading) set `mean_data[i] = 0` and `rstd_data[i] = NAN`, then return.
> 4. Resolve `weight_data`/`bias_data` pointers (nullptr when the corresponding optional is absent).
> 5. For each `i` in [0,leading) (the flattened (batch,group) index): let `x = input_data + i*inner_size` (the group's contiguous span). Compute `sum = reduce_add(x, inner_size)` and `sq_sum = vec_powerf(x, inner_size)` as `float`. `mean_value = (double)sum / inner_size`; `variance = (double)sq_sum / inner_size - mean_value*mean_value`; `std = sqrt(variance + eps)`; `rstd_value = 1.0 / std`. Note statistic accumulation is done in float (via `reduce_add`/`vec_powerf`) then finalized in double.
> 6. Output write:
>    - If both `weight_data` and `bias_data` are null: `y = out_data + i*inner_size`; for each `j` in [0,inner_size): `y[j] = (CTYPE)(((double)x[j] - mean_value) * rstd_value)` (per-group affine-free normalization).
>    - Else (weight and/or bias present): affine is applied per channel. `g = i % G` (this group's index within its batch). For each `j` in [0,D): channel `ch = g*D + j`; `scale = rstd_value * (weight ? weight_data[ch] : 1.0)`; `beta = -scale*mean_value + (bias ? bias_data[ch] : 0.0)`; then over the channel's `HxW` span at `input_data + (i*D+j)*HxW` → `out_data + (i*D+j)*HxW`: `y[k] = (CTYPE)(scale*(double)x[k] + beta)` for each `k` in [0,HxW).
> 7. Store `mean_data[i] = (CTYPE)mean_value` and `rstd_data[i] = (CTYPE)rstd_value`.

> [spec:et:def:op-native-group-norm.torch.executor.native.native-group-norm-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> native_group_norm_out( KernelRuntimeContext& ctx, const Tensor& input, const std::optional<Tensor>& weight, const std::optional<Tensor>& bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double...

> [spec:et:sem:op-native-group-norm.torch.executor.native.native-group-norm-out-fn]
> Entry point for `native_group_norm.out`: validates arguments, resizes the three outputs, and dispatches to the `group_norm<CTYPE>` worker. Returns tuple `(out, mean_out, rstd_out)`.
>
> Steps:
> 1. Build `ret_val = (out, mean_out, rstd_out)`.
> 2. ET_KERNEL_CHECK: `check_group_norm_args(input, weight, bias, N, C, HxW, group, out, mean_out, rstd_out)` per `[spec:et:sem:normalization-ops-util.torch.executor.check-group-norm-args-fn]`; on failure set `Error::InvalidArgument` on `ctx`, return `ret_val`.
> 3. Set target `mean_rstd_sizes = {N, group}` (ndim 2).
> 4. ET_KERNEL_CHECK: `resize_tensor(out, input.sizes())` == Ok; else InvalidArgument, return `ret_val`.
> 5. ET_KERNEL_CHECK: `resize_tensor(mean_out, {N, group})` == Ok; else InvalidArgument, return `ret_val`.
> 6. ET_KERNEL_CHECK: `resize_tensor(rstd_out, {N, group})` == Ok; else InvalidArgument, return `ret_val`.
> 7. ET_KERNEL_CHECK: `tensor_is_default_dim_order(input)`; else InvalidArgument, return `ret_val`.
> 8. ET_KERNEL_CHECK: `tensors_have_same_dim_order(input, out, mean_out, rstd_out)`; else InvalidArgument, return `ret_val`.
> 9. If `weight` present, ET_KERNEL_CHECK same dim order as `input`; if `bias` present, likewise. Failure → InvalidArgument, return `ret_val`.
> 10. Dispatch on `input.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `ret_val`. Invoke `group_norm<CTYPE>(input, weight, bias, N, C, HxW, group, eps, out, mean_out, rstd_out)` per `[spec:et:sem:op-native-group-norm.torch.executor.native.group-norm-fn]`.
> 11. Return `ret_val`.

