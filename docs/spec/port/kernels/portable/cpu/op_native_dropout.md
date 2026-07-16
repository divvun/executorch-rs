# kernels/portable/cpu/op_native_dropout.cpp

> [spec:et:def:op-native-dropout.torch.executor.native.native-dropout-out-fn]
> std::tuple<Tensor&, Tensor&> native_dropout_out( KernelRuntimeContext& ctx, const Tensor& input, double prob, std::optional<bool> train, Tensor& out, Tensor& mask)

> [spec:et:sem:op-native-dropout.torch.executor.native.native-dropout-out-fn]
> Applies dropout to `input`: produces `out` (input with dropped elements zeroed, not rescaled) and a boolean `mask` (true = kept). Returns tuple `(out, mask)`.
>
> Steps:
> 1. Build `ret = (out, mask)`.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dtype(input, out)`; on failure set `Error::InvalidArgument` on `ctx`, return `ret` unchanged.
> 3. ET_KERNEL_CHECK: `tensors_have_same_dim_order(input, out, mask)`; else InvalidArgument, return `ret`.
> 4. ET_KERNEL_CHECK: `resize_tensor(out, input.sizes())` == Ok; else InvalidArgument, return `ret`.
> 5. ET_KERNEL_CHECK: `resize_tensor(mask, input.sizes())` == Ok; else InvalidArgument, return `ret`.
> 6. ET_KERNEL_CHECK: `tensor_is_bool_type(mask)` (mask must be Bool dtype); else InvalidArgument, return `ret`.
> 7. ET_KERNEL_CHECK_MSG: `prob >= 0 && prob <= 1` (message includes the offending value); else InvalidArgument, return `ret`.
> 8. Active-dropout case — if `(train has no value OR train == true) AND prob != 0`:
>    a. Seed a `std::mt19937` from `std::random_device` and a `std::uniform_real_distribution<double>` over the default [0,1) range. For each element index `ii` in `mask.numel()`, set `mask[ii] = (dist(gen) >= prob)` (true means kept). Note: nondeterministic (freshly seeded per call), so exact drawn values are not reproducible.
>    b. Dispatch on `input.scalar_type()` over FLOATHBF16 (Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `ret`. Apply the binary elementwise functor per `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]` over `input` (loaded as FLOATHBF16) and `mask` (loaded as BOOL_OR_BYTE), computing `mask_val ? val : 0` in `CTYPE_COMPUTE`, storing to `out` with the SAME_AS_COMMON policy (out dtype equals input dtype). No 1/(1-prob) rescaling is applied.
> 9. Inference/no-op case — else if `input.numel() > 0`: `memcpy` all of `input`'s bytes into `out` (identity copy) and `memset` `mask` to all-true (every byte set true). If `input.numel() == 0`, both outputs are left as the empty resized tensors.
> 10. Return `ret`.

