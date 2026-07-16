# kernels/optimized/cpu/op_fft_c2r.cpp

> [spec:et:def:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn]
> Tensor& opt_fft_c2r_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef dim, int64_t normalization, int64_t last_dim_size, Tensor& out)

> [spec:et:sem:op-fft-c2r.torch.executor.native.opt-fft-c2r-out-fn]
> Out-variant complex-to-real inverse FFT (`_fft_c2r.out`). Given a
> one-sided complex input tensor `in`, transform axes `dim`, a
> `normalization` mode, the real length `last_dim_size` of the last
> transformed axis, and a pre-allocated `out`, computes the N-D
> complex->real inverse DFT of `in` along `dim`, writing the real result
> into `out`, and returns `out`.
>
> Steps:
> 1. Read `in.sizes()`. Kernel-check `in.dim() <= kTensorDimensionLimit`
>    (return `out` on failure).
> 2. Kernel-check `!dim.empty()`.
> 3. Kernel-check `last_dim_size >= 1`.
> 4. Build `out_sizes` as a copy of `in.sizes()` into a
>    `kTensorDimensionLimit`-length stack buffer sliced to the input rank,
>    then set `out_sizes[dim.back()] = last_dim_size` (expand the last
>    transformed axis from its one-sided length to the real length).
> 5. Kernel-check `tensors_have_same_dim_order(in, out)`.
> 6. Kernel-check `in.scalar_type() == toComplexType(out.scalar_type())`
>    (the input must be the complex type corresponding to the output real
>    type).
> 7. For each `d` in `dim`, kernel-check `d >= 0 && d < in.dim()` (message
>    "dims must be in bounds (got <d>)").
> 8. Kernel-check `resize_tensor(out, out_sizes) == Error::Ok` (message
>    "Failed to resize output tensor (last dim <n>).").
> 9. Build pocketfft `axes` from `dim`, `out_shape` from `out.sizes()`, and
>    byte strides `in_stride`/`out_stride` from `in`/`out`.
> 10. Dispatch over the OUTPUT scalar type with ET_SWITCH_FLOAT_TYPES
>    (float/double only, matching upstream). In the lambda:
>    a. Compute `fct = compute_fct<CTYPE_OUT>(ctx, out, dim, normalization)`
>       — note the factor is computed from the OUTPUT tensor's sizes over
>       `dim`. If nullopt (unsupported normalization recorded the failure),
>       return from the lambda without writing output.
>    b. Call `pocketfft::c2r<CTYPE_OUT>(out_shape, in_stride, out_stride,
>       axes, forward=false, in_cdata, out_data, *fct)` — the inverse
>       complex->real transform along `axes`, scaled by `fct`, reading the
>       one-sided complex spectrum and writing `last_dim_size` real samples
>       along the last axis into `out`.
> 11. Return `out`.

