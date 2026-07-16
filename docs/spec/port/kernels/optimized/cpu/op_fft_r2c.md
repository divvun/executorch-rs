# kernels/optimized/cpu/op_fft_r2c.cpp

> [spec:et:def:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn]
> Tensor& opt_fft_r2c_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef dim, int64_t normalization, bool onesided, Tensor& out)

> [spec:et:sem:op-fft-r2c.torch.executor.native.opt-fft-r2c-out-fn]
> Out-variant real-to-complex FFT (`_fft_r2c.out`). Given a real input
> tensor `in`, a list of transform axes `dim`, a `normalization` mode, a
> `onesided` flag, and a pre-allocated `out`, computes the N-D real->complex
> DFT of `in` along the axes in `dim`, writing the (one-sided) complex
> spectrum into `out`, and returns `out`.
>
> Steps:
> 1. Read `in.sizes()`. Kernel-check `in.dim() <= kTensorDimensionLimit`
>    (return `out` on failure).
> 2. Build `out_sizes` as a copy of `in.sizes()` into a
>    `kTensorDimensionLimit`-length stack buffer sliced to the input rank.
> 3. Kernel-check `!dim.empty()`.
> 4. Kernel-check `tensors_have_same_dim_order(in, out)`.
> 5. Kernel-check `onesided` is true (message "onesided=False is not
>    supported yet in _fft_r2c") — the non-onesided path is unimplemented.
> 6. Kernel-check `out.scalar_type() == toComplexType(in.scalar_type())`
>    (the output must be the complex type corresponding to the input real
>    type).
> 7. For each `d` in `dim`, kernel-check `d >= 0 && d < in.dim()` (message
>    "dims must be in bounds (got <d>)").
> 8. If `onesided`, halve the last transformed axis:
>    `out_sizes[dim.back()] = out_sizes[dim.back()] / 2 + 1`.
> 9. Kernel-check `resize_tensor(out, out_sizes) == Error::Ok` (message
>    "Failed to resize output tensor (last dim <n>).").
> 10. Build pocketfft `axes` from `dim`, `in_shape` from `in.sizes()`, and
>    byte strides `in_stride`/`out_stride` from `in`/`out`.
> 11. Dispatch over the input scalar type with ET_SWITCH_FLOAT_TYPES
>    (float/double only, matching upstream). In the lambda:
>    a. Compute `fct = compute_fct<CTYPE_IN>(ctx, in, dim, normalization)`.
>       If it is nullopt (unsupported normalization recorded the failure),
>       return from the lambda without writing output.
>    b. Call `pocketfft::r2c<CTYPE_IN>(in_shape, in_stride, out_stride, axes,
>       forward=true, in_data, out_cdata, *fct)` — the forward real->complex
>       transform along `axes`, scaled by `fct`, writing the one-sided
>       spectrum (last axis has N/2+1 complex samples, DC at index 0) into
>       `out`. (Conjugate-symmetry fill for the non-onesided case is a TODO
>       and not performed.)
> 12. Return `out`.

