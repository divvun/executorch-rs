# kernels/optimized/cpu/fft_utils.h

> [spec:et:def:fft-utils.torch.executor.native.compute-fct-fn]
> std::optional<T> compute_fct(KernelRuntimeContext& ctx, int64_t size, int64_t normalization)

> [spec:et:sem:fft-utils.torch.executor.native.compute-fct-fn]
> Compute the scalar normalization factor `T` for an FFT.
> Size overload `compute_fct(ctx, size, normalization)`: cast `normalization`
> to `fft_norm_mode` and switch — `none` returns 1; `by_n` returns
> `1 / size`; `by_root_n` returns `1 / sqrt(size)` (all in `T`). Any other
> value fails the kernel check with InvalidArgument ("Unsupported
> normalization type: <n>") and returns nullopt.
> Tensor overload `compute_fct(ctx, t, dim, normalization)`: if
> `normalization` is `none`, return 1 immediately; otherwise compute the
> signal size `n` as the product of `t.sizes()[idx]` over every `idx` in
> `dim`, then delegate to the size overload.

> [spec:et:def:fft-utils.torch.executor.native.fft-norm-mode]
> enum class fft_norm_mode {
>   none;
>   by_root_n;
>   by_n;
> }

> [spec:et:def:fft-utils.torch.executor.native.shape-from-tensor-fn]
> inline pocketfft::shape_t shape_from_tensor(const Tensor& t)

> [spec:et:sem:fft-utils.torch.executor.native.shape-from-tensor-fn]
> Build a pocketfft `shape_t` (vector of unsigned extents) by copying the
> tensor's `sizes()` element-for-element. It is the logical N-D extent used
> to drive the transform.

> [spec:et:def:fft-utils.torch.executor.native.stride-from-tensor-fn]
> inline pocketfft::stride_t stride_from_tensor(const Tensor& t)

> [spec:et:sem:fft-utils.torch.executor.native.stride-from-tensor-fn]
> Build a pocketfft `stride_t` (vector of signed byte strides) from the
> tensor's element strides: copy `t.strides()` element-for-element, then
> multiply each by `t.element_size()` to convert element strides to byte
> strides. pocketfft-specific; folded into the op-level contiguous packing
> under the realfft substitution.

> [spec:et:def:fft-utils.torch.executor.native.tensor-cdata-fn]
> inline std::complex<T>* tensor_cdata(Tensor& t)

> [spec:et:sem:fft-utils.torch.executor.native.tensor-cdata-fn]
> Reinterpret a tensor's typed data pointer (`etensor::complex<T>*` /
> `const etensor::complex<T>*`) as a `std::complex<T>*` / `const
> std::complex<T>*` (UB reinterpret_cast, matching PyTorch core) so pocketfft
> can read/write the complex buffer. pocketfft-specific; the realfft
> substitution reads/writes the interleaved re/im layout directly, so the ops
> do not use this helper.

