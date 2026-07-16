# kernels/portable/cpu/op__adaptive_avg_pool2d.cpp

> [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn]
> inline int64_t

> [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn]
> Computes the exclusive end index into an input dimension for output index
> `out_idx`, given the output dimension length `out_size` and input dimension
> length `in_size`. Returns `floor((out_idx + 1) * in_size / out_size)`,
> computed as: form the integer product `(out_idx + 1) * in_size` in int64,
> cast it to 32-bit `float`, divide by `out_size` (also cast to float) in
> float arithmetic, apply `std::ceil`, then cast the result back to int64.
> Because the division uses single-precision `float`, this is the exact ceil
> only up to float rounding for very large operands. Together with the start
> index the half-open range `[start(out_idx), end(out_idx))` covers the input
> elements that are averaged into output element `out_idx`.

> [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn]
> inline int64_t

> [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn]
> Computes the inclusive start index into an input dimension for output index
> `out_idx`, given the output dimension length `out_size` and input dimension
> length `in_size`. Returns `floor(out_idx * in_size / out_size)`, computed as:
> form the integer product `out_idx * in_size` in int64, cast it to 32-bit
> `float`, divide by `out_size` (also cast to float) in float arithmetic, apply
> `std::floor`, then cast the result back to int64. The single-precision `float`
> division means this is the exact floor only up to float rounding for very
> large operands.

> [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn]
> Tensor& _adaptive_avg_pool2d_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef output_size, Tensor& out)

> [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn]
> Implements `_adaptive_avg_pool2d.out(in, output_size, out)`: adaptive 2D
> average pooling of `in` (shape `[C, H, W]` or `[N, C, H, W]`) to spatial size
> `output_size = [out_H, out_W]`, writing into `out`.
>
> Validation, in order (each is `ET_KERNEL_CHECK`; on failure the context Error
> is set to `InvalidArgument` and `out` is returned unmodified):
> 1. `check_adaptive_avg_pool2d_args(in, output_size, out)` — validates rank,
>    channel/batch agreement, dtype compatibility and `output_size` shape per
>    that util's contract.
> 2. `tensors_have_same_dim_order(in, out)` — `in` and `out` share the same
>    dim order.
> 3. `tensor_is_default_dim_order(in)` — `in` is in default (contiguous) dim
>    order.
>
> Then computes the target output sizes via
> `get_adaptive_avg_pool2d_out_target_size(in, output_size, ...)` (input batch
> and channel dims preserved, last two dims replaced by `out_H`, `out_W`),
> checks `output_size_is_valid({sizes, ndim}, 2)` (`ET_KERNEL_CHECK` →
> `InvalidArgument`), then `resize_tensor(out, target_sizes)` must return
> `Error::Ok` (`ET_KERNEL_CHECK` → `InvalidArgument`).
>
> Dtype dispatch: switches on `in.scalar_type()` over
> `ET_SWITCH_FLOATHBF16_TYPES_AND(Long, ...)`, i.e. the accepted CTYPE set is
> {Half, Float, Double, BFloat16, Long}; any other input dtype fails the switch
> (kernel error). `out` uses the same CTYPE as `in`.
>
> Algorithm (all pointer arithmetic in the selected CTYPE):
> - `ndim = in.dim()`; `in_H = in.size(ndim-2)`, `in_W = in.size(ndim-1)`,
>   `out_H = output_size[0]`, `out_W = output_size[1]`,
>   `channels = in.size(ndim-3)`, `batch_size = (ndim == 4) ? in.size(0) : 1`.
> - `in_plane_size = in_H * in_W`, `out_plane_size = out_H * out_W`.
> - For each batch `b` in `[0, batch_size)`, for each channel `c` in
>   `[0, channels)`: `plane_idx = b * channels + c`; the input plane starts at
>   `in_ptr + plane_idx * in_plane_size` and the output plane at
>   `out_ptr + plane_idx * out_plane_size` (contiguous NCHW layout).
> - For each output row `oh` in `[0, out_H)`: `ih0 = adaptive_start_index(oh,
>   out_H, in_H)` per `[spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn]`,
>   `ih1 = adaptive_end_index(oh, out_H, in_H)` per
>   `[spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn]`.
>   For each output col `ow` in `[0, out_W)`: `iw0`, `iw1` computed the same way
>   over the W dimension.
> - Accumulate `sum` in `float` (single precision, regardless of CTYPE) over
>   the input window: for `ih` in `[ih0, ih1)`, `iw` in `[iw0, iw1)`, add
>   `plane_in[ih * in_W + iw]` to `sum`.
> - `count = (ih1 - ih0) * (iw1 - iw0)`; write `plane_out[oh * out_W + ow] =
>   static_cast<CTYPE>(sum / static_cast<float>(count))`.
>
> Windows always non-empty (count >= 1) for valid output sizes. Returns `out`.

