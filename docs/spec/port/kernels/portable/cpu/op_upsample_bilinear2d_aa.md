# kernels/portable/cpu/op_upsample_bilinear2d_aa.cpp

> [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn]
> inline T bilinear_aa_filter(T x)

> [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn]
> The triangular (tent) anti-aliasing filter, templated on `T`. Given `x`:
> take `x = abs(x)`; if `x < 1.0` return `1.0 - x`, otherwise return `0.0`
> (all constants and arithmetic in `T`). Matches PyTorch's bilinear AA filter.

> [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.check-upsample-bilinear2d-aa-args-fn]
> bool check_upsample_bilinear2d_aa_args( const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t>& output_size, const bool align_corners, const executorch::aten::OptionalArrayRef<double>& scale_factors, Tensor& out)

> [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.check-upsample-bilinear2d-aa-args-fn]
> Argument-validation predicate for the AA variant. It simply delegates to the
> non-AA bilinear check: returns
> `check_upsample_bilinear2d_args(in, output_size, align_corners,
> scale_factors, out)` unchanged (same dtype/rank-4/matching N,C/dim-order and
> exactly-one-of output_size/scale_factors rules). Note this predicate is not
> actually invoked by the AA entry point, which performs its own inline checks.

> [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn]
> void compute_aa_weights_for_pixel( int64_t output_idx, T scale, int64_t input_size, int64_t* indices, T* weights, int64_t* num_contributors)

> [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn]
> Computes, for one output pixel `output_idx` along a single axis, up to 4
> input contributor indices and their normalized weights. Templated on `T`
> (used as `float` in this op). Inputs: `output_idx`, `scale` (the kernel ratio
> along this axis, i.e. whatever `area_pixel_compute_scale` produced for this
> dimension per
> `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]`; it is
> passed through and used directly, not recomputed), and `input_size`. Outputs
> written through pointers: `indices[4]`, `weights[4]`,
> and `*num_contributors`. Matches PyTorch's AA weight algorithm.
>
> Steps (all arithmetic in `T`):
> 1. `center = scale * (output_idx + 0.5)`. (AA always uses this center,
>    independent of align_corners.)
> 2. `support = (scale >= 1.0) ? scale : 1.0` (bilinear interp_size 2 → base
>    support 1.0, scaled up when downsampling).
> 3. Contributor range:
>    `xmin = max( (int64)(center - support + 0.5), 0 )`,
>    `xmax = min( (int64)(center + support + 0.5), input_size )`.
>    The casts to int64 truncate toward zero.
> 4. `*num_contributors = min(xmax - xmin, 4)`.
> 5. Degenerate case: if `*num_contributors <= 0`, set `*num_contributors = 1`,
>    `indices[0] = clamp((int64)center, 0, input_size - 1)`, `weights[0] = 1.0`,
>    zero `weights[1..3]`, and return.
> 6. Otherwise compute raw weights. Let
>    `invscale = (scale >= 1.0) ? 1.0/scale : 1.0`. For `j` in
>    `[0, *num_contributors)`: `x = xmin + j`;
>    `arg = (j + xmin - center + 0.5) * invscale`;
>    `weight = bilinear_aa_filter(arg)` (see
>    `[spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.bilinear-aa-filter-fn]`);
>    store `indices[j] = x`, `weights[j] = weight`, and accumulate
>    `total_weight += weight`.
> 7. Normalize: if `total_weight > 0`, divide each of `weights[0..num-1]` by
>    `total_weight`; else fall back to equal weights `1.0/num_contributors` for
>    each of the first `*num_contributors` slots.
> 8. Zero the remaining `weights[*num_contributors .. 3]`.
> Returns void; `indices`/`weights` beyond `*num_contributors` (up to 4) are
> zeroed weights (index slots left unspecified).

> [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn]
> void upsample_bilinear2d_aa_kernel_impl( KernelRuntimeContext& ctx, const Tensor& in, bool align_corners, const float scale_h, const float scale_w, Tensor& out)

> [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn]
> Anti-aliased bilinear upsample kernel, templated on element type `CTYPE`.
> `scale_h`/`scale_w` are the precomputed `float` kernel ratios. Both `in` and
> `out` are rank 4 with matching N (dim0) and C (dim1). Determines layout from
> `in`'s dim order: `is_nchw = is_contiguous_dim_order(in.dim_order())`. This
> variant assumes *plain contiguous* geometry within each plane/batch (it
> indexes by computed offsets, not by tensor strides).
>
> Obtain `in_data = in.const_data_ptr<CTYPE>()`,
> `out_data = out.mutable_data_ptr<CTYPE>()`.
>
> For every output pixel it computes height contributors once per output row
> and width contributors once per output column via
> `[spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.compute-aa-weights-for-pixel-fn]`
> (instantiated with `T = float`), yielding `h_indices/h_weights/h_num` and
> `w_indices/w_weights/w_num`. The output value is a separable weighted sum:
> `value = sum over ih in h_contributors, iw in w_contributors of
> in[ih, iw] * h_weight[ih] * w_weight[iw]`, accumulated in `CTYPE`
> (initialized to 0), with each input element multiplied by
> `h_weight * w_weight` per term.
>
> NCHW branch: for `n` in [0,out.size(0)), `c` in [0,out.size(1)): set
> `in_plane = in_data + (n*in.size(1) + c)*in.size(2)*in.size(3)` and
> `out_plane = out_data + (n*out.size(1) + c)*out.size(2)*out.size(3)`. For
> `oh` in [0,out.size(2)) compute the height weights from `(oh, scale_h,
> in.size(2))`; for `ow` in [0,out.size(3)) compute width weights from `(ow,
> scale_w, in.size(3))`; index input as `in_plane[ih*in.size(3) + iw]` and store
> the accumulated `value` at `out_plane[oh*out.size(3) + ow]`.
>
> NHWC branch: for `n` in [0,out.size(0)): set
> `in_batch = in_data + n*in.size(1)*in.size(2)*in.size(3)` and
> `out_batch = out_data + n*out.size(1)*out.size(2)*out.size(3)`. For `oh` then
> `ow` (weights computed as above), loop `c` in [0,out.size(1)): index input as
> `in_batch[(ih*in.size(3) + iw)*in.size(1) + c]` and store `value` at
> `out_batch[(oh*out.size(3) + ow)*out.size(1) + c]`.
> Returns void.

> [spec:et:def:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn]
> Tensor& _upsample_bilinear2d_aa_out( KernelRuntimeContext& ctx, const Tensor& in, const executorch::aten::ArrayRef<int64_t> output_size, bool align_corners, const std::optional<double> scale_h, const std::optional<double> scale_w, Tensor...

> [spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-out-fn]
> Entry point for `_upsample_bilinear2d_aa.out`. Arguments: input `in`,
> `output_size` (an `ArrayRef<int64_t>` of exactly the target H,W),
> `align_corners`, optional per-axis `scale_h`/`scale_w`, and preallocated
> `out`. Returns `out`.
>
> Inline validation (each `ET_KERNEL_CHECK` on failure sets InvalidArgument on
> `ctx` and returns `out` unchanged; it does NOT resize `out` — the caller must
> supply a correctly sized output):
> 1. `in.dim() == 4`.
> 2. `out.dim() == 4`.
> 3. `in.scalar_type() == out.scalar_type()`.
> 4. `output_size.size() == 2`.
> 5. `output_size[0] > 0 && output_size[1] > 0`.
> 6. `out.size(0) == in.size(0)` (batch).
> 7. `out.size(1) == in.size(1)` (channels).
> 8. `out.size(2) == output_size[0]` (height).
> 9. `out.size(3) == output_size[1]` (width).
>
> Scale resolution: if both `scale_h` and `scale_w` have values, use
> `final_scale_h = scale_h.value()`, `final_scale_w = scale_w.value()`;
> otherwise compute `final_scale_h = output_size[0] / in.size(2)` and
> `final_scale_w = output_size[1] / in.size(3)` as doubles. (Note: the branch
> is all-or-nothing — it only uses provided scales when both are present.)
>
> Kernel ratios: `kernel_scale_h = area_pixel_compute_scale<double>(in.sizes()[2],
> out.sizes()[2], align_corners, final_scale_h)` and likewise `kernel_scale_w`
> from dim 3, per
> `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]`.
>
> Dispatch on `in.scalar_type()` with `ET_SWITCH_REALHBF16_TYPES` (Byte, Char,
> Short, Int, Long, Float, Double, Half, BFloat16 — real numeric plus Half and
> BFloat16, excluding Bool and complex). For the selected `CTYPE` call
> `[spec:et:sem:op-upsample-bilinear2d-aa.torch.executor.native.upsample-bilinear2d-aa-kernel-impl-fn]`
> with `(ctx, in, align_corners, kernel_scale_h, kernel_scale_w, out)`.
> Return `out`.

