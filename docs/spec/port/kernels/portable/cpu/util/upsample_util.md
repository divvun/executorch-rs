# kernels/portable/cpu/util/upsample_util.cpp, kernels/portable/cpu/util/upsample_util.h

> [spec:et:def:upsample-util.torch.executor.area-pixel-compute-scale-fn]
> inline scalar_t area_pixel_compute_scale( int64_t input_size, int64_t output_size, bool align_corners, const std::optional<double>& scale)

> [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-scale-fn]
> Template on `scalar_t`. Computes the source/destination scale ratio for one
> spatial dimension, following ATen's `area_pixel_compute_scale`.
> If `align_corners`: return `(input_size - 1) / (output_size - 1)` as `scalar_t`
> when `output_size > 1`, else return `0`.
> If not `align_corners`: return
> `compute_scales_value<scalar_t>(scale, input_size, output_size)` per
> `[spec:et:sem:upsample-util.torch.executor.compute-scales-value-fn]` (uses the
> explicit `scale` if provided, else `input_size / output_size`).

> [spec:et:def:upsample-util.torch.executor.area-pixel-compute-source-index-fn]
> inline scalar_t area_pixel_compute_source_index( scalar_t scale, int64_t dst_index, bool align_corners, bool cubic)

> [spec:et:sem:upsample-util.torch.executor.area-pixel-compute-source-index-fn]
> Template on `scalar_t`. Maps a destination index `dst_index` to a (fractional)
> source index using `scale`, following ATen's `area_pixel_compute_source_index`.
> If `align_corners`: return `scale * dst_index`.
> Else: compute `src_idx = scale * (dst_index + 0.5) - 0.5`; then if `!cubic` and
> `src_idx < 0`, return `0` (clamp negative for non-cubic modes), otherwise return
> `src_idx`.

> [spec:et:def:upsample-util.torch.executor.check-upsample-2d-common-args-fn]
> bool check_upsample_2d_common_args( const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t>& output_size, const executorch::aten::OptionalArrayRef<double>& scale_factors, Tensor& out)

> [spec:et:sem:upsample-util.torch.executor.check-upsample-2d-common-args-fn]
> Validates the common arguments shared by 2D upsample ops. Returns `true` if all
> checks pass; on first failure logs and returns `false`.
>
> Checks, in order:
> 1. `tensors_have_same_dtype(in, out)`.
> 2. `tensors_have_same_dim_order(in, out)`.
> 3. `in.dim() == 4` and `out.dim() == 4` (NCHW / NHWC 4D tensors).
> 4. `tensor_is_default_or_channels_last_dim_order(in)` and same for `out` (only
>    contiguous or channels-last layouts allowed).
> 5. Exactly one of `output_size` / `scale_factors` is present:
>    `output_size.has_value() ^ scale_factors.has_value()` must be true.
> 6. If `scale_factors` is present: its size is 2 and both entries `> 0`.
>    Else if `output_size` is present: its size is 2 and both entries `> 0`.

> [spec:et:def:upsample-util.torch.executor.check-upsample-bilinear2d-args-fn]
> bool check_upsample_bilinear2d_args( const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t>& output_size, ET_UNUSED const bool align_corners, const executorch::aten::OptionalArrayRef<double>& scale_factors, Tensor& out)

> [spec:et:sem:upsample-util.torch.executor.check-upsample-bilinear2d-args-fn]
> Validates arguments for `upsample_bilinear2d`. The `align_corners` argument is
> unused. Delegates directly to `check_upsample_2d_common_args(in, output_size,
> scale_factors, out)` per
> `[spec:et:sem:upsample-util.torch.executor.check-upsample-2d-common-args-fn]`,
> returning its result.

> [spec:et:def:upsample-util.torch.executor.check-upsample-nearest2d-args-fn]
> bool check_upsample_nearest2d_args( const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t>& output_size, const executorch::aten::OptionalArrayRef<double>& scale_factors, Tensor& out)

> [spec:et:sem:upsample-util.torch.executor.check-upsample-nearest2d-args-fn]
> Validates arguments for `upsample_nearest2d`. Delegates directly to
> `check_upsample_2d_common_args(in, output_size, scale_factors, out)` per
> `[spec:et:sem:upsample-util.torch.executor.check-upsample-2d-common-args-fn]`,
> returning its result.

> [spec:et:def:upsample-util.torch.executor.compute-scales-value-fn]
> inline scalar_t compute_scales_value( const std::optional<double>& scale, int64_t input_size, int64_t output_size)

> [spec:et:sem:upsample-util.torch.executor.compute-scales-value-fn]
> Template on `scalar_t`. Returns the source-per-destination scale for one
> dimension: if `scale` (a `std::optional<double>`) has a value, return
> `1.0 / scale.value()` as `scalar_t` (an explicit `scale` is the output/input
> factor, so its reciprocal is source-per-destination); otherwise return
> `input_size / output_size` as `scalar_t`.

> [spec:et:def:upsample-util.torch.executor.compute-source-index-and-lambda-fn]
> inline void compute_source_index_and_lambda( int64_t& input_index0, int64_t& input_index1, scalar_t& lambda0, scalar_t& lambda1, opmath_t ratio, int64_t output_index, int64_t input_size, int64_t output_size, bool align_corners)

> [spec:et:sem:upsample-util.torch.executor.compute-source-index-and-lambda-fn]
> Template on `scalar_t` (lambda type) and `opmath_t` (compute type). For a given
> `output_index`, computes the two neighboring source indices and their
> interpolation weights for bilinear/linear upsampling. Outputs via reference:
> `input_index0`, `input_index1`, `lambda0`, `lambda1`.
>
> If `output_size == input_size` (scale factor 1, exact copy): set
> `input_index0 = input_index1 = output_index`, `lambda0 = 1`, `lambda1 = 0`.
> Otherwise:
> 1. `real_input_index = area_pixel_compute_source_index<opmath_t>(ratio, output_index,
>    align_corners, cubic=false)` per
>    `[spec:et:sem:upsample-util.torch.executor.area-pixel-compute-source-index-fn]`.
> 2. `guard_index_and_lambda(real_input_index, input_size, input_index0, lambda1)`
>    per `[spec:et:sem:upsample-util.torch.executor.guard-index-and-lambda-fn]` —
>    fills `input_index0` (the floored, clamped lower index) and `lambda1` (the
>    fractional weight toward the upper neighbor).
> 3. `offset = (input_index0 < input_size - 1) ? 1 : 0`;
>    `input_index1 = input_index0 + offset` (clamp the upper neighbor at the edge).
> 4. `lambda0 = 1 - lambda1` (weight toward the lower neighbor).

> [spec:et:def:upsample-util.torch.executor.guard-index-and-lambda-fn]
> inline void guard_index_and_lambda( const opmath_t& real_input_index, const int64_t& input_size, int64_t& input_index, scalar_t& lambda)

> [spec:et:sem:upsample-util.torch.executor.guard-index-and-lambda-fn]
> Template on `scalar_t` (lambda) and `opmath_t` (compute type). Converts a
> fractional `real_input_index` into an integer lower `input_index` and a
> fractional `lambda`, guarding against overflow when `real_input_index` exceeds
> the range the float type can represent.
> 1. `input_index = min(int64_t(floor(real_input_index)), input_size - 1)` — floor,
>    clamped so it never exceeds the last valid index.
> 2. `lambda = min(max(real_input_index - input_index, 0), 1)` — the fractional part,
>    clamped to `[0, 1]`.
> Both outputs are written through reference parameters.

> [spec:et:def:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]
> inline int64_t nearest_neighbor_compute_source_index( const float scale, int64_t dst_index, int64_t input_size)

> [spec:et:sem:upsample-util.torch.executor.nearest-neighbor-compute-source-index-fn]
> Maps a destination index `dst_index` to a source index for nearest-neighbor
> upsampling, matching OpenCV's (buggy, kept for backward compatibility)
> INTER_NEAREST behavior: return `min(int64_t(floor(dst_index * scale)), input_size - 1)`.
> `scale` is a `float` source-per-destination factor; `floor` of the product is
> clamped to the last valid input index.

> [spec:et:def:upsample-util.torch.executor.resize-upsample-2d-fn]
> Error resize_upsample_2d( const Tensor& in, const executorch::aten::OptionalArrayRef<int64_t>& output_size, const executorch::aten::OptionalArrayRef<double>& scale_factors, double& scale_h_out, double& scale_w_out, Tensor& out)

> [spec:et:sem:upsample-util.torch.executor.resize-upsample-2d-fn]
> Resizes `out` to the target 2D-upsampled shape and reports the effective height
> and width scales (source-per-destination) via `scale_h_out` / `scale_w_out`.
> Assumes exactly one of `output_size` / `scale_factors` is set (enforced by the
> check functions). Returns `Error` (`Ok` on success). `in` is 4D; the spatial
> dims are the last two: `dim = in.dim()`, height at `dim-2`, width at `dim-1`.
>
> Steps:
> 1. Initialize `target_size` as a copy of `in.sizes()`.
> 2. If `scale_factors` present: `scale_h_out = scale_factors[0]`,
>    `scale_w_out = scale_factors[1]`; then
>    `target_size[dim-2] = SizesType(in.size(dim-2) * scale_h_out)` and
>    `target_size[dim-1] = SizesType(in.size(dim-1) * scale_w_out)` (double multiply
>    then truncating cast to integer size).
> 3. Else if `output_size` present: `target_size[dim-2] = output_size[0]`,
>    `target_size[dim-1] = output_size[1]`, and the scales are computed as
>    `scale_h_out = double(output_size[0]) / in.size(dim-2)`,
>    `scale_w_out = double(output_size[1]) / in.size(dim-1)`.
> 4. Else: log "Invalid output_size or scale_factors" and return
>    `Error::InvalidArgument`.
> 5. `ET_CHECK_OR_RETURN_ERROR`: both `target_size[dim-2] > 0` and
>    `target_size[dim-1] > 0` (non-empty output), else `Error::InvalidArgument`.
> 6. Return `resize_tensor(out, {target_size, dim})`.

