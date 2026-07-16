# kernels/portable/cpu/util/grid_sampler_2d_util.cpp, kernels/portable/cpu/util/grid_sampler_2d_util.h

> [spec:et:def:grid-sampler-2d-util.torch.executor.check-grid-sampler-2d-args-and-resize-out-fn]
> Error check_grid_sampler_2d_args_and_resize_out( const Tensor& input, const Tensor& grid, Tensor& out)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.check-grid-sampler-2d-args-and-resize-out-fn]
> Validates the arguments of a 2D grid-sampler op and resizes `out` to the
> required output shape. Returns `Error::Ok` on success; on the first failed
> check returns `Error::InvalidArgument` immediately (via
> `ET_CHECK_OR_RETURN_ERROR`, which logs the message and returns the error
> without further checks). `out` is only resized after every argument check
> passes.
>
> Checks, in order (each returns `Error::InvalidArgument` on failure):
> 1. `input.dim() == 4` — input must be 4D, laid out as (N, C, H, W).
> 2. `tensor_is_default_dim_order(input)` — input must be in the default
>    (contiguous NCHW) dim order.
> 3. `grid.dim() == 4` — grid must be 4D, laid out as (N, H_out, W_out, 2).
> 4. `grid.size(3) == 2` — the grid's last dimension must be exactly 2 (the
>    x,y sampling coordinates).
> 5. `input.size(0) == grid.size(0)` — batch sizes N must match.
> 6. `tensors_have_same_dtype(input, grid)` — input and grid must share dtype.
> 7. `tensors_have_same_dtype(input, out)` — input and out must share dtype.
>
> After all checks pass, compute the target output shape as the 4-element
> array `[input.size(0), input.size(1), grid.size(1), grid.size(2)]`, i.e.
> `[N, C, H_out, W_out]`, each cast to the tensor `SizesType`. Call
> `resize_tensor(out, {out_sizes, 4})`; if it does not return `Error::Ok`,
> return `Error::InvalidArgument` ("Failed to resize output tensor").
> Otherwise return `Error::Ok`.
>
> This function only checks dimensionality, dim order, batch match, and dtype
> equality; it does not validate C, H, or W magnitudes, nor the
> interpolation/padding/align_corners parameters (those are handled by the
> caller). No accepted-dtype set is enforced here beyond input==grid==out.

> [spec:et:def:grid-sampler-2d-util.torch.executor.clip-coordinates-fn]
> inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.clip-coordinates-fn]
> Clamps a floating-point coordinate `in` into the closed pixel-index range
> `[0, clip_limit - 1]`, used for `GridSamplerPadding::Border` (and after
> reflection).
>
> Computes `std::min(clip_limit - 1, std::max(in, 0))` with `clip_limit - 1`
> and `0` cast to `scalar_t`. Concretely: the lower clamp `std::max(in, 0)`
> raises values below 0 up to 0, then the upper clamp `std::min(..., clip_limit
> - 1)` lowers values above `clip_limit - 1` down to `clip_limit - 1`. The
> result lies in `[0, clip_limit - 1]`.
>
> NaN handling follows `std::min`/`std::max` C++ semantics: since these return
> their first argument when a comparison is unordered, a NaN `in` propagates
> as follows — `std::max(NaN, 0)` returns `NaN`, then `std::min(clip_limit-1,
> NaN)` returns `clip_limit-1`; so a NaN input clamps to `clip_limit - 1`.
> Note the `clip_limit - 1` subtraction is done in `int64_t` before the cast,
> so `clip_limit == 0` yields an upper bound of `-1`.

> [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn]
> inline scalar_t cubic_convolution1(scalar_t x, scalar_t A)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn]
> Evaluates the cubic convolution kernel used for interpolation sample points
> within 1 unit of the query point (the two nearest neighbors). `A` is the
> cubic spline parameter (bicubic uses `A = -0.75`).
>
> Returns `((A + 2) * x - (A + 3)) * x * x + 1`, evaluated left-to-right in
> `scalar_t` arithmetic. Equivalently: `(A+2)*x^3 - (A+3)*x^2 + 1`. At `x = 0`
> this is `1`; at `x = 1` it is `0`. Used by
> `[spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]`
> for the two central coefficients.

> [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn]
> inline scalar_t cubic_convolution2(scalar_t x, scalar_t A)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn]
> Evaluates the cubic convolution kernel used for interpolation sample points
> between 1 and 2 units from the query point (the two outer neighbors). `A` is
> the cubic spline parameter (bicubic uses `A = -0.75`).
>
> Returns `((A * x - 5 * A) * x + 8 * A) * x - 4 * A`, evaluated left-to-right
> in `scalar_t` arithmetic. Equivalently: `A*x^3 - 5*A*x^2 + 8*A*x - 4*A`. At
> `x = 1` this is `0`; at `x = 2` it is `0`. Used by
> `[spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]`
> for the two outer coefficients.

> [spec:et:def:grid-sampler-2d-util.torch.executor.cubic-interp1d-fn]
> inline scalar_t

> [spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-interp1d-fn]
> Performs 1D cubic interpolation across 4 evenly-spaced sample values
> `x0, x1, x2, x3` at fractional position `t` in `[0, 1]`, where `t` is the
> offset from `x1` toward `x2`.
>
> Algorithm:
> 1. Compute the 4 coefficients into a local array `coeffs[4]` via
>    `[spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]`
>    with argument `t`.
> 2. Return the dot product
>    `x0*coeffs[0] + x1*coeffs[1] + x2*coeffs[2] + x3*coeffs[3]`, summed
>    left-to-right in `scalar_t`.
>
> The coefficients sum to 1 for `t` in `[0,1]`, so this is a partition-of-unity
> interpolation. Used to build 2D bicubic sampling by first interpolating four
> rows then interpolating the resulting column.

> [spec:et:def:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]
> inline void get_cubic_upsample_coefficients(scalar_t coeffs[4], scalar_t t)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.get-cubic-upsample-coefficients-fn]
> Computes the 4 bicubic interpolation coefficients for a fractional position
> `t` in `[0, 1]` and writes them into the caller-provided array `coeffs[4]`.
> Uses the standard bicubic spline parameter `A = -0.75`.
>
> Algorithm:
> 1. `A = -0.75` (as `scalar_t`).
> 2. `x1 = t`.
>    - `coeffs[0] = cubic_convolution2(x1 + 1.0, A)` — outer-left neighbor,
>      per `[spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution2-fn]`.
>    - `coeffs[1] = cubic_convolution1(x1, A)` — near-left neighbor, per
>      `[spec:et:sem:grid-sampler-2d-util.torch.executor.cubic-convolution1-fn]`.
> 3. `x2 = 1.0 - t`.
>    - `coeffs[2] = cubic_convolution1(x2, A)` — near-right neighbor.
>    - `coeffs[3] = cubic_convolution2(x2 + 1.0, A)` — outer-right neighbor.
>
> The distances used are: `coeffs[0]` at distance `t+1`, `coeffs[1]` at
> distance `t`, `coeffs[2]` at distance `1-t`, `coeffs[3]` at distance `2-t`
> from the query point. Returns nothing; results are stored in `coeffs`.

> [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]
> inline scalar_t grid_sampler_compute_source_index( scalar_t coord, int64_t size, GridSamplerPadding padding_mode, bool align_corners)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]
> Converts one normalized grid coordinate `coord` in `[-1, 1]` into a pixel
> source index along an axis of length `size`, applying unnormalization and
> the requested padding mode.
>
> Algorithm:
> 1. Unnormalize: `coord = grid_sampler_unnormalize(coord, size,
>    align_corners)` per
>    `[spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn]`.
> 2. If `padding_mode == GridSamplerPadding::Border`: clamp to image bounds
>    with `coord = clip_coordinates(coord, size)` per
>    `[spec:et:sem:grid-sampler-2d-util.torch.executor.clip-coordinates-fn]`.
> 3. Else if `padding_mode == GridSamplerPadding::Reflection`: reflect the
>    coordinate about the image borders using
>    `[spec:et:sem:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn]`:
>    - if `align_corners`: `coord = reflect_coordinates(coord, 0, 2*(size-1))`.
>    - else: `coord = reflect_coordinates(coord, -1, 2*size - 1)`.
>    Then additionally clamp with `coord = clip_coordinates(coord, size)` to
>    guard against floating-point drift outside `[0, size-1]`.
> 4. For `padding_mode == GridSamplerPadding::Zeros`: no clamping/reflection is
>    applied here; the returned index may be outside `[0, size-1]` and the
>    caller is responsible for treating out-of-bounds samples as zero.
> 5. Return the resulting (possibly fractional) `coord`.

> [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-interpolation]
> enum class GridSamplerInterpolation {
>   Bilinear;
>   Nearest;
>   Bicubic;
> }

> [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-padding]
> enum class GridSamplerPadding {
>   Zeros;
>   Border;
>   Reflection;
> }

> [spec:et:def:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn]
> inline scalar_t

> [spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-unnormalize-fn]
> Maps a normalized coordinate `coord` (nominally in `[-1, 1]`) to a
> continuous pixel index along an axis of length `size`. Each pixel is viewed
> as the area between `(idx - 0.5)` and `(idx + 0.5)`. All arithmetic is in
> `scalar_t`.
>
> - If `align_corners == true`: the extreme normalized values map to the
>   centers of the corner pixels, i.e. `-1 -> 0` and `+1 -> size - 1`. Returns
>   `((coord + 1) / 2) * (size - 1)`.
> - If `align_corners == false`: the extreme normalized values map to the
>   image edges, i.e. `-1 -> -0.5` and `+1 -> size - 0.5`. Returns
>   `((coord + 1) * size - 1) / 2`.
>
> The result is unclamped and may fall outside `[0, size-1]`; padding-mode
> handling is applied by the caller in
> `[spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]`.

> [spec:et:def:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn]
> inline scalar_t

> [spec:et:sem:grid-sampler-2d-util.torch.executor.reflect-coordinates-fn]
> Reflects a coordinate `in` back and forth until it lands within the closed
> interval `[twice_low/2, twice_high/2]`. The bounds are passed pre-doubled
> (`twice_low`, `twice_high`) so that half-integer bounds can be expressed as
> integers. All arithmetic below is in `scalar_t` except where noted.
>
> Algorithm:
> 1. Degenerate span: if `twice_low == twice_high`, return `0`.
> 2. `min = twice_low / 2` (integer `twice_low` cast to `scalar_t`, then
>    halved).
> 3. `span = (twice_high - twice_low) / 2` (subtraction in `int64_t`, then
>    cast and halved) — the half-open reflection period.
> 4. `in = fabs(in - min)` — shift so the low bound is at 0 and fold to the
>    non-negative side.
> 5. `extra = fmod(in, span)` — remainder within one span (non-negative since
>    `in` is non-negative after step 4).
> 6. `flips = floor(in / span)` cast to `int` — how many span-widths were
>    crossed.
> 7. If `flips` is even, return `extra + min`; if odd, return
>    `span - extra + min` (the reflected position).
>
> This implements the standard PyTorch grid-sampler reflection padding. The
> caller in
> `[spec:et:sem:grid-sampler-2d-util.torch.executor.grid-sampler-compute-source-index-fn]`
> follows this with a clip to absorb floating-point rounding.

> [spec:et:def:grid-sampler-2d-util.torch.executor.within-bounds-2d-fn]
> inline bool within_bounds_2d(scalar_t h, scalar_t w, int64_t H, int64_t W)

> [spec:et:sem:grid-sampler-2d-util.torch.executor.within-bounds-2d-fn]
> Tests whether a 2D pixel position `(h, w)` lies inside a grid of height `H`
> and width `W`.
>
> Returns `true` iff `h >= 0 && h < H && w >= 0 && w < W`, i.e. `h` is in the
> half-open range `[0, H)` and `w` is in `[0, W)`. Comparisons mix `scalar_t`
> (`h`, `w`) with `int64_t` (`H`, `W`) under the usual C++ arithmetic
> conversions; when `h`/`w` are floating and non-integral the check is on the
> raw value, so callers typically pass already-integral (floored) coordinates.
> A NaN coordinate makes every comparison false, so NaN yields `false`. Used
> to gate zero-padding reads in the sampler.

