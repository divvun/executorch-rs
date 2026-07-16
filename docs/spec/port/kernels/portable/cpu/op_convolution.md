# kernels/portable/cpu/op_convolution.cpp

> [spec:et:def:op-convolution.torch.executor.native.conv2d-impl-fn]
> void conv2d_impl( const CTYPE* const in_ptr, SizesArrayRef in_sizes, StridesArrayRef in_strides, const CTYPE* const w_ptr, SizesArrayRef w_sizes, StridesArrayRef w_strides, const std::optional<Tensor>& bias, const char* const bias_ptr, L...

> [spec:et:sem:op-convolution.torch.executor.native.conv2d-impl-fn]
> Computes the 2D (or transposed-2D) convolution contribution for a single
> `(batch, group, out_c)` output channel. Tensors are 4D (N, C, H, W); indices
> into them are computed with explicit strides via `calculate_linear_index(coord,
> strides, 4)`. `load_bias` reads a bias element from `bias_ptr` in the bias
> tensor's own dtype and converts to CTYPE.
>
> Setup: `in_C = in_sizes[1]`, `out_C = out_sizes[1]`, spatial extents
> `out_H/in_H/w_H = *_sizes[2]`, `out_W/in_W/w_W = *_sizes[3]`. Per-group input
> channels `in_C_per_group = in_C/groups`, `in_c_start = group*in_C_per_group`;
> similarly `out_C_per_group`, `out_c_start`. `in_coord[0]=batch`; for the
> non-transposed path `out_coord = {batch, out_c, out_y, out_x}` and
> `w_coord[0]=out_c`. Read stride/padding/dilation as
> `stride_y=val_at(stride,0)`, `padding_y=val_at(padding,0,0)`,
> `dilation_y=val_at(dilation,0)`, and the x variants at index 1.
>
> Non-transposed path (`transposed == false`): for each `out_y` in [0,out_H) and
> `out_x` in [0,out_W): initialize `accum = 0`. For each input channel `in_c` in
> [in_c_start, in_c_start+in_C_per_group) with `w_coord[1] = in_c - in_c_start`:
> for each `w_y` in [0,w_H): `in_y = stride_y*out_y + dilation_y*w_y - padding_y`;
> if `0 <= in_y < in_H`, for each `w_x` in [0,w_W):
> `in_x = stride_x*out_x + dilation_x*w_x - padding_x`; if `0 <= in_x < in_W`,
> `accum += in_ptr[in_idx] * w_ptr[w_idx]` (inputs outside bounds contribute
> nothing, i.e. implicit zero padding). After the channel loops, if
> `bias_ptr != nullptr` add `load_bias(&bias_ptr[out_c * bias.element_size()])`,
> then store `out_ptr[out_idx] = accum` (assignment — this channel is written
> once).
>
> Transposed path (`transposed == true`): `w_coord[1] = out_c - out_c_start`.
> Scatter from input to output: for each `in_y` in [0,in_H) and `in_x` in
> [0,in_W): for each `in_c` in the group (with `w_coord[0]=in_c`), load
> `in_val = in_ptr[in_idx]`; for each `w_y`: `out_y = stride_y*in_y +
> dilation_y*w_y - padding_y`; if in bounds, for each `w_x`:
> `out_x = stride_x*in_x + dilation_x*w_x - padding_x`; if in bounds,
> `out_ptr[out_idx] += in_val * w_ptr[w_idx]` (accumulate; the caller
> pre-initializes `out` with bias or zero before scattering).

> [spec:et:def:op-convolution.torch.executor.native.conv3d-impl-fn]
> void conv3d_impl( const CTYPE* const in_ptr, SizesArrayRef in_sizes, StridesArrayRef in_strides, const CTYPE* const w_ptr, SizesArrayRef w_sizes, StridesArrayRef w_strides, const std::optional<Tensor>& bias, const char* const bias_ptr, L...

> [spec:et:sem:op-convolution.torch.executor.native.conv3d-impl-fn]
> Computes the forward 3D convolution contribution for a single
> `(batch, group, out_c)` output channel. Tensors are 5D (N, C, D, H, W); indices
> are computed via `calculate_linear_index(coord, strides, 5)`. This impl handles
> only the non-transposed case.
>
> Setup: `in_C = in_sizes[1]`; depth extents `out_D/in_D/w_D = *_sizes[2]`,
> height `*_sizes[3]`, width `*_sizes[4]`. `in_C_per_group = in_C/groups`,
> `in_c_start = group*in_C_per_group`. `in_coord[0]=batch`,
> `out_coord = {batch, out_c, out_z, out_y, out_x}`, `w_coord[0]=out_c`.
> stride/padding/dilation are read for z at index 0, y at index 1, x at index 2
> (`val_at`, padding default 0).
>
> Loop: for each `out_z` in [0,out_D), `out_y` in [0,out_H), `out_x` in [0,out_W):
> initialize `accum = 0`. For each `in_c` in [in_c_start, in_c_start+in_C_per_group)
> with `w_coord[1] = in_c - in_c_start`: for each `w_z`:
> `in_z = stride_z*out_z + dilation_z*w_z - padding_z`; skip (continue) if
> `in_z < 0 || in_z >= in_D`. For each `w_y`:
> `in_y = stride_y*out_y + dilation_y*w_y - padding_y`; skip if out of [0,in_H).
> For each `w_x`: `in_x = stride_x*out_x + dilation_x*w_x - padding_x`; if
> `0 <= in_x < in_W`, `accum += in_ptr[in_idx] * w_ptr[w_idx]`. Out-of-bounds
> input positions contribute nothing (implicit zero padding). After the loops, if
> `bias_ptr != nullptr` add `load_bias(&bias_ptr[out_c * bias.element_size()])`,
> then store `out_ptr[out_idx] = accum`.

> [spec:et:def:op-convolution.torch.executor.native.convolution-wrapper-fn]
> void convolution_wrapper( const Tensor& in, const Tensor& weight, const std::optional<Tensor>& bias, LoadFn load_bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, int64_t groups, Tensor& out)

> [spec:et:sem:op-convolution.torch.executor.native.convolution-wrapper-fn]
> Prepares tensor geometry and dispatches per (batch, group, out_channel) to the
> 2D or 3D impl. Handles the 1D-as-2D case and strides from dim orders.
>
> Steps:
> 1. Capture `in_sizes`, `weight_sizes`, `out_sizes` and their dim orders, and
>    local copies of stride/padding/dilation.
> 2. 1D handling: if `in.dim() == 3`, unsqueeze a height dim of size 1 for input,
>    weight, and output (`get_unsqueezed_sizes(...,2,...)` and
>    `get_unsqueezed_dim_order`), producing 4D shapes. Adjust the params so the
>    inserted height dim is trivial: `stride = {1, stride[0]}`,
>    `padding = {0, padding[0]}`, `dilation = {1, dilation.size()>0 ? dilation[0]
>    : 1}`. This lets the 2D impl handle 1D convolution.
> 3. Compute contiguous element-strides for input, weight, and output from their
>    (possibly unsqueezed) sizes and dim orders via
>    `dim_order_to_stride_nocheck`.
> 4. Obtain typed pointers: `out_ptr` (mutable CTYPE), `in_ptr`/`w_ptr` (const
>    CTYPE), and `bias_ptr` (raw `const char*` into the bias tensor, or nullptr
>    if no bias).
> 5. `out_N = out.size(0)`, `out_C = out.size(1)`, `out_C_per_group = out_C/groups`,
>    `is_conv3d = (in_sizes.size() == 5)`.
> 6. If `transposed`: pre-initialize the whole output before scattering. If there
>    is no bias, `memset(out_ptr, 0, out.nbytes())`. If there is bias, set each
>    output element to its channel's bias: for out_ix in [0, out.numel()),
>    `out_ptr[out_ix] = load_bias(&bias_ptr[((out_ix / out_strides[1]) % out_C) *
>    bias.element_size()])`. (Non-transposed impls assign, so no pre-init is
>    needed there.)
> 7. For each `batch` in [0, out_N), each `group` in [0, groups), and each output
>    channel `out_c` in [group*out_C_per_group, (group+1)*out_C_per_group): call
>    `conv3d_impl` (5D strides) if `is_conv3d`, else `conv2d_impl` (4D strides,
>    passing `transposed`), forwarding the prepared sizes/strides/params, bias,
>    and `load_bias`. See
>    `[spec:et:sem:op-convolution.torch.executor.native.conv2d-impl-fn]` and
>    `[spec:et:sem:op-convolution.torch.executor.native.conv3d-impl-fn]`.

> [spec:et:def:op-convolution.torch.executor.native.convolution-out-fn]
> Tensor& convolution_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& weight, const std::optional<Tensor>& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, in...

> [spec:et:sem:op-convolution.torch.executor.native.convolution-out-fn]
> Implements the general `convolution.out` (covers 1D/2D/3D forward and
> transposed convolution). The context is used for checks.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_convolution_args(in, weight, bias, stride, padding,
>    dilation, transposed, output_padding, groups, out)` (rank/channel/group
>    divisibility, param list lengths, dtype compatibility, non-negative
>    padding/stride, etc.). On failure set Error::InvalidArgument on the context
>    and return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. Compute `output_sizes` via `get_convolution_out_target_size(in, weight,
>    stride, padding, dilation, transposed, output_padding, groups, ...)`
>    (standard conv output-size formula; for transposed convolution it uses the
>    transposed formula including `output_padding`).
> 4. ET_KERNEL_CHECK: `output_size_is_valid({output_sizes, output_ndim},
>    in.dim() - 2)` (spatial dims valid); on failure set Error::InvalidArgument
>    and return `out` unchanged.
> 5. Resize `out` to `{output_sizes, output_ndim}`; if resize fails set
>    Error::InvalidArgument and return `out` unchanged.
> 6. If `out.numel() == 0`, return `out` immediately (nothing to compute).
> 7. Dispatch on `in.scalar_type()` over REALHBF16 = {Byte, Char, Short, Int,
>    Long, Half, Float, Double, BFloat16}; other dtypes set Error::InvalidArgument
>    and return `out` unchanged. Build `load_bias`: if `bias` is present, a
>    dtype-aware loader that reads a bias element (allowed dtypes REALHBF16) and
>    converts to CTYPE (`utils::internal::get_load_to_compute_fn`); otherwise
>    nullptr.
> 8. Call `convolution_wrapper<CTYPE>(in, weight, bias, load_bias, stride,
>    padding, dilation, transposed, groups, out)` per
>    `[spec:et:sem:op-convolution.torch.executor.native.convolution-wrapper-fn]`,
>    which runs the actual convolution.
> 9. Return `out`.

