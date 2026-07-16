# kernels/portable/cpu/op_convolution_backward.cpp

> [spec:et:def:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn]
> bool check_convolution_backward_args( const Tensor& grad_output, const Tensor& input, const Tensor& weight, ET_UNUSED const OptIntArrayRef bias_sizes_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, In...

> [spec:et:sem:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn]
> Validates the argument set for `convolution_backward.out`. Returns `true` if all
> checks pass; on the first failing check it logs and returns `false` (the caller
> turns a `false` into `Error::InvalidArgument`). Checks are performed in this order:
> 1. `transposed == false`; otherwise fail ("Transposed Convolution Backward not supported yet").
> 2. `weight.dim() == 4`; otherwise fail (only 2D convolution backward is supported).
> 3. `weight` and `input` have the same dtype.
> 4. `grad_output` and `input` have the same dtype.
> 5. If `output_mask[0]` (grad_input requested): `grad_input` and `input` have the same dtype.
> 6. If `output_mask[1]` (grad_weight requested): `grad_weight` and `input` have the same dtype.
> 7. If `output_mask[2]` (grad_bias requested): `grad_bias` and `input` have the same dtype.
> 8. The forward convolution arguments are valid per the shared
>    `check_convolution_args(input, weight, /*bias=*/none, stride, padding, dilation,
>    transposed, output_padding, groups, grad_output)` check (which treats `grad_output`
>    as the forward "out" tensor).
> 9. Compute the forward convolution output shape into `output_sizes`/`output_ndim` via the
>    shared `get_convolution_out_target_size(...)`, then assert
>    `output_size_is_valid({output_sizes, output_ndim}, input.dim() - 2)` (spatial output dims valid).
> 10. `grad_output.dim() == input.dim()`.
> 11. `grad_output` has exactly shape `{output_sizes, output_ndim}` from step 9 (`tensor_has_expected_size`).
>
> `bias_sizes_opt` is unused by this function.

> [spec:et:def:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn]
> void conv2d_backward_impl( const Tensor& grad_output, const Tensor& input, const Tensor& weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups, executorch::aten::ArrayRef<bool> output_mask, Tensor& grad_in...

> [spec:et:sem:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn]
> Templated on `CTYPE` (element type). Computes the three gradients of a 2D non-transposed
> convolution directly, writing into whichever of `grad_input`, `grad_weight`, `grad_bias`
> are selected by `output_mask`. No validation here; the caller has already validated/resized.
>
> Shape variables (NCHW layout, weight is `[out_channels, in_channels/groups, kH, kW]`):
> `batch_size=input.size(0)`, `in_channels=input.size(1)`, `out_channels=weight.size(0)`,
> `in_height=input.size(2)`, `in_width=input.size(3)`, `out_height=grad_output.size(2)`,
> `out_width=grad_output.size(3)`, `kernel_height=weight.size(2)`, `kernel_width=weight.size(3)`.
>
> Hyperparameters via `val_at`: `stride_h=stride[0]`, `stride_w=stride[1]`,
> `padding_h=padding[0]` (default 0 if absent), `padding_w=padding[1]` (default 0),
> `dilation_h=dilation[0]`, `dilation_w=dilation[1]`. `val_at(arr,i)` returns `arr[i]` when
> `arr` has more than one element else `arr[0]` (with the given default when empty). Grouping:
> `in_channels_per_group=in_channels/groups`, `out_channels_per_group=out_channels/groups`.
>
> For each requested output selected by `output_mask`, obtain its mutable data pointer and
> zero-fill its entire buffer with memset (over `nbytes()`) before accumulating.
>
> Iterate, using each tensor's actual `strides()` to compute linear indices via
> `calculate_linear_index(coord, strides, 4)`, building 4-element coordinate arrays
> `out_coord`/`in_coord`/`weight_coord`:
> - for each batch `b` in `[0,batch_size)`: `in_coord[0]=b`, `out_coord[0]=b`.
>   - for each group `g` in `[0,groups)`:
>     - for each output row `h` in `[0,out_height)`: `out_coord[2]=h`.
>       - for each output col `w` in `[0,out_width)`: `out_coord[3]=w`.
>         - for each in-group output channel `oc` in `[0,out_channels_per_group)`:
>           `oc_global=oc+g*out_channels_per_group`; `weight_coord[0]=oc_global`,
>           `out_coord[1]=oc_global`; `out_idx=calculate_linear_index(out_coord, grad_output.strides(), 4)`.
>           - If `output_mask[2]`: `grad_bias[oc_global] += grad_output[out_idx]`.
>           - for each in-group input channel `ic` in `[0,in_channels_per_group)`:
>             `ic_global=ic+g*in_channels_per_group`; `in_coord[1]=ic_global`, `weight_coord[1]=ic`.
>             - for each kernel row `kh` in `[0,kernel_height)`:
>               `in_h=h*stride_h - padding_h + kh*dilation_h`; skip unless `0<=in_h<in_height`;
>               `in_coord[2]=in_h`, `weight_coord[2]=kh`.
>               - for each kernel col `kw` in `[0,kernel_width)`:
>                 `in_w=w*stride_w - padding_w + kw*dilation_w`; skip unless `0<=in_w<in_width`;
>                 `in_coord[3]=in_w`, `weight_coord[3]=kw`;
>                 `in_idx=calculate_linear_index(in_coord, input.strides(), 4)`,
>                 `weight_idx=calculate_linear_index(weight_coord, weight.strides(), 4)`.
>                 - If `output_mask[0]`: `grad_input[in_idx] += grad_output[out_idx]*weight[weight_idx]`.
>                 - If `output_mask[1]`: `grad_weight[weight_idx] += grad_output[out_idx]*input[in_idx]`.
>
> All accumulation is done in `CTYPE`. Padding positions falling outside the input are skipped
> (they contribute nothing to grad_input or grad_weight).

> [spec:et:def:op-convolution-backward.torch.executor.native.convolution-backward-out-fn]
> std::tuple<Tensor&, Tensor&, Tensor&> convolution_backward_out( KernelRuntimeContext& ctx, const Tensor& grad_output, const Tensor& input, const Tensor& weight, const OptIntArrayRef bias_sizes_opt, IntArrayRef stride, IntArrayRef padding...

> [spec:et:sem:op-convolution-backward.torch.executor.native.convolution-backward-out-fn]
> Entry point for `convolution_backward.out`. Builds return value
> `ret_val = (grad_input, grad_weight, grad_bias)` (references to the three out tensors);
> every failure path returns this tuple with the out tensors unchanged.
>
> Steps:
> 1. ET_KERNEL_CHECK: run
>    `[spec:et:sem:op-convolution-backward.torch.executor.native.check-convolution-backward-args-fn]`;
>    on failure set `Error::InvalidArgument` on `ctx` and return `ret_val`.
> 2. If `output_mask[0]`: resize `grad_input` to `input.sizes()` (ET_KERNEL_CHECK on
>    `resize_tensor == Error::Ok`, else InvalidArgument, return `ret_val`).
> 3. If `output_mask[1]`: resize `grad_weight` to `weight.sizes()` similarly.
> 4. If `bias_sizes_opt.has_value() && output_mask[2]`: resize `grad_bias` to
>    `bias_sizes_opt.value()` similarly. (grad_bias is not resized when `bias_sizes_opt` is
>    absent, even if `output_mask[2]` is set.)
> 5. Dispatch over `input.scalar_type()` restricted to the FLOATHBF16 set
>    (`{Half, Float, Double, BFloat16}`) via `ET_SWITCH_FLOATHBF16_TYPES`; an unsupported
>    dtype sets an error via `ctx` and returns. For the selected `CTYPE`, call
>    `conv2d_backward_impl<CTYPE>(grad_output, input, weight, stride, padding, dilation,
>    groups, output_mask, grad_input, grad_weight, grad_bias)` per
>    `[spec:et:sem:op-convolution-backward.torch.executor.native.conv2d-backward-impl-fn]`.
> 6. Return `ret_val`.

