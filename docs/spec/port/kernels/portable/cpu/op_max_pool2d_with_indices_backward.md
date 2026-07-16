# kernels/portable/cpu/op_max_pool2d_with_indices_backward.cpp

> [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn]
> bool check_max_pool2d_backward_args( const Tensor& grad_output, const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices, const Tensor& grad_input)

> [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn]
> Argument validator for the max_pool2d backward pass; returns `true` if all
> checks pass, else logs and returns `false` (each `ET_LOG_AND_RETURN_IF_FALSE` /
> `ET_CHECK_OR_RETURN_FALSE` returns `false` on its first failure).
>
> Checks, in order:
> 1. `grad_output` and `input` have the same dtype.
> 2. `grad_input` and `input` have the same dtype.
> 3. `check_max_pool2d_with_indices_args(input, kernel_size, stride, padding,
>    dilation, ceil_mode, grad_output, indices)` passes — i.e. the same forward
>    argument validation, using `grad_output` in the `out` slot and `indices` as
>    the indices tensor (message "Invalid max_pool_2d arguments" on failure).
> 4. Compute the forward output size for `input` into
>    `output_sizes`/`output_ndim` via
>    `get_max_pool2d_with_indices_out_target_size` and check
>    `output_size_is_valid({output_sizes, output_ndim}, 2)`.
> 5. `grad_output.dim() == input.dim()` (else logs the two dims).
> 6. `grad_output` has exactly the expected size `{output_sizes, output_ndim}`
>    (`tensor_has_expected_size`).
> Returns `true` only if all pass.

> [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn]
> void max_pool_backward_impl( const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices)

> [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn]
> Templated core of the backward pass (`is_3d` is a compile-time flag; for
> max_pool2d it is `false`). Scatters `grad_output` back into `grad_input` at the
> argmax positions recorded in `indices`, accumulating. The caller must have
> zero-initialized `grad_input` (this function only adds).
>
> Layout: `grad_output.dim()` is `ndim`. Treat batch and channels as one flat
> dimension `channels`: for 2-D pooling (`is_3d == false`), `channels =
> grad_output.size(0)` when `ndim == 3` (CHW), else `grad_output.size(0) *
> grad_output.size(1)` (NCHW). `input_depth = 1` (2-D). Spatial extents:
> `input_height = grad_input.size(ndim-2)`, `input_width = grad_input.size(ndim-1)`,
> `output_depth = 1`, `output_height = grad_output.size(ndim-2)`,
> `output_width = grad_output.size(ndim-1)`.
>
> Algorithm: for each channel `c` in `[0, channels)`, take the sub-slices
> `grad_input_ptr = grad_input_data + c*input_depth*input_height*input_width`,
> `grad_output_ptr` and `indices_ptr` = the corresponding output-sized slices at
> `c*output_depth*output_height*output_width`. For each output position
> `(od, oh, ow)` (od over `output_depth`, oh over `output_height`, ow over
> `output_width`), let `index = od*output_height*output_width + oh*output_width
> + ow`, and `maxindex = indices_ptr[index]`. If `maxindex != -1`, do
> `grad_input_ptr[maxindex] += grad_output_ptr[index]` (accumulate; multiple
> output cells whose argmax collided add together). `maxindex == -1` (no valid
> max, e.g. an all-padding window) contributes nothing.
>
> `grad_output`, `indices` (int64), and `grad_input` share element type CTYPE for
> the gradient tensors; `indices` is always int64. No return value (writes into
> `grad_input`).

> [spec:et:def:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn]
> Tensor& max_pool2d_with_indices_backward_out( KernelRuntimeContext& ctx, const Tensor& grad_output, const Tensor& input, ET_UNUSED IntArrayRef kernel_size, ET_UNUSED IntArrayRef stride, ET_UNUSED IntArrayRef padding, ET_UNUSED IntArrayRe...

> [spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool2d-with-indices-backward-out-fn]
> Backward (gradient) of max_pool2d_with_indices: given `grad_output` and the
> forward `indices`, produces `grad_input` (gradient w.r.t. `input`) by routing
> each output gradient to the input element that was the max. `kernel_size`,
> `stride`, `padding`, `dilation`, `ceil_mode` are only used for validation (the
> actual scatter is index-driven).
>
> Steps:
> 1. Validate via `check_max_pool2d_backward_args(grad_output, input,
>    kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input)`
>    (see
>    `[spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.check-max-pool2d-backward-args-fn]`)
>    (ET_KERNEL_CHECK: on failure sets Error::InvalidArgument and returns
>    `grad_input` unchanged).
> 2. Resize `grad_input` to `input.sizes()` (ET_KERNEL_CHECK: InvalidArgument,
>    returns `grad_input`).
> 3. Dispatch on `input.scalar_type()` over FLOATHBF16 {Half, Float, Double,
>    BFloat16} (CTYPE) and call
>    `[spec:et:sem:op-max-pool2d-with-indices-backward.torch.executor.native.max-pool-backward-impl-fn]`
>    with `is_3d = false`, which scatters `grad_output` into `grad_input` at the
>    argmax positions from `indices`.
> 4. Return `grad_input`.
>
> Note: the resize in step 2 does not zero the buffer, and the impl only
> accumulates; `grad_input` is expected to be logically zero-initialized before
> the scatter (positions never selected as a max remain at their prior value,
> which for a correctly allocated gradient output is zero).

