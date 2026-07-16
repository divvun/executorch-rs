# kernels/portable/cpu/op_repeat.cpp

> [spec:et:def:op-repeat.torch.executor.native.calculate-output-size-fn]
> bool calculate_output_size( const executorch::aten::ArrayRef<executorch::aten::SizesType>& self_sizes, const executorch::aten::ArrayRef<int64_t>& repeats, Tensor::SizesType* out_sizes_ptr)

> [spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn]
> Computes the output shape for `repeat`, writing `repeats.size()` entries into
> `out_sizes_ptr`. Returns `bool` (validation). Steps:
>
> - ET_LOG_AND_RETURN_IF_FALSE: `repeats.size() < kTensorDimensionLimit`; if not,
>   log and return false.
> - ET_CHECK_OR_RETURN_FALSE: `repeats.size() >= self_sizes.size()` (message
>   "Repeats vector size is %zu must be >= self_sizes %zu."); else return false.
> - Let `d = repeats.size() - self_sizes.size()` be the number of leading
>   dimensions with no corresponding self dim. For `i` in `[0, d)`: `out_sizes_ptr[i]
>   = repeats[i]` (cast to SizesType).
> - Then for the remaining positions, walking `j` from 0 over `self_sizes`: for
>   `i` in `[d, repeats.size())`, `out_sizes_ptr[i] = repeats[i] * self_sizes[j]`
>   (cast to SizesType), incrementing `j` each step. So each output dim aligned
>   with a self dim is `repeats[i] * self_sizes[j]`; leading extra dims are just
>   `repeats[i]`.
> - Returns true on success.

> [spec:et:def:op-repeat.torch.executor.native.repeat-out-fn]
> Tensor& repeat_out( KernelRuntimeContext& ctx, const Tensor& self, executorch::aten::ArrayRef<int64_t> repeats, Tensor& out)

> [spec:et:sem:op-repeat.torch.executor.native.repeat-out-fn]
> Tiles `self` according to `repeats` into `out`. Steps:
>
> - Compute `expected_output_size` into a stack buffer via
>   `calculate_output_size(self.sizes(), repeats, ...)` (see
>   `[spec:et:sem:op-repeat.torch.executor.native.calculate-output-size-fn]`).
>   ET_KERNEL_CHECK on its bool result; on failure `Error::InvalidArgument`,
>   return `out`.
> - ET_KERNEL_CHECK: `self`/`out` same dim order; else `Error::InvalidArgument`,
>   return `out`.
> - ET_KERNEL_CHECK: `self` is in default (contiguous) dim order
>   (`tensor_is_default_dim_order`); else `Error::InvalidArgument`, return `out`.
> - Resize `out` to `{expected_output_size, repeats.size()}` (so `out.dim() ==
>   repeats.size()`); on failure `Error::InvalidArgument` (message "Failed to
>   resize output tensor."), return `out`.
> - ET_KERNEL_CHECK: `repeat_tensor(self, repeats, out) == Error::Ok` (see
>   `[spec:et:sem:repeat-util.torch.executor.repeat-tensor-fn]`, which performs
>   the actual dtype-agnostic byte-copy tiling of `self` across the repeated
>   dimensions); on failure `Error::InvalidArgument`, return `out`.
> - Returns `out`.

