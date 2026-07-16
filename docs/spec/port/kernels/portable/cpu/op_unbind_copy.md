# kernels/portable/cpu/op_unbind_copy.cpp

> [spec:et:def:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn]
> void unbind_copy_int_out( KernelRuntimeContext& ctx, const Tensor& input, int64_t dim, TensorList out)

> [spec:et:sem:op-unbind-copy.torch.executor.native.unbind-copy-int-out-fn]
> Removes dimension `dim` from `input`, writing each slice along `dim` into the
> corresponding tensor in `out`. Implements `unbind_copy.int_out(Tensor input,
> int dim=0, *, Tensor(a!)[] out)`. Returns void. Step by step:
>
> - Normalize `dim`: if `dim < 0`, `dim += input.dim()`.
> - ET_KERNEL_CHECK `check_unbind_copy_args(input, dim, out)` (see
>   `[spec:et:sem:copy-ops-util...check-unbind-copy-args-fn]`): validates `dim`
>   valid, `out.size() == input.size(dim)`, and each out tensor has the expected
>   (input shape minus `dim`) shape and matching dtype. On failure sets
>   Error::InvalidArgument and returns.
> - For each `i` in `[0, out.size())` ET_KERNEL_CHECK
>   `tensors_have_same_dim_order(input, out[i])`; else Error::InvalidArgument.
> - ET_KERNEL_CHECK `tensor_is_default_dim_order(input)`; else Error::InvalidArgument.
> - If `input.numel() == 0`, return.
> - Let `leading_dims = getLeadingDims(input, dim)`, `trailing_dims =
>   getTrailingDims(input, dim)`, `step = input.size(dim) * trailing_dims`.
> - Dtype dispatch: in_type and out[0] type from REALHBBF16 = {Byte, Char, Short,
>   Int, Long, Bool, Half, Float, Double, BFloat16} as CTYPE_IN / CTYPE_OUT.
> - For each output tensor `i` (the slice at index `i` along `dim`):
>   `input_offset = i * trailing_dims`, `dest_offset = 0`. For each `j` in
>   `[0, leading_dims)`: copy `trailing_dims` elements `dest[dest_offset + k] =
>   convert<CTYPE_OUT, CTYPE_IN>(input_data[input_offset + k])`, then advance
>   `input_offset += step` and `dest_offset += trailing_dims`. This gathers, for
>   fixed index `i` along `dim`, the trailing block from every leading slab.
> - Returns void.

