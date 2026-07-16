# kernels/portable/cpu/op_split_copy.cpp

> [spec:et:def:op-split-copy.torch.executor.native.split-copy-tensor-out-fn]
> void split_copy_Tensor_out( KernelRuntimeContext& ctx, const Tensor& input, int64_t split_size, int64_t dim, TensorList out)

> [spec:et:sem:op-split-copy.torch.executor.native.split-copy-tensor-out-fn]
> Splits `input` into equal-`split_size` chunks along `dim`, writing each chunk
> into the corresponding tensor in `out`. Implements `split_copy.Tensor_out(Tensor
> input, int split_size, int dim=0, *, Tensor(a!)[] out)`. The last chunk is
> smaller if `input.size(dim)` is not divisible by `split_size`. Returns void
> (writes through `out`). Step by step:
>
> - Normalize `dim`: if `dim < 0`, `dim += input.dim()`.
> - ET_KERNEL_CHECK `check_split_copy_args(input, split_size, dim, out)` (see
>   `[spec:et:sem:copy-ops-util...check-split-copy-args-fn]`): validates
>   `split_size >= 1`, `dim` valid, `out.size()` equals the expected number of
>   chunks `ceil(input.size(dim)/split_size)`, and each out tensor has the
>   expected shape and matching dtype. On failure sets Error::InvalidArgument and
>   returns (empty return value).
> - For each `i` in `[0, out.size())` ET_KERNEL_CHECK
>   `tensors_have_same_dim_order(input, out[i])`; else Error::InvalidArgument.
> - Let `leading_dims = getLeadingDims(input, dim)` (product of sizes before
>   `dim`), `trailing_dims = getTrailingDims(input, dim)` (product after `dim`),
>   `step = input.size(dim) * trailing_dims` (elements per leading slab of input).
> - Dtype dispatch: in_type and out[0] type both from REALHBBF16 = {Byte, Char,
>   Short, Int, Long, Bool, Half, Float, Double, BFloat16}, switched
>   independently as CTYPE_IN and CTYPE_OUT.
> - `input_data` starts at input base. For each chunk `i`:
>   `out_step = out[i].size(dim) * trailing_dims`; if `out_step == 0` skip this
>   chunk (do not advance `input_data`). Otherwise copy: for each `j` in
>   `[0, leading_dims)` copy `out_step` elements `dest[k] =
>   convert<CTYPE_OUT, CTYPE_IN>(src[k])` (numeric cast), then advance `src` by
>   `step` and `dest` by `out_step`; after the leading loop advance `input_data`
>   by `out_step` so the next chunk starts at the next block along `dim`.
> - Returns void.

