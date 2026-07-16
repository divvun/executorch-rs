# kernels/portable/cpu/util/stack_util.cpp

> [spec:et:def:stack-util.torch.executor.native.utils.stack-out-shape-fn]
> std::tuple<

> [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-shape-fn]
> Validates a `stack` operation and computes its output shape without touching an
> output tensor. Returns a tuple `(Error, out_sizes, out_dim)` where `out_sizes`
> is a `std::array<SizesType, kTensorDimensionLimit>` and `out_dim` a `size_t`. On
> any validation failure it returns `Error::InvalidArgument` with a zero-initialized
> `out_sizes` and `out_dim == 0`; on success `Error::Ok` with the filled shape.
>
> Steps:
> 1. If `tensors.size() == 0`, return `InvalidArgument` (need at least one input).
> 2. Normalize `dim`: `normalized_dim = dim`; if `< 0`, `normalized_dim += tensors[0].dim() + 1`
>    (the result rank is `tensors[0].dim() + 1`, so the valid inclusive range is
>    `[0, tensors[0].dim()]`).
> 3. If `normalized_dim < 0 || normalized_dim > tensors[0].dim()`, return
>    `InvalidArgument`.
> 4. Shape agreement: for every `i` in `[1, tensors.size())`, require
>    `tensors[i].dim() == tensors[0].dim()` and `tensors[i].size(d) == tensors[0].size(d)`
>    for all `d`; any mismatch returns `InvalidArgument`.
> 5. Compute the output shape via `get_stack_out_target_size(tensors, normalized_dim, ...)`
>    per `[spec:et:sem:copy-ops-util.torch.executor.get-stack-out-target-size-fn]`
>    (inserts a new dimension of size `tensors.size()` at `normalized_dim`), and
>    return `(Error::Ok, out_sizes, out_dim)`.

> [spec:et:def:stack-util.torch.executor.native.utils.stack-out-impl-fn]
> Tensor& stack_out_impl( KernelRuntimeContext& ctx, executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, Tensor& out)

> [spec:et:sem:stack-util.torch.executor.native.utils.stack-out-impl-fn]
> Stacks `tensors` along `dim` into `out`, inserting a new dimension of size
> `tensors.size()`. Returns a reference to `out`. Uses `ctx` for error reporting;
> on any failed `ET_KERNEL_CHECK` it sets `Error::InvalidArgument` on `ctx` and
> returns `out` unchanged.
>
> Steps:
> 1. Normalize `dim`: if `dim < 0`, `dim += out.dim()` (out already has the stacked
>    rank).
> 2. `ET_KERNEL_CHECK`: `check_stack_args(tensors, dim, out)` per
>    `[spec:et:sem:copy-ops-util.torch.executor.check-stack-args-fn]`.
> 3. `ET_KERNEL_CHECK`: for every input `i`, `tensors_have_same_dim_order(tensors[i], out)`.
> 4. `ET_KERNEL_CHECK`: `tensor_is_default_dim_order(out)` (out must be contiguous
>    default dim order).
> 5. Compute expected shape via `get_stack_out_target_size(tensors, dim, ...)` per
>    `[spec:et:sem:copy-ops-util.torch.executor.get-stack-out-target-size-fn]` and
>    `ET_KERNEL_CHECK` that `resize_tensor(out, expected)` returns `Error::Ok`.
> 6. Compute geometry relative to the stack dimension: `outer = getLeadingDims(out, dim)`
>    (product of out sizes before `dim`), `inner = getTrailingDims(out, dim)` (product
>    after `dim`), `ninputs = tensors.size()`.
> 7. Dtype dispatch: `ET_SWITCH_REALHBBF16_TYPES` on `out.scalar_type()` selects
>    `CTYPE_OUT` — the accepted output dtypes are the REALHBBF16 set: Byte(uint8),
>    Char(int8), Short(int16), Int(int32), Long(int64), Half(float16),
>    Float(float32), Double(float64), Bool, and BFloat16. Any other dtype fails the
>    switch (ctx error). Copy loop, writing `out` contiguously: for `i` in `[0, outer)`,
>    for `j` in `[0, ninputs)`, dispatch again on `tensors[j].scalar_type()` to get
>    `CTYPE_IN` (same REALHBBF16 set), take `in_ptr = tensors[j] data + i*inner`, and
>    for `k` in `[0, inner)` write `out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k])`,
>    then advance `out_ptr += inner`. Element values are converted from each input's
>    dtype to the output dtype via C++ `static_cast` (inputs may differ in dtype from
>    the output and from each other).
> 8. Return `out`.
