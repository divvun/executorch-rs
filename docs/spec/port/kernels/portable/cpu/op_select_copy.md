# kernels/portable/cpu/op_select_copy.cpp

> [spec:et:def:op-select-copy.torch.executor.native.select-copy-int-out-fn]
> Tensor& select_copy_int_out( KernelRuntimeContext& ctx, const Tensor& in, int64_t dim, int64_t index, Tensor& out)

> [spec:et:sem:op-select-copy.torch.executor.native.select-copy-int-out-fn]
> `select_copy.int_out`: copies the `index`-th slice of `in` along `dim` into
> `out` (which has one fewer dimension than `in`). Thin wrapper:
>
> - Call `torch::executor::select_copy_util(in, dim, index, out)` (see
>   `[spec:et:sem:select-copy-util.torch.executor.select-copy-util-fn]`), which
>   performs all argument validation (dim/index normalization and bounds, shape
>   and dtype checks, dim-order/resize) and the actual byte-level slice copy,
>   returning an `Error`.
> - If the returned `err != Error::Ok`, propagate it by calling `ctx.fail(err)`
>   (records the error on the context).
> - Returns `out` (whether or not the copy succeeded).

