# kernels/portable/cpu/op_stack.cpp

> [spec:et:def:op-stack.torch.executor.native.stack-out-fn]
> Tensor& stack_out( KernelRuntimeContext& ctx, executorch::aten::ArrayRef<Tensor> tensors, int64_t dim, Tensor& out)

> [spec:et:sem:op-stack.torch.executor.native.stack-out-fn]
> Concatenates the tensors in `tensors` along a new dimension `dim`, producing a
> tensor with one more dimension than the inputs. Implements `stack.out(Tensor[]
> tensors, int dim=0, *, Tensor(a!) out)`. This function is a thin wrapper that
> forwards all arguments unchanged to `utils::stack_out_impl(ctx, tensors, dim,
> out)` and returns its result. The full behavior is defined by
> `[spec:et:sem:stack-util...stack-out-impl-fn]`: it validates the input list is
> non-empty and all inputs share the same shape and dtype, normalizes negative
> `dim` against `out.dim()` (the stacked rank = input rank + 1), resizes `out` so
> that dimension `dim` has size `tensors.size()`, checks dim order, and writes
> each input tensor into the corresponding slice of `out` along the new
> dimension. Returns `out` (with error propagated via the context on any check
> failure, in which case `out` is returned unchanged).

