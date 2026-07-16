# kernels/portable/cpu/op_lift_fresh_copy.cpp

> [spec:et:def:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn]
> Tensor&

> [spec:et:sem:op-lift-fresh-copy.torch.executor.native.lift-fresh-copy-out-fn]
> Copies `in` into `out` verbatim (a raw byte copy); functionally an identity
> copy for the `lift_fresh_copy` op.
>
> Steps:
> 1. Check `in` and `out` have the same dtype (ET_KERNEL_CHECK: on failure sets
>    Error::InvalidArgument on the context and returns `out` unchanged).
> 2. Resize `out` to `in.sizes()` (ET_KERNEL_CHECK: InvalidArgument, returns
>    `out`).
> 3. Check `in` and `out` have the same dim order (ET_KERNEL_CHECK:
>    InvalidArgument, returns `out`).
> 4. If `in.nbytes() > 0`, `memcpy` `in.nbytes()` bytes from `in`'s data pointer
>    to `out`'s data pointer. The guard is required because a tensor with numel 0
>    may legally have a null data pointer, and passing null to memcpy is invalid
>    in some environments even for size 0; so when `in.nbytes() == 0` no copy is
>    performed.
> 5. Return `out`.
>
> Accepts any dtype (no dtype dispatch; the copy is byte-for-byte, so element
> layout must already match, which is guaranteed by the same-dtype and
> same-dim-order checks). No value conversion occurs.

