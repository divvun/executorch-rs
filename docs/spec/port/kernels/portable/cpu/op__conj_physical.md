# kernels/portable/cpu/op__conj_physical.cpp

> [spec:et:def:op-conj-physical.torch.executor.native.conj-physical-out-fn]
> Tensor&

> [spec:et:sem:op-conj-physical.torch.executor.native.conj-physical-out-fn]
> Implements `_conj_physical.out(in, out)`: physically conjugates every element
> of complex tensor `in` into `out` (negates the imaginary part).
>
> Validation, in order:
> 1. `resize_tensor(out, in.sizes()) == Error::Ok` (`ET_KERNEL_CHECK_MSG`; on
>    failure sets context Error to `InvalidArgument`, message "Failed to resize
>    output tensor.", returns `out`).
> 2. `tensors_have_same_dtype(in, out)` (`ET_KERNEL_CHECK` → `InvalidArgument`,
>    returns `out`).
> 3. `tensors_have_same_dim_order(in, out)` (`ET_KERNEL_CHECK` →
>    `InvalidArgument`, returns `out`).
>
> Dtype dispatch: switches on `in.scalar_type()` over `ET_SWITCH_COMPLEXH_TYPES`
> (accepted set {ComplexHalf, ComplexFloat, ComplexDouble}); any other dtype
> fails the switch. `out` shares the same complex CTYPE as `in`.
>
> Applies element-wise via `apply_unary_map_fn<CTYPE, CTYPE>` over all
> `in.numel()` elements in flat order: for each input value `v`, writes
> `CTYPE(v.real_, -v.imag_)` (real part unchanged, imaginary part negated).
> `apply_unary_map_fn` walks the flat contiguous index range `[0, numel)`.
>
> Returns `out`.

