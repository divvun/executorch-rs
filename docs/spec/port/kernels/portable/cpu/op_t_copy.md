# kernels/portable/cpu/op_t_copy.cpp

> [spec:et:def:op-t-copy.torch.executor.native.t-copy-out-fn]
> Tensor& t_copy_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-t-copy.torch.executor.native.t-copy-out-fn]
> Transposes dimensions 0 and 1 of a <= 2-D tensor into `out`. Implements
> `t_copy.out(Tensor self, *, Tensor(a!) out)`. 0-D and 1-D tensors are copied
> unchanged; a 2-D tensor is transposed (equivalent to transpose(in, 0, 1)).
> Step by step:
>
> - ET_KERNEL_CHECK `check_t_copy_args(in, out)` (see
>   `[spec:et:sem:transpose-util...check-t-copy-args-fn]`): validates `in.dim() <=
>   2`, and `in`/`out` share dtype with the correct transposed shape. On failure
>   sets Error::InvalidArgument and returns `out` unchanged.
> - If `in.dim() < 2` (0-D or 1-D): resize `out` to `in.sizes()` (on failure
>   Error::InvalidArgument); if `in.numel() > 0`, switch over ALL types (any
>   ScalarType) and `memcpy(out_data, in_data, in.nbytes())` (byte-for-byte
>   copy). Return `out`.
> - Otherwise (2-D): ET_KERNEL_CHECK `tensors_have_same_dim_order(in, out)` and
>   `tensor_is_default_dim_order(in)`; else Error::InvalidArgument. Compute
>   `expected_out_size` via `get_transpose_out_target_size(in, 1, 0, ...)` (swap
>   dims 1 and 0) and resize `out`; on failure Error::InvalidArgument. Then switch
>   over ALL types and call `transpose_tensors<CTYPE>(in, 1, 0, out)` (see
>   `[spec:et:sem:transpose-util...transpose-tensors-fn]`), which writes the
>   transposed data densely (contiguous) into `out`.
> - Returns `out`.

