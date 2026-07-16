# kernels/portable/cpu/op_addmm.cpp

> [spec:et:def:op-addmm.torch.executor.native.addmm-out-fn]
> Tensor& addmm_out( KernelRuntimeContext& ctx, const Tensor& in, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& out)

> [spec:et:sem:op-addmm.torch.executor.native.addmm-out-fn]
> Implements `addmm.out(in, mat1, mat2, beta, alpha, out)`: computes
> `out = beta * in + alpha * (mat1 @ mat2)`, where `mat1` is `[m, n]`, `mat2` is
> `[n, p]`, the matmul result is `[m, p]`, and `in` is broadcastable to `[m, p]`.
>
> Validation, in order (each `ET_KERNEL_CHECK` → `InvalidArgument`, returns
> `out`):
> 1. `check_addmm_args(in, mat1, mat2, beta, alpha, out)` — shapes/dtypes valid
>    for addmm.
> 2. Compute target output size via `get_mm_out_target_size(mat1, mat2, ...)`
>    (`= [mat1.size(0), mat2.size(1)]`), then `resize_tensor(out, target)`.
> 3. `tensor_is_broadcastable_to(in, out)` — `in` broadcasts to `out`'s shape.
> 4. `tensors_have_same_dim_order(in, mat1, mat2, out)`.
> 5. `tensor_is_default_dim_order(in)`.
>
> Dtype dispatch: `ET_SWITCH_REALHBF16_TYPES` on `in.scalar_type()` (CTYPE ∈
> {Byte, Char, Short, Int, Long, Half, Float, Double, BFloat16}); all tensors use
> this same CTYPE. Computes `alpha_val = scalar_to<CTYPE>(alpha)`, `beta_val =
> scalar_to<CTYPE>(beta)`, `m = mat1.size(0)`, `n = mat1.size(1)`, `p =
> mat2.size(1)`.
>
> Two paths:
> - No broadcast (`out.sizes() == in.sizes()`): calls `vec_addmm<CTYPE,CTYPE>
>   (out_ptr, in_ptr, mat1_ptr, mat2_ptr, m, n, p, beta_val, alpha_val)`, which
>   computes `out = beta_val * in + alpha_val * (mat1 @ mat2)` in one fused pass
>   (no broadcasting).
> - Broadcast needed: first `vec_matmul<CTYPE,CTYPE>(out_ptr, mat1_ptr, mat2_ptr,
>   m, n, p)` writes `mat1 @ mat2` into `out`; then
>   `utils::apply_bitensor_elementwise_fn<CTYPE, op_name, REALHBF16>` applies
>   `val_a * alpha_val + val_b * beta_val` with `a = out` (the matmul result) and
>   `b = in`, broadcasting `in` to `out`'s shape, writing back into `out`. Both
>   operand tensors are read as `REALHBF16` dtypes.
>
> Returns `out`.

