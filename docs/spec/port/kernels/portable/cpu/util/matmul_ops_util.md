# kernels/portable/cpu/util/matmul_ops_util.cpp, kernels/portable/cpu/util/matmul_ops_util.h

> [spec:et:def:matmul-ops-util.torch.executor.check-addmm-args-fn]
> bool check_addmm_args( const Tensor& in, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, Tensor& out)

> [spec:et:sem:matmul-ops-util.torch.executor.check-addmm-args-fn]
> Validates the argument shapes/dtypes for `addmm` (out = beta*in + alpha*(mat1 @ mat2)).
> Full signature: `check_addmm_args(in, mat1, mat2, beta, alpha, out)`. `beta`
> and `alpha` are accepted but not inspected here (only mat/out shapes and
> dtypes are checked).
>
> Returns `bool`. Each check below uses `ET_LOG_AND_RETURN_IF_FALSE`: if the
> predicate is false the function logs and immediately returns `false`;
> otherwise it falls through. If all checks pass it returns `true`. Checks are
> evaluated in this order (short-circuiting on the first failure):
> 1. `mat1` is rank 2 (`mat1.dim() == 2`).
> 2. `mat2` is rank 2.
> 3. `out` is rank 2.
> 4. `in`, `mat1`, `mat2` all share the same dtype.
> 5. `in` and `out` share the same dtype (transitively all four tensors share
>    one dtype).
> 6. The inner matmul dimensions agree: `mat1.size(1) == mat2.size(0)`.
>
> `in` may be any rank/shape here (its broadcastability against the `[mat1.size(0),
> mat2.size(1)]` result is not validated by this function). No output is
> written; this is a pure validation predicate.

> [spec:et:def:matmul-ops-util.torch.executor.check-bmm-args-fn]
> bool check_bmm_args(const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:matmul-ops-util.torch.executor.check-bmm-args-fn]
> Validates argument shapes/dtypes for batched matmul `bmm` (out[b] = in[b] @
> mat2[b]). Returns `bool`. Each check uses `ET_LOG_AND_RETURN_IF_FALSE`
> (returns `false` on the first failed predicate); returns `true` if all pass.
> Checks in order:
> 1. `in` is rank 3.
> 2. `mat2` is rank 3.
> 3. `out` is rank 3.
> 4. `in`, `mat2`, `out` all share the same dtype.
> 5. Batch dims match: `in.size(0) == mat2.size(0)`.
> 6. Contraction dims match: `in.size(2) == mat2.size(1)`.
>
> The output dimensions (`in.size(1)` and `mat2.size(2)`) are not checked here;
> they are produced by `[spec:et:sem:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]`.
> Pure validation predicate; writes no output.

> [spec:et:def:matmul-ops-util.torch.executor.check-linear-args-fn]
> bool check_linear_args(const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:matmul-ops-util.torch.executor.check-linear-args-fn]
> Validates argument shapes/dtypes for `linear` (out = in @ mat2^T, where
> `mat2` is the weight matrix of shape `[out_features, in_features]`). Returns
> `bool`. Each check uses `ET_LOG_AND_RETURN_IF_FALSE` (returns `false` on the
> first failed predicate); returns `true` if all pass. Checks in order:
> 1. `in.dim() == out.dim()` (input and output have equal rank).
> 2. `in.dim() >= 2`.
> 3. `mat2` is rank 2.
> 4. `in`, `mat2`, `out` all share the same dtype.
> 5. The contraction dim matches: `in.size(in.dim() - 1) == mat2.size(1)` (the
>    trailing/feature dim of `in` equals the number of columns of `mat2`, i.e.
>    `in_features`).
>
> Note the transposed convention: `mat2.size(1)` is contracted (not
> `mat2.size(0)` as in `mm`). Leading dims of `in` are not otherwise
> constrained. Pure validation predicate; writes no output.

> [spec:et:def:matmul-ops-util.torch.executor.check-mm-args-fn]
> bool check_mm_args(const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:matmul-ops-util.torch.executor.check-mm-args-fn]
> Validates argument shapes/dtypes for 2-D matmul `mm` (out = in @ mat2).
> Returns `bool`. Each check uses `ET_LOG_AND_RETURN_IF_FALSE` (returns `false`
> on the first failed predicate); returns `true` if all pass. Checks in order:
> 1. `in` is rank 2.
> 2. `mat2` is rank 2.
> 3. `out` is rank 2.
> 4. `in`, `mat2`, `out` all share the same dtype.
> 5. Contraction dims match: `in.size(1) == mat2.size(0)`.
>
> Output dims (`in.size(0)`, `mat2.size(1)`) are produced by
> `[spec:et:sem:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]` and
> not checked here. Pure validation predicate; writes no output.

> [spec:et:def:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]
> void get_bmm_out_target_size( const Tensor& mat1, const Tensor& mat2, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]
> Computes the target output shape for batched matmul `bmm`. Signature
> `get_bmm_out_target_size(mat1, mat2, out_sizes, out_ndim)` where `out_sizes`
> is a caller-provided buffer and `out_ndim` an out-param. No return value; no
> validation (caller is expected to have run
> `[spec:et:sem:matmul-ops-util.torch.executor.check-bmm-args-fn]`).
>
> Behavior:
> 1. Set `*out_ndim = 3`.
> 2. `out_sizes[0] = mat1.size(0)` (batch).
> 3. `out_sizes[1] = mat1.size(1)` (M, rows of each left matrix).
> 4. `out_sizes[2] = mat2.size(2)` (P, cols of each right matrix).
>
> The result shape is `[batch, M, P]`.

> [spec:et:def:matmul-ops-util.torch.executor.get-linear-out-target-size-fn]
> void get_linear_out_target_size( const Tensor& mat1, const Tensor& mat2, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:matmul-ops-util.torch.executor.get-linear-out-target-size-fn]
> Computes the target output shape for `linear` (out = mat1 @ mat2^T).
> Signature `get_linear_out_target_size(mat1, mat2, out_sizes, out_ndim)`;
> `out_sizes` is a caller buffer, `out_ndim` an out-param. No return value; no
> validation.
>
> Behavior:
> 1. Set `*out_ndim = mat1.dim()` (output has the same rank as the input).
> 2. For each `ii` in `[0, mat1.dim() - 1)`: copy the leading dims through,
>    `out_sizes[ii] = mat1.sizes()[ii]`.
> 3. Set the last dim to the number of output features:
>    `out_sizes[mat1.dim() - 1] = mat2.size(0)`.
>
> So all of `mat1`'s dims except the last are preserved and the trailing
> in_features dim is replaced by `mat2.size(0)` (out_features), reflecting the
> transposed weight convention used by
> `[spec:et:sem:matmul-ops-util.torch.executor.check-linear-args-fn]`.

> [spec:et:def:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]
> void get_mm_out_target_size( const Tensor& mat1, const Tensor& mat2, Tensor::SizesType* out_sizes, size_t* out_ndim)

> [spec:et:sem:matmul-ops-util.torch.executor.get-mm-out-target-size-fn]
> Computes the target output shape for 2-D matmul `mm` (out = mat1 @ mat2).
> Signature `get_mm_out_target_size(mat1, mat2, out_sizes, out_ndim)`;
> `out_sizes` is a caller buffer, `out_ndim` an out-param. No return value; no
> validation.
>
> Behavior:
> 1. Set `*out_ndim = 2`.
> 2. `out_sizes[0] = mat1.size(0)` (M).
> 3. `out_sizes[1] = mat2.size(1)` (P).
>
> Result shape `[M, P]`.

> [spec:et:def:matmul-ops-util.torch.executor.internal.bmm-out-impl-fn]
> void bmm_out_impl(const Tensor& in, const Tensor& mat2, Tensor& out)

> [spec:et:sem:matmul-ops-util.torch.executor.internal.bmm-out-impl-fn]
> Templated batched-matmul kernel `bmm_out_impl<CTYPE>(in, mat2, out)` that
> writes `out[b] = in[b] @ mat2[b]` for every batch `b`, computing over the
> element type `CTYPE`. No return value; results are written in place into
> `out`'s data buffer. Assumes shapes already validated by
> `[spec:et:sem:matmul-ops-util.torch.executor.check-bmm-args-fn]` and `out`
> already sized per
> `[spec:et:sem:matmul-ops-util.torch.executor.get-bmm-out-target-size-fn]`; it
> performs no checks itself.
>
> Setup:
> - `in_data`, `mat2_data`, `out_data` are the raw contiguous `CTYPE` buffers
>   (obtained via `const_data_ptr`/`mutable_data_ptr`).
> - `batch_size = in.size(0)`, `m = in.size(1)`, `n = in.size(2)`,
>   `p = mat2.size(2)`.
>
> Algorithm (row-major contiguous layout assumed throughout):
> For each `b` in `[0, batch_size)`:
> - `in_data_offset = in_data + b*m*n`, `mat2_data_offset = mat2_data + b*n*p`,
>   `out_data_offset = out_data + b*m*p`.
> - For each `i` in `[0, m)`, for each `j` in `[0, p)`:
>   - Initialize accumulator `sum = static_cast<CTYPE>(0.0)` (accumulation is in
>     `CTYPE`, NOT a widened type — porting must match this to reproduce
>     rounding for low-precision dtypes).
>   - For each `k` in `[0, n)`: `sum += in_data_offset[i*n + k] *
>     mat2_data_offset[k*p + j]`.
>   - Write `out_data_offset[i*p + j] = sum`.
>
> Iteration order is batch-major, then row `i`, then column `j`, then
> contraction `k`. Empty dims (any of `batch_size`, `m`, `p`, `n` == 0) simply
> produce empty loops (for `n == 0`, each output element is `0`).

