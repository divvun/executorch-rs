# kernels/optimized/cpu/op_linear.cpp

> [spec:et:def:op-linear.torch.executor.native.initialize-scalar-fn]
> void initialize_scalar( const ssize_t out_numel, const scalar_t init, scalar_t* out)

> [spec:et:sem:op-linear.torch.executor.native.initialize-scalar-fn]
> Fill the first `out_numel` elements of the buffer `out` with the scalar value
> `init`. The C++ broadcasts `init` into a `Vectorized<scalar_t>` and stores it in
> vector-length chunks (`d += Vec::size()`), then does one sub-vector-length store
> of the `out_numel - d` remainder. The observable result is exactly `out[i] =
> init` for every `i` in `[0, out_numel)`.

> [spec:et:def:op-linear.torch.executor.native.initialize-to-vector-fn]
> void initialize_to_vector( const ssize_t n, const ssize_t m, const scalar_t* bias, scalar_t* out)

> [spec:et:sem:op-linear.torch.executor.native.initialize-to-vector-fn]
> `out` is an `n x m` row-major matrix of `scalar_t`; `bias` is an `m`-element
> vector. Broadcast `bias` into every row of `out`: for each `col` in `[0, n)`,
> `memcpy` `m * sizeof(scalar_t)` bytes from `bias` to `out + col * m`. After the
> loop each of the `n` rows equals `bias`.

> [spec:et:def:op-linear.torch.executor.native.opt-linear-out-fn]
> Tensor& opt_linear_out( RuntimeContext& ctx, const Tensor& in, const Tensor& mat2, const optional<Tensor>& bias, Tensor& out)

> [spec:et:sem:op-linear.torch.executor.native.opt-linear-out-fn]
> Optimized `linear.out`: `out = in @ mat2^T (+ bias)` where `in` is `[*, K]`,
> `mat2` (the weight) is `[N, K]`, `out` is `[*, N]`. Steps: (1) `check_linear_args
> (in, mat2, out)`; InvalidArgument + return `out` on failure. (2) Compute the
> target output sizes via `get_linear_out_target_size` and `resize_tensor(out,
> ...)`; InvalidArgument on failure. (3) If `out.numel() == 0`, return `out`
> (GEMM doesn't tolerate empty input). (4) Compute `n = product of in.sizes[0..
> in.dim()-1]` (the flattened leading dims), `k = in.sizes[in.dim()-1]`, `m =
> mat2.size(0)`. (5) If `bias` is present, check `bias.dtype() == out.dtype()`
> and that `bias` is 1-D with `size(0) == m` or `== 1` (InvalidArgument otherwise).
> (6) Switch over `out.scalar_type()` across real + Half + BFloat16: if `bias`
> has exactly one element, prefill `out` with that scalar via `initialize_scalar
> (out.numel(), *bias_data, out_data)`; else if `bias` is present, broadcast the
> `m`-vector bias into every row via `initialize_to_vector(n, m, bias_data,
> out_data)`. Set `beta = 1` when `bias` is present (GEMM accumulates onto the
> prefilled bias) else `beta = 0` (GEMM fully overwrites). Then call `cpublas::
> gemm(Transpose, NoTranspose, m, n, k, 1, mat2_data, k, in_data, k, beta,
> out_data, m)` — weight transposed, `in` untransposed. (7) Return `out`.

