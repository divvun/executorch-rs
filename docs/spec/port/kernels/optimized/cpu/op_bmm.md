# kernels/optimized/cpu/op_bmm.cpp

> [spec:et:def:op-bmm.torch.executor.native.bmm-kernel-fn]
> void bmm_kernel(const Tensor& self, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-bmm.torch.executor.native.bmm-kernel-fn]
> Batched matrix multiply of the 3-D tensors `self` (b×n×k) and `mat2` (b×k×m)
> into `out` (b×n×m), one column-major GEMM per batch. If any of `self`/`mat2`/
> `out` is empty (`numel() == 0`), return immediately. Let `b_data = self`,
> `a_data = mat2`, `c_data = out`, `batch_size = self.size(0)`, `n = self.size(1)`,
> `k = self.size(2)`, `m = mat2.size(2)`. For each batch `i` in `[0, batch_size)`:
> `a = a_data + i*m*k`, `b = b_data + i*k*n`, `c = c_data + i*m*n`, then
> `cpublas::gemm(NoTranspose, NoTranspose, m, n, k, 1, a, m, b, k, 0, c, m)`. As
> in mm, the column-major GEMM with untransposed `(mat2, self)` produces the
> row-major per-batch product `self_i @ mat2_i`.

> [spec:et:def:op-bmm.torch.executor.native.check-bmm-out-args-fn]
> bool check_bmm_out_args(const Tensor& self, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-bmm.torch.executor.native.check-bmm-out-args-fn]
> Validate the bmm operands, logging and returning `false` on the first failed
> check (else `true`). In order: `self.dim() == mat2.dim()`; `self.dim() ==
> out.dim()`; `self.dim() == 3`; `self.size(0) >= 0`; `self.size(0) ==
> mat2.size(0)`; `self.size(0) == out.size(0)`; `mat2.size(2) == out.size(2)`;
> `self.size(1) == out.size(1)`; and finally `tensors_have_same_dtype(self, mat2,
> out)`.

> [spec:et:def:op-bmm.torch.executor.native.opt-bmm-out-fn]
> Tensor& opt_bmm_out( KernelRuntimeContext& ctx, const Tensor& self, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-bmm.torch.executor.native.opt-bmm-out-fn]
> Optimized `bmm.out(self, mat2, *, out)`. First `resize_out_tensor(self, mat2,
> out)` (InvalidArgument + return `out` on non-Ok), then `check_bmm_out_args(self,
> mat2, out)` (InvalidArgument on failure). Dispatch on `self.scalar_type()`: for
> complex dtypes switch over ComplexHalf/ComplexFloat/ComplexDouble and call
> `bmm_kernel<CTYPE>(self, mat2, out)`; otherwise switch over real + Half +
> BFloat16 and call `bmm_kernel<CTYPE>(self, mat2, out)`. Return `out`.

> [spec:et:def:op-bmm.torch.executor.native.resize-out-tensor-fn]
> Error resize_out_tensor(const Tensor& self, const Tensor& mat2, Tensor& out)

> [spec:et:sem:op-bmm.torch.executor.native.resize-out-tensor-fn]
> Compute the bmm output shape and resize `out`. Let `m_dim = self.dim() - 2`,
> `n_dim = self.dim() - 1`. Copy `self.size(i)` into `expected_output_size[i]` for
> every leading dim `i` in `[0, m_dim)`. If `m_dim >= self.dim()` or `n_dim >=
> mat2.dim()`, log "Incompatible matrix multiply dimensions." and return
> `Error::InvalidArgument`. Otherwise set `expected_output_size[m_dim] =
> self.size(m_dim)` and `expected_output_size[n_dim] = mat2.size(n_dim)`, form an
> `ArrayRef` of length `out.dim()` over `expected_output_size`, and return
> `resize_tensor(out, that)`.

