# kernels/optimized/cpu/op_log_softmax.cpp

> [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-kernel-fn]
> void log_softmax_kernel(const Tensor& input, int64_t dim, Tensor& out)

> [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-kernel-fn]
> Computes log_softmax over `dim` of `input` into `out` (IN_T input, OUT_T
> output). If `input.dim() == 0`: set `out[0] = 0` and return. Else compute
> `dim_size = input.size(dim)`, `outer_size = prod(input.size(i) for i<dim)`,
> `inner_size = prod(input.size(i) for i>dim)`. Two cases:
> - Last-dim case (`dim == input.dim()-1`, inner_size == 1): parallel_for over
>   `[0, outer_size)` with GRAIN_SIZE; each outer row of `dim_size` contiguous
>   elements is a lastdim log_softmax. For each row: m = max over the row;
>   s = sum over the row of exp(x - m) (accumulated in float for reduced types);
>   log_s = log(s); out[j] = (x[j] - m) - log_s. (ATen's
>   serial_vec_log_softmax_lastdim_range / chunked vectorization collapses to
>   this scalar per-row max-subtract-logsumexp.)
> - General case: chunk sizing splits the inner dimension into `num_chunks` of
>   `chunk_size` (ATen uses BLOCK_SIZE = 64*1024, halved from server default for
>   mobile caches) and parallel_for over `[0, outer_size * num_chunks)`. For each
>   (outer, inner) position the reduction is over the `dim_size` elements at
>   stride `inner_size`: m = max; s = sum exp(x - m); log_s = log(s);
>   out = (x - m) - log_s. (serial_vec_logsoftmax_range collapses to this.)
> The max-subtract then subtract-logsumexp is the numerically stable formulation
> and must be identical to the portable kernel.

> [spec:et:def:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn]
> bool log_softmax_wrapper(const Tensor& X, int64_t dim, Tensor& out)

> [spec:et:sem:op-log-softmax.torch.executor.native.log-softmax-wrapper-fn]
> Selects the (IN_T, OUT_T) instantiation of log_softmax_kernel. If OUT_T is
> BFloat16 or Half: input dtype equals output dtype (enforced by
> check_log_softmax_args), call `log_softmax_kernel<OUT_T, OUT_T>(X, dim, out)`
> and return true (if constexpr avoids instantiating unsupported cross-type
> combos). Otherwise switch on `X.scalar_type()`: Float ->
> `log_softmax_kernel<float, OUT_T>` and return true; any other -> return false
> (unsupported input dtype; Double not yet supported).

> [spec:et:def:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn]
> Tensor& opt_log_softmax_out( KernelRuntimeContext& context, const Tensor& self, int64_t dim, bool half_to_float, Tensor& out)

> [spec:et:sem:op-log-softmax.torch.executor.native.opt-log-softmax-out-fn]
> Out-variant of _log_softmax. Steps:
> 1. ET_KERNEL_CHECK(check_log_softmax_args(self, dim, half_to_float, out)); else
>    InvalidArgument, return out.
> 2. ET_KERNEL_CHECK(resize_tensor(out, self.sizes()) == Ok); else
>    InvalidArgument, return out.
> 3. Normalize dim: `dim = dim < 0 ? dim + nonzero_dim(self) : dim`.
> 4. Switch on out.scalar_type(): Float -> `log_softmax_wrapper<float>`;
>    BFloat16 -> `log_softmax_wrapper<BFloat16>`; Half ->
>    `log_softmax_wrapper<Half>`; each captures `success` and
>    ET_KERNEL_CHECK(success) InvalidArgument. Default (Double etc.) ->
>    ET_KERNEL_CHECK(false) InvalidArgument.
> 5. Return out. `(void)context;` otherwise.
</content>
