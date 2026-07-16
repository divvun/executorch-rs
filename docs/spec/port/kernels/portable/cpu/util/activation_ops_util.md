# kernels/portable/cpu/util/activation_ops_util.cpp

> [spec:et:def:activation-ops-util.torch.executor.check-gelu-args-fn]
> bool check_gelu_args( const Tensor& in, std::string_view approximate, Tensor& out)

> [spec:et:sem:activation-ops-util.torch.executor.check-gelu-args-fn]
> Validates arguments for the gelu op; returns `true` if all checks pass, `false`
> on the first failed check (each failing check also logs an Error). Checks, in
> order (short-circuit on first failure):
> 1. `tensors_have_same_dtype(in, out)` — `in` and `out` must have identical dtype.
> 2. `in.scalar_type() != ScalarType::Bool` — Bool input is rejected.
> 3. `approximate == "tanh" || approximate == "none"` — the approximation mode
>    string must be exactly "tanh" or "none", otherwise logs "Invalid
>    approximation format: <approximate> for gelu" and returns false.
> Returns true only if all three hold.

> [spec:et:def:activation-ops-util.torch.executor.check-glu-args-fn]
> bool check_glu_args(const Tensor& in, int64_t dim, Tensor& out)

> [spec:et:sem:activation-ops-util.torch.executor.check-glu-args-fn]
> Validates arguments for the glu (gated linear unit) op, which halves the input
> along `dim`. Returns `true` iff all checks pass, `false` on the first failure
> (each logs an Error). Checks in order (short-circuit):
> 1. `dim_is_valid(dim, in.dim())` — `dim` must be in `[-in.dim(), in.dim()-1]`.
> 2. `tensor_is_floating_type(in)` — input must be a floating type.
> 3. Compute `non_negative_dim = dim < 0 ? dim + in.dim() : dim` and
>    `dim_size = in.size(non_negative_dim)`. Require `dim_size % 2 == 0` (the
>    halving dimension must be even); else log "Halving dimension must be even..."
>    and return false.
> 4. `tensor_is_floating_type(out)` — output must be a floating type.
> 5. `tensors_have_same_rank(in, out)` — same number of dimensions.
> 6. `out.size(non_negative_dim) == dim_size / 2` — output's size along the split
>    dim must be half the input's; else log and return false.
> 7. For every other dimension `i != non_negative_dim` (0..in.dim()), require
>    `out.size(i) == in.size(i)`; on the first mismatch log the output/input
>    shapes and return false.
> Returns true only if all checks hold. Note: does NOT check input/output dtype
> equality — only that both are floating types (they may differ, e.g. Float
> in, Double out).

> [spec:et:def:activation-ops-util.torch.executor.check-log-softmax-args-fn]
> bool check_log_softmax_args( const Tensor& in, int64_t dim, bool half_to_float, Tensor& out)

> [spec:et:sem:activation-ops-util.torch.executor.check-log-softmax-args-fn]
> Validates arguments for the log_softmax op. Returns `true` iff all checks pass,
> `false` on the first failure (each logs an Error). Checks in order
> (short-circuit):
> 1. `!half_to_float` — require `half_to_float` to be false; else log "half to
>    float conversion is not supported on CPU" and return false.
> 2. `tensors_have_same_dtype(in, out)` — `in` and `out` must have identical dtype.
> 3. `tensor_has_dim(in, dim)` — `dim` must be a valid dimension of `in`
>    (supports negative indexing).
> 4. `tensor_is_default_or_channels_last_dim_order(in)` — input dim order must be
>    default (contiguous) or channels-last.
> 5. `tensor_is_default_or_channels_last_dim_order(out)` — same for output.
> Returns true only if all hold.

> [spec:et:def:activation-ops-util.torch.executor.check-softmax-args-fn]
> bool check_softmax_args( const Tensor& in, int64_t dim, bool half_to_float, Tensor& out)

> [spec:et:sem:activation-ops-util.torch.executor.check-softmax-args-fn]
> Validates arguments for the softmax op. Identical validation to log_softmax:
> directly returns `check_log_softmax_args(in, dim, half_to_float, out)` per
> `[spec:et:sem:activation-ops-util.torch.executor.check-log-softmax-args-fn]`
> (rejects `half_to_float`, requires same dtype for `in`/`out`, `dim` valid for
> `in`, and both tensors in default or channels-last dim order).

> [spec:et:def:activation-ops-util.torch.executor.resize-glu-out-fn]
> Error resize_glu_out(const Tensor& in, int64_t dim, Tensor& out)

> [spec:et:sem:activation-ops-util.torch.executor.resize-glu-out-fn]
> Resizes the glu output tensor to the input shape with the split dimension
> halved. Steps:
> 1. Allocate a local `expected_output_size` array of `kTensorDimensionLimit`
>    SizesType entries.
> 2. Compute `non_negative_dim = dim < 0 ? dim + in.dim() : dim`.
> 3. For each `i` in `0..in.dim()`: set `expected_output_size[i] = in.size(i)/2`
>    if `i == non_negative_dim`, else `in.size(i)` (integer division halves the
>    split dim).
> 4. Build an ArrayRef view over the first `out.dim()` entries of
>    `expected_output_size` (note the length uses `out.dim()`, not `in.dim()`).
> 5. Return `resize_tensor(out, output_size)` — propagating its Error (Error::Ok
>    on success). Performs no dtype/validity checks itself (those belong to
>    `check_glu_args`); this only sets the output shape.

