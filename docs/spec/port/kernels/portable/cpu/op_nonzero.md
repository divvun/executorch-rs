# kernels/portable/cpu/op_nonzero.cpp

> [spec:et:def:op-nonzero.torch.executor.native.increment-index-fn]
> void increment_index(size_t* index, const ArrayRef<SizesType> sizes)

> [spec:et:sem:op-nonzero.torch.executor.native.increment-index-fn]
> Advances a multi-dimensional coordinate `index` (one entry per dim, sizes given by `sizes`) by one position in row-major (last-dim-fastest, C-contiguous) order, with carry.
>
> Steps: iterate `i` from `sizes.size()-1` down to 0: increment `index[i]`; if `index[i]` now equals `sizes[i]`, reset `index[i] = 0` and continue to the next-higher dim (carry); otherwise return immediately. If every dim carries (index was at the last element), `index` wraps back to all-zeros. Mutates `index` in place; no return value.

> [spec:et:def:op-nonzero.torch.executor.native.nonzero-fn]
> void nonzero(KernelRuntimeContext& ctx, const Tensor& input, Tensor& output)

> [spec:et:sem:op-nonzero.torch.executor.native.nonzero-fn]
> Templated (on input element type `CTYPE`) two-pass worker that fills `output` with the coordinates of every nonzero element of `input`. `output` is int64, shaped `[num_nonzero, input.dim()]`, one row per nonzero element. No return value.
>
> Steps:
> 1. Let `in_data = input.const_data_ptr<CTYPE>()`, `lim = input.numel()`.
> 2. First pass: count `num_nonzero` = number of `i` in [0,lim) with `in_data[i] != 0` (linear over flat memory).
> 3. Resize `output` to shape `{num_nonzero, input.dim()}`: ET_KERNEL_CHECK `resize_tensor(...) == Error::Ok`; on failure set `Error::InvalidArgument` on `ctx` and return (void) — `output` left in whatever state resize produced.
> 4. Initialize a coordinate buffer `index[kTensorDimensionLimit]` to all zeros; `out_data = output.mutable_data_ptr<int64_t>()`; `out_idx = 0`.
> 5. Second pass: for each flat index `i` in [0,lim): if `in_data[i] != 0`, write the current coordinate — for each dim `j` in [0,input.dim()) set `out_data[out_idx++] = index[j]`. Then, unconditionally, advance `index` by one via `increment_index(index, input.sizes())` per `[spec:et:sem:op-nonzero.torch.executor.native.increment-index-fn]` (keeping `index` in sync with `i` in row-major order). The nonzero test uses `!= 0`, so for floating types negative zero and any NaN/inf follow C++ `!= 0` semantics (`-0.0 != 0` is false, `NaN != 0` is true).

> [spec:et:def:op-nonzero.torch.executor.native.nonzero-out-fn]
> Tensor& nonzero_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out)

> [spec:et:sem:op-nonzero.torch.executor.native.nonzero-out-fn]
> Entry point for `nonzero.out`: validates and dispatches to the templated `nonzero` worker. Returns `out`, a 2-D int64 tensor whose rows are the coordinates of the nonzero elements of `in`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_nonzero_args(in, out)` (validates `out` is int64 and 2-D / dynamically-shaped as required); on failure set `Error::InvalidArgument` on `ctx`, return `out` unchanged.
> 2. Dispatch on `in.scalar_type()` over REALHBBF16 (ET_SWITCH_REALHBBF16_TYPES = Byte, Char, Short, Int, Long, Float, Double, Half, BFloat16); unsupported dtype → InvalidArgument, return `out`.
> 3. Invoke `nonzero<CTYPE>(ctx, in, out)` per `[spec:et:sem:op-nonzero.torch.executor.native.nonzero-fn]`, which counts nonzeros, resizes `out` to `[num_nonzero, in.dim()]`, and writes the coordinates.
> 4. Return `out`.

