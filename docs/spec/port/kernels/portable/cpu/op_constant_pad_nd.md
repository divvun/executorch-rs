# kernels/portable/cpu/op_constant_pad_nd.cpp

> [spec:et:def:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn]
> void apply_padding_to_dim( KernelRuntimeContext& ctx, size_t ndim, const CTYPE* self_data, IntArrayRef self_sizes, IntArrayRef self_strides, CTYPE* out_data, CTYPE* out_data_end, IntArrayRef out_sizes, IntArrayRef out_strides, IntArrayRe...

> [spec:et:sem:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn]
> Recursively copies `self` into `out` while inserting the pad fill value before
> and after each dimension, walking from the outermost dim `dim` toward the
> innermost. `out_step_len = out_strides[dim]`, `in_step_len = self_strides[dim]`
> (strides here are element counts = trailing-dim products). `out_data_end` is a
> one-past-the-end guard pointer for the out buffer. `last_padded_dim` is the
> deepest dim that has any nonzero padding.
>
> Steps for one invocation at `dim`:
> 1. If `dim >= ndim`, return (base of recursion).
> 2. Determine this dim's padding: `pad_i = ndim - 1 - dim`. If
>    `pad_i < pad.size()/2`, read `pb = pad[2*pad_i]` (before) and
>    `pa = pad[2*pad_i+1]` (after); ET_KERNEL_CHECK_MSG both must be >= 0
>    ("Padding values must be non-negative.") — on failure set
>    Error::InvalidArgument on the context and return (void). Otherwise
>    `pad_before = pad_after = 0`.
> 3. Leading padding: if `pad_before > 0`, bounds-check that `out_data <=
>    out_data_end` and (using division to avoid overflow) that the remaining
>    space `out_data_end - out_data` divided by `out_step_len` is >= pad_before;
>    on failure set Error::InvalidArgument and return. Then for each of
>    `pad_before` iterations, `set_all_to_value(out_data, out_step_len, value)`
>    per `[spec:et:sem:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn]`
>    and advance `out_data += out_step_len`.
> 4. Body:
>    - If `dim >= last_padded_dim` (no deeper dim is padded), the whole remaining
>      block is contiguous in both tensors: `copy_len = in_step_len *
>      self_sizes[dim]`. If `copy_len > 0`: ET_KERNEL_CHECK_MSG that `out_data`
>      and `self_data` regions do not overlap and that out has room
>      (`out_data <= out_data_end` and remaining >= copy_len); on failure set
>      Error::InvalidArgument and return. Then memcpy `copy_len` elements from
>      `self_data` to `out_data`, and advance both pointers by `copy_len`.
>    - Else recurse: for each i in [0, self_sizes[dim]) call
>      `apply_padding_to_dim(..., dim+1)`; after each recursive call, if
>      `ctx.failure_state() != Error::Ok` return early (propagating an error set
>      deeper in the recursion); otherwise advance `out_data += out_step_len` and
>      `self_data += in_step_len`.
> 5. Trailing padding: if `pad_after > 0`, apply the same bounds checks as step 3
>    (set Error::InvalidArgument and return on failure), then write `pad_after`
>    runs of the fill value, advancing `out_data` by `out_step_len` each time.

> [spec:et:def:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn]
> void constant_pad_nd_out_impl( KernelRuntimeContext& ctx, const Tensor& self, IntArrayRef pad, CTYPE value_v, Tensor& out)

> [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn]
> Sets up geometry and drives the recursive pad copy for one CTYPE.
>
> Steps:
> 1. Get `self_data` (const) and `out_data` (mutable) pointers; `ndim = self.dim()`.
> 2. Zero-dim special case: if `ndim == 0`, copy the single scalar
>    `out_data[0] = self_data[0]` and return.
> 3. For each dim i in [0, ndim): record `self_sizes[i] = self.size(i)`,
>    `self_strides[i] = getTrailingDims(self, i)` (element count of all dims after
>    i), `out_sizes[i] = out.size(i)`, `out_strides[i] = getTrailingDims(out, i)`.
>    While iterating, compute `pad_i = ndim - 1 - i`; if `pad_i < pad.size()/2`
>    and `pad[2*pad_i] + pad[2*pad_i+1] > 0`, set `last_padded_dim = i`. (Because
>    i ascends, `last_padded_dim` ends as the deepest padded dim, or 0 if none.)
> 4. Wrap the size/stride arrays as IntArrayRefs of length `ndim`. Set
>    `out_data_end = out_data + out.numel()`.
> 5. Call `apply_padding_to_dim(ctx, ndim, self_data, self_sizes, self_strides,
>    out_data, out_data_end, out_sizes, out_strides, pad, value_v,
>    last_padded_dim, 0)` per
>    `[spec:et:sem:op-constant-pad-nd.torch.executor.native.apply-padding-to-dim-fn]`,
>    which performs the copy-with-padding starting at dim 0. Errors are reported
>    through the context's failure state.

> [spec:et:def:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn]
> void set_all_to_value(CTYPE* out_data, size_t step_len, CTYPE value)

> [spec:et:sem:op-constant-pad-nd.torch.executor.native.set-all-to-value-fn]
> Fills `step_len` consecutive elements starting at `out_data` with `value`:
> `for i in [0, step_len): out_data[i] = value`. Used to write a contiguous run
> of the pad fill value. Pure write, no bounds checking of its own (callers
> guarantee bounds).

> [spec:et:def:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn]
> Tensor& constant_pad_nd_out( KernelRuntimeContext& ctx, const Tensor& in, IntArrayRef pad, const Scalar& value, Tensor& out)

> [spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-fn]
> Implements `constant_pad_nd.out`: pads `in` with the constant scalar `value`
> according to the `pad` list (pairs of before/after per trailing dimension) and
> writes to `out`. The context is used only for checks.
>
> Steps:
> 1. ET_KERNEL_CHECK: `check_constant_pad_args(in, pad, value, out)` (pad length
>    is even and does not exceed 2*in.dim(), dtype compatibility, etc.). On
>    failure set Error::InvalidArgument and return `out` unchanged.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 3. Resize `out` for the padded shape via `resize_constant_pad_output(in, pad,
>    out)` (each dim's output size = input size + pad_before + pad_after for the
>    corresponding pad pair, unpadded dims unchanged); if it does not return
>    Error::Ok set Error::InvalidArgument (message "Failed to resize output
>    tensor.") and return `out` unchanged.
> 4. Dispatch on `in.scalar_type()` over REALHBBF16 = {Byte, Char, Short, Int,
>    Long, Half, Float, Double, Bool, BFloat16}; other dtypes set
>    Error::InvalidArgument and return `out` unchanged.
> 5. Cast `value` to CTYPE with overflow checking
>    (`utils::internal::check_overflow_scalar_cast<CTYPE>(value)`);
>    ET_KERNEL_CHECK that the cast is representable (has_value) — on overflow set
>    Error::InvalidArgument and return (`out` unchanged).
> 6. Call `constant_pad_nd_out_impl<CTYPE>(ctx, in, pad, value_casted, out)` per
>    `[spec:et:sem:op-constant-pad-nd.torch.executor.native.constant-pad-nd-out-impl-fn]`.
> 7. Return `out`.

