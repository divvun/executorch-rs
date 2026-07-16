# kernels/portable/cpu/op_expand_copy.cpp

> [spec:et:def:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn]
> size_t map_expand_to_repeats( executorch::aten::ArrayRef<SizesType> self_sizes, executorch::aten::ArrayRef<int64_t> expand_sizes, int64_t* repeats, const size_t repeats_size)

> [spec:et:sem:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn]
> Converts an `expand` target-size vector into an equivalent per-dimension `repeat` count vector
> (used to drive `repeat_tensor`). Writes `expand_sizes.size()` entries into `repeats[]` and returns
> that count. `repeats_size` is the buffer capacity (assumed large enough).
> Right-aligns `self_sizes` against `expand_sizes` (expand may add leading dims). Walk from the
> trailing (rightmost) dimension toward the front:
> 1. Let `j = expand_sizes.size()`. For `i` descending from `self_sizes.size()` while `i > 0 && j > 0`:
>    decrement both `i` and `j`; set `repeats[j] = expand_sizes[j]` by default, but if
>    `expand_sizes[j] == -1` (keep existing size) OR `expand_sizes[j] == self_sizes[i]`, set
>    `repeats[j] = 1` (no repetition needed since the dim already matches / is preserved).
> 2. For any remaining leading expand dims (`j > 0` after the aligned pass): decrement `j` and set
>    `repeats[j] = expand_sizes[j]` (these are brand-new dims with no corresponding `self` dim, so
>    the full expand size becomes the repeat count).
> 3. Return `expand_sizes.size()`.

> [spec:et:def:op-expand-copy.torch.executor.native.expand-copy-out-fn]
> Tensor& expand_copy_out( KernelRuntimeContext& ctx, const Tensor& self, ArrayRef<int64_t> expand_sizes, bool implicit, Tensor& out)

> [spec:et:sem:op-expand-copy.torch.executor.native.expand-copy-out-fn]
> Implements `expand_copy.out(self, expand_sizes, implicit, *, out)`: materializes a broadcast/expand
> of `self` to `expand_sizes` into a fresh contiguous `out`. Every failure path sets the error on
> `ctx` and returns `out` unchanged.
> 1. ET_KERNEL_CHECK `check_expand_copy_args(self, expand_sizes, implicit, out)` (validates the expand
>    is legal: rank not shrinking, each non-(-1) target dim either equals self's dim or self's dim
>    is 1, etc.), else InvalidArgument.
> 2. Resolve `-1` entries in `expand_sizes` to the corresponding `self` dim sizes via
>    `get_expand_copy_out_target_size(self.sizes(), expand_sizes, output_sizes, &output_rank)`;
>    ET_KERNEL_CHECK its success, else InvalidArgument.
> 3. ET_KERNEL_CHECK `resize_tensor(out, {output_sizes, output_rank}) == Error::Ok`, else InvalidArgument.
> 4. ET_KERNEL_CHECK `tensors_have_same_dim_order(self, out)`, else InvalidArgument.
> 5. ET_KERNEL_CHECK `tensor_is_default_dim_order(self)`, else InvalidArgument.
> 6. Convert `expand_sizes` to per-dim repeat counts via `map_expand_to_repeats(self.sizes(),
>    expand_sizes, repeats, kTensorDimensionLimit)` per
>    `[spec:et:sem:op-expand-copy.torch.executor.native.map-expand-to-repeats-fn]`, yielding
>    `repeats_size` counts.
> 7. ET_KERNEL_CHECK `repeat_tensor(self, makeArrayRef(repeats, repeats_size), out) == Error::Ok`
>    (tiles `self` per the repeat counts into `out`), else InvalidArgument.
> 8. Return `out`. (Dtype-agnostic: `repeat_tensor` copies raw elements of any dtype.)

