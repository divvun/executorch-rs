# kernels/portable/cpu/op_clamp.cpp

> [spec:et:def:op-clamp.torch.executor.native.check-bounds-fn]
> ET_NODISCARD bool check_bounds( KernelRuntimeContext& ctx, const Scalar& val_scalar, const torch::executor::native::ScalarType& val_type, const torch::executor::native::ScalarType& out_type, const char* val_name)

> [spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn]
> Validates that a clamp bound scalar `val_scalar` fits in the output dtype
> `out_type`. Returns a bool (true = valid) and logs an error on failure. It does
> not itself set any context error; the caller wraps it in ET_KERNEL_CHECK.
> `val_type` and `val_name` are used only for messaging.
>
> Steps:
> - If `out_type` is an integral type excluding Bool: read the scalar as
>   `int64_t` (`utils::scalar_to<int64_t>`), then dispatch `out_type` over the
>   integer types and check `is_out_of_bounds<CTYPE_OUT, int64_t>(val_long)`
>   per `[spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn]`. If
>   out of bounds, log "<val_name> value out of bounds" and set the result
>   invalid.
> - Else if `out_type` is a floating type: dispatch `out_type` over FLOATHBF16,
>   read the scalar as `double`, and only if the value is finite
>   (`std::isfinite`) check `is_out_of_bounds<CTYPE_OUT, double>(val_double)`;
>   if out of bounds, log the message and set the result invalid. Non-finite
>   (inf/NaN) bounds are accepted for floating outputs.
> - Otherwise (e.g. Bool output) no bound check is applied; the result stays
>   valid.
> - Return the accumulated validity flag.

> [spec:et:def:op-clamp.torch.executor.native.is-out-of-bounds-fn]
> bool is_out_of_bounds(CTYPE_CAST val_cast)

> [spec:et:sem:op-clamp.torch.executor.native.is-out-of-bounds-fn]
> Templated on `CTYPE_OUT` (target dtype) and `CTYPE_CAST` (the type the value is
> already held as). Returns true iff `val_cast` lies outside the representable
> range of `CTYPE_OUT`, i.e.
> `val_cast < std::numeric_limits<CTYPE_OUT>::lowest()` OR
> `val_cast > std::numeric_limits<CTYPE_OUT>::max()`. The comparison is performed
> in `CTYPE_CAST`. Pure function, no side effects.

> [spec:et:def:op-clamp.torch.executor.native.clamp-out-fn]
> Tensor& clamp_out( KernelRuntimeContext& ctx, const Tensor& in, const std::optional<Scalar>& min_opt, const std::optional<Scalar>& max_opt, Tensor& out)

> [spec:et:sem:op-clamp.torch.executor.native.clamp-out-fn]
> Elementwise clamp with scalar bounds:
> `out = min(max(in, min_opt), max_opt)`, with each bound applied only if
> present.
>
> Steps:
> 1. `has_min = min_opt.has_value()`, `has_max = max_opt.has_value()`.
>    ET_KERNEL_CHECK_MSG: at least one of them must be present ("At least one of
>    'min' or 'max' must not be None"); otherwise set Error::InvalidArgument and
>    return `out` unchanged.
> 2. Determine dtypes: `in_type`, `out_type`, and (for present bounds) each
>    bound's scalar dtype via `utils::get_scalar_dtype`; absent bounds default
>    to `in_type`.
> 3. Compute `common_type` starting from `in_type`, promoting with each present
>    scalar bound via `utils::promote_type_with_scalar`.
> 4. ET_KERNEL_CHECK: `common_type == out_type` (exact equality, not just
>    castable); on failure set Error::InvalidArgument and return `out` unchanged.
> 5. For each present bound, ET_KERNEL_CHECK `check_bounds(...)` per
>    `[spec:et:sem:op-clamp.torch.executor.native.check-bounds-fn]` (the bound
>    must fit `out_type`); on failure set Error::InvalidArgument and return `out`
>    unchanged.
> 6. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, out)`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 7. Resize `out` to `in.sizes()`; on failure set Error::InvalidArgument and
>    return `out` unchanged.
> 8. `compute_type = utils::get_compute_type(common_type)`; dispatch over
>    REALB = {Byte, Char, Short, Int, Long, Float, Double, Bool}
>    (ET_SWITCH_REALB_TYPES); other compute types set Error::InvalidArgument and
>    return `out` unchanged.
> 9. Elementwise over `in` (unitensor helper, input loaded from REALHBBF16,
>    output written as SAME_AS_COMMON): starting from `val_in`, if `has_min`
>    apply `utils::max_override(val_out, scalar_to<CTYPE_COMPUTE>(min))`; if
>    `has_max` apply `utils::min_override(val_out, scalar_to<CTYPE_COMPUTE>(max))`.
>    `max_override`/`min_override` are NaN-propagating max/min. Store the result
>    to `out`.
> 10. Return `out`.

> [spec:et:def:op-clamp.torch.executor.native.clamp-tensor-out-fn]
> Tensor& clamp_tensor_out( KernelRuntimeContext& ctx, const Tensor& in, const std::optional<Tensor>& min_opt, const std::optional<Tensor>& max_opt, Tensor& out)

> [spec:et:sem:op-clamp.torch.executor.native.clamp-tensor-out-fn]
> Elementwise clamp with tensor bounds and broadcasting:
> `out = min(max(in, min), max)`, each bound applied only if present.
>
> Steps:
> 1. `has_min = min_opt.has_value()`, `has_max = max_opt.has_value()`.
>    ET_KERNEL_CHECK_MSG: at least one must be present; otherwise set
>    Error::InvalidArgument and return `out` unchanged.
> 2. Let `min = has_min ? min_opt.value() : in` and
>    `max = has_max ? max_opt.value() : in` (absent bounds alias `in` so the
>    tritensor helper always has three real tensors, but the lambda ignores the
>    aliased bound).
> 3. Compute `common_type` from `in.scalar_type()`, promoting with
>    `min.scalar_type()` if `has_min` and `max.scalar_type()` if `has_max`
>    (`promoteTypes`).
> 4. ET_KERNEL_CHECK: `canCast(common_type, out.scalar_type())`; on failure set
>    Error::InvalidArgument and return `out` unchanged.
> 5. ET_KERNEL_CHECK: `tensors_have_same_dim_order(in, min, max, out)`; on
>    failure set Error::InvalidArgument and return `out` unchanged.
> 6. Resize `out` to the broadcast of `in`, `min`, `max`
>    (`resize_to_broadcast_target_size`); if it does not return Error::Ok set
>    Error::InvalidArgument and return `out` unchanged.
> 7. `compute_type = utils::get_compute_type(common_type)`; dispatch over REALB =
>    {Byte, Char, Short, Int, Long, Float, Double, Bool}; other compute types set
>    Error::InvalidArgument and return `out` unchanged.
> 8. Tritensor elementwise over the broadcasted (in, min, max), all loaded from
>    REALHBBF16, output written as REALHBBF16: starting from `val_in`, if
>    `has_min` apply `utils::max_override(val_out, val_min)`; if `has_max` apply
>    `utils::min_override(val_out, val_max)` (NaN-propagating). Store to `out`.
> 9. Return `out`.

