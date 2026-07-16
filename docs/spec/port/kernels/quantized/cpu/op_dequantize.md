# kernels/quantized/cpu/op_dequantize.cpp

> [spec:et:def:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn]
> void apply_over_unpacked_dim( const Fn& fn, const executorch::aten::Tensor& in, const int64_t& dim)

> [spec:et:sem:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn]
> Iterates a callback `fn(inner_size, outer_idx, unpacked_dim_idx)` over one
> selected dimension `dim` of tensor `in`, treating that dimension as "unpacked".
>
> Steps:
> 1. If `in.numel() == 0`, return immediately (no calls).
> 2. ET_CHECK_MSG that `in.dim() > 0` (abort/fatal on failure).
> 3. ET_CHECK_VALID_DIM(dim, in.dim()): assert `-in.dim() <= dim < in.dim()`
>    (abort on failure).
> 4. Normalize the dimension: `d = dim < 0 ? dim + in.dim() : dim` (ET_NORMALIZE_IX).
> 5. Compute `dim_size = in.size(d)`; `outer_size = getLeadingDims(in, d)` (the
>    product of sizes of all dims strictly before `d`, i.e. 1 if `d == 0`);
>    `inner_size = getTrailingDims(in, d)` (the product of sizes of all dims
>    strictly after `d`, i.e. 1 if `d` is the last dim).
> 6. For `outer_idx` in `[0, outer_size)` (outer loop), then for
>    `unpacked_dim_idx` in `[0, dim_size)` (inner loop), invoke
>    `fn(inner_size, outer_idx, unpacked_dim_idx)`. The callback receives
>    `inner_size` as its first argument (named `numel` at call sites).
>
> Note: the callback is responsible for the actual per-slice work; this function
> only supplies the loop structure and the three indices.

> [spec:et:def:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]
> bool can_use_optimized_dequantize_per_channel( const Tensor& in, const ScalarType in_dtype, std::optional<ScalarType>& out_dtype)

> [spec:et:sem:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]
> Returns whether the NEON/optimized per-channel path may be used.
>
> Steps:
> 1. Determine `is_contiguous`: in ATen mode (`USE_ATEN_LIB`), `in.is_contiguous()`;
>    otherwise it is `is_contiguous_dim_order(in.dim_order().data(), in.dim())`
>    (i.e. the dim order is the identity/row-major order 0,1,2,...).
> 2. Return `false` if any of the following hold:
>    - not contiguous, OR
>    - `in_dtype != ScalarType::Char` (i.e. input is not signed int8), OR
>    - `out_dtype` has a value and that value is not `ScalarType::Float`.
> 3. Otherwise return `true`.
>
> The optimized path therefore requires: contiguous int8 input producing float
> output.

> [spec:et:def:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]
> void check_dequantize_per_tensor_args( const Tensor& input, int64_t quant_min, int64_t quant_max, ScalarType dtype, std::optional<ScalarType>& out_dtype, Tensor& out)

> [spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]
> Validates the arguments common to the dequantize kernels. All checks use
> ET_CHECK_MSG, which aborts (fatal) on failure — it does not set a context error
> or return. Order of checks:
> 1. `input.scalar_type()` must be one of Byte (uint8), Char (int8), Bits16,
>    UInt16 (uint16), Short (int16), or Int (int32). Otherwise abort.
> 2. `input.scalar_type()` must equal the `dtype` argument. Otherwise abort.
> 3. If `out_dtype` has a value, `out.scalar_type()` must equal `out_dtype.value()`.
>    Otherwise abort.
> 4. `quant_min <= quant_max`. Otherwise abort.
>
> Note: `quant_min`/`quant_max` are only range-checked here; they are metadata and
> are not otherwise used to clamp during dequantization.

> [spec:et:def:op-dequantize.torch.executor.native.dequantize-optimized-fn]
> void dequantize_optimized( const int8_t* in, const double scale, const int64_t zero_point, float* out, int64_t quant_min, int64_t quant_max, size_t numel)

> [spec:et:sem:op-dequantize.torch.executor.native.dequantize-optimized-fn]
> Dequantizes `numel` contiguous int8 values from `in` into `float` output `out`
> using a single scalar `scale` (double) and `zero_point` (int64).
>
> Steps:
> 1. ET_CHECK_MSG `zero_point >= quant_min` (abort on failure).
> 2. ET_CHECK_MSG `zero_point <= quant_max` (abort on failure).
> 3. For each element index `i` in `[0, numel)`, compute
>    `out[i] = (in[i] - zero_point) * scale`, where `in[i]` is the int8 value
>    promoted to a wider integer/float, the subtraction is done against
>    `zero_point`, and the product uses `scale` as the (double) multiplier. The
>    result is stored as `float`.
>
> On aarch64/ARM NEON builds this is implemented with a vectorized loop processing
> 16 int8 lanes at a time (subtract zero_point, widen to int32, convert to f32,
> multiply by `float(scale)`), with a scalar remainder loop for the tail; the
> numerical result is equivalent to the scalar formula above (the vector path
> narrows `scale` to `float` before multiplying, matching the scalar path's
> effective float multiply). Unlike the general per-tensor kernel, quant_min/
> quant_max here additionally bound `zero_point`.

> [spec:et:def:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn]
> void dequantize_per_channel_optimized( const Tensor& in, const Tensor& scales, const std::optional<Tensor>& opt_zero_points, Tensor& out, int64_t axis, int64_t quant_min, int64_t quant_max, ScalarType in_dtype, std::optional<ScalarType>&...

> [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn]
> Optimized per-channel dequantization for contiguous int8 input to float output.
> Precondition: caller has confirmed eligibility via
> `[spec:et:sem:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]`.
>
> Steps:
> 1. Call `check_dequantize_per_tensor_args(in, quant_min, quant_max, in_dtype,
>    out_dtype, out)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]`.
> 2. ET_CHECK_MSG `in_dtype == ScalarType::Char` (int8) — abort otherwise.
> 3. If `out_dtype` has a value, ET_CHECK_MSG it equals `ScalarType::Float` —
>    abort otherwise.
> 4. `in_data = in.const_data_ptr<int8_t>()`; `out_data = out.mutable_data_ptr<float>()`.
> 5. `zero_points_data`: if `opt_zero_points` has a value, its
>    `const_data_ptr<int64_t>()`; else `nullptr`.
> 6. `axis_stride = in.strides()[axis]`; `outer_stride = in.size(axis) * axis_stride`.
> 7. Invoke `apply_over_unpacked_dim(fn, in, axis)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.apply-over-unpacked-dim-fn]`.
>    For each `(numel=inner_size, outer_idx, unpacked_dim_idx)` the callback:
>    - computes the local input pointer
>      `in_data + outer_idx*outer_stride + unpacked_dim_idx*axis_stride`;
>    - `scale = get_scale(scales, unpacked_dim_idx)` per
>      `[spec:et:sem:op-dequantize.torch.executor.native.get-scale-fn]` (channel =
>      the unpacked dim index);
>    - `zero_point = zero_points_data ? zero_points_data[unpacked_dim_idx] : 0`;
>    - computes the local output pointer with the same offset formula;
>    - calls `dequantize_optimized(in_local, scale, zero_point, out_local,
>      quant_min, quant_max, numel)` per
>      `[spec:et:sem:op-dequantize.torch.executor.native.dequantize-optimized-fn]`,
>      which additionally asserts `quant_min <= zero_point <= quant_max`.
>
> Effectively each channel slice along `axis` shares one scale and zero_point, and
> the inner `numel` (trailing-dims product) contiguous elements of the slice are
> dequantized together.

> [spec:et:def:op-dequantize.torch.executor.native.get-scale-fn]
> float get_scale(const Tensor& scale, size_t channel_ix)

> [spec:et:sem:op-dequantize.torch.executor.native.get-scale-fn]
> Returns the scale value for a given channel index as a `float`.
>
> Steps:
> 1. ET_CHECK_MSG that `scale.scalar_type()` is `Double` or `Float` (abort
>    otherwise).
> 2. If `Double`: read `scale.const_data_ptr<double>()[channel_ix]` and narrow it
>    to `float` (static_cast).
> 3. Else (`Float`): return `scale.const_data_ptr<float>()[channel_ix]` directly.
>
> No bounds checking on `channel_ix` is performed here.

> [spec:et:def:op-dequantize.torch.executor.dequantize-per-channel-out-fn]
> Tensor& dequantize_per_channel_out( const Tensor& input, const Tensor& scale, const std::optional<Tensor>& opt_zero_points, int64_t axis, int64_t quant_min, int64_t quant_max, ScalarType dtype, std::optional<ScalarType> out_dtype, Tensor...

> [spec:et:sem:op-dequantize.torch.executor.dequantize-per-channel-out-fn]
> Dequantizes `input` per channel along `axis`: each channel index along `axis`
> has its own scale (and optional zero_point). Result stored into `out`; `out` is
> returned. This is the non-context overload; the context overload (below, same
> rule id) first resizes `out` to `input.sizes()` (ET_CHECK_MSG on failure) then
> delegates here.
>
> Steps:
> 1. ET_CHECK_MSG `tensor_has_dim(input, axis)`, i.e.
>    `-input.dim() <= axis < input.dim()` (abort otherwise).
> 2. Normalize axis: if `axis < 0`, `axis += nonzero_dim(input)` (nonzero_dim =
>    number of dims, treating a 0-dim tensor as rank 1).
> 3. ET_CHECK_MSG `scale.numel() == input.size(axis)` (abort otherwise).
> 4. If `opt_zero_points` has a value:
>    - ET_CHECK_MSG its scalar_type is `Int` (int32) or `Long` (int64);
>    - ET_CHECK_MSG its `numel() == input.size(axis)`.
> 5. Call `check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype,
>    out_dtype, out)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]`.
> 6. If `can_use_optimized_dequantize_per_channel(input, dtype, out_dtype)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.can-use-optimized-dequantize-per-channel-fn]`
>    is true, call `dequantize_per_channel_optimized(...)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-channel-optimized-fn]`
>    and return `out`.
> 7. Otherwise use the generic path. Build `dims[]`, the list of all input
>    dimension indices except `axis`, in ascending order (for i in
>    `[0, input.dim()-1)`: `dims[i] = i` if `i < axis` else `i+1`).
> 8. `zero_point_data = opt_zero_points ? const_data_ptr<int64_t>() : nullptr`.
>    Note: even though step 4 accepts Int or Long zero_points, the generic path
>    reads them as int64_t here (assumes Long layout).
> 9. Dispatch on `input.scalar_type()` (accepted set: Byte, Char, Short, Int
>    (`ET_FORALL_INT_TYPES`), plus Bits16 and UInt16 both read as uint16_t) and
>    then on `out.scalar_type()` (accepted set from `ET_FORALL_FLOATH_TYPES_WITH`:
>    Float, Double, Half; other out dtypes abort via ET_CHECK_MSG). For the chosen
>    (CTYPE_IN, CTYPE_OUT):
>    - If `input.dim() == 1`: ET_CHECK_MSG `axis == 0`; call `apply_over_dim` over
>      the whole tensor; for each flat index `current_ix`, compute
>      `_scale = get_scale(scale, current_ix)`,
>      `zero_point = zero_point_data ? zero_point_data[current_ix] : 0`, and
>      `out[current_ix] = CTYPE_OUT(input[current_ix] - int32_t(zero_point)) * _scale`
>      (subtraction performed in the input's promoted integer type, cast to
>      CTYPE_OUT, then multiplied by scale as float).
>    - Else (multi-dim): for each `channel_ix` in `[0, input.size(axis))`, load
>      `_scale = get_scale(scale, channel_ix)` and
>      `_zero_point = zero_point_data ? zero_point_data[channel_ix] : 0`, then use
>      `apply_over_dim_list` over `optional_dim_list` (= all dims except axis)
>      pinned at `channel_ix`; for each visited flat index `in_ix`:
>      `out[in_ix] = CTYPE_OUT((input[in_ix] - int32_t(_zero_point)) * _scale)`.
> 10. Return `out`.
>
> Dequantization formula per element: `(quantized - zero_point) * scale`,
> selecting scale/zero_point by the element's position along `axis`. quant_min/
> quant_max are not used to clamp in the generic path.

> [spec:et:def:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]
> Tensor& dequantize_per_tensor_out( KernelRuntimeContext& context, const Tensor& input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, std::optional<ScalarType> out_dtype, Tensor& out)

> [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]
> Dequantizes `input` with a single scalar `scale` (double) and `zero_point`
> (int64) into `out`; returns `out`. Formula per element:
> `(input - zero_point) * float(scale)`.
>
> Steps:
> 1. `resize_tensor(out, input.sizes())`; ET_CHECK_MSG the result is `Error::Ok`
>    (abort otherwise). `out` ends up with `input`'s shape.
> 2. `check_dequantize_per_tensor_args(input, quant_min, quant_max, dtype,
>    out_dtype, out)` per
>    `[spec:et:sem:op-dequantize.torch.executor.native.check-dequantize-per-tensor-args-fn]`.
> 3. Dispatch on `input.scalar_type()` (accepted: Byte, Char, Short, Int via
>    `ET_FORALL_INT_TYPES`, plus Bits16 and UInt16 both handled as uint16_t; any
>    other input dtype aborts via ET_CHECK_MSG), then on `out.scalar_type()`
>    (accepted from `ET_FORALL_FLOATH_TYPES_WITH`: Float, Double, Half; other out
>    dtypes abort).
> 4. For the chosen (IN_CTYPE, OUT_CTYPE): iterate flat index `i` over
>    `[0, input.numel())` and set
>    `out[i] = OUT_CTYPE((input[i] - int32_t(zero_point)) * float(scale))`. The
>    subtraction is `input[i]` (promoted) minus `zero_point` cast to int32; the
>    product multiplies by `scale` narrowed to `float` (matching fbgemm
>    behavior); the result is cast to OUT_CTYPE.
> 5. Return `out`.
>
> quant_min/quant_max are validated only (step 2) and are not used to clamp.

> [spec:et:def:op-dequantize.torch.executor.dequantize-per-tensor-tensor-args-out-fn]
> Tensor& dequantize_per_tensor_tensor_args_out( const Tensor& input, const Tensor& scale, const Tensor& zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, std::optional<ScalarType> out_dtype, Tensor& out)

> [spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-tensor-args-out-fn]
> Variant of per-tensor dequantize where `scale` and `zero_point` are supplied as
> single-element tensors. Returns `out`.
>
> Steps:
> 1. ET_CHECK_MSG `scale.scalar_type() == ScalarType::Double` (abort otherwise).
> 2. ET_CHECK_MSG `zero_point.scalar_type() == ScalarType::Long` (int64) (abort
>    otherwise).
> 3. ET_CHECK_MSG `scale.numel() == 1` (abort otherwise).
> 4. ET_CHECK_MSG `zero_point.numel() == 1` (abort otherwise).
> 5. Call `dequantize_per_tensor_out(input, scale.const_data_ptr<double>()[0],
>    zero_point.const_data_ptr<int64_t>()[0], quant_min, quant_max, dtype,
>    out_dtype, out)` per
>    `[spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]`.
> 6. Return `out`.
>
> A context wrapper overload with the same behavior exists that ignores the
> `KernelRuntimeContext` and forwards to this function.

> [spec:et:def:op-dequantize.torch.executor.dequantize-per-token-out-fn]
> Tensor& dequantize_per_token_out( const Tensor& input, const Tensor& scale, const Tensor& zero_points, int64_t quant_min, int64_t quant_max, ScalarType dtype, ScalarType out_dtype, Tensor& out)

> [spec:et:sem:op-dequantize.torch.executor.dequantize-per-token-out-fn]
> Per-token dequantization: each "token" is the last-dimension vector of `input`,
> and all leading dimensions are collapsed into a channel/token axis. Returns `out`.
>
> Steps:
> 1. Compute `num_channels = product of input.size(i) for i in [0, input.dim()-1)`
>    (product of all sizes except the last dim; 1 if `input.dim() == 1`).
> 2. Form a 2-D logical view `reshaped_input` of shape
>    `[num_channels, input.size(input.dim()-1)]` over `input`'s data (no copy):
>    - ATen mode: `at::from_blob` over `input`'s data with those sizes and
>      `input.scalar_type()`.
>    - Portable mode: build a `TensorImpl` with rank 2, the computed sizes,
>      contiguous dim order {0,1}, strides derived from that dim order, STATIC
>      dynamism, pointing at `input`'s data. Also `resize_tensor(out,
>      input.sizes())` and ET_CHECK_MSG it is `Error::Ok` (this resize happens
>      only in portable mode).
> 3. Call `dequantize_per_channel_out(reshaped_input, scale, zero_points,
>    axis=0, quant_min, quant_max, dtype, out_dtype, out)` per
>    `[spec:et:sem:op-dequantize.torch.executor.dequantize-per-channel-out-fn]`
>    (the non-context overload). So each row (token) uses `scale[row]` and
>    `zero_points[row]`.
> 4. Return `out`.
>
> A context wrapper overload with the same behavior exists that ignores the
> `RuntimeContext` and forwards to this function. Note the per-token op takes a
> non-optional `out_dtype` (ScalarType, not optional), which is passed through as
> the optional `out_dtype` to the per-channel call.

> [spec:et:def:op-dequantize.torch.executor.native.dequantize-per-tensor-out-fn]
> Tensor& dequantize_per_tensor_out( const Tensor& input, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, ScalarType dtype, std::optional<ScalarType> out_dtype, Tensor& out)

> [spec:et:sem:op-dequantize.torch.executor.native.dequantize-per-tensor-out-fn]
> Context-taking overload of per-tensor dequantize. Ignores the
> `KernelRuntimeContext` argument (cast to void) and forwards all remaining
> arguments unchanged to the non-context `dequantize_per_tensor_out` per
> `[spec:et:sem:op-dequantize.torch.executor.dequantize-per-tensor-out-fn]`,
> returning its result (`out`). No additional behavior.

