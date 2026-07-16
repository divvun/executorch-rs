# kernels/portable/cpu/util/elementwise_util.h

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]
> inline void apply_bitensor_elementwise_fn( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& a, SupportedTensorDtypes a_dtypes, const Tensor& b, SupportedTensorDtypes b_dtypes, const Tensor& out, SupportNoncontiguousInputTe...

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-bitensor-elementwise-fn-fn]
> Public convenience wrapper for binary (two-input) elementwise
> operators. Template over `<CTYPE_COMPUTE, op_name, out_dtypes, Op>`.
> The annotated overload takes a trailing `SupportNoncontiguousInputTensors`
> tag and forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]`
> with `<CTYPE_COMPUTE, op_name, out_dtypes,
> support_noncontiguous_tensors=true>`, passing `(compute_fun, ctx, out,
> std::make_pair(&a, a_dtypes), std::make_pair(&b, b_dtypes))` — two
> input tensors `a`, `b` each paired with its supported-dtype set, in
> that order.
>
> Sibling overloads (not separately annotated): a tag-free overload
> forwards with `support_noncontiguous_tensors=false`; a DEPRECATED
> runtime-`out_dtypes` overload forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]`.
> For each output element, `compute_fun(a_elem, b_elem)` is evaluated
> with both operands converted to `CTYPE_COMPUTE` and the result
> converted to `out`'s dtype; broadcasting between `a`, `b`, and `out`
> is applied. See [NOTE: Generic lambdas] for the `Op` contract.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]
> inline void apply_elementwise_fn( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& out, Args... inputs)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]
> Template over `<CTYPE_COMPUTE, op_name, out_dtypes,
> support_noncontiguous_tensors, Op, Args...>` where `out_dtypes` is a
> compile-time `SupportedTensorDtypes`. Primary apply driver. Steps:
> - Run
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]`
>   for `<CTYPE_COMPUTE>` on `(ctx, out, out_dtypes, inputs...)`. If
>   false, return immediately without writing `out` (`ctx` holds
>   `InvalidArgument`).
> - `compute_type = CppTypeToScalarType<CTYPE_COMPUTE>::value`;
>   `out_specialized_scalar_type =
>   specialized_output_scalar_type<CTYPE_COMPUTE>(out_dtypes)` per
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn]`.
> - Fast path (only if `should_include_kernel_dtype(op_name,
>   out_specialized_scalar_type)` — a selective-build gate that is true
>   unless dtype-selective build excluded that dtype): if every input
>   tensor's `scalar_type()` equals `compute_type` AND `out`'s
>   `scalar_type()` equals `out_specialized_scalar_type`, then with
>   `CTYPE_OUT = ScalarTypeToCppType<out_specialized_scalar_type>::type`
>   call
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]`
>   with `<CTYPE_COMPUTE, CTYPE_OUT, support_noncontiguous_tensors>` and
>   `(compute_fun, ctx, out, inputs...)`, then return.
> - Slow path (otherwise): call
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]`
>   with `<CTYPE_COMPUTE, op_name, support_noncontiguous_tensors>` and
>   `(compute_fun, ctx, out, out_dtypes, inputs...)`.
> Both paths produce identical numeric results; the fast path only
> avoids per-element dtype conversion when all dtypes already match. Does
> not resize `out`.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]
> inline void apply_elementwise_fn_generic_impl( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& out, SupportedTensorDtypes out_dtypes, Args... inputs)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]
> Template over `<CTYPE_COMPUTE, op_name, support_noncontiguous_tensors,
> Op, Args...>`. The general (non-dtype-specialized) elementwise loop:
> it loads each input via a dtype-erased loader that converts to
> `CTYPE_COMPUTE`, applies `compute_fun`, and stores the result via a
> dtype-erased storer that converts to the output tensor's dtype. Steps:
> - `kNumInputs = sizeof...(inputs)`.
> - For each input pair `(tensor*, dtypes)`, build an `InputInfo`
>   holding: (a) `load_to_compute` = the loader function pointer from
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn]`
>   for `<CTYPE_COMPUTE, op_name>` given `(ctx, *tensor, dtypes)`; (b)
>   `data_ptr` = raw byte pointer to the tensor's data; (c)
>   `element_size` = the tensor's element size in bytes.
> - Obtain `store_compute_to_out` = the storer function pointer from
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]`
>   for `<CTYPE_COMPUTE, op_name>` given `(ctx, out, out_dtypes)`, and
>   record `out`'s raw byte pointer and `out_element_size`.
>   (If any input dtype or the out dtype was unsupported these getters
>   would have failed `ctx` and returned null; this generic impl assumes
>   validation already passed — its callers run
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]`
>   first.)
> - Iterate `[0, out.numel())` in parallel via
>   `executorch::extension::parallel_for` (grain `GRAIN_SIZE`). Each task
>   builds a `BroadcastIndexesRange<kNumInputs,
>   support_noncontiguous_tensors>(out, *inputs.first...)` per
>   `[spec:et:sem:broadcast-indexes-range.broadcast-indexes-range]`,
>   advances its iterator by `begin`, and loops while the flat output
>   index `indexes[0] < end`. For each position: for each input `i`,
>   compute the source byte address `data_ptr + indexes[i+1] *
>   element_size` and call `load_to_compute` to obtain a `CTYPE_COMPUTE`
>   value; apply `compute_fun` to the loaded inputs (argument order =
>   input order); call `store_compute_to_out(result, data_out +
>   indexes[0] * out_element_size)` to convert and write the element.
> Broadcasting and (optional) stride handling are entirely delegated to
> the BroadcastIndexesRange. Does not resize `out`. NaN/inf propagate as
> `compute_fun` defines.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]
> inline void apply_elementwise_fn_runtime_out_dtypes( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& out, SupportedTensorDtypes out_dtypes, Args... inputs)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]
> Template over `<CTYPE_COMPUTE, op_name, Op, Args...>`. Variant of the
> apply driver used by the DEPRECATED overloads that receive
> `out_dtypes` as a runtime argument rather than a template parameter,
> so it cannot compute a compile-time output specialization. Steps:
> - Run
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]`
>   for `<CTYPE_COMPUTE>` on `(ctx, out, out_dtypes, inputs...)`. If it
>   returns false (some dtype unsupported; `ctx` already carries
>   `InvalidArgument`), return immediately without writing `out`.
> - Otherwise call
>   `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-generic-impl-fn]`
>   with `<CTYPE_COMPUTE, op_name, support_noncontiguous_tensors=false>`,
>   passing `(compute_fun, ctx, out, out_dtypes, inputs...)`.
> Always uses the generic (non-specialized) loop and always treats
> inputs as contiguous. Does not resize `out`.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn]
> inline void apply_tritensor_elementwise_fn( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& a, SupportedTensorDtypes a_dtypes, const Tensor& b, SupportedTensorDtypes b_dtypes, const Tensor& c, SupportedTensorDtypes c_dtyp...

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-tritensor-elementwise-fn-fn]
> Public convenience wrapper for ternary (three-input) elementwise
> operators. Template over `<CTYPE_COMPUTE, op_name, out_dtypes, Op>`.
> The annotated overload takes a trailing `SupportNoncontiguousInputTensors`
> tag and forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]`
> with `<CTYPE_COMPUTE, op_name, out_dtypes,
> support_noncontiguous_tensors=true>`, passing `(compute_fun, ctx, out,
> std::make_pair(&a, a_dtypes), std::make_pair(&b, b_dtypes),
> std::make_pair(&c, c_dtypes))` — three input tensors `a`, `b`, `c`
> each paired with its supported-dtype set, in that order. To mitigate
> build-time blowup, all three inputs are passed to `compute_fun` as
> `CTYPE_COMPUTE` regardless of their tensor dtypes.
>
> Sibling overloads (not separately annotated): a tag-free overload
> forwards with `support_noncontiguous_tensors=false`; a DEPRECATED
> runtime-`out_dtypes` overload forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]`.
> For each output element, `compute_fun(a_elem, b_elem, c_elem)` is
> evaluated and the result stored to `out` with broadcasting across all
> four tensors. `op_name` must be a `static constexpr const char[]`
> (C++17 cannot bind a string literal to the template param). See
> [NOTE: Generic lambdas] for the `Op` contract.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]
> inline void apply_unitensor_elementwise_fn( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& a, SupportedTensorDtypes a_dtypes, const Tensor& out, SupportNoncontiguousInputTensors)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-unitensor-elementwise-fn-fn]
> Public convenience wrapper for unary elementwise operators. Template
> over `<CTYPE_COMPUTE, op_name, out_dtypes, Op>`. The annotated
> overload takes a trailing `SupportNoncontiguousInputTensors` tag,
> which selects stride-respecting iteration. It forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]`
> with `<CTYPE_COMPUTE, op_name, out_dtypes,
> support_noncontiguous_tensors=true>`, passing `(compute_fun, ctx, out,
> std::make_pair(&a, a_dtypes))` — i.e. exactly one input tensor `a`
> with its supported-dtype set `a_dtypes`.
>
> Two sibling overloads (not separately annotated) exist: one without
> the tag forwards with `support_noncontiguous_tensors=false`; one
> DEPRECATED overload that takes `out_dtypes` as a runtime argument
> forwards to
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-runtime-out-dtypes-fn]`.
> For each output element, `compute_fun` is called with the single
> broadcast-mapped input element (as `CTYPE_COMPUTE`) and the result is
> converted to `out`'s dtype and stored. See [NOTE: Generic lambdas] in
> the source for the SFINAE/vectorization contract on `Op`.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn]
> constexpr bool can_use_vectorized()

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn]
> Compile-time (`constexpr`) predicate, template over `<CTYPE_COMPUTE,
> Op, Args...>`, only defined when building with PyTorch headers
> (`ET_USE_PYTORCH_HEADERS`). Let `Vec = at::vec::Vectorized<
> CTYPE_COMPUTE>`. Returns true iff the compute functor `Op` can be
> invoked with `sizeof...(Args)` arguments each of type `Vec` AND the
> result of that invocation is itself exactly `Vec`:
> - First test `std::is_invocable_v<Op, Vec, Vec, ...>` (one `Vec` per
>   entry in `Args...`). If not invocable, return false.
> - If invocable, additionally require
>   `std::is_same_v<std::invoke_result_t<Op, Vec, ...>, Vec>` and return
>   that. The result-type check is required because `Vectorized` is
>   implicitly convertible to a pointer and hence to bool, which would
>   otherwise make even a scalar-`bool`-returning `Op` spuriously appear
>   invocable-with-Vec; requiring the result be `Vec` rejects that.
> Purely a code-generation gate: it decides whether the SIMD fast path
> in
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]`
> is instantiated. When PyTorch headers are absent, the vectorized path
> does not exist at all and this predicate is effectively false. A Rust
> port targeting scalar semantics may treat it as always false.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]
> inline void dtype_specialized_elementwise_fn_impl( const Op& compute_fun, KernelRuntimeContext& ctx, const Tensor& out, Args... inputs)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.dtype-specialized-elementwise-fn-impl-fn]
> Template over `<CTYPE_COMPUTE, CTYPE_OUT, support_noncontiguous_tensors,
> Op, Args...>`. Fast path used when every input tensor's dtype already
> equals the compute type and the output tensor's dtype equals
> `CTYPE_OUT`; there is therefore no per-element load/store conversion —
> inputs are read directly as `CTYPE_COMPUTE`, results written directly
> as `CTYPE_OUT`. Each entry of `inputs` is a
> `std::pair<const Tensor*, SupportedTensorDtypes>` (only the tensor
> pointer `.first` is used here). `kNumInputs = sizeof...(inputs)`.
> Precondition (debug-asserted via `ET_DCHECK`): every input's
> `scalar_type()` equals `CppTypeToScalarType<CTYPE_COMPUTE>::value`.
>
> Two implementations:
>
> (A) Vectorized path — only compiled with PyTorch headers and only
> taken when
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.can-use-vectorized-fn]`
> is true for `<CTYPE_COMPUTE, Op, Args...>` AND no input is broadcast.
> "No input is broadcast" means for every input,
> `[spec:et:sem:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]`
> holds between that input's `sizes()` and `out.sizes()`. If both hold,
> iterate over `[0, out.numel())` in parallel via
> `executorch::extension::parallel_for` (grain size
> `GRAIN_SIZE`); each task processes a contiguous `[begin, end)` slice
> of flat output indices. Within a slice, using `Vec =
> at::vec::Vectorized<CTYPE_COMPUTE>` and `Vec::size()` lanes:
> gather raw input pointers as `const CTYPE_COMPUTE*` and the output
> pointer as `CTYPE_OUT*`; compute a `vectorized_begin` (the first index
> >= `begin` that is a multiple of `Vec::size()`) and `vectorized_end`
> (largest multiple of `Vec::size()` <= `end`). Process a scalar
> prologue `[begin, vectorized_begin)`, a main SIMD loop
> `[vectorized_begin, vectorized_end)` stepping by `Vec::size()`, and a
> scalar epilogue `[vectorized_end, end)`. In the scalar sections
> element `idx` is computed as `data_out[idx] = compute_fun(input_0[idx],
> input_1[idx], ...)`. In the SIMD loop, load each input with
> `Vec::loadu`, apply `compute_fun` to the `Vec` operands, and
> `store` the resulting `Vec` to `&data_out[idx]`. (In NDEBUG-off/debug
> builds the scalar sections also route through `Vec` with a
> single-element store, purely to exercise the vectorized lambda; the
> numeric result is identical.) After processing, return.
>
> (B) Generic parallel path — always available, taken when the
> vectorized path is unavailable or an input is broadcast. Iterate
> `[0, out.numel())` in parallel via `parallel_for` (grain `GRAIN_SIZE`).
> Each task builds a `BroadcastIndexesRange<kNumInputs,
> support_noncontiguous_tensors>(out, *inputs.first...)` per
> `[spec:et:sem:broadcast-indexes-range.broadcast-indexes-range]`,
> advances its iterator by `begin`, and iterates while the produced flat
> output index `indexes[0] < end`. For each position, `indexes[0]` is the
> flat output offset and `indexes[i+1]` is the flat offset into input
> `i` (already accounting for broadcasting and, if
> `support_noncontiguous_tensors`, strides). Load each input directly as
> `CTYPE_COMPUTE` from `input_i[indexes[i+1]]`, apply `compute_fun`, and
> write the result to `data_out[indexes[0]]` as `CTYPE_OUT` (implicit
> `CTYPE_COMPUTE`->`CTYPE_OUT` conversion on assignment).
>
> Does not resize `out` and does not validate dtypes (the caller
> guarantees them). NaN/inf propagate exactly as `compute_fun` defines.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]
> inline ScalarType get_compute_type(ScalarType& common_type)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.get-compute-type-fn]
> Given a `common_type` ScalarType (the promoted common dtype of the
> operands), return the ScalarType to actually compute in. If
> `common_type` is `Half` or `BFloat16`, return `Float` (16-bit floats
> are computed in fp32 for accuracy); for every other value, return
> `common_type` unchanged. Does not mutate the argument despite the
> reference parameter. This is the runtime companion to the
> compile-time float-widening embodied by the SAME_AS_COMMON rules in
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]`.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors]
> struct SupportNoncontiguousInputTensors

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors.support-noncontiguous-input-tensors-fn]
> explicit SupportNoncontiguousInputTensors() = default

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.support-noncontiguous-input-tensors.support-noncontiguous-input-tensors-fn]
> Defaulted explicit default constructor of the empty tag type
> `SupportNoncontiguousInputTensors`. It has no fields and no behavior:
> constructing one produces an empty value. Its sole purpose is to be
> passed as a trailing argument to the `apply_*_elementwise_fn`
> overloads to select the variant that respects input tensor strides
> (i.e. sets `support_noncontiguous_tensors = true` in the underlying
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]`
> call). `explicit` prevents implicit/brace conversion from `{}`, so the
> caller must name the type. A Rust port models this as a unit
> marker/type, or simply a boolean flag threaded through the apply
> functions.

> [spec:et:def:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]
> inline bool validate_elementwise_fn_inputs( KernelRuntimeContext& ctx, const Tensor& out, SupportedTensorDtypes out_dtypes, Args... inputs)

> [spec:et:sem:elementwise-util.torch.executor.native.utils.internal.validate-elementwise-fn-inputs-fn]
> Template over `<CTYPE_COMPUTE, Args...>`. Validates that every input
> tensor's dtype fits its declared supported-dtype set and that the
> output tensor's dtype fits `out_dtypes`, all relative to the compute
> type. Steps:
> - `compute_type = CppTypeToScalarType<CTYPE_COMPUTE>::value`.
> - For each input pair `(tensor*, dtypes)` in `inputs`, evaluate
>   `check_tensor_dtype(*tensor, dtypes, compute_type)` per
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]`.
> - Also evaluate `check_tensor_dtype(out, out_dtypes, compute_type)`.
> - Combine all checks with logical AND and wrap in `ET_KERNEL_CHECK`
>   with error `InvalidArgument` and failure-return value `false`: if
>   any check is false, set `Error::InvalidArgument` on `ctx` and return
>   `false` immediately (the `&&` short-circuits, but all that matters
>   is the combined result).
> - If all checks pass, return `true`.
> Performs no shape/broadcast checking and does not touch `out`'s data;
> it is purely a dtype gate. Returning `false` signals the caller to
> abort without writing output.

