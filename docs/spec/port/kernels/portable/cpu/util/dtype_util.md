# kernels/portable/cpu/util/dtype_util.cpp, kernels/portable/cpu/util/dtype_util.h

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]
> bool check_tensor_dtype( const Tensor t, SupportedTensorDtypes dtypes, const ScalarType compute_type)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]
> Returns true iff tensor `t`'s scalar type is acceptable for the given
> `dtypes` category, where `compute_type` is the compute scalar type of
> the operation. Dispatches on `dtypes`:
> - REALHBBF16: true iff `t`'s dtype is a "real HBBF16" type, i.e. one
>   of the real dtypes {Byte, Char, Short, Int, Long, Float, Double}
>   plus Half, BFloat16, and Bool (the H, B for Bool, BF16 additions);
>   delegates to `executorch::runtime::tensor_is_realhbbf16_type(t)`.
> - REALHBF16: true iff `t`'s dtype is a "real HBF16" type, i.e. one of
>   the real dtypes {Byte, Char, Short, Int, Long, Float, Double} plus
>   Half and BFloat16 (no Bool); delegates to
>   `tensor_is_realhbf16_type(t)`.
> - FLOATHBF16: true iff `t`'s dtype is a floating type, i.e. one of
>   {Double, Float, Half, BFloat16}; delegates to
>   `tensor_is_floating_type(t)`.
> - INTB: true iff `t`'s dtype is an integral type including Bool, i.e.
>   one of {Byte, Char, Short, Int, Long, Bool}; delegates to
>   `tensor_is_integral_type(t, /*includeBool=*/true)`.
> - BOOL: true iff `t`'s dtype is exactly Bool; delegates to
>   `tensor_is_type(t, ScalarType::Bool)`.
> - BOOL_OR_BYTE: true iff `t`'s dtype is Bool or Byte; delegates to
>   `tensor_is_type(t, ScalarType::Bool, ScalarType::Byte)`.
> - SAME_AS_COMPUTE: true iff `t`'s dtype equals `compute_type`;
>   delegates to `tensor_is_type(t, compute_type)`.
> - SAME_AS_COMMON: if `compute_type == ScalarType::Float`, true iff
>   `t`'s dtype is one of {Float, Half, BFloat16}; otherwise true iff
>   `t`'s dtype equals `compute_type`.
>
> The `dtypes` enum is total, so all cases are covered; the trailing
> `ET_CHECK(false)` after the switch is unreachable and only guards
> against an out-of-range enum value (which would abort).

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]
> void convert_and_store(From f, void* dst)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]
> Template over `<To, From>`. Takes a value `f` of type `From` and a raw
> destination pointer `dst`. Converts `f` to type `To` using a C++
> `static_cast<To>(f)` and writes the result into `*dst` (treating `dst`
> as a `To*`). No bounds/alignment checks; `dst` must point to writable
> storage for one `To`. The `static_cast` conversion follows C++
> numeric conversion rules for the concrete `From`/`To`: integer<->
> float truncates toward zero, out-of-range float->integer is UB in
> C++ but in practice used only for in-range values, float narrowing
> (e.g. Float->Half/BFloat16) rounds to nearest representable, and
> bool conversions map nonzero to true / zero to false. Inverse of
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_bool( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `load_to_compute_fn<CTYPE_COMPUTE>`. Initialize `result = nullptr`.
> Test `t.scalar_type()` directly (no ET_SWITCH):
> - if it is NOT `ScalarType::Bool`, call
>   `context.fail(Error::InvalidArgument)`, log Error "Unhandled dtype
>   <name> for <op_name>", and leave `result == nullptr`;
> - otherwise set `result = load_and_convert<CTYPE_COMPUTE, bool>`
>   (loads a `bool` element and converts to the compute type per
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]`).
> Return `result`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-or-byte-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_bool_or_byte( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-bool-or-byte-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `load_to_compute_fn<CTYPE_COMPUTE>`. Initialize `result = nullptr`,
> then dispatch over exactly two types Bool and Byte via
> `ET_SWITCH_TWO_TYPES(Bool, Byte, t.scalar_type(), context, op_name,
> TENSOR_CTYPE, ...)`. If `t.scalar_type()` is Bool or Byte, set
> `result = load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>` (TENSOR_CTYPE
> = bool or uint8 respectively). For any other dtype, the ET_SWITCH
> default fails the context with `Error::InvalidArgument`, logs the
> unhandled-dtype message, and leaves `result == nullptr`. Return
> `result`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-floathbf16-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_floathbf16( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-floathbf16-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]`
> except it dispatches over the FLOATHBF16 dtype set — the floating
> types {Double, Float, Half, BFloat16} only — via
> `ET_SWITCH_FLOATHBF16_TYPES`. For an accepted floating
> `t.scalar_type()`, sets `result = load_and_convert<CTYPE_COMPUTE,
> TENSOR_CTYPE>`; for any integral/bool type, fails the context with
> `Error::InvalidArgument`, logs the unhandled-dtype message, and
> returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn( KernelRuntimeContext& context, const Tensor& t, SupportedTensorDtypes dtypes)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Public entry point for
> obtaining a load function; forwards `(context, t, dtypes)` to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn]`.
> The `op_name` template argument passed to the impl depends on the
> selective-build mode:
> - if `EXECUTORCH_SELECTIVE_BUILD_DTYPE` is defined, the caller's real
>   `op_name` is forwarded (so dtype-selective build can strip
>   per-operator dtype specializations);
> - otherwise a single shared constant name `kGenericElementwiseOpName`
>   (= "generic_elementwise_op") is used for all operators, so every
>   operator shares one template instantiation and binary size is
>   reduced.
> Behavior (which loader is returned, failure/null semantics) is
> otherwise exactly that of the impl. A Rust port with no selective
> build simply forwards to the impl unconditionally.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_impl( KernelRuntimeContext& context, const Tensor& t, SupportedTensorDtypes dtypes)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-impl-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Given `context`, tensor `t`,
> and a `SupportedTensorDtypes dtypes` value, returns the appropriate
> `load_to_compute_fn<CTYPE_COMPUTE>` by switching on `dtypes` and
> delegating to the matching per-category getter with `<CTYPE_COMPUTE,
> op_name>`:
> - REALHBBF16 -> `get_load_to_compute_fn_realhbbf16`
> - REALHBF16 -> `get_load_to_compute_fn_realhbf16`
> - FLOATHBF16 -> `get_load_to_compute_fn_realhbf16` (NOTE: the load
>   path intentionally reuses the REALHBF16 getter for FLOATHBF16;
>   loading integral inputs to a floating compute type is harmless, so
>   the wider accept set is used on load. This differs from the store
>   path in
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]`,
>   which uses the true FLOATHBF16 getter.)
> - INTB -> `get_load_to_compute_fn_intb`
> - BOOL -> `get_load_to_compute_fn_bool`
> - BOOL_OR_BYTE -> `get_load_to_compute_fn_bool_or_byte`
> - SAME_AS_COMPUTE -> `get_load_to_compute_fn_same_as_compute`
> - SAME_AS_COMMON -> `get_load_to_compute_fn_same_as_common`
> Each delegate may fail the context and return `nullptr` if `t`'s dtype
> is not in the category (see the individual rules). The `dtypes` enum
> is total; the post-switch `ET_CHECK(false)` is unreachable and aborts
> only on an out-of-range enum value.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-intb-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_intb( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-intb-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]`
> except it dispatches over the integral-plus-Bool dtype set — the
> integer types {Byte, Char, Short, Int, Long} plus Bool — via
> `ET_SWITCH_INT_TYPES_AND(Bool, ...)`. For an accepted `t.scalar_type()`,
> sets `result = load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>`; for any
> floating type (Half/BFloat16/Float/Double), fails the context with
> `Error::InvalidArgument`, logs the unhandled-dtype message, and
> returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_realhbbf16( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Given `context` and tensor
> `t`, returns a `load_to_compute_fn<CTYPE_COMPUTE>` (function pointer
> `CTYPE_COMPUTE(*)(const void*)`) that loads one element of `t` and
> converts it to `CTYPE_COMPUTE`, selected by `t.scalar_type()`.
>
> Initialize `result = nullptr`. Dispatch over the REALHBBF16 dtype set
> — the real types {Byte(uint8), Char(int8), Short(int16), Int(int32),
> Long(int64), Float, Double} plus Half, Bool, and BFloat16 — via
> `ET_SWITCH_REALHBBF16_TYPES(t.scalar_type(), context, op_name,
> TENSOR_CTYPE, ...)`. If `t.scalar_type()` is one of those types, set
> `result = load_and_convert<CTYPE_COMPUTE, TENSOR_CTYPE>` (the loader
> per
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]`
> specialized to that element type). If `t.scalar_type()` is not in the
> set, the ET_SWITCH default path calls `context.fail(Error::
> InvalidArgument)`, logs "Unhandled dtype <name> for <op_name>", and
> leaves `result == nullptr`. Returns `result` (a null pointer signals
> the caller that dispatch failed and the context is already in error).

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbf16-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_realhbf16( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbf16-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-realhbbf16-fn]`
> except it dispatches over the REALHBF16 dtype set — the real types
> {Byte, Char, Short, Int, Long, Float, Double} plus Half and BFloat16
> (NOT Bool) — via `ET_SWITCH_REALHBF16_TYPES`. For an accepted
> `t.scalar_type()`, sets `result = load_and_convert<CTYPE_COMPUTE,
> TENSOR_CTYPE>`; for Bool or any unaccepted type, fails the context
> with `Error::InvalidArgument`, logs the unhandled-dtype message, and
> returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-common-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_same_as_common( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-common-fn]
> Template over `<CTYPE_COMPUTE, op_name>`, with two overloads selected
> by whether `CTYPE_COMPUTE` is `float`:
> - When `CTYPE_COMPUTE == float`: initialize `result = nullptr`, then
>   dispatch over exactly three types Float, Half, BFloat16 via
>   `ET_SWITCH_THREE_TYPES(Float, Half, BFloat16, t.scalar_type(),
>   context, op_name, T, ...)`. If `t.scalar_type()` is Float, Half, or
>   BFloat16, set `result = load_and_convert<CTYPE_COMPUTE, T>` (loading
>   Half/BFloat16 promotes to float). For any other dtype, the ET_SWITCH
>   default fails the context with `Error::InvalidArgument`, logs the
>   unhandled-dtype message, leaving `result == nullptr`. Return
>   `result`.
> - When `CTYPE_COMPUTE != float`: delegate directly to
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn]`
>   with the same `<CTYPE_COMPUTE, op_name>` (i.e. accept only the exact
>   compute dtype).
>
> This mirrors the SAME_AS_COMMON acceptance rule in
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.check-tensor-dtype-fn]`:
> a float compute type also accepts Half/BFloat16 inputs.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn]
> load_to_compute_fn<CTYPE_COMPUTE> get_load_to_compute_fn_same_as_compute( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-load-to-compute-fn-same-as-compute-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `load_to_compute_fn<CTYPE_COMPUTE>`. Initialize `result = nullptr`.
> Compute `common_scalar_type = CppTypeToScalarType<CTYPE_COMPUTE>::value`
> (the ScalarType corresponding to the compute C++ type). Test
> `t.scalar_type()` directly:
> - if it does NOT equal `common_scalar_type`, call
>   `context.fail(Error::InvalidArgument)`, log the unhandled-dtype
>   message, leave `result == nullptr`;
> - otherwise set `result = load_and_convert<CTYPE_COMPUTE,
>   CTYPE_COMPUTE>` (an identity load: element type equals compute
>   type, so the `static_cast` is a no-op copy).
> Return `result`. Only accepts a tensor whose dtype is exactly the
> compute type.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn_bool( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `store_compute_to_tensor_fn<CTYPE_COMPUTE>` (function pointer
> `void(*)(CTYPE_COMPUTE, void*)`) that converts a compute-typed value
> and stores it into a `t`-typed tensor element. Initialize `result =
> nullptr`. Test `t.scalar_type()` directly:
> - if it is NOT `ScalarType::Bool`, call
>   `context.fail(Error::InvalidArgument)`, log "Unhandled dtype <name>
>   for <op_name>", leave `result == nullptr`;
> - otherwise set `result = convert_and_store<bool, CTYPE_COMPUTE>` (per
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]`,
>   casts the compute value to `bool` — nonzero->true, zero->false — and
>   writes it).
> Return `result`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-or-byte-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-bool-or-byte-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `store_compute_to_tensor_fn<CTYPE_COMPUTE>`. Initialize `result =
> nullptr`, then dispatch over exactly Bool and Byte via
> `ET_SWITCH_TWO_TYPES(Bool, Byte, t.scalar_type(), context, op_name,
> TENSOR_CTYPE, ...)`. If `t.scalar_type()` is Bool or Byte, set
> `result = convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>` (cast the
> compute value to bool/uint8 and write). For any other dtype, the
> ET_SWITCH default fails the context with `Error::InvalidArgument`,
> logs the unhandled-dtype message, and leaves `result == nullptr`.
> Return `result`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-floathbf16-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-floathbf16-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]`
> except it dispatches over the FLOATHBF16 dtype set — floating types
> {Double, Float, Half, BFloat16} only — via `ET_SWITCH_FLOATHBF16_TYPES`.
> For an accepted floating dtype sets `result =
> convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>`; for any integral/bool
> dtype fails the context with `Error::InvalidArgument`, logs the
> unhandled-dtype message, and returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn( KernelRuntimeContext& context, const Tensor& t, SupportedTensorDtypes dtypes)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Given `context`, tensor `t`,
> and `SupportedTensorDtypes dtypes`, returns the appropriate
> `store_compute_to_tensor_fn<CTYPE_COMPUTE>` by switching on `dtypes`
> and delegating to the matching per-category store getter with
> `<CTYPE_COMPUTE, op_name>`:
> - REALHBBF16 -> `get_store_compute_to_tensor_fn_realhbbf16`
> - REALHBF16 -> `get_store_compute_to_tensor_fn_realhbf16`
> - FLOATHBF16 -> `get_store_compute_to_tensor_fn_floathbf16` (the true
>   FLOATHBF16 getter, unlike the load dispatcher which substitutes the
>   REALHBF16 getter)
> - INTB -> `get_store_compute_to_tensor_fn_intb`
> - BOOL -> `get_store_compute_to_tensor_fn_bool`
> - BOOL_OR_BYTE -> `get_store_compute_to_tensor_fn_bool_or_byte`
> - SAME_AS_COMPUTE -> `get_store_compute_to_tensor_fn_same_as_compute`
> - SAME_AS_COMMON -> `get_store_compute_to_tensor_fn_same_as_common`
> Each delegate may fail the context and return `nullptr` if `t`'s dtype
> is not in the category. Unlike the load path, there is no separate
> `op_name`-substitution wrapper: this function is called directly with
> the caller's `op_name`. The `dtypes` enum is total; the post-switch
> `ET_CHECK(false)` is unreachable.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-intb-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE> get_store_compute_to_tensor_fn_intb( KernelRuntimeContext& context, const Tensor& t)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-intb-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]`
> except it dispatches over the integral-plus-Bool dtype set — integer
> types {Byte, Char, Short, Int, Long} plus Bool — via
> `ET_SWITCH_INT_TYPES_AND(Bool, ...)`. For an accepted dtype sets
> `result = convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>` (float->int
> stores truncate toward zero); for any floating dtype fails the context
> with `Error::InvalidArgument`, logs the unhandled-dtype message, and
> returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `store_compute_to_tensor_fn<CTYPE_COMPUTE>`. Initialize `result =
> nullptr`, then dispatch over the REALHBBF16 dtype set — real types
> {Byte, Char, Short, Int, Long, Float, Double} plus Half, Bool, and
> BFloat16 — via `ET_SWITCH_REALHBBF16_TYPES(t.scalar_type(), context,
> op_name, TENSOR_CTYPE, ...)`. For an accepted `t.scalar_type()`, set
> `result = convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>` (cast the
> compute value to the tensor element type and write, per
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]`).
> For an unaccepted dtype, the ET_SWITCH default fails the context with
> `Error::InvalidArgument`, logs the unhandled-dtype message, leaving
> `result == nullptr`. Return `result`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbf16-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbf16-fn]
> Identical structure to
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-realhbbf16-fn]`
> except it dispatches over the REALHBF16 dtype set — real types {Byte,
> Char, Short, Int, Long, Float, Double} plus Half and BFloat16 (NOT
> Bool) — via `ET_SWITCH_REALHBF16_TYPES`. For an accepted dtype sets
> `result = convert_and_store<TENSOR_CTYPE, CTYPE_COMPUTE>`; for Bool or
> any unaccepted dtype fails the context with `Error::InvalidArgument`,
> logs the unhandled-dtype message, and returns `nullptr`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-common-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-common-fn]
> Template over `<CTYPE_COMPUTE, op_name>`, with two overloads selected
> by whether `CTYPE_COMPUTE` is `float`:
> - When `CTYPE_COMPUTE == float`: initialize `result = nullptr`, then
>   dispatch over exactly three types Float, Half, BFloat16 via
>   `ET_SWITCH_THREE_TYPES(Float, Half, BFloat16, t.scalar_type(),
>   context, op_name, CTYPE, ...)`. If `t.scalar_type()` is Float, Half,
>   or BFloat16, set `result = convert_and_store<CTYPE, CTYPE_COMPUTE>`
>   (storing to Half/BFloat16 narrows the float, rounding to nearest).
>   For any other dtype the ET_SWITCH default fails the context with
>   `Error::InvalidArgument`, logs the unhandled-dtype message, leaving
>   `result == nullptr`. Return `result`.
> - When `CTYPE_COMPUTE != float`: delegate directly to
>   `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn]`
>   with the same `<CTYPE_COMPUTE, op_name>` (accept only the exact
>   compute dtype).

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn]
> store_compute_to_tensor_fn<CTYPE_COMPUTE>

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.get-store-compute-to-tensor-fn-same-as-compute-fn]
> Template over `<CTYPE_COMPUTE, op_name>`. Returns a
> `store_compute_to_tensor_fn<CTYPE_COMPUTE>`. Initialize `result =
> nullptr`. Compute `common_scalar_type =
> CppTypeToScalarType<CTYPE_COMPUTE>::value`. Test `t.scalar_type()`
> directly:
> - if it does NOT equal `common_scalar_type`, call
>   `context.fail(Error::InvalidArgument)`, log the unhandled-dtype
>   message, leave `result == nullptr`;
> - otherwise set `result = convert_and_store<CTYPE_COMPUTE,
>   CTYPE_COMPUTE>` (identity store: element type equals compute type,
>   so the cast is a no-op copy).
> Return `result`. Only accepts a tensor whose dtype is exactly the
> compute type.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]
> To load_and_convert(const void* fromPtr)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.load-and-convert-fn]
> Template over `<To, From>`. Takes a raw source pointer `fromPtr`,
> reinterprets it as a `const From*`, loads the single `From` value at
> `*fromPtr`, converts it to `To` via a C++ `static_cast<To>(...)`, and
> returns the converted value. No bounds/alignment checks; `fromPtr`
> must point to a readable `From`. Conversion follows the same C++
> numeric `static_cast` rules described in
> `[spec:et:sem:dtype-util.torch.executor.native.utils.internal.convert-and-store-fn]`.
> Used to build a `load_to_compute_fn<CTYPE_COMPUTE>` (a function
> pointer `CTYPE_COMPUTE(*)(const void*)`) that loads one tensor
> element of dtype `From` and promotes it to the compute type `To`.

> [spec:et:def:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn]
> inline constexpr ScalarType specialized_output_scalar_type( SupportedTensorDtypes out_dtypes)

> [spec:et:sem:dtype-util.torch.executor.native.utils.internal.specialized-output-scalar-type-fn]
> Template over `<CTYPE_COMPUTE>`; `constexpr`. Given the output tensor's
> supported-dtypes category `out_dtypes`, returns the single ScalarType
> the elementwise machinery is willing to generate specialized
> (dtype-monomorphized) code for:
> - BOOL -> `ScalarType::Bool`
> - BOOL_OR_BYTE -> `ScalarType::Bool` (only Bool is specialized; Byte
>   outputs fall through to the generic path)
> - REALHBBF16, REALHBF16, FLOATHBF16, INTB, SAME_AS_COMPUTE,
>   SAME_AS_COMMON -> `CppTypeToScalarType<CTYPE_COMPUTE>::value` (i.e.
>   the compute type itself).
> Used by
> `[spec:et:sem:elementwise-util.torch.executor.native.utils.internal.apply-elementwise-fn-fn]`
> to decide whether the actual output tensor dtype matches the one
> specialized case worth a fast path. The enum is total; there is no
> post-switch return (all cases return), so an out-of-range value is
> undefined.

> [spec:et:def:dtype-util.torch.executor.native.utils.supported-tensor-dtypes]
> enum class SupportedTensorDtypes {
>   REALHBBF16;
>   REALHBF16;
>   FLOATHBF16;
>   INTB;
>   BOOL;
>   BOOL_OR_BYTE;
>   SAME_AS_COMPUTE;
>   SAME_AS_COMMON;
> }

