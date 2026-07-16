# runtime/executor/tensor_parser.h

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
> NamedData* get_data_by_key(const char* key, Span<NamedData> entries)

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
> Declares the lookup helper implemented in tensor_parser_exec_aten.cpp; see
> `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]`
> for the full behavior.
>
> Iterate over `entries` in ascending index order from 0 to `entries.size()-1`.
> For each entry compare its `key` field against the requested `key` using C
> string comparison (byte-for-byte until a NUL terminator; equal when strcmp
> returns 0). On the first match return a pointer to that entry (a mutable
> `NamedData*` into the span's backing storage). If no entry matches, return
> nullptr. Both `key` and every `entries[i].key` are assumed NUL-terminated;
> the function does not bounds-check the strings. An empty span always yields
> nullptr.

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.named-data]
> struct NamedData {
>   const char* key;
>   FreeableBuffer buffer;
> }

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
> ET_NODISCARD Result<BoxedEvalueList<executorch::aten::Tensor>> parseTensorList(

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
> Declares `parseTensorList`, implemented in tensor_parser_exec_aten.cpp; see
> `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]`
> for the full behavior.
>
> Builds a `BoxedEvalueList<Tensor>` from a flatbuffer vector of int32 indices
> into an already-populated `values` EValue array of length `values_len`, using
> the method allocator on `memory_manager`. Allocates two parallel arrays of
> length `tensor_indices->size()`: an array of `EValue*` pointers and an array of
> `Tensor` objects; on any allocation failure returns
> `Error::MemoryAllocationFailed`. For each index (in vector order): validate
> `0 <= index < values_len` (else `Error::InvalidProgram`), call `tryToTensor()`
> on that EValue (propagating its error on failure), placement-new-move the tensor
> into the tensor array slot, and store `&values[index]` into the pointer array
> slot. Returns the resulting boxed list. See
> `[spec:et:def:evalue.boxed-evalue-list]` for boxed-list semantics.

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-std.optional-t-parse-list-optional-type-fn]
> ET_NODISCARD Result<BoxedEvalueList<std::optional<T>>> parseListOptionalType(

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-std.optional-t-parse-list-optional-type-fn]
> Template function (defined inline in the header) that deserializes a list of
> optional `T` (e.g. `optional<Tensor>`, `optional<float>`) into a
> `BoxedEvalueList<std::optional<T>>`. Same algorithm for every `T`, hence the
> template.
>
> Inputs: `value_indices` (flatbuffer vector of int32), the already-populated
> `values` EValue array of length `values_len`, and `memory_manager`.
>
> Steps:
> 1. From the method allocator allocate `evalp_list`, an array of `EValue*` of
>    length `value_indices->size()`; if allocation returns null, return
>    `Error::MemoryAllocationFailed`.
> 2. From the method allocator allocate `optional_tensor_list`, an array of
>    `std::optional<T>` of the same length; if null, return
>    `Error::MemoryAllocationFailed`.
> 3. Set `output_idx = 0`. Iterate over `value_indices` in vector order. For each
>    `index`:
>    - If `index == -1` (the serialized sentinel for None): placement-new an
>      empty `std::optional<T>` (nullopt) into `optional_tensor_list[output_idx]`,
>      and set `evalp_list[output_idx] = nullptr`. (A null pointer here is later
>      converted to nullopt by the boxed list.)
>    - Otherwise: validate `index >= 0 && index < values_len`, else return
>      `Error::InvalidProgram` ("Invalid value index ... for ListOptional").
>      Call `values[index].tryToOptional<T>()`; on error propagate that error.
>      Placement-new-move the resulting `std::optional<T>` into
>      `optional_tensor_list[output_idx]`, and set `evalp_list[output_idx] =
>      &values[index]`.
>    - Increment `output_idx`.
> 4. Return `BoxedEvalueList<std::optional<T>>(evalp_list, optional_tensor_list,
>    value_indices->size())`.
>
> Placement-new (not assignment) is used because the freshly allocated list
> elements are uninitialized and `T` may be non-trivial. See
> `[spec:et:def:evalue.boxed-evalue-list]`.

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-executorch.aten.tensor-parse-tensor-fn]
> ET_NODISCARD Result<executorch::aten::Tensor> parseTensor(

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-executorch.aten.tensor-parse-tensor-fn]
> Declaration only; `parseTensor` has two mutually exclusive implementations
> selected at build time by the tensor mode. In portable/ETensor mode the body is
> `[spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn]`;
> in ATen mode it is
> `[spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.parse-tensor-fn]`.
>
> Both variants deserialize one `executorch_flatbuffer::Tensor` (`s_tensor`) into
> a runtime `Tensor`: they validate `storage_offset == 0`, validate the scalar
> type, read `sizes` and `dim_order` (validating rank <= `kTensorDimensionLimit`
> and that `dim_order` size equals the number of sizes), compute strides via
> `dim_order_to_stride`, construct a data-less tensor to learn its byte size, then
> resolve the data pointer through
> `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]`
> and attach it. `named_data_map` and `external_constants` (defaulting to null /
> empty) are forwarded to the data-pointer resolution to satisfy external
> mutable and external constant tensors respectively. On any check failure the
> corresponding non-Ok `Error` is returned as the `Result`.

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
> ET_NODISCARD Result<void*> getTensorDataPtr(

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
> Declares `getTensorDataPtr`, implemented in tensor_parser_exec_aten.cpp; see
> `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]`
> for the full behavior.
>
> Returns the data pointer to use for `s_tensor`, given the `program` (for
> constant-buffer data and mutable subsegments), the tensor's computed `nbytes`,
> the planned-memory `allocator`, and optional `named_data_map` /
> `external_constants` for external data. A tensor is classified by
> `data_buffer_idx` and `allocation_info`:
> - external constant: returns the pointer from `external_constants` matched by
>   fully-qualified name;
> - external mutable: validates layout against the named_data_map layout, gets a
>   memory-planned pointer, and loads the external data into it;
> - constant in PTE (`data_buffer_idx > 0`, no `allocation_info`): returns the
>   program constant-buffer pointer;
> - memory-planned with initial state (`data_buffer_idx > 0` + `allocation_info`):
>   gets a planned pointer and loads the mutable subsegment into it;
> - memory-planned no initial state (`data_buffer_idx == 0` + `allocation_info`):
>   returns just the planned pointer;
> - input/placeholder (both zero/null): returns nullptr (pointer supplied at
>   runtime).
> Any resolution error is returned as a non-Ok `Error`.

> [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
> ET_NODISCARD Error validateTensorLayout( const executorch_flatbuffer::Tensor* s_tensor, const TensorLayout& expected_layout)

> [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
> Declares `validateTensorLayout`, implemented in tensor_parser_exec_aten.cpp;
> see
> `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]`
> for the full behavior.
>
> Checks that the `scalar_type`, `sizes` and `dim_order` of the serialized
> `s_tensor` match those of the `expected_layout` (from the external data map),
> returning `Error::Ok` on full agreement and `Error::InvalidExternalData` on the
> first mismatch or missing field.

