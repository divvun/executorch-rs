# runtime/executor/tensor_parser_exec_aten.cpp

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
> NamedData* get_data_by_key(const char* key, Span<NamedData> entries)

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
> Linear search of `entries` for the entry whose `key` matches the requested
> `key`. Iterate `i` from 0 to `entries.size()-1` (ascending). For each entry
> compare `entries[i].key` against `key` with C string comparison (`strcmp == 0`,
> i.e. byte-for-byte until a NUL terminator). On the first match, return a
> pointer to that entry (`&entries[i]`, mutable, into the span's backing store).
> If no entry matches, return nullptr. An empty span returns nullptr. Both the
> query key and stored keys are assumed NUL-terminated; no bounds checking is
> performed on the strings.

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
> ET_NODISCARD Result<BoxedEvalueList<executorch::aten::Tensor>> parseTensorList(

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
> Builds a `BoxedEvalueList<Tensor>` referencing tensors that already live inside
> the `values` EValue array. Inputs: `tensor_indices` (flatbuffer vector of
> int32), the populated `values` array of length `values_len`, and
> `memory_manager`.
>
> Steps:
> 1. Open a profiling scope named "TensorParser::parseTensorList".
> 2. From the method allocator allocate `tensor_list`, an array of `Tensor` of
>    length `tensor_indices->size()`; if null, return
>    `Error::MemoryAllocationFailed`.
> 3. From the method allocator allocate `evalp_list`, an array of `EValue*` of
>    the same length; if null, return `Error::MemoryAllocationFailed`.
> 4. Set `output_idx = 0`. Iterate over `tensor_indices` in vector order. For
>    each `tensor_index`:
>    - Require `tensor_index >= 0 && tensor_index < values_len`, else
>      `Error::InvalidProgram` ("Invalid value index ... for TensorList").
>    - Call `values[tensor_index].tryToTensor()`; on error propagate that error.
>    - Placement-new-move the resulting `Tensor` into `tensor_list[output_idx]`
>      (placement-new because the slot is uninitialized and `Tensor` is
>      non-trivial).
>    - Set `evalp_list[output_idx] = &values[tensor_index]`.
>    - Increment `output_idx`.
> 5. Return `BoxedEvalueList<Tensor>(evalp_list, tensor_list,
>    tensor_indices->size())`. See `[spec:et:def:evalue.boxed-evalue-list]`.

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-mem-planned-ptr-fn]
> ET_NODISCARD Result<void*> getMemPlannedPtr(

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-mem-planned-ptr-fn]
> Resolves the memory-planned data pointer described by `allocation_info` from a
> `HierarchicalAllocator`. File-local helper. Inputs: `allocation_info` (from the
> serialized tensor), the tensor's `nbytes`, and `allocator`.
>
> Steps:
> 1. Require `allocator != nullptr`, else `Error::InvalidState`
>    ("HierarchicalAllocator must not be null for memory-planned tensor").
> 2. Compute `memory_id = allocation_info->memory_id() - 1`. The `-1` un-biases
>    the serialized id: id 0 was historically reserved, so serialized ids are
>    1-based and the allocator uses 0-based hierarchy ids.
> 3. Reconstruct the 64-bit memory offset from two 32-bit fields:
>    `memory_offset_low = allocation_info->memory_offset_low()` and
>    `memory_offset_high = allocation_info->memory_offset_high()`. If `size_t` is
>    wider than 32 bits (64-bit platform), `memory_offset = memory_offset_low |
>    (size_t(memory_offset_high) << 32)`. On a 32-bit platform, require
>    `memory_offset_high == 0` (else `Error::NotSupported`, "size_t cannot hold
>    memory offset ...") and use `memory_offset = memory_offset_low`.
> 4. Return `allocator->get_offset_address(memory_id, memory_offset, nbytes)`
>    (see `[spec:et:sem:hierarchical-allocator.get-offset-address-fn]`), which
>    yields the pointer into arena `memory_id` at `memory_offset`, validating that
>    `nbytes` fits, or a non-Ok `Error` which is propagated.

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
> ET_NODISCARD Result<void*> getTensorDataPtr(

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
> Returns the data pointer for `s_tensor`, dispatching on how the tensor's data is
> stored. Inputs: `s_tensor`, `program`, the tensor's computed `nbytes`, the
> planned-memory `allocator`, and optional `named_data_map` /
> `external_constants`.
>
> Read `data_buffer_idx = s_tensor->data_buffer_idx()` and `allocation_info =
> s_tensor->allocation_info()`. If `allocation_info != nullptr`, require
> `allocator != nullptr` up front, else `Error::InvalidState`
> ("HierarchicalAllocator is null but tensor has allocation_info ...").
>
> Then, in order:
>
> 1. External tensor: if `s_tensor->extra_tensor_info() != nullptr` AND its
>    `location() == TensorDataLocation::EXTERNAL`:
>    - Require the `fully_qualified_name()` is non-null, else
>      `Error::InvalidExternalData` ("Fully qualified name of external tensor is
>      null"). Let `fqn` be its C string.
>    - If `allocation_info == nullptr` (external constant): call
>      `get_data_by_key(fqn, external_constants)` (see
>      `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]`).
>      If found, return `const_cast<void*>(data->buffer.data())`. If not found,
>      return `Error::Internal` (these should have been resolved earlier in
>      `Method::parse_external_constants`).
>    - Else (external mutable): require `named_data_map != nullptr`, else
>      `Error::InvalidExternalData` ("Cannot retrieve external tensor with fqn:
>      ... The named_data_map is null; most likely no external .ptd file was
>      provided."). Call `named_data_map->get_tensor_layout(fqn)`; propagate its
>      error if not ok. Call `validateTensorLayout(s_tensor, tensor_layout)` (see
>      `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]`);
>      if it returns non-Ok, return that error. Resolve the destination via
>      `getMemPlannedPtr(allocation_info, nbytes, allocator)` (see
>      `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-mem-planned-ptr-fn]`);
>      propagate its error. Call `named_data_map->load_data_into(fqn,
>      planned_ptr, nbytes)`; if non-Ok return that error. Return the planned
>      pointer.
> 2. Else if `data_buffer_idx > 0 && allocation_info == nullptr` (constant stored
>    in the PTE): call `program->get_constant_buffer_data(data_buffer_idx,
>    nbytes)`; propagate its error, otherwise return
>    `const_cast<void*>(const_data.get())` (const_cast is safe because this data
>    is never modified at runtime).
> 3. Else if `data_buffer_idx > 0 && allocation_info != nullptr` (memory-planned
>    with an initial state): resolve `planned_ptr` via `getMemPlannedPtr(...)`
>    (propagate error), then call
>    `TensorParser::load_mutable_subsegment_into(program,
>    mutable_data_segments_index=0, offset_index=s_tensor->data_buffer_idx(),
>    size=nbytes, buffer=planned_ptr)` (see
>    `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn]`);
>    if non-Ok return that error, else return `planned_ptr`.
> 4. Else if `data_buffer_idx == 0 && allocation_info != nullptr` (memory-planned,
>    no initial state): return `getMemPlannedPtr(allocation_info, nbytes,
>    allocator)` directly.
> 5. Else (`data_buffer_idx == 0 && allocation_info == nullptr`, an
>    input/placeholder tensor): return nullptr — the data pointer is supplied at
>    runtime.

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser]
> class TensorParser final

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn]
> ET_NODISCARD static Error load_mutable_subsegment_into( const Program* program, size_t mutable_data_segments_index, size_t offset_index, size_t size, void* buffer)

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.tensor-parser.load-mutable-subsegment-into-fn]
> Thin static forwarder that exists solely to give the free functions in this
> translation unit access to a private `Program` method (the class
> `TensorParser` is declared a friend of `Program`). It takes `program`,
> `mutable_data_segments_index`, `offset_index`, `size`, and a destination
> `buffer` pointer, and returns `program->load_mutable_subsegment_into(
> mutable_data_segments_index, offset_index, size, buffer)` verbatim, propagating
> that `Error` unchanged. It performs no validation or transformation of its own;
> the actual copy of the mutable subsegment bytes into `buffer` is defined by
> `[spec:et:sem:program.load-mutable-subsegment-into-fn]`. In Rust, where there is
> no friend-access restriction, this indirection collapses to a direct call to the
> program's load-mutable-subsegment routine.

> [spec:et:def:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
> ET_NODISCARD Error validateTensorLayout( const executorch_flatbuffer::Tensor* s_tensor, const TensorLayout& expected_layout)

> [spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
> Verifies that the serialized tensor `s_tensor` agrees with the `expected_layout`
> obtained from the external `NamedDataMap`, so that external data can be loaded
> into a memory-planned buffer safely. Returns `Error::Ok` when everything matches;
> on the first failing check returns `Error::InvalidExternalData` (each check is an
> `ET_CHECK_OR_RETURN_ERROR`, which on failure logs and returns that error
> immediately). Checks are performed in this exact order:
>
> 1. Scalar type: require `static_cast<ScalarType>(s_tensor->scalar_type()) ==
>    expected_layout.scalar_type()`. (Note: the diagnostic message's "Expected"/
>    "got" labels are swapped relative to the operands, but the comparison itself
>    is a plain equality of the two scalar-type enum values.)
> 2. Require `s_tensor->sizes() != nullptr` ("Missing sizes field").
> 3. Require `s_tensor->dim_order() != nullptr` ("Missing dim_order field").
> 4. Let `dim = s_tensor->sizes()->size()` (an `int`). Require `dim >= 0`
>    ("Dim is negative").
> 5. Require `static_cast<size_t>(dim) == expected_layout.sizes().size()`
>    (rank must match the expected layout's number of sizes).
> 6. Require `s_tensor->dim_order()->size() == static_cast<size_t>(dim)`
>    (serialized dim_order length must equal the number of sizes).
> 7. For `i` from 0 to `dim-1` (ascending), for each `i` in this single loop
>    require both:
>    - `s_tensor->sizes()->Get(i) == expected_layout.sizes()[i]`, and
>    - `s_tensor->dim_order()->Get(i) == expected_layout.dim_order()[i]`.
>    The size check for a given `i` is evaluated before the dim_order check for the
>    same `i`; the first mismatch (in this element-then-field order) returns.
> 8. If all checks pass, return `Error::Ok`.
>
> No promotion or normalization is applied: comparisons are exact integer/enum
> equality against the layout the external `.ptd` producer recorded. See
> `[spec:et:def:tensor-layout]`.

