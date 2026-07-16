# runtime/executor/tensor_parser_portable.cpp

> [spec:et:def:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn]
> Result<Tensor> parseTensor( const Program* program, MemoryManager* memory_manager, const executorch_flatbuffer::Tensor* s_tensor, const NamedDataMap* named_data_map, Span<NamedData> external_constants)

> [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn]
> Portable/ETensor-mode implementation of `parseTensor`: deserializes one
> `executorch_flatbuffer::Tensor` (`s_tensor`) into a portable `Tensor` (a handle
> over a `TensorImpl`) backed by executorch-owned memory. Returns
> `Result<Tensor>`; on any failing `ET_CHECK_OR_RETURN_ERROR` it logs and returns
> the named non-Ok `Error` immediately. Allocations come from the method allocator
> (`memory_manager->method_allocator()`); a null allocation returns
> `Error::MemoryAllocationFailed`.
>
> Steps:
> 1. Open a profiling scope named "TensorParser::parseTensor". Bind
>    `method_allocator = memory_manager->method_allocator()`.
> 2. Require `s_tensor->storage_offset() == 0`, else `Error::NotSupported`
>    ("Non-zero storage offset ... not supported").
> 3. `scalar_type = static_cast<ScalarType>(s_tensor->scalar_type())`. Require
>    `isValid(scalar_type)` (a recognized, supported ScalarType), else
>    `Error::InvalidProgram`.
> 4. `dynamism = static_cast<TensorShapeDynamism>(s_tensor->shape_dynamism())`.
>    Require `dynamism != DYNAMIC_UNBOUND`, else `Error::NotSupported` ("Fully
>    dynamic tensor shapes not yet supported"). (Fully dynamic shapes are handled
>    in ATen mode but rejected here.)
> 5. Require `s_tensor->sizes() != nullptr`, else `Error::InvalidProgram`
>    ("Missing sizes field"). Let `serialized_sizes = s_tensor->sizes()->data()`
>    (pointer to int32 in flatbuffer memory) and `dim = s_tensor->sizes()->size()`.
> 6. Require `dim <= kTensorDimensionLimit`, else `Error::InvalidProgram`
>    ("Tensor rank too large").
> 7. Require `s_tensor->dim_order() != nullptr` (else `Error::InvalidProgram`
>    "Missing dim_order field") and `s_tensor->dim_order()->size() == dim` (else
>    `Error::InvalidProgram` "dim_order size != dim"). Let `serialized_dim_order =
>    s_tensor->dim_order()->data()`.
> 8. Resolve the `sizes` and `dim_order` buffers used to construct the tensor,
>    branching on dynamism:
>    - If `dynamism != STATIC` (i.e. bounded dynamic): allocate a mutable
>      `SizesType[dim]` (`sizes_buf`) and a mutable `DimOrderType[dim]`
>      (`dim_order_buf`) from the method allocator (each null -> return
>      `Error::MemoryAllocationFailed`), then `memcpy` `dim` elements of the
>      serialized sizes and dim_order into them respectively. Use these local
>      buffers as `sizes` / `dim_order` so the tensor can be resized later.
>    - If `dynamism == STATIC`: do not copy; set `sizes` and `dim_order` to point
>      directly at the (const) flatbuffer memory `serialized_sizes` /
>      `serialized_dim_order` (const-cast away constness — safe because static
>      tensors are never resized so these fields are never written).
> 9. Validate sizes: for `i` from 0 to `dim-1`, require `sizes[i] >= 0`, else
>    `Error::InvalidProgram` ("Negative size ..."). (Bad positive values cannot be
>    detected; negative values would otherwise panic in the `TensorImpl` ctor.
>    `dim_order` itself is validated by `dim_order_to_stride` below.)
> 10. Allocate a mutable `StridesType[dim]` (`strides`) from the method allocator
>     (null -> `Error::MemoryAllocationFailed`) and fill it via
>     `dim_order_to_stride(sizes, dim_order, dim, strides)` (see
>     `[spec:et:sem:dim-order-util.dim-order-to-stride-fn]`). Require its returned
>     status is `Error::Ok`, else `Error::Internal` ("dim_order_to_stride returned
>     invalid status").
> 11. Extract device info: default `device_type = DeviceType::CPU` and
>     `device_index = 0`. If `s_tensor->extra_tensor_info() != nullptr`, override
>     with `static_cast<DeviceType>(extra_tensor_info()->device_type())` and
>     `static_cast<DeviceIndex>(extra_tensor_info()->device_index())`. (The default
>     preserves backward compatibility with older PTE files lacking device
>     annotations.)
> 12. Allocate one `TensorImpl` instance from the method allocator (null ->
>     `Error::MemoryAllocationFailed`) and placement-new it in place with
>     `TensorImpl(scalar_type, dim, sizes, /*data=*/nullptr, dim_order, strides,
>     dynamism, device_type, device_index)` — i.e. constructed first with a null
>     data pointer so its byte size is known before its memory is assigned.
> 13. Resolve the data pointer via `getTensorDataPtr(s_tensor, program,
>     tensor_impl->nbytes(), memory_manager->planned_memory(), named_data_map,
>     external_constants)` (see
>     `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]`).
>     On failure, log an Error with the error code and return that error.
> 14. Assign the resolved pointer with `tensor_impl->set_data(data_ptr)` (may be
>     nullptr for input/placeholder tensors, to be supplied at runtime).
> 15. Return `Tensor(tensor_impl)` — a Tensor handle wrapping the allocated impl.

