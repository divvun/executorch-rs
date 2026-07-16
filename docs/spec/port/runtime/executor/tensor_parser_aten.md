# runtime/executor/tensor_parser_aten.cpp

> [spec:et:def:tensor-parser-aten.executorch.runtime.aten.deserialization.delete-nothing-fn]
> void deleteNothing(void*)

> [spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.delete-nothing-fn]
> A no-op deleter with signature `void(void*)`. Ignores its argument and does
> nothing. Passed to `at::from_blob` as the storage deleter so that ATen never
> frees the tensor's data buffer: the buffer is owned by executorch (planned
> memory, constant buffer, or external data) and outlives the tensor, so the ATen
> Tensor must not attempt to deallocate it when its storage refcount drops to
> zero. In Rust this corresponds to a deleter closure that does not run any
> destructor / free on the pointer.

> [spec:et:def:tensor-parser-aten.executorch.runtime.aten.deserialization.parse-tensor-fn]
> Result<at::Tensor> parseTensor( const Program* program, MemoryManager* memory_manager, const executorch_flatbuffer::Tensor* s_tensor, const NamedDataMap* named_data_map, Span<NamedData> external_constants)

> [spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.parse-tensor-fn]
> ATen-mode implementation of `parseTensor`: deserializes one
> `executorch_flatbuffer::Tensor` (`s_tensor`) into an `at::Tensor` backed by
> executorch-owned memory. Returns `Result<at::Tensor>`; on any check failure
> returns the corresponding non-Ok `Error`.
>
> Steps:
> 1. Open a profiling scope named "TensorParser::parseTensor".
> 2. Require `s_tensor->storage_offset() == 0`, else `Error::NotSupported`
>    ("Non-zero storage offset ... not supported").
> 3. Read `type = static_cast<at::ScalarType>(s_tensor->scalar_type())`. Require
>    `isValid(type)` (the type is a recognized ScalarType), else
>    `Error::InvalidProgram`. Build ATen tensor `options` from `at::CPU(type)`
>    (CPU backend, dtype = `type`).
> 4. Require `s_tensor->sizes() != nullptr`, else `Error::InvalidProgram`
>    ("Missing sizes field"). Let `ndim = sizes->size()`.
> 5. Require `ndim <= kTensorDimensionLimit`, else `Error::InvalidProgram`
>    ("Tensor rank too large").
> 6. Require `s_tensor->dim_order() != nullptr` (else `Error::InvalidProgram`
>    "Missing dim_order field") and `dim_order()->size() == ndim` (else
>    `Error::InvalidProgram` "dim_order size != ndim").
> 7. Copy the serialized int32 sizes into a `std::vector<int64_t> sizes`
>    (widening each element to int64 for ATen). Allocate
>    `std::vector<int64_t> strides(ndim)` and fill it by calling
>    `dim_order_to_stride(sizes_ptr, dim_order_ptr, ndim, strides.data())` (see
>    `[spec:et:sem:dim-order-util.dim-order-to-stride-fn]`); require its status
>    is `Error::Ok`, else `Error::Internal` ("dim_order_to_stride returned
>    invalid status").
> 8. Create a data-less tensor first, to learn its byte size before allocating
>    memory: `tensor = at::from_blob(data=nullptr, sizes, strides,
>    storage_offset=0, deleter=deleteNothing, options)` (see
>    `[spec:et:sem:tensor-parser-aten.executorch.runtime.aten.deserialization.delete-nothing-fn]`).
> 9. Branch on shape dynamism:
>    - If `s_tensor->shape_dynamism() == DYNAMIC_UNBOUND` (fully dynamic): give
>      the tensor a real resizable allocator so ATen kernels can resize it. On
>      its underlying StorageImpl: set allocator to `at::getCPUAllocator()`, set
>      resizable = true, set nbytes = 0; then set the tensor's sizes to the
>      empty/0 contiguous shape (`impl->set_sizes_contiguous(0)`). Leave data as
>      nullptr (it will be reallocated by ATen on resize).
>    - Otherwise (STATIC or bounded dynamic): resolve the data pointer via
>      `getTensorDataPtr(s_tensor, program, tensor.nbytes(),
>      memory_manager->planned_memory(), named_data_map, external_constants)`
>      (see
>      `[spec:et:sem:tensor-parser-exec-aten.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]`).
>      If that fails, log an Error with the error code and return the error.
>      Otherwise set the tensor storage's data pointer to
>      `at::DataPtr(data_ptr, DeviceType::CPU)` (no owning deleter, since the
>      memory is executorch-owned).
> 10. Return the `at::Tensor`.

