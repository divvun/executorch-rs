# runtime/backend/backend_execution_context.h

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context]
> class BackendExecutionContext final {
>   EventTracer* event_tracer_ = nullptr;
>   MemoryAllocator* temp_allocator_ = nullptr;
>   const char* method_name_ = nullptr;
> }

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.allocate-fn]
> void* allocate( size_t size, size_t alignment = MemoryAllocator::kDefaultAlignment)

> [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.allocate-fn]
> Allocates `size` bytes with the given `alignment` (default `MemoryAllocator::kDefaultAlignment`) from the context's temp allocator and returns the resulting pointer. Implemented as `return temp_allocator_->allocate(size, alignment);`. It performs no null-check on `temp_allocator_`: if the context was constructed without a temp allocator (`temp_allocator_ == nullptr`), calling `allocate` dereferences a null pointer (undefined behavior). The returned memory lives only until the temp allocator is reset, which the runtime does after every delegate call/instruction during execution.

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.backend-execution-context-fn]
> BackendExecutionContext( EventTracer* event_tracer = nullptr, MemoryAllocator* temp_allocator = nullptr, const char* method_name = nullptr) : event_tracer_(event_tracer), temp_allocator_(temp_allocator), method_name_(method_name)

> [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.backend-execution-context-fn]
> Constructor. Takes three optional pointers, each defaulting to `nullptr`: `event_tracer`, `temp_allocator`, and `method_name`. Stores them directly into the corresponding member fields (`event_tracer_`, `temp_allocator_`, `method_name_`) with no validation, copying, or ownership transfer — the object holds non-owning raw pointers whose lifetimes the caller must guarantee. A default-constructed context therefore has all three members null.

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.event-tracer-fn]
> EventTracer* event_tracer()

> [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.event-tracer-fn]
> Accessor. Returns the stored `event_tracer_` pointer verbatim (may be `nullptr` if none was installed). No side effects, no ownership transfer.

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-method-name-fn]
> const char* get_method_name() const

> [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-method-name-fn]
> Const accessor. Returns the stored `method_name_` C-string pointer verbatim (may be `nullptr`). This is the name of the method currently executing (e.g. "forward"). No side effects, no ownership transfer, no copy.

> [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-temp-allocator-fn]
> MemoryAllocator* get_temp_allocator()

> [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-temp-allocator-fn]
> Accessor. Returns the stored `temp_allocator_` pointer verbatim (may be `nullptr`). This temp allocator is reset by the runtime after every instruction/delegate call. No side effects, no ownership transfer.

