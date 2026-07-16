# runtime/kernel/kernel_runtime_context.h

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context]
> class KernelRuntimeContext {
>   EventTracer* event_tracer_ = nullptr;
>   MemoryAllocator* temp_allocator_ = nullptr;
>   Error failure_state_ = Error::Ok;
> }

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn]
> Result<void*> allocate_temp( size_t size, size_t alignment = MemoryAllocator::kDefaultAlignment)

> [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn]
> Allocates scratch memory for a kernel from the context's optional temporary
> allocator (`temp_allocator_`), conceptually valid for the duration of the kernel
> call. Signature `allocate_temp(size_t size, size_t alignment =
> MemoryAllocator::kDefaultAlignment) -> Result<void*>`. `alignment` must be a
> power of 2 (the caller's contract; enforced downstream by the allocator).
>
> Steps:
> 1. Require `temp_allocator_ != nullptr`, else return `Error::NotFound`
>    ("No temp allocator provided"). (No temp allocator was supplied at
>    construction, so temp allocation is unavailable.)
> 2. Call `temp_memory = temp_allocator_->allocate(size, alignment)` (see
>    `[spec:et:sem:memory-allocator.allocate-fn]`).
> 3. Require `temp_memory != nullptr`, else return `Error::MemoryAllocationFailed`
>    ("Failed to allocate temp memory. Bytes requested: ...").
> 4. Return `temp_memory` as the `Result<void*>` success value.
>
> The context does not own the returned memory and performs no bookkeeping of it;
> lifetime/reclamation is entirely the responsibility of the supplied
> `temp_allocator_` (temp allocators are typically reset by the runtime between
> kernel invocations). `size == 0` is not special-cased here — it is forwarded to
> the allocator, whose result then flows through the null check above.

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn]
> void fail(Error error)

> [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn]
> Records a kernel failure by storing the given `error` into the context's
> `failure_state_` member (`failure_state_ = error`). Signature `void fail(Error
> error)`; no return value. This is the non-fatal error-propagation path for
> portable kernels, which cannot return errors through their PyTorch-compatible
> signatures: instead of aborting via `ET_CHECK_*`, a kernel calls `fail(...)` and
> returns, and the runtime later reads `[spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn]`
> to observe the outcome. The write is an unconditional overwrite: it does not
> check the current state or preserve a prior error, so if called multiple times
> the last `error` wins. Passing `Error::Ok` therefore clears any previously set
> failure. If `fail` is never called, `failure_state_` retains its
> construction-time default of `Error::Ok`, which the runtime treats as success.

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn]
> ET_NODISCARD Error failure_state() const

> [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn]
> Const accessor returning the context's current `failure_state_` by value.
> Signature `Error failure_state() const` (marked `ET_NODISCARD`, so callers must
> use the returned value). It performs no computation and no mutation: it simply
> returns whatever `Error` was last stored, which is `Error::Ok` unless a kernel
> called `fail(...)` (see
> `[spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn]`).
> The runtime calls this after a kernel returns to decide whether the kernel
> succeeded (`Error::Ok`) or failed (any other value).

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.internal-event-tracer-fn]
> EventTracer* internal_event_tracer()

> [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.internal-event-tracer-fn]
> Internal-only accessor returning the raw `event_tracer_` pointer held by the
> context (`EventTracer*`), which may be nullptr when no tracer was supplied at
> construction. Signature `EventTracer* internal_event_tracer()`. It returns the
> stored pointer verbatim with no null-check, no transformation, and no ownership
> transfer (the context does not own the tracer). It exists so the generated
> codegen layer can emit profiling/debugging events; general kernel authors and
> users are not intended to call it.

> [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn]
> KernelRuntimeContext( EventTracer* event_tracer = nullptr, MemoryAllocator* temp_allocator = nullptr) : event_tracer_(event_tracer), temp_allocator_(temp_allocator)

> [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn]
> Constructor. Signature `KernelRuntimeContext(EventTracer* event_tracer =
> nullptr, MemoryAllocator* temp_allocator = nullptr)`. Both parameters default to
> nullptr, so a default-constructed context has no tracer and no temp allocator.
> It stores the two arguments into the corresponding members via a member
> initializer list: `event_tracer_ = event_tracer` and `temp_allocator_ =
> temp_allocator`. The third member, `failure_state_`, is not listed and takes its
> in-class default initializer `Error::Ok`. The context does NOT take ownership of
> either pointer; both must outlive the context instance. No validation is
> performed (either pointer may legitimately be null): a null `temp_allocator_`
> later causes
> `[spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn]`
> to return `Error::NotFound`, and a null `event_tracer_` disables event tracing.

