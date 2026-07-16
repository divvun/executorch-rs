//! Literal port of runtime/backend/backend_execution_context.h.

use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::memory_allocator::MemoryAllocatorBase;

/// BackendExecutionContext will be used to inject run time context.
// [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context]
pub struct BackendExecutionContext {
    event_tracer_: *mut dyn EventTracer,
    temp_allocator_: *mut dyn MemoryAllocatorBase,
    method_name_: *const core::ffi::c_char,
}

impl BackendExecutionContext {
    // [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.backend-execution-context-fn]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.backend-execution-context-fn]
    //
    // PORT-NOTE: C++ default args (`event_tracer = nullptr, temp_allocator =
    // nullptr, method_name = nullptr`). Rust has no default args; callers pass
    // the three pointers explicitly. A null `*mut dyn EventTracer` is
    // constructed from a null data pointer with a valid vtable slot; use
    // `core::ptr::null_mut::<ConcreteType>() as *mut dyn EventTracer` at the
    // call site, or `BackendExecutionContext::default()` for the all-null case.
    pub fn new(
        event_tracer: *mut dyn EventTracer,
        temp_allocator: *mut dyn MemoryAllocatorBase,
        method_name: *const core::ffi::c_char,
    ) -> Self {
        BackendExecutionContext {
            event_tracer_: event_tracer,
            temp_allocator_: temp_allocator,
            method_name_: method_name,
        }
    }

    /// Returns a pointer to an instance of EventTracer to do profiling/debugging
    /// logging inside the delegate backend. Users will need access to this pointer
    /// to use any of the event tracer APIs.
    // [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.event-tracer-fn]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.event-tracer-fn]
    pub fn event_tracer(&mut self) -> *mut dyn EventTracer {
        self.event_tracer_
    }

    /// Returns a pointer to the address allocated by temp allocator. This
    /// allocator will be reset after every delegate call during execution.
    // [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.allocate-fn]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.allocate-fn]
    pub fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
        // TODO(chenlai): depends on the need, we may expose more functionality for
        // memory allocation.
        //
        // PORT-NOTE: mirrors C++'s unchecked `temp_allocator_->allocate(...)`:
        // no null-check on `temp_allocator_`, so a null allocator is UB exactly
        // as in the original.
        unsafe { (*self.temp_allocator_).allocate(size, alignment) }
    }

    /// Returns the temp allocator. This allocator will be reset every instruction.
    // [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-temp-allocator-fn]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-temp-allocator-fn]
    pub fn get_temp_allocator(&mut self) -> *mut dyn MemoryAllocatorBase {
        self.temp_allocator_
    }

    /// Get the name of the executing method from the ExecuTorch runtime.
    // [spec:et:def:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-method-name-fn]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-method-name-fn]
    pub fn get_method_name(&self) -> *const core::ffi::c_char {
        self.method_name_
    }
}

// No C++ test file targets backend_execution_context.h directly; these focused
// unit tests pin the constructor/accessor semantics from the sem rules against
// the same MemoryAllocator the runtime installs as the temp allocator.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::memory_allocator::MemoryAllocator;
    use crate::runtime::platform::runtime::runtime_init;

    fn null_event_tracer() -> *mut dyn EventTracer {
        crate::extension::module::module::null_event_tracer()
    }
    fn null_allocator() -> *mut dyn MemoryAllocatorBase {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }

    // Constructor stores the three pointers verbatim; the accessors return them
    // unchanged. A context built from all-null yields null accessors.
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.backend-execution-context-fn/test]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.event-tracer-fn/test]
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.get-temp-allocator-fn/test]
    #[test]
    fn backend_execution_context_stores_and_returns_pointers() {
        runtime_init();

        let mut null_ctx =
            BackendExecutionContext::new(null_event_tracer(), null_allocator(), core::ptr::null());
        assert!(null_ctx.event_tracer().is_null());
        assert!(null_ctx.get_temp_allocator().is_null());
        assert!(null_ctx.get_method_name().is_null());

        let mut buffer = [0u8; 64];
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());
        let allocator_ptr = &mut allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase;
        let method_name = c"forward";

        let mut ctx =
            BackendExecutionContext::new(null_event_tracer(), allocator_ptr, method_name.as_ptr());
        // get_temp_allocator returns the same allocator pointer that was stored.
        assert_eq!(
            ctx.get_temp_allocator() as *mut MemoryAllocator,
            allocator_ptr as *mut MemoryAllocator
        );
        // get_method_name returns the same C-string pointer verbatim.
        assert_eq!(ctx.get_method_name(), method_name.as_ptr());
    }

    // allocate delegates to temp_allocator_->allocate(size, alignment); the
    // returned pointer falls inside the allocator's backing buffer.
    // [spec:et:sem:backend-execution-context.executorch.et-runtime-namespace.backend-execution-context.allocate-fn/test]
    #[test]
    fn backend_execution_context_allocate_delegates_to_temp_allocator() {
        runtime_init();

        let mut buffer = [0u8; 64];
        let base = buffer.as_mut_ptr() as usize;
        let mut allocator = MemoryAllocator::new(buffer.len() as u32, buffer.as_mut_ptr());
        let allocator_ptr = &mut allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase;

        let mut ctx =
            BackendExecutionContext::new(null_event_tracer(), allocator_ptr, core::ptr::null());
        let p = ctx.allocate(16, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert!(!p.is_null());
        let addr = p as usize;
        assert!(addr >= base && addr + 16 <= base + buffer.len());
        assert_eq!(addr % MemoryAllocator::K_DEFAULT_ALIGNMENT, 0);
    }
}
