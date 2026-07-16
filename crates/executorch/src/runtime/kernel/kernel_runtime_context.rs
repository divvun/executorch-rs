//! Literal port of runtime/kernel/kernel_runtime_context.h.

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::event_tracer::EventTracer;
use crate::runtime::core::memory_allocator::MemoryAllocatorBase;

/// Runtime state and functionality for kernel implementations.
///
/// NOTE: Will not be passed to operators if running in ATen mode as those
/// operators do not expect to receive a KernelRuntimeContext argument.
// [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context]
//
// PORT-NOTE: the C++ members are raw `EventTracer*` / `MemoryAllocator*` that
// the context does not own; they are mirrored as raw pointers so the non-owning,
// nullable, mutate-through-pointer semantics survive verbatim. Both are base
// pointers to abstract/virtual types, so they are `dyn EventTracer` /
// `dyn MemoryAllocatorBase` trait objects (matching the C++ base-pointer
// polymorphism per PORTING.md) — the temp allocator thus dispatches to the
// concrete allocator's overridden `allocate`.
pub struct KernelRuntimeContext<'a> {
    event_tracer_: *mut (dyn EventTracer + 'a),
    temp_allocator_: *mut (dyn MemoryAllocatorBase + 'a),
    failure_state_: Error,
}

impl<'a> KernelRuntimeContext<'a> {
    /// Construct a new kernel runtime context.
    ///
    /// KernelRuntimeContext does not take ownership
    /// of these pointers, so they must outlive the context instance.
    // [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn]
    //
    // PORT-NOTE: C++ defaults both parameters to `nullptr`. Rust has no default
    // arguments; `new` takes both raw pointers (pass a null `dyn EventTracer`
    // pointer / null `MemoryAllocator` pointer for the default-constructed
    // context). `failure_state_` takes its in-class default of `Error::Ok`.
    pub fn new(
        event_tracer: *mut (dyn EventTracer + 'a),
        temp_allocator: *mut (dyn MemoryAllocatorBase + 'a),
    ) -> Self {
        KernelRuntimeContext {
            event_tracer_: event_tracer,
            temp_allocator_: temp_allocator,
            failure_state_: Error::Ok,
        }
    }

    /// Tells the runtime that the kernel call has failed. Prefer this over
    /// ET_CHECK_*(), which fatally panics the process/system.
    ///
    /// If this is not called, the runtime will treat the kernel call as
    /// successful.
    // [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn]
    pub fn fail(&mut self, error: Error) {
        self.failure_state_ = error;
    }

    /// Returns the current failure state.
    // [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn]
    #[must_use]
    pub fn failure_state(&self) -> Error {
        self.failure_state_
    }

    /// INTERNAL ONLY
    ///
    /// Returns a pointer to an instance of EventTracer to do profiling/debugging
    /// logging inside the codegen layer. This is only for internal usage inside
    /// the codegen layer and users should not be accessing this.
    // [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.internal-event-tracer-fn]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.internal-event-tracer-fn]
    pub fn internal_event_tracer(&mut self) -> *mut (dyn EventTracer + 'a) {
        self.event_tracer_
    }

    /// Allocates temporary memory that will be freed when the kernel returns. This
    /// returns a pointer to the allocated memory or an error if the allocation
    /// fails.
    ///
    /// @param[in] size Number of bytes to allocate.
    /// @param[in] alignment Minimum alignment for the returned pointer. Must be a
    ///     power of 2.
    ///
    /// @returns A result object containing either a pointer to the allocated
    ///     memory or an error to indicate failure
    // [spec:et:def:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn]
    //
    // PORT-NOTE: C++ defaults `alignment` to `MemoryAllocator::kDefaultAlignment`.
    // Rust has no default arguments, so callers must pass
    // `MemoryAllocator::K_DEFAULT_ALIGNMENT` explicitly for the default case.
    pub fn allocate_temp(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<*mut core::ffi::c_void> {
        crate::et_check_or_return_error!(
            !self.temp_allocator_.is_null(),
            NotFound,
            "No temp allocator provided"
        );
        let temp_memory = unsafe { (*self.temp_allocator_).allocate(size, alignment) };
        crate::et_check_or_return_error!(
            !temp_memory.is_null(),
            MemoryAllocationFailed,
            "Failed to allocate temp memory. Bytes requested: {}",
            size
        );
        Ok(temp_memory)
    }

    // TODO(T147221312): Add a way to resize a tensor.
}

/// If `cond` is false, log `Check failed (<cond>): <message>`, record `error`
/// on the kernel `context`'s failure state, and return `retval` from the
/// current kernel to allow for early exit.
///
/// Ports `ET_KERNEL_CHECK_MSG(context, cond, error, retval, message, ...)` from
/// runtime/core/exec_aten/util/tensor_util.h. `$error` is the `Error` enum
/// variant name without the `Error::` prefix (e.g. `InvalidArgument`).
///
/// PORT-NOTE: PORTING.md assigns the `et_kernel_check!` macro family to this
/// module even though the C++ macros live in `tensor_util.h`; they are hosted
/// here so kernel authors get them alongside `KernelRuntimeContext`.
#[macro_export]
macro_rules! et_kernel_check_msg {
    ($context:expr, $cond:expr, $error:ident, $retval:expr, $fmt:literal $(, $($args:tt)*)?) => {{
        if !($cond) {
            $crate::et_log!(
                Error,
                ::core::concat!("Check failed ({}): ", $fmt),
                ::core::stringify!($cond),
                $($($args)*)?
            );
            $context.fail($crate::runtime::core::error::Error::$error);
            return $retval;
        }
    }};
}

/// If `cond` is false, log `Check failed (<cond>): `, record `error` on the
/// kernel `context`'s failure state, and return `retval` from the current
/// kernel to allow for early exit.
///
/// Ports `ET_KERNEL_CHECK(context, cond, error, retval)` from
/// runtime/core/exec_aten/util/tensor_util.h. `$error` is the `Error` enum
/// variant name without the `Error::` prefix (e.g. `InvalidArgument`).
#[macro_export]
macro_rules! et_kernel_check {
    ($context:expr, $cond:expr, $error:ident, $retval:expr) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            $context.fail($crate::runtime::core::error::Error::$error);
            return $retval;
        }
    }};
    ($context:expr, $cond:expr, $error:ident $(,)?) => {{
        if !($cond) {
            $crate::et_log!(Error, "Check failed ({}): ", ::core::stringify!($cond));
            $context.fail($crate::runtime::core::error::Error::$error);
            return;
        }
    }};
}

// Port of runtime/kernel/test/kernel_runtime_context_test.cpp.
#[cfg(test)]
mod tests {
    use super::KernelRuntimeContext;
    use crate::runtime::core::error::Error;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::result::ResultExt;
    use crate::runtime::platform::runtime::runtime_init;

    // Mirrors the fixture `SetUp()` (`runtime_init()`).
    fn setup() {
        runtime_init();
    }

    // Null `dyn` fat pointers for the default-constructed context. C++ defaults
    // both ctor args to `nullptr`; built per the established null-fat-pointer
    // pattern.
    fn null_event_tracer() -> *mut (dyn crate::runtime::core::event_tracer::EventTracer + 'static) {
        crate::extension::module::module::null_event_tracer()
    }
    fn null_allocator() -> *mut (dyn MemoryAllocatorBase + 'static) {
        core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase
    }

    // class TestMemoryAllocator : public MemoryAllocator — overrides `allocate`
    // to record the alignment it was called with, then delegates to the base
    // bump allocator. Composed over a `MemoryAllocator` (its fields are private)
    // so the override delegates to the base behavior, matching the C++ subclass.
    struct TestMemoryAllocator {
        base: MemoryAllocator,
        last_seen_alignment: usize,
    }

    impl TestMemoryAllocator {
        fn new(size: u32, base_address: *mut u8) -> Self {
            TestMemoryAllocator {
                base: MemoryAllocator::new(size, base_address),
                last_seen_alignment: 0,
            }
        }
    }

    impl MemoryAllocatorBase for TestMemoryAllocator {
        fn allocate(&mut self, size: usize, alignment: usize) -> *mut core::ffi::c_void {
            self.last_seen_alignment = alignment;
            self.base.allocate(size, alignment)
        }
        fn base_address(&self) -> *mut u8 {
            self.base.base_address()
        }
        fn size(&self) -> u32 {
            self.base.size()
        }
        fn used_size(&self) -> usize {
            self.base.used_size()
        }
        fn free_size(&self) -> usize {
            self.base.free_size()
        }
        fn reset(&mut self) {
            self.base.reset()
        }
    }

    // The default-constructed context (null tracer + null allocator) starts in
    // the Ok failure state, so this pins the constructor's in-class default
    // `failure_state_ = Error::Ok`.
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn/test]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn/test]
    #[test]
    fn kernel_runtime_context_test_failure_state_defaults_to_ok() {
        setup();
        let context = KernelRuntimeContext::new(null_event_tracer(), null_allocator());

        assert_eq!(context.failure_state(), Error::Ok);
    }

    // `internal_event_tracer()` returns the stored `event_tracer_` pointer
    // verbatim: it returns the exact data-pointer + vtable it was constructed
    // with (here the null-tracer fat pointer), with no null-check and no
    // transformation. Comparing the returned raw fat pointer against the one
    // passed in pins "returns the stored pointer verbatim" rather than merely
    // "returns null".
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.internal-event-tracer-fn/test]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.kernel-runtime-context-fn/test]
    #[test]
    fn kernel_runtime_context_test_internal_event_tracer_returns_stored_pointer() {
        setup();

        let tracer_ptr = null_event_tracer();
        let mut ctx = KernelRuntimeContext::new(tracer_ptr, null_allocator());
        assert_eq!(ctx.internal_event_tracer(), tracer_ptr);
        assert!(ctx.internal_event_tracer().is_null());
    }

    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.fail-fn/test]
    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.failure-state-fn/test]
    #[test]
    fn kernel_runtime_context_test_failure_state_reflects_failure() {
        setup();
        let mut context = KernelRuntimeContext::new(null_event_tracer(), null_allocator());

        // Starts off Ok.
        assert_eq!(context.failure_state(), Error::Ok);

        // Failing should update the failure state.
        context.fail(Error::MemoryAllocationFailed);
        assert_eq!(context.failure_state(), Error::MemoryAllocationFailed);

        // State can be overwritten.
        context.fail(Error::Internal);
        assert_eq!(context.failure_state(), Error::Internal);

        // And can be cleared.
        context.fail(Error::Ok);
        assert_eq!(context.failure_state(), Error::Ok);
    }

    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn/test]
    #[test]
    fn kernel_runtime_context_test_failure_no_memory_allocator_provided() {
        setup();
        let mut context = KernelRuntimeContext::new(null_event_tracer(), null_allocator());
        let allocated_memory = context.allocate_temp(4, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(ResultExt::error(&allocated_memory), Error::NotFound);
    }

    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn/test]
    #[test]
    fn kernel_runtime_context_test_successful_memory_allocation() {
        setup();
        const TEMP_MEMORY_ALLOCATOR_POOL_SIZE: usize = 4;
        let mut temp_memory_allocator_pool = vec![0u8; TEMP_MEMORY_ALLOCATOR_POOL_SIZE];
        let mut temp_allocator = MemoryAllocator::new(
            TEMP_MEMORY_ALLOCATOR_POOL_SIZE as u32,
            temp_memory_allocator_pool.as_mut_ptr(),
        );
        let mut context = KernelRuntimeContext::new(
            null_event_tracer(),
            &mut temp_allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
        );
        let allocated_memory = context.allocate_temp(4, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(ResultExt::ok(&allocated_memory), true);
    }

    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn/test]
    #[test]
    fn kernel_runtime_context_test_failure_memory_allocation_insufficient_space() {
        setup();
        const TEMP_MEMORY_ALLOCATOR_POOL_SIZE: usize = 4;
        let mut temp_memory_allocator_pool = vec![0u8; TEMP_MEMORY_ALLOCATOR_POOL_SIZE];
        let mut temp_allocator = MemoryAllocator::new(
            TEMP_MEMORY_ALLOCATOR_POOL_SIZE as u32,
            temp_memory_allocator_pool.as_mut_ptr(),
        );
        let mut context = KernelRuntimeContext::new(
            null_event_tracer(),
            &mut temp_allocator as *mut MemoryAllocator as *mut dyn MemoryAllocatorBase,
        );
        let allocated_memory = context.allocate_temp(8, MemoryAllocator::K_DEFAULT_ALIGNMENT);
        assert_eq!(
            ResultExt::error(&allocated_memory),
            Error::MemoryAllocationFailed
        );
    }

    // [spec:et:sem:kernel-runtime-context.executorch.et-runtime-namespace.kernel-runtime-context.allocate-temp-fn/test]
    // [spec:et:sem:memory-allocator.executorch.runtime.memory-allocator.allocate-fn/test]
    #[test]
    fn kernel_runtime_context_test_memory_allocator_alignment_passed() {
        setup();
        const TEMP_MEMORY_ALLOCATOR_POOL_SIZE: usize = 4;
        let mut temp_memory_allocator_pool = vec![0u8; TEMP_MEMORY_ALLOCATOR_POOL_SIZE];
        let mut temp_allocator = TestMemoryAllocator::new(
            TEMP_MEMORY_ALLOCATOR_POOL_SIZE as u32,
            temp_memory_allocator_pool.as_mut_ptr(),
        );
        let mut context = KernelRuntimeContext::new(
            null_event_tracer(),
            &mut temp_allocator as *mut TestMemoryAllocator as *mut dyn MemoryAllocatorBase,
        );
        let allocated_memory = context.allocate_temp(4, 2);
        assert_eq!(ResultExt::ok(&allocated_memory), true);
        assert_eq!(temp_allocator.last_seen_alignment, 2);
    }
}
