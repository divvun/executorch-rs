//! Literal port of runtime/core/event_tracer_hooks.h.
//!
//! This file contains the hooks that are inserted across various parts of the
//! core runtime code to call into the EventTracer class for logging of profiling
//! and debugging events. Any calls made to the EventTracer from the runtime must
//! be made via these hooks.
//!
//! The benefit of defining these hooks is that we can easily control whether or
//! not we want to compile in the EventTracer code based on the status of the
//! ET_EVENT_TRACER_ENABLED flag.
//!
//! PORT-NOTE: `ET_EVENT_TRACER_ENABLED` (a compile-time `#ifdef`) maps to the
//! `event-tracer` cargo feature. Both compiled shapes are kept literal: with the
//! feature on the hooks null-check the tracer and dispatch; with it off they are
//! no-ops. The C++ `EventTracer*` (raw, nullable) maps to `*mut dyn EventTracer`
//! so the null-check semantics survive; callers pass `core::ptr::null_mut()` for
//! "no tracer".

use crate::runtime::core::evalue::EValue;
#[cfg(not(feature = "event-tracer"))]
use crate::runtime::core::event_tracer::AllocatorID;
#[cfg(feature = "event-tracer")]
use crate::runtime::core::event_tracer::{
    AllocatorID, EventTracerDebugLogLevel, EventTracerProfilingLevel, LoggedEValueType,
};
use crate::runtime::core::event_tracer::{
    ChainID, DebugHandle, EventTracer, EventTracerEntry, K_UNSET_CHAIN_ID, K_UNSET_DEBUG_HANDLE,
};

/// This class enables scope based profiling where needed using RAII for
/// operators only. If operator profiling is disabled then this class is a no-op.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope]
pub struct EventTracerProfileOpScope {
    #[cfg(feature = "event-tracer")]
    event_tracer_: *mut dyn EventTracer,
    #[cfg(feature = "event-tracer")]
    event_entry_: EventTracerEntry,
}

impl EventTracerProfileOpScope {
    // [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn]
    pub fn new(event_tracer: *mut dyn EventTracer, name: *const core::ffi::c_char) -> Self {
        #[cfg(feature = "event-tracer")]
        {
            let event_tracer_ = event_tracer;
            let mut event_entry_ = EventTracerEntry::default();
            if event_tracer_.is_null() {
                return EventTracerProfileOpScope {
                    event_tracer_,
                    event_entry_,
                };
            }
            if unsafe { (*event_tracer_).event_tracer_profiling_level() }
                > EventTracerProfilingLevel::KProfileMethodOnly
            {
                event_entry_ = unsafe {
                    (*event_tracer).start_profiling(name, K_UNSET_CHAIN_ID, K_UNSET_DEBUG_HANDLE)
                };
            }
            EventTracerProfileOpScope {
                event_tracer_,
                event_entry_,
            }
        }
        #[cfg(not(feature = "event-tracer"))]
        {
            let _ = event_tracer;
            let _ = name;
            EventTracerProfileOpScope {}
        }
    }
}

impl Drop for EventTracerProfileOpScope {
    fn drop(&mut self) {
        #[cfg(feature = "event-tracer")]
        {
            if self.event_tracer_.is_null() {
                return;
            }
            if unsafe { (*self.event_tracer_).event_tracer_profiling_level() }
                > EventTracerProfilingLevel::KProfileMethodOnly
            {
                unsafe { (*self.event_tracer_).end_profiling(self.event_entry_) };
            }
        }
    }
}

pub type EventTracerProfileScope = EventTracerProfileOpScope;

/// This class enables scope based profiling where needed using RAII.
/// Profiling will be started when the object is created and will end
/// when the object goes out of scope. This is specifically intended to
/// be used for profiling methods in the runtime.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope]
pub struct EventTracerProfileMethodScope {
    #[cfg(feature = "event-tracer")]
    event_tracer_: *mut dyn EventTracer,
    #[cfg(feature = "event-tracer")]
    event_entry_: EventTracerEntry,
}

impl EventTracerProfileMethodScope {
    // [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn]
    pub fn new(event_tracer: *mut dyn EventTracer, name: *const core::ffi::c_char) -> Self {
        #[cfg(feature = "event-tracer")]
        {
            let event_tracer_ = event_tracer;
            let mut event_entry_ = EventTracerEntry::default();
            if event_tracer_.is_null() {
                return EventTracerProfileMethodScope {
                    event_tracer_,
                    event_entry_,
                };
            }
            event_entry_ = unsafe {
                (*event_tracer).start_profiling(name, K_UNSET_CHAIN_ID, K_UNSET_DEBUG_HANDLE)
            };
            EventTracerProfileMethodScope {
                event_tracer_,
                event_entry_,
            }
        }
        #[cfg(not(feature = "event-tracer"))]
        {
            let _ = event_tracer;
            let _ = name;
            EventTracerProfileMethodScope {}
        }
    }
}

impl Drop for EventTracerProfileMethodScope {
    fn drop(&mut self) {
        #[cfg(feature = "event-tracer")]
        {
            if self.event_tracer_.is_null() {
                return;
            }
            unsafe { (*self.event_tracer_).end_profiling(self.event_entry_) };
        }
    }
}

/// This class helps us set and then clear out the chain id and debug handle
/// values stored in the event tracer class using RAII. This is typically called
/// in the executor loop before entering the codegen layer to configure the chain
/// id and debug handle of the current instruction being executed.
/// After we return from the kernel execution we can then reset the chain id and
/// debug handle to defaults when this object goes out of scope.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope]
pub struct EventTracerProfileInstructionScope {
    #[cfg(feature = "event-tracer")]
    event_tracer_: *mut dyn EventTracer,
}

impl EventTracerProfileInstructionScope {
    // [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope.event-tracer-profile-instruction-scope-fn]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope.event-tracer-profile-instruction-scope-fn]
    pub fn new(
        event_tracer: *mut dyn EventTracer,
        chain_idx: ChainID,
        debug_handle: DebugHandle,
    ) -> Self {
        #[cfg(feature = "event-tracer")]
        {
            let event_tracer_ = event_tracer;
            if event_tracer_.is_null() {
                return EventTracerProfileInstructionScope { event_tracer_ };
            }
            unsafe { (*event_tracer_).set_chain_debug_handle(chain_idx, debug_handle) };
            EventTracerProfileInstructionScope { event_tracer_ }
        }
        #[cfg(not(feature = "event-tracer"))]
        {
            let _ = event_tracer;
            let _ = chain_idx;
            let _ = debug_handle;
            EventTracerProfileInstructionScope {}
        }
    }
}

impl Drop for EventTracerProfileInstructionScope {
    fn drop(&mut self) {
        #[cfg(feature = "event-tracer")]
        {
            if self.event_tracer_.is_null() {
                return;
            }
            unsafe {
                (*self.event_tracer_).set_chain_debug_handle(K_UNSET_CHAIN_ID, K_UNSET_DEBUG_HANDLE)
            };
        }
    }
}

// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-enabled-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-enabled-fn]
#[inline]
pub fn event_tracer_enabled() -> bool {
    #[cfg(feature = "event-tracer")]
    {
        true
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        false
    }
}

/// Create a new event block with the specified name. Any events logged
/// after this will be associated with this new event block.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-create-event-block-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-create-event-block-fn]
#[inline]
pub fn event_tracer_create_event_block(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            unsafe { (*event_tracer).create_event_block(name) };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = name;
    }
}

/// Explicitly mark the beginning of a new profiling event. This returns
/// an instance of an EventTracerEntry object that the user needs to keep
/// around and pass into the corresponding event_tracer_end_profiling_event
/// call.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn]
#[inline]
pub fn event_tracer_begin_profiling_event(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
) -> EventTracerEntry {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            return unsafe {
                (*event_tracer).start_profiling(name, K_UNSET_CHAIN_ID, K_UNSET_DEBUG_HANDLE)
            };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = name;
    }
    // There is no active tracer; this value will be ignored.
    EventTracerEntry::default()
}

/// Mark the end of a profiling event passing in the entry token
/// returned by a previous call to ET_EVENT_TRACER_BEGIN_PROFILING_EVENT.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn]
#[inline]
pub fn event_tracer_end_profiling_event(
    event_tracer: *mut dyn EventTracer,
    event: EventTracerEntry,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            unsafe { (*event_tracer).end_profiling(event) };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = event;
    }
}

/// Start the tracking of the allocator represented by this name and returns
/// an AllocatorID that will be used to track all subsequent allocations done by
/// this allocator.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn]
#[inline]
pub fn event_tracer_track_allocator(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
) -> AllocatorID {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            return unsafe { (*event_tracer).track_allocator(name) };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = name;
    }
    // There is no active tracer; this value will be ignored.
    0
}

/// Log the allocation event done via the allocator represented by id.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn]
#[inline]
pub fn event_tracer_track_allocation(
    event_tracer: *mut dyn EventTracer,
    id: AllocatorID,
    size: usize,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            unsafe { (*event_tracer).track_allocation(id, size) };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = id;
        let _ = size;
    }
}

/// Log an intermediate value.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-fn]
#[inline]
pub fn event_tracer_log_evalue(event_tracer: *mut dyn EventTracer, evalue: &mut EValue) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            if unsafe { (*event_tracer).event_tracer_debug_level() }
                >= EventTracerDebugLogLevel::KIntermediateOutputs
            {
                let _ = unsafe {
                    (*event_tracer).log_evalue(evalue, LoggedEValueType::KIntermediateOutput)
                };
            }
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = evalue;
    }
}

/// Log a program output.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn]
#[inline]
pub fn event_tracer_log_evalue_output(event_tracer: *mut dyn EventTracer, evalue: &EValue) {
    #[cfg(feature = "event-tracer")]
    {
        // If debugging via event tracer is enabled but intermediate output
        // logging is disabled then we want to only log the outputs.
        if !event_tracer.is_null() {
            if unsafe { (*event_tracer).event_tracer_debug_level() }
                >= EventTracerDebugLogLevel::KProgramOutputs
            {
                let _ =
                    unsafe { (*event_tracer).log_evalue(evalue, LoggedEValueType::KProgramOutput) };
            }
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = evalue;
    }
}

// Set the bundled input index of the current bundled input being used by the
// method.
// [spec:et:def:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-set-bundled-input-index-fn]
// [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-set-bundled-input-index-fn]
#[inline]
pub fn event_tracer_set_bundled_input_index(
    event_tracer: *mut dyn EventTracer,
    bundled_input_index: i32,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            unsafe { (*event_tracer).set_bundled_input_index(bundled_input_index) };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = bundled_input_index;
    }
}

// PORT-NOTE: no C++ counterpart. event_tracer_test.cpp exercises most hooks
// through the tracer but never `event_tracer_enabled` or
// `event_tracer_set_bundled_input_index`; these focused tests pin them.
#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-enabled-fn/test]
    #[test]
    fn event_tracer_hooks_test_event_tracer_enabled() {
        // Mirrors the compile-time `ET_EVENT_TRACER_ENABLED` branch: `true` iff the
        // `event-tracer` feature is on.
        #[cfg(feature = "event-tracer")]
        assert!(event_tracer_enabled());
        #[cfg(not(feature = "event-tracer"))]
        assert!(!event_tracer_enabled());
    }

    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-set-bundled-input-index-fn/test]
    #[cfg(feature = "event-tracer")]
    #[test]
    fn event_tracer_hooks_test_set_bundled_input_index() {
        use crate::runtime::core::event_tracer::{
            EventTracer, EventTracerState, K_UNSET_BUNDLED_INPUT_INDEX,
        };

        struct MinimalTracer {
            state_: EventTracerState,
        }
        // Only the accessors are used here; the abstract methods are unreachable.
        impl EventTracer for MinimalTracer {
            fn state(&self) -> &EventTracerState {
                &self.state_
            }
            fn state_mut(&mut self) -> &mut EventTracerState {
                &mut self.state_
            }
            fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
                unimplemented!()
            }
            fn start_profiling(
                &mut self,
                _name: *const core::ffi::c_char,
                _chain_id: crate::runtime::core::event_tracer::ChainID,
                _debug_handle: DebugHandle,
            ) -> EventTracerEntry {
                unimplemented!()
            }
            fn start_profiling_delegate(
                &mut self,
                _name: *const core::ffi::c_char,
                _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            ) -> EventTracerEntry {
                unimplemented!()
            }
            fn end_profiling_delegate(
                &mut self,
                _e: EventTracerEntry,
                _m: *const core::ffi::c_void,
                _l: usize,
            ) {
                unimplemented!()
            }
            fn log_profiling_delegate(
                &mut self,
                _name: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _s: crate::runtime::platform::types::et_timestamp_t,
                _e: crate::runtime::platform::types::et_timestamp_t,
                _m: *const core::ffi::c_void,
                _l: usize,
            ) {
                unimplemented!()
            }
            fn end_profiling(&mut self, _p: EventTracerEntry) {
                unimplemented!()
            }
            fn track_allocation(
                &mut self,
                _id: crate::runtime::core::event_tracer::AllocatorID,
                _size: usize,
            ) {
                unimplemented!()
            }
            fn track_allocator(
                &mut self,
                _name: *const core::ffi::c_char,
            ) -> crate::runtime::core::event_tracer::AllocatorID {
                unimplemented!()
            }
            fn log_evalue(
                &mut self,
                _e: &EValue,
                _t: crate::runtime::core::event_tracer::LoggedEValueType,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn log_intermediate_output_delegate_tensor(
                &mut self,
                _n: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _o: &crate::runtime::core::portable_type::tensor::Tensor,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn log_intermediate_output_delegate_tensor_array(
                &mut self,
                _n: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _o: crate::runtime::core::array_ref::ArrayRef<
                    crate::runtime::core::portable_type::tensor::Tensor,
                >,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn log_intermediate_output_delegate_int(
                &mut self,
                _n: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _o: &i32,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn log_intermediate_output_delegate_bool(
                &mut self,
                _n: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _o: &bool,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn log_intermediate_output_delegate_double(
                &mut self,
                _n: *const core::ffi::c_char,
                _i: crate::runtime::core::event_tracer::DelegateDebugIntId,
                _o: &f64,
            ) -> crate::runtime::core::result::Result<bool> {
                unimplemented!()
            }
            fn set_delegation_intermediate_output_filter(
                &mut self,
                _f: *mut dyn crate::runtime::core::event_tracer::EventTracerFilterBase,
            ) {
                unimplemented!()
            }
        }

        let mut tracer = MinimalTracer {
            state_: EventTracerState::default(),
        };
        assert_eq!(tracer.bundled_input_index(), K_UNSET_BUNDLED_INPUT_INDEX);

        let tracer_ptr: *mut dyn EventTracer = &mut tracer;
        event_tracer_set_bundled_input_index(tracer_ptr, 5);
        assert_eq!(tracer.bundled_input_index(), 5);

        // Null tracer: no-op, must not dereference.
        let null_ptr: *mut dyn EventTracer =
            core::ptr::null_mut::<MinimalTracer>() as *mut dyn EventTracer;
        event_tracer_set_bundled_input_index(null_ptr, 9);
        assert_eq!(tracer.bundled_input_index(), 5);
    }
}
