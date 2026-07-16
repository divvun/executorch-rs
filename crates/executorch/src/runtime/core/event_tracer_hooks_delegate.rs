//! Literal port of runtime/core/event_tracer_hooks_delegate.h.
//!
//! This file contains the hooks that can be used by runtime delegate backend
//! authors to log profiling and debugging events from backend code.
//!
//! The benefit of defining these hooks is that we can easily control whether or
//! not we want to compile in the EventTracer code based on the status of the
//! ET_EVENT_TRACER_ENABLED flag.
//!
//! PORT-NOTE: `ET_EVENT_TRACER_ENABLED` maps to the `event-tracer` cargo
//! feature, and `EventTracer*` to `*mut dyn EventTracer`, as in
//! event_tracer_hooks.rs.

use crate::runtime::core::array_ref::ArrayRef;
#[cfg(feature = "event-tracer")]
use crate::runtime::core::event_tracer::DelegateDebugIntId;
use crate::runtime::core::event_tracer::{DebugHandle, EventTracer, EventTracerEntry};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::platform::types::et_timestamp_t;

/// Start the profiling of a delegate event. Similar to start_profiling it will
/// return an instance of EventTracerEntry that contains the details of this
/// event. Can be left in production code as these hooks compile conditionally.
// [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn]
// [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn]
#[inline]
pub fn event_tracer_start_profiling_delegate(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
    delegate_debug_id: DebugHandle,
) -> EventTracerEntry {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            // PORT-NOTE: bug-for-bug — the C++ passes a `DebugHandle`
            // (`uint32_t`) `delegate_debug_id` into `start_profiling_delegate`
            // whose parameter is `DelegateDebugIntId` (`int32_t`), relying on an
            // implicit narrowing conversion. Reproduced with an explicit `as`.
            return unsafe {
                (*event_tracer)
                    .start_profiling_delegate(name, delegate_debug_id as DelegateDebugIntId)
            };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = name;
        let _ = delegate_debug_id;
    }
    // There is no active tracer; this value will be ignored.
    EventTracerEntry::default()
}

/// Signal the end of the delegate profiling event contained in
/// event_tracer_entry. Users also have the option to log some some free-from
/// string based metadata along with this. Can be left in production code as
/// these hooks compile conditionally.
// [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn]
// [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn]
#[inline]
pub fn event_tracer_end_profiling_delegate(
    event_tracer: *mut dyn EventTracer,
    event_tracer_entry: EventTracerEntry,
    metadata: *const core::ffi::c_void,
    metadata_len: usize,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            unsafe {
                (*event_tracer).end_profiling_delegate(event_tracer_entry, metadata, metadata_len)
            };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer_entry;
        let _ = metadata;
        let _ = metadata_len;
    }
}

/// Some delegates get access to the profiling details only after the complete
/// graph has been executed. This interface is to support such use cases. It
/// can be called in a loop etc. to log any number of profiling events that are
/// part of this delegate. Can be left in production code as these hooks
/// compile conditionally.
// [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-profiling-delegate-fn]
// [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-profiling-delegate-fn]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn event_tracer_log_profiling_delegate(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
    delegate_debug_id: DebugHandle,
    start_time: et_timestamp_t,
    end_time: et_timestamp_t,
    metadata: *const core::ffi::c_void,
    metadata_len: usize,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            // PORT-NOTE: same `DebugHandle`->`DelegateDebugIntId` narrowing as
            // event_tracer_start_profiling_delegate, ported bug-for-bug.
            unsafe {
                (*event_tracer).log_profiling_delegate(
                    name,
                    delegate_debug_id as DelegateDebugIntId,
                    start_time,
                    end_time,
                    metadata,
                    metadata_len,
                )
            };
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = name;
        let _ = delegate_debug_id;
        let _ = start_time;
        let _ = end_time;
        let _ = metadata;
        let _ = metadata_len;
    }
}

/// PORT-NOTE: the C++ `event_tracer_log_output_delegate<T>` is a function
/// template whose `static_assert` restricts `T` to exactly `int`, `bool`,
/// `double`, `executorch::aten::Tensor`, or `ArrayRef<executorch::aten::Tensor>`
/// and then dispatches to the matching `log_intermediate_output_delegate`
/// overload. This trait `DelegateOutput` is implemented only for those five
/// types (so any other `T` is a compile error, mirroring the static_assert) and
/// carries the overload selection. The generic `event_tracer_log_output_delegate`
/// is bounded on it.
pub trait DelegateOutput {
    #[cfg(feature = "event-tracer")]
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    );
}

#[cfg(feature = "event-tracer")]
impl DelegateOutput for &i32 {
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    ) {
        let _ = unsafe {
            (*event_tracer).log_intermediate_output_delegate_int(name, delegate_debug_id, self)
        };
    }
}

#[cfg(feature = "event-tracer")]
impl DelegateOutput for &bool {
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    ) {
        let _ = unsafe {
            (*event_tracer).log_intermediate_output_delegate_bool(name, delegate_debug_id, self)
        };
    }
}

#[cfg(feature = "event-tracer")]
impl DelegateOutput for &f64 {
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    ) {
        let _ = unsafe {
            (*event_tracer).log_intermediate_output_delegate_double(name, delegate_debug_id, self)
        };
    }
}

#[cfg(feature = "event-tracer")]
impl<'a> DelegateOutput for &Tensor<'a> {
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    ) {
        let _ = unsafe {
            (*event_tracer).log_intermediate_output_delegate_tensor(name, delegate_debug_id, self)
        };
    }
}

#[cfg(feature = "event-tracer")]
impl<'a> DelegateOutput for ArrayRef<Tensor<'a>> {
    fn log_intermediate_output_delegate(
        self,
        event_tracer: *mut dyn EventTracer,
        name: *const core::ffi::c_char,
        delegate_debug_id: DelegateDebugIntId,
    ) {
        let _ = unsafe {
            (*event_tracer).log_intermediate_output_delegate_tensor_array(
                name,
                delegate_debug_id,
                self,
            )
        };
    }
}

#[cfg(not(feature = "event-tracer"))]
impl DelegateOutput for &i32 {}
#[cfg(not(feature = "event-tracer"))]
impl DelegateOutput for &bool {}
#[cfg(not(feature = "event-tracer"))]
impl DelegateOutput for &f64 {}
#[cfg(not(feature = "event-tracer"))]
impl<'a> DelegateOutput for &Tensor<'a> {}
#[cfg(not(feature = "event-tracer"))]
impl<'a> DelegateOutput for ArrayRef<Tensor<'a>> {}

/// This templated interfaces can be called in a loop etc. to log any number of
/// debug events that are part of this delegate. Supported values types are int,
/// bool, double, tensor and array of tensors. Can be left in production code as
/// these hooks compile conditionally.
// [spec:et:def:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn]
// [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn]
#[inline]
pub fn event_tracer_log_output_delegate<T: DelegateOutput>(
    event_tracer: *mut dyn EventTracer,
    name: *const core::ffi::c_char,
    delegate_debug_id: DebugHandle,
    output: T,
) {
    #[cfg(feature = "event-tracer")]
    {
        if !event_tracer.is_null() {
            // PORT-NOTE: same `DebugHandle`->`DelegateDebugIntId` narrowing as
            // the other delegate hooks; the C++ passes the `DebugHandle` through
            // to `log_intermediate_output_delegate` whose id parameter is
            // `DelegateDebugIntId`.
            output.log_intermediate_output_delegate(
                event_tracer,
                name,
                delegate_debug_id as DelegateDebugIntId,
            );
        }
    }
    #[cfg(not(feature = "event-tracer"))]
    {
        let _ = event_tracer;
        let _ = name;
        let _ = delegate_debug_id;
        let _ = output;
    }
}

// PORT-NOTE: no C++ counterpart — event_tracer_test.cpp's
// RunSimpleTracerTestDelegate stops at the profiling delegate hooks and never
// calls event_tracer_log_output_delegate. These focused tests pin it, following
// the event_tracer_hooks.rs pattern (a minimal in-module EventTracer mock).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::event_tracer::{
        AllocatorID, ChainID, DelegateDebugIntId, EventTracerFilterBase, EventTracerState,
        LoggedEValueType,
    };
    use crate::runtime::core::result::Result;

    // Records only the log_intermediate_output_delegate_* overloads that the
    // hook under test dispatches to; everything else is unreachable here.
    #[allow(dead_code)]
    struct RecordingTracer {
        state_: EventTracerState,
        last_name_: *const core::ffi::c_char,
        last_id_: DelegateDebugIntId,
        last_int_: i32,
        int_calls_: usize,
        last_double_: f64,
        double_calls_: usize,
    }

    // `new()` is only called by the feature-gated dispatch test below.
    #[allow(dead_code)]
    impl RecordingTracer {
        fn new() -> Self {
            RecordingTracer {
                state_: EventTracerState::default(),
                last_name_: core::ptr::null(),
                last_id_: -1,
                last_int_: 0,
                int_calls_: 0,
                last_double_: 0.0,
                double_calls_: 0,
            }
        }
    }

    impl EventTracer for RecordingTracer {
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
            _chain_id: ChainID,
            _debug_handle: DebugHandle,
        ) -> EventTracerEntry {
            unimplemented!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
        ) -> EventTracerEntry {
            unimplemented!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unimplemented!()
        }
        fn log_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _start_time: et_timestamp_t,
            _end_time: et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unimplemented!()
        }
        fn end_profiling(&mut self, _prof_entry: EventTracerEntry) {
            unimplemented!()
        }
        fn track_allocation(&mut self, _id: AllocatorID, _size: usize) {
            unimplemented!()
        }
        fn track_allocator(&mut self, _name: *const core::ffi::c_char) -> AllocatorID {
            unimplemented!()
        }
        fn log_evalue(&mut self, _evalue: &EValue, _evalue_type: LoggedEValueType) -> Result<bool> {
            unimplemented!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &Tensor,
        ) -> Result<bool> {
            unimplemented!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: ArrayRef<Tensor>,
        ) -> Result<bool> {
            unimplemented!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            name: *const core::ffi::c_char,
            delegate_debug_index: DelegateDebugIntId,
            output: &i32,
        ) -> Result<bool> {
            self.last_name_ = name;
            self.last_id_ = delegate_debug_index;
            self.last_int_ = *output;
            self.int_calls_ += 1;
            Ok(true)
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &bool,
        ) -> Result<bool> {
            unimplemented!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            name: *const core::ffi::c_char,
            delegate_debug_index: DelegateDebugIntId,
            output: &f64,
        ) -> Result<bool> {
            self.last_name_ = name;
            self.last_id_ = delegate_debug_index;
            self.last_double_ = *output;
            self.double_calls_ += 1;
            Ok(true)
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn EventTracerFilterBase,
        ) {
            unimplemented!()
        }
    }

    // Null tracer: no-op, must not dereference — with the `event-tracer`
    // feature on this exercises the `if (event_tracer)` guard; with it off,
    // the !ET_EVENT_TRACER_ENABLED branch that discards every argument.
    // [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn/test]
    #[test]
    fn event_tracer_hooks_delegate_test_log_output_delegate_null_tracer_noop() {
        let null_ptr: *mut dyn EventTracer =
            core::ptr::null_mut::<RecordingTracer>() as *mut dyn EventTracer;
        event_tracer_log_output_delegate(null_ptr, c"test_event".as_ptr(), 1, &5i32);
        event_tracer_log_output_delegate(null_ptr, core::ptr::null(), 1, &1.5f64);
        event_tracer_log_output_delegate(null_ptr, c"test_event".as_ptr(), 1, &true);
    }

    // With a live tracer the hook forwards name/output verbatim to the
    // per-type log_intermediate_output_delegate overload, narrowing the
    // DebugHandle id to DelegateDebugIntId (bug-for-bug with the C++).
    // [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-output-delegate-fn/test]
    #[cfg(feature = "event-tracer")]
    #[test]
    fn event_tracer_hooks_delegate_test_log_output_delegate_dispatches_by_type() {
        let mut tracer = RecordingTracer::new();
        let tracer_ptr: *mut dyn EventTracer = &mut tracer;

        let name = c"test_event";
        event_tracer_log_output_delegate(tracer_ptr, name.as_ptr(), 7 as DebugHandle, &42i32);
        assert_eq!(tracer.int_calls_, 1);
        assert_eq!(tracer.last_name_, name.as_ptr());
        assert_eq!(tracer.last_id_, 7);
        assert_eq!(tracer.last_int_, 42);

        event_tracer_log_output_delegate(tracer_ptr, name.as_ptr(), 9 as DebugHandle, &2.5f64);
        assert_eq!(tracer.double_calls_, 1);
        assert_eq!(tracer.last_id_, 9);
        assert_eq!(tracer.last_double_, 2.5);
        // The int overload was not re-invoked for the double output.
        assert_eq!(tracer.int_calls_, 1);
    }
}
