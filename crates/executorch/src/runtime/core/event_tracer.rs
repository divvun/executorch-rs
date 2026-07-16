//! Literal port of runtime/core/event_tracer.h.
//!
//! PORT-NOTE: The C++ `EventTracer` is an abstract base class mixing pure
//! virtual methods (the tracer contract) with non-virtual inline getters/setters
//! over protected data members. Per PORTING.md (virtual interfaces -> traits),
//! this is modeled as an `EventTracer` trait carrying both the abstract methods
//! and, as provided default methods over an accessor to a shared
//! `EventTracerState` struct, the concrete getters/setters. The protected data
//! members live in `EventTracerState`; an implementor stores one and returns
//! `&mut` / `&` to it via `state_mut()` / `state()`, mirroring the base-class
//! storage. This keeps the non-virtual accessor bodies literal while allowing
//! trait-object (`dyn EventTracer`) dispatch where C++ held `EventTracer*`.

use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::evalue::EValue;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::result::Result;
use crate::runtime::platform::types::et_timestamp_t;

/// Represents an allocator id returned by track_allocator.
// [spec:et:def:event-tracer.executorch.runtime.allocator-id]
pub type AllocatorID = u32;
/// Represents the chain id that will be passed in by the user during
/// event logging.
// [spec:et:def:event-tracer.executorch.runtime.chain-id]
pub type ChainID = i32;
/// Represents the debug handle that is generally associated with each
/// op executed in the runtime.
// [spec:et:def:event-tracer.executorch.runtime.debug-handle]
pub type DebugHandle = u32;
// Represents the delegate debug id that is generally associated with each
// delegate event.
// [spec:et:def:event-tracer.executorch.runtime.delegate-debug-int-id]
pub type DelegateDebugIntId = i32;

/// Default id's for chain id and debug handle.
pub const K_UNSET_CHAIN_ID: ChainID = -1;
pub const K_UNSET_DEBUG_HANDLE: DebugHandle = 0;
pub const K_UNSET_DELEGATE_DEBUG_INT_ID: DelegateDebugIntId = -1;
// Default bundled input index to indicate that it hasn't been set yet.
pub const K_UNSET_BUNDLED_INPUT_INDEX: i32 = -1;

/// Different types of delegate debug identifiers that are supported currently.
// [spec:et:def:event-tracer.executorch.runtime.delegate-debug-id-type]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DelegateDebugIdType {
    /// Default value, indicates that it's not a delegate event.
    KNone,
    /// Indicates a delegate event logged using an integer delegate debug
    /// identifier.
    KInt,
    /// Indicates a delegate event logged using a string delegate debug
    /// identifier i.e. the delegate debug id is a pointer to a string table
    /// managed by the class implementing EventTracer functionality.
    KStr,
}

/// Indicates the type of the EValue that was logged. These values could be
/// serialized and should not be changed.
// [spec:et:def:event-tracer.executorch.runtime.logged-e-value-type]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoggedEValueType {
    /// Intermediate output from an operator.
    KIntermediateOutput = 0,
    /// Output at the program level. This is essentially the output
    /// of the model.
    KProgramOutput = 1,
}

/// Indicates the level of event tracer debug logging. Verbosity of the logging
/// increases as we go down the enum list.
// [spec:et:def:event-tracer.executorch.runtime.event-tracer-debug-log-level]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EventTracerDebugLogLevel {
    /// No logging.
    KNoLogging,
    /// When set to this only the program level outputs will be logged.
    KProgramOutputs,
    /// When set to this all intermediate outputs and program level outputs
    /// will be logged.
    KIntermediateOutputs,
}

/// EventTracerFilterBase is an abstract base class that provides an interface
/// for filtering events based on their name or delegate debug index.
/// Derived classes should implement the filter method to define specific
/// filtering logic.
// [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base]
// [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base.event-tracer-filter-base-fn]
// [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.event-tracer-filter-base-fn]
pub trait EventTracerFilterBase {
    /// Filters events based on the given name or delegate debug index.
    ///
    /// Note that only one of either the name or delegate_debug_index should be
    /// passed in.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn]
    fn filter(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
    ) -> Result<bool>;
}

/// Indicates the level of profiling that should be enabled. Profiling
/// events will be logged in increasing order of verbosity as we go down the
/// enum list. Thus it is important to keep the enum values in the right order.
// [spec:et:def:event-tracer.executorch.runtime.event-tracer-profiling-level]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EventTracerProfilingLevel {
    /// No operator profiling.
    KProfileMethodOnly,
    /// All profiling events enabled.
    KProfileAllEvents,
}

/// This is the struct which should be returned when a profiling event is
/// started. This is used to uniquely identify that profiling event and will be
/// required to be passed into the end_profiling call to signal that the event
/// identified by this struct has completed.
// [spec:et:def:event-tracer.executorch.runtime.event-tracer-entry]
#[derive(Clone, Copy)]
pub struct EventTracerEntry {
    /// An event id to uniquely identify this event that was generated during a
    /// call to start the tracking of an event.
    pub event_id: i64,
    /// The chain to which this event belongs to.
    pub chain_id: ChainID,
    /// The debug handle corresponding to this event.
    pub debug_handle: DebugHandle,
    /// The time at which this event was started to be tracked.
    pub start_time: et_timestamp_t,
    /// When delegate_event_id_type != DelegateDebugIdType::kNone it indicates
    /// that event_id represents a delegate event.
    pub delegate_event_id_type: DelegateDebugIdType,
}

/// PORT-NOTE: the C++ `EventTracerEntry` has no explicit constructor; the hooks
/// return a default-constructed value (`EventTracerEntry()`) whose fields are
/// value-initialized (zero/`kNone`) and ignored by callers. `Default` mirrors
/// that value-initialized default.
impl Default for EventTracerEntry {
    fn default() -> Self {
        EventTracerEntry {
            event_id: 0,
            chain_id: 0,
            debug_handle: 0,
            start_time: 0,
            delegate_event_id_type: DelegateDebugIdType::KNone,
        }
    }
}

/// PORT-NOTE: the protected data members of the abstract `EventTracer` base
/// class. An implementor embeds one of these and exposes it via
/// `EventTracer::state`/`state_mut`, letting the base's non-virtual accessors be
/// ported as default trait methods over that shared storage.
pub struct EventTracerState {
    pub chain_id_: ChainID,
    pub debug_handle_: DebugHandle,
    pub event_tracer_enable_debugging_: bool,
    pub log_intermediate_tensors_: bool,
    pub bundled_input_index_: i32,
    pub event_tracer_debug_level_: EventTracerDebugLogLevel,
    pub event_tracer_profiling_level_: EventTracerProfilingLevel,
}

impl Default for EventTracerState {
    fn default() -> Self {
        EventTracerState {
            chain_id_: K_UNSET_CHAIN_ID,
            debug_handle_: K_UNSET_DEBUG_HANDLE,
            event_tracer_enable_debugging_: false,
            log_intermediate_tensors_: false,
            bundled_input_index_: K_UNSET_BUNDLED_INPUT_INDEX,
            event_tracer_debug_level_: EventTracerDebugLogLevel::KNoLogging,
            event_tracer_profiling_level_: EventTracerProfilingLevel::KProfileAllEvents,
        }
    }
}

/// EventTracer is a class that users can inherit and implement to
/// log/serialize/stream etc. the profiling and debugging events that are
/// generated at runtime for a model. An example of this is the ETDump
/// implementation in the devtools codebase that serializes these events to a
/// flatbuffer.
// [spec:et:def:event-tracer.executorch.runtime.event-tracer]
// [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-fn]
// [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-fn]
pub trait EventTracer {
    /// Accessor to the base-class protected data members (see PORT-NOTE on
    /// `EventTracerState`); backs the non-virtual accessors/mutators below.
    fn state(&self) -> &EventTracerState;
    fn state_mut(&mut self) -> &mut EventTracerState;

    /// Start a new event block (can consist of profiling and/or debugging
    /// events.) identified by this name.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.create-event-block-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.create-event-block-fn]
    fn create_event_block(&mut self, name: *const core::ffi::c_char);

    /// Start the profiling of the event identified by name and debug_handle.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-fn]
    fn start_profiling(
        &mut self,
        name: *const core::ffi::c_char,
        chain_id: ChainID,
        debug_handle: DebugHandle,
    ) -> EventTracerEntry;

    /// Start the profiling of a delegate event.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn]
    fn start_profiling_delegate(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
    ) -> EventTracerEntry;

    /// Signal the end of the delegate profiling event contained in
    /// event_tracer_entry.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn]
    fn end_profiling_delegate(
        &mut self,
        event_tracer_entry: EventTracerEntry,
        metadata: *const core::ffi::c_void,
        metadata_len: usize,
    );

    /// Some delegates get access to the profiling details only after the
    /// complete graph has been executed.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-profiling-delegate-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-profiling-delegate-fn]
    fn log_profiling_delegate(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        start_time: et_timestamp_t,
        end_time: et_timestamp_t,
        metadata: *const core::ffi::c_void,
        metadata_len: usize,
    );

    /// End the profiling of the event identified by prof_entry
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.end-profiling-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-fn]
    fn end_profiling(&mut self, prof_entry: EventTracerEntry);

    /// Track this allocation done via a MemoryAllocator which had profiling
    /// enabled on it.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.track-allocation-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocation-fn]
    fn track_allocation(&mut self, id: AllocatorID, size: usize);

    /// Generate an allocator id for this memory allocator that will be used in
    /// the future to identify all the allocations done by this allocator.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.track-allocator-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocator-fn]
    fn track_allocator(&mut self, name: *const core::ffi::c_char) -> AllocatorID;

    /// Log an evalue during the execution of the model.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-evalue-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-evalue-fn]
    fn log_evalue(&mut self, evalue: &EValue, evalue_type: LoggedEValueType) -> Result<bool>;

    /// Log an intermediate tensor output from a delegate.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn]
    fn log_intermediate_output_delegate_tensor(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        output: &Tensor,
    ) -> Result<bool>;

    /// Log an intermediate tensor array output from a delegate.
    fn log_intermediate_output_delegate_tensor_array(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        output: ArrayRef<Tensor>,
    ) -> Result<bool>;

    /// Log an intermediate int output from a delegate.
    fn log_intermediate_output_delegate_int(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        output: &i32,
    ) -> Result<bool>;

    /// Log an intermediate bool output from a delegate.
    fn log_intermediate_output_delegate_bool(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        output: &bool,
    ) -> Result<bool>;

    /// Log an intermediate double output from a delegate.
    fn log_intermediate_output_delegate_double(
        &mut self,
        name: *const core::ffi::c_char,
        delegate_debug_index: DelegateDebugIntId,
        output: &f64,
    ) -> Result<bool>;

    /// Set the filter of event tracer for delegation intermediate outputs.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn]
    fn set_delegation_intermediate_output_filter(
        &mut self,
        event_tracer_filter: *mut dyn EventTracerFilterBase,
    );

    /// Helper function to set the chain id ands debug handle.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn]
    fn set_chain_debug_handle(&mut self, chain_id: ChainID, debug_handle: DebugHandle) {
        self.state_mut().chain_id_ = chain_id;
        self.state_mut().debug_handle_ = debug_handle;
    }

    /// When running a program wrapped in a bundled program, log the bundled
    /// input index of the current bundled input being tested out on this method.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn]
    fn set_bundled_input_index(&mut self, bundled_input_index: i32) {
        self.state_mut().bundled_input_index_ = bundled_input_index;
    }

    /// Return the current bundled input index.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn]
    fn bundled_input_index(&self) -> i32 {
        self.state().bundled_input_index_
    }

    /// Set the level of event tracer debug logging that is desired.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn]
    fn set_event_tracer_debug_level(&mut self, log_level: EventTracerDebugLogLevel) {
        self.state_mut().event_tracer_debug_level_ = log_level;
    }

    /// Return the current level of event tracer debug logging.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn]
    fn event_tracer_debug_level(&self) -> EventTracerDebugLogLevel {
        self.state().event_tracer_debug_level_
    }

    /// Set the level of event tracer profiling that is desired.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn]
    fn set_event_tracer_profiling_level(&mut self, profiling_level: EventTracerProfilingLevel) {
        self.state_mut().event_tracer_profiling_level_ = profiling_level;
    }

    /// Return the current level of event tracer profiling.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn]
    fn event_tracer_profiling_level(&self) -> EventTracerProfilingLevel {
        self.state().event_tracer_profiling_level_
    }

    /// Return the current status of intermediate outputs logging mode.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.intermediate-outputs-logging-status-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.intermediate-outputs-logging-status-fn]
    fn intermediate_outputs_logging_status(&self) -> bool {
        self.state().log_intermediate_tensors_
    }

    /// Get the current chain id.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn]
    fn current_chain_id(&self) -> ChainID {
        self.state().chain_id_
    }

    /// Get the current debug handle.
    // [spec:et:def:event-tracer.executorch.runtime.event-tracer.current-debug-handle-fn]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-debug-handle-fn]
    fn current_debug_handle(&self) -> DebugHandle {
        self.state().debug_handle_
    }
}

// PORT-NOTE: event_tracer_test.cpp begins with `#define ET_EVENT_TRACER_ENABLED`
// before including the hooks, i.e. it exercises the tracer-enabled compile shape.
// That maps to the `event-tracer` cargo feature; the whole ported suite is gated
// on it and only compiles/runs under `--features event-tracer`.
#[cfg(all(test, feature = "event-tracer"))]
mod tests {
    use super::*;
    use crate::runtime::core::event_tracer_hooks::{
        EventTracerProfileInstructionScope, EventTracerProfileMethodScope,
        EventTracerProfileOpScope, event_tracer_begin_profiling_event,
        event_tracer_create_event_block, event_tracer_end_profiling_event, event_tracer_log_evalue,
        event_tracer_log_evalue_output, event_tracer_track_allocation,
        event_tracer_track_allocator,
    };
    use crate::runtime::core::event_tracer_hooks_delegate::{
        event_tracer_end_profiling_delegate, event_tracer_log_profiling_delegate,
        event_tracer_start_profiling_delegate,
    };
    use crate::runtime::core::result::ResultExt;

    // class DummyEventTracer : public EventTracer
    struct DummyEventTracer {
        state_: EventTracerState,
        logged_evalue_: EValue<'static>,
        logged_evalue_type_: LoggedEValueType,
        event_name_: [u8; 1024],
    }

    impl DummyEventTracer {
        fn new() -> Self {
            DummyEventTracer {
                state_: EventTracerState::default(),
                // EValue logged_evalue_ = EValue(false);
                logged_evalue_: EValue::from_bool(false),
                logged_evalue_type_: LoggedEValueType::KIntermediateOutput,
                event_name_: [0u8; 1024],
            }
        }

        // EValue logged_evalue() { return logged_evalue_; }
        fn logged_evalue(&self) -> EValue<'static> {
            EValue::from_ref(&self.logged_evalue_)
        }

        // LoggedEValueType logged_evalue_type() { return logged_evalue_type_; }
        fn logged_evalue_type(&self) -> LoggedEValueType {
            self.logged_evalue_type_
        }

        // char* get_event_name() { return event_name_; }
        //
        // Returned as a `&str` truncated at the NUL, so tests can compare like the
        // C++ `strcmp(get_event_name(), "...")`.
        fn get_event_name(&self) -> &str {
            let end = self
                .event_name_
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(self.event_name_.len());
            core::str::from_utf8(&self.event_name_[..end]).unwrap()
        }

        // void reset_logged_value() { logged_evalue_ = EValue(false); }
        fn reset_logged_value(&mut self) {
            self.logged_evalue_ = EValue::from_bool(false);
        }
    }

    impl EventTracer for DummyEventTracer {
        fn state(&self) -> &EventTracerState {
            &self.state_
        }
        fn state_mut(&mut self) -> &mut EventTracerState {
            &mut self.state_
        }

        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {}

        fn start_profiling(
            &mut self,
            name: *const core::ffi::c_char,
            _chain_id: ChainID,
            _debug_handle: DebugHandle,
        ) -> EventTracerEntry {
            // ET_CHECK(strlen(name) + 1 < sizeof(event_name_));
            // memcpy(event_name_, name, strlen(name) + 1);
            let cstr = unsafe { core::ffi::CStr::from_ptr(name) };
            let bytes = cstr.to_bytes_with_nul();
            assert!(bytes.len() < self.event_name_.len());
            self.event_name_[..bytes.len()].copy_from_slice(bytes);
            EventTracerEntry::default()
        }

        fn end_profiling(&mut self, _prof_entry: EventTracerEntry) {
            // memset(event_name_, 0, sizeof(event_name_));
            self.event_name_ = [0u8; 1024];
        }

        fn track_allocation(&mut self, _id: AllocatorID, _size: usize) {}

        fn track_allocator(&mut self, _name: *const core::ffi::c_char) -> AllocatorID {
            0
        }

        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
        ) -> EventTracerEntry {
            EventTracerEntry::default()
        }

        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
        }

        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn EventTracerFilterBase,
        ) {
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
        }

        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &Tensor,
        ) -> Result<bool> {
            Ok(true)
        }

        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: ArrayRef<Tensor>,
        ) -> Result<bool> {
            Ok(true)
        }

        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &i32,
        ) -> Result<bool> {
            Ok(true)
        }

        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &bool,
        ) -> Result<bool> {
            Ok(true)
        }

        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &f64,
        ) -> Result<bool> {
            Ok(true)
        }

        // Result<bool> log_evalue(const EValue& evalue, LoggedEValueType type)
        fn log_evalue(&mut self, evalue: &EValue, evalue_type: LoggedEValueType) -> Result<bool> {
            // logged_evalue_ = evalue;
            self.logged_evalue_ = EValue::from_ref(unsafe {
                // Extend the borrow to 'static: the stored EValue only ever holds
                // a bool payload in these tests (no borrowed Tensor), so there is
                // no dangling reference. Mirrors the C++ by-value copy.
                core::mem::transmute::<&EValue, &EValue<'static>>(evalue)
            });
            self.logged_evalue_type_ = evalue_type;
            Ok(true)
        }
    }

    // Exercise all the event_tracer API's for a basic sanity check.
    fn run_simple_tracer_test(event_tracer: *mut dyn EventTracer) {
        let example = c"ExampleEvent";
        event_tracer_create_event_block(event_tracer, example.as_ptr());
        event_tracer_create_event_block(event_tracer, example.as_ptr());
        let event_entry = event_tracer_begin_profiling_event(event_tracer, example.as_ptr());
        event_tracer_end_profiling_event(event_tracer, event_entry);
        {
            let _event_tracer_profile_scope =
                EventTracerProfileMethodScope::new(event_tracer, c"ExampleScope".as_ptr());
        }
        {
            let _event_tracer_profile_instruction_scope =
                EventTracerProfileInstructionScope::new(event_tracer, 0, 1);
        }
        let allocator_id = event_tracer_track_allocator(event_tracer, c"AllocatorName".as_ptr());
        event_tracer_track_allocation(event_tracer, allocator_id, 64);
    }

    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-create-event-block-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-begin-profiling-event-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-end-profiling-event-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocator-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-track-allocation-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-instruction-scope.event-tracer-profile-instruction-scope-fn/test]
    // The hooks driven here dispatch through the tracer's own methods; this test
    // exercises the following EventTracer trait methods via the DummyEventTracer:
    // create_event_block, start_profiling, end_profiling, track_allocator,
    // track_allocation.
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.create-event-block-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocator-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.track-allocation-fn/test]
    #[test]
    fn test_event_tracer_simple_event_tracer_test() {
        // Call all the EventTracer macro's with a valid pointer to an event
        // tracer and also with a null pointer (to test that the null case works).
        let mut dummy = DummyEventTracer::new();
        let dummy_ptr: *mut dyn EventTracer = &mut dummy;
        let dummy_event_tracer_arr: [*mut DummyEventTracer; 2] =
            [&mut dummy as *mut DummyEventTracer, core::ptr::null_mut()];
        for _i in 0..dummy_event_tracer_arr.len() {
            run_simple_tracer_test(dummy_ptr);
            run_simple_tracer_test(null_tracer());
        }
    }

    // Exercise all the event_tracer API's for delegates as a basic sanity check.
    fn run_simple_tracer_test_delegate(event_tracer: *mut dyn EventTracer) {
        let event_tracer_entry = event_tracer_start_profiling_delegate(
            event_tracer,
            c"test_event".as_ptr(),
            K_UNSET_DELEGATE_DEBUG_INT_ID as DebugHandle,
        );
        event_tracer_end_profiling_delegate(event_tracer, event_tracer_entry, core::ptr::null(), 0);
        event_tracer_start_profiling_delegate(event_tracer, core::ptr::null(), 1);
        let metadata = c"test_metadata";
        event_tracer_end_profiling_delegate(
            event_tracer,
            event_tracer_entry,
            metadata.as_ptr() as *const core::ffi::c_void,
            metadata.to_bytes().len(),
        );
        event_tracer_log_profiling_delegate(
            event_tracer,
            c"test_event".as_ptr(),
            K_UNSET_DELEGATE_DEBUG_INT_ID as DebugHandle,
            0,
            1,
            core::ptr::null(),
            0,
        );
        event_tracer_log_profiling_delegate(
            event_tracer,
            core::ptr::null(),
            1,
            0,
            1,
            core::ptr::null(),
            0,
        );
    }

    // [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-start-profiling-delegate-fn/test]
    // [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-end-profiling-delegate-fn/test]
    // [spec:et:sem:event-tracer-hooks-delegate.executorch.runtime.event-tracer-log-profiling-delegate-fn/test]
    // The delegate hooks dispatch through the tracer's own delegate methods.
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.start-profiling-delegate-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.end-profiling-delegate-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-profiling-delegate-fn/test]
    #[test]
    fn test_event_tracer_simple_event_tracer_test_delegate() {
        let mut dummy = DummyEventTracer::new();
        let dummy_ptr: *mut dyn EventTracer = &mut dummy;
        let dummy_event_tracer_arr: [*mut DummyEventTracer; 2] =
            [&mut dummy as *mut DummyEventTracer, core::ptr::null_mut()];
        for _i in 0..dummy_event_tracer_arr.len() {
            run_simple_tracer_test_delegate(dummy_ptr);
            run_simple_tracer_test_delegate(null_tracer());
        }
    }

    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-log-evalue-output-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-debug-level-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-evalue-fn/test]
    // The log-evalue hooks gate on `event_tracer_debug_level()`; the assertions
    // on which evalues get logged fail if that getter returns the wrong level.
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-debug-level-fn/test]
    #[test]
    fn test_event_tracer_simple_event_tracer_test_logging() {
        let mut test_eval = EValue::from_bool(true);

        {
            // By default there should be no logging enabled.
            let mut dummy = DummyEventTracer::new();
            let dummy_ptr: *mut dyn EventTracer = &mut dummy;
            event_tracer_log_evalue(dummy_ptr, &mut test_eval);
            assert_eq!(dummy.logged_evalue().to_bool(), false);
        }

        {
            // Enable only program outputs to be logged. So event_tracer_log_evalue
            // should have no effect but event_tracer_log_evalue_output should work.
            let mut dummy = DummyEventTracer::new();
            dummy.set_event_tracer_debug_level(EventTracerDebugLogLevel::KProgramOutputs);
            let dummy_ptr: *mut dyn EventTracer = &mut dummy;
            event_tracer_log_evalue(dummy_ptr, &mut test_eval);
            assert_eq!(dummy.logged_evalue().to_bool(), false);
            event_tracer_log_evalue_output(dummy_ptr, &test_eval);
            assert_eq!(dummy.logged_evalue().to_bool(), true);
            assert_eq!(dummy.logged_evalue_type(), LoggedEValueType::KProgramOutput);
        }

        {
            // Enable all outputs to be logged. So event_tracer_log_evalue and
            // event_tracer_log_evalue_output should both work.
            let mut dummy = DummyEventTracer::new();
            dummy.set_event_tracer_debug_level(EventTracerDebugLogLevel::KIntermediateOutputs);
            let dummy_ptr: *mut dyn EventTracer = &mut dummy;
            event_tracer_log_evalue(dummy_ptr, &mut test_eval);
            assert_eq!(dummy.logged_evalue().to_bool(), true);
            assert_eq!(
                dummy.logged_evalue_type(),
                LoggedEValueType::KIntermediateOutput
            );
            dummy.reset_logged_value();
            event_tracer_log_evalue_output(dummy_ptr, &test_eval);
            assert_eq!(dummy.logged_evalue().to_bool(), true);
            assert_eq!(dummy.logged_evalue_type(), LoggedEValueType::KProgramOutput);
        }

        // Test with nullptr's to make sure it goes through smoothly.
        event_tracer_log_evalue(null_tracer(), &mut test_eval);
        event_tracer_log_evalue_output(null_tracer(), &test_eval);
    }

    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-op-scope.event-tracer-profile-op-scope-fn/test]
    // [spec:et:sem:event-tracer-hooks.executorch.et-runtime-namespace.internal.event-tracer-profile-method-scope.event-tracer-profile-method-scope-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-event-tracer-profiling-level-fn/test]
    // The op scope gates on `event_tracer_profiling_level()`; the event-name
    // assertions fail if that getter returns the wrong level.
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-profiling-level-fn/test]
    #[test]
    fn test_event_tracer_event_tracer_profile_op_control() {
        let mut dummy = DummyEventTracer::new();
        let dummy_ptr: *mut dyn EventTracer = &mut dummy;
        // Op profiling is enabled by default. Test that it works.
        {
            {
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(dummy_ptr, c"ExampleOpScope".as_ptr());
                assert_eq!(dummy.get_event_name(), "ExampleOpScope");
            }
            assert_eq!(dummy.get_event_name(), "");

            // Normal profiling should still work.
            {
                let _event_tracer_profiler_scope =
                    EventTracerProfileMethodScope::new(dummy_ptr, c"ExampleProfilerScope".as_ptr());
                assert_eq!(dummy.get_event_name(), "ExampleProfilerScope");
            }

            dummy.set_event_tracer_profiling_level(EventTracerProfilingLevel::KProfileMethodOnly);

            // Op profiling should be disabled now.
            {
                let _event_tracer_op_scope =
                    EventTracerProfileOpScope::new(dummy_ptr, c"ExampleOpScope".as_ptr());
                assert_eq!(dummy.get_event_name(), "");
            }

            // Normal profiling should still work.
            {
                let _event_tracer_profiler_scope = EventTracerProfileMethodScope::new(
                    dummy_ptr,
                    c"1ExampleProfilerScope".as_ptr(),
                );
                assert_eq!(dummy.get_event_name(), "1ExampleProfilerScope");
            }
        }
    }

    // Helper producing a null `*mut dyn EventTracer` (a fat pointer needs a
    // concrete pointee to synthesize null metadata).
    fn null_tracer() -> *mut dyn EventTracer {
        core::ptr::null_mut::<DummyEventTracer>() as *mut dyn EventTracer
    }

    // PORT-NOTE: no C++ counterpart. Focused unit tests pinning the non-virtual
    // base-class state accessors that the ported C++ suite never exercises
    // directly. Each mirrors the getter/setter body in event_tracer.h and its
    // default value.

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-chain-debug-handle-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-chain-id-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.current-debug-handle-fn/test]
    #[test]
    fn test_event_tracer_chain_and_debug_handle_accessors() {
        let mut dummy = DummyEventTracer::new();
        // Defaults: chain_id_ = kUnsetChainId (-1), debug_handle_ =
        // kUnsetDebugHandle (0).
        assert_eq!(dummy.current_chain_id(), K_UNSET_CHAIN_ID);
        assert_eq!(dummy.current_debug_handle(), K_UNSET_DEBUG_HANDLE);

        dummy.set_chain_debug_handle(7, 42);
        assert_eq!(dummy.current_chain_id(), 7);
        assert_eq!(dummy.current_debug_handle(), 42);
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-bundled-input-index-fn/test]
    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.bundled-input-index-fn/test]
    #[test]
    fn test_event_tracer_bundled_input_index_accessor() {
        let mut dummy = DummyEventTracer::new();
        // Default: bundled_input_index_ = kUnsetBundledInputIndex (-1).
        assert_eq!(dummy.bundled_input_index(), K_UNSET_BUNDLED_INPUT_INDEX);

        dummy.set_bundled_input_index(3);
        assert_eq!(dummy.bundled_input_index(), 3);
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.intermediate-outputs-logging-status-fn/test]
    #[test]
    fn test_event_tracer_intermediate_outputs_logging_status_accessor() {
        let mut dummy = DummyEventTracer::new();
        // Default: log_intermediate_tensors_ = false.
        assert!(!dummy.intermediate_outputs_logging_status());

        dummy.state_mut().log_intermediate_tensors_ = true;
        assert!(dummy.intermediate_outputs_logging_status());
    }

    // PORT-NOTE: no C++ counterpart. The C++ event_tracer_test.cpp declares
    // `set_delegation_intermediate_output_filter`, the five
    // `log_intermediate_output_delegate` overloads, and never constructs an
    // `EventTracerFilterBase`, so none of these are exercised by the ported
    // suite. The tests below pin the abstract contracts of those pure-virtual
    // methods and the two `= default`/empty-body virtual destructors, mirroring
    // the sem-rule contracts in docs/spec/port/runtime/core/event_tracer.md.

    // A concrete EventTracerFilterBase honoring the "exactly one of name /
    // delegate_debug_index is set" contract from the filter-fn sem rule: match
    // an event named "keep", or the delegate debug index 7.
    struct NameOrIndexFilter {
        dropped: *mut bool,
    }

    impl EventTracerFilterBase for NameOrIndexFilter {
        fn filter(
            &mut self,
            name: *const core::ffi::c_char,
            delegate_debug_index: DelegateDebugIntId,
        ) -> Result<bool> {
            if !name.is_null() {
                // name is set -> delegate_debug_index must be unset (-1).
                assert_eq!(delegate_debug_index, K_UNSET_DELEGATE_DEBUG_INT_ID);
                let cstr = unsafe { core::ffi::CStr::from_ptr(name) };
                Ok(cstr.to_bytes() == b"keep")
            } else {
                // name is unset -> a real delegate_debug_index is supplied.
                Ok(delegate_debug_index == 7)
            }
        }
    }

    impl Drop for NameOrIndexFilter {
        fn drop(&mut self) {
            if !self.dropped.is_null() {
                unsafe { *self.dropped = true };
            }
        }
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.filter-fn/test]
    #[test]
    fn test_event_tracer_filter_base_filter() {
        let mut filter = NameOrIndexFilter {
            dropped: core::ptr::null_mut(),
        };
        // Identify by name (delegate_debug_index unset): "keep" matches, others
        // do not.
        assert_eq!(
            filter.filter(c"keep".as_ptr(), K_UNSET_DELEGATE_DEBUG_INT_ID),
            Ok(true)
        );
        assert_eq!(
            filter.filter(c"drop".as_ptr(), K_UNSET_DELEGATE_DEBUG_INT_ID),
            Ok(false)
        );
        // Identify by delegate debug index (name null): 7 matches, others do not.
        assert_eq!(filter.filter(core::ptr::null(), 7), Ok(true));
        assert_eq!(filter.filter(core::ptr::null(), 3), Ok(false));
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer-filter-base.event-tracer-filter-base-fn/test]
    #[test]
    fn test_event_tracer_filter_base_destructor() {
        // The `= default` base destructor performs no teardown of its own; a
        // filter dropped through a `Box<dyn EventTracerFilterBase>` runs exactly
        // the concrete type's Drop.
        let mut dropped = false;
        {
            let _filter: alloc::boxed::Box<dyn EventTracerFilterBase> =
                alloc::boxed::Box::new(NameOrIndexFilter {
                    dropped: &mut dropped as *mut bool,
                });
            assert!(!dropped);
        }
        assert!(dropped);
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.set-delegation-intermediate-output-filter-fn/test]
    #[test]
    fn test_event_tracer_set_delegation_intermediate_output_filter() {
        // DummyEventTracer's override retains no filter (no-op body); the
        // contract here is just that a filter pointer can be installed via the
        // base API. Exercise with both a real filter and a null pointer.
        let mut dummy = DummyEventTracer::new();
        let mut filter = NameOrIndexFilter {
            dropped: core::ptr::null_mut(),
        };
        let filter_ptr: *mut dyn EventTracerFilterBase = &mut filter;
        dummy.set_delegation_intermediate_output_filter(filter_ptr);
        dummy.set_delegation_intermediate_output_filter(
            core::ptr::null_mut::<NameOrIndexFilter>() as *mut dyn EventTracerFilterBase
        );
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.log-intermediate-output-delegate-fn/test]
    #[test]
    fn test_event_tracer_log_intermediate_output_delegate() {
        use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

        let mut dummy = DummyEventTracer::new();
        let tf = TensorFactory::<f32>::new();
        let tensor = tf.make(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

        // The tensor overload named by the def rule, invoked by name with the
        // delegate index unset per the identifier convention.
        assert_eq!(
            dummy.log_intermediate_output_delegate_tensor(
                c"delegate_event".as_ptr(),
                K_UNSET_DELEGATE_DEBUG_INT_ID,
                &tensor,
            ),
            Ok(true)
        );

        // The rest of the overload family, invoked by delegate index with name
        // null.
        let arr = ArrayRef::from_single(&tensor);
        assert_eq!(
            dummy.log_intermediate_output_delegate_tensor_array(core::ptr::null(), 1, arr),
            Ok(true)
        );
        assert_eq!(
            dummy.log_intermediate_output_delegate_int(core::ptr::null(), 1, &5),
            Ok(true)
        );
        assert_eq!(
            dummy.log_intermediate_output_delegate_bool(core::ptr::null(), 1, &true),
            Ok(true)
        );
        assert_eq!(
            dummy.log_intermediate_output_delegate_double(core::ptr::null(), 1, &2.5),
            Ok(true)
        );
    }

    // A tracer that records its own destruction, to pin that dropping through a
    // `Box<dyn EventTracer>` runs the concrete Drop with no base teardown.
    struct DropTrackingTracer {
        state_: EventTracerState,
        dropped: *mut bool,
    }

    impl Drop for DropTrackingTracer {
        fn drop(&mut self) {
            if !self.dropped.is_null() {
                unsafe { *self.dropped = true };
            }
        }
    }

    impl EventTracer for DropTrackingTracer {
        fn state(&self) -> &EventTracerState {
            &self.state_
        }
        fn state_mut(&mut self) -> &mut EventTracerState {
            &mut self.state_
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {}
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: ChainID,
            _debug_handle: DebugHandle,
        ) -> EventTracerEntry {
            EventTracerEntry::default()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
        ) -> EventTracerEntry {
            EventTracerEntry::default()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
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
        }
        fn end_profiling(&mut self, _prof_entry: EventTracerEntry) {}
        fn track_allocation(&mut self, _id: AllocatorID, _size: usize) {}
        fn track_allocator(&mut self, _name: *const core::ffi::c_char) -> AllocatorID {
            0
        }
        fn log_evalue(&mut self, _evalue: &EValue, _evalue_type: LoggedEValueType) -> Result<bool> {
            Ok(true)
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &Tensor,
        ) -> Result<bool> {
            Ok(true)
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: ArrayRef<Tensor>,
        ) -> Result<bool> {
            Ok(true)
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &i32,
        ) -> Result<bool> {
            Ok(true)
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &bool,
        ) -> Result<bool> {
            Ok(true)
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: DelegateDebugIntId,
            _output: &f64,
        ) -> Result<bool> {
            Ok(true)
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn EventTracerFilterBase,
        ) {
        }
    }

    // [spec:et:sem:event-tracer.executorch.runtime.event-tracer.event-tracer-fn/test]
    #[test]
    fn test_event_tracer_destructor() {
        // The empty-body base destructor performs no teardown; dropping the
        // tracer through a `Box<dyn EventTracer>` dispatches to the concrete
        // type's Drop.
        let mut dropped = false;
        {
            let _tracer: alloc::boxed::Box<dyn EventTracer> =
                alloc::boxed::Box::new(DropTrackingTracer {
                    state_: EventTracerState::default(),
                    dropped: &mut dropped as *mut bool,
                });
            assert!(!dropped);
        }
        assert!(dropped);
    }
}
