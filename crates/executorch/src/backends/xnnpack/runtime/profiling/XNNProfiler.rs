//! Literal port of backends/xnnpack/runtime/profiling/XNNProfiler.cpp +
//! backends/xnnpack/runtime/profiling/XNNProfiler.h.
//!
//! PORT-NOTE: The C++ compiles two shapes of `XNNProfiler`:
//!   * the "profiler compiled in" shape, guarded by `#if
//!     defined(ET_EVENT_TRACER_ENABLED) || defined(ENABLE_XNNPACK_PROFILING)`,
//!   * a "stub" shape otherwise, whose methods just return `Error::Ok`.
//! These map to the crate features `event-tracer` (ET_EVENT_TRACER_ENABLED) and
//! `xnnpack-profiling` (ENABLE_XNNPACK_PROFILING). The legacy broad
//! `profiling-enabled` feature also enables this code for compatibility.
//!
//! Depends on the XNNPACK C API (`xnn_runtime_t`, profiling queries), so the
//! module is gated behind the `xnnpack` feature.
#![cfg(feature = "xnnpack")]
#![allow(non_upper_case_globals)]

use crate::runtime::core::error::Error;
use crate::runtime::core::event_tracer::EventTracer;

// [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler-state]
#[cfg(any(
    feature = "event-tracer",
    feature = "xnnpack-profiling",
    feature = "profiling-enabled"
))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XNNProfilerState {
    Uninitialized,
    Ready,
    Running,
}

#[cfg(any(
    feature = "event-tracer",
    feature = "xnnpack-profiling",
    feature = "profiling-enabled"
))]
mod imp {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::{
        xnn_get_runtime_profiling_info, xnn_profile_info_num_operators,
        xnn_profile_info_operator_name, xnn_profile_info_operator_timing, xnn_runtime_t,
        xnn_status_out_of_memory, xnn_status_success,
    };
    use crate::runtime::core::event_tracer::DebugHandle;
    use crate::runtime::core::event_tracer_hooks_delegate::event_tracer_log_profiling_delegate;
    use crate::runtime::platform::platform::{pal_current_ticks, pal_ticks_to_ns_multiplier};
    use crate::runtime::platform::types::et_timestamp_t;

    use std::collections::HashMap;

    use core::ffi::c_void;

    // PORT-NOTE: `ET_CHECK(cond)` aborts on failure. Not exported crate-wide as
    // a name-shared macro; mirrored here as a file-local macro over
    // `runtime_abort`, matching the portable-kernel pattern.
    macro_rules! et_check {
        ($cond:expr) => {
            if !($cond) {
                crate::runtime::platform::abort::runtime_abort();
            }
        };
    }

    // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler]
    pub struct XNNProfiler {
        event_tracer_: *mut dyn EventTracer,
        runtime_: xnn_runtime_t,
        state_: XNNProfilerState,

        op_count_: usize,
        op_names_: Vec<u8>,
        op_timings_: Vec<u64>,
        run_count_: u64,
        start_time_: et_timestamp_t,

        // State needed to track average timing. Only used when
        // ENABLE_XNNPACK_PROFILING is defined.
        #[cfg(any(feature = "xnnpack-profiling", feature = "profiling-enabled"))]
        op_timings_sum_: Vec<u64>,
    }

    impl XNNProfiler {
        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
        //
        // PORT-NOTE: The C++ ctor initializes `state_` and `run_count_`; the
        // remaining members are default-initialized. In Rust every field must be
        // given a value, so `runtime_`/`event_tracer_`/`start_time_` get null /
        // zero placeholders (overwritten by `initialize`/`start`), matching the
        // C++ "left in their default state until populated".
        pub fn new() -> Self {
            XNNProfiler {
                event_tracer_: null_event_tracer(),
                runtime_: xnn_runtime_t(core::ptr::null_mut()),
                state_: XNNProfilerState::Uninitialized,
                op_count_: 0,
                op_names_: Vec::new(),
                op_timings_: Vec::new(),
                run_count_: 0,
                start_time_: 0,
                #[cfg(any(feature = "xnnpack-profiling", feature = "profiling-enabled"))]
                op_timings_sum_: Vec::new(),
            }
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
        #[must_use]
        pub fn initialize(&mut self, runtime: xnn_runtime_t) -> Error {
            self.runtime_ = runtime;

            // Fetch the runtime operator information from XNNPACK.
            crate::et_check_ok_or_return_error!(self.get_runtime_num_operators());
            crate::et_check_ok_or_return_error!(self.get_runtime_operator_names());

            self.state_ = XNNProfilerState::Ready;

            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
        #[must_use]
        pub fn start(&mut self, event_tracer: *mut dyn EventTracer) -> Error {
            // Validate profiler state.
            if self.state_ == XNNProfilerState::Uninitialized {
                crate::et_log!(
                    Error,
                    "XNNProfiler must be initialized prior to calling begin_execution."
                );
                return Error::InvalidState;
            } else if self.state_ == XNNProfilerState::Running {
                crate::et_log!(
                    Error,
                    "XNNProfiler is already running. Call end_execution() before calling begin_execution()."
                );
                return Error::InvalidState;
            }

            self.event_tracer_ = event_tracer;
            self.state_ = XNNProfilerState::Running;

            // Log the start of execution timestamp.
            self.start_time_ = pal_current_ticks();

            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
        #[must_use]
        pub fn end(&mut self) -> Error {
            // Validate profiler state.
            crate::et_check_or_return_error!(
                self.state_ == XNNProfilerState::Running,
                InvalidState,
                "XNNProfiler is not running. Ensure begin_execution() is called before end_execution()."
            );

            // Retrieve operator timing from XNNPACK.
            crate::et_check_ok_or_return_error!(self.get_runtime_operator_timings());

            if !event_tracer_is_null(self.event_tracer_) {
                self.submit_trace();
            }

            self.log_operator_timings();

            self.state_ = XNNProfilerState::Ready;
            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn]
        #[must_use]
        fn get_runtime_operator_names(&mut self) -> Error {
            let mut required_size: usize = 0;

            // First call returns xnn_status_out_of_memory, but sets required_size
            // to the correct size of the buffer to store the result.
            let mut status = unsafe {
                xnn_get_runtime_profiling_info(
                    self.runtime_,                  // runtime
                    xnn_profile_info_operator_name, // param_name
                    0,                              // param_value_size
                    core::ptr::null_mut(),          // param_value
                    &mut required_size,             // param_value_size_ret
                )
            };

            if status == xnn_status_out_of_memory {
                self.op_names_.resize(required_size, 0);
                status = unsafe {
                    xnn_get_runtime_profiling_info(
                        self.runtime_,
                        xnn_profile_info_operator_name,
                        self.op_names_.len(),
                        self.op_names_.as_mut_ptr() as *mut c_void,
                        &mut required_size,
                    )
                };
            }

            if status != xnn_status_success {
                crate::et_log!(Error, "Failed to get XNNPACK operator names: {}", status.0);
                return Error::Internal;
            }

            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]
        #[must_use]
        fn get_runtime_num_operators(&mut self) -> Error {
            let mut required_size: usize = 0;

            let status = unsafe {
                xnn_get_runtime_profiling_info(
                    self.runtime_,
                    xnn_profile_info_num_operators,
                    core::mem::size_of::<usize>(),
                    &mut self.op_count_ as *mut usize as *mut c_void,
                    &mut required_size,
                )
            };

            if status != xnn_status_success {
                crate::et_log!(Error, "Failed to get XNNPACK operator count: {}", status.0);
                return Error::Internal;
            }

            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-timings-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-timings-fn]
        #[must_use]
        fn get_runtime_operator_timings(&mut self) -> Error {
            let mut required_size: usize = 0;

            // Get number of runtime operators for timing_stats.size
            self.op_timings_.resize(self.op_count_, 0);
            let status = unsafe {
                xnn_get_runtime_profiling_info(
                    self.runtime_,
                    xnn_profile_info_operator_timing,
                    self.op_timings_.len() * core::mem::size_of::<u64>(),
                    self.op_timings_.as_mut_ptr() as *mut c_void,
                    &mut required_size,
                )
            };

            if status != xnn_status_success {
                crate::et_log!(Error, "Failed to get XNNPACK operator timing: {}", status.0);
                return Error::Internal;
            }

            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.log-operator-timings-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.log-operator-timings-fn]
        #[cfg(any(feature = "xnnpack-profiling", feature = "profiling-enabled"))]
        fn log_operator_timings(&mut self) {
            // Update running average state and log average timing for each op.
            self.run_count_ += 1;
            let mut name_len: usize = 0;
            let mut total_time: f32 = 0.0f32;

            if self.op_timings_sum_.len() != self.op_count_ {
                self.op_timings_sum_ = vec![0u64; self.op_count_];
            }

            for i in 0..self.op_count_ {
                let op_name = c_string_at(&self.op_names_, name_len);
                name_len += strlen(&self.op_names_, name_len) + 1;

                self.op_timings_sum_[i] += self.op_timings_[i];
                let avg_op_time = self.op_timings_sum_[i] as f32 / self.run_count_ as f32;
                total_time += avg_op_time;

                #[cfg(feature = "xnnpack-profiling")]
                eprintln!(
                    ">>, {}, {} ({})",
                    op_name, self.op_timings_[i], avg_op_time
                );
                #[cfg(all(
                    feature = "profiling-enabled",
                    not(feature = "xnnpack-profiling")
                ))]
                crate::et_log!(
                    Info,
                    ">>, {}, {} ({})",
                    op_name,
                    self.op_timings_[i],
                    avg_op_time
                );
            }
            #[cfg(feature = "xnnpack-profiling")]
            eprintln!(">>, Total Time, {}", total_time);
            #[cfg(all(
                feature = "profiling-enabled",
                not(feature = "xnnpack-profiling")
            ))]
            crate::et_log!(Info, ">>, Total Time, {}", total_time);
        }

        #[cfg(not(any(feature = "xnnpack-profiling", feature = "profiling-enabled")))]
        fn log_operator_timings(&mut self) {
            self.run_count_ += 1;
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.submit-trace-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.submit-trace-fn]
        fn submit_trace(&mut self) {
            // Retrieve the system tick rate (ratio between ticks and
            // nanoseconds).
            let tick_ns_conv_multiplier = pal_ticks_to_ns_multiplier();

            // ET_CHECK(op_timings_.size() == op_count_);
            et_check!(self.op_timings_.len() == self.op_count_);
            let mut name_len: usize = 0;
            let mut time: et_timestamp_t = self.start_time_;
            let mut op_counts: HashMap<String, u32> = HashMap::new();

            for i in 0..self.op_count_ {
                let op_name = c_string_at(&self.op_names_, name_len);
                name_len += strlen(&self.op_names_, name_len) + 1;

                // Format the op name as {name} #{count}.
                let op_name_str = op_name.clone();
                let count = op_counts.entry(op_name_str.clone()).or_insert(0);
                *count += 1;
                let name_formatted = format!("{} #{}", op_name_str, *count);

                // Convert from microseconds (XNNPACK) to PAL ticks (ET).
                let interval_ticks: et_timestamp_t =
                    (self.op_timings_[i] * 1000 * tick_ns_conv_multiplier.denominator
                        / tick_ns_conv_multiplier.numerator) as et_timestamp_t;

                let end_time = time + interval_ticks;

                let name_c = std::ffi::CString::new(name_formatted).unwrap();
                event_tracer_log_profiling_delegate(
                    self.event_tracer_,
                    name_c.as_ptr(),
                    /*delegate_debug_id=*/ (-1i64) as u32 as DebugHandle,
                    time,
                    end_time,
                    core::ptr::null(),
                    0,
                );

                // Assume that the next op starts immediately after the previous
                // op. This may not be strictly true, but it should be close
                // enough.
                time = end_time;
            }
        }
    }

    // PORT-NOTE: In C++ `event_tracer_` is a raw pointer compared against
    // `nullptr`. A `*mut dyn EventTracer` is a fat pointer; a null one is spelled
    // via the never-instantiated `NULL_EVENT_TRACER` idiom used elsewhere, and
    // "is null" checks the data pointer.
    fn event_tracer_is_null(ptr: *mut dyn EventTracer) -> bool {
        ptr.is_null()
    }

    fn null_event_tracer() -> *mut dyn EventTracer {
        NULL_EVENT_TRACER
    }

    const NULL_EVENT_TRACER: *mut dyn EventTracer =
        core::ptr::null_mut::<NullEventTracer>() as *mut dyn EventTracer;

    // Never-instantiated concrete implementor used only to spell the null
    // `*mut dyn EventTracer` placeholder for a freshly-constructed profiler
    // (mirroring the C++ default-initialized `event_tracer_`).
    struct NullEventTracer;
    impl EventTracer for NullEventTracer {
        fn state(&self) -> &crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn state_mut(&mut self) -> &mut crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
            unreachable!()
        }
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: crate::runtime::core::event_tracer::ChainID,
            _debug_handle: crate::runtime::core::event_tracer::DebugHandle,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: crate::runtime::core::event_tracer::EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn log_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _start_time: et_timestamp_t,
            _end_time: et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn end_profiling(
            &mut self,
            _prof_entry: crate::runtime::core::event_tracer::EventTracerEntry,
        ) {
            unreachable!()
        }
        fn track_allocation(
            &mut self,
            _id: crate::runtime::core::event_tracer::AllocatorID,
            _size: usize,
        ) {
            unreachable!()
        }
        fn track_allocator(
            &mut self,
            _name: *const core::ffi::c_char,
        ) -> crate::runtime::core::event_tracer::AllocatorID {
            unreachable!()
        }
        fn log_evalue(
            &mut self,
            _evalue: &crate::runtime::core::evalue::EValue,
            _evalue_type: crate::runtime::core::event_tracer::LoggedEValueType,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &crate::runtime::core::portable_type::tensor::Tensor,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: crate::runtime::core::array_ref::ArrayRef<
                crate::runtime::core::portable_type::tensor::Tensor,
            >,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &i32,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &bool,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &f64,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn crate::runtime::core::event_tracer::EventTracerFilterBase,
        ) {
            unreachable!()
        }
    }

    // PORT-NOTE: The C++ walks the packed name buffer as C strings via
    // `&op_names_[name_len]` and `strlen`. These helpers reproduce that over the
    // `Vec<char>` (ported as `Vec<u8>`): `strlen` measures to the next NUL,
    // `c_string_at` decodes the run of bytes as a UTF-8 `String` (lossy — the
    // op names are ASCII).
    fn strlen(buf: &[u8], start: usize) -> usize {
        let mut end = start;
        while end < buf.len() && buf[end] != 0 {
            end += 1;
        }
        end - start
    }

    fn c_string_at(buf: &[u8], start: usize) -> String {
        let len = strlen(buf, start);
        String::from_utf8_lossy(&buf[start..start + len]).into_owned()
    }
}

#[cfg(any(
    feature = "event-tracer",
    feature = "xnnpack-profiling",
    feature = "profiling-enabled"
))]
pub use imp::XNNProfiler;

// ------------------------------------------------------------------------
// Stub implementation for when profiling is disabled (`#else` branch).
// ------------------------------------------------------------------------
#[cfg(not(any(
    feature = "event-tracer",
    feature = "xnnpack-profiling",
    feature = "profiling-enabled"
)))]
mod stub {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::xnn_runtime_t;

    // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler]
    //
    // PORT-NOTE: The stub build has an empty class (no members) whose methods
    // unconditionally return Ok.
    pub struct XNNProfiler;

    impl XNNProfiler {
        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
        pub fn new() -> Self {
            XNNProfiler
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
        #[must_use]
        pub fn initialize(&mut self, runtime: xnn_runtime_t) -> Error {
            let _ = runtime;
            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
        #[must_use]
        pub fn start(&mut self, event_tracer: *mut dyn EventTracer) -> Error {
            let _ = event_tracer;
            Error::Ok
        }

        // [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
        // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
        #[must_use]
        pub fn end(&mut self) -> Error {
            Error::Ok
        }
    }
}

#[cfg(not(any(
    feature = "event-tracer",
    feature = "xnnpack-profiling",
    feature = "profiling-enabled"
)))]
pub use stub::XNNProfiler;

// ------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------
//
// PORT-NOTE: There is no upstream XNNProfiler_test.cpp; these tests are
// synthesized from the C++ source + sem rules. The state-machine guards
// exercisable WITHOUT a live `xnn_runtime_t` are covered directly: the
// constructor leaves the profiler in `Uninitialized`, so `start()` on a fresh
// profiler short-circuits to `InvalidState` (before any XNNPACK C call or
// `pal_current_ticks()`), and `end()` on a non-`Running` profiler
// short-circuits to `InvalidState` (before `get_runtime_operator_timings()`).
// `initialize`, the `get_runtime_*` queries, `log_operator_timings`, and
// `submit_trace` require a real runtime; `initialize_and_end_profile_real_runtime`
// builds one directly through the linked XNNPACK C API (single binary-add
// subgraph, runtime created with XNN_FLAG_BASIC_PROFILING) — no delegated
// `.pte` needed, XNNPACK is only the driver for the ported query/trace logic.
#[cfg(all(
    test,
    feature = "xnnpack",
    any(
        feature = "event-tracer",
        feature = "xnnpack-profiling",
        feature = "profiling-enabled"
    )
))]
mod tests {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::xnn_runtime_t;
    use crate::extension::module::module::null_event_tracer;

    // A freshly-constructed profiler is `Uninitialized`, so `start()` must
    // reject with `InvalidState` before touching the runtime. This pins both
    // the constructor's `state_ = Uninitialized` initialization and the
    // Uninitialized guard in `start`.
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn/test]
    #[test]
    fn start_before_initialize_is_invalid_state() {
        crate::runtime::platform::platform::pal_init();
        let mut profiler = XNNProfiler::new();
        assert_eq!(profiler.start(null_event_tracer()), Error::InvalidState);
    }

    // `end()` on a profiler that is not `Running` (a fresh `Uninitialized` one)
    // must reject with `InvalidState` before querying operator timings.
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn/test]
    #[test]
    fn end_before_start_is_invalid_state() {
        crate::runtime::platform::platform::pal_init();
        let mut profiler = XNNProfiler::new();
        assert_eq!(profiler.end(), Error::InvalidState);
    }

    // Guard against a `xnn_runtime_t` unused-import warning under the
    // event-tracer-only build (the type documents the runtime the state guards
    // deliberately do not touch).
    #[test]
    fn null_runtime_handle_is_null() {
        let rt = xnn_runtime_t(core::ptr::null_mut());
        assert!(rt.0.is_null());
    }

    // Event tracer that records the delegate profiling events `submit_trace`
    // emits (name per op, formatted "{name} #{count}"). Only
    // `log_profiling_delegate` is reachable from `submit_trace`'s
    // `event_tracer_log_profiling_delegate` hook; every other trait method is
    // unreachable here (mirrors the NullEventTracer pattern).
    struct RecordingTracer {
        events: Vec<String>,
    }

    impl crate::runtime::core::event_tracer::EventTracer for RecordingTracer {
        fn state(&self) -> &crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn state_mut(&mut self) -> &mut crate::runtime::core::event_tracer::EventTracerState {
            unreachable!()
        }
        fn create_event_block(&mut self, _name: *const core::ffi::c_char) {
            unreachable!()
        }
        fn start_profiling(
            &mut self,
            _name: *const core::ffi::c_char,
            _chain_id: crate::runtime::core::event_tracer::ChainID,
            _debug_handle: crate::runtime::core::event_tracer::DebugHandle,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn start_profiling_delegate(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
        ) -> crate::runtime::core::event_tracer::EventTracerEntry {
            unreachable!()
        }
        fn end_profiling_delegate(
            &mut self,
            _event_tracer_entry: crate::runtime::core::event_tracer::EventTracerEntry,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            unreachable!()
        }
        fn log_profiling_delegate(
            &mut self,
            name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _start_time: crate::runtime::platform::types::et_timestamp_t,
            _end_time: crate::runtime::platform::types::et_timestamp_t,
            _metadata: *const core::ffi::c_void,
            _metadata_len: usize,
        ) {
            let name = unsafe { core::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
            self.events.push(name);
        }
        fn end_profiling(
            &mut self,
            _prof_entry: crate::runtime::core::event_tracer::EventTracerEntry,
        ) {
            unreachable!()
        }
        fn track_allocation(
            &mut self,
            _id: crate::runtime::core::event_tracer::AllocatorID,
            _size: usize,
        ) {
            unreachable!()
        }
        fn track_allocator(
            &mut self,
            _name: *const core::ffi::c_char,
        ) -> crate::runtime::core::event_tracer::AllocatorID {
            unreachable!()
        }
        fn log_evalue(
            &mut self,
            _evalue: &crate::runtime::core::evalue::EValue,
            _evalue_type: crate::runtime::core::event_tracer::LoggedEValueType,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &crate::runtime::core::portable_type::tensor::Tensor,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_tensor_array(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: crate::runtime::core::array_ref::ArrayRef<
                crate::runtime::core::portable_type::tensor::Tensor,
            >,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_int(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &i32,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_bool(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &bool,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn log_intermediate_output_delegate_double(
            &mut self,
            _name: *const core::ffi::c_char,
            _delegate_debug_index: crate::runtime::core::event_tracer::DelegateDebugIntId,
            _output: &f64,
        ) -> crate::runtime::core::result::Result<bool> {
            unreachable!()
        }
        fn set_delegation_intermediate_output_filter(
            &mut self,
            _event_tracer_filter: *mut dyn crate::runtime::core::event_tracer::EventTracerFilterBase,
        ) {
            unreachable!()
        }
    }

    // Drives the profiler against a real XNNPACK runtime built directly through
    // the linked C API: a single binary-add fp32 subgraph, compiled into a
    // runtime created with XNN_FLAG_BASIC_PROFILING (the flag the profiling
    // queries require). `initialize` performs the num-operators query
    // (`get_runtime_num_operators`) and the two-phase out_of_memory-then-fill
    // operator-names query (`get_runtime_operator_names`); after a real invoke,
    // `end` performs the operator-timings query
    // (`get_runtime_operator_timings`), walks the packed name buffer and emits
    // one "{name} #{count}" delegate event per op to the non-null tracer
    // (`submit_trace`), and updates the run-count/average state
    // (`log_operator_timings`).
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-timings-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.log-operator-timings-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.submit-trace-fn/test]
    #[test]
    fn initialize_and_end_profile_real_runtime() {
        use crate::backends::xnnpack::runtime::sys::{
            XNN_FLAG_BASIC_PROFILING, XNN_INVALID_VALUE_ID, XNN_VALUE_FLAG_EXTERNAL_INPUT,
            XNN_VALUE_FLAG_EXTERNAL_OUTPUT, pthreadpool_t, xnn_binary_add, xnn_create_runtime_v4,
            xnn_create_subgraph, xnn_create_workspace, xnn_datatype_fp32, xnn_define_binary,
            xnn_define_tensor_value, xnn_delete_runtime, xnn_delete_subgraph, xnn_external_value,
            xnn_initialize, xnn_invoke_runtime, xnn_release_workspace, xnn_setup_runtime,
            xnn_status_success, xnn_subgraph_t, xnn_weights_cache_t, xnn_workspace_t,
        };

        crate::runtime::platform::platform::pal_init();

        let dims: [usize; 4] = [1, 4, 4, 4];
        const N: usize = 64;
        // Extra floats beyond the logical extent provide XNN_EXTRA_BYTES
        // padding for SIMD reads.
        let mut in0 = vec![1.0f32; N + 16];
        let mut in1 = vec![2.0f32; N + 16];
        let mut out = vec![0.0f32; N + 16];

        let mut subgraph = xnn_subgraph_t(core::ptr::null_mut());
        let mut workspace = xnn_workspace_t(core::ptr::null_mut());
        let mut rt = xnn_runtime_t(core::ptr::null_mut());
        unsafe {
            assert_eq!(xnn_initialize(core::ptr::null()), xnn_status_success);
            assert_eq!(xnn_create_subgraph(3, 0, &mut subgraph), xnn_status_success);

            let mut in0_id: u32 = XNN_INVALID_VALUE_ID;
            let mut in1_id: u32 = XNN_INVALID_VALUE_ID;
            let mut out_id: u32 = XNN_INVALID_VALUE_ID;
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    0,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut in0_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    1,
                    XNN_VALUE_FLAG_EXTERNAL_INPUT,
                    &mut in1_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_tensor_value(
                    subgraph,
                    xnn_datatype_fp32,
                    dims.len(),
                    dims.as_ptr(),
                    core::ptr::null(),
                    2,
                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                    &mut out_id,
                ),
                xnn_status_success
            );
            assert_eq!(
                xnn_define_binary(
                    subgraph,
                    xnn_binary_add,
                    core::ptr::null(),
                    in0_id,
                    in1_id,
                    out_id,
                    0,
                ),
                xnn_status_success
            );

            assert_eq!(xnn_create_workspace(&mut workspace), xnn_status_success);
            assert_eq!(
                xnn_create_runtime_v4(
                    subgraph,
                    xnn_weights_cache_t(core::ptr::null_mut()),
                    workspace,
                    pthreadpool_t(core::ptr::null_mut()),
                    XNN_FLAG_BASIC_PROFILING,
                    &mut rt,
                ),
                xnn_status_success
            );
        }

        // initialize(): fetches op count + names from the profiled runtime.
        let mut profiler = XNNProfiler::new();
        assert_eq!(profiler.initialize(rt), Error::Ok);

        // Bind externals and run a real inference so end() reads real timings.
        let externals = [
            xnn_external_value {
                id: 0,
                data: in0.as_mut_ptr() as *mut core::ffi::c_void,
            },
            xnn_external_value {
                id: 1,
                data: in1.as_mut_ptr() as *mut core::ffi::c_void,
            },
            xnn_external_value {
                id: 2,
                data: out.as_mut_ptr() as *mut core::ffi::c_void,
            },
        ];
        unsafe {
            assert_eq!(
                xnn_setup_runtime(rt, externals.len(), externals.as_ptr()),
                xnn_status_success
            );
            assert_eq!(xnn_invoke_runtime(rt), xnn_status_success);
        }
        assert_eq!(out[0], 3.0);

        // start() with a non-null tracer, then end(): timings query + trace
        // submission + timing log; the profiler returns to Ready.
        let mut tracer = RecordingTracer { events: Vec::new() };
        let tracer_ptr = &mut tracer as *mut RecordingTracer
            as *mut dyn crate::runtime::core::event_tracer::EventTracer;
        assert_eq!(profiler.start(tracer_ptr), Error::Ok);
        assert_eq!(profiler.end(), Error::Ok);

        // The `event_tracer_log_profiling_delegate` hook only reaches the
        // tracer when event tracing is compiled in; under
        // profiling-enabled-only builds `submit_trace` still runs (name walk +
        // tick conversion) but the hook is a no-op.
        #[cfg(feature = "event-tracer")]
        {
            assert!(!tracer.events.is_empty());
            // Every submitted event carries the "{name} #{count}" format.
            for event in &tracer.events {
                assert!(event.contains(" #"), "unexpected event name: {}", event);
            }
        }
        #[cfg(not(feature = "event-tracer"))]
        assert!(tracer.events.is_empty());

        // Ready again: a second start()+end() round succeeds (run_count grows).
        assert_eq!(profiler.start(tracer_ptr), Error::Ok);
        assert_eq!(profiler.end(), Error::Ok);

        unsafe {
            assert_eq!(xnn_delete_runtime(rt), xnn_status_success);
            assert_eq!(xnn_delete_subgraph(subgraph), xnn_status_success);
            assert_eq!(xnn_release_workspace(workspace), xnn_status_success);
        }
    }
}

// PORT-NOTE: Stub-build tests. When neither profiling feature is enabled the
// C++ compiles the empty `XNNProfiler` whose `initialize`/`start`/`end`
// unconditionally return `Error::Ok`. These pin exactly that.
#[cfg(all(
    test,
    feature = "xnnpack",
    not(any(
        feature = "event-tracer",
        feature = "xnnpack-profiling",
        feature = "profiling-enabled"
    ))
))]
mod stub_tests {
    use super::*;
    use crate::backends::xnnpack::runtime::sys::xnn_runtime_t;
    use crate::extension::module::module::null_event_tracer;

    // In the stub build every method is a no-op returning Ok, and the state
    // guards of the real build are absent.
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn/test]
    // [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn/test]
    #[test]
    fn stub_methods_return_ok() {
        let mut profiler = XNNProfiler::new();
        assert_eq!(
            profiler.initialize(xnn_runtime_t(core::ptr::null_mut())),
            Error::Ok
        );
        assert_eq!(profiler.start(null_event_tracer()), Error::Ok);
        assert_eq!(profiler.end(), Error::Ok);
    }
}
