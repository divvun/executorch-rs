# backends/xnnpack/runtime/profiling/XNNProfiler.cpp, backends/xnnpack/runtime/profiling/XNNProfiler.h

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
> Error XNNProfiler::end()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]
> Ends a profiling session. This behavior applies only when the profiler is
> compiled in (`ET_EVENT_TRACER_ENABLED` or `ENABLE_XNNPACK_PROFILING`); the
> stub build described in `[spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.end-fn]`
> unconditionally returns Ok.
>
> Steps:
> 1. Validate state: if `state_` is not `Running`, log an Error message
>    ("XNNProfiler is not running. Ensure begin_execution() is called before
>    end_execution().") and return `Error::InvalidState`.
> 2. Call `get_runtime_operator_timings()`; if it returns a non-Ok Error,
>    propagate that Error immediately.
> 3. If `event_tracer_` is non-null, call `submit_trace()`.
> 4. Call `log_operator_timings()`.
> 5. Set `state_` to `Ready`.
> 6. Return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]
> Error XNNProfiler::get_runtime_num_operators()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]
> Queries XNNPACK for the number of operators in the runtime and stores it in
> the member `op_count_` (a `size_t`).
>
> Steps:
> 1. Initialize a local `required_size` to 0.
> 2. Call `xnn_get_runtime_profiling_info(runtime_,
>    xnn_profile_info_num_operators, sizeof(op_count_), &op_count_,
>    &required_size)`, requesting the operator count directly into `op_count_`.
> 3. If the returned status is not `xnn_status_success`, log an Error message
>    ("Failed to get XNNPACK operator count: %d" with the numeric status) and
>    return `Error::Internal`.
> 4. Otherwise return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn]
> Error XNNProfiler::get_runtime_operator_names()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn]
> Queries XNNPACK for the concatenated operator-name buffer and stores it in
> the member `op_names_` (a `std::vector<char>`). The buffer holds the operator
> names as consecutive NUL-terminated C strings, one per operator, in operator
> order.
>
> Steps:
> 1. Initialize a local `required_size` to 0.
> 2. Make a sizing call: `xnn_get_runtime_profiling_info(runtime_,
>    xnn_profile_info_operator_name, 0 /*param_value_size*/,
>    nullptr /*param_value*/, &required_size)`. By contract this first call
>    returns `xnn_status_out_of_memory` and writes the needed buffer size into
>    `required_size`.
> 3. If the status is `xnn_status_out_of_memory`, resize `op_names_` to
>    `required_size`, then call again:
>    `xnn_get_runtime_profiling_info(runtime_, xnn_profile_info_operator_name,
>    op_names_.size(), op_names_.data(), &required_size)` to fill the buffer.
> 4. If the (final) status is not `xnn_status_success`, log an Error message
>    ("Failed to get XNNPACK operator names: %d" with the numeric status) and
>    return `Error::Internal`.
> 5. Otherwise return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-timings-fn]
> Error XNNProfiler::get_runtime_operator_timings()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-timings-fn]
> Queries XNNPACK for per-operator timing values and stores them in the member
> `op_timings_` (a `std::vector<uint64_t>`), one entry per operator, in operator
> order. Each value is an elapsed time in microseconds.
>
> Steps:
> 1. Declare a local `required_size` (uninitialized).
> 2. Resize `op_timings_` to `op_count_` (the count established by
>    `get_runtime_num_operators`, see
>    `[spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]`).
> 3. Call `xnn_get_runtime_profiling_info(runtime_,
>    xnn_profile_info_operator_timing, op_timings_.size() * sizeof(uint64_t),
>    op_timings_.data(), &required_size)`.
> 4. If the status is not `xnn_status_success`, log an Error message ("Failed to
>    get XNNPACK operator timing: %d" with the numeric status) and return
>    `Error::Internal`.
> 5. Otherwise return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
> Error XNNProfiler::initialize(xnn_runtime_t runtime)

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.initialize-fn]
> Initializes the profiler against a compiled XNNPACK runtime. This behavior
> applies only when the profiler is compiled in (`ET_EVENT_TRACER_ENABLED` or
> `ENABLE_XNNPACK_PROFILING`); the stub build simply ignores `runtime` and
> returns Ok.
>
> Steps:
> 1. Store `runtime` in the member `runtime_`.
> 2. Call `get_runtime_num_operators()` (see
>    `[spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-num-operators-fn]`);
>    if it returns a non-Ok Error, propagate that Error immediately.
> 3. Call `get_runtime_operator_names()` (see
>    `[spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.get-runtime-operator-names-fn]`);
>    if it returns a non-Ok Error, propagate that Error immediately.
> 4. Set `state_` to `Ready`.
> 5. Return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.log-operator-timings-fn]
> void XNNProfiler::log_operator_timings()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.log-operator-timings-fn]
> Updates the running-average timing state and, when the verbose profiling
> build is enabled, logs per-operator average timings. Two build variants:
>
> When `ENABLE_XNNPACK_PROFILING` is defined:
> 1. Increment `run_count_`.
> 2. Initialize a `name_len` cursor to 0, a `total_time` accumulator to 0.0f
>    (float), and null the `op_name` pointer.
> 3. If `op_timings_sum_.size()` is not equal to `op_count_`, replace
>    `op_timings_sum_` with a fresh `std::vector<uint64_t>` of length `op_count_`
>    filled with 0.
> 4. For each operator index `i` from 0 to `op_count_ - 1`:
>    a. Set `op_name` to `&op_names_[name_len]` (a C string within the packed
>       name buffer).
>    b. Advance `name_len` by `strlen(op_name) + 1` to skip past this name and
>       its NUL terminator to the start of the next name.
>    c. Add `op_timings_[i]` (this run's microsecond timing) into the running
>       sum `op_timings_sum_[i]`.
>    d. Compute `avg_op_time = op_timings_sum_[i] / static_cast<float>(run_count_)`.
>    e. Add `avg_op_time` into `total_time`.
>    f. Log at Info level: ">>, %s, %" PRId64 " (%f)" with `op_name`,
>       `op_timings_[i]`, and `avg_op_time`.
> 5. After the loop, log at Info level ">>, Total Time, %f" with `total_time`.
>
> When `ENABLE_XNNPACK_PROFILING` is not defined (event-tracer-only build):
> 1. Only increment `run_count_`; produce no log output.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
> Error XNNProfiler::start(EventTracer* event_tracer)

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.start-fn]
> Starts a new profiling session, capturing the start timestamp. This behavior
> applies only when the profiler is compiled in (`ET_EVENT_TRACER_ENABLED` or
> `ENABLE_XNNPACK_PROFILING`); the stub build ignores `event_tracer` and
> returns Ok.
>
> Steps:
> 1. Validate state:
>    - If `state_` is `Uninitialized`, log an Error message ("XNNProfiler must
>      be initialized prior to calling begin_execution.") and return
>      `Error::InvalidState`.
>    - Else if `state_` is `Running`, log an Error message ("XNNProfiler is
>      already running. Call end_execution() before calling begin_execution().")
>      and return `Error::InvalidState`.
> 2. Store `event_tracer` in the member `event_tracer_` (may be null).
> 3. Set `state_` to `Running`.
> 4. Set the member `start_time_` to `runtime::pal_current_ticks()` (the current
>    PAL tick timestamp).
> 5. Return `Error::Ok`.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.submit-trace-fn]
> void XNNProfiler::submit_trace()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.submit-trace-fn]
> Converts the collected per-operator XNNPACK timings into a sequence of
> delegate profiling events and submits them to the ET event tracer. Only called
> from `end()` when `event_tracer_` is non-null.
>
> Steps:
> 1. Fetch the tick-to-nanosecond conversion via
>    `runtime::pal_ticks_to_ns_multiplier()`, which returns a ratio with fields
>    `numerator` and `denominator` expressing nanoseconds per tick.
> 2. Assert `op_timings_.size() == op_count_` (ET_CHECK; a hard failure/abort if
>    violated).
> 3. Initialize a `name_len` cursor to 0, a running timestamp `time` set to the
>    member `start_time_`, and an empty `std::unordered_map<std::string,uint32_t>`
>    `op_counts` used to disambiguate repeated operator names.
> 4. For each operator index `i` from 0 to `op_count_ - 1`:
>    a. Set `op_name` to `&op_names_[name_len]` (C string in the packed name
>       buffer) and advance `name_len` by `strlen(op_name) + 1`.
>    b. Build `op_name_str = std::string(op_name)`. Pre-increment
>       `op_counts[op_name_str]` (so the first occurrence becomes 1), then form
>       the display name `name_formatted = op_name_str + " #" +
>       std::to_string(op_counts[op_name_str])`.
>    c. Convert the operator's XNNPACK timing (microseconds) to ET ticks:
>       `interval_ticks = static_cast<et_timestamp_t>(op_timings_[i] * 1000 *
>       multiplier.denominator / multiplier.numerator)`. (Multiply us by 1000 to
>       get ns, then divide by ns/tick = numerator/denominator.) Integer
>       arithmetic in `et_timestamp_t`.
>    d. Compute `end_time = time + interval_ticks`.
>    e. Call `executorch::runtime::event_tracer_log_profiling_delegate(
>       event_tracer_, name_formatted.c_str(), delegate_debug_id =
>       static_cast<DebugHandle>(-1), time, end_time)` to emit a delegate
>       profiling event spanning [time, end_time].
>    f. Set `time = end_time`, so the next operator is assumed to begin
>       immediately where the previous one ended.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
> XNNProfiler::XNNProfiler()

> [spec:et:sem:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler.xnn-profiler-fn]
> Default constructor.
>
> When the profiler is compiled in (`ET_EVENT_TRACER_ENABLED` or
> `ENABLE_XNNPACK_PROFILING`): initialize `state_` to
> `XNNProfilerState::Uninitialized` and `run_count_` to 0. Other members
> (`runtime_`, `event_tracer_`, `op_count_`, `op_names_`, `op_timings_`,
> `start_time_`, and `op_timings_sum_` when present) are left in their default
> state until `initialize`/`start`/`end` populate them.
>
> When profiling is not compiled in (stub build): an empty constructor that
> initializes no members.

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler]
> class XNNProfiler

> [spec:et:def:xnn-profiler.executorch.backends.xnnpack.delegate.profiling.xnn-profiler-state]
> enum class XNNProfilerState {
>   Uninitialized;
>   Ready;
>   Running;
> }

