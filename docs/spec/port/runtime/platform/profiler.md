# runtime/platform/profiler.cpp, runtime/platform/profiler.h

> [spec:et:def:profiler.executorch.runtime.begin-profiling-fn]
> uint32_t begin_profiling(const char* name)

> [spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]
> Records the start of a profiling event named `name` in the current profiling
> block and returns a token (a `uint32_t` index) identifying it for a later
> `end_profiling` call.
>
> Module state (file-local statics): a byte buffer `prof_buf` of size
> `prof_buf_size * MAX_PROFILE_BLOCKS`; `prof_header` points into the current
> block's header; `prof_arr` points into the current block's `prof_event_t`
> array; `num_blocks`; and `profile_state_tls` (a `prof_state_t`). See
> `[spec:et:def:profiler.executorch.runtime.prof-header-t]` and
> `[spec:et:def:profiler.executorch.runtime.prof-event-t]` for layouts.
>
> Steps:
> 1. `ET_CHECK_MSG(prof_header->prof_entries < MAX_PROFILE_EVENTS, ...)`: assert
>    there is free space in the event array; on failure the runtime aborts (via
>    the platform abort path) with the message "Out of profiling buffer space.
>    Increase MAX_PROFILE_EVENTS and re-compile."
> 2. Let `curr_counter = prof_header->prof_entries` (the next free slot index).
> 3. Increment `prof_header->prof_entries`.
> 4. Initialize the event at `prof_arr[curr_counter]`:
>    - `end_time = 0`;
>    - `name_str = name` (stores the raw pointer via the struct's anonymous union
>      `{ const char* name_str; char name[PROF_NAME_MAX_LEN]; }` — the string is
>      NOT copied here; copying is deferred to `dump_profile_stats`, see
>      `[spec:et:sem:profiler.executorch.runtime.dump-profile-stats-fn]`);
>    - `chain_idx = profile_state_tls.chain_idx` and
>      `instruction_idx = profile_state_tls.instruction_idx`, read via
>      `get_profile_tls_state()` (see
>      `[spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn]`).
> 5. Last, set `start_time = et_pal_current_ticks()` (see
>    `[spec:et:sem:platform.et-pal-current-ticks-fn]`). It is deliberately set
>    last so the timestamp excludes this function's bookkeeping overhead.
> 6. Return `curr_counter`.
>
> `name` is stored by reference and must remain valid until stats are dumped.

> [spec:et:def:profiler.executorch.runtime.dump-profile-stats-fn]
> void dump_profile_stats(prof_result_t* prof_result)

> [spec:et:sem:profiler.executorch.runtime.dump-profile-stats-fn]
> Populates the caller-provided `prof_result` with pointers/sizes describing the
> whole profiling buffer, and (once) materializes each event's name string into
> the fixed-size name field of its `prof_event_t`. Takes a non-null
> `prof_result_t* prof_result`; returns nothing.
>
> Module state used: `prof_buf`, `num_blocks`, and the file-local flag
> `prof_stats_dumped` (initially `false`).
>
> Steps:
> 1. Fill the result struct:
>    - `prof_result->prof_data = (uint8_t*)prof_buf` (base of the buffer);
>    - `prof_result->num_bytes = num_blocks * prof_buf_size`;
>    - `prof_result->num_blocks = num_blocks`.
> 2. If `prof_stats_dumped` is `false`, materialize names (this is done only the
>    first time, to keep it out of the hot begin/end path). For each block index
>    `i` in `[0, num_blocks)`:
>    a. Compute the block's header pointer `prof_buf + prof_buf_size * i` and its
>       event array pointer `prof_buf + prof_buf_size * i + prof_events_offset`.
>    b. For each event index `j` in `[0, header->prof_entries)`:
>       - Let `str_ptr = event[j].name_str` and `str_len = strlen(str_ptr)` (the
>         pointer stored by `begin_profiling`).
>       - `memset(event[j].name, 0, PROF_NAME_MAX_LEN)` to zero the inline name
>         buffer (which aliases `name_str` in the union — this overwrites the
>         pointer).
>       - If `str_len > PROF_NAME_MAX_LEN`, copy exactly `PROF_NAME_MAX_LEN`
>         bytes; otherwise copy `str_len` bytes, via `memcpy` into `event[j].name`.
>         (Note the boundary uses `>` not `>=`, so a name of exactly
>         `PROF_NAME_MAX_LEN` bytes is copied whole with no room left for a NUL
>         terminator.)
> 3. Set `prof_stats_dumped = true`.
>
> Because the union overlays `name_str` and `name`, after this call the event's
> pointer field is destroyed; the name now lives inline. Calling again is a no-op
> for name materialization (guard flag) but still refreshes the result struct.
> `reset_profile_stats` clears the flag to allow a fresh dump (see
> `[spec:et:sem:profiler.executorch.runtime.reset-profile-stats-fn]`).

> [spec:et:def:profiler.executorch.runtime.end-profiling-fn]
> void end_profiling(uint32_t token_id)

> [spec:et:sem:profiler.executorch.runtime.end-profiling-fn]
> Records the end time of the profiling event identified by `token_id`. Takes the
> `uint32_t` token returned by `begin_profiling` (see
> `[spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]`), returns
> nothing.
>
> Steps:
> 1. `ET_CHECK_MSG(token_id < MAX_PROFILE_EVENTS, "Invalid token id.")`: assert
>    the token is a valid event index; on failure the runtime aborts with the
>    message "Invalid token id." (Note: this only bounds-checks against the array
>    capacity, not against the number of events actually begun.)
> 2. Set `prof_arr[token_id].end_time = et_pal_current_ticks()` (see
>    `[spec:et:sem:platform.et-pal-current-ticks-fn]`), stamping the event's end
>    timestamp in system ticks in the current block's event array.
>
> No return value; no other fields are modified.

> [spec:et:def:profiler.executorch.runtime.executorch-profiler]
> class ExecutorchProfiler {
>   uint32_t prof_tok;
> }

> [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope]
> class ExecutorchProfilerInstructionScope {
>   prof_state_t old_state_;
> }

> [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope.executorch-profiler-instruction-scope-fn]
> ExecutorchProfilerInstructionScope::ExecutorchProfilerInstructionScope( const prof_state_t& state) : old_state_(get_profile_tls_state())

> [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.executorch-profiler-instruction-scope-fn]
> Constructor of the RAII scope guard `ExecutorchProfilerInstructionScope`, which
> temporarily installs a `prof_state_t` (chain/instruction indices) into the
> profiler thread-local state for the lifetime of the object.
>
> Steps (constructor, given `const prof_state_t& state`):
> 1. Initialize the member `old_state_` from `get_profile_tls_state()` (see
>    `[spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn]`),
>    saving the current TLS state so it can be restored later.
> 2. In the constructor body, call `set_profile_tls_state(state)` (see
>    `[spec:et:sem:profiler.executorch.runtime.set-profile-tls-state-fn]`) to
>    install the caller's `state`, so subsequent `begin_profiling` calls tag
>    events with this state's `chain_idx`/`instruction_idx`.
>
> The paired destructor (`~ExecutorchProfilerInstructionScope`) restores the
> saved state by calling `set_profile_tls_state(old_state_)`. The class is
> non-copyable and non-movable. A Rust port would implement this as a guard whose
> `Drop` restores the previous state.

> [spec:et:def:profiler.executorch.runtime.executorch-profiler.executorch-profiler-fn]
> ExecutorchProfiler::ExecutorchProfiler(const char* name)

> [spec:et:sem:profiler.executorch.runtime.executorch-profiler.executorch-profiler-fn]
> Constructor of the RAII scope guard `ExecutorchProfiler`, which profiles the
> block it is scoped to: it begins an event on construction and ends it on
> destruction.
>
> Steps (constructor, given `const char* name`):
> 1. Call `begin_profiling(name)` (see
>    `[spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]`) and store the
>    returned `uint32_t` token in the member `prof_tok`.
>
> The paired destructor (`~ExecutorchProfiler`) calls `end_profiling(prof_tok)`
> (see `[spec:et:sem:profiler.executorch.runtime.end-profiling-fn]`), stamping the
> event's end time when the object leaves scope. A Rust port would implement this
> as a guard holding the token whose `Drop` calls `end_profiling`.

> [spec:et:def:profiler.executorch.runtime.mem-prof-event-t]
> typedef struct alignas(8)

> [spec:et:def:profiler.executorch.runtime.prof-allocator-t]
> typedef struct alignas(8)

> [spec:et:def:profiler.executorch.runtime.prof-event-t]
> typedef struct alignas(8)

> [spec:et:def:profiler.executorch.runtime.prof-header-t]
> typedef struct alignas(8)

> [spec:et:def:profiler.executorch.runtime.prof-result-t]
> typedef struct alignas(8)

> [spec:et:def:profiler.executorch.runtime.prof-state-t]
> typedef struct

> [spec:et:def:profiler.executorch.runtime.profiler-init-fn]
> void profiler_init(void)

> [spec:et:sem:profiler.executorch.runtime.profiler-init-fn]
> Initializes the profiler against the statically allocated profiling buffer.
> Takes no arguments, returns nothing.
>
> Behavior: calls `profiling_create_block("default")` (see
> `[spec:et:sem:profiler.executorch.runtime.profiling-create-block-fn]`), which
> allocates/initializes the first profiling block, names it "default", writes the
> header (version, capacities), resets the counters, and sets up the array base
> pointers. That is the entire body.

> [spec:et:def:profiler.executorch.runtime.profiling-create-block-fn]
> void profiling_create_block(const char* name)

> [spec:et:sem:profiler.executorch.runtime.profiling-create-block-fn]
> Starts a new named profiling block (or reuses the current unused one) and
> repoints all module base pointers at it. Takes `const char* name`, returns
> nothing.
>
> Module state: `prof_buf`, `prof_header`, `prof_arr`, `mem_allocator_arr`,
> `mem_prof_arr`, `num_blocks` (initially 0).
>
> Steps:
> 1. Decide whether to advance to a new block. If the current block already has
>    any recorded entries — `prof_header->prof_entries != 0` OR
>    `mem_prof_entries != 0` OR `allocator_entries != 0` — OR `num_blocks == 0`
>    (no block yet exists), then:
>    - Increment `num_blocks` by 1.
>    - `ET_CHECK_MSG(num_blocks <= MAX_PROFILE_BLOCKS, ...)`: assert capacity; on
>      failure the runtime aborts with a message noting only `MAX_PROFILE_BLOCKS`
>      blocks are supported. Otherwise (current block is empty and at least one
>      block exists) reuse it without incrementing.
> 2. Compute the copy length: `str_len = min(strlen(name), PROF_NAME_MAX_LEN)`
>    (via `strlen(name) >= PROF_NAME_MAX_LEN ? PROF_NAME_MAX_LEN : strlen(name)`).
> 3. Compute the block base `base = prof_buf + (num_blocks - 1) * prof_buf_size`
>    and point `prof_header = (prof_header_t*)(base + prof_header_offset)`.
> 4. `memset(prof_header->name, 0, PROF_NAME_MAX_LEN)` then
>    `memcpy(prof_header->name, name, str_len)` to store the (possibly truncated)
>    block name.
> 5. Write header metadata: `prof_ver = ET_PROF_VER` (0x00000001);
>    `max_prof_entries = MAX_PROFILE_EVENTS`;
>    `max_allocator_entries = MEM_PROFILE_MAX_ALLOCATORS`;
>    `max_mem_prof_entries = MAX_MEM_PROFILE_EVENTS`.
> 6. Call `reset_profile_stats()` (see
>    `[spec:et:sem:profiler.executorch.runtime.reset-profile-stats-fn]`) to zero
>    the three live entry counters and clear the dumped flag.
> 7. Repoint the array bases into this block:
>    `prof_arr = base + prof_events_offset`,
>    `mem_allocator_arr = base + prof_mem_alloc_info_offset`,
>    `mem_prof_arr = base + prof_mem_alloc_events_offset`. Offsets are the
>    constants defined alongside the buffer layout (header, then events, then
>    allocator info, then memory-profile events).
>
> No return value.

> [spec:et:def:profiler.executorch.runtime.reset-profile-stats-fn]
> void reset_profile_stats()

> [spec:et:sem:profiler.executorch.runtime.reset-profile-stats-fn]
> Clears the live entry counters of the current profiling block so it can be
> reused, and re-arms the stats-dump path. Takes no arguments, returns nothing.
>
> Steps:
> 1. Set the file-local `prof_stats_dumped = false` (so the next
>    `dump_profile_stats` will re-materialize event names, see
>    `[spec:et:sem:profiler.executorch.runtime.dump-profile-stats-fn]`).
> 2. On the current block's header (`prof_header`), set `prof_entries = 0`,
>    `allocator_entries = 0`, and `mem_prof_entries = 0`.
>
> It does NOT clear the max_* capacity fields, the block name, the version, or the
> underlying event/allocator array contents; only the three "count" fields and the
> dumped flag are affected.

> [spec:et:def:profiler.executorch.runtime.set-profile-tls-state-fn]
> void set_profile_tls_state(const prof_state_t& state)

> [spec:et:sem:profiler.executorch.runtime.set-profile-tls-state-fn]
> Stores a `prof_state_t` (`{ int32_t chain_idx; uint32_t instruction_idx; }`)
> as the current profiler state. Takes `const prof_state_t& state`, returns
> nothing.
>
> Behavior: copy-assigns `state` into the module-level variable
> `profile_state_tls` (`profile_state_tls = state`). This state is read by
> `begin_profiling` to tag new events (see
> `[spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]`) and by
> `get_profile_tls_state` (see
> `[spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn]`).
>
> Note: despite the "tls"/thread-local naming, in this source `profile_state_tls`
> is a plain namespace-scope global (initialized to `{-1, 0u}`), not declared
> `thread_local`; the whole profiler module operates on shared global state.

> [spec:et:def:profiler.executorch.runtime.track-allocation-fn]
> void track_allocation(int32_t id, uint32_t size)

> [spec:et:sem:profiler.executorch.runtime.track-allocation-fn]
> Records one memory-allocation event (allocator id + byte size) into the current
> block's memory-profile array. Takes `int32_t id` and `uint32_t size`, returns
> nothing.
>
> Steps:
> 1. If `id == -1` (the null allocator id), return immediately without recording
>    anything.
> 2. `ET_CHECK_MSG(prof_header->mem_prof_entries < MAX_MEM_PROFILE_EVENTS, ...)`:
>    assert space in the memory-event array; on failure the runtime aborts with a
>    message to increase `MAX_MEM_PROFILE_EVENTS` (the message interpolates the
>    current `mem_prof_entries` count).
> 3. At index `prof_header->mem_prof_entries` of `mem_prof_arr`, set
>    `allocator_id = id` and `allocation_size = size`.
> 4. Increment `prof_header->mem_prof_entries`.
>
> No return value. See `[spec:et:def:profiler.executorch.runtime.mem-prof-event-t]`
> for the record layout and
> `[spec:et:sem:profiler.executorch.runtime.track-allocator-fn]` for obtaining an
> allocator id.

> [spec:et:def:profiler.executorch.runtime.track-allocator-fn]
> uint32_t track_allocator(const char* name)

> [spec:et:sem:profiler.executorch.runtime.track-allocator-fn]
> Registers a named memory allocator in the current block and returns its
> allocator id (a `uint32_t`). Takes `const char* name`.
>
> Steps:
> 1. `ET_CHECK_MSG(prof_header->allocator_entries < MEM_PROFILE_MAX_ALLOCATORS,
>    ...)`: assert space in the allocator-info array; on failure the runtime
>    aborts with a message to increase `MEM_PROFILE_MAX_ALLOCATORS`.
> 2. Let `str_len = strlen(name)` and `num_allocators =
>    prof_header->allocator_entries` (the slot to fill).
> 3. `memset(mem_allocator_arr[num_allocators].name, 0, PROF_NAME_MAX_LEN)`, then
>    copy the name: if `str_len > PROF_NAME_MAX_LEN`, `memcpy` exactly
>    `PROF_NAME_MAX_LEN` bytes; otherwise `memcpy` `str_len` bytes (again the
>    boundary is `>`, so a name of exactly `PROF_NAME_MAX_LEN` is copied whole
>    with no NUL room).
> 4. Set `mem_allocator_arr[num_allocators].allocator_id = num_allocators` (the id
>    equals the slot index).
> 5. Return `prof_header->allocator_entries++` — i.e. return the pre-increment
>    value (the id just assigned) and then increment the counter.
>
> The returned id is the value to pass as `track_allocation(id, size)`'s `id` (see
> `[spec:et:sem:profiler.executorch.runtime.track-allocation-fn]`). Layout:
> `[spec:et:def:profiler.executorch.runtime.prof-allocator-t]`.

> [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope.operator-fn]
> ExecutorchProfilerInstructionScope& operator=(

> [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.operator-fn]
> The copy-assignment operator of `ExecutorchProfilerInstructionScope`. It is
> explicitly deleted (`= delete`): the class is a non-copyable, non-movable scope
> guard, so copy-assignment (and likewise copy-construction, move-construction,
> and move-assignment) is prohibited at compile time. There is no runtime
> behavior — any attempt to copy-assign one of these objects is a compile error.
> A Rust port models this simply by not deriving/implementing `Clone`/`Copy` and
> not providing assignment for the guard type.

> [spec:et:def:profiler.executorch.runtime.get-profile-tls-state-fn]
> const prof_state_t& get_profile_tls_state()

> [spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn]
> Returns the current profiler state. Takes no arguments; returns a
> `const prof_state_t&` (a reference to the module-level `profile_state_tls`).
>
> Behavior: returns a reference to the module-global `profile_state_tls`
> (`{ int32_t chain_idx; uint32_t instruction_idx; }`, initialized to
> `{-1, 0u}`). No copy, no validation, no side effects. It is written by
> `set_profile_tls_state` (see
> `[spec:et:sem:profiler.executorch.runtime.set-profile-tls-state-fn]`) and read
> by `begin_profiling` (see
> `[spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]`). Despite the
> "tls" naming, the underlying storage is a plain namespace-scope global, not a
> `thread_local`.

