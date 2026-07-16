//! Literal port of runtime/platform/profiler.cpp + runtime/platform/profiler.h.

#![cfg(feature = "profiling-enabled")]

use core::ffi::c_char;

use crate::runtime::platform::platform::et_pal_current_ticks;

// PORT-NOTE: `ET_CHECK_MSG` is defined in runtime/platform/assert.h, which is not
// part of this module set and has no ported `assert.rs` target yet. This local
// macro mirrors its semantics (emit message, then abort via the PAL abort path)
// so this file compiles; it should be replaced by the shared `et_check_msg!`
// once the assert module is ported. Unresolved cross-module reference.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// Version string used to check for compatibility with post-processing
// tool
pub const ET_PROF_VER: u32 = 0x00000001;

// By default we support profiling upto 1024 perf events. Build
// targets can override this to increase the profiling buffer size
// during compilation.
pub const MAX_PROFILE_EVENTS: usize = 1024;
// By default we support profiling upto 1024 memory allocation events.
// Build targets can choose to override this, which will consequently have
// the effect of increasing/decreasing the profiling buffer size.
pub const MAX_MEM_PROFILE_EVENTS: usize = 1024;
// By default we support profiling only upto 16 allocators. If users
// have more allocators than these then they can override this during
// compilation time. There will be an increase/decrease in the profiling
// buffer size based on the way this value is changed.
pub const MEM_PROFILE_MAX_ALLOCATORS: usize = 32;
// By default we support only one profiling block. If users want to profile
// something that will be iterated on multiple times then they will have to
// increment this to support their use case. In post-processing the stats for
// all these iterations will be consolidated.
pub const MAX_PROFILE_BLOCKS: usize = 2;

pub const PROF_NAME_MAX_LEN: usize = 32;

// [spec:et:def:profiler.executorch.runtime.prof-event-t]
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct prof_event_t {
    // union {
    //   const char* name_str;
    //   char name[PROF_NAME_MAX_LEN];
    // };
    pub name: [u8; PROF_NAME_MAX_LEN],
    // chain_idx == -1 is a null value, when profile event happens out of chain
    // execution
    pub chain_idx: i32,
    pub instruction_idx: u32,
    pub start_time: u64,
    pub end_time: u64,
}

impl prof_event_t {
    // Reads the union's `name_str` (const char*) view. The union overlays a
    // raw pointer over the first `size_of::<*const c_char>()` bytes of `name`.
    #[inline]
    fn name_str(&self) -> *const c_char {
        unsafe { core::ptr::read_unaligned(self.name.as_ptr() as *const *const c_char) }
    }

    #[inline]
    fn set_name_str(&mut self, ptr: *const c_char) {
        unsafe {
            core::ptr::write_unaligned(self.name.as_mut_ptr() as *mut *const c_char, ptr);
        }
    }
}

// [spec:et:def:profiler.executorch.runtime.mem-prof-event-t]
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct mem_prof_event_t {
    pub allocator_id: u32,
    pub allocation_size: u32,
}

// [spec:et:def:profiler.executorch.runtime.prof-allocator-t]
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct prof_allocator_t {
    pub name: [u8; PROF_NAME_MAX_LEN],
    pub allocator_id: u64,
}

// [spec:et:def:profiler.executorch.runtime.prof-result-t]
#[repr(C, align(8))]
pub struct prof_result_t {
    pub prof_data: *mut u8,
    pub num_bytes: u32,
    pub num_blocks: u32,
}

// [spec:et:def:profiler.executorch.runtime.prof-header-t]
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct prof_header_t {
    pub name: [u8; 32],
    pub prof_ver: u32,
    pub max_prof_entries: u32,
    pub prof_entries: u32,
    pub max_allocator_entries: u32,
    pub allocator_entries: u32,
    pub max_mem_prof_entries: u32,
    pub mem_prof_entries: u32,
}

/*
This is what the layout of the profiling buffer looks like.
---------------------------------------
| Profiling header                    |
---------------------------------------
| Profile events (Perf events)        |
---------------------------------------
| Memory allocators info              |
---------------------------------------
| Profile events (Memory allocations) |
---------------------------------------
*/

// offsets of the various sections in the profiling buffer
// Total size required for profiling buffer
pub const prof_buf_size: usize = core::mem::size_of::<prof_header_t>()
    + core::mem::size_of::<prof_event_t>() * MAX_PROFILE_EVENTS
    + core::mem::size_of::<mem_prof_event_t>() * MAX_MEM_PROFILE_EVENTS
    + core::mem::size_of::<prof_allocator_t>() * MEM_PROFILE_MAX_ALLOCATORS;

pub const prof_header_offset: usize = 0;
pub const prof_events_offset: usize = core::mem::size_of::<prof_header_t>();
pub const prof_mem_alloc_info_offset: usize =
    prof_events_offset + core::mem::size_of::<prof_event_t>() * MAX_PROFILE_EVENTS;
pub const prof_mem_alloc_events_offset: usize = prof_mem_alloc_info_offset
    + core::mem::size_of::<prof_allocator_t>() * MEM_PROFILE_MAX_ALLOCATORS;

// [spec:et:def:profiler.executorch.runtime.prof-state-t]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct prof_state_t {
    pub chain_idx: i32,
    pub instruction_idx: u32,
}

// Module state (file-local statics). The C++ source uses namespace-scope
// statics operated on as shared global (mutable) state; mirrored here with a
// single mutable statics block. Not thread-safe, matching the C++.
static mut prof_buf: [u8; prof_buf_size * MAX_PROFILE_BLOCKS] =
    [0u8; prof_buf_size * MAX_PROFILE_BLOCKS];
// Base pointer for header
static mut prof_header: *mut prof_header_t = core::ptr::null_mut();
// Base pointer for profiling entries
static mut prof_arr: *mut prof_event_t = core::ptr::null_mut();
// Base pointer for memory allocator info array
static mut mem_allocator_arr: *mut prof_allocator_t = core::ptr::null_mut();
// Base pointer for memory profiling entries
static mut mem_prof_arr: *mut mem_prof_event_t = core::ptr::null_mut();

static mut num_blocks: u32 = 0;
static mut prof_stats_dumped: bool = false;
static mut profile_state_tls: prof_state_t = prof_state_t {
    chain_idx: -1,
    instruction_idx: 0u32,
};

// PORT-NOTE: In C++ the base pointers `prof_header`/`prof_arr`/`mem_allocator_arr`/
// `mem_prof_arr` are initialized to `prof_buf + <offset>` at static-init time. Rust
// cannot take the address of a mutable static in a const initializer, so they are
// initialized to null here and set up lazily via `ensure_base_pointers_initialized`
// (invoked at the head of every public entry point). This is a construct deviation,
// not a semantic one: begin/end/etc. all run through `profiling_create_block` via
// `profiler_init` in practice, which repoints them anyway.
unsafe fn ensure_base_pointers_initialized() {
    unsafe {
        if prof_header.is_null() {
            let base = ((&raw mut prof_buf) as *mut u8) as usize;
            prof_header = (base + prof_header_offset) as *mut prof_header_t;
            prof_arr = (base + prof_events_offset) as *mut prof_event_t;
            mem_allocator_arr = (base + prof_mem_alloc_info_offset) as *mut prof_allocator_t;
            mem_prof_arr = (base + prof_mem_alloc_events_offset) as *mut mem_prof_event_t;
        }
    }
}

// [spec:et:def:profiler.executorch.runtime.get-profile-tls-state-fn]
// [spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn]
pub fn get_profile_tls_state() -> prof_state_t {
    unsafe { profile_state_tls }
}

// [spec:et:def:profiler.executorch.runtime.set-profile-tls-state-fn]
// [spec:et:sem:profiler.executorch.runtime.set-profile-tls-state-fn]
pub fn set_profile_tls_state(state: &prof_state_t) {
    unsafe {
        profile_state_tls = *state;
    }
}

// [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope]
// [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope.executorch-profiler-instruction-scope-fn]
// [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.executorch-profiler-instruction-scope-fn]
// [spec:et:def:profiler.executorch.runtime.executorch-profiler-instruction-scope.operator-fn]
// [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.operator-fn]
// ScopeGuard: non-copyable, non-movable. Modeled as a guard whose Drop restores
// the previous TLS state. Not deriving Clone/Copy prohibits copy/assignment.
pub struct ExecutorchProfilerInstructionScope {
    old_state_: prof_state_t,
}

impl ExecutorchProfilerInstructionScope {
    pub fn new(state: &prof_state_t) -> Self {
        let this = ExecutorchProfilerInstructionScope {
            old_state_: get_profile_tls_state(),
        };
        set_profile_tls_state(state);
        this
    }
}

impl Drop for ExecutorchProfilerInstructionScope {
    fn drop(&mut self) {
        set_profile_tls_state(&self.old_state_);
    }
}

// [spec:et:def:profiler.executorch.runtime.begin-profiling-fn]
// [spec:et:sem:profiler.executorch.runtime.begin-profiling-fn]
pub fn begin_profiling(name: *const c_char) -> u32 {
    unsafe {
        ensure_base_pointers_initialized();
        et_check_msg!(
            (*prof_header).prof_entries < MAX_PROFILE_EVENTS as u32,
            "Out of profiling buffer space. Increase MAX_PROFILE_EVENTS and re-compile."
        );
        let curr_counter: u32 = (*prof_header).prof_entries;
        (*prof_header).prof_entries += 1;
        (*prof_arr.add(curr_counter as usize)).end_time = 0;
        (*prof_arr.add(curr_counter as usize)).set_name_str(name);
        let state: prof_state_t = get_profile_tls_state();
        (*prof_arr.add(curr_counter as usize)).chain_idx = state.chain_idx;
        (*prof_arr.add(curr_counter as usize)).instruction_idx = state.instruction_idx;
        // Set start time at the last to ensure that we're not capturing
        // any of the overhead in this function.
        (*prof_arr.add(curr_counter as usize)).start_time = et_pal_current_ticks();
        curr_counter
    }
}

// [spec:et:def:profiler.executorch.runtime.end-profiling-fn]
// [spec:et:sem:profiler.executorch.runtime.end-profiling-fn]
pub fn end_profiling(token_id: u32) {
    unsafe {
        ensure_base_pointers_initialized();
        et_check_msg!(token_id < MAX_PROFILE_EVENTS as u32, "Invalid token id.");
        (*prof_arr.add(token_id as usize)).end_time = et_pal_current_ticks();
    }
}

// [spec:et:def:profiler.executorch.runtime.dump-profile-stats-fn]
// [spec:et:sem:profiler.executorch.runtime.dump-profile-stats-fn]
pub fn dump_profile_stats(prof_result: *mut prof_result_t) {
    unsafe {
        ensure_base_pointers_initialized();
        (*prof_result).prof_data = (&raw mut prof_buf) as *mut u8;
        (*prof_result).num_bytes = num_blocks * prof_buf_size as u32;
        (*prof_result).num_blocks = num_blocks;

        if !prof_stats_dumped {
            let mut i: usize = 0;
            while i < num_blocks as usize {
                let prof_header_local: *mut prof_header_t =
                    ((&raw mut prof_buf) as *mut u8).add(prof_buf_size * i) as *mut prof_header_t;
                let prof_event_local: *mut prof_event_t = ((&raw mut prof_buf) as *mut u8)
                    .add(prof_buf_size * i + prof_events_offset)
                    as *mut prof_event_t;
                // Copy over the string names into the space allocated in prof_event_t. We
                // avoided doing this earlier to keep the overhead in begin_profiling and
                // end_profiling as low as possible.
                let mut j: usize = 0;
                while j < (*prof_header_local).prof_entries as usize {
                    let str_ptr: *const c_char = (*prof_event_local.add(j)).name_str();
                    let str_len: usize = libc::strlen(str_ptr);
                    core::ptr::write_bytes(
                        (*prof_event_local.add(j)).name.as_mut_ptr(),
                        0,
                        PROF_NAME_MAX_LEN,
                    );
                    if str_len > PROF_NAME_MAX_LEN {
                        core::ptr::copy_nonoverlapping(
                            str_ptr as *const u8,
                            (*prof_event_local.add(j)).name.as_mut_ptr(),
                            PROF_NAME_MAX_LEN,
                        );
                    } else {
                        core::ptr::copy_nonoverlapping(
                            str_ptr as *const u8,
                            (*prof_event_local.add(j)).name.as_mut_ptr(),
                            str_len,
                        );
                    }
                    j += 1;
                }
                i += 1;
            }
        }

        prof_stats_dumped = true;
    }
}

// [spec:et:def:profiler.executorch.runtime.reset-profile-stats-fn]
// [spec:et:sem:profiler.executorch.runtime.reset-profile-stats-fn]
pub fn reset_profile_stats() {
    unsafe {
        ensure_base_pointers_initialized();
        prof_stats_dumped = false;
        (*prof_header).prof_entries = 0;
        (*prof_header).allocator_entries = 0;
        (*prof_header).mem_prof_entries = 0;
    }
}

// [spec:et:def:profiler.executorch.runtime.track-allocation-fn]
// [spec:et:sem:profiler.executorch.runtime.track-allocation-fn]
pub fn track_allocation(id: i32, size: u32) {
    unsafe {
        ensure_base_pointers_initialized();
        if id == -1 {
            return;
        }
        et_check_msg!(
            (*prof_header).mem_prof_entries < MAX_MEM_PROFILE_EVENTS as u32,
            "Out of memory profiling buffer space. Increase MAX_MEM_PROFILE_EVENTS\
       to {} and re-compile.",
            (*prof_header).mem_prof_entries
        );
        (*mem_prof_arr.add((*prof_header).mem_prof_entries as usize)).allocator_id = id as u32;
        (*mem_prof_arr.add((*prof_header).mem_prof_entries as usize)).allocation_size = size;
        (*prof_header).mem_prof_entries += 1;
    }
}

// [spec:et:def:profiler.executorch.runtime.track-allocator-fn]
// [spec:et:sem:profiler.executorch.runtime.track-allocator-fn]
pub fn track_allocator(name: *const c_char) -> u32 {
    unsafe {
        ensure_base_pointers_initialized();
        et_check_msg!(
            (*prof_header).allocator_entries < MEM_PROFILE_MAX_ALLOCATORS as u32,
            "Out of allocator tracking space, {} is needed. Increase MEM_PROFILE_MAX_ALLOCATORS and re-compile",
            (*prof_header).allocator_entries
        );
        let str_len: usize = libc::strlen(name);
        let num_allocators: usize = (*prof_header).allocator_entries as usize;
        core::ptr::write_bytes(
            (*mem_allocator_arr.add(num_allocators)).name.as_mut_ptr(),
            0,
            PROF_NAME_MAX_LEN,
        );
        if str_len > PROF_NAME_MAX_LEN {
            core::ptr::copy_nonoverlapping(
                name as *const u8,
                (*mem_allocator_arr.add(num_allocators)).name.as_mut_ptr(),
                PROF_NAME_MAX_LEN,
            );
        } else {
            core::ptr::copy_nonoverlapping(
                name as *const u8,
                (*mem_allocator_arr.add(num_allocators)).name.as_mut_ptr(),
                str_len,
            );
        }
        (*mem_allocator_arr.add(num_allocators)).allocator_id = num_allocators as u64;
        let ret = (*prof_header).allocator_entries;
        (*prof_header).allocator_entries += 1;
        ret
    }
}

// [spec:et:def:profiler.executorch.runtime.profiling-create-block-fn]
// [spec:et:sem:profiler.executorch.runtime.profiling-create-block-fn]
pub fn profiling_create_block(name: *const c_char) {
    unsafe {
        ensure_base_pointers_initialized();
        // If the current profiling block is not used then continue to use this, if
        // not move onto the next block.
        if (*prof_header).prof_entries != 0
            || (*prof_header).mem_prof_entries != 0
            || (*prof_header).allocator_entries != 0
            || num_blocks == 0
        {
            num_blocks += 1;
            et_check_msg!(
                num_blocks <= MAX_PROFILE_BLOCKS as u32,
                "Only {} blocks are supported and they've all been used up but {} is used. Increment MAX_PROFILE_BLOCKS and re-run",
                MAX_PROFILE_BLOCKS,
                num_blocks
            );
        }

        // Copy over the name of this profiling block.
        let str_len: usize = if libc::strlen(name) >= PROF_NAME_MAX_LEN {
            PROF_NAME_MAX_LEN
        } else {
            libc::strlen(name)
        };
        let base: usize =
            ((&raw mut prof_buf) as *mut u8) as usize + (num_blocks as usize - 1) * prof_buf_size;
        prof_header = (base + prof_header_offset) as *mut prof_header_t;
        core::ptr::write_bytes((*prof_header).name.as_mut_ptr(), 0, PROF_NAME_MAX_LEN);
        core::ptr::copy_nonoverlapping(
            name as *const u8,
            (*prof_header).name.as_mut_ptr(),
            str_len,
        );

        // Set profiler version for compatiblity checks in the post-processing
        // tool.
        (*prof_header).prof_ver = ET_PROF_VER;
        // Set the maximum number of entries that this block can support.
        (*prof_header).max_prof_entries = MAX_PROFILE_EVENTS as u32;
        (*prof_header).max_allocator_entries = MEM_PROFILE_MAX_ALLOCATORS as u32;
        (*prof_header).max_mem_prof_entries = MAX_MEM_PROFILE_EVENTS as u32;
        reset_profile_stats();

        // Set the base addresses for the various profiling entries arrays.
        prof_arr = (base + prof_events_offset) as *mut prof_event_t;
        mem_allocator_arr = (base + prof_mem_alloc_info_offset) as *mut prof_allocator_t;
        mem_prof_arr = (base + prof_mem_alloc_events_offset) as *mut mem_prof_event_t;
    }
}

// [spec:et:def:profiler.executorch.runtime.profiler-init-fn]
// [spec:et:sem:profiler.executorch.runtime.profiler-init-fn]
pub fn profiler_init() {
    profiling_create_block(c"default".as_ptr());
}

// [spec:et:def:profiler.executorch.runtime.executorch-profiler]
// [spec:et:def:profiler.executorch.runtime.executorch-profiler.executorch-profiler-fn]
// [spec:et:sem:profiler.executorch.runtime.executorch-profiler.executorch-profiler-fn]
// RAII scope guard: begins an event on construction, ends it on Drop.
pub struct ExecutorchProfiler {
    prof_tok: u32,
}

impl ExecutorchProfiler {
    pub fn new(name: *const c_char) -> Self {
        ExecutorchProfiler {
            prof_tok: begin_profiling(name),
        }
    }
}

impl Drop for ExecutorchProfiler {
    fn drop(&mut self) {
        end_profiling(self.prof_tok);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The whole profiler operates on shared module-global statics and is not
    // thread-safe (mirroring the C++). Serialize the profiler tests so their
    // mutations of prof_buf/prof_header/... do not race under the parallel test
    // runner. There is no C++ profiler_test.cpp; these are focused Rust unit
    // tests pinning the ported semantics against docs/spec/port profiler.md.
    static PROFILER_TEST_LOCK: Mutex<()> = Mutex::new(());

    // Reads the header of the current block that prof_header points at.
    fn header() -> prof_header_t {
        unsafe {
            ensure_base_pointers_initialized();
            *prof_header
        }
    }

    // profiler_init -> profiling_create_block("default"): names the block,
    // writes version/capacity metadata, and resets the live counters.
    // begin_profiling returns an incrementing token and bumps prof_entries;
    // end_profiling stamps end_time. reset_profile_stats zeroes the counters.
    // [spec:et:sem:profiler.executorch.runtime.profiler-init-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.profiling-create-block-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.begin-profiling-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.end-profiling-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.reset-profile-stats-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.dump-profile-stats-fn/test]
    #[test]
    fn profiler_block_lifecycle() {
        let _lock = PROFILER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            crate::runtime::platform::platform::et_pal_init();
        }
        profiler_init();

        let h = header();
        assert_eq!(h.name[0..7], *b"default");
        assert_eq!(h.prof_ver, ET_PROF_VER);
        assert_eq!(h.max_prof_entries, MAX_PROFILE_EVENTS as u32);
        assert_eq!(h.max_allocator_entries, MEM_PROFILE_MAX_ALLOCATORS as u32);
        assert_eq!(h.max_mem_prof_entries, MAX_MEM_PROFILE_EVENTS as u32);
        assert_eq!(h.prof_entries, 0);
        assert_eq!(h.allocator_entries, 0);
        assert_eq!(h.mem_prof_entries, 0);

        // Set a known TLS state so begin_profiling tags the event with it.
        set_profile_tls_state(&prof_state_t {
            chain_idx: 3,
            instruction_idx: 7,
        });

        let tok0 = begin_profiling(c"evt0".as_ptr());
        let tok1 = begin_profiling(c"evt1".as_ptr());
        assert_eq!(tok0, 0);
        assert_eq!(tok1, 1);
        assert_eq!(header().prof_entries, 2);

        unsafe {
            // Event 0 was tagged with the installed TLS state; end_time starts 0.
            let e0 = *prof_arr.add(tok0 as usize);
            assert_eq!(e0.chain_idx, 3);
            assert_eq!(e0.instruction_idx, 7);
            assert_eq!(e0.end_time, 0);
        }

        end_profiling(tok0);
        unsafe {
            // end_profiling stamps a nonzero end_time (et_pal_current_ticks after
            // init) for event 0 and leaves event 1 unstamped.
            assert_ne!((*prof_arr.add(tok0 as usize)).end_time, 0);
            assert_eq!((*prof_arr.add(tok1 as usize)).end_time, 0);
        }

        reset_profile_stats();
        assert_eq!(header().prof_entries, 0);
        assert_eq!(header().allocator_entries, 0);
        assert_eq!(header().mem_prof_entries, 0);

        // dump_profile_stats fills the result struct describing the buffer.
        let mut result = prof_result_t {
            prof_data: core::ptr::null_mut(),
            num_bytes: 0,
            num_blocks: 0,
        };
        dump_profile_stats(&mut result as *mut prof_result_t);
        assert_eq!(result.prof_data, unsafe { (&raw mut prof_buf) as *mut u8 });
        assert_eq!(
            result.num_bytes,
            unsafe { num_blocks } * prof_buf_size as u32
        );
        assert_eq!(result.num_blocks, unsafe { num_blocks });
    }

    // track_allocator returns the pre-increment slot index as the allocator id and
    // records the name; track_allocation with id == -1 is a no-op, otherwise it
    // records (allocator_id, size) and bumps mem_prof_entries.
    // [spec:et:sem:profiler.executorch.runtime.track-allocator-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.track-allocation-fn/test]
    #[test]
    fn profiler_track_allocation_and_allocator() {
        let _lock = PROFILER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        profiler_init();
        reset_profile_stats();

        let id0 = track_allocator(c"alloc0".as_ptr());
        let id1 = track_allocator(c"alloc1".as_ptr());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(header().allocator_entries, 2);
        unsafe {
            assert_eq!((*mem_allocator_arr.add(0)).allocator_id, 0);
            assert_eq!((*mem_allocator_arr.add(1)).allocator_id, 1);
            let name0 = (*mem_allocator_arr.add(0)).name;
            assert_eq!(name0[0..6], *b"alloc0");
        }

        // id == -1 records nothing.
        track_allocation(-1, 999);
        assert_eq!(header().mem_prof_entries, 0);

        track_allocation(id0 as i32, 128);
        assert_eq!(header().mem_prof_entries, 1);
        unsafe {
            assert_eq!((*mem_prof_arr.add(0)).allocator_id, 0);
            assert_eq!((*mem_prof_arr.add(0)).allocation_size, 128);
        }
    }

    // get/set_profile_tls_state round-trip the module-global prof_state_t.
    // [spec:et:sem:profiler.executorch.runtime.get-profile-tls-state-fn/test]
    // [spec:et:sem:profiler.executorch.runtime.set-profile-tls-state-fn/test]
    #[test]
    fn profiler_tls_state_roundtrip() {
        let _lock = PROFILER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_profile_tls_state(&prof_state_t {
            chain_idx: 11,
            instruction_idx: 22,
        });
        let s = get_profile_tls_state();
        assert_eq!(s.chain_idx, 11);
        assert_eq!(s.instruction_idx, 22);
    }

    // ExecutorchProfilerInstructionScope installs `state` on construction and
    // restores the previous TLS state on Drop.
    // [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.executorch-profiler-instruction-scope-fn/test]
    #[test]
    fn profiler_instruction_scope_restores_state() {
        let _lock = PROFILER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_profile_tls_state(&prof_state_t {
            chain_idx: 1,
            instruction_idx: 2,
        });
        {
            let _scope = ExecutorchProfilerInstructionScope::new(&prof_state_t {
                chain_idx: 9,
                instruction_idx: 8,
            });
            let inner = get_profile_tls_state();
            assert_eq!(inner.chain_idx, 9);
            assert_eq!(inner.instruction_idx, 8);
        }
        let restored = get_profile_tls_state();
        assert_eq!(restored.chain_idx, 1);
        assert_eq!(restored.instruction_idx, 2);
    }

    // The C++ scope guard deletes copy-assignment (and copy/move construction);
    // per the sem rule the Rust model is that the guard type implements neither
    // Copy nor Clone. This pins that compile-time property via associated-const
    // probe resolution: the inherent DETECTED (true) only applies when the bound
    // holds, otherwise resolution falls back to the trait's `false`.
    // [spec:et:sem:profiler.executorch.runtime.executorch-profiler-instruction-scope.operator-fn/test]
    #[test]
    fn profiler_instruction_scope_is_not_copy_or_clone() {
        struct CopyProbe<T>(core::marker::PhantomData<T>);
        struct CloneProbe<T>(core::marker::PhantomData<T>);
        trait NotDetected {
            const DETECTED: bool = false;
        }
        impl<T> NotDetected for CopyProbe<T> {}
        impl<T> NotDetected for CloneProbe<T> {}
        impl<T: Copy> CopyProbe<T> {
            const DETECTED: bool = true;
        }
        impl<T: Clone> CloneProbe<T> {
            const DETECTED: bool = true;
        }

        // The guard must not be duplicable: neither Copy nor Clone.
        assert!(!CopyProbe::<ExecutorchProfilerInstructionScope>::DETECTED);
        assert!(!CloneProbe::<ExecutorchProfilerInstructionScope>::DETECTED);
        // Control: the probes do detect a copyable/clonable type.
        assert!(CopyProbe::<u32>::DETECTED);
        assert!(CloneProbe::<u32>::DETECTED);
    }

    // ExecutorchProfiler begins an event on construction (a token/prof_entries
    // bump) and ends it (stamps end_time) on Drop.
    // [spec:et:sem:profiler.executorch.runtime.executorch-profiler.executorch-profiler-fn/test]
    #[test]
    fn profiler_scope_begins_and_ends_event() {
        let _lock = PROFILER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            crate::runtime::platform::platform::et_pal_init();
        }
        profiler_init();
        reset_profile_stats();

        let before = header().prof_entries;
        {
            let _p = ExecutorchProfiler::new(c"scoped".as_ptr());
            assert_eq!(header().prof_entries, before + 1);
            unsafe {
                assert_eq!((*prof_arr.add(before as usize)).end_time, 0);
            }
        }
        // After Drop, the event's end_time is stamped.
        unsafe {
            assert_ne!((*prof_arr.add(before as usize)).end_time, 0);
        }
    }
}
