//! Literal port of runtime/platform/runtime.cpp.

use crate::runtime::platform::platform::et_pal_init;

/// Initialize the ExecuTorch global runtime.
// [spec:et:def:runtime.executorch.runtime.runtime-init-fn]
// [spec:et:sem:runtime.executorch.runtime.runtime-init-fn]
pub fn runtime_init() {
    unsafe {
        et_pal_init();
    }
    // EXECUTORCH_PROFILE_CREATE_BLOCK("default");
    //
    // PORT-NOTE: `PROFILING_ENABLED` is not defined by default, so this macro
    // expands to a no-op that merely discards its `"default"` argument. The
    // profiling-enabled expansion (which would call the profiler module's
    // `profiling_create_block`) lives in that module's group and is not wired in
    // here.
    let _ = "default";
}
