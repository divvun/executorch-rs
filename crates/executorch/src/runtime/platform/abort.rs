//! Literal port of runtime/platform/abort.cpp.

use crate::runtime::platform::platform::pal_abort;

/// Trigger the ExecuTorch global runtime to immediately exit without cleaning
/// up, and set an abnormal exit status (platform-defined).
// [spec:et:def:abort.executorch.runtime.runtime-abort-fn]
// [spec:et:sem:abort.executorch.runtime.runtime-abort-fn]
pub fn runtime_abort() -> ! {
    pal_abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    // There is no dedicated C++ test file for abort.cpp; its contract (immediate
    // abnormal process exit via pal_abort, gtest-style EXPECT_DEATH) cannot be
    // observed with `#[should_panic]` because the default posix PAL abort calls
    // `libc::abort()` (no unwinding). Instead this is a re-exec death test: the
    // parent re-runs this test binary filtered to this one test with a marker
    // env var set; the child branch calls `runtime_abort()` and must terminate
    // abnormally (SIGABRT on unix hosts).
    // [spec:et:sem:abort.executorch.runtime.runtime-abort-fn/test]
    #[test]
    fn runtime_abort_terminates_process_abnormally() {
        const ENV_KEY: &str = "ET_RUNTIME_ABORT_DEATH_TEST_CHILD";
        if std::env::var_os(ENV_KEY).is_some() {
            runtime_abort();
        }

        let exe = std::env::current_exe().expect("current_exe");
        let output = std::process::Command::new(exe)
            .args([
                "runtime_abort_terminates_process_abnormally",
                "--test-threads=1",
            ])
            .env(ENV_KEY, "1")
            .output()
            .expect("failed to spawn death-test child");

        assert!(
            !output.status.success(),
            "runtime_abort() child exited successfully: {:?}",
            output.status
        );
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(output.status.signal(), Some(libc::SIGABRT));
        }
    }
}
