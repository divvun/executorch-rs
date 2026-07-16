//! Literal port of runtime/platform/platform.cpp + runtime/platform/platform.h.
//!
//! Platform abstraction layer to allow individual platform libraries to
//! override symbols in ExecuTorch. PAL functions are defined as C functions so
//! a platform library implementer can use C in lieu of C++.
//!
//! The `et_pal_` methods should not be called directly. Use the corresponding
//! methods in the `executorch::runtime` namespace (this module's `pal_*`
//! functions) instead to appropriately dispatch through the PAL function table.

#![allow(non_camel_case_types)]

use core::ffi::c_char;

use crate::runtime::platform::types::et_timestamp_t;

/// Represents the conversion ratio from system ticks to nanoseconds.
/// To convert, use nanoseconds = ticks * numerator / denominator.
// [spec:et:def:platform.et-tick-ratio-t]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct et_tick_ratio_t {
    pub numerator: u64,
    pub denominator: u64,
}

// The weak PAL hooks (`et_pal_*`). These are declared here (mirroring
// platform.h) and defined by the platform-defaults group
// (`runtime/platform/default/*`, POSIX under `#[cfg(unix)]`). In C++ they are
// weak symbols selectable via `ET_INTERNAL_PLATFORM_WEAKNESS`; in Rust the
// concrete strong definitions are resolved at link time from the defaults
// module.
unsafe extern "C" {
    // [spec:et:def:platform.et-pal-init-fn]
    // [spec:et:sem:platform.et-pal-init-fn]
    pub fn et_pal_init();

    pub fn et_pal_abort() -> !;

    // [spec:et:def:platform.et-pal-current-ticks-fn]
    // [spec:et:sem:platform.et-pal-current-ticks-fn]
    pub fn et_pal_current_ticks() -> et_timestamp_t;

    // [spec:et:def:platform.et-pal-ticks-to-ns-multiplier-fn]
    // [spec:et:sem:platform.et-pal-ticks-to-ns-multiplier-fn]
    pub fn et_pal_ticks_to_ns_multiplier() -> et_tick_ratio_t;

    // [spec:et:def:platform.et-pal-emit-log-message-fn]
    // [spec:et:sem:platform.et-pal-emit-log-message-fn]
    pub fn et_pal_emit_log_message(
        timestamp: et_timestamp_t,
        level: et_pal_log_level_t,
        filename: *const c_char,
        function: *const c_char,
        line: usize,
        message: *const c_char,
        length: usize,
    );

    // [spec:et:def:platform.et-pal-allocate-fn]
    // [spec:et:sem:platform.et-pal-allocate-fn]
    pub fn et_pal_allocate(size: usize) -> *mut core::ffi::c_void;

    // [spec:et:def:platform.et-pal-free-fn]
    // [spec:et:sem:platform.et-pal-free-fn]
    pub fn et_pal_free(ptr: *mut core::ffi::c_void);
}

pub type pal_init_method = extern "C" fn();
pub type pal_abort_method = extern "C" fn();
// [spec:et:def:platform.et-pal-current-ticks-t-void]
pub type et_pal_current_ticks_t = extern "C" fn() -> et_timestamp_t;
pub type pal_current_ticks_method = extern "C" fn() -> et_timestamp_t;
pub type pal_ticks_to_ns_multiplier_method = extern "C" fn() -> et_tick_ratio_t;
pub type pal_emit_log_message_method = extern "C" fn(
    timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    filename: *const c_char,
    function: *const c_char,
    line: usize,
    message: *const c_char,
    length: usize,
);
pub type pal_allocate_method = extern "C" fn(size: usize) -> *mut core::ffi::c_void;
pub type pal_free_method = extern "C" fn(ptr: *mut core::ffi::c_void);

/// Severity level of a log message. Values must map to printable 7-bit ASCII
/// uppercase letters.
// [spec:et:def:platform.et-pal-log-level-t]
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum et_pal_log_level_t {
    kDebug = b'D' as i32,
    kInfo = b'I' as i32,
    kError = b'E' as i32,
    kFatal = b'F' as i32,
    // Exception to the "uppercase letter" rule.
    kUnknown = b'?' as i32,
}

// PORT-NOTE: In C++ the `et_pal_*` weak hooks have a plain C signature and are
// stored directly in the `PalImpl` table (whose `*_method` fields are C
// function-pointer types). Rust cannot coerce an `unsafe extern "C" {}` item to
// a safe `extern "C" fn` pointer, so trampolines wrap each hook. Behavior is
// identical (they forward verbatim); only the construct differs.
extern "C" fn et_pal_init_trampoline() {
    unsafe { et_pal_init() }
}
extern "C" fn et_pal_abort_trampoline() {
    unsafe { et_pal_abort() }
}
extern "C" fn et_pal_current_ticks_trampoline() -> et_timestamp_t {
    unsafe { et_pal_current_ticks() }
}
extern "C" fn et_pal_ticks_to_ns_multiplier_trampoline() -> et_tick_ratio_t {
    unsafe { et_pal_ticks_to_ns_multiplier() }
}
extern "C" fn et_pal_emit_log_message_trampoline(
    timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    filename: *const c_char,
    function: *const c_char,
    line: usize,
    message: *const c_char,
    length: usize,
) {
    unsafe { et_pal_emit_log_message(timestamp, level, filename, function, line, message, length) }
}
extern "C" fn et_pal_allocate_trampoline(size: usize) -> *mut core::ffi::c_void {
    unsafe { et_pal_allocate(size) }
}
extern "C" fn et_pal_free_trampoline(ptr: *mut core::ffi::c_void) {
    unsafe { et_pal_free(ptr) }
}

/// Table of pointers to platform abstraction layer functions.
// [spec:et:def:platform.executorch.runtime.pal-impl]
#[derive(Clone, Copy)]
pub struct PalImpl {
    pub init: Option<pal_init_method>,
    pub abort: Option<pal_abort_method>,
    pub current_ticks: Option<pal_current_ticks_method>,
    pub ticks_to_ns_multiplier: Option<pal_ticks_to_ns_multiplier_method>,
    pub emit_log_message: Option<pal_emit_log_message_method>,
    pub allocate: Option<pal_allocate_method>,
    pub free: Option<pal_free_method>,

    // An optional metadata field, indicating the name of the source
    // file that registered the PAL implementation.
    pub source_filename: *const c_char,
}

impl PalImpl {
    pub fn create_log_only(
        emit_log_message: Option<pal_emit_log_message_method>,
        source_filename: *const c_char,
    ) -> PalImpl {
        PalImpl::create(
            None, // init
            None, // abort
            None, // current_ticks
            None, // ticks_to_ns_multiplier
            emit_log_message,
            None, // allocate
            None, // free
            source_filename,
        )
    }

    // [spec:et:def:platform.executorch.runtime.pal-impl.create-fn]
    // [spec:et:sem:platform.executorch.runtime.pal-impl.create-fn]
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        init: Option<pal_init_method>,
        abort: Option<pal_abort_method>,
        current_ticks: Option<pal_current_ticks_method>,
        ticks_to_ns_multiplier: Option<pal_ticks_to_ns_multiplier_method>,
        emit_log_message: Option<pal_emit_log_message_method>,
        allocate: Option<pal_allocate_method>,
        free: Option<pal_free_method>,
        source_filename: *const c_char,
    ) -> PalImpl {
        PalImpl {
            init,
            abort,
            current_ticks,
            ticks_to_ns_multiplier,
            emit_log_message,
            allocate,
            free,
            source_filename,
        }
    }
}

/// The singleton instance of the PAL function table.
//
// PORT-NOTE: C++ statically initializes `pal_impl` with the `__FILE__` string
// literal (a `const char*`). Rust `static` initializers require the raw pointer
// to be constructed from a `&CStr`; `SOURCE_FILENAME` holds this file's path.
static SOURCE_FILENAME: &core::ffi::CStr = c"runtime/platform/platform.rs";

static mut PAL_IMPL: PalImpl = PalImpl {
    init: Some(et_pal_init_trampoline),
    abort: Some(et_pal_abort_trampoline),
    current_ticks: Some(et_pal_current_ticks_trampoline),
    ticks_to_ns_multiplier: Some(et_pal_ticks_to_ns_multiplier_trampoline),
    emit_log_message: Some(et_pal_emit_log_message_trampoline),
    allocate: Some(et_pal_allocate_trampoline),
    free: Some(et_pal_free_trampoline),
    source_filename: SOURCE_FILENAME.as_ptr(),
};

/// Tracks whether the PAL has been overridden. This is used to warn when
/// multiple callers override the PAL.
static mut IS_PAL_OVERRIDDEN: bool = false;

/// Override the PAL functions with user implementations. Any null entries in the
/// table are unchanged and will keep the default implementation.
// [spec:et:def:platform.executorch.runtime.register-pal-fn]
// [spec:et:sem:platform.executorch.runtime.register-pal-fn]
pub fn register_pal(impl_: PalImpl) -> bool {
    unsafe {
        if IS_PAL_OVERRIDDEN {
            let source = if !impl_.source_filename.is_null() {
                core::ffi::CStr::from_ptr(impl_.source_filename)
                    .to_str()
                    .unwrap_or("unknown")
            } else {
                "unknown"
            };
            crate::et_log!(
                Error,
                "register_pal() called multiple times. Subsequent calls will override the previous implementation. Previous implementation was registered from {}.",
                source
            );
        }
        IS_PAL_OVERRIDDEN = true;

        let pal_impl = &raw mut PAL_IMPL;

        if impl_.abort.is_some() {
            (*pal_impl).abort = impl_.abort;
        }

        if impl_.current_ticks.is_some() {
            (*pal_impl).current_ticks = impl_.current_ticks;
        }

        if impl_.ticks_to_ns_multiplier.is_some() {
            (*pal_impl).ticks_to_ns_multiplier = impl_.ticks_to_ns_multiplier;
        }

        if impl_.emit_log_message.is_some() {
            (*pal_impl).emit_log_message = impl_.emit_log_message;
        }

        if impl_.allocate.is_some() {
            (*pal_impl).allocate = impl_.allocate;
        }

        if impl_.free.is_some() {
            (*pal_impl).free = impl_.free;
        }

        if impl_.init.is_some() {
            (*pal_impl).init = impl_.init;
            if let Some(init) = (*pal_impl).init {
                init();
            }
        }

        true
    }
}

// [spec:et:def:platform.executorch.runtime.get-pal-impl-fn]
// [spec:et:sem:platform.executorch.runtime.get-pal-impl-fn]
pub fn get_pal_impl() -> *const PalImpl {
    &raw const PAL_IMPL
}

// [spec:et:def:platform.executorch.runtime.pal-init-fn]
// [spec:et:sem:platform.executorch.runtime.pal-init-fn]
pub fn pal_init() {
    unsafe { (PAL_IMPL.init.unwrap())() }
}

// [spec:et:def:platform.executorch.runtime.pal-abort-fn]
// [spec:et:sem:platform.executorch.runtime.pal-abort-fn]
pub fn pal_abort() -> ! {
    unsafe {
        (PAL_IMPL.abort.unwrap())();
    }
    // This should be unreachable, but in case the PAL implementation doesn't
    // abort, force it here.
    std::process::abort();
}

// [spec:et:def:platform.executorch.runtime.pal-current-ticks-fn]
// [spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn]
pub fn pal_current_ticks() -> et_timestamp_t {
    unsafe { (PAL_IMPL.current_ticks.unwrap())() }
}

// [spec:et:def:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]
// [spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn]
pub fn pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
    unsafe { (PAL_IMPL.ticks_to_ns_multiplier.unwrap())() }
}

// [spec:et:def:platform.executorch.runtime.pal-emit-log-message-fn]
// [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn]
#[allow(clippy::too_many_arguments)]
pub fn pal_emit_log_message(
    timestamp: et_timestamp_t,
    level: et_pal_log_level_t,
    filename: *const c_char,
    function: *const c_char,
    line: usize,
    message: *const c_char,
    length: usize,
) {
    unsafe {
        (PAL_IMPL.emit_log_message.unwrap())(
            timestamp, level, filename, function, line, message, length,
        )
    }
}

// [spec:et:def:platform.executorch.runtime.pal-allocate-fn]
// [spec:et:sem:platform.executorch.runtime.pal-allocate-fn]
pub fn pal_allocate(size: usize) -> *mut core::ffi::c_void {
    unsafe { (PAL_IMPL.allocate.unwrap())(size) }
}

// [spec:et:def:platform.executorch.runtime.pal-free-fn]
// [spec:et:sem:platform.executorch.runtime.pal-free-fn]
pub fn pal_free(ptr: *mut core::ffi::c_void) {
    unsafe { (PAL_IMPL.free.unwrap())(ptr) }
}

/// Shared test-only PAL spy harness.
///
/// PORT-NOTE: The C++ platform tests intercept the PAL by linking an alternate
/// set of weak `et_pal_*` symbols (`stub_platform.cpp`) together with a
/// `PlatformIntercept`/`InterceptWith` RAII installer. In Rust the `et_pal_*`
/// symbols are strong `#[unsafe(no_mangle)]` definitions in the `default/*`
/// modules and cannot be swapped out per test. The behavioral core those tests
/// exercise — dispatch through the singleton `PalImpl` table and its
/// `register_pal` override — IS reachable, so this harness installs a spy via
/// `register_pal` (the runtime-override API path) instead. `PalSpy` mirrors the
/// C++ spy fields; `ACTIVE_SPY` mirrors the runtime-override test's `active_spy`
/// pointer; the extern "C" trampolines forward to it.
#[cfg(test)]
pub(crate) mod test_spy {
    use super::*;

    /// Serializes every test that installs a PAL spy: the `PAL_IMPL` table and
    /// `IS_PAL_OVERRIDDEN` flag are process-global mutable statics, and Rust runs
    /// tests in parallel by default. Also mirrors gtest's serial execution of the
    /// override fixtures.
    pub static PAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Mirror of the C++ `PalSpy` in stub_platform.cpp / pal_spy.h.
    pub struct PalSpy {
        pub init_call_count: usize,
        pub current_ticks_call_count: usize,
        pub emit_log_message_call_count: usize,
        pub tick_ns_multiplier: et_tick_ratio_t,
        pub allocate_call_count: usize,
        pub free_call_count: usize,
        pub last_allocated_size: usize,
        pub last_allocated_ptr: *mut core::ffi::c_void,
        pub last_freed_ptr: *mut core::ffi::c_void,
        pub last_log_message_args: LogMessageArgs,
    }

    #[derive(Default)]
    pub struct LogMessageArgs {
        pub timestamp: et_timestamp_t,
        pub level: Option<et_pal_log_level_t>,
        pub filename: String,
        pub function: String,
        pub line: usize,
        pub message: String,
        pub length: usize,
    }

    impl PalSpy {
        pub const K_TIMESTAMP: et_timestamp_t = 1234;

        pub fn new() -> Self {
            PalSpy {
                init_call_count: 0,
                current_ticks_call_count: 0,
                emit_log_message_call_count: 0,
                tick_ns_multiplier: et_tick_ratio_t {
                    numerator: 1,
                    denominator: 1,
                },
                allocate_call_count: 0,
                free_call_count: 0,
                last_allocated_size: 0,
                last_allocated_ptr: core::ptr::null_mut(),
                last_freed_ptr: core::ptr::null_mut(),
                last_log_message_args: LogMessageArgs::default(),
            }
        }
    }

    // Mirror of the runtime-override test's file-local `active_spy`: the
    // trampolines below forward into whatever spy the currently-installed guard
    // registered. Guarded by PAL_TEST_LOCK so only one spy is ever active.
    static mut ACTIVE_SPY: *mut PalSpy = core::ptr::null_mut();

    unsafe fn spy() -> &'static mut PalSpy {
        unsafe { &mut *ACTIVE_SPY }
    }

    extern "C" fn pal_init() {
        unsafe { spy().init_call_count += 1 }
    }

    extern "C" fn pal_current_ticks() -> et_timestamp_t {
        unsafe {
            let s = spy();
            s.current_ticks_call_count += 1;
            PalSpy::K_TIMESTAMP
        }
    }

    extern "C" fn pal_ticks_to_ns_multiplier() -> et_tick_ratio_t {
        unsafe { spy().tick_ns_multiplier }
    }

    fn cstr_to_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe {
            core::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    extern "C" fn pal_emit_log_message(
        timestamp: et_timestamp_t,
        level: et_pal_log_level_t,
        filename: *const c_char,
        function: *const c_char,
        line: usize,
        message: *const c_char,
        length: usize,
    ) {
        unsafe {
            let s = spy();
            s.emit_log_message_call_count += 1;
            // The C++ spy copies message via std::string(message) which stops at
            // the first NUL; the emitter always passes a NUL-terminated buffer,
            // so `length` may exceed the copied string length (it is stored
            // separately). Mirror that: store the C-string content and the raw
            // length independently.
            s.last_log_message_args.timestamp = timestamp;
            s.last_log_message_args.level = Some(level);
            s.last_log_message_args.filename = cstr_to_string(filename);
            s.last_log_message_args.function = cstr_to_string(function);
            s.last_log_message_args.line = line;
            s.last_log_message_args.message = cstr_to_string(message);
            s.last_log_message_args.length = length;
        }
    }

    extern "C" fn pal_allocate(size: usize) -> *mut core::ffi::c_void {
        unsafe {
            let s = spy();
            s.allocate_call_count += 1;
            s.last_allocated_size = size;
            s.last_allocated_ptr = 0x1234usize as *mut core::ffi::c_void;
            core::ptr::null_mut()
        }
    }

    extern "C" fn pal_free(ptr: *mut core::ffi::c_void) {
        unsafe {
            let s = spy();
            s.free_call_count += 1;
            s.last_freed_ptr = ptr;
        }
    }

    /// RAII installer mirroring `InterceptWith` and the runtime-override
    /// fixture's SetUp/TearDown: captures the current `PalImpl`, points
    /// `ACTIVE_SPY` at the caller's spy, and installs the spy trampolines via
    /// `register_pal`; on drop it restores the captured table and clears
    /// `ACTIVE_SPY`. The caller must hold `PAL_TEST_LOCK`.
    pub struct SpyGuard {
        original: PalImpl,
    }

    impl SpyGuard {
        pub fn install(spy: &mut PalSpy) -> SpyGuard {
            let original = unsafe { *get_pal_impl() };
            unsafe {
                ACTIVE_SPY = spy as *mut PalSpy;
            }
            register_pal(PalImpl::create(
                Some(pal_init),
                None, // abort
                Some(pal_current_ticks),
                Some(pal_ticks_to_ns_multiplier),
                Some(pal_emit_log_message),
                Some(pal_allocate),
                Some(pal_free),
                c"stub_platform.rs".as_ptr(),
            ));
            SpyGuard { original }
        }
    }

    impl Drop for SpyGuard {
        fn drop(&mut self) {
            // Restore the original table. `register_pal` runs `init`, which for
            // the default posix table is the strong `et_pal_init` trampoline —
            // harmless and idempotent. Clear ACTIVE_SPY afterward so no dangling
            // pointer remains.
            register_pal(self.original);
            unsafe {
                ACTIVE_SPY = core::ptr::null_mut();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // [spec:et:sem:platform.et-pal-init-fn/test]
    // Also verifies the strong posix et_pal_init: this calls it directly and pins
    // its idempotence (second call is a no-op because INITIALIZED is already set).
    // [spec:et:sem:posix.et-pal-init-fn/test]
    // TEST(ExecutorPalTest, Initialization)
    #[test]
    fn executor_pal_test_initialization() {
        // Ensure `et_pal_init` can be called multiple times.
        //
        // PORT-NOTE: the C++ comment notes et_pal_init was already called once in
        // main(); the Rust default posix `et_pal_init` is idempotent (guards on an
        // INITIALIZED flag), so calling it here is the equivalent check.
        unsafe {
            et_pal_init();
            et_pal_init();
        }
    }

    // [spec:et:sem:posix.et-pal-current-ticks-fn/test]
    // Also exercises the weak PAL hook declaration through the default posix
    // table: et_pal_current_ticks() must return a monotonically non-decreasing
    // et_timestamp_t — the header hook contract.
    // [spec:et:sem:platform.et-pal-current-ticks-fn/test]
    // TEST(ExecutorPalTest, TimestampCoherency)
    #[test]
    fn executor_pal_test_timestamp_coherency() {
        unsafe {
            et_pal_init();

            let time_a: et_timestamp_t = et_pal_current_ticks();
            assert!(time_a >= 0);

            let time_b: et_timestamp_t = et_pal_current_ticks();
            assert!(time_b >= time_a);
        }
    }

    // [spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn/test]
    // Also exercises the weak PAL hook declaration through the default posix
    // table: et_pal_ticks_to_ns_multiplier() must return an et_tick_ratio_t with
    // positive numerator and denominator — the header hook contract.
    // [spec:et:sem:platform.et-pal-ticks-to-ns-multiplier-fn/test]
    // TEST(ExecutorPalTest, TickRateRatioSanity)
    #[test]
    fn executor_pal_test_tick_rate_ratio_sanity() {
        let tick_ns_ratio = unsafe { et_pal_ticks_to_ns_multiplier() };
        assert!(tick_ns_ratio.numerator > 0);
        assert!(tick_ns_ratio.denominator > 0);
    }

    // executor_pal_death_test.cpp — TEST(ExecutorPalTest, UninitializedPalDeath)
    //
    // PORT-NOTE: The C++ death test links stub_platform's `et_pal_current_ticks`
    // /`et_pal_emit_log_message`, which `__builtin_trap()` when no intercept is
    // installed, and uses a `main()` that skips PAL init. In Rust the strong
    // posix `et_pal_current_ticks` aborts (via `_assert_pal_initialized!`) only
    // when INITIALIZED is false AND `debug_assertions` are on; there is no
    // process-global "no intercept installed" state to trip, and `INITIALIZED`
    // is a shared mutable static that other tests in this binary set to true, so
    // the uninitialized-abort precondition cannot be reliably reproduced in an
    // in-process test. The abort-on-uninitialized behavior is a property of the
    // posix default PAL; testing it belongs to that module. Ported as an ignored
    // placeholder to preserve the 1:1 case mapping.
    #[test]
    #[ignore = "PORT-NOTE: uninitialized-PAL death depends on weak-symbol stub + separate process main() that don't map to Rust; see comment"]
    fn executor_pal_test_uninitialized_pal_death() {}

    // executor_pal_override_test.cpp — TEST(ExecutorPalOverrideTest, DiesIfNotIntercepted)
    //
    // PORT-NOTE: In C++ this links stub_platform's `et_pal_init`, which traps if
    // no intercept is installed, and asserts `runtime_init()` dies. In Rust
    // `runtime_init` calls the strong posix `et_pal_init` directly (per
    // runtime.md step 1, the table override is bypassed); that default never
    // traps, so there is nothing to die on. The "dies if the PAL init hook is a
    // trapping stub" behavior is inseparable from the weak-symbol link trick.
    #[test]
    #[ignore = "PORT-NOTE: runtime_init calls the strong posix et_pal_init directly; the trapping stub can't be linked per-test in Rust"]
    fn executor_pal_override_test_dies_if_not_intercepted() {}

    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // [spec:et:sem:runtime.executorch.runtime.runtime-init-fn/test]
    // TEST(ExecutorPalOverrideTest, InitIsRegistered)
    //
    // PORT-NOTE: The C++ test intercepts et_pal_init and checks runtime_init()
    // bumps init_call_count. In Rust runtime_init calls the strong posix
    // et_pal_init (not the table), so the spy's init hook is not invoked by
    // runtime_init. Instead this pins the reachable equivalent: register_pal
    // invokes the freshly-installed init exactly once (register-pal-fn step 4).
    #[test]
    fn executor_pal_override_test_init_is_registered() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        assert_eq!(spy.init_call_count, 0);
        {
            let _guard = test_spy::SpyGuard::install(&mut spy);
            // register_pal (inside install) runs the installed init once.
            assert_eq!(spy.init_call_count, 1);
        }
        // runtime_init bypasses the table (strong posix et_pal_init), so it does
        // not touch the spy — assert it stays at 1 after restore.
        assert_eq!(spy.init_call_count, 1);
    }

    // [spec:et:sem:log.executorch.runtime.internal.logf-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn/test]
    // Also verifies get_log_timestamp: the et_log! expansion calls it, and the
    // asserted timestamp == K_TIMESTAMP / current_ticks_call_count == 1 both hold
    // only because it forwards to pal_current_ticks.
    // [spec:et:sem:log.executorch.runtime.internal.get-log-timestamp-fn/test]
    // TEST(ExecutorPalOverrideTest, LogSmokeTest)  (guarded by ET_LOG_ENABLED)
    #[test]
    fn executor_pal_override_test_log_smoke_test() {
        if !crate::runtime::platform::log::ET_LOG_ENABLED {
            return;
        }
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        assert_eq!(spy.current_ticks_call_count, 0);
        assert_eq!(spy.emit_log_message_call_count, 0);

        // Use the highest log level, which isn't likely to be disabled.
        use crate::runtime::platform::log::{ET_MIN_LOG_LEVEL, LogLevel};
        assert!(LogLevel::Fatal >= ET_MIN_LOG_LEVEL);
        crate::et_log!(Fatal, "Test log");

        assert_eq!(spy.emit_log_message_call_count, 1);
        // Logging a message should also cause et_pal_current_ticks to be called
        // once.
        assert_eq!(spy.current_ticks_call_count, 1);

        let args = &spy.last_log_message_args;
        assert_eq!(args.timestamp, test_spy::PalSpy::K_TIMESTAMP);
        assert_eq!(args.level, Some(et_pal_log_level_t::kFatal));
        // Ignore filename/function/line to avoid fragility.
        assert_eq!(args.message, "Test log");
        assert_eq!(args.length, "Test log".len());
    }

    // [spec:et:sem:log.executorch.runtime.internal.vlogf-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn/test]
    // TEST(ExecutorPalOverrideTest, LogLevels)
    #[test]
    fn executor_pal_override_test_log_levels() {
        if !crate::runtime::platform::log::ET_LOG_ENABLED {
            return;
        }
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        use crate::runtime::platform::log::{ET_MIN_LOG_LEVEL, LogLevel};

        if LogLevel::Debug >= ET_MIN_LOG_LEVEL {
            crate::et_log!(Debug, "Test log");
            assert_eq!(
                spy.last_log_message_args.level,
                Some(et_pal_log_level_t::kDebug)
            );
        }

        if LogLevel::Info >= ET_MIN_LOG_LEVEL {
            crate::et_log!(Info, "Test log");
            assert_eq!(
                spy.last_log_message_args.level,
                Some(et_pal_log_level_t::kInfo)
            );
        }

        if LogLevel::Error >= ET_MIN_LOG_LEVEL {
            crate::et_log!(Error, "Test log");
            assert_eq!(
                spy.last_log_message_args.level,
                Some(et_pal_log_level_t::kError)
            );
        }

        if LogLevel::Fatal >= ET_MIN_LOG_LEVEL {
            crate::et_log!(Fatal, "Test log");
            assert_eq!(
                spy.last_log_message_args.level,
                Some(et_pal_log_level_t::kFatal)
            );
        }

        // An invalid LogLevel should map to kUnknown.
        crate::et_log!(NumLevels, "Test log");
        assert_eq!(
            spy.last_log_message_args.level,
            Some(et_pal_log_level_t::kUnknown)
        );
    }

    // PORT-NOTE: The C++ suite has no standalone test for the `pal_init()`
    // dispatch wrapper (runtime_init calls the strong et_pal_init, and the
    // override tests observe init via register_pal's own invocation). This pins
    // the wrapper's sole contract directly: it dispatches to the currently
    // installed PalImpl.init slot.
    // [spec:et:sem:platform.executorch.runtime.pal-init-fn/test]
    #[test]
    fn pal_init_dispatches_to_installed_slot() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        // install (register_pal) already ran the installed init once.
        assert_eq!(spy.init_call_count, 1);
        pal_init();
        assert_eq!(spy.init_call_count, 2);
    }

    // [spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn/test]
    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // TEST(ExecutorPalOverrideTest, TickToNsMultiplier)
    #[test]
    fn executor_pal_override_test_tick_to_ns_multiplier() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        // Validate that tick to ns multipliers are overridden.
        spy.tick_ns_multiplier = et_tick_ratio_t {
            numerator: 2,
            denominator: 3,
        };
        assert_eq!(pal_ticks_to_ns_multiplier().numerator, 2);
        assert_eq!(pal_ticks_to_ns_multiplier().denominator, 3);

        spy.tick_ns_multiplier = et_tick_ratio_t {
            numerator: 3,
            denominator: 1,
        };
        assert_eq!(pal_ticks_to_ns_multiplier().numerator, 3);
        assert_eq!(pal_ticks_to_ns_multiplier().denominator, 1);
    }

    // [spec:et:sem:platform.executorch.runtime.pal-allocate-fn/test]
    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // TEST(ExecutorPalOverrideTest, AllocateSmokeTest)
    #[test]
    fn executor_pal_override_test_allocate_smoke_test() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        // Validate that et_pal_allocate is overridden.
        assert_eq!(spy.allocate_call_count, 0);
        assert!(spy.last_allocated_ptr.is_null());
        pal_allocate(4);
        assert_eq!(spy.allocate_call_count, 1);
        assert_eq!(spy.last_allocated_size, 4);
        assert_eq!(
            spy.last_allocated_ptr,
            0x1234usize as *mut core::ffi::c_void
        );
    }

    // [spec:et:sem:platform.executorch.runtime.pal-free-fn/test]
    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // TEST(ExecutorPalOverrideTest, FreeSmokeTest)
    #[test]
    fn executor_pal_override_test_free_smoke_test() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();
        let _guard = test_spy::SpyGuard::install(&mut spy);

        pal_allocate(4);
        assert_eq!(spy.last_allocated_size, 4);
        assert_eq!(
            spy.last_allocated_ptr,
            0x1234usize as *mut core::ffi::c_void
        );

        // Validate that et_pal_free is overridden.
        assert_eq!(spy.free_call_count, 0);
        assert!(spy.last_freed_ptr.is_null());
        pal_free(spy.last_allocated_ptr);
        assert_eq!(spy.free_call_count, 1);
        assert_eq!(spy.last_freed_ptr, 0x1234usize as *mut core::ffi::c_void);
    }

    // ---- executor_pal_runtime_override_test.cpp ----
    // TEST_F(RuntimePalOverrideTest, SmokeTest)
    //
    // The C++ fixture builds its own set of C-ABI trampolines forwarding to a
    // PalSpy via a file-local `active_spy`, registers them with register_pal, and
    // exercises the `pal_*` dispatch wrappers. `SpyGuard` above is exactly that
    // pattern; this test drives the same assertions through it. The fixture
    // SetUp/TearDown capture+restore of the original PalImpl is what SpyGuard's
    // Drop does.
    //
    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-allocate-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-free-fn/test]
    // Also verifies get_pal_impl (SpyGuard::install captures + restores the table
    // through it) and PalImpl::create (SpyGuard builds the spy table with it, and
    // each dispatch below reads back the exact slots it aggregated).
    // [spec:et:sem:platform.executorch.runtime.get-pal-impl-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-impl.create-fn/test]
    #[test]
    fn runtime_pal_override_test_smoke_test() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();

        assert_eq!(spy.init_call_count, 0);
        assert_eq!(spy.current_ticks_call_count, 0);
        assert_eq!(spy.allocate_call_count, 0);
        assert_eq!(spy.free_call_count, 0);

        let _guard = test_spy::SpyGuard::install(&mut spy);

        // Expect register to call init.
        assert_eq!(spy.init_call_count, 1);

        assert_eq!(pal_current_ticks(), 1234);
        assert_eq!(spy.current_ticks_call_count, 1);

        let ticks_to_ns_multiplier = pal_ticks_to_ns_multiplier();
        assert_eq!(ticks_to_ns_multiplier.numerator, 1);
        assert_eq!(ticks_to_ns_multiplier.denominator, 1);

        pal_emit_log_message(
            5,
            et_pal_log_level_t::kError,
            c"test.cpp".as_ptr(),
            c"test_function".as_ptr(),
            6,
            c"test message".as_ptr(),
            7,
        );
        assert_eq!(spy.emit_log_message_call_count, 1);
        assert_eq!(spy.last_log_message_args.timestamp, 5);
        assert_eq!(
            spy.last_log_message_args.level,
            Some(et_pal_log_level_t::kError)
        );
        assert_eq!(spy.last_log_message_args.filename, "test.cpp");
        assert_eq!(spy.last_log_message_args.function, "test_function");
        assert_eq!(spy.last_log_message_args.line, 6);
        assert_eq!(spy.last_log_message_args.message, "test message");
        assert_eq!(spy.last_log_message_args.length, 7);

        pal_allocate(16);
        assert_eq!(spy.allocate_call_count, 1);

        pal_free(core::ptr::null_mut());
        assert_eq!(spy.free_call_count, 1);
    }

    // ---- executor_pal_static_runtime_override_test.cpp ----
    // TEST(RuntimePalOverrideTest, SmokeTest)
    //
    // PORT-NOTE: The C++ variant statically registers the spy PAL at load time
    // (a namespace-scope `bool registration_result = register_pal(...)`), so
    // et_pal_init has run once by the time the test body executes (init_call_count
    // == 1) and the spy stays installed for the whole process. Rust has no
    // load-time static-initializer registration hook equivalent that would also
    // isolate to this test binary; reproducing the "statically registered before
    // main" state in-process is the same observable path as the runtime-override
    // smoke test above (register_pal installs the spy and runs init once). This
    // ports that identical behavioral core; the static-vs-runtime registration
    // timing is the only deviation.
    //
    // [spec:et:sem:platform.executorch.runtime.register-pal-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-current-ticks-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-ticks-to-ns-multiplier-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-emit-log-message-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-allocate-fn/test]
    // [spec:et:sem:platform.executorch.runtime.pal-free-fn/test]
    #[test]
    fn static_runtime_pal_override_test_smoke_test() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut spy = test_spy::PalSpy::new();

        assert_eq!(spy.current_ticks_call_count, 0);
        assert_eq!(spy.allocate_call_count, 0);
        assert_eq!(spy.free_call_count, 0);

        let _guard = test_spy::SpyGuard::install(&mut spy);

        // Expect registration to call init.
        assert_eq!(spy.init_call_count, 1);

        assert_eq!(pal_current_ticks(), 1234);
        assert_eq!(spy.current_ticks_call_count, 1);

        let ticks_to_ns_multiplier = pal_ticks_to_ns_multiplier();
        assert_eq!(ticks_to_ns_multiplier.numerator, 1);
        assert_eq!(ticks_to_ns_multiplier.denominator, 1);

        pal_emit_log_message(
            5,
            et_pal_log_level_t::kError,
            c"test.cpp".as_ptr(),
            c"test_function".as_ptr(),
            6,
            c"test message".as_ptr(),
            7,
        );
        assert_eq!(spy.emit_log_message_call_count, 1);
        assert_eq!(spy.last_log_message_args.timestamp, 5);
        assert_eq!(
            spy.last_log_message_args.level,
            Some(et_pal_log_level_t::kError)
        );
        assert_eq!(spy.last_log_message_args.filename, "test.cpp");
        assert_eq!(spy.last_log_message_args.function, "test_function");
        assert_eq!(spy.last_log_message_args.line, 6);
        assert_eq!(spy.last_log_message_args.message, "test message");
        assert_eq!(spy.last_log_message_args.length, 7);

        pal_allocate(16);
        assert_eq!(spy.allocate_call_count, 1);

        pal_free(core::ptr::null_mut());
        assert_eq!(spy.free_call_count, 1);
    }

    // Direct exercise of the weak PAL allocate/free hook declarations through the
    // default posix table (no spy installed): et_pal_allocate(size) must return a
    // usable, non-null block for a nonzero size that is releasable via
    // et_pal_free(), and et_pal_free(nullptr) must be a no-op. The hooks are the
    // reference default (malloc/free); this pins the header contract that the
    // spy-based override tests bypass.
    // [spec:et:sem:platform.et-pal-allocate-fn/test]
    // [spec:et:sem:platform.et-pal-free-fn/test]
    #[test]
    fn platform_default_allocate_free_hooks() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unsafe {
            let ptr = et_pal_allocate(16);
            assert!(!ptr.is_null());
            // Block must be writable through its full extent.
            core::ptr::write_bytes(ptr as *mut u8, 0xAB, 16);
            assert_eq!(*(ptr as *const u8).add(15), 0xAB);
            et_pal_free(ptr);
            // Freeing a null pointer is a no-op (must not abort/crash).
            et_pal_free(core::ptr::null_mut());
        }
    }

    // Direct exercise of the weak PAL emit-log-message hook declaration through
    // the default posix table (no spy installed): calling et_pal_emit_log_message
    // with a well-formed record must emit to stderr and return without side
    // effects on the caller. The spy-based log tests intercept before this default
    // runs, so this pins the header hook declaration binding to the default impl.
    // [spec:et:sem:platform.et-pal-emit-log-message-fn/test]
    #[test]
    fn platform_default_emit_log_message_hook() {
        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let message = c"hook smoke test";
        unsafe {
            et_pal_init();
            et_pal_emit_log_message(
                et_pal_current_ticks(),
                et_pal_log_level_t::kInfo,
                c"platform.rs".as_ptr(),
                c"platform_default_emit_log_message_hook".as_ptr(),
                42,
                message.as_ptr(),
                message.to_bytes().len(),
            );
        }
    }

    // pal_abort step 1: dispatch to the currently-installed PalImpl.abort slot;
    // step 2: if that slot returns (violating its noreturn contract), force
    // termination via std::process::abort(). Both outcomes are process-fatal, so
    // each variant runs in a forked child (the gtest EXPECT_DEATH equivalent).
    // The child performs only async-signal-safe work: it pokes PAL_IMPL.abort
    // directly rather than going through register_pal, whose et_log! path may
    // allocate (unsafe after fork in a threaded test process).
    // [spec:et:sem:platform.executorch.runtime.pal-abort-fn/test]
    #[cfg(unix)]
    #[test]
    fn pal_abort_dispatches_then_forces_abort() {
        extern "C" fn exiting_abort() {
            unsafe { libc::_exit(17) }
        }
        extern "C" fn returning_abort() {}

        // Returns (exited_normally, exit_code, term_signal) for the child.
        unsafe fn run_child(hook: Option<pal_abort_method>) -> (bool, i32, i32) {
            unsafe {
                let pid = libc::fork();
                assert!(pid >= 0);
                if pid == 0 {
                    if let Some(h) = hook {
                        (*(&raw mut PAL_IMPL)).abort = Some(h);
                    }
                    pal_abort();
                }
                let mut status: core::ffi::c_int = 0;
                assert_eq!(libc::waitpid(pid, &mut status, 0), pid);
                if libc::WIFEXITED(status) {
                    (true, libc::WEXITSTATUS(status), 0)
                } else {
                    assert!(libc::WIFSIGNALED(status));
                    (false, 0, libc::WTERMSIG(status))
                }
            }
        }

        let _lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Step 1: the installed abort slot is what runs (child exits 17 from the
        // hook, never reaching the safety net).
        let (exited, code, _) = unsafe { run_child(Some(exiting_abort)) };
        assert!(exited);
        assert_eq!(code, 17);

        // Step 2: an abort hook that returns still ends in a forced abort.
        let (exited, _, sig) = unsafe { run_child(Some(returning_abort)) };
        assert!(!exited);
        assert_eq!(sig, libc::SIGABRT);

        // Default table: abort trampoline -> et_pal_abort -> libc::abort.
        let (exited, _, sig) = unsafe { run_child(None) };
        assert!(!exited);
        assert_eq!(sig, libc::SIGABRT);
    }
}
