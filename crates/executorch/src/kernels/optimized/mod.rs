pub mod blas;
pub mod cpu;
pub mod register;
pub mod utils;
pub mod vec;

use core::ffi::CStr;

use crate::runtime::core::error::Error;
use crate::runtime::kernel::operator_registry::{Kernel, get_registered_kernels, register_kernel};

/// Registers the merged optimized+portable CPU operator set into the global
/// operator registry so `Method::execute` dispatches to the fast optimized
/// kernels where they exist and falls back to the portable kernels for every
/// other op. This is the Rust analogue of the C++ `optimized_native_cpu_ops_lib`
/// target, which links `optimized_kernels` + `portable_kernels` under a
/// `merge_yaml`'d `functions.yaml` (optimized wins per-op).
///
/// PORT-NOTE (codegen deviation): the C++ has no runtime `register()` — the
/// operator table is emitted at build time by the `functions.yaml`/`merge_yaml`
/// codegen and pulled in via `+whole-archive` static-initializer linkage.
/// Rust has no load-time static initializer, so registration is surfaced as an
/// explicit entry point a consumer calls once at startup, mirroring the xnnpack
/// backend's `register()` seam.
///
/// The merge is realized as a skip-if-already-registered pass: the optimized
/// kernels are registered first (so they win the op name), then the portable
/// `#[et_kernel]` table (`kernels::registry::ET_KERNELS`, the port's stand-in
/// for `portable_kernels`) is registered, each op skipped if its name is already
/// present. That same skip makes the whole function idempotent — calling it more
/// than once is a no-op that returns `Ok`, and it composes safely with a prior
/// `kernels::registry::register_all`.
#[must_use]
pub fn register() -> Error {
    let opt = register::optimized_kernels();
    for kernel in &opt {
        let err = register_if_absent(kernel);
        if err != Error::Ok {
            return err;
        }
    }

    for reg in crate::kernels::registry::ET_KERNELS.iter() {
        let kernel = Kernel::new_fallback(reg.name.as_ptr(), reg.op);
        let err = register_if_absent(&kernel);
        if err != Error::Ok {
            return err;
        }
    }

    Error::Ok
}

/// Registers `kernel` only if no fallback kernel is already registered under its
/// op name. Skipping keeps `register()` idempotent and lets the optimized
/// override the portable set (optimized is registered first) without tripping
/// `register_kernels`' duplicate-name abort.
///
/// PORT-NOTE: This Rust-only idempotence probe scans the live registry directly.
/// `registry_has_op_function` delegates to the diagnostic lookup, which reports
/// a normal first-registration miss as an error.
// [spec:et:req:kernel-registration.quiet-idempotent-probe]
fn register_if_absent(kernel: &Kernel) -> Error {
    if registry_has_fallback_kernel(kernel.name_) {
        return Error::Ok;
    }
    register_kernel(kernel)
}

fn registry_has_fallback_kernel(name: *const core::ffi::c_char) -> bool {
    if name.is_null() {
        return false;
    }

    let kernels = get_registered_kernels();
    for index in 0..kernels.size() {
        let registered = unsafe { *kernels.index(index) };
        if registered.name_.is_null() || !registered.kernel_key_.is_fallback() {
            continue;
        }
        if unsafe { CStr::from_ptr(registered.name_) == CStr::from_ptr(name) } {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::core::evalue::EValue;
    use crate::runtime::core::span::Span;
    use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;
    use crate::runtime::kernel::operator_registry::{
        OPERATOR_REGISTRY_TEST_LOCK, OpFunction, clear_registry_for_test,
    };
    use crate::runtime::platform::platform::test_spy;

    fn optimized_kernel(_ctx: &mut KernelRuntimeContext, _stack: Span<*mut EValue>) {}
    fn portable_kernel(_ctx: &mut KernelRuntimeContext, _stack: Span<*mut EValue>) {}

    // [spec:et:req:kernel-registration.quiet-idempotent-probe/test]
    #[test]
    fn first_registration_is_quiet_and_keeps_optimized_precedence() {
        let _registry_lock = OPERATOR_REGISTRY_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _pal_lock = test_spy::PAL_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        clear_registry_for_test();

        let mut spy = test_spy::PalSpy::new();
        let _spy_guard = test_spy::SpyGuard::install(&mut spy);
        let optimized =
            Kernel::new_fallback(c"test::quiet_registration.out".as_ptr(), optimized_kernel);
        let portable =
            Kernel::new_fallback(c"test::quiet_registration.out".as_ptr(), portable_kernel);

        assert_eq!(register_if_absent(&optimized), Error::Ok);
        assert_eq!(spy.emit_log_message_call_count, 0);
        assert_eq!(register_if_absent(&portable), Error::Ok);
        assert_eq!(spy.emit_log_message_call_count, 0);

        let kernels = get_registered_kernels();
        assert_eq!(kernels.size(), 1);
        let registered = unsafe { *kernels.index(0) };
        assert!(std::ptr::fn_addr_eq(
            registered.op_.unwrap(),
            optimized_kernel as OpFunction,
        ));
    }
}
