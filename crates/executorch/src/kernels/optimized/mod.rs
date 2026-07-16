pub mod blas;
pub mod cpu;
pub mod register;
pub mod utils;
pub mod vec;

use crate::runtime::core::error::Error;
use crate::runtime::core::span::Span;
use crate::runtime::kernel::operator_registry::{
    Kernel, register_kernel, registry_has_op_function,
};

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
fn register_if_absent(kernel: &Kernel) -> Error {
    if registry_has_op_function(kernel.name_, Span::new()) {
        return Error::Ok;
    }
    register_kernel(kernel)
}
