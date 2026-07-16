//! Proves the vendored XNNPACK static library is built and linked AND that the
//! XNNPACK delegate registers into the global backend registry the executor
//! consults during `Method` load.
//!
//! Run: `cargo run -p executorch --features xnnpack --example xnn_link_smoke`

#[cfg(feature = "xnnpack")]
fn main() {
    use executorch::backends::xnnpack::runtime::sys;
    use executorch::runtime::backend::interface::{get_backend_class, get_num_registered_backends};
    use executorch::runtime::core::error::Error;

    // 1. Linking: xnn_initialize resolves and succeeds at runtime.
    // SAFETY: xnn_initialize accepts a null allocator (use the default).
    let status = unsafe { sys::xnn_initialize(core::ptr::null()) };
    assert_eq!(
        status,
        sys::xnn_status::SUCCESS,
        "xnn_initialize -> {status:?}"
    );
    println!("xnn_initialize: SUCCESS — XNNPACK linked and initialized");

    // 2. Registration wiring: register the delegate, then confirm the executor
    //    would find it by the same key stored in .pte delegate calls.
    assert_eq!(executorch::backends::xnnpack::register(), Error::Ok);
    let key = b"XnnpackBackend\0".as_ptr() as *const core::ffi::c_char;
    assert!(
        !get_backend_class(key).is_null(),
        "XnnpackBackend not found in registry after register()"
    );
    // Idempotent: a second call is a no-op, still one backend registered.
    assert_eq!(executorch::backends::xnnpack::register(), Error::Ok);
    assert_eq!(get_num_registered_backends(), 1);
    println!("register(): XnnpackBackend present in backend registry (idempotent)");
}

#[cfg(not(feature = "xnnpack"))]
fn main() {
    eprintln!("build with --features xnnpack to run this smoke test");
}
