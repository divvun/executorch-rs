//! Build script for the ExecuTorch Rust port.
//!
//! When the `xnnpack` feature is enabled, compiles the vendored XNNPACK
//! static library (and its cpuinfo / pthreadpool dependencies) and emits the
//! link directives so the `sys.rs` FFI declarations resolve. Mirrors the flags
//! in `backends/xnnpack/cmake/Dependencies.cmake`. Without the feature the
//! script is a no-op, keeping default `cargo build`/`cargo test` fast.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(et_use_libdl)");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_XNNPACK").is_none() {
        return;
    }
    build_and_link_xnnpack();
}

fn build_and_link_xnnpack() {
    // rust/executorch/ -> repo root is three levels up.
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .to_path_buf();
    let tp = repo.join("backends/xnnpack/third-party");
    let xnnpack = tp.join("XNNPACK");
    assert!(
        xnnpack.join("CMakeLists.txt").exists(),
        "XNNPACK submodule not checked out at {}; run `git submodule update --init --recursive backends/xnnpack/third-party/XNNPACK`",
        xnnpack.display()
    );

    for dir in ["XNNPACK", "cpuinfo", "pthreadpool", "FP16", "FXdiv"] {
        println!("cargo:rerun-if-changed={}", tp.join(dir).display());
    }

    let kleidi = std::env::var_os("CARGO_FEATURE_XNNPACK_KLEIDI").is_some();

    let mut cfg = cmake::Config::new(&xnnpack);
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("XNNPACK_LIBRARY_TYPE", "static")
        .define("XNNPACK_BUILD_BENCHMARKS", "OFF")
        .define("XNNPACK_BUILD_TESTS", "OFF")
        .define("XNNPACK_BUILD_ALL_MICROKERNELS", "OFF")
        .define("XNNPACK_ENABLE_AVXVNNI", "OFF")
        .define("XNNPACK_ENABLE_AVX512VNNIGFNI", "OFF")
        .define("XNNPACK_ENABLE_KLEIDIAI", if kleidi { "ON" } else { "OFF" })
        // Point at the vendored deps instead of letting XNNPACK download them.
        .define("CPUINFO_SOURCE_DIR", tp.join("cpuinfo"))
        .define("PTHREADPOOL_SOURCE_DIR", tp.join("pthreadpool"))
        .define("FP16_SOURCE_DIR", tp.join("FP16"))
        .define("FXDIV_SOURCE_DIR", tp.join("FXdiv"));

    // XNNPACK only installs libXNNPACK.a + libxnnpack-microkernels-prod.a, but
    // cpuinfo/pthreadpool archives land in the build tree. Build without the
    // install step and link straight out of the build directory.
    let dst = cfg.build_target("XNNPACK").build();
    let build = dst.join("build");

    // The two XNNPACK archives plus its object-library deps.
    for dir in ["", "cpuinfo", "pthreadpool"] {
        println!(
            "cargo:rustc-link-search=native={}",
            build.join(dir).display()
        );
    }
    println!("cargo:rustc-link-lib=static=XNNPACK");
    println!("cargo:rustc-link-lib=static=xnnpack-microkernels-prod");
    println!("cargo:rustc-link-lib=static=cpuinfo");
    println!("cargo:rustc-link-lib=static=pthreadpool");

    // C++ runtime + platform frameworks XNNPACK's objects pull in.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
        "linux" | "android" => println!("cargo:rustc-link-lib=stdc++"),
        _ => {}
    }
}
