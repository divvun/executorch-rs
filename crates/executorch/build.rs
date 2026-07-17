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
    // crates/executorch/ -> repo root is two levels up.
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root")
        .to_path_buf();
    let tp = repo.join("third-party");
    let xnnpack = tp.join("XNNPACK");
    assert!(
        xnnpack.join("CMakeLists.txt").exists(),
        "XNNPACK submodule not checked out at {}; run `git submodule update --init --recursive third-party/XNNPACK`",
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

    // Match XNNPACK's MSVC runtime library to the Rust binary's CRT. The build
    // script sees `+crt-static` (from RUSTFLAGS / .cargo/config.toml) via
    // CARGO_CFG_TARGET_FEATURE. Otherwise CMake's CMP0091 default links the
    // dynamic runtime (/MD), whose `__declspec(dllimport)` CRT references
    // (atan2f, _wassert, …) fail to resolve against a static-CRT Rust binary.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let crt_static = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .map(|feats| feats.split(',').any(|f| f == "crt-static"))
            .unwrap_or(false);
        cfg.define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW").define(
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            if crt_static {
                "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            } else {
                "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
            },
        );
    }

    // XNNPACK only installs libXNNPACK.a + libxnnpack-microkernels-prod.a, but
    // cpuinfo/pthreadpool archives land in the build tree. Build without the
    // install step and link straight out of the build directory.
    let dst = cfg.build_target("XNNPACK").build();
    let build = dst.join("build");

    // The two XNNPACK archives plus its object-library deps. Single-config
    // generators (Ninja/Make) drop the archives straight into each directory;
    // MSVC multi-config generators (Visual Studio) nest them in a per-config
    // subdir (Debug / Release / RelWithDebInfo), which the cmake crate selects
    // from the cargo profile. Emit both layouts — the linker ignores search
    // paths that don't exist, and only the built config's subdir is present.
    let is_msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    let configs: &[&str] = if is_msvc {
        &["", "RelWithDebInfo", "Release", "Debug", "MinSizeRel"]
    } else {
        &[""]
    };
    for dir in ["", "cpuinfo", "pthreadpool"] {
        let base = build.join(dir);
        for config in configs {
            println!(
                "cargo:rustc-link-search=native={}",
                base.join(config).display()
            );
        }
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
