# executorch (Rust port)

A literal Rust re-implementation of the
[**ExecuTorch**](https://github.com/pytorch/executorch) on-device inference
runtime: it loads `.pte` programs exported by the (unchanged, Python)
ExecuTorch ahead-of-time pipeline and executes them on CPU, including the
XNNPACK delegate. The port covers the runtime core (platform layer, core
types, `EValue`, program/method executor), the portable, quantized and
optimized CPU kernels, the data-loader/module/tensor extensions, and the
XNNPACK backend with the vendored C library built from source.

This repository contains:

- `crates/executorch/` — the Rust port (single crate; the module tree mirrors
  the upstream C++ tree one-to-one).
- `crates/executorch-macros/` — the `#[et_kernel]` proc-macro that replaces
  the upstream `functions.yaml` registration codegen with link-time kernel
  registration.
- `docs/spec/port/` — the behavioral specification (per-symbol `def`/`sem`
  rules) that pins the port to the C++ behavior of
  [upstream ExecuTorch](https://github.com/pytorch/executorch), which served
  as the porting reference (pinned to `ec52125f1c` in `plan/main.styx`).
- `docs/port/rust-conventions.md` — the conventions the port was built under.
- `third-party/` — the vendored XNNPACK build dependencies (git submodules),
  used only with the `xnnpack` feature.

## Status

The port is complete and in production use (it powers
[divvun-speech](https://github.com/divvun) text-to-speech). It was built in
three waves:

1. **Spec** — a per-symbol behavioral spec (`def` + `sem` rules) extracted
   from the C++, under `docs/spec/port/`.
2. **Literal port** — every symbol translated 1:1 (bug-for-bug) into a
   matching Rust module, one module per C++ file.
3. **Tests** — the upstream C++ test suite ported absolutely: **3,040
   passing tests** pinning the spec, including the documented C++ quirks.

There is deliberately no idiomatization pass: the literal port is the
maintained form, so a reviewer can diff any Rust module against its C++
source file line by line. Deviations from the C++ (pure-Rust substitutions
for Eigen/pocketfft/sleef, the proc-macro registration seam, trait objects
for virtual interfaces) are marked `DEVIATION`/`PORT-NOTE` in place and
catalogued in `docs/port/rust-conventions.md`.

## Using

```toml
[dependencies]
executorch = { path = "crates/executorch", features = ["xnnpack"] }
```

```rust
use executorch::extension::module::module::{LoadMode, Module};

executorch::backends::xnnpack::register(); // once, with the xnnpack feature
let mut module = Module::from_file_path("model.pte", LoadMode::File, None, None, None, false)?;
let outputs = module.forward(inputs)?;
```

Kernel registration is explicit: call `kernels::optimized::register()` (which
also registers the portable set) and/or `backends::xnnpack::register()` at
startup, depending on how the `.pte` was exported.

## Building

```sh
cargo test                       # portable runtime + kernels, no C deps
git submodule update --init third-party/XNNPACK third-party/cpuinfo \
    third-party/pthreadpool third-party/FP16 third-party/FXdiv
cargo test --features xnnpack    # builds the vendored XNNPACK via cmake
```

Feature flags: `xnnpack` (delegate + vendored C build), `event-tracer`,
`profiling-enabled`, `bundled-program`, `aten`, `android`, `zephyr` (PAL
variants).

## License

BSD-3-Clause, matching upstream ExecuTorch (see `LICENSE`).
