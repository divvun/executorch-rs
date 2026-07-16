# executorch-rs

Literal Rust port of the ExecuTorch runtime. The upstream C++ lives in
pytorch/executorch (porting reference pinned in `plan/main.styx`); this repo
carries only the Rust target.

## Layout

- `crates/executorch/src/` mirrors the upstream C++ tree one module per
  source file; `crates/executorch-macros/` is the `#[et_kernel]` registration
  proc-macro.
- `docs/spec/port/` is the nspec conformance corpus; every symbol carries
  `[spec:et:...]` markers in the Rust source.
- `docs/port/rust-conventions.md` documents the translation conventions and
  deviation policy. Read it before editing port code.

## Rules

- The port is literal and bug-for-bug by design; there is no idiomatization
  pass. Do not "clean up" ported code — its shape matches the C++ on purpose
  (quality gates exempt `crates/executorch/src/**` for this reason).
- Behavioral changes must update the matching `sem` rule in `docs/spec/port/`
  (bump the rule version) and keep the tests green.
- Keep `[spec:et:...]` annotation comments intact when moving or editing
  code; nspec coverage is computed from them.
- Commit via `nplan commit` (the pre-commit hook blocks bare `git commit`).

## Commands

```sh
cargo test                             # full suite
cargo fmt --check                      # format gate
cargo check -p executorch --all-features   # lint gate (builds XNNPACK once)
nplan check                            # witness all gates before committing
```
