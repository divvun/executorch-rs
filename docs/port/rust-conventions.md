# Wave-2 translation conventions

Literal, bug-for-bug port of the ExecuTorch C++ runtime. One Rust module
per source file: `runtime/core/evalue.{h,cpp}` → `src/runtime/core/evalue.rs`.
The module tree is pre-generated; fill your assigned files only.

## Non-negotiables

- **Literal.** Same control flow, same decomposition, same names. A
  reviewer must see line-by-line correspondence with the C++. Idiom is
  Wave 4; do not `.iter().map().sum()` a manual loop.
- **Bug-for-bug.** Reproduce off-by-ones, overflow quirks, precision,
  silent early exits. Never fix. Flag suspected bugs with a
  `// PORT-NOTE:` comment and mention them in your final report.
- **Annotations.** Carry each symbol's markers verbatim above the Rust
  item, e.g.
  ```rust
  // [spec:et:def:evalue.executorch.runtime.e-value.to-tensor-fn]
  // [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-fn]
  pub fn to_tensor(...)
  ```
  The `sem` rule body in docs/spec/port/ is the porting instruction —
  read it alongside the C++.
- **No cross-module redesign.** If your module wants another module to
  change, add a `// PORT-NOTE:` and keep both literal.

## Type mappings

| C++ | Rust |
|---|---|
| `executorch::runtime::Error` (error.h) | `crate::runtime::core::error::Error` — `#[repr(u8)]` enum, same discriminants |
| `Result<T>` (result.h) | `crate::runtime::core::error::Result<T>` = `core::result::Result<T, Error>` (plus literal ports of `Result` helpers where call sites need them) |
| `Span<T>` / `ArrayRef<T>` | literal structs over `*const T`/len with slice accessors (`span.rs`, `array_ref.rs`) — not bare `&[T]`, so pointer identity semantics survive |
| `optional<T>` | `Option<T>` |
| `c10::Half` / `BFloat16` | `half::f16` / `half::bf16`, re-exported from `runtime/core/portable_type` |
| `char*` C strings in APIs | `&core::ffi::CStr` at PAL/extern boundaries, `&str` internally |
| virtual interfaces (`DataLoader`, `EventTracer`, `BackendInterface`, `NamedDataMap`) | traits with `&mut self` methods mirroring the vtable; `dyn` objects where C++ held base pointers |
| non-owning `Tensor`/`TensorImpl*` | `Tensor<'a>` holding `&'a TensorImpl` (or raw pointer + unsafe where mutation aliasing is inherent — mirror `tensor_impl.rs`'s established pattern) |
| `ET_NORETURN void et_pal_abort()` etc. | `pub extern "C" fn et_pal_*` in `runtime/platform/` (posix default under `#[cfg(unix)]`, minimal fallback otherwise) |

## Macro mappings (defined once, use everywhere)

- `ET_CHECK_OR_RETURN_ERROR(cond, Err, ...)` → `et_check_or_return_error!(cond, Err, "fmt", args)` (`runtime/platform/log.rs` owns logging macros; check macros live in `runtime/core/error.rs`)
- `ET_KERNEL_CHECK(ctx, cond, Err, retval)` → `et_kernel_check!(ctx, cond, Err, retval)` (`runtime/kernel/kernel_runtime_context.rs`)
- `ET_SWITCH_<SET>_TYPES(dtype, ctx, name, CTYPE, body)` → `et_switch_<set>_types!(dtype, |CTYPE| { ... })` — `macro_rules!` in `runtime/core/exec_aten/util/scalar_type_util.rs`; the generic-closure-over-dtype shape is established there, follow it.
- `ET_LOG(Level, ...)` → `et_log!(Level, ...)`

## Discipline

- `std` is allowed where the C++ used equivalents (extension/*); the
  core runtime avoids allocation exactly where the C++ does.
- `unsafe` is expected at PAL, flatbuffer, and tensor-aliasing
  boundaries — keep each block minimal and mirror what the C++ does.
- After your files: `cargo check -p executorch 2>&1` and fix every
  error *in your files*. Errors rooted in other modules still being
  stubs are acceptable — say so in your report.
- Generated flatbuffer types: `crate::schema::generated::executorch_flatbuffer`
  (from `schema/program.fbs`). Do not hand-write program schema types.

# Wave-3 test conventions

- A C++ test file `X_test.cpp` for module `M` ports into `#[cfg(test)] mod tests`
  at the bottom of `crates/executorch/src/<M>.rs`. Port the suite absolutely — no
  test dropped; obsolete/flaky ones get ported then `#[ignore]` with a PORT-NOTE.
- gtest mapping: `TEST(Suite, Name)` → `#[test] fn suite_name()`; `TEST_F`
  fixtures → plain fns calling a `setup()` helper; `EXPECT_EQ/NE/TRUE...` →
  `assert_eq!/assert_ne!/assert!`; `EXPECT_TENSOR_EQ/CLOSE` and `TensorFactory`
  come from `crate::runtime::core::exec_aten::testing_util`;
  `ET_EXPECT_KERNEL_FAILURE(ctx, expr)` → run expr, assert the context recorded
  a failure (mirroring the C++ macro's semantics); death tests on `ET_CHECK` →
  `#[should_panic]`.
- Facets: above each ported test add `// [spec:et:sem:<id>/test]` for every
  function rule the test genuinely exercises (the primary op/function first;
  util helpers only when the test meaningfully pins their behavior). Never
  annotate aspirationally.
- Failures are translation bugs: fix the WAVE-2 MODULE (referee = C++ source +
  sem rule), never weaken the ported assertion. Cross-module fixes get a
  PORT-NOTE and a report line instead.
- Fixture-dependent tests (executor/module .pte models): read the same env vars
  the C++ tests use (e.g. ET_MODULE_ADD_PATH); if unset, print a skip note and
  return early. PORT-NOTE the fixture dependency.
- xnnpack-gated tests compile under `--features xnnpack` but cannot link/run
  until the XNNPACK C library is wired into a build script — port + facet them,
  `#[cfg(feature = "xnnpack")]`, and PORT-NOTE the link gap.

# Optimized kernels (kernels/optimized) — dependency substitutions

The optimized CPU kernels exist to be FAST; their C++ leans on Eigen BLAS,
pocketfft, sleef, and a Vectorized<T> SIMD abstraction. The Rust port keeps the
op CONTROL FLOW literal (bug-for-bug) but at the leaf where C++ calls an external
math dependency, substitutes a pure-Rust crate. Each substitution is a deliberate
DEVIATION — mark it with a `// DEVIATION:` note; keep the `[spec:et:...]`
annotations and the surrounding algorithm identical.

Substitution table (do NOT literally port the C++ dependency internals):
- `executorch::cpublas::gemm(...)` (blas/CPUBlas.rs, blas/BlasKernel.rs) → the
  `gemm` crate. Port CPUBlas's `gemm`/`TransposeType`/`gemm_impl` SIGNATURE and
  the transpose/alpha/beta/column-major bookkeeping literally, but the inner
  triple-loop / Eigen call becomes a single `gemm::gemm(...)` invocation. This
  is the module linear/mm/bmm depend on — get its API right first.
- pocketfft (fft_utils.rs, op_fft_r2c.rs, op_fft_c2r.rs) → `realfft` (already a
  dep). Port the op's packing/normalization/one-sided-spectrum bookkeeping
  literally; the transform itself is a realfft plan.
- sleef vectorized transcendentals (exp/gelu/etc.) → scalar `libm`/`f32::exp`
  in a plain loop (Rust autovectorizes). DEVIATION note; behavior must match to
  the op's tolerance. `wide`/`pulp` only if a benchmark shows the scalar loop is
  the bottleneck — not preemptively.
- `Vectorized<T>` / vec/functional.rs → scalar loops (or std::simd-free plain
  code). The optimized op's *structure* (blocked/unrolled loops) is preserved;
  the SIMD lane type collapses to the scalar element type.
- `llvmMathExtras.h` helpers → Rust std where one exists (`leading_zeros`,
  `next_power_of_two`, `checked_*`, `isqrt`, etc.); thin literal ports otherwise.
  Most are one-liners. Annotate each; they're small but the gate counts them.
- cpuinfo runtime dispatch → `std::arch::is_aarch64_feature_detected!` /
  `std::is_x86_feature_detected!` (no FFI).

Registration: these ops override the portable set. They must be registered into
the `operator_registry` (runtime/kernel/operator_registry.rs) so `Method::execute`
dispatches to them — the Rust analogue of the C++ `functions.yaml` codegen +
`optimized_native_cpu_ops_lib`. Provide an explicit, idempotent
`kernels::optimized::register()` (mirrors the xnnpack `register()` seam) that
registers each optimized op under its ATen op name + overload, so a consumer
calls it once at startup. PORT-NOTE the codegen deviation.
