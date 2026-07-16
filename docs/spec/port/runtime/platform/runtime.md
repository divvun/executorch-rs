# runtime/platform/runtime.cpp

> [spec:et:def:runtime.executorch.runtime.runtime-init-fn]
> void runtime_init()

> [spec:et:sem:runtime.executorch.runtime.runtime-init-fn]
> Initializes the ExecuTorch global runtime. Takes no arguments, returns nothing.
>
> Steps:
> 1. Call `et_pal_init()` directly (the weak C-ABI PAL hook, see
>    `[spec:et:sem:platform.et-pal-init-fn]`) to initialize global platform
>    state. Note this calls the `et_pal_*` symbol directly rather than the
>    `pal_init()` dispatch wrapper, so it invokes the link-time-resolved default
>    (or a strong client override) even if a table override was installed via
>    `register_pal`.
> 2. Invoke the `EXECUTORCH_PROFILE_CREATE_BLOCK("default")` macro (see
>    `[spec:et:sem:profiler.executorch.runtime.profiling-create-block-fn]`).
>    When `PROFILING_ENABLED` is defined this creates/initializes the "default"
>    profiling block; otherwise the macro expands to a no-op that merely discards
>    the `"default"` argument.
>
> No validation and no error propagation; the function is idempotent-safe to the
> extent the underlying `et_pal_init` is.

