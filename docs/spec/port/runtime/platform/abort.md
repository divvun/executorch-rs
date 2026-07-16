# runtime/platform/abort.cpp

> [spec:et:def:abort.executorch.runtime.runtime-abort-fn]
> ET_NORETURN void runtime_abort()

> [spec:et:sem:abort.executorch.runtime.runtime-abort-fn]
> Thin wrapper that triggers immediate, uncleaned abnormal termination of the
> runtime. It takes no arguments and never returns (`ET_NORETURN`).
>
> Behavior: unconditionally calls the runtime's PAL abort dispatcher
> `pal_abort()` (declared in `runtime/platform/platform.h`), which routes to the
> currently registered `et_pal_abort` implementation in the PAL function table.
> There is no argument validation, no cleanup, no logging, and no return path;
> control does not come back to the caller. The abnormal exit status and the
> exact mechanism (e.g. `std::abort()`, `_Exit`, trap instruction) are defined by
> the selected platform backend — see the per-platform abort rules such as
> `[spec:et:sem:posix.et-pal-abort-fn]`, `[spec:et:sem:android.et-pal-abort-fn]`,
> `[spec:et:sem:minimal.et-pal-abort-fn]`, and `[spec:et:sem:zephyr.et-pal-abort-fn]`.

