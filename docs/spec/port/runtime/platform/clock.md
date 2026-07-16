# runtime/platform/clock.h

> [spec:et:def:clock.executorch.runtime.ticks-to-ns-fn]
> inline uint64_t ticks_to_ns(et_timestamp_t ticks)

> [spec:et:sem:clock.executorch.runtime.ticks-to-ns-fn]
> Converts an interval measured in system ticks into nanoseconds using the
> platform-defined conversion ratio. `ticks` has type `et_timestamp_t`, which is
> `uint64_t` (see `[spec:et:def:platform.et-tick-ratio-t]` for the ratio type).
>
> Steps:
> 1. Obtain the conversion ratio by calling `et_pal_ticks_to_ns_multiplier()`,
>    which returns an `et_tick_ratio_t { uint64_t numerator; uint64_t denominator; }`
>    (see the platform rule `[spec:et:sem:platform.et-pal-ticks-to-ns-multiplier-fn]`
>    and per-backend variants such as `[spec:et:sem:posix.et-pal-ticks-to-ns-multiplier-fn]`).
> 2. Compute the result as `(uint64_t)ticks * ratio.numerator / ratio.denominator`,
>    evaluated left-to-right in unsigned 64-bit arithmetic: first the product
>    `ticks * numerator` (wrapping modulo 2^64 on overflow, matching C++ unsigned
>    overflow semantics), then integer (truncating toward zero) division by
>    `denominator`.
> 3. Return the `uint64_t` result.
>
> There is no argument validation and no division-by-zero guard: `denominator` is
> assumed nonzero (all in-tree backends return a denominator of 1). No rounding is
> applied beyond the truncating integer division.

