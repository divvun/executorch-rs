# kernels/portable/cpu/util/arange_util.cpp

> [spec:et:def:arange-util.torch.executor.native.arange-out-impl-fn]
> void arange_out_impl( KernelRuntimeContext& ctx, double start, double end, double step, Tensor& out)

> [spec:et:sem:arange-util.torch.executor.native.arange-out-impl-fn]
> Fills `out` with an arithmetic sequence. There are two overloads plus two
> convenience wrappers:
>
> Four-argument form `arange_out_impl(ctx, start, end, step, out)` (all
> doubles):
> 1. `ctx` is unused (cast to void).
> 2. Compute `numel = compute_arange_out_size(start, end, step)` per
>    `[spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn]`.
>    This may abort via ET_CHECK_MSG if the computed count is negative.
> 3. Dispatch on `out.scalar_type()` over the REALHBF16 set, which is the set of
>    real (non-complex) dtypes plus Half and BFloat16: Byte (uint8), Char
>    (int8), Short (int16), Int (int32), Long (int64), Float (float32), Double
>    (float64), Half (float16), BFloat16. (It excludes Bool and complex types.)
>    If `out.scalar_type()` is outside this set, ET_SWITCH aborts/sets an error
>    for the unhandled type.
> 4. For the selected element type CTYPE, take `out_data = out.mutable_data_ptr<CTYPE>()`
>    and for each `i` in `[0, numel)` (ascending) write
>    `out_data[i] = static_cast<CTYPE>(start + i * step)`. The arithmetic
>    `start + i * step` is computed in double precision, then narrowed to CTYPE
>    by C++ static_cast (truncation toward zero for integer CTYPEs; standard
>    round-to-nearest float narrowing for floating CTYPEs).
> 5. `out` is written in place; the caller is responsible for having resized
>    `out` to hold `numel` elements. No return value.
>
> Two-argument form `arange_out_impl(ctx, end, out)`: equivalent to the
> four-argument form with `start = 0.0`, `step = 1.0`, i.e. fills `out` with
> `0, 1, 2, ... numel-1` cast to CTYPE, where `numel = ceil(end)`.
>
> The header also defines two inline convenience overloads that omit `ctx`:
> `arange_out_impl(start, end, step, out)` and `arange_out_impl(end, out)`.
> Each constructs a fresh default `KernelRuntimeContext` and forwards to the
> corresponding context-taking overload.

> [spec:et:def:arange-util.torch.executor.native.compute-arange-out-size-fn]
> executorch::aten::SizesType

> [spec:et:sem:arange-util.torch.executor.native.compute-arange-out-size-fn]
> Computes the number of elements a torch.arange-style output tensor must hold
> for the range `[start, end)` with the given `step`.
>
> Primary overload `compute_arange_out_size(start, end, step)` (all doubles):
> 1. Compute `numel = static_cast<SizesType>(std::ceil((end - start) / step))`.
>    The division and ceil are done in double precision; the ceil result is then
>    narrowed to SizesType (a signed 32-bit `int` for portable ExecuTorch) by
>    truncation toward zero. Note: `step` is not validated to be non-zero here;
>    a zero step yields inf/nan from the division before the cast.
> 2. ET_CHECK_MSG asserts `numel >= 0`; if it is negative (e.g. `end < start`
>    with positive step, or `start < end` with negative step) the check fails
>    and aborts the program with a message reporting numel, start, end, step.
> 3. Return `numel`.
>
> Inline convenience overload `compute_arange_out_size(end)` returns
> `compute_arange_out_size(0.0, end, 1.0)`, i.e. `ceil(end)`.

