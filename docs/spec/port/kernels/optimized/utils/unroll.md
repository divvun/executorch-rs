# kernels/optimized/utils/unroll.h

> [spec:et:def:unroll.executorch.utils.forced-unroll]
> struct ForcedUnroll

> [spec:et:def:unroll.executorch.utils.forced-unroll-1]
> struct ForcedUnroll<1>

> [spec:et:def:unroll.executorch.utils.forced-unroll-1.operator-fn]
> ET_INLINE void operator()(const Func& f) const

> [spec:et:sem:unroll.executorch.utils.forced-unroll-1.operator-fn]
> Base case of the compile-time unroll for n == 1: invoke the callable f
> exactly once with the index 0, i.e. call f(0).

> [spec:et:def:unroll.executorch.utils.forced-unroll.operator-fn]
> ET_INLINE void operator()(const Func& f) const

> [spec:et:sem:unroll.executorch.utils.forced-unroll.operator-fn]
> Recursive case of the compile-time unroll for n > 1: first invoke
> ForcedUnroll<n-1>{}(f) (which expands to f(0), f(1), ..., f(n-2) in order),
> then invoke f(n - 1). The net effect is calling f(0); f(1); ...; f(n-1) in
> ascending index order, fully unrolled at compile time.
