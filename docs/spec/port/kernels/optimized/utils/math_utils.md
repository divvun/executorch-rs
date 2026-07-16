# kernels/optimized/utils/math_utils.h

> [spec:et:def:math-utils.executorch.utils.ceil-log2-fn]
> T CeilLog2(const T& x)

> [spec:et:sem:math-utils.executorch.utils.ceil-log2-fn]
> Compute ceil(log2(x)) for a generic numeric type T. If x <= 2, return 1.
> Otherwise compute the index of the last (most-significant) set bit of
> (x - 1) cast to uint64_t via findLastSet (which for a nonzero value equals
> floor(log2(x-1))), then add 1 and cast the result back to T. Subtracting 1
> before taking findLastSet makes exact powers of two return their true
> ceil(log2) rather than one too many.

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits]
> struct ComputeDTypeTraits

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-c10-b-float16]
> struct ComputeDTypeTraits<c10::BFloat16>

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-c10-half]
> struct ComputeDTypeTraits<c10::Half>

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-int16-t]
> struct ComputeDTypeTraits<int16_t>

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-int8-t]
> struct ComputeDTypeTraits<int8_t>

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-uint16-t]
> struct ComputeDTypeTraits<uint16_t>

> [spec:et:def:math-utils.executorch.utils.compute-d-type-traits-uint8-t]
> struct ComputeDTypeTraits<uint8_t>

> [spec:et:def:math-utils.executorch.utils.divup-fn]
> inline int64_t divup(int64_t x, int64_t y)

> [spec:et:sem:math-utils.executorch.utils.divup-fn]
> Integer ceiling division of x by y for int64_t operands: return
> (x + y - 1) / y using truncating integer division. Assumes y > 0 and
> x >= 0 (the callers pass positive counts); no overflow or divide-by-zero
> guarding is performed.
