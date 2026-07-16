# kernels/optimized/utils/llvmMathExtras.h

> [spec:et:def:llvm-math-extras.bit-scan-forward-fn]
> unsigned char _BitScanForward(unsigned long* _Index, unsigned long _Mask)

> [spec:et:sem:llvm-math-extras.bit-scan-forward-fn]
> MSVC intrinsic declaration only (no body in this header). Scans the 32-bit
> `_Mask` from the least-significant bit toward the most-significant, writes the
> bit index of the first set bit into `*_Index`, and returns nonzero if any bit
> was set (returns 0, leaving `*_Index` undefined, when `_Mask == 0`). Only
> referenced under `_MSC_VER`; on the Rust target it is unused and replaced by
> `u32::trailing_zeros`.

> [spec:et:def:llvm-math-extras.bit-scan-forward64-fn]
> unsigned char _BitScanForward64(unsigned long* _Index, unsigned __int64 _Mask)

> [spec:et:sem:llvm-math-extras.bit-scan-forward64-fn]
> MSVC intrinsic declaration only. Same as `_BitScanForward` but scans a 64-bit
> `_Mask`: writes the index of the lowest set bit into `*_Index` and returns
> nonzero iff `_Mask != 0`. Replaced on the Rust target by `u64::trailing_zeros`.

> [spec:et:def:llvm-math-extras.bit-scan-reverse-fn]
> unsigned char _BitScanReverse(unsigned long* _Index, unsigned long _Mask)

> [spec:et:sem:llvm-math-extras.bit-scan-reverse-fn]
> MSVC intrinsic declaration only. Scans the 32-bit `_Mask` from the
> most-significant bit toward the least, writes the index of the highest set bit
> into `*_Index`, and returns nonzero iff `_Mask != 0`. Replaced on the Rust
> target by `31 - u32::leading_zeros`.

> [spec:et:def:llvm-math-extras.bit-scan-reverse64-fn]
> unsigned char _BitScanReverse64(unsigned long* _Index, unsigned __int64 _Mask)

> [spec:et:sem:llvm-math-extras.bit-scan-reverse64-fn]
> MSVC intrinsic declaration only. Same as `_BitScanReverse` but over a 64-bit
> `_Mask`: writes the index of the highest set bit into `*_Index`, returns
> nonzero iff `_Mask != 0`. Replaced on the Rust target by `63 - u64::leading_zeros`.

> [spec:et:def:llvm-math-extras.executorch.llvm.absolute-difference-fn]
> typename std::enable_if<std::is_unsigned<T>::value, T>::type AbsoluteDifference( T X, T Y)

> [spec:et:sem:llvm-math-extras.executorch.llvm.absolute-difference-fn]
> For unsigned T, returns |X - Y| without underflow: compute max(X, Y) - min(X, Y).
> Equivalently the larger minus the smaller of the two operands.

> [spec:et:def:llvm-math-extras.executorch.llvm.align-addr-fn]
> inline uintptr_t alignAddr(const void* Addr, size_t Alignment)

> [spec:et:sem:llvm-math-extras.executorch.llvm.align-addr-fn]
> Rounds the pointer value `Addr` up to the next multiple of `Alignment` (which
> must be a power of two). Asserts Alignment is a nonzero power of two and that
> `Addr + Alignment - 1` does not wrap. Returns
> `(Addr + Alignment - 1) & ~(Alignment - 1)` as a `uintptr_t`.

> [spec:et:def:llvm-math-extras.executorch.llvm.align-down-fn]
> inline uint64_t alignDown(uint64_t Value, uint64_t Align, uint64_t Skew = 0)

> [spec:et:sem:llvm-math-extras.executorch.llvm.align-down-fn]
> Returns the largest u64 that is <= Value and congruent to Skew mod Align. Align
> must be nonzero (asserted). First reduces `Skew %= Align`, then returns
> `(Value - Skew) / Align * Align + Skew` using wrapping/truncating u64 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.align-to]
> struct AlignTo

> [spec:et:def:llvm-math-extras.executorch.llvm.align-to-fn]
> inline uint64_t alignTo(uint64_t Value, uint64_t Align, uint64_t Skew = 0)

> [spec:et:sem:llvm-math-extras.executorch.llvm.align-to-fn]
> Returns the smallest u64 (mod 2**64) that is >= Value and equal to
> `Align * N + Skew` for some integer N. Align must be nonzero (asserted). First
> reduces `Skew %= Align`, then returns
> `(Value + Align - 1 - Skew) / Align * Align + Skew` with wrapping u64 arithmetic
> (so `alignTo(~0LL, 8) == 0`).

> [spec:et:def:llvm-math-extras.executorch.llvm.align-to.from-value]
> struct from_value {
>   static const uint64_t value = (Value + Align - 1) / Align * Align;
> }

> [spec:et:def:llvm-math-extras.executorch.llvm.alignment-adjustment-fn]
> inline size_t alignmentAdjustment(const void* Ptr, size_t Alignment)

> [spec:et:sem:llvm-math-extras.executorch.llvm.alignment-adjustment-fn]
> Returns the number of bytes that must be added to `Ptr` to align it up to
> `Alignment`: `alignAddr(Ptr, Alignment) - (uintptr_t)Ptr`.

> [spec:et:def:llvm-math-extras.executorch.llvm.bits-to-double-fn]
> inline double BitsToDouble(uint64_t Bits)

> [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-double-fn]
> Reinterprets the 64-bit pattern `Bits` as an IEEE-754 double (bitwise memcpy).
> Rust: `f64::from_bits(Bits)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.bits-to-float-fn]
> inline float BitsToFloat(uint32_t Bits)

> [spec:et:sem:llvm-math-extras.executorch.llvm.bits-to-float-fn]
> Reinterprets the 32-bit pattern `Bits` as an IEEE-754 float (bitwise memcpy).
> Rust: `f32::from_bits(Bits)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.count-leading-ones-fn]
> std::size_t countLeadingOnes(T Value, ZeroBehavior ZB = ZB_Width)

> [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-ones-fn]
> For unsigned T, counts the number of consecutive 1 bits from the
> most-significant bit down to the first 0 bit. Implemented as
> `countLeadingZeros(~Value, ZB)`. ZB (ZB_Width/ZB_Undefined) governs the
> all-ones input.

> [spec:et:def:llvm-math-extras.executorch.llvm.count-leading-zeros-fn]
> std::size_t countLeadingZeros(T Val, ZeroBehavior ZB = ZB_Width)

> [spec:et:sem:llvm-math-extras.executorch.llvm.count-leading-zeros-fn]
> For unsigned T, counts the number of consecutive 0 bits from the
> most-significant bit down to the first 1 bit. Dispatches to the sized
> LeadingZerosCounter<T, sizeof(T)>::count(Val, ZB). For nonzero Val equals the
> hardware clz. For Val == 0 the result is numeric_limits<T>::digits (the bit
> width) when ZB is ZB_Width, and undefined when ZB is ZB_Undefined.

> [spec:et:def:llvm-math-extras.executorch.llvm.count-population-fn]
> inline unsigned countPopulation(T Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.count-population-fn]
> For unsigned T, returns the number of set (1) bits in `Value` (0 if Value == 0).
> Dispatches to PopulationCounter<T, sizeof(T)>::count. Rust: `Value.count_ones()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.count-trailing-ones-fn]
> std::size_t countTrailingOnes(T Value, ZeroBehavior ZB = ZB_Width)

> [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-ones-fn]
> For unsigned T, counts the number of consecutive 1 bits from the
> least-significant bit up to the first 0 bit. Implemented as
> `countTrailingZeros(~Value, ZB)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.count-trailing-zeros-fn]
> std::size_t countTrailingZeros(T Val, ZeroBehavior ZB = ZB_Width)

> [spec:et:sem:llvm-math-extras.executorch.llvm.count-trailing-zeros-fn]
> For unsigned T, counts the number of consecutive 0 bits from the
> least-significant bit up to the first 1 bit. Dispatches to
> TrailingZerosCounter<T, sizeof(T)>::count(Val, ZB). For nonzero Val equals the
> hardware ctz. For Val == 0 the result is numeric_limits<T>::digits (bit width)
> when ZB is ZB_Width, undefined when ZB is ZB_Undefined.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter]
> struct LeadingZerosCounter

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4]
> struct LeadingZerosCounter<T, 4>

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4.count-fn]
> static std::size_t count(T Val, ZeroBehavior ZB)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-4.count-fn]
> 32-bit specialization. If ZB != ZB_Undefined and Val == 0, returns 32.
> Otherwise returns __builtin_clz(Val), the count of leading zero bits of the
> 32-bit value (undefined for Val == 0 under ZB_Undefined). Rust: for Val != 0,
> `Val.leading_zeros()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8]
> struct LeadingZerosCounter<T, 8>

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8.count-fn]
> static std::size_t count(T Val, ZeroBehavior ZB)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter-t-8.count-fn]
> 64-bit specialization. If ZB != ZB_Undefined and Val == 0, returns 64.
> Otherwise returns __builtin_clzll(Val), the count of leading zero bits of the
> 64-bit value. Rust: for Val != 0, `Val.leading_zeros()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter.count-fn]
> static std::size_t count(T Val, ZeroBehavior)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.leading-zeros-counter.count-fn]
> Generic (portable, non-intrinsic) leading-zero count for any integer width via
> the bisection method. If Val == 0 returns numeric_limits<T>::digits (the bit
> width). Otherwise: ZeroBits = 0; for Shift = digits/2 down to 1 (halving each
> iteration), set Tmp = Val >> Shift; if Tmp is nonzero replace Val with Tmp,
> else add Shift into ZeroBits. Returns ZeroBits. ZeroBehavior is ignored (the
> Val == 0 path always returns the width).

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter]
> struct PopulationCounter

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter-t-8]
> struct PopulationCounter<T, 8>

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter-t-8.count-fn]
> static unsigned count(T Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter-t-8.count-fn]
> 64-bit population count. With GCC returns __builtin_popcountll(Value).
> Otherwise the SWAR fallback: v = Value; v -= (v>>1)&0x5555555555555555;
> v = (v&0x3333333333333333) + ((v>>2)&0x3333333333333333);
> v = (v + (v>>4)) & 0x0F0F0F0F0F0F0F0F; return (v * 0x0101010101010101) >> 56.
> Rust: `Value.count_ones()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.population-counter.count-fn]
> static unsigned count(T Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.population-counter.count-fn]
> Generic (<= 32-bit) population count. With GCC returns
> __builtin_popcount(Value). Otherwise the 32-bit SWAR fallback:
> v = Value; v = v - ((v>>1)&0x55555555);
> v = (v&0x33333333) + ((v>>2)&0x33333333);
> return ((v + (v>>4) & 0xF0F0F0F) * 0x1010101) >> 24. Note the source's
> parenthesization gives `(v + (v>>4) & 0xF0F0F0F)` = `(v + ((v>>4) & 0xF0F0F0F))`
> because `&` binds looser than `+`. Rust: `(Value as u32).count_ones()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter]
> struct TrailingZerosCounter

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4]
> struct TrailingZerosCounter<T, 4>

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4.count-fn]
> static std::size_t count(T Val, ZeroBehavior ZB)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-4.count-fn]
> 32-bit specialization. If ZB != ZB_Undefined and Val == 0, returns 32.
> Otherwise returns __builtin_ctz(Val), the count of trailing zero bits of the
> 32-bit value. Rust: for Val != 0, `Val.trailing_zeros()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8]
> struct TrailingZerosCounter<T, 8>

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8.count-fn]
> static std::size_t count(T Val, ZeroBehavior ZB)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter-t-8.count-fn]
> 64-bit specialization. If ZB != ZB_Undefined and Val == 0, returns 64.
> Otherwise returns __builtin_ctzll(Val), the count of trailing zero bits of the
> 64-bit value. Rust: for Val != 0, `Val.trailing_zeros()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter.count-fn]
> static std::size_t count(T Val, ZeroBehavior)

> [spec:et:sem:llvm-math-extras.executorch.llvm.detail.trailing-zeros-counter.count-fn]
> Generic (portable) trailing-zero count via bisection. If Val == 0 returns
> numeric_limits<T>::digits. If Val is odd (Val & 1) returns 0. Otherwise:
> ZeroBits = 0; Shift = digits/2; Mask = max >> Shift; while Shift != 0: if
> (Val & Mask) == 0 then Val >>= Shift and ZeroBits |= Shift; Shift >>= 1;
> Mask >>= Shift. Returns ZeroBits. ZeroBehavior is ignored.

> [spec:et:def:llvm-math-extras.executorch.llvm.divide-ceil-fn]
> inline uint64_t divideCeil(uint64_t Numerator, uint64_t Denominator)

> [spec:et:sem:llvm-math-extras.executorch.llvm.divide-ceil-fn]
> Returns ceil(Numerator / Denominator) computed as
> `alignTo(Numerator, Denominator) / Denominator`, i.e. rounds Numerator up to a
> multiple of Denominator then divides. Denominator must be nonzero.

> [spec:et:def:llvm-math-extras.executorch.llvm.double-to-bits-fn]
> inline uint64_t DoubleToBits(double Double)

> [spec:et:sem:llvm-math-extras.executorch.llvm.double-to-bits-fn]
> Reinterprets the double `Double` as its 64-bit pattern (bitwise memcpy). NaN
> bits may change on some hosts. Rust: `Double.to_bits()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.find-first-set-fn]
> T findFirstSet(T Val, ZeroBehavior ZB = ZB_Max)

> [spec:et:sem:llvm-math-extras.executorch.llvm.find-first-set-fn]
> Returns the index of the lowest set bit of Val. If ZB == ZB_Max and Val == 0,
> returns numeric_limits<T>::max(). Otherwise returns
> `countTrailingZeros(Val, ZB_Undefined)` cast to T.

> [spec:et:def:llvm-math-extras.executorch.llvm.find-last-set-fn]
> T findLastSet(T Val, ZeroBehavior ZB = ZB_Max)

> [spec:et:sem:llvm-math-extras.executorch.llvm.find-last-set-fn]
> Returns the index of the highest set bit of Val. If ZB == ZB_Max and Val == 0,
> returns numeric_limits<T>::max(). Otherwise returns
> `countLeadingZeros(Val, ZB_Undefined) ^ (numeric_limits<T>::digits - 1)`, i.e.
> (width - 1) minus the leading-zero count.

> [spec:et:def:llvm-math-extras.executorch.llvm.float-to-bits-fn]
> inline uint32_t FloatToBits(float Float)

> [spec:et:sem:llvm-math-extras.executorch.llvm.float-to-bits-fn]
> Reinterprets the float `Float` as its 32-bit pattern (bitwise memcpy). NaN bits
> may change on some hosts. Rust: `Float.to_bits()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.greatest-common-divisor64-fn]
> inline uint64_t GreatestCommonDivisor64(uint64_t A, uint64_t B)

> [spec:et:sem:llvm-math-extras.executorch.llvm.greatest-common-divisor64-fn]
> Euclid's algorithm for gcd of two u64. While B != 0: T = B; B = A % B; A = T.
> Returns A. gcd(A, 0) == A.

> [spec:et:def:llvm-math-extras.executorch.llvm.hi-32-fn]
> constexpr inline uint32_t Hi_32(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.hi-32-fn]
> Returns the high 32 bits of `Value` as u32: `(Value >> 32) as u32`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-int-16-fn]
> constexpr inline bool isInt<16>(int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-16-fn]
> 16-bit specialization of isInt: true iff x fits in a signed 16-bit integer,
> i.e. `(x as i16) as i64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-int-32-fn]
> constexpr inline bool isInt<32>(int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-32-fn]
> 32-bit specialization of isInt: true iff x fits in a signed 32-bit integer,
> i.e. `(x as i32) as i64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-int-8-fn]
> constexpr inline bool isInt<8>(int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-8-fn]
> 8-bit specialization of isInt: true iff x fits in a signed 8-bit integer,
> i.e. `(x as i8) as i64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-int-fn]
> constexpr inline bool isInt(int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-fn]
> Generic N-bit fit check for a signed value: returns true iff x fits in an N-bit
> two's complement integer. Returns `N >= 64 || (-(1 << (N-1)) <= x && x < (1 << (N-1)))`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-int-n-fn]
> inline bool isIntN(unsigned N, int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-int-n-fn]
> Dynamic-width signed fit check: returns `N >= 64 || (minIntN(N) <= x && x <= maxIntN(N))`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-mask-32-fn]
> constexpr inline bool isMask_32(uint32_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-32-fn]
> True iff Value is a non-empty run of 1 bits starting at bit 0 with the rest
> zero (e.g. 0x0000FFFF). Computed as `Value != 0 && ((Value + 1) & Value) == 0`
> using wrapping u32 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-mask-64-fn]
> constexpr inline bool isMask_64(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-mask-64-fn]
> 64-bit version of isMask_32: `Value != 0 && ((Value + 1) & Value) == 0` with
> wrapping u64 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-power-of2-32-fn]
> constexpr inline bool isPowerOf2_32(uint32_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-32-fn]
> True iff Value is a power of two > 0: `Value != 0 && (Value & (Value - 1)) == 0`
> with wrapping u32 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-power-of2-64-fn]
> constexpr inline bool isPowerOf2_64(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-power-of2-64-fn]
> 64-bit version: `Value != 0 && (Value & (Value - 1)) == 0` with wrapping u64
> arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-int-fn]
> constexpr inline bool isShiftedInt(int64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-int-fn]
> Template<N, S>. True iff x is an N-bit signed value shifted left by S:
> `isInt<N + S>(x) && (x as u64 % (1u64 << S) == 0)`. Static-asserts N > 0 and
> N + S <= 64.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-mask-32-fn]
> constexpr inline bool isShiftedMask_32(uint32_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-32-fn]
> True iff Value is a non-empty contiguous run of 1 bits (anywhere) with the rest
> zero (e.g. 0x0000FF00). Computed as `Value != 0 && isMask_32((Value - 1) | Value)`
> with wrapping u32 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-mask-64-fn]
> constexpr inline bool isShiftedMask_64(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-mask-64-fn]
> 64-bit version: `Value != 0 && isMask_64((Value - 1) | Value)` with wrapping
> u64 arithmetic.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-shifted-u-int-fn]
> constexpr inline bool isShiftedUInt(uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-shifted-u-int-fn]
> Template<N, S>. True iff x is an N-bit unsigned value shifted left by S:
> `isUInt<N + S>(x) && (x % (1u64 << S) == 0)`. Static-asserts N > 0 and
> N + S <= 64.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-16-fn]
> constexpr inline bool isUInt<16>(uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-16-fn]
> 16-bit specialization of isUInt: true iff x fits in an unsigned 16-bit integer,
> i.e. `(x as u16) as u64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-32-fn]
> constexpr inline bool isUInt<32>(uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-32-fn]
> 32-bit specialization of isUInt: true iff x fits in an unsigned 32-bit integer,
> i.e. `(x as u32) as u64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-8-fn]
> constexpr inline bool isUInt<8>(uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-8-fn]
> 8-bit specialization of isUInt: true iff x fits in an unsigned 8-bit integer,
> i.e. `(x as u8) as u64 == x`.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-fn]
> constexpr inline typename std::enable_if<(N < 64), bool>::type isUInt( uint64_t X)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-fn]
> Generic N-bit unsigned fit check. For N < 64: static-asserts N > 0 and returns
> `X < (1u64 << N)`. For N >= 64 the companion overload returns true unconditionally.

> [spec:et:def:llvm-math-extras.executorch.llvm.is-u-int-n-fn]
> inline bool isUIntN(unsigned N, uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.is-u-int-n-fn]
> Dynamic-width unsigned fit check: returns `N >= 64 || x <= maxUIntN(N)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.lo-32-fn]
> constexpr inline uint32_t Lo_32(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.lo-32-fn]
> Returns the low 32 bits of `Value` as u32: `Value as u32` (truncation).

> [spec:et:def:llvm-math-extras.executorch.llvm.log2-32-ceil-fn]
> inline unsigned Log2_32_Ceil(uint32_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-ceil-fn]
> Returns ceil(log2(Value)) for a u32, and 32 when Value == 0. Computed as
> `32 - countLeadingZeros(Value - 1)` with wrapping subtraction (so Value == 0
> passes 0xFFFFFFFF, whose clz is 0, giving 32; Value == 1 passes 0, clz 32,
> giving 0).

> [spec:et:def:llvm-math-extras.executorch.llvm.log2-32-fn]
> inline unsigned Log2_32(uint32_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.log2-32-fn]
> Returns floor(log2(Value)) for a u32, and (unsigned)(-1) == 0xFFFFFFFF when
> Value == 0. Computed as `31 - countLeadingZeros(Value)`: for Value == 0,
> countLeadingZeros returns 32 (ZB_Width default), so 31 - 32 wraps to 0xFFFFFFFF.

> [spec:et:def:llvm-math-extras.executorch.llvm.log2-64-ceil-fn]
> inline unsigned Log2_64_Ceil(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-ceil-fn]
> Returns ceil(log2(Value)) for a u64, and 64 when Value == 0. Computed as
> `64 - countLeadingZeros(Value - 1)` with wrapping subtraction.

> [spec:et:def:llvm-math-extras.executorch.llvm.log2-64-fn]
> inline unsigned Log2_64(uint64_t Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.log2-64-fn]
> Returns floor(log2(Value)) for a u64, and (unsigned)(-1) == 0xFFFFFFFF when
> Value == 0. Computed as `63 - countLeadingZeros(Value)`; for Value == 0,
> countLeadingZeros returns 64, so 63 - 64 wraps to 0xFFFFFFFF.

> [spec:et:def:llvm-math-extras.executorch.llvm.log2-fn]
> inline double Log2(double Value)

> [spec:et:sem:llvm-math-extras.executorch.llvm.log2-fn]
> Returns log base 2 of a double: `log2(Value)` (on old Android, log(Value)/log(2)).
> Rust: `Value.log2()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.make-64-fn]
> constexpr inline uint64_t Make_64(uint32_t High, uint32_t Low)

> [spec:et:sem:llvm-math-extras.executorch.llvm.make-64-fn]
> Combines two u32 into a u64: `((High as u64) << 32) | (Low as u64)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.mask-leading-ones-fn]
> T maskLeadingOnes(unsigned N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-ones-fn]
> Returns an unsigned T with the N most-significant bits set to 1 and the rest 0.
> Computed as `!maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N)` (bitwise NOT of the
> low-N-cleared mask).

> [spec:et:def:llvm-math-extras.executorch.llvm.mask-leading-zeros-fn]
> T maskLeadingZeros(unsigned N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.mask-leading-zeros-fn]
> Returns an unsigned T with the N most-significant bits set to 0 and the rest 1.
> Computed as `maskTrailingOnes<T>(CHAR_BIT * sizeof(T) - N)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.mask-trailing-ones-fn]
> T maskTrailingOnes(unsigned N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-ones-fn]
> Returns an unsigned T with the N least-significant bits set to 1 and the rest 0.
> Bits = CHAR_BIT * sizeof(T); asserts N <= Bits. Returns 0 when N == 0, else
> `T(-1) >> (Bits - N)` (all-ones shifted right).

> [spec:et:def:llvm-math-extras.executorch.llvm.mask-trailing-zeros-fn]
> T maskTrailingZeros(unsigned N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.mask-trailing-zeros-fn]
> Returns an unsigned T with the N least-significant bits set to 0 and the rest 1.
> Computed as `maskLeadingOnes<T>(CHAR_BIT * sizeof(T) - N)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.max-int-n-fn]
> inline int64_t maxIntN(int64_t N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.max-int-n-fn]
> Maximum value of an N-bit signed integer. Asserts 0 < N <= 64. Returns
> `(1u64 << (N - 1)) - 1` computed in u64 then cast to i64 (so N == 64 relies on
> two's-complement wraparound to give i64::MAX).

> [spec:et:def:llvm-math-extras.executorch.llvm.max-u-int-n-fn]
> inline uint64_t maxUIntN(uint64_t N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.max-u-int-n-fn]
> Maximum value of an N-bit unsigned integer. Asserts 0 < N <= 64. Returns
> `UINT64_MAX >> (64 - N)` (branch-free, avoids the UB of `1 << 64`).

> [spec:et:def:llvm-math-extras.executorch.llvm.min-align-fn]
> constexpr inline uint64_t MinAlign(uint64_t A, uint64_t B)

> [spec:et:sem:llvm-math-extras.executorch.llvm.min-align-fn]
> Given two alignments/offsets, returns the largest power of two dividing both:
> `(A | B) & (1 + ~(A | B))` (isolates the lowest set bit of A | B) with wrapping
> u64 arithmetic. Returns 0 when both A and B are 0.

> [spec:et:def:llvm-math-extras.executorch.llvm.min-int-n-fn]
> inline int64_t minIntN(int64_t N)

> [spec:et:sem:llvm-math-extras.executorch.llvm.min-int-n-fn]
> Minimum value of an N-bit signed integer. Asserts 0 < N <= 64. Returns
> `-(1u64 << (N - 1))` where the negation is on the u64 value (MSVC warning 4146)
> and the result is cast to i64.

> [spec:et:def:llvm-math-extras.executorch.llvm.next-power-of2-fn]
> inline uint64_t NextPowerOf2(uint64_t A)

> [spec:et:sem:llvm-math-extras.executorch.llvm.next-power-of2-fn]
> Returns the next power of two strictly greater than A, or 0 on overflow.
> Smears the highest set bit down through all lower bits via
> A |= A >> {1,2,4,8,16,32}, then returns A + 1 (wrapping). NextPowerOf2(0) == 1.

> [spec:et:def:llvm-math-extras.executorch.llvm.offset-to-alignment-fn]
> inline uint64_t OffsetToAlignment(uint64_t Value, uint64_t Align)

> [spec:et:sem:llvm-math-extras.executorch.llvm.offset-to-alignment-fn]
> Returns the offset from Value up to the next multiple of Align:
> `alignTo(Value, Align) - Value`.

> [spec:et:def:llvm-math-extras.executorch.llvm.power-of2-ceil-fn]
> inline uint64_t PowerOf2Ceil(uint64_t A)

> [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-ceil-fn]
> Returns the smallest power of two >= A. Returns 0 when A == 0; otherwise
> `NextPowerOf2(A - 1)`.

> [spec:et:def:llvm-math-extras.executorch.llvm.power-of2-floor-fn]
> inline uint64_t PowerOf2Floor(uint64_t A)

> [spec:et:sem:llvm-math-extras.executorch.llvm.power-of2-floor-fn]
> Returns the largest power of two <= A. Returns 0 when A == 0; otherwise
> `1u64 << (63 - countLeadingZeros(A, ZB_Undefined))`.

> [spec:et:def:llvm-math-extras.executorch.llvm.reverse-bits-fn]
> T reverseBits(T Val)

> [spec:et:sem:llvm-math-extras.executorch.llvm.reverse-bits-fn]
> Reverses the bit order of Val across its full byte width. Copies the bytes of
> Val, replaces each input byte i with BitReverseTable256[byte] and writes it to
> output position (sizeof - i - 1) (reversing byte order too), then copies back.
> Rust: `Val.reverse_bits()`.

> [spec:et:def:llvm-math-extras.executorch.llvm.saturating-add-fn]
> typename std::enable_if<std::is_unsigned<T>::value, T>::type SaturatingAdd( T X, T Y, bool* ResultOverflowed = nullptr)

> [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-add-fn]
> Unsigned saturating add. Z = X + Y (wrapping). Overflowed = (Z < X || Z < Y).
> If overflowed, writes true through ResultOverflowed (if non-null) and returns
> numeric_limits<T>::max(); else writes false and returns Z.

> [spec:et:def:llvm-math-extras.executorch.llvm.saturating-multiply-add-fn]
> typename std::enable_if<std::is_unsigned<T>::value, T>::type SaturatingMultiplyAdd(T X, T Y, T A, bool* ResultOverflowed = nullptr)

> [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-add-fn]
> Unsigned saturating X*Y + A. Product = SaturatingMultiply(X, Y, &Overflowed);
> if the multiply overflowed, sets ResultOverflowed and returns Product (already
> saturated to max). Otherwise returns SaturatingAdd(A, Product, &Overflowed).

> [spec:et:def:llvm-math-extras.executorch.llvm.saturating-multiply-fn]
> typename std::enable_if<std::is_unsigned<T>::value, T>::type SaturatingMultiply( T X, T Y, bool* ResultOverflowed = nullptr)

> [spec:et:sem:llvm-math-extras.executorch.llvm.saturating-multiply-fn]
> Unsigned saturating multiply. Overflowed = false. Estimate
> Log2Z = Log2_64(X) + Log2_64(Y) (if X or Y is 0, Log2_64 returns 0xFFFFFFFF, so
> the sum is small/negative in int arithmetic and the product 0 is returned).
> Log2Max = Log2_64(max). If Log2Z < Log2Max return X*Y. If Log2Z > Log2Max set
> Overflowed and return max. Otherwise borderline: Z = (X >> 1) * Y; if
> `Z & ~(max >> 1)` set Overflowed and return max; Z <<= 1; if (X & 1) return
> SaturatingAdd(Z, Y, ResultOverflowed); else return Z. All arithmetic on T.

> [spec:et:def:llvm-math-extras.executorch.llvm.sign-extend32-fn]
> constexpr inline int32_t SignExtend32(uint32_t X)

> [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend32-fn]
> Template<B> sign-extends the low B bits of X to a full i32. Static-asserts
> 0 < B <= 32. Returns `(i32)(X << (32 - B)) >> (32 - B)` (left shift to the top
> then arithmetic right shift). A companion dynamic overload takes B as a runtime
> argument with the same body.

> [spec:et:def:llvm-math-extras.executorch.llvm.sign-extend64-fn]
> constexpr inline int64_t SignExtend64(uint64_t x)

> [spec:et:sem:llvm-math-extras.executorch.llvm.sign-extend64-fn]
> Template<B> sign-extends the low B bits of x to a full i64. Static-asserts
> 0 < B <= 64. Returns `(i64)(x << (64 - B)) >> (64 - B)`. A companion dynamic
> overload takes B as a runtime argument with the same body.

> [spec:et:def:llvm-math-extras.executorch.llvm.zero-behavior]
> enum ZeroBehavior {
>   ZB_Undefined;
>   ZB_Max;
>   ZB_Width;
> }
