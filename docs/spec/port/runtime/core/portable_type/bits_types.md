# runtime/core/portable_type/bits_types.h

> [spec:et:def:bits-types.executorch.runtime.etensor.bits16]
> struct alignas(2) bits16 {
>   uint16_t val_;
> }

> [spec:et:def:bits-types.executorch.runtime.etensor.bits16.bits16-fn]
> bits16() = default

> [spec:et:sem:bits-types.executorch.runtime.etensor.bits16.bits16-fn]
> Compiler-defaulted default constructor for `bits16`. Constructs a `bits16`
> whose single member `val_` (a `uint16_t`) is left uninitialized — no
> zeroing, no side effects. `bits16` is a 16-bit uninterpreted dtype
> (alignment 2, size 2 bytes) carrying no numeric semantics; it merely holds
> 16 raw bits. In a Rust port this maps to a default that leaves the wrapped
> `u16` in an unspecified/uninitialized state (matching C++ default-init of a
> trivial member); callers that need a defined value must use the explicit
> value constructor `bits16(uint16_t)` which sets `val_` to the given bits.

> [spec:et:def:bits-types.executorch.runtime.etensor.bits1x8]
> struct alignas(1) bits1x8 {
>   uint8_t val_;
> }

> [spec:et:def:bits-types.executorch.runtime.etensor.bits1x8.bits1x8-fn]
> bits1x8() = default

> [spec:et:sem:bits-types.executorch.runtime.etensor.bits1x8.bits1x8-fn]
> Compiler-defaulted default constructor for `bits1x8`. Constructs a `bits1x8`
> whose single member `val_` (a `uint8_t`, also exposed as `using underlying =
> uint8_t`) is left uninitialized — no zeroing, no side effects. `bits1x8` is
> an uninterpreted dtype representing 1 bit packed to a byte boundary
> (alignment 1, size 1 byte); it defines no numeric semantics and just holds 8
> raw bits. In a Rust port this maps to a default that leaves the wrapped `u8`
> unspecified (matching C++ default-init of a trivial member); to obtain a
> defined value use the explicit constructor `bits1x8(uint8_t)` which sets
> `val_` to the given byte.

> [spec:et:def:bits-types.executorch.runtime.etensor.bits2x4]
> struct alignas(1) bits2x4 {
>   uint8_t val_;
> }

> [spec:et:def:bits-types.executorch.runtime.etensor.bits2x4.bits2x4-fn]
> bits2x4() = default

> [spec:et:sem:bits-types.executorch.runtime.etensor.bits2x4.bits2x4-fn]
> Compiler-defaulted default constructor for `bits2x4`. Constructs a `bits2x4`
> whose single member `val_` (a `uint8_t`, also exposed as `using underlying =
> uint8_t`) is left uninitialized — no zeroing, no side effects. `bits2x4` is
> an uninterpreted dtype representing 2 bits packed to a byte boundary
> (alignment 1, size 1 byte); it defines no numeric semantics and just holds 8
> raw bits. In a Rust port this maps to a default that leaves the wrapped `u8`
> unspecified (matching C++ default-init of a trivial member); to obtain a
> defined value use the explicit constructor `bits2x4(uint8_t)` which sets
> `val_` to the given byte.

> [spec:et:def:bits-types.executorch.runtime.etensor.bits4x2]
> struct alignas(1) bits4x2 {
>   uint8_t val_;
> }

> [spec:et:def:bits-types.executorch.runtime.etensor.bits4x2.bits4x2-fn]
> bits4x2() = default

> [spec:et:sem:bits-types.executorch.runtime.etensor.bits4x2.bits4x2-fn]
> Compiler-defaulted default constructor for `bits4x2`. Constructs a `bits4x2`
> whose single member `val_` (a `uint8_t`, also exposed as `using underlying =
> uint8_t`) is left uninitialized — no zeroing, no side effects. `bits4x2` is
> an uninterpreted dtype representing 4 bits packed to a byte boundary
> (alignment 1, size 1 byte); it defines no numeric semantics and just holds 8
> raw bits. In a Rust port this maps to a default that leaves the wrapped `u8`
> unspecified (matching C++ default-init of a trivial member); to obtain a
> defined value use the explicit constructor `bits4x2(uint8_t)` which sets
> `val_` to the given byte.

> [spec:et:def:bits-types.executorch.runtime.etensor.bits8]
> struct alignas(1) bits8 {
>   uint8_t val_;
> }

> [spec:et:def:bits-types.executorch.runtime.etensor.bits8.bits8-fn]
> bits8() = default

> [spec:et:sem:bits-types.executorch.runtime.etensor.bits8.bits8-fn]
> Compiler-defaulted default constructor for `bits8`. Constructs a `bits8`
> whose single member `val_` (a `uint8_t`) is left uninitialized — no zeroing,
> no side effects. Unlike the other bits types, `bits8` does not declare a
> `using underlying` alias. `bits8` is an uninterpreted 8-bit dtype (alignment
> 1, size 1 byte) with no numeric semantics; it merely holds 8 raw bits. In a
> Rust port this maps to a default that leaves the wrapped `u8` unspecified
> (matching C++ default-init of a trivial member); to obtain a defined value
> use the explicit constructor `bits8(uint8_t)` which sets `val_` to the given
> byte.

