# runtime/core/portable_type/qint_types.h

> [spec:et:def:qint-types.executorch.runtime.etensor.qint32]
> struct alignas(4) qint32 {
>   int32_t val_;
> }

> [spec:et:def:qint-types.executorch.runtime.etensor.qint32.qint32-fn]
> qint32() = default

> [spec:et:sem:qint-types.executorch.runtime.etensor.qint32.qint32-fn]
> Compiler-defaulted default constructor for `qint32`. Constructs a `qint32`
> whose single member `val_` (an `int32_t`, also exposed as `using underlying
> = int32_t`) is left uninitialized — no zeroing, no side effects. `qint32` is
> the storage type for signed 32-bit quantized tensor elements (alignment 4,
> size 4 bytes); the struct itself carries no scale/zero-point and applies no
> interpretation to the stored bits. In a Rust port this maps to a default
> that leaves the wrapped `i32` unspecified (matching C++ default-init of a
> trivial member); to obtain a defined value use the explicit constructor
> `qint32(int32_t)` which sets `val_` to the given value.

> [spec:et:def:qint-types.executorch.runtime.etensor.qint8]
> struct alignas(1) qint8 {
>   int8_t val_;
> }

> [spec:et:def:qint-types.executorch.runtime.etensor.qint8.qint8-fn]
> qint8() = default

> [spec:et:sem:qint-types.executorch.runtime.etensor.qint8.qint8-fn]
> Compiler-defaulted default constructor for `qint8`. Constructs a `qint8`
> whose single member `val_` (an `int8_t`, also exposed as `using underlying =
> int8_t`) is left uninitialized — no zeroing, no side effects. `qint8` is the
> storage type for signed 8-bit quantized tensor elements (alignment 1, size 1
> byte); the struct carries no scale/zero-point and applies no interpretation
> to the stored bits. In a Rust port this maps to a default that leaves the
> wrapped `i8` unspecified (matching C++ default-init of a trivial member); to
> obtain a defined value use the explicit constructor `qint8(int8_t)` which
> sets `val_` to the given value.

> [spec:et:def:qint-types.executorch.runtime.etensor.quint2x4]
> struct alignas(1) quint2x4 {
>   uint8_t val_;
> }

> [spec:et:def:qint-types.executorch.runtime.etensor.quint2x4.quint2x4-fn]
> quint2x4() = default

> [spec:et:sem:qint-types.executorch.runtime.etensor.quint2x4.quint2x4-fn]
> Compiler-defaulted default constructor for `quint2x4`. Constructs a
> `quint2x4` whose single member `val_` (a `uint8_t`, also exposed as `using
> underlying = uint8_t`) is left uninitialized — no zeroing, no side effects.
> `quint2x4` is the storage type for unsigned 2-bit quantized tensor elements
> packed to a byte boundary (alignment 1, size 1 byte, four 2-bit values per
> byte); the struct carries no scale/zero-point and applies no interpretation
> to the packed bits. In a Rust port this maps to a default that leaves the
> wrapped `u8` unspecified (matching C++ default-init of a trivial member); to
> obtain a defined value use the explicit constructor `quint2x4(uint8_t)`
> which sets `val_` to the given byte.

> [spec:et:def:qint-types.executorch.runtime.etensor.quint4x2]
> struct alignas(1) quint4x2 {
>   uint8_t val_;
> }

> [spec:et:def:qint-types.executorch.runtime.etensor.quint4x2.quint4x2-fn]
> quint4x2() = default

> [spec:et:sem:qint-types.executorch.runtime.etensor.quint4x2.quint4x2-fn]
> Compiler-defaulted default constructor for `quint4x2`. Constructs a
> `quint4x2` whose single member `val_` (a `uint8_t`, also exposed as `using
> underlying = uint8_t`) is left uninitialized — no zeroing, no side effects.
> `quint4x2` is the storage type for unsigned 4-bit quantized tensor elements
> packed to a byte boundary (alignment 1, size 1 byte, two 4-bit values per
> byte); the struct carries no scale/zero-point and applies no interpretation
> to the packed bits. In a Rust port this maps to a default that leaves the
> wrapped `u8` unspecified (matching C++ default-init of a trivial member); to
> obtain a defined value use the explicit constructor `quint4x2(uint8_t)`
> which sets `val_` to the given byte.

> [spec:et:def:qint-types.executorch.runtime.etensor.quint8]
> struct alignas(1) quint8 {
>   uint8_t val_;
> }

> [spec:et:def:qint-types.executorch.runtime.etensor.quint8.quint8-fn]
> quint8() = default

> [spec:et:sem:qint-types.executorch.runtime.etensor.quint8.quint8-fn]
> Compiler-defaulted default constructor for `quint8`. Constructs a `quint8`
> whose single member `val_` (a `uint8_t`, also exposed as `using underlying =
> uint8_t`) is left uninitialized — no zeroing, no side effects. `quint8` is
> the storage type for unsigned 8-bit quantized tensor elements (alignment 1,
> size 1 byte); the struct carries no scale/zero-point and applies no
> interpretation to the stored bits. In a Rust port this maps to a default
> that leaves the wrapped `u8` unspecified (matching C++ default-init of a
> trivial member); to obtain a defined value use the explicit constructor
> `quint8(uint8_t)` which sets `val_` to the given value.

