# runtime/core/portable_type/scalar_type.h

> [spec:et:def:scalar-type.executorch.runtime.etensor.scalar-type]
> enum class ScalarType : int8_t {
>   Undefined;
>   NumOptions;
> }

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn]
> struct alignas(1) Float8_e4m3fn {
>   uint8_t x;
> }

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn.float8-e4m3fn-fn]
> Float8_e4m3fn() = default

> [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fn.float8-e4m3fn-fn]
> Compiler-defaulted default constructor for `unused_dtype::Float8_e4m3fn`.
> Constructs the struct leaving its single member `x` (a `uint8_t`, also
> exposed as `using underlying = uint8_t`) uninitialized — no zeroing, no side
> effects. This is a placeholder/storage-only type (alignment 1, size 1 byte)
> that ExecuTorch defines solely to keep the `ScalarType` enum indices aligned
> with ATen; it is an "unused dtype" with no float8 arithmetic implemented —
> the byte is not interpreted as an FP8 e4m3fn value here. In a Rust port this
> maps to a default that leaves the wrapped `u8` unspecified (matching C++
> default-init of a trivial member); to store a defined byte use the explicit
> constructor `Float8_e4m3fn(uint8_t)` which sets `x` to the given value.

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz]
> struct alignas(1) Float8_e4m3fnuz {
>   uint8_t x;
> }

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz.float8-e4m3fnuz-fn]
> Float8_e4m3fnuz() = default

> [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e4m3fnuz.float8-e4m3fnuz-fn]
> Compiler-defaulted default constructor for `unused_dtype::Float8_e4m3fnuz`.
> Constructs the struct leaving its single member `x` (a `uint8_t`, also
> exposed as `using underlying = uint8_t`) uninitialized — no zeroing, no side
> effects. This is a placeholder/storage-only type (alignment 1, size 1 byte)
> that ExecuTorch defines solely to keep the `ScalarType` enum indices aligned
> with ATen; it is an "unused dtype" with no float8 arithmetic implemented —
> the byte is not interpreted as an FP8 e4m3fnuz value here. In a Rust port
> this maps to a default that leaves the wrapped `u8` unspecified (matching C++
> default-init of a trivial member); to store a defined byte use the explicit
> constructor `Float8_e4m3fnuz(uint8_t)` which sets `x` to the given value.

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2]
> struct alignas(1) Float8_e5m2 {
>   uint8_t x;
> }

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2.float8-e5m2-fn]
> Float8_e5m2() = default

> [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2.float8-e5m2-fn]
> Compiler-defaulted default constructor for `unused_dtype::Float8_e5m2`.
> Constructs the struct leaving its single member `x` (a `uint8_t`, also
> exposed as `using underlying = uint8_t`) uninitialized — no zeroing, no side
> effects. This is a placeholder/storage-only type (alignment 1, size 1 byte)
> that ExecuTorch defines solely to keep the `ScalarType` enum indices aligned
> with ATen; it is an "unused dtype" with no float8 arithmetic implemented —
> the byte is not interpreted as an FP8 e5m2 value here. In a Rust port this
> maps to a default that leaves the wrapped `u8` unspecified (matching C++
> default-init of a trivial member); to store a defined byte use the explicit
> constructor `Float8_e5m2(uint8_t)` which sets `x` to the given value.

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz]
> struct alignas(1) Float8_e5m2fnuz {
>   uint8_t x;
> }

> [spec:et:def:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz.float8-e5m2fnuz-fn]
> Float8_e5m2fnuz() = default

> [spec:et:sem:scalar-type.executorch.runtime.etensor.unused-dtype.float8-e5m2fnuz.float8-e5m2fnuz-fn]
> Compiler-defaulted default constructor for `unused_dtype::Float8_e5m2fnuz`.
> Constructs the struct leaving its single member `x` (a `uint8_t`, also
> exposed as `using underlying = uint8_t`) uninitialized — no zeroing, no side
> effects. This is a placeholder/storage-only type (alignment 1, size 1 byte)
> that ExecuTorch defines solely to keep the `ScalarType` enum indices aligned
> with ATen; it is an "unused dtype" with no float8 arithmetic implemented —
> the byte is not interpreted as an FP8 e5m2fnuz value here. In a Rust port
> this maps to a default that leaves the wrapped `u8` unspecified (matching C++
> default-init of a trivial member); to store a defined byte use the explicit
> constructor `Float8_e5m2fnuz(uint8_t)` which sets `x` to the given value.

