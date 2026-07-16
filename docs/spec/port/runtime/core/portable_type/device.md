# runtime/core/portable_type/device.h

> [spec:et:def:device.executorch.runtime.etensor.device]
> struct Device final {
>   DeviceType type_;
>   DeviceIndex index_ = 0;
> }

> [spec:et:def:device.executorch.runtime.etensor.device-type]
> enum class DeviceType : int8_t {
>   CPU = 0;
>   CUDA = 1;
> }

> [spec:et:def:device.executorch.runtime.etensor.device.device-fn]
> Device(DeviceType type, DeviceIndex index = 0)

> [spec:et:sem:device.executorch.runtime.etensor.device.device-fn]
> Constructs a `Device` from a `DeviceType type` and an optional
> `DeviceIndex index` (an `int8_t`) that defaults to `0`. Member-initializes
> `type_` from `type` and `index_` from `index`; performs no validation of
> either argument (any `int8_t` index and any `DeviceType` enumerator, even an
> out-of-range value, is accepted verbatim). The constructor is implicit
> (`/* implicit */`), so a bare `DeviceType` can convert to a `Device` with
> index 0. No side effects, cannot fail. In a Rust port: `Device::new(type:
> DeviceType, index: i8) -> Device`, with a convenience default of `index = 0`.

> [spec:et:def:device.executorch.runtime.etensor.device.index-fn]
> DeviceIndex index() const noexcept

> [spec:et:sem:device.executorch.runtime.etensor.device.index-fn]
> Returns the stored device index `index_` (a `DeviceIndex`, i.e. `int8_t`) by
> value. Pure `const noexcept` accessor: no mutation, no validation, cannot
> fail. In a Rust port: `fn index(&self) -> i8 { self.index }`.

> [spec:et:def:device.executorch.runtime.etensor.device.is-cpu-fn]
> bool is_cpu() const noexcept

> [spec:et:sem:device.executorch.runtime.etensor.device.is-cpu-fn]
> Returns `true` iff the stored `type_` equals `DeviceType::CPU` (enum value
> 0), otherwise `false` (e.g. for `DeviceType::CUDA`). Ignores `index_`. Pure
> `const noexcept` predicate; no mutation, cannot fail. In a Rust port: `fn
> is_cpu(&self) -> bool { self.type_ == DeviceType::CPU }`.

> [spec:et:def:device.executorch.runtime.etensor.device.operator-fn]
> bool operator==(const Device& other) const noexcept

> [spec:et:sem:device.executorch.runtime.etensor.device.operator-fn]
> Equality comparison. Returns `true` iff both fields match: `type_ ==
> other.type_` AND `index_ == other.index_`. Short-circuits on the type
> comparison (per `&&`). Pure `const noexcept`; no mutation, cannot fail. The
> sibling `operator!=` (not separately specced) is defined as `!(*this ==
> other)`. In a Rust port this is a derived `PartialEq`/`Eq` comparing both the
> `type` and `index` fields.

> [spec:et:def:device.executorch.runtime.etensor.device.type-fn]
> DeviceType type() const noexcept

> [spec:et:sem:device.executorch.runtime.etensor.device.type-fn]
> Returns the stored device type `type_` (a `DeviceType`) by value. Pure
> `const noexcept` accessor: no mutation, no validation, cannot fail. In a Rust
> port: `fn type_(&self) -> DeviceType { self.type_ }`.

