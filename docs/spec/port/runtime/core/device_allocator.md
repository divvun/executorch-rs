# runtime/core/device_allocator.cpp, runtime/core/device_allocator.h

> [spec:et:def:device-allocator.executorch.runtime.device-allocator]
> class DeviceAllocator {
>   static constexpr size_t kDefaultAlignment = MemoryAllocator::kDefaultAlignment;
> }

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry]
> class DeviceAllocatorRegistry {
>   static DeviceAllocatorRegistry& instance();
>   DeviceAllocator* allocators_[etensor::kNumDeviceTypes] = {};
> }

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn]
> DeviceAllocatorRegistry() = default

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn]
> Private default constructor (`= default`) of `DeviceAllocatorRegistry`. It
> is private so instances can only be created by `instance()`
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]`),
> enforcing the singleton pattern. It performs no explicit work: the sole data
> member `allocators_[etensor::kNumDeviceTypes]` is default-member-initialized
> to all-null (`= {}`), giving a registry with no allocators registered for
> any device type. `etensor::kNumDeviceTypes` is 2 (indices for `CPU` = 0 and
> `CUDA` = 1), so the array has 2 null slots. The registry is non-copyable and
> non-movable (copy/move ctors and assignments are deleted).

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn]
> DeviceAllocator* DeviceAllocatorRegistry::get_allocator( etensor::DeviceType type)

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn]
> Looks up the registered allocator for a device type. Steps:
> 1. Compute `index = static_cast<size_t>(type)` (the `DeviceType` enum's
>    underlying integer; `CPU` = 0, `CUDA` = 1).
> 2. Bounds check: if `index >= etensor::kNumDeviceTypes` (i.e. `>= 2`),
>    return `nullptr`. This guards against out-of-range/invalid enum values
>    without aborting.
> 3. Otherwise return `allocators_[index]`, which is either the previously
>    registered `DeviceAllocator*` for that type or `nullptr` if none was
>    registered.
>
> No mutation of registry state occurs, so this is safe to call concurrently
> with other `get_allocator()` calls (but not concurrently with
> `register_allocator()`). Returns a raw non-owning pointer; the registry does
> not own the allocator's lifetime (allocators are expected to be static
> singletons).

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]
> void DeviceAllocatorRegistry::register_allocator(DeviceAllocator* alloc)

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]
> Registers a `DeviceAllocator` into the singleton registry's fixed-size
> array, keyed by the allocator's own device type. Not thread-safe; intended
> to run once per device type during static initialization. Steps:
> 1. `ET_CHECK_MSG(alloc != nullptr, "Cannot register a null allocator")`: if
>    `alloc` is null, abort the program with that message (`ET_CHECK_MSG` is a
>    fatal assertion, not a recoverable error).
> 2. Query the allocator's type: `type = alloc->device_type()`
>    (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-type-fn]`)
>    and compute `index = static_cast<size_t>(type)`.
> 3. `ET_CHECK_MSG(index < etensor::kNumDeviceTypes, ...)`: if the index is
>    out of range (`>= 2`), abort with "Invalid device type: %d" (type cast to
>    `int`).
> 4. `ET_CHECK_MSG(allocators_[index] == nullptr, ...)`: if a non-null
>    allocator is already registered for that device type, abort with
>    "Allocator already registered for device type: %d". Each device type may
>    be registered at most once.
> 5. Store the pointer: `allocators_[index] = alloc`. The registry does not
>    take ownership; `alloc` must have static lifetime.
>
> All failure paths abort via `ET_CHECK_MSG`; there is no error return value
> (return type is `void`).

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.allocate-fn]
> virtual Result<void*> allocate( size_t nbytes, etensor::DeviceIndex index, size_t alignment = kDefaultAlignment) = 0

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]
> Pure virtual (`= 0`) method on the abstract `DeviceAllocator`; the base
> class provides no body. Contract for concrete implementations: allocate
> `nbytes` bytes of memory on the device identified by `index`, with the
> returned pointer aligned to at least `alignment` bytes.
>
> Parameters:
> - `nbytes`: number of bytes to allocate.
> - `index` (`etensor::DeviceIndex`, an `int8_t`): which device (e.g. GPU 0 vs
>   GPU 1) to allocate on.
> - `alignment` (defaults to `DeviceAllocator::kDefaultAlignment`, which is
>   `MemoryAllocator::kDefaultAlignment`): minimum alignment of the returned
>   pointer, in bytes; must be a power of 2.
>
> Return value: `Result<void*>`. On success, holds a device pointer to at
> least `nbytes` bytes aligned to at least `alignment`. On failure, holds an
> `Error` (e.g. `Error::MemoryAllocationFailed`). The returned pointer is a
> device pointer and is not necessarily dereferenceable from the host. The
> memory must later be released via
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.deallocate-fn]`
> on the same allocator with the same `index`.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn]
> virtual Error copy_device_to_host( void* dst, const void* src, size_t nbytes, etensor::DeviceIndex index) = 0

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn]
> Pure virtual (`= 0`); no base-class body. Contract: copy `nbytes` bytes from
> `src` (a device pointer, e.g. memory obtained from
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]`)
> into `dst` (a host pointer), for the device identified by `index`. The
> mirror of
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn]`.
>
> Parameters: `dst` (destination, host memory), `src` (source, device memory),
> `nbytes` (byte count), `index` (`etensor::DeviceIndex`).
>
> Return value: `Error::Ok` on success, or an appropriate error code (e.g.
> `Error::AccessFailed`) on failure. Both buffers must be at least `nbytes`
> long; the direction (device -> host) distinguishes this from the
> host-to-device variant.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn]
> virtual Error copy_host_to_device( void* dst, const void* src, size_t nbytes, etensor::DeviceIndex index) = 0

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-host-to-device-fn]
> Pure virtual (`= 0`); no base-class body. Contract: copy `nbytes` bytes from
> `src` (a host pointer) into `dst` (a device pointer, e.g. memory obtained
> from
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]`),
> for the device identified by `index`. The mirror of
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.copy-device-to-host-fn]`.
>
> Parameters: `dst` (destination, device memory), `src` (source, host memory),
> `nbytes` (byte count), `index` (`etensor::DeviceIndex`).
>
> Return value: `Error::Ok` on success, or an appropriate error code (e.g.
> `Error::AccessFailed`) on failure. Both buffers must be at least `nbytes`
> long; the direction (host -> device) distinguishes this from the
> device-to-host variant.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.deallocate-fn]
> virtual void deallocate(void* ptr, etensor::DeviceIndex index) = 0

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.deallocate-fn]
> Pure virtual (`= 0`); no base-class body. Contract: free device memory
> previously returned by
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator.allocate-fn]`
> from the same allocator. Parameters: `ptr` (the device pointer to free) and
> `index` (`etensor::DeviceIndex`, the device the memory lives on). Returns
> `void`; there is no error channel. Implementations are expected to tolerate
> a null `ptr` as a no-op (mirroring how
> `[spec:et:sem:device-memory-buffer.executorch-runtime.device-memory-buffer.operator-fn]`
> and the `DeviceMemoryBuffer` destructor only call `deallocate` when `ptr` is
> non-null, so a well-behaved caller never passes null). Passing a pointer not
> obtained from this allocator, or the wrong `index`, is undefined behavior.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.device-allocator-fn]
> virtual ~DeviceAllocator() = default

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-allocator-fn]
> Virtual destructor of the abstract `DeviceAllocator` base class, defaulted
> (`= default`). Performs no work of its own — the base class has no data
> members. It is `virtual` so that destroying a concrete allocator through a
> base-class pointer dispatches to the derived destructor. In practice
> allocators are static singletons, so this rarely runs; concrete
> implementations that hold device-side state are responsible for their own
> cleanup in their derived destructors.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator.device-type-fn]
> virtual etensor::DeviceType device_type() const = 0

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator.device-type-fn]
> Pure virtual (`= 0`), `const`; no base-class body. Contract: return the
> `etensor::DeviceType` this allocator handles (e.g. `DeviceType::CUDA`). Takes
> no arguments and has no side effects. Used by the registry
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]`)
> to determine the array slot under which the allocator is registered, and it
> must return the same value for the lifetime of the allocator.

> [spec:et:def:device-allocator.executorch.runtime.get-device-allocator-fn]
> DeviceAllocator* get_device_allocator(etensor::DeviceType type)

> [spec:et:sem:device-allocator.executorch.runtime.get-device-allocator-fn]
> Free convenience function. Delegates to the singleton registry: returns
> `DeviceAllocatorRegistry::instance()`
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]`)
> `.get_allocator(type)`
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.get-allocator-fn]`).
> Returns the registered `DeviceAllocator*` for `type`, or `nullptr` if none
> is registered or `type` is out of range. Non-owning pointer.

> [spec:et:def:device-allocator.executorch.runtime.register-device-allocator-fn]
> void register_device_allocator(DeviceAllocator* alloc)

> [spec:et:sem:device-allocator.executorch.runtime.register-device-allocator-fn]
> Free convenience function returning `void`. Delegates to the singleton
> registry: calls `DeviceAllocatorRegistry::instance()`
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]`)
> `.register_allocator(alloc)`
> (`[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.register-allocator-fn]`).
> Inherits that method's full contract, including its `ET_CHECK_MSG`
> abort-on-failure behavior for a null allocator, an out-of-range device type,
> or a device type already registered. `alloc` must have static lifetime and
> is not owned by the registry. Not thread-safe; call during static
> initialization.

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]
> DeviceAllocatorRegistry& DeviceAllocatorRegistry::instance()

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.instance-fn]
> Static accessor returning a reference to the process-wide singleton
> `DeviceAllocatorRegistry`. Uses a function-local `static
> DeviceAllocatorRegistry registry;` that is constructed lazily on first call
> (via its private default constructor,
> `[spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.device-allocator-registry-fn]`,
> yielding an all-null allocator array) and thereafter returns the same
> reference on every call. C++ guarantees the initialization of a function-
> local static is thread-safe (constructed exactly once even under concurrent
> first calls). The returned reference is never null and lives for the
> remainder of the program. In a Rust port this maps to a lazily-initialized
> global singleton (e.g. `OnceLock`/`OnceCell`).

> [spec:et:def:device-allocator.executorch.runtime.device-allocator-registry.operator-fn]
> DeviceAllocatorRegistry& operator=(const DeviceAllocatorRegistry&) = delete

> [spec:et:sem:device-allocator.executorch.runtime.device-allocator-registry.operator-fn]
> Deleted copy-assignment operator (`= delete`). It has no body and cannot be
> called; any attempt to copy-assign one `DeviceAllocatorRegistry` to another
> is a compile-time error. This, together with the deleted copy constructor
> and deleted move constructor/assignment, enforces that the registry is a
> non-copyable, non-movable singleton — preventing a shallow copy whose
> mutations would silently diverge from the real singleton returned by
> `instance()`. In a Rust port this is a non-concern: the singleton is simply
> never cloned or moved out of its global.

