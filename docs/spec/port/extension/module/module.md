# extension/module/module.cpp, extension/module/module.h

> [spec:et:def:module.executorch.extension.et-module-namespace.make-data-loader-fn]
> runtime::Result<std::unique_ptr<runtime::DataLoader>> make_data_loader( const std::string& file_path, Module::LoadMode mode)

> [spec:et:sem:module.executorch.extension.et-module-namespace.make-data-loader-fn]
> File-local helper that constructs a concrete `DataLoader` for a file path
> according to the `Module::LoadMode` enum.
>
> Switch on `mode`:
> - `File`: call `FileDataLoader::from(file_path.c_str())`. This reads the
>   whole file into a buffer.
> - `Mmap`: call `MmapDataLoader::from(file_path.c_str(),
>   MmapDataLoader::MlockConfig::NoMlock)` (mmap without memory locking).
> - `MmapUseMlock`: call `MmapDataLoader::from(file_path.c_str())` with the
>   default MlockConfig (memory locking that reports errors).
> - `MmapUseMlockIgnoreErrors`: call `MmapDataLoader::from(file_path.c_str(),
>   MmapDataLoader::MlockConfig::UseMlockIgnoreErrors)`.
> - `MmapUseMadvise`: call `MmapDataLoader::from(file_path.c_str(),
>   MmapDataLoader::MlockConfig::UseMadvise)`.
>
> For whichever branch runs: if the underlying `...::from` Result is not ok,
> return its `Error` immediately (propagate). Otherwise move the loaded loader
> value onto the heap (into a `unique_ptr` of the concrete loader type) and
> assign it to the local `data_loader`.
>
> On success return the owning `unique_ptr<DataLoader>` (upcast to the base
> DataLoader type). The enum is exhaustive; no default case exists.

> [spec:et:def:module.executorch.extension.et-module-namespace.module]
> class Module {
>   enum class LoadMode { /// Load the whole file as a buffer. File, /// Use mmap to load pages into memory. Mmap, /// Use memory locking and handle errors. Mmap...;
>   ET_NODISCARD virtual runtime::Error load( const Program::Verification verification = Program::Verification::Minimal);
>   ET_NODISCARD virtual runtime::Error load( const LoadBackendOptionsMap& backend_options, const Program::Verification verification = Program::Verification::Min...;
>   ET_NODISCARD runtime::Error load_method( const std::string& method_name, runtime::HierarchicalAllocator* planned_memory = nullptr, torch::executor::EventTrac...;
>   ET_DEPRECATED ET_NODISCARD;
>   ET_DEPRECATED ET_NODISCARD;
>   ET_DEPRECATED ET_NODISCARD;
>   ET_NODISCARD virtual runtime::Result<std::vector<runtime::EValue>> execute( const std::string& method_name, const std::vector<runtime::EValue>& input_values);
>   ET_NODISCARD runtime::Error set_input( const std::string& method_name, const runtime::EValue& input_value, size_t input_index);
>   ET_NODISCARD runtime::Error set_inputs( const std::string& method_name, const std::vector<runtime::EValue>& input_values);
>   ET_NODISCARD runtime::Error set_output( const std::string& method_name, runtime::EValue output_value, size_t output_index = 0);
>   ET_NODISCARD runtime::Error set_outputs( const std::string& method_name, const std::vector<runtime::EValue>& output_values);
>   ET_NODISCARD runtime::Result<std::vector<runtime::EValue>> get_outputs( const std::string& method_name);
>   ET_NODISCARD runtime::Result<runtime::EValue> get_output( const std::string& method_name, size_t output_index = 0);
>   ET_DEPRECATED ET_NODISCARD;
>   struct PlannedMemory { std::vector<std::vector<uint8_t>> planned_buffers; std::vector<runtime::Span<uint8_t>> planned_spans; std::vector<runtime::DeviceMemor...;
>   struct MethodHolder { std::unique_ptr<PlannedMemory> planned_memory; std::unique_ptr<runtime::MemoryManager> memory_manager; std::unique_ptr<Method> method; ...;
>   std::string file_path_;
>   std::vector<std::string> data_files_;
>   LoadMode load_mode_{LoadMode::File};
>   std::shared_ptr<Program> program_;
>   std::unique_ptr<runtime::DataLoader> data_loader_;
>   std::unique_ptr<runtime::MemoryAllocator> memory_allocator_;
>   std::unique_ptr<runtime::MemoryAllocator> temp_allocator_;
>   std::unique_ptr<runtime::EventTracer> event_tracer_;
>   std::vector<std::unique_ptr<runtime::DataLoader>> data_map_loaders_;
>   std::vector<std::unique_ptr<NamedDataMap>> named_data_maps_;
>   std::unique_ptr<NamedDataMap> merged_data_map_;
>   std::vector<std::vector<uint8_t>> shared_arenas_;
>   ET_DEPRECATED std::vector<uint8_t> debug_buffer_;
>   std::vector<std::vector<runtime::BackendOption>> backend_options_storage_;
>   LoadBackendOptionsMap backend_options_map_;
>   bool share_memory_arenas_;
>   ET_NODISCARD runtime::Error load_internal( const Program::Verification verification);
>   std::unordered_map<std::string, MethodHolder> methods_;
> }

> [spec:et:def:module.executorch.extension.et-module-namespace.module.debug-buffer-fn]
> runtime::Span<uint8_t> debug_buffer()

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.debug-buffer-fn]
> Deprecated accessor. Returns a `Span<uint8_t>` constructed from
> `debug_buffer_.data()` and `debug_buffer_.size()` — i.e. a view over the
> Module's `debug_buffer_` member vector.
>
> This vector is never populated by the Module, so the returned span is
> always empty (size 0). The actual debug buffer lives inside the
> `EventTracer` attached to the Module; callers should use that instead.
> No loading or validation is performed.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.event-tracer-fn]
> inline runtime::EventTracer* event_tracer() const

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.event-tracer-fn]
> Const accessor. Returns the raw `EventTracer*` held by the Module's
> `event_tracer_` member unique_ptr (i.e. `event_tracer_.get()`).
>
> Returns nullptr if no EventTracer was supplied to the constructor (the
> Module does not own or create a default tracer). Ownership stays with the
> Module; the returned pointer is a non-owning borrow.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.execute-fn]
> runtime::Result<std::vector<runtime::EValue>> Module::execute( const std::string& method_name, const std::vector<runtime::EValue>& input_values)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.execute-fn]
> Executes a method with the given inputs and returns its outputs.
>
> Steps:
> 1. Call `load_method(method_name)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`.
>    If it returns a non-Ok Error, return that Error (via `Result`).
> 2. Look up the cached `Method` for `method_name` in `methods_` (using
>    `.at()`, which assumes the entry now exists — load_method guarantees it
>    on success).
> 3. For each `index` from 0 to `input_values.size() - 1` in order, call
>    `method->set_input(input_values[index], index)`. If any returns non-Ok,
>    return that Error. (Extra method inputs beyond `input_values.size()` are
>    left at whatever value they previously held; fewer inputs than the method
>    expects are simply not set here.)
> 4. Call `method->execute()`; on non-Ok Error, return it.
> 5. Read `outputs_size = method->outputs_size()`. Allocate a
>    `std::vector<EValue>` of that length (default-constructed EValues), then
>    call `method->get_outputs(outputs.data(), outputs_size)` to fill it; on
>    non-Ok Error return it.
> 6. Return the outputs vector by value.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.get-max-mem-planned-buffer-sizes-fn]
> runtime::Result<std::vector<size_t>>

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.get-max-mem-planned-buffer-sizes-fn]
> Private helper. Computes, per memory-plan buffer index, the maximum buffer
> size needed across ALL methods in the program. Used to size shared arenas.
>
> Steps:
> 1. Initialize an empty `result` vector<size_t>.
> 2. Call `method_names()` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.method-names-fn]`
>    (which loads the program if needed). On non-Ok, return that Error.
> 3. For each method name in the returned set (iteration order is the
>    unordered_set's order and is unspecified — irrelevant since the result is
>    an elementwise max):
>    a. Call `get_mem_planned_buffer_sizes(name)` per
>       `[spec:et:sem:module.executorch.extension.et-module-namespace.module.get-mem-planned-buffer-sizes-fn]`;
>       on non-Ok, return that Error.
>    b. Let `sizes` be that method's per-buffer sizes. If `sizes.size() >
>       result.size()`, grow `result` to `sizes.size()`, filling new entries
>       with 0.
>    c. For each `i` in `[0, sizes.size())`, set `result[i] = max(result[i],
>       sizes[i])`.
> 4. Return `result`. If the program has no methods, returns an empty vector.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.get-mem-planned-buffer-sizes-fn]
> runtime::Result<std::vector<size_t>> Module::get_mem_planned_buffer_sizes( const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.get-mem-planned-buffer-sizes-fn]
> Private helper. Returns the memory-planned buffer sizes for a single method,
> in buffer-index order.
>
> Steps:
> 1. Call `program_->method_meta(method_name.c_str())`. NOTE: this does NOT
>    call `load()` first; it assumes `program_` is already loaded (callers
>    ensure this). If the meta Result's error is non-Ok, return that Error.
> 2. Get the `MethodMeta` value `meta`.
> 3. Reserve a `sizes` vector<size_t> of length
>    `meta.num_memory_planned_buffers()`.
> 4. For each `i` in `[0, meta.num_memory_planned_buffers())` in order: call
>    `meta.memory_planned_buffer_size(i)`; if its error is non-Ok return that
>    Error; otherwise append its size value to `sizes`.
> 5. Return `sizes`.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.get-output-fn]
> runtime::Result<runtime::EValue> Module::get_output( const std::string& method_name, size_t output_index)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.get-output-fn]
> Retrieves a single current output value of a method without executing it.
>
> Steps:
> 1. Call `load_method(method_name)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`;
>    on non-Ok Error, return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`.
> 3. Bounds check: if `output_index >= method->outputs_size()`, fail with
>    `Error::InvalidArgument` (log message "output index: <output_index> is out
>    of range") and return that Error.
> 4. Otherwise return `method->get_output(output_index)` (a `Result<EValue>`;
>    the current value of that output slot, which may be uninitialized/default
>    if the method has not been executed).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.get-outputs-fn]
> runtime::Result<std::vector<runtime::EValue>> Module::get_outputs( const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.get-outputs-fn]
> Retrieves all current output values of a method without executing it.
>
> Steps:
> 1. Call `load_method(method_name)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`;
>    on non-Ok Error, return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`.
> 3. Read `outputs_size = method->outputs_size()`. Allocate a
>    `std::vector<EValue>` of that length (default-constructed EValues).
> 4. Call `method->get_outputs(outputs.data(), outputs_size)` to fill it; on
>    non-Ok Error return it.
> 5. Return the outputs vector by value. Values reflect the method's current
>    output state (may be default/uninitialized if never executed).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.is-loaded-fn]
> virtual inline bool is_loaded() const

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.is-loaded-fn]
> Const, virtual accessor. Returns true iff the Module's `program_` shared
> pointer is non-null (`program_ != nullptr`), i.e. the program has been
> loaded (or was supplied directly to the constructor). Returns false
> otherwise. Performs no loading and no side effects.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.is-method-loaded-fn]
> inline bool is_method_loaded(const std::string& method_name) const

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.is-method-loaded-fn]
> Const accessor. Returns true iff a method with name `method_name` is present
> in the `methods_` cache map (`methods_.count(method_name) != 0`), i.e. the
> method has been loaded via load_method. Returns false otherwise. Performs no
> loading and no side effects.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.load-fn]
> runtime::Error Module::load( const LoadBackendOptionsMap& backend_options, const Program::Verification verification)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.load-fn]
> This rule covers the `load(LoadBackendOptionsMap, verification)` overload
> that installs per-delegate backend options. (The simpler
> `load(verification)` overload just returns `load_internal(verification)` per
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-internal-fn]`.)
>
> Behavior (transactional — on any failure the previously-installed backend
> options remain in place and the input is not committed):
> 1. Call `load_internal(verification)` first. This does not read backend
>    options, so on failure no copy work happens. If it returns non-Ok, return
>    that Error.
> 2. Build the deep copy in locals so a mid-way failure leaves member state
>    untouched:
>    a. Create `local_storage` (`vector<vector<BackendOption>>`) and
>       `reserve(backend_options.size())` so it never reallocates while being
>       filled (keeping inner-vector addresses stable). Create an empty local
>       `LoadBackendOptionsMap local_map`.
>    b. For each `i` in `[0, backend_options.size())` in order: read
>       `entry = backend_options.entry_at(i)`; deep-copy that entry's options
>       into `local_storage` by emplacing a new inner vector from
>       `[entry.options.begin(), entry.options.end())`; then call
>       `local_map.set_options(entry.backend_id, Span(owned.data(),
>       owned.size()))` where `owned` is the just-added inner vector. If
>       `set_options` returns non-Ok, return that Error (the input map was
>       already valid so this is not expected).
> 3. Commit both members together at the single commit point: move
>    `local_storage` into `backend_options_storage_` and `local_map` into
>    `backend_options_map_`. The move of the outer storage vector is an O(1)
>    buffer transfer that keeps the inner heap buffers (and thus the map's
>    spans) valid.
> 4. Return `Error::Ok`.
>
> Rust note: the deep-copy + span-into-owned-storage dance exists to satisfy
> C++ lifetime/ownership; a Rust port owns the option arrays directly and only
> needs to preserve the transactional "all-or-nothing commit" semantics.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.load-forward-fn]
> inline runtime::Error load_forward( torch::executor::EventTracer* event_tracer)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.load-forward-fn]
> Deprecated convenience overload of `load_forward` taking only an
> `EventTracer*`. Delegates to the non-deprecated
> `load_forward(planned_memory=nullptr, event_tracer, backend_options=nullptr)`,
> which in turn calls `load_method("forward", nullptr, event_tracer, nullptr)`
> per
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`.
> Returns the resulting Error. (The primary non-deprecated `load_forward` is an
> inline forwarder that calls `load_method("forward", planned_memory,
> event_tracer, backend_options, kernel_registry)`.)

> [spec:et:def:module.executorch.extension.et-module-namespace.module.load-internal-fn]
> runtime::Error Module::load_internal(const Program::Verification verification)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.load-internal-fn]
> Loads the program (and any external data maps) if not already loaded.
> Idempotent: does nothing if already loaded.
>
> Steps:
> 1. If `is_loaded()` (program_ non-null) — return `Error::Ok` immediately,
>    doing nothing.
> 2. If `data_loader_` is null: build it by calling `make_data_loader(file_path_,
>    load_mode_)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.make-data-loader-fn]`.
>    On non-Ok, return the Error; otherwise store the loader into
>    `data_loader_`. (If a data_loader/program was provided via constructor,
>    `data_loader_` is already set or the program is already loaded.)
> 3. If `data_files_` is non-empty: for each `data_file` in order, call
>    `make_data_loader(data_file, load_mode_)`; on non-Ok return the Error;
>    otherwise append the loader to `data_map_loaders_`.
> 4. If `data_map_loaders_` is non-empty (either from step 3 or supplied via
>    constructor):
>    a. For each `i` in `[0, data_map_loaders_.size())` in order, call
>       `FlatTensorDataMap::load(data_map_loaders_[i].get())`; on non-Ok return
>       the Error; otherwise move the resulting map onto the heap and append to
>       `named_data_maps_`.
>    b. Build a `vector<const NamedDataMap*>` of the raw pointers of every entry
>       in `named_data_maps_` (in order), then call `MergedDataMap::load(Span(...))`
>       over that; on non-Ok return the Error; otherwise store the merged map
>       into `merged_data_map_`.
> 5. Call `Program::load(data_loader_.get(), verification)`; on non-Ok return
>    the Error; otherwise move the Program onto the heap and store it into the
>    shared_ptr member `program_` (with a plain `delete` deleter).
> 6. Return `Error::Ok`.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.load-method-fn]
> runtime::Error Module::load_method( const std::string& method_name, runtime::HierarchicalAllocator* planned_memory, torch::executor::EventTracer* event_tracer, const LoadBackendOptionsMap* backend_options, std::vector<Kernel> kernel_regi...

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]
> Loads and caches a single method, setting up its memory management. This is
> the primary lazy-init path used by execute/set_input/get_output/etc.
> Idempotent: does nothing if the method is already loaded.
>
> Parameters: `method_name`; optional `planned_memory` (HierarchicalAllocator*,
> default null); optional per-method `event_tracer` (default null); optional
> `backend_options` (LoadBackendOptionsMap*, default null); `kernel_registry`
> (vector<Kernel>, default empty, moved-from).
>
> Steps:
> 1. If `is_method_loaded(method_name)` (method already in `methods_`) — return
>    `Error::Ok`, doing nothing.
> 2. Call `load()` (default verification) to ensure the program is loaded per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-internal-fn]`;
>    on non-Ok return the Error.
> 3. Determine effective backend options: use the passed `backend_options`
>    pointer if non-null; else use `&backend_options_map_` if that stored map
>    has `size() > 0`; else null. (An empty stored map behaves identically to
>    null downstream.)
> 4. Create a local `MethodHolder`.
> 5. If `planned_memory` is null, allocate planned memory into
>    `method_holder.planned_memory` and point `planned_memory` at
>    `method_holder.planned_memory->planned_memory.get()`:
>    a. Fetch `meta = program_->method_meta(method_name.c_str())`; on non-Ok
>       return the Error. Scan its memory-planned buffers: set
>       `has_device_buffers` true if any `meta.memory_planned_buffer_device(i)`
>       is ok and reports non-CPU.
>    b. If `has_device_buffers`: require `!share_memory_arenas_`, else fail with
>       `Error::NotSupported` (device buffers not compatible with shared
>       arenas). Then build device-aware planned memory via
>       `make_planned_memory_with_devices(meta)` per
>       `[spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-devices-fn]`.
>    c. Else if `!share_memory_arenas_`: get per-buffer sizes via
>       `get_mem_planned_buffer_sizes(method_name)` (on non-Ok return Error) and
>       build via `make_planned_memory(sizes)` per
>       `[spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-fn]`.
>    d. Else (shared arenas enabled): get this method's sizes via
>       `get_mem_planned_buffer_sizes(method_name)` (on non-Ok return Error). If
>       `shared_arenas_` is empty, lazily populate it: call
>       `get_max_mem_planned_buffer_sizes()` (on non-Ok return Error), then for
>       the first `min(2, max_sizes.size())` buffer indices (i.e. only mem_id=1
>       and mem_id=2), append a `max_sizes[i]`-byte arena to `shared_arenas_`.
>       Build via `make_planned_memory_with_shared_arenas(sizes, shared_arenas_)`
>       per
>       `[spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-shared-arenas-fn]`,
>       which shares arenas for indices < shared_arenas_.size() and allocates
>       fresh buffers for the rest.
>    (If `planned_memory` was passed non-null, none of the above runs and the
>    caller-provided allocator is used directly.)
> 6. Build `method_holder.memory_manager` as a `MemoryManager(memory_allocator_,
>    planned_memory, temp_allocator_)`. Move `kernel_registry` into
>    `method_holder.kernel_registry`.
> 7. Call `program_->load_method(method_name.c_str(), memory_manager,
>    event_tracer ? event_tracer : this->event_tracer(), merged_data_map_.get(),
>    effective_backend_options, Span<const Kernel>(kernel_registry...))`. That
>    is: the per-method event tracer takes precedence over the Module's tracer;
>    the merged external data map (may be null) and effective backend options
>    (may be null) are forwarded. On non-Ok return the Error.
> 8. Move the loaded Method onto the heap into `method_holder.method`, then
>    insert `(method_name -> method_holder)` into `methods_`.
> 9. Return `Error::Ok`.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.load-mode]
> enum class LoadMode {
>   File;
>   Mmap;
>   MmapUseMlock;
>   MmapUseMlockIgnoreErrors;
>   MmapUseMadvise;
> }

> [spec:et:def:module.executorch.extension.et-module-namespace.module.make-planned-memory-fn]
> std::unique_ptr<Module::PlannedMemory> Module::make_planned_memory( const std::vector<size_t>& buffer_sizes)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-fn]
> Private helper. Allocates fresh CPU memory-planned buffers of the given
> sizes and wraps them in a HierarchicalAllocator.
>
> Steps:
> 1. Create a `PlannedMemory`. Reserve `planned_buffers` and `planned_spans`
>    to `buffer_sizes.size()`.
> 2. For each `size` in `buffer_sizes` in order: append a new
>    `vector<uint8_t>` of that `size` to `planned_buffers`, then append a
>    `Span<uint8_t>(that buffer's data(), size)` to `planned_spans`. The span
>    references the just-added buffer's storage (reserve above prevents
>    reallocation that would invalidate earlier spans).
> 3. Set `planned_memory` to a `HierarchicalAllocator` constructed over
>    `Span(planned_spans.data(), planned_spans.size())`.
> 4. Return the owning `PlannedMemory`. No device buffers are used
>    (`device_buffers`/`planned_devices` stay empty).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-devices-fn]
> std::unique_ptr<Module::PlannedMemory> Module::make_planned_memory_with_devices( const ET_RUNTIME_NAMESPACE::MethodMeta& method_meta)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-devices-fn]
> Private helper. Builds memory-planned buffers where some buffers may live on
> non-CPU devices, based on per-buffer device metadata from the MethodMeta.
>
> Steps:
> 1. Create a `PlannedMemory`. Let `num_buffers =
>    method_meta.num_memory_planned_buffers()`; reserve `planned_buffers`,
>    `planned_spans`, `device_buffers`, and `planned_devices` to that count.
> 2. For each `i` in `[0, num_buffers)` in order:
>    a. `size = method_meta.memory_planned_buffer_size(i)`; if not ok, this is a
>       hard failure — `ET_CHECK_MSG` aborts the process ("Failed to get buffer
>       size for index i"). (Not a recoverable Error; it is a fatal assertion.)
>    b. `device = method_meta.memory_planned_buffer_device(i)`; if not ok,
>       fatal `ET_CHECK_MSG` abort. Append `device.get()` to `planned_devices`.
>    c. If `device->is_cpu()`: append a fresh `size`-byte `vector<uint8_t>` to
>       `planned_buffers` and a `Span(that buffer's data(), size)` to
>       `planned_spans`.
>    d. Else (device buffer): append an empty CPU placeholder vector to
>       `planned_buffers`; create device memory via
>       `DeviceMemoryBuffer::create(size, device->type(), device->index())` —
>       if not ok, fatal `ET_CHECK_MSG` abort. Append that buffer's `as_span()`
>       to `planned_spans` and move the `DeviceMemoryBuffer` into
>       `device_buffers`.
> 3. Set `planned_memory` to a `HierarchicalAllocator` constructed over BOTH
>    `Span(planned_spans...)` and `Span<const Device>(planned_devices...)`, so
>    the allocator carries per-buffer device metadata (exposed later via the
>    MemoryManager's planned_buffer_devices()). `planned_devices` must outlive
>    the allocator.
> 4. Return the owning `PlannedMemory`.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-shared-arenas-fn]
> std::unique_ptr<Module::PlannedMemory>

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.make-planned-memory-with-shared-arenas-fn]
> Private helper. Builds memory-planned buffers where the first buffers reuse
> caller-provided shared arenas and the rest are freshly allocated.
>
> Steps:
> 1. Create a `PlannedMemory`. Reserve `planned_buffers` and `planned_spans` to
>    `buffer_sizes.size()`.
> 2. For each `i` in `[0, buffer_sizes.size())` in order:
>    - If `i < shared_arenas.size()`: append an EMPTY `vector<uint8_t>` to
>      `planned_buffers` (the arena is not owned here), and append a
>      `Span(shared_arenas[i].data(), shared_arenas[i].size())` to
>      `planned_spans` — i.e. this buffer aliases the externally-owned shared
>      arena (note the span uses the arena's actual size, not
>      `buffer_sizes[i]`).
>    - Else: append a fresh `buffer_sizes[i]`-byte `vector<uint8_t>` to
>      `planned_buffers` and a `Span(that buffer's data(), buffer_sizes[i])` to
>      `planned_spans`.
> 3. Set `planned_memory` to a `HierarchicalAllocator` over
>    `Span(planned_spans...)`.
> 4. Return the owning `PlannedMemory`. The shared arenas remain owned by the
>    caller (the Module's `shared_arenas_`) and must outlive this
>    PlannedMemory.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.method-fn]
> runtime::Result<Method*> method( const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.method-fn]
> Deprecated accessor that returns a raw pointer to a loaded `Method` by name.
> This rule is the header declaration; the definition is specified in
> `[spec:et:sem:module.executorch.extension.et-module-namespace.runtime.result-method-module.method-fn]`.
>
> Behavior: call `load_method(method_name)` per
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`;
> on non-Ok Error return it. Otherwise return
> `methods_[method_name].method.get()` — a non-owning `Method*` borrowed from
> the Module's cache (the Module retains ownership).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.method-holder]
> struct MethodHolder {
>   std::unique_ptr<PlannedMemory> planned_memory;
>   std::unique_ptr<runtime::MemoryManager> memory_manager;
>   std::unique_ptr<Method> method;
>   std::vector<Kernel> kernel_registry;
> }

> [spec:et:def:module.executorch.extension.et-module-namespace.module.method-meta-fn]
> runtime::Result<MethodMeta> Module::method_meta( const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.method-meta-fn]
> Returns the `MethodMeta` metadata for a method, loading the program if
> needed.
>
> Steps:
> 1. Call `load()` (default verification) per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-internal-fn]`;
>    on non-Ok Error return it.
> 2. Return `program_->method_meta(method_name.c_str())` (a `Result<MethodMeta>`,
>    which itself errors if `method_name` is not a valid method). Does not load
>    or cache the method itself.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.method-names-fn]
> runtime::Result<std::unordered_set<std::string>> Module::method_names()

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.method-names-fn]
> Returns the set of method names in the loaded program.
>
> Steps:
> 1. Call `load()` (default verification); on non-Ok Error return it.
> 2. Let `method_count = program_->num_methods()`.
> 3. Create an empty `unordered_set<string> result` reserved to `method_count`.
> 4. For each `index` in `[0, method_count)` in order, call
>    `program_->get_method_name(index)` and insert its `.get()` string value
>    into `result`. (These name lookups are assumed to succeed; `.get()`
>    unwraps the Result directly.)
> 5. Return `result` (set — no ordering guarantee, duplicates impossible).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.module-fn]
> Module::Module( const std::string& file_path, const std::string& data_map_path, const LoadMode load_mode, std::unique_ptr<runtime::EventTracer> event_tracer, std::unique_ptr<runtime::MemoryAllocator> memory_allocator, std::unique_ptr<run...

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.module-fn]
> Constructor. The annotated overload is
> `Module(file_path, data_map_path, load_mode, event_tracer, memory_allocator,
> temp_allocator, share_memory_arenas)`; all five Module constructors share the
> same initialization discipline described here.
>
> Common initialization:
> - `memory_allocator_`: if the caller passed a non-null `memory_allocator`,
>   take ownership of it; otherwise default-construct a `MallocMemoryAllocator`.
> - `temp_allocator_`: same rule with the passed `temp_allocator`.
> - `event_tracer_`: move-in the passed tracer (may be null).
> - `share_memory_arenas_`: set from the argument.
> - Program is NOT loaded here; loading is deferred to `load()`/`load_method()`.
> - Finally call `runtime::runtime_init()` (idempotent global runtime init).
>
> Per-overload specifics:
> - This overload additionally sets `file_path_ = file_path`,
>   `load_mode_ = load_mode`, and if `data_map_path` is non-empty pushes it
>   onto `data_files_` (empty path is ignored).
> - The `(file_path, load_mode, ...)` overload: sets `file_path_`, `load_mode_`;
>   no data files.
> - The `(file_path, vector<string> data_files, load_mode, ...)` overload: sets
>   `file_path_`, `load_mode_`, and moves the whole `data_files` vector into
>   `data_files_`.
> - The `(unique_ptr<DataLoader> data_loader, ...)` overload: moves the loader
>   into `data_loader_`; if a `data_map_loader` is provided (non-null) pushes it
>   onto `data_map_loaders_`. Argument order for this overload is
>   (data_loader, memory_allocator, temp_allocator, event_tracer,
>   data_map_loader, share_memory_arenas).
> - The `(shared_ptr<Program> program, ...)` overload: moves the program into
>   `program_` (so `is_loaded()` is immediately true); if a `data_map_loader` is
>   provided (non-null) pushes it onto `data_map_loaders_`. Argument order:
>   (program, memory_allocator, temp_allocator, event_tracer, data_map_loader,
>   share_memory_arenas).
>
> The Module is non-copyable and non-movable.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.num-methods-fn]
> runtime::Result<size_t> Module::num_methods()

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.num-methods-fn]
> Returns the number of methods in the loaded program.
>
> Steps:
> 1. Call `load()` (default verification) to ensure the program is loaded; on
>    non-Ok Error return it.
> 2. Return `program_->num_methods()` (a `size_t`, wrapped in `Result`).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.planned-memory]
> struct PlannedMemory {
>   std::vector<std::vector<uint8_t>> planned_buffers;
>   std::vector<runtime::Span<uint8_t>> planned_spans;
>   std::vector<runtime::DeviceMemoryBuffer> device_buffers;
>   std::vector<runtime::etensor::Device> planned_devices;
>   std::unique_ptr<runtime::HierarchicalAllocator> planned_memory;
> }

> [spec:et:def:module.executorch.extension.et-module-namespace.module.program-fn]
> inline std::shared_ptr<Program> program() const

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.program-fn]
> Const accessor. Returns a copy of the `program_` shared pointer (shared
> ownership of the loaded Program), or a null shared_ptr if the program has not
> been loaded yet. Performs no loading and no side effects. The Program's data
> loader is guaranteed valid for the Program's lifetime.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.set-input-fn]
> runtime::Error Module::set_input( const std::string& method_name, const runtime::EValue& input_value, size_t input_index)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.set-input-fn]
> Sets a single input value for a method at a given input index.
>
> Steps:
> 1. Call `load_method(method_name)`; on non-Ok Error return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`.
> 3. Return `method->set_input(input_value, input_index)` (the Method validates
>    the index and value type and propagates any Error).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.set-inputs-fn]
> runtime::Error Module::set_inputs( const std::string& method_name, const std::vector<runtime::EValue>& input_values)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.set-inputs-fn]
> Sets all input values for a method at once.
>
> Steps:
> 1. Call `load_method(method_name)`; on non-Ok Error return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`.
> 3. Return `method->set_inputs(ArrayRef<EValue>(input_values.data(),
>    input_values.size()))` — passes the whole input vector as an ArrayRef. The
>    Method validates count/types and propagates any Error.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.set-output-fn]
> runtime::Error Module::set_output( const std::string& method_name, runtime::EValue output_value, size_t output_index)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.set-output-fn]
> Sets a method's output buffer to point at a caller-provided tensor's data.
> Only Tensor outputs are supported.
>
> Steps:
> 1. Call `load_method(method_name)`; on non-Ok Error return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`.
> 3. Validate `output_value.isTensor()`; if not, fail with
>    `Error::InvalidArgument` (log "output type: <tag> is not tensor" using the
>    numeric EValue tag) and return that Error.
> 4. Take `output_tensor = output_value.toTensor()`.
> 5. Return `method->set_output_data_ptr(output_tensor.mutable_data_ptr(),
>    output_tensor.nbytes(), output_index)` — i.e. bind the method's output slot
>    at `output_index` to the tensor's mutable data pointer and byte size. The
>    Method propagates any Error (e.g. for memory-planned or constant outputs).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.set-outputs-fn]
> runtime::Error Module::set_outputs( const std::string& method_name, const std::vector<runtime::EValue>& output_values)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.set-outputs-fn]
> Sets all output buffers for a method from a vector of tensor EValues.
>
> Steps:
> 1. Call `load_method(method_name)`; on non-Ok Error return it.
> 2. Look up the cached `Method` via `methods_.at(method_name)`; read
>    `outputs_size = method->outputs_size()`.
> 3. Validate `output_values.size() == outputs_size`; if not, fail with
>    `Error::InvalidArgument` (log "output size: <given> is not equal to method
>    output size: <outputs_size>") and return that Error.
> 4. For each `index` in `[0, outputs_size)` in order, call
>    `set_output(method_name, output_values[index], index)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.set-output-fn]`;
>    on the first non-Ok Error, return it (partial application may have already
>    bound earlier outputs).
> 5. Return `Error::Ok`. Only Tensor outputs are supported; fails for
>    memory-planned or constant outputs.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.unload-forward-fn]
> inline bool unload_forward()

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.unload-forward-fn]
> Unloads the "forward" method. Delegates to `unload_method("forward")` per
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.unload-method-fn]`.
> Returns true if the "forward" method was cached and is now removed, false if
> it was not loaded (no-op).

> [spec:et:def:module.executorch.extension.et-module-namespace.module.unload-method-fn]
> inline bool unload_method(const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.unload-method-fn]
> Unloads (drops from the cache) the method named `method_name`. Returns
> `methods_.erase(method_name)` interpreted as bool: true if an entry existed
> and was removed (erase count 1), false if no such method was loaded (count 0,
> no-op). Erasing destroys the cached Method and its associated planned memory /
> memory manager.

> [spec:et:def:module.executorch.extension.et-module-namespace.runtime.result-method-module.method-fn]
> ET_NODISCARD runtime::Result<Method*> Module::method( const std::string& method_name)

> [spec:et:sem:module.executorch.extension.et-module-namespace.runtime.result-method-module.method-fn]
> Definition of the deprecated `method(method_name)` accessor (declared in
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.method-fn]`).
>
> Steps:
> 1. Call `load_method(method_name)` per
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`;
>    on non-Ok Error, return that Error (via `Result<Method*>`).
> 2. Return `methods_[method_name].method.get()` — a non-owning `Method*` into
>    the Module's cache. Uses `operator[]`, so it assumes the entry exists
>    (load_method guarantees it on success). Ownership stays with the Module.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.backend-options-fn]
> inline const LoadBackendOptionsMap& backend_options() const

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.backend-options-fn]
> Const accessor. Returns a const reference to the Module-owned
> `backend_options_map_` — the deep copy most recently installed via
> `load(LoadBackendOptionsMap, ...)` per
> `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-fn]`.
> If that overload was never called, returns a default-constructed empty map
> (`size() == 0`). The reference stays valid until the next
> `load(LoadBackendOptionsMap, ...)` or Module destruction. No side effects.

> [spec:et:def:module.executorch.extension.et-module-namespace.module.operator-fn]
> Module& operator=(const Module&) = delete

> [spec:et:sem:module.executorch.extension.et-module-namespace.module.operator-fn]
> The copy-assignment operator is explicitly deleted (`= delete`). Module is
> non-copyable (and, per the surrounding declarations, non-copy-constructible
> and non-movable). Any attempt to copy-assign a Module is a compile-time
> error. A Rust port models this as a non-Clone type with unique ownership.

