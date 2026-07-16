# extension/module/bundled_module.cpp, extension/module/bundled_module.h

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module]
> class BundledModule : public Module {
>   ET_NODISCARD static runtime::Result<std::unique_ptr<BundledModule>> from_file( const std::string& file_path, std::unique_ptr<runtime::MemoryAllocator> memory...;
>   ET_NODISCARD runtime::Result<std::vector<runtime::EValue>> execute( const std::string& method_name, const size_t testset_idx);
>   ET_NODISCARD runtime::Error verify_method_outputs( const std::string& method_name, const size_t testset_idx, double rtol = 1e-5, double atol = 1e-8);
>   const void* bundled_program_ptr_;
>   bool is_loaded_from_file_ = false;
> }

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.bundled-module-fn]
> BundledModule::BundledModule( const void* bundled_program_ptr, std::unique_ptr<runtime::MemoryAllocator> memory_allocator, std::unique_ptr<runtime::MemoryAllocator> temp_allocator, std::unique_ptr<runtime::EventTracer> event_tracer, std:...

> [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.bundled-module-fn]
> Constructs a `BundledModule` from a raw pointer to the in-memory serialized
> bundled program buffer (`bundled_program_ptr`), plus the optional allocators,
> event tracer, and external-data-map loader that a plain `Module` accepts (all
> defaulting to null / not provided).
>
> The base `Module` subobject is constructed with a data loader obtained by
> calling the free function `program_data_loader(bundled_program_ptr)` (see
> `[spec:et:sem:bundled-module.executorch.extension.program-data-loader-fn]`),
> which parses the bundled-program flatbuffer and returns a `BufferDataLoader`
> viewing the embedded plain-program bytes. The `memory_allocator`,
> `temp_allocator`, `event_tracer`, and `data_map_loader` arguments are moved
> (ownership transferred) into the base `Module` constructor unchanged.
>
> After the base is constructed, the member `bundled_program_ptr_` is set to the
> same `bundled_program_ptr` value, preserving access to the parts of the bundle
> outside the plain program (namely the test-set inputs and expected outputs used
> by `execute` and `verify_method_outputs`). The member `is_loaded_from_file_`
> retains its default value of `false`; it is only set to `true` by `from_file`
> (see
> `[spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn]`).
> The constructor does not take ownership of the buffer pointed to by
> `bundled_program_ptr`: the caller must keep it alive for the lifetime of the
> module. On destruction, the buffer is deleted (as `delete[]` over `const
> uint8_t*`) only when `is_loaded_from_file_` is `true`.

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn]
> runtime::Result<std::vector<runtime::EValue>> BundledModule::execute( const std::string& method_name, const size_t testset_idx)

> [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn]
> Runs the method named `method_name`, feeding it the bundled test-set inputs at
> index `testset_idx`, and returns the resulting output `EValue`s. Returns a
> `Result<std::vector<EValue>>`.
>
> Steps, in order; any step returning a non-`Ok` `Error` aborts and that `Error`
> is returned immediately (via the `ET_CHECK_OK_OR_RETURN_ERROR` pattern):
>
> 1. Ensure the method is loaded by calling `load_method(method_name)` on the
>    base `Module` (see
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`).
>    This is idempotent: if the program and this method are already loaded it is a
>    no-op returning `Ok`; otherwise it loads the program and this method, caching
>    it. Propagates any load error.
> 2. Obtain a reference to the loaded method object: `methods_.at(method_name).method`.
>    After step 1 succeeded, the entry is guaranteed to exist in the cache.
> 3. Load the bundled input for this test set by calling the bundled-program
>    helper `load_bundled_input(*method, bundled_program_ptr_, testset_idx)`,
>    which copies/sets the `testset_idx`-th serialized input values from the
>    bundle into the method's input slots. Propagates any error (e.g. an
>    out-of-range `testset_idx` or a shape/dtype mismatch surfaces here).
> 4. Execute the method by calling `method->execute()`. Propagates any execution
>    error.
> 5. Query the number of outputs via `method->outputs_size()` and allocate a
>    `std::vector<EValue>` of exactly that length (default-constructed elements).
> 6. Copy the method's outputs into that vector by calling
>    `method->get_outputs(outputs.data(), outputs_size)` (see
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.get-outputs-fn]`
>    for the analogous copy semantics). Propagates any error.
>
> On success returns the populated `outputs` vector by value. The returned
> `EValue`s alias tensor data owned by the method (they are not deep copies);
> they remain valid only while the method's output memory is valid.

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn]
> runtime::Result<std::unique_ptr<BundledModule>> BundledModule::from_file( const std::string& file_path, std::unique_ptr<runtime::MemoryAllocator> memory_allocator, std::unique_ptr<runtime::MemoryAllocator> temp_allocator, std::unique_ptr...

> [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.from-file-fn]
> Static factory that loads a bundled program from the filesystem path
> `file_path`, copies the entire file into a heap buffer owned by the returned
> module, and constructs a `BundledModule` over it. Returns
> `Result<std::unique_ptr<BundledModule>>`. The optional `memory_allocator`,
> `temp_allocator`, `event_tracer`, and `data_map_loader` (all defaulting to
> null) are forwarded to the constructor.
>
> Steps, in order; each `Result`/`Error` check returns the error immediately on
> failure:
>
> 1. Create a file data loader via `FileDataLoader::from(file_path.c_str())` (see
>    `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.from-fn]`).
>    If the result is not `ok()`, return its `error()` (e.g. file not found /
>    open failure).
> 2. Query the file size via `loader->size()` (see
>    `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.size-fn]`).
>    If not `ok()`, return its `error()`. Extract the `size_t file_size`.
> 3. Allocate a heap byte buffer of exactly `file_size` bytes
>    (`std::make_unique<uint8_t[]>(file_size)`).
> 4. Read the whole file into that buffer via
>    `loader->load_into(0, file_size, {}, file_data.get())` — offset 0, length
>    `file_size`, default (empty) segment info, destination = the buffer (see
>    `[spec:et:sem:file-data-loader.executorch.extension.file-data-loader.load-into-fn]`).
>    If the returned `Error` is not `Ok`, return it.
> 5. Construct a `BundledModule` via `std::make_unique<BundledModule>(...)`,
>    passing `file_data.release()` as the `bundled_program_ptr` (transferring raw
>    ownership of the buffer to the module) together with the moved
>    `memory_allocator`, `temp_allocator`, `event_tracer`, and `data_map_loader`.
> 6. Set the module's `is_loaded_from_file_` flag to `true` so that the
>    destructor will `delete[]` the buffer.
>
> Return the `unique_ptr<BundledModule>`. The file loader created in step 1 is
> destroyed when this function returns (only its buffered copy of the bytes is
> retained). Because the module now owns the byte buffer, callers do not manage
> its lifetime — unlike the raw-pointer constructor.

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn]
> runtime::Error BundledModule::verify_method_outputs( const std::string& method_name, const size_t testset_idx, double rtol, double atol)

> [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.verify-method-outputs-fn]
> Compares the outputs currently produced by the method `method_name` against the
> bundle's expected outputs for test set `testset_idx`, within the given
> tolerances. Returns a bare `runtime::Error` (`Ok` if all outputs match).
> `rtol` defaults to `1e-5` and `atol` defaults to `1e-8` (defaults declared in
> the header; the definition uses whatever the caller passed).
>
> Steps:
>
> 1. Ensure the method is loaded by calling `load_method(method_name)` (see
>    `[spec:et:sem:module.executorch.extension.et-module-namespace.module.load-method-fn]`);
>    if it returns a non-`Ok` `Error`, return that error immediately.
> 2. Obtain the loaded method reference `methods_.at(method_name).method`
>    (guaranteed present after step 1).
> 3. Delegate to the bundled-program helper
>    `verify_method_outputs(*method, bundled_program_ptr_, testset_idx, rtol,
>    atol)` and return its `Error` directly. That helper reads the method's
>    current output tensors and, for each, compares elementwise against the
>    bundled expected output using tolerance test `|actual - expected| <= atol +
>    rtol * |expected|`, returning `Ok` on match or an error (e.g.
>    `Error::InvalidArgument` for shape/dtype/count mismatch, or a mismatch
>    error) otherwise.
>
> This function does not itself execute the method or load bundled inputs; it
> assumes the method's outputs already hold the values to be verified (typically
> set by a prior
> `[spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.execute-fn]`
> call for the same `testset_idx`).

> [spec:et:def:bundled-module.executorch.extension.program-data-loader-fn]
> std::unique_ptr<BufferDataLoader> program_data_loader( const void* bundled_program_ptr)

> [spec:et:sem:bundled-module.executorch.extension.program-data-loader-fn]
> File-local (anonymous-namespace) helper that, given a raw pointer to a
> serialized bundled-program flatbuffer (`bundled_program_ptr`), returns a
> `BufferDataLoader` viewing the plain ExecuTorch program bytes embedded inside
> that bundle.
>
> Steps:
>
> 1. Interpret the buffer as a bundled-program flatbuffer root by calling
>    `bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr)`. This is
>    a zero-copy cast to the generated flatbuffer accessor; no validation is
>    performed here (the caller is trusted to pass a valid bundle buffer).
> 2. Access the embedded plain program via `bundled_program->program()`, which is
>    a flatbuffer vector-of-bytes field holding the serialized ExecuTorch program.
> 3. Construct and return `std::make_unique<BufferDataLoader>(program->data(),
>    program->size())` (see
>    `[spec:et:sem:buffer-data-loader.executorch.extension.buffer-data-loader.buffer-data-loader-fn]`),
>    i.e. a data loader that borrows — does not copy — the program bytes at
>    `program->data()` with length `program->size()`.
>
> The returned loader aliases memory owned by the bundled-program buffer; that
> buffer must outlive the loader (and hence the `Module` built from it).

> [spec:et:def:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.operator-fn]
> BundledModule& operator=(const BundledModule&) = delete

> [spec:et:sem:bundled-module.executorch.extension.et-bundled-module-namespace.bundled-module.operator-fn]
> The copy-assignment operator `operator=(const BundledModule&)` is explicitly
> deleted, making `BundledModule` non-copy-assignable. Together with the deleted
> copy constructor and deleted move constructor / move-assignment, instances are
> neither copyable nor movable. This is required because a `BundledModule` may
> exclusively own a heap-allocated program buffer (when
> `is_loaded_from_file_` is `true`, freed in the destructor) and holds a data
> loader aliasing that buffer; copying or moving would risk double-free or
> dangling aliases. In a Rust port there is no direct equivalent to synthesize —
> the type should simply not implement `Copy`/`Clone` and ownership should be
> moved via the owning smart pointer (`Box`/`unique_ptr` analog) rather than the
> value itself.

