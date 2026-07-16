# runtime/executor/program.cpp, runtime/executor/program.h

> [spec:et:def:program.executorch.et-runtime-namespace.get-execution-plan-fn]
> Result<executorch_flatbuffer::ExecutionPlan*> get_execution_plan( const executorch_flatbuffer::Program* program, const char* method_name)

> [spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn]
> Free function (file-local, anonymous namespace) that finds the flatbuffer
> `ExecutionPlan` whose name matches `method_name`.
>
> Steps:
> 1. Read `program->execution_plan()`, a flatbuffer vector of `ExecutionPlan`.
> 2. Iterate `i` from 0 to `execution_plans->size() - 1` in order. For each,
>    obtain the mutable plan object at index `i`.
> 3. Match a plan when all of: the plan pointer is non-null, `plan->name()` is
>    non-null, and `strcmp(plan->name()->c_str(), method_name) == 0` (exact,
>    NUL-terminated C-string comparison). Return the first matching plan pointer
>    (a `Result` wrapping the raw `ExecutionPlan*`).
> 4. If no plan matches after the full scan, log an Error ("No method named
>    '<method_name>' in program") and return `Error::InvalidArgument`.
>
> The vector is assumed non-null (callers reach this only after a valid program
> was loaded). Returns a mutable pointer into the flatbuffer buffer owned by the
> Program; lifetime is tied to the Program's `program_data_`.

> [spec:et:def:program.executorch.et-runtime-namespace.is-aligned-fn]
> bool IsAligned(const void* data)

> [spec:et:sem:program.executorch.et-runtime-namespace.is-aligned-fn]
> Free function (file-local, anonymous namespace). Returns true iff the pointer
> `data` is aligned to `kMinimumAlignment`, where `kMinimumAlignment ==
> alignof(std::max_align_t)` (the alignment guaranteed by `malloc`/`new`; on
> typical targets 16, but must be treated as a platform constant that is a power
> of 2).
>
> Steps:
> 1. Reinterpret the pointer value as an unsigned integer address (`uintptr_t`).
> 2. Return `(addr % kMinimumAlignment) == 0`.
>
> No dereference of `data` occurs; a null pointer yields true (address 0 is
> divisible by any alignment).

> [spec:et:def:program.executorch.et-runtime-namespace.program]
> class Program final {
>   enum class Verification : uint8_t { /** * Do minimal verification of the data, ensuring that the header appears * correct. * * Has minimal runtime overhead. ...;
>   ET_NODISCARD static Result<Program>;
>   ET_DEPRECATED ET_NODISCARD;
>   ET_DEPRECATED Result<const char*>;
>   enum HeaderStatus { /** * An ExecuTorch program header is present, and its version is compatible * with this version of the runtime. */ CompatibleVersion, /*...;
>   static constexpr size_t kMinHeadBytes = 64;
>   ET_NODISCARD Result<FreeableBuffer>;
>   FreeableBuffer program_data_;
>   DataLoader* loader_;
>   const executorch_flatbuffer::Program* internal_program_;
>   size_t segment_base_offset_;
>   FreeableBuffer constant_segment_data_;
>   std::optional<internal::PteDataMap> pte_data_map_;
> }

> [spec:et:def:program.executorch.et-runtime-namespace.program.check-header-fn]
> Program::HeaderStatus Program::check_header( const void* data, size_t size)

> [spec:et:sem:program.executorch.et-runtime-namespace.program.check-header-fn]
> Static method. Classifies the leading bytes of a candidate program file into a
> `HeaderStatus` (`[spec:et:def:program.executorch.et-runtime-namespace.program.header-status]`).
> Does not require an aligned buffer and does not parse the flatbuffer.
>
> Steps:
> 1. If `size < kMinHeadBytes` (kMinHeadBytes == 64), return
>    `HeaderStatus::ShortData` immediately.
> 2. If the flatbuffer file identifier at `data` matches the schema's expected
>    Program identifier (a 4-byte tag located at bytes 4..7 of the flatbuffer,
>    i.e. `ProgramBufferHasIdentifier(data)` is true), return
>    `HeaderStatus::CompatibleVersion`.
> 3. Otherwise read the 4-byte buffer identifier (bytes 4..7). If its first two
>    bytes are ASCII 'E' and 'T', it looks like an ExecuTorch file of a
>    different version: return `HeaderStatus::IncompatibleVersion`.
> 4. Otherwise return `HeaderStatus::NotPresent`.
>
> The caller must guarantee `data` points to at least `size` readable bytes; the
> function reads at most the first 8 bytes (offset + file identifier).

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-backend-delegate-data-fn]
> Error Program::get_backend_delegate_data( size_t index, const void** out_data, size_t* out_size) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-backend-delegate-data-fn]
> Private method (friend-accessible, used by `Method`) that returns a pointer to
> and size of the inline blob for the backend delegate at `index` in the
> program's `backend_delegate_data` table.
>
> Steps:
> 1. Obtain `data_list = internal_program_->backend_delegate_data()` (assumed
>    non-null for a valid program).
> 2. ET_CHECK_OR_RETURN_ERROR: if `index >= data_list->size()`, log and return
>    `Error::NotFound` without writing `out_data`/`out_size`.
> 3. Fetch `data = data_list->Get(index)->data()`, the flatbuffer byte vector
>    for that delegate blob.
> 4. Write `*out_data = data->data()` (pointer into the program flatbuffer
>    buffer) and `*out_size = data->size()`.
> 5. Return `Error::Ok`.
>
> On error only the return value signals failure; the out-params are left
> unmodified.

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-constant-buffer-data-fn]
> Result<const void*> Program::get_constant_buffer_data( size_t buffer_index, size_t nbytes) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-constant-buffer-data-fn]
> Returns a read-only pointer to the constant tensor data at `buffer_index`,
> validating that at least `nbytes` are readable there. Constant data lives
> either in a separate segment (loaded into `constant_segment_data_` during
> `load`) or, on the deprecated path, inline in the flatbuffer's
> `constant_buffer`. The two are mutually exclusive; the branch is selected by
> whether `constant_segment_data_.data()` is non-null.
>
> Segment path (constant_segment_data_ present):
> 1. Let `constant_segment = internal_program_->constant_segment()`. Compute
>    `num_elems` = size of `constant_segment->offsets()` if both the segment and
>    its offsets vector are non-null, else 0. The offsets vector always contains
>    a leading placeholder entry (index 0) for non-const tensors.
> 2. ET_CHECK_OR_RETURN_ERROR: if `buffer_index >= num_elems`, return
>    `Error::InvalidArgument`.
> 3. Read `offset = (uint64_t)(*constant_segment->offsets())[buffer_index]` —
>    the byte offset of this tensor relative to the start of the constant
>    segment.
> 4. Let `size = constant_segment_data_.size()`. ET_CHECK_OR_RETURN_ERROR: if
>    NOT (`offset <= size && nbytes <= size - offset`) return
>    `Error::InvalidArgument` (the `size - offset` form is computed only after
>    `offset <= size` is known, avoiding underflow).
> 5. Return `constant_segment_data_.data() + offset` as `const void*`.
>
> Inline path (constant_segment_data_ null), only when
> ET_ENABLE_DEPRECATED_CONSTANT_BUFFER is enabled:
> 1. Let `constant_buffer_ptr = internal_program_->constant_buffer()`;
>    `num_elems` = its size if non-null else 0.
> 2. ET_CHECK_OR_RETURN_ERROR: if `buffer_index >= num_elems`, return
>    `Error::InvalidArgument`.
> 3. Get `storage = constant_buffer[buffer_index]->storage()`; `storage_size` =
>    its size if non-null else 0.
> 4. ET_CHECK_OR_RETURN_ERROR: if `nbytes > storage_size` return
>    `Error::InvalidArgument` (storage may be larger than nbytes due to tensor
>    alignment padding, so only an upper bound is required).
> 5. Return `storage->data()`.
> When ET_ENABLE_DEPRECATED_CONSTANT_BUFFER is disabled, the inline path instead
> logs and returns `Error::InvalidProgram`.

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-internal-program-fn]
> const executorch_flatbuffer::Program* get_internal_program() const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-internal-program-fn]
> Private inline accessor (friend-accessible). Returns the stored
> `internal_program_` pointer (the raw flatbuffer `Program*` that points into
> the owned `program_data_` buffer). No validation, no copy; pure getter.

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-method-name-fn]
> Result<const char*> Program::get_method_name(size_t plan_index) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-method-name-fn]
> Returns the name of the method (execution plan) at `plan_index`.
>
> Steps:
> 1. If `plan_index >= num_methods()`
>    (`[spec:et:sem:program.executorch.et-runtime-namespace.program.num-methods-fn]`),
>    log an Error and return `Error::InvalidArgument`.
> 2. Otherwise the execution_plan vector is known present. Fetch
>    `name = internal_program_->execution_plan()->Get(plan_index)->name()`.
> 3. If `name` is null, log an Error and return `Error::InvalidProgram`.
> 4. Return `name->c_str()` — a NUL-terminated pointer owned by the Program,
>    valid for the Program's lifetime.

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-named-data-map-fn]
> Result<const NamedDataMap*> Program::get_named_data_map() const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-named-data-map-fn]
> Returns a pointer to the program's `NamedDataMap` for resolving named data
> stored in the PTE.
>
> Steps:
> 1. If `pte_data_map_` (the `std::optional<internal::PteDataMap>`) has a value,
>    return a pointer to the contained `PteDataMap` (upcast to `NamedDataMap*`).
>    The returned pointer is owned by the Program.
> 2. Otherwise (no named_data in the program) return `Error::NotFound`.

> [spec:et:def:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn]
> Result<const char*> Program::get_output_flattening_encoding( const char* method_name) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.get-output-flattening-encoding-fn]
> Deprecated. Returns the pytree encoding string describing how the named
> method's outputs are flattened. `method_name` defaults to "forward".
>
> Steps:
> 1. Resolve the execution plan via
>    `[spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn]`;
>    on failure return that error (e.g. `Error::InvalidArgument` if no such
>    method).
> 2. `container_meta_type = plan->container_meta_type()`. ET_CHECK_OR_RETURN_ERROR:
>    if null, return `Error::InvalidProgram`.
> 3. `encoded_out_str = container_meta_type->encoded_out_str()`.
>    ET_CHECK_OR_RETURN_ERROR: if null, return `Error::InvalidProgram`.
> 4. Return `encoded_out_str->c_str()` (NUL-terminated, owned by the Program).

> [spec:et:def:program.executorch.et-runtime-namespace.program.header-status]
> enum HeaderStatus {
>   CompatibleVersion;
>   IncompatibleVersion;
>   NotPresent;
>   ShortData;
> }

> [spec:et:def:program.executorch.et-runtime-namespace.program.load-fn]
> Result<Program> Program::load( DataLoader* loader, Program::Verification verification)

> [spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn]
> Static factory. Loads and parses a program from `loader`, performing
> verification at the requested `verification` level, and returns a constructed
> `Program`. The Program holds `loader` (which must outlive it). Compile flags:
> ET_ENABLE_PROGRAM_VERIFICATION (default 1) and
> ET_ENABLE_DEPRECATED_CONSTANT_BUFFER (default 1).
>
> Phase A — determine sizes from the extended header:
> 1. Initialise `program_size = 0`, `segment_base_offset = 0`,
>    `segment_data_size = 0`.
> 2. Load the first `ExtendedHeader::kNumHeadBytes` bytes at offset 0 with
>    SegmentInfo type Program. If the load fails, return its error.
> 3. Parse an ExtendedHeader from those bytes.
>    - If parse succeeds: set `program_size = eh.program_size`,
>      `segment_base_offset = eh.segment_base_offset`,
>      `segment_data_size = eh.segment_data_size`. Then, for backward
>      compatibility (segment_data_size was added in ET 1.0), only run the file
>      size sanity check when `(segment_data_size == 0 && segment_base_offset ==
>      0)` OR `segment_data_size > 0`:
>        - ET_CHECK_OR_RETURN_ERROR(`segment_base_offset <= SIZE_MAX -
>          segment_data_size`, else `Error::InvalidProgram`) to guard overflow.
>        - Compute `expected = (segment_base_offset == 0) ? program_size :
>          segment_base_offset + segment_data_size`, and `actual =
>          loader->size().get()`. ET_CHECK_OR_RETURN_ERROR(`expected <= actual`,
>          else `Error::InvalidProgram`).
>    - If parse returns `Error::NotFound` (no extended header): the program
>      consumes the whole file with no segments. Read `loader->size()`; on
>      failure return its error; else set `program_size = size`.
>    - Any other parse error: log "Extended header may be corrupt" and return
>      that error.
>
> Phase B — load and sanity-check the flatbuffer:
> 4. Load `program_size` bytes at offset 0 (SegmentInfo type Program) into
>    `program_data`. On failure return its error.
> 5. ET_CHECK_OR_RETURN_ERROR(`IsAligned(program_data.data())`, else
>    `Error::InvalidArgument`) —
>    `[spec:et:sem:program.executorch.et-runtime-namespace.is-aligned-fn]`.
> 6. Let `kMinBufferSize = ExtendedHeader::kHeaderOffset` (the flatbuffer header
>    size before the extended header, 8 bytes). ET_CHECK_OR_RETURN_ERROR(
>    `program_data.size() >= kMinBufferSize`, else `Error::InvalidProgram`).
> 7. If the flatbuffer Program file identifier is absent/mismatched
>    (`ProgramBufferHasIdentifier(data)` false): log the mismatch and return
>    `Error::InvalidProgram`.
>
> Phase C — verification:
> 8. If `verification == InternalConsistency`:
>    - When ET_ENABLE_PROGRAM_VERIFICATION: run the flatbuffer `Verifier` over
>      the whole buffer (`VerifyProgramBuffer`). ET_CHECK_OR_RETURN_ERROR(ok,
>      else `Error::InvalidProgram`). Then obtain the root Program and call
>      `validate_program`
>      (`[spec:et:sem:program-validation.executorch.runtime.validate-program-fn]`);
>      ET_CHECK_OR_RETURN_ERROR(result == Error::Ok, else
>      `Error::InvalidProgram`).
>    - When verification is compiled out: log Info that it falls back to Minimal.
> 9. If `verification == Minimal` (or, when verification is compiled out, also
>    InternalConsistency): manually bounds-check the root table. Read the root
>    offset as a little-endian `uoffset_t` from the first 4 bytes.
>    ET_CHECK_OR_RETURN_ERROR(`root_offset >= kMinBufferSize && root_offset <=
>    program_data.size() - sizeof(soffset_t)`, else `Error::InvalidProgram`).
>    (In InternalConsistency+verification-enabled mode this check is instead
>    covered by the Verifier.)
> 10. Obtain the root `flatbuffer_program = GetProgram(program_data.data())`.
>
> Phase D — named data map:
> 11. `named_data = flatbuffer_program->named_data()`. If non-null, call
>     `PteDataMap::create(loader, segment_base_offset, named_data,
>     flatbuffer_program->segments())`
>     (`[spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]`);
>     on failure return that error; on success store the created map into the
>     optional `pte_data_map`. If `named_data` is null, `pte_data_map` stays
>     `nullopt`.
>
> Phase E — constant data and construction:
> 12. `constant_segment = flatbuffer_program->constant_segment()`. If
>     `constant_segment != nullptr && constant_segment->offsets() != nullptr &&
>     offsets->size() > 0`, use the segment path:
>     - If `offsets->size() == 1` (only the index-0 placeholder, no real
>       constants): construct and return a Program with an empty
>       `constant_segment_data` FreeableBuffer.
>     - Else (real constants in a separate segment):
>       - `constant_buffer = flatbuffer_program->constant_buffer()`.
>         ET_CHECK_OR_RETURN_ERROR(`constant_buffer == nullptr ||
>         constant_buffer->size() == 0`, else `Error::InvalidProgram`) — inline
>         and segment constants must not both be present.
>       - `segments = flatbuffer_program->segments()`.
>         ET_CHECK_OR_RETURN_ERROR(`segments != nullptr`, else
>         `Error::InvalidProgram`).
>       - ET_CHECK_OR_RETURN_ERROR(`constant_segment->segment_index() <
>         segments->size()`, else `Error::InvalidProgram`).
>       - `data_segment = segments->Get(constant_segment->segment_index())`.
>         Load `data_segment->size()` bytes at offset `segment_base_offset +
>         data_segment->offset()` with SegmentInfo type Constant and the
>         segment index. On failure return its error.
>       - Construct and return a Program with that loaded buffer as
>         `constant_segment_data`.
> 13. Else (no constant segment) use the deprecated inline `constant_buffer`
>     path:
>     - When ET_ENABLE_DEPRECATED_CONSTANT_BUFFER: log a deprecation Error and
>       construct/return a Program with an empty `constant_segment_data`.
>     - Otherwise: log an Error and return `Error::InvalidProgram`.
>
> Every returned Program is built via the private constructor
> (`[spec:et:sem:program.executorch.et-runtime-namespace.program.program-fn]`),
> moving `program_data`, `flatbuffer_program`, the constant buffer, and the
> pte_data_map optional into it, along with `loader` and `segment_base_offset`.

> [spec:et:def:program.executorch.et-runtime-namespace.program.load-method-fn]
> Result<Method> Program::load_method( const char* method_name, MemoryManager* memory_manager, EventTracer* event_tracer, const NamedDataMap* named_data_map, const LoadBackendOptionsMap* backend_options, Span<const Kernel> kernel_registry)...

> [spec:et:sem:program.executorch.et-runtime-namespace.program.load-method-fn]
> Loads the named method and prepares it for execution, returning a `Method`.
> `event_tracer`, `named_data_map`, `backend_options`, and `kernel_registry` are
> optional (default null / empty span).
>
> Steps:
> 1. Emit event-tracer setup: create an event block named "Default" and open a
>    profiling scope "Program::load_method" on `event_tracer` (no-ops if the
>    tracer is null).
> 2. Compute `method_meta(method_name)`
>    (`[spec:et:sem:program.executorch.et-runtime-namespace.program.method-meta-fn]`);
>    if it fails, return its error. (Method::method_meta() later assumes this
>    succeeds, so failure must be caught here.)
> 3. Resolve the execution plan via
>    `[spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn]`;
>    on failure return its error.
> 4. Delegate to `Method::load(plan, this, memory_manager, event_tracer,
>    named_data_map, backend_options, kernel_registry)` and return its result
>    (the constructed Method or an error).

> [spec:et:def:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn]
> Error Program::load_mutable_subsegment_into( size_t mutable_data_segments_index, size_t offset_index, size_t size, void* buffer) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.load-mutable-subsegment-into-fn]
> Private method (friend-accessible). Loads `size` bytes from a mutable data
> subsegment directly into a caller-provided `buffer` (which must have at least
> `size` bytes). Addressed by a two-level index: `mutable_data_segments_index`
> selects a mutable-data-segment offsets group, `offset_index` selects an offset
> within it.
>
> Steps:
> 1. If `loader_ == nullptr || segment_base_offset_ == 0` (no segments in
>    program), log and return `Error::NotFound`.
> 2. If `internal_program_->mutable_data_segments() == nullptr`, log and return
>    `Error::NotFound`.
> 3. If `mutable_data_segments_index >= mutable_data_segments()->size()`, log
>    and return `Error::NotFound`.
> 4. `segment_offsets = mutable_data_segments()->Get(mutable_data_segments_index)`.
>    If `segment_offsets->offsets() == nullptr`, log and return
>    `Error::NotFound`.
> 5. If `offset_index >= segment_offsets->offsets()->size()`, log and return
>    `Error::NotFound`.
> 6. `offset = segment_offsets->offsets()->Get(offset_index)` — byte offset
>    relative to the start of the referenced segment.
> 7. ET_CHECK_OR_RETURN_ERROR(`internal_program_->segments() != nullptr`, else
>    `Error::InvalidProgram`). Let `num_segments = segments()->size()`.
> 8. If `segment_offsets->segment_index() >= num_segments`, log and return
>    `Error::NotFound`.
> 9. `segment = segments()->Get(segment_offsets->segment_index())`.
> 10. Overflow-checked `end_offset = offset + size` (via `c10::add_overflows`);
>     on overflow ET_CHECK_OR_RETURN_ERROR → `Error::InvalidProgram`.
> 11. If `end_offset > segment->size()`, log and return `Error::InvalidArgument`.
> 12. Build SegmentInfo{type Mutable, index = segment_offsets->segment_index()}.
> 13. Compute the absolute file offset with overflow guards on a `uint64_t`
>     accumulator: `base_plus_seg = segment_base_offset_ + segment->offset()`
>     (add_overflows check AND `<= SIZE_MAX`, else `Error::InvalidProgram`),
>     then `total_offset = base_plus_seg + offset` (add_overflows check, else
>     `Error::InvalidProgram`).
> 14. Return `loader_->load_into(total_offset, size, info, buffer)` — writes the
>     bytes into `buffer` and returns Ok or a loader error.

> [spec:et:def:program.executorch.et-runtime-namespace.program.load-segment-fn]
> Result<FreeableBuffer> Program::LoadSegment( const DataLoader::SegmentInfo& segment_info) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn]
> Private method (friend-accessible). Loads the whole segment identified by
> `segment_info.segment_index` from the program's `segments` table and returns
> it as a `FreeableBuffer`. The other `segment_info` fields (type, descriptor)
> are passed through to the loader unchanged.
>
> Steps:
> 1. `index = segment_info.segment_index`.
> 2. If `loader_ == nullptr || segment_base_offset_ == 0` (no segments), log and
>    return `Error::NotFound`.
> 3. ET_CHECK_OR_RETURN_ERROR(`internal_program_->segments() != nullptr`, else
>    `Error::InvalidProgram`). Let `num_segments = segments()->size()`.
> 4. If `index >= num_segments`, log and return `Error::NotFound`.
> 5. `segment = segments()->Get(index)`; `seg_offset = segment->offset()`.
> 6. Compute `absolute_offset = segment_base_offset_ + seg_offset` on a
>    `uint64_t`; ET_CHECK_OR_RETURN_ERROR that the add does not overflow AND
>    `absolute_offset <= SIZE_MAX`, else `Error::InvalidProgram`.
> 7. Return `loader_->load((size_t)absolute_offset, segment->size(),
>    segment_info)` — the loaded buffer or a loader error.

> [spec:et:def:program.executorch.et-runtime-namespace.program.method-meta-fn]
> Result<MethodMeta> Program::method_meta(const char* method_name) const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.method-meta-fn]
> Gathers metadata for the named method, returning a `MethodMeta` wrapping its
> execution plan.
>
> Steps:
> 1. Resolve the execution plan via
>    `[spec:et:sem:program.executorch.et-runtime-namespace.get-execution-plan-fn]`;
>    on failure return its error.
> 2. Validate the plan fields whose accessors do not return `Result<>`, each via
>    ET_CHECK_OR_RETURN_ERROR returning `Error::InvalidProgram` if null:
>    `plan->name()`, `plan->non_const_buffer_sizes()`, `plan->inputs()`,
>    `plan->outputs()`.
> 3. Return `MethodMeta(plan)` constructed over that plan pointer.

> [spec:et:def:program.executorch.et-runtime-namespace.program.num-methods-fn]
> size_t Program::num_methods() const

> [spec:et:sem:program.executorch.et-runtime-namespace.program.num-methods-fn]
> Returns the number of methods (execution plans) in the program.
>
> Steps:
> 1. Take `internal_program_` (the raw flatbuffer `Program*`) and read
>    `execution_plan()`, its flatbuffer vector of `ExecutionPlan`.
> 2. If that vector pointer is non-null, return its `size()` (a `size_t`).
> 3. If it is null, return 0.
>
> Pure query; no validation, logging, or error path. Callers use the result as
> the exclusive upper bound for valid method indices.

> [spec:et:def:program.executorch.et-runtime-namespace.program.program-fn]
> Program( DataLoader* loader, size_t segment_base_offset, FreeableBuffer&& program_data, const executorch_flatbuffer::Program* internal_program, FreeableBuffer&& constant_segment_data, std::optional<internal::PteDataMap>&& pte_data_map) :...

> [spec:et:sem:program.executorch.et-runtime-namespace.program.program-fn]
> Private constructor. Assembles a `Program` from the pieces produced by
> `[spec:et:sem:program.executorch.et-runtime-namespace.program.load-fn]`. Takes
> `loader`, `segment_base_offset`, an rvalue `program_data` FreeableBuffer, the
> parsed `internal_program` flatbuffer pointer (pointing into `program_data`), an
> rvalue `constant_segment_data` FreeableBuffer, and an rvalue
> `std::optional<internal::PteDataMap>` `pte_data_map`.
>
> Member initialization (in declaration order):
> 1. `program_data_` = move of `program_data` (owns the serialized bytes that
>    `internal_program_` and constant/tensor data point into).
> 2. `loader_` = `segment_base_offset > 0 ? loader : nullptr`. When the program
>    has no segments (base offset 0) the loader is dropped, so later segment
>    loads short-circuit to `Error::NotFound`
>    (`[spec:et:sem:program.executorch.et-runtime-namespace.program.load-segment-fn]`).
> 3. `internal_program_` = `internal_program`.
> 4. `segment_base_offset_` = `segment_base_offset`.
> 5. `constant_segment_data_` = move of `constant_segment_data` (empty
>    FreeableBuffer when constants are inline or absent).
> 6. `pte_data_map_` = move of `pte_data_map` (`nullopt` when the program has no
>    named data).
>
> No validation or logging; all consistency checks happen in `load` before this
> is called.

> [spec:et:def:program.executorch.et-runtime-namespace.program.verification]
> enum class Verification : uint8_t {
>   Minimal;
>   InternalConsistency;
> }

> [spec:et:def:program.executorch.et-runtime-namespace.program.operator-fn]
> Program& operator=(Program&& rhs) noexcept = delete

> [spec:et:sem:program.executorch.et-runtime-namespace.program.operator-fn]
> Deleted move-assignment operator (`= delete`). `Program` is move-constructible
> (to be compatible with `Result<Program>`) but neither move-assignable nor
> copy-assignable nor copy-constructible. Any attempt to move-assign one Program
> onto another is a compile-time error; there is no runtime behavior to
> reimplement. In a Rust port, model this as a type that is movable (Rust move
> semantics) but exposes no assignment-through-reference operation.

