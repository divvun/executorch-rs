# runtime/executor/method.cpp, runtime/executor/method.h

> [spec:et:def:method.et-check-msg-fn]
> ET_CHECK_MSG(

> [spec:et:sem:method.et-check-msg-fn]
> `ET_CHECK_MSG(cond, fmt, ...)` is the runtime's fatal-assertion macro. It
> evaluates `cond`; if `cond` is truthy it does nothing and execution
> continues. If `cond` is falsy it logs a fatal message formatted from `fmt`
> and the trailing printf-style arguments (including source file/line context)
> and then aborts the process — it does NOT return an Error, it terminates.
> This is used for invariants that must hold in a correct program (e.g.
> bounds checks in `get_value`/`mutable_value`), so a Rust port models these
> call sites as `assert!`/`panic!`-style unrecoverable checks rather than
> recoverable `Error` returns. Here it guards `i < n_value_` before indexing
> `values_[i]`: when `i >= n_value_` it panics with the message
> "`<i>` >= `<n_value_>`".

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate]
> class BackendDelegate final {
>   FreeableBuffer segment_;
>   const BackendInterface* backend_;
>   DelegateHandle* handle_;
> }

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.backend-delegate-fn]
> ~BackendDelegate()

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.backend-delegate-fn]
> Destructor `~BackendDelegate()`. If `backend_` is non-null, calls
> `backend_->destroy(handle_)` to let the backend release any resources it
> associated with `handle_` (note `handle_` may itself be null if init never
> completed — the backend must tolerate that). If `backend_` is null (a
> zero/allocated-but-uninitialized delegate slot), does nothing. The
> `segment_` FreeableBuffer member is destroyed by its own destructor as part
> of normal member destruction, which frees the processed-data buffer if it
> owns one.

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.execute-fn]
> Error Execute( BackendExecutionContext& backend_execution_context, Span<EValue*> args) const

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.execute-fn]
> `Error Execute(BackendExecutionContext&, Span<EValue*> args) const`. Opens a
> profiling scope named "delegate_execute" (no-op when profiling is disabled).
> Delegates directly to `backend_->execute(backend_execution_context, handle_,
> args)` and returns that backend's Error verbatim. `backend_` is assumed
> non-null (guaranteed for a successfully-initialized delegate). Does not
> validate `args`; the backend consumes the EValue pointer list, reading its
> inputs and writing its outputs in place through those pointers.

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn]
> static Result<FreeableBuffer> GetProcessedData( const executorch_flatbuffer::BackendDelegate& delegate, const Program* program)

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn]
> `static Result<FreeableBuffer> GetProcessedData(const flatbuffer
> BackendDelegate& delegate, const Program* program)`. Reads
> `delegate.processed()` (a `BackendDelegateDataReference`) and switches on
> its `location()`:
>
> - `DataLocation::INLINE`: calls `program->get_backend_delegate_data(
>   processed->index(), &data, &size)`. On error returns that Error. On
>   success returns a `FreeableBuffer(data, size, free_fn=nullptr)` — i.e. a
>   non-owning view into program-owned inline data (freeing it is a no-op).
> - `DataLocation::SEGMENT`: takes `backend_id = delegate.id()->c_str()` and
>   returns `program->LoadSegment(SegmentInfo(Type::Backend, processed->index(),
>   backend_id))`, propagating that Result (the returned FreeableBuffer may own
>   loaded memory and free it on destruction).
> - any other value: logs "Unknown data location `<n>`" and returns
>   `Error::Internal`.

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.init-fn]
> static Error Init( const executorch_flatbuffer::BackendDelegate& delegate, const Program* program, BackendInitContext& backend_init_context, BackendDelegate* out)

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.init-fn]
> `static Error Init(const flatbuffer BackendDelegate& delegate, const
> Program* program, BackendInitContext& backend_init_context, BackendDelegate*
> out)`. Initializes the already-allocated `*out` in place. Steps:
>
> 1. Require `delegate.id() != nullptr` else return `Error::InvalidProgram`
>    ("Missing backend id"). Take `backend_id = delegate.id()->c_str()`.
> 2. Look up the backend: `backend = get_backend_class(backend_id)`. Require
>    non-null else return `Error::NotFound` ("Backend `<id>` is not
>    registered."). Require `backend->is_available()` else return
>    `Error::NotFound` ("Backend `<id>` is not available.").
> 3. Get the processed blob via `GetProcessedData(delegate, program)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.get-processed-data-fn]`.
>    On error, log and return that Error.
> 4. Compile specs: if `delegate.compile_specs() != nullptr`, call
>    `PopulateCompileSpecs(...)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn]`
>    into a local `compile_specs`; on error log and return it, else set
>    `num_compile_specs = delegate.compile_specs()->size()`. If null, leave
>    `compile_specs = nullptr`, `num_compile_specs = 0`.
> 5. Set `out->backend_ = backend`, `out->handle_ = nullptr`, and
>    placement-new `out->segment_` as a `FreeableBuffer` move-constructed from
>    the processed data (transferring ownership into the delegate; the buffer
>    outlives the backend and may be aliased by its handle).
> 6. Call `backend->init(backend_init_context, &out->segment_,
>    ArrayRef<CompileSpec>(compile_specs, num_compile_specs))`. On error: log
>    the error code, call `out->segment_.Free()`, and return the Error. On
>    success set `out->handle_ = handle.get()` and return `Error::Ok`.
>
> Ordering matters: `backend_` is set before `init`, so a later
> `~BackendDelegate` will call `destroy` even if `init` failed after step 5
> (though on init failure the segment is freed here).

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn]
> static Error PopulateCompileSpecs( const flatbuffers::Vector<flatbuffers::Offset< executorch_flatbuffer::CompileSpec>>* compile_specs_in_program, BackendInitContext& backend_init_context, CompileSpec** out_spec)

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.populate-compile-specs-fn]
> `static Error PopulateCompileSpecs(const flatbuffers::Vector<...CompileSpec>*
> compile_specs_in_program, BackendInitContext&, CompileSpec** out_spec)`.
> Builds a runtime `CompileSpec[]` from the serialized specs:
>
> 1. `number_of_compile_specs = compile_specs_in_program->size()`.
> 2. Allocate `CompileSpec[number_of_compile_specs]` from
>    `backend_init_context.get_runtime_allocator()` via `allocateList`. If
>    allocation returns null, return `Error::MemoryAllocationFailed`.
> 3. For each `j` in `[0, number_of_compile_specs)`: read
>    `compile_spec_in_program = compile_specs_in_program->Get(j)`; set
>    `list[j].key = compile_spec_in_program->key()->c_str()` (a pointer into
>    the flatbuffer, not copied) and `list[j].value = { buffer =
>    const_cast<void*> of value()->Data(), nbytes = value()->size() }` (also
>    aliasing flatbuffer memory).
> 4. Write the list pointer to `*out_spec` and return `Error::Ok`. The
>    resulting array and the data it points to remain valid only as long as
>    the allocator's memory and the program flatbuffer live.

> [spec:et:def:method.executorch.et-runtime-namespace.chain]
> struct Chain {
>   const executorch_flatbuffer::Chain* s_chain_;
>   Span<InstructionArgs> argument_lists_;
>   OpFunction* kernels_;
> }

> [spec:et:def:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn]
> Result<InstructionArgs> gen_instruction_arguments( MemoryAllocator* method_allocator, size_t num_values, EValue* values, size_t num_args, const int32_t* arg_idxs)

> [spec:et:sem:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn]
> `Result<InstructionArgs> gen_instruction_arguments(MemoryAllocator*
> method_allocator, size_t num_values, EValue* values, size_t num_args, const
> int32_t* arg_idxs)`. Builds the argument pointer list for one instruction:
>
> 1. Allocate `EValue*[num_args]` from `method_allocator` via `allocateList`.
>    If null, return `Error::MemoryAllocationFailed`.
> 2. For each `i` in `[0, num_args)`: read `arg_idx = arg_idxs[i]`; require
>    `static_cast<size_t>(arg_idx) < num_values` else return
>    `Error::InvalidProgram` ("Arg index `<arg_idx>` >= `<num_values>`") — note
>    the cast to `size_t` means a negative `arg_idx` becomes a huge value and
>    also fails this check. Set `arg_list[i] = &values[arg_idx]` (a pointer
>    into the master `values` table).
> 3. Return `InstructionArgs(arg_list, num_args)` (a `Span<EValue*>`).

> [spec:et:def:method.executorch.et-runtime-namespace.method]
> class Method final {
>   ET_DEPRECATED ET_NODISCARD;
>   ET_NODISCARD Result<executorch::aten::Tensor>;
>   ET_EXPERIMENTAL ET_NODISCARD;
>   ET_DEPRECATED ET_NODISCARD;
>   ET_EXPERIMENTAL ET_NODISCARD;
>   ET_EXPERIMENTAL ET_NODISCARD;
>   ET_DEPRECATED ET_NODISCARD;
>   const EValue& get_output(size_t i) const;
>   ET_DEPRECATED const EValue& get_input(size_t i) const;
>   ET_DEPRECATED EValue& mutable_input(size_t i);
>   ET_DEPRECATED EValue& mutable_output(size_t i);
>   enum class InitializationState : uint8_t { Uninitialized, Initialized, InitializationFailed, };
>   struct StepState { size_t chain_idx; size_t instr_idx; };
>   ET_NODISCARD static Result<Method>;
>   const EValue& get_value(size_t i) const;
>   EValue& mutable_value(size_t i);
>   StepState step_state_;
>   const Program* program_;
>   MemoryManager* memory_manager_;
>   MemoryAllocator* temp_allocator_;
>   executorch_flatbuffer::ExecutionPlan* serialization_plan_;
>   EventTracer* event_tracer_;
>   size_t n_value_;
>   EValue* values_;
>   bool* input_set_;
>   size_t n_delegate_;
>   BackendDelegate* delegates_;
>   size_t n_chains_;
>   Chain* chains_;
>   internal::MergedDataMap* merged_data_map_;
>   NamedData* external_constants_;
>   size_t n_external_constants_ = 0;
>   Span<const Kernel> kernel_registry_;
>   InitializationState init_state_;
>   ET_NODISCARD Result<size_t>;
> }

> [spec:et:def:method.executorch.et-runtime-namespace.method.execute-fn]
> ET_NODISCARD Error execute()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-fn]
> Public declaration of `Error execute()`. Behavior is defined by its
> definition; see `[spec:et:sem:method.method.execute-fn]`. Summary: runs all
> chains/instructions to completion, requiring the method to be initialized,
> not mid-`step()`, and all inputs set; resets execution state on both success
> and error so the method can be re-run.

> [spec:et:def:method.executorch.et-runtime-namespace.method.execute-instruction-fn]
> ET_NODISCARD Error execute_instruction()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.execute-instruction-fn]
> Private declaration of `Error execute_instruction()`. Behavior is defined by
> its definition; see `[spec:et:sem:method.method.execute-instruction-fn]`.
> Summary: executes the single instruction at `step_state_` (kernel, delegate,
> jump-false, move, or free), resets the temp allocator afterward, and on
> success advances `step_state_.instr_idx`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.experimental-reset-execution-fn]
> Error experimental_reset_execution()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-reset-execution-fn]
> Public declaration of the deprecated `Error experimental_reset_execution()`.
> Behavior defined by its definition; see
> `[spec:et:sem:method.method.experimental-reset-execution-fn]`. It simply
> forwards to `reset_execution()`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.experimental-step-fn]
> Error experimental_step()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.experimental-step-fn]
> Public declaration of the deprecated `Error experimental_step()`. Behavior
> defined by its definition; see
> `[spec:et:sem:method.method.experimental-step-fn]`. It simply forwards to
> `step()`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-attribute-fn]
> get_attribute( std::string_view name)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-attribute-fn]
> `Result<executorch::aten::Tensor> get_attribute(std::string_view name)`.
> Searches the serialized values for a constant/attribute tensor whose fully
> qualified name equals `name` and returns the corresponding runtime tensor:
>
> 1. Let `flatbuffer_values = serialization_plan_->values()`. Initialize
>    `counter = 0`.
> 2. For each `i` in `[0, flatbuffer_values->size())`: read
>    `serialization_value = flatbuffer_values->Get(i)`. If its `val_type()` is
>    `Tensor`, cast to a flatbuffer `Tensor`; if it has a non-null
>    `extra_tensor_info()` with a non-null `fully_qualified_name()` and that
>    name (`strcmp`) equals `name.data()`, then: if `values_[counter]` is not a
>    tensor, log a malformed-.pte error and return `Error::Internal`; otherwise
>    return `values_[counter].toTensor()`.
> 3. `++counter` at the end of every iteration (for every value, tensor or
>    not, matched or not).
> 4. If no match is found, return `Error::NotFound`.
>
> Note the search compares `name` against every tensor's qualified name but
> indexes the runtime `values_` array by `counter` (the raw value index), so
> the matched flatbuffer index and the returned `values_` index coincide.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-event-tracer-fn]
> EventTracer* get_event_tracer()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-event-tracer-fn]
> `EventTracer* get_event_tracer()`. Returns the `event_tracer_` pointer
> stored at construction (may be null if no tracer was provided). Pure
> accessor; no validation, no side effects.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-input-index-fn]
> size_t get_input_index(size_t i) const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn]
> `size_t get_input_index(size_t i) const`. Translates a zero-based input
> ordinal `i` into the corresponding index in the master `values_` table.
> `ET_CHECK_MSG(i < inputs_size(), ...)` — fatal panic if `i` is out of range
> (see `[spec:et:sem:method.et-check-msg-fn]`). Returns
> `static_cast<size_t>(serialization_plan_->inputs()->Get(i))`, i.e. the i-th
> entry of the serialized inputs index list.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-inputs-fn]
> Error

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-inputs-fn]
> Public (deprecated) declaration of `Error get_inputs(EValue* input_evalues,
> size_t length)`. Behavior defined by its definition; see
> `[spec:et:sem:method.method.get-inputs-fn]`. Summary: shallow-copies the
> method's inputs into `input_evalues[0..inputs_size())`, marks those inputs
> as set, and fills the remaining slots up to `length` with None EValues.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]
> Result<size_t> Method::get_num_external_constants()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]
> `Result<size_t> Method::get_num_external_constants()`. Counts serialized
> tensors tagged as external constants (and, as a side effect, validates that
> every serialized value is non-null so subsequent loops can skip null checks):
>
> 1. `flatbuffer_values = serialization_plan_->values()`, `n_value =
>    flatbuffer_values->size()`. Initialize `n_external_constants = 0`.
> 2. For each `i` in `[0, n_value)`: read `serialization_value =
>    flatbuffer_values->Get(i)`. Require it non-null AND (its `val_type()` is
>    `Null` OR `val() != nullptr`) — else return `Error::InvalidProgram`
>    ("Null value at index `<i>`").
> 3. If `val_type() != Tensor`, skip. Otherwise cast `val()` to a flatbuffer
>    `Tensor`. If `s_tensor->extra_tensor_info() != nullptr` AND its
>    `location() == TensorDataLocation::EXTERNAL` AND `s_tensor->
>    allocation_info() == nullptr` (external constant, not memory-planned/
>    mutable), increment `n_external_constants`.
> 4. Return `n_external_constants`. This is an upper bound on the number of
>    distinct external constants (multiple tensors may share one qualified
>    name / buffer).

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-output-index-fn]
> size_t get_output_index(size_t i) const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn]
> `size_t get_output_index(size_t i) const`. Translates a zero-based output
> ordinal `i` into the corresponding index in the master `values_` table.
> `ET_CHECK_MSG(i < outputs_size(), ...)` — fatal panic if out of range (see
> `[spec:et:sem:method.et-check-msg-fn]`). Returns
> `static_cast<size_t>(serialization_plan_->outputs()->Get(i))`, the i-th
> entry of the serialized outputs index list.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-outputs-fn]
> ET_NODISCARD Error get_outputs(EValue* output_evalues, size_t length)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-outputs-fn]
> Public declaration of `Error get_outputs(EValue* output_evalues, size_t
> length)`. Behavior defined by its definition; see
> `[spec:et:sem:method.method.get-outputs-fn]`. Summary: shallow-copies the
> method's outputs into `output_evalues[0..outputs_size())` and fills the
> remaining slots up to `length` with None EValues; requires the method
> initialized and `length >= outputs_size()`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.in-progress-fn]
> bool in_progress() const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.in-progress-fn]
> Public declaration of `bool in_progress() const`. Behavior defined by its
> definition; see `[spec:et:sem:method.method.in-progress-fn]`. Summary:
> returns true iff `step_state_` is neither the initial `{0,0}` position nor
> at/past the end (`chain_idx < n_chains_`).

> [spec:et:def:method.executorch.et-runtime-namespace.method.init-fn]
> Error Method::init( executorch_flatbuffer::ExecutionPlan* s_plan, const NamedDataMap* external_data_map, const LoadBackendOptionsMap* backend_options)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.init-fn]
> `Error Method::init(ExecutionPlan* s_plan, const NamedDataMap*
> external_data_map, const LoadBackendOptionsMap* backend_options)`.
> Initializes the method from its serialized plan. Opens an event-tracer scope
> "Method::init". Steps:
>
> 1. Require `init_state_ == InitializationState::Uninitialized` (so a second
>    init, or init after a prior failure, is rejected) else return
>    `Error::InvalidState`. Immediately set `init_state_ =
>    InitializationFailed` (pessimistic; flipped to `Initialized` only on full
>    success). Set `serialization_plan_ = s_plan`. Take `method_allocator =
>    memory_manager_->method_allocator()`.
> 2. Values: call `parse_values(external_data_map)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.parse-values-fn]`;
>    on error return it.
> 3. Delegates: read `delegates = serialization_plan_->delegates()`; require
>    non-null else `Error::InvalidProgram` ("Missing delegates field").
>    `n_delegate = delegates->size()`. Allocate `BackendDelegate[n_delegate]`
>    from `method_allocator`; if null return `Error::MemoryAllocationFailed`.
>    Obtain the program's named data map via `program_->get_named_data_map()`;
>    require it either ok or `Error::NotFound` else `Error::InvalidProgram`.
>    Choose the effective `named_data_map`: if both `external_data_map` and the
>    PTE data map are present, load a `MergedDataMap::load(external_data_map,
>    pte_data_map)` (on error return it), allocate and placement-new
>    `merged_data_map_`, and use it; else if only `external_data_map`, use it;
>    else if only the PTE map, use it; else null. Set `n_delegate_ = 0`, then
>    for each `i` in `[0, n_delegate)`: gather per-delegate runtime specs from
>    `backend_options->get_options(delegate.id())` when both are present,
>    build a `BackendInitContext(method_allocator, event_tracer_,
>    method_name, named_data_map, runtime_specs)`, call
>    `BackendDelegate::Init(delegate, program_, ctx, &delegates_[i])` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.init-fn]`;
>    on error return it, else set `n_delegate_ = i + 1` (so the destructor only
>    tears down successfully-inited delegates).
> 4. Chains: read `chains = serialization_plan_->chains()`; require non-null
>    and `size() > 0` else `Error::InvalidProgram` ("No chains"). `n_chains_ =
>    chains->size()`. Allocate `Chain[n_chains_]`; if null return
>    `Error::MemoryAllocationFailed`. Track `delayed_error = Error::Ok` and
>    `num_instructions_missing_op = 0` so all operators are attempted before
>    failing. For each chain `i`: require its `instructions()` non-null else
>    `Error::InvalidProgram`. Allocate `OpFunction[num_instructions]` and
>    `InstructionArgs[num_instructions]` (null → `MemoryAllocationFailed`).
>    For each `instr_idx`: require the instruction and its `instr_args()`
>    non-null else `Error::InvalidProgram`. Switch on `instr_args_type()`:
>    - `KernelCall`: require `args()` non-null; build the arg list via
>      `gen_instruction_arguments(method_allocator, n_value_, values_,
>      args->size(), args->data())` per
>      `[spec:et:sem:method.executorch.et-runtime-namespace.gen-instruction-arguments-fn]`
>      (return on error), store it, then `resolve_operator(op_index, kernels,
>      instr_idx, args, n_args)` per
>      `[spec:et:sem:method.executorch.et-runtime-namespace.method.resolve-operator-fn]`.
>      If that returns `Error::OperatorMissing` increment
>      `num_instructions_missing_op`; if `Error::MemoryAllocationFailed` return
>      immediately; otherwise record it in `delayed_error`.
>    - `DelegateCall`: require `args()` non-null; build and store the arg list
>      via `gen_instruction_arguments`. No operator resolution.
>    - `JumpFalseCall`: validate `cond_value_index()` is in `[0, n_value_)`
>      (`ET_CHECK_VALID_VALUE_INDEX`, else `Error::InvalidProgram`); store an
>      empty `InstructionArgs`.
>    - `MoveCall`: validate both `move_from()` and `move_to()` are in
>      `[0, n_value_)`; store an empty `InstructionArgs`.
>    - `FreeCall`: validate `value_index()` is in `[0, n_value_)`; store an
>      empty `InstructionArgs`.
>    - default: log and return `Error::InvalidProgram`.
>    After its instructions, set `chains_[i] = Chain{s_chain, arg_lists span,
>    kernels}`.
> 5. After all chains: require `num_instructions_missing_op == 0` else return
>    `Error::OperatorMissing`. If `delayed_error != Error::Ok`, return it.
> 6. Set `step_state_ = {0, 0}`, `init_state_ = Initialized`, return
>    `Error::Ok`.
>
> The `ET_CHECK_VALID_VALUE_INDEX(index, n_value)` helper requires `index >= 0
> && (size_t)index < n_value` else returns `Error::InvalidProgram`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.initialization-state]
> enum class InitializationState : uint8_t {
>   Uninitialized;
>   Initialized;
>   InitializationFailed;
> }

> [spec:et:def:method.executorch.et-runtime-namespace.method.initialized-fn]
> inline bool initialized() const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.initialized-fn]
> `inline bool initialized() const`. Returns `init_state_ ==
> InitializationState::Initialized`. That is, true only when init completed
> fully and successfully; false for both `Uninitialized` and
> `InitializationFailed`. Pure accessor, no side effects. Used to gate the
> public execution/input/output APIs.

> [spec:et:def:method.executorch.et-runtime-namespace.method.inputs-size-fn]
> size_t inputs_size() const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.inputs-size-fn]
> `size_t inputs_size() const`. Reads `serialization_plan_->inputs()` (the
> serialized input index list). If that pointer is null, returns 0; otherwise
> returns `inputs()->size()`. Pure accessor, no side effects. Defines the
> valid range for input ordinals used by `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn]`
> and `[spec:et:sem:method.method.set-inputs-fn]`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.load-fn]
> Result<Method> Method::load( executorch_flatbuffer::ExecutionPlan* s_plan, const Program* program, MemoryManager* memory_manager, EventTracer* event_tracer, const NamedDataMap* external_data_map, const LoadBackendOptionsMap* backend_opti...

> [spec:et:sem:method.executorch.et-runtime-namespace.method.load-fn]
> `static Result<Method> Method::load(ExecutionPlan* s_plan, const Program*
> program, MemoryManager* memory_manager, EventTracer* event_tracer, const
> NamedDataMap* external_data_map, const LoadBackendOptionsMap* backend_options
> = nullptr, Span<const Kernel> kernel_registry = {})`. Static factory (called
> by `Program`). Steps:
>
> 1. Obtain `temp_allocator = memory_manager->temp_allocator()`. If it is null,
>    allocate a `PlatformMemoryAllocator` instance from the method allocator
>    (`memory_manager->method_allocator()->allocateInstance<...>()`); if that
>    allocation returns null, log and return `Error::MemoryAllocationFailed`.
>    Placement-new the `PlatformMemoryAllocator` and use it as `temp_allocator`.
>    (So the method always has a usable temp allocator, defaulting to a
>    platform malloc/free allocator.)
> 2. Construct a `Method` in place via its private ctor `Method(program,
>    memory_manager, event_tracer, temp_allocator, kernel_registry)` — this
>    zero-initializes all state and sets `init_state_ = Uninitialized`.
> 3. Log the method name (`s_plan->name()->c_str()`).
> 4. Call `method.init(s_plan, external_data_map, backend_options)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.init-fn]`. On
>    error return that Error.
> 5. On success, `ET_CHECK(method.initialized())` (a fatal assert that init
>    really flipped the state; see `[spec:et:sem:method.et-check-msg-fn]`) and
>    return the moved `Method` by value (move-constructed into the Result per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn]`).

> [spec:et:def:method.executorch.et-runtime-namespace.method.log-outputs-fn]
> void log_outputs()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.log-outputs-fn]
> Private declaration of `void log_outputs()`. Behavior defined by its
> definition; see `[spec:et:sem:method.method.log-outputs-fn]`. Summary: when
> event tracing is compiled in and enabled at the program-outputs debug level,
> logs each output EValue to the event tracer; otherwise a no-op.

> [spec:et:def:method.executorch.et-runtime-namespace.method.method-fn]
> Method(Method&& rhs) noexcept

> [spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn]
> `Method(Method&& rhs) noexcept`. Move constructor. Takes ownership of all of
> `rhs`'s resources and leaves `rhs` in a safe, uninitialized, destructible
> state. Steps:
>
> 1. Member-copy every field from `rhs` into the new object: `step_state_`,
>    `program_`, `memory_manager_`, `temp_allocator_`, `serialization_plan_`,
>    `event_tracer_`, `n_value_`, `values_`, `input_set_`, `n_delegate_`,
>    `delegates_`, `n_chains_`, `chains_`, `merged_data_map_` (moved),
>    `external_constants_`, `n_external_constants_`, `kernel_registry_`,
>    `init_state_`. All are raw pointers/PODs, so this is a shallow transfer of
>    ownership (no deep copy).
> 2. Null out / zero the fields in `rhs` that its destructor inspects, so the
>    moved-from object frees nothing twice: `rhs.n_value_ = 0`, `rhs.values_ =
>    nullptr`, `rhs.input_set_ = nullptr`, `rhs.n_delegate_ = 0`,
>    `rhs.delegates_ = nullptr`, `rhs.merged_data_map_ = nullptr`,
>    `rhs.n_external_constants_ = 0`, `rhs.external_constants_ = nullptr`.
> 3. Also reset the remaining `rhs` fields to fail-safe values so any further
>    use of the moved-from object fails: `rhs.init_state_ =
>    InitializationState::Uninitialized`, `rhs.step_state_ = {}`, `rhs.program_
>    = nullptr`, `rhs.memory_manager_ = nullptr`, `rhs.serialization_plan_ =
>    nullptr`, `rhs.event_tracer_ = nullptr`, `rhs.n_chains_ = 0`, `rhs.chains_
>    = nullptr`.
>
> `noexcept`; never fails. A Rust port models this as the natural move of the
> owned handle where the moved-from value is not reused.

> [spec:et:def:method.executorch.et-runtime-namespace.method.method-meta-fn]
> MethodMeta method_meta() const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.method-meta-fn]
> Public declaration of `MethodMeta method_meta() const`. Behavior defined by
> its definition; see `[spec:et:sem:method.method.method-meta-fn]`. Summary:
> looks up this method's `MethodMeta` from the program by name; fatally asserts
> the lookup succeeds.

> [spec:et:def:method.executorch.et-runtime-namespace.method.outputs-size-fn]
> size_t outputs_size() const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.outputs-size-fn]
> Public declaration of `size_t outputs_size() const`. Behavior defined by its
> definition; see `[spec:et:sem:method.method.outputs-size-fn]`. Summary:
> returns the number of serialized outputs (0 if the outputs list is null).

> [spec:et:def:method.executorch.et-runtime-namespace.method.parse-external-constants-fn]
> Error Method::parse_external_constants(const NamedDataMap* external_data_map)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-external-constants-fn]
> `Error Method::parse_external_constants(const NamedDataMap* external_data_map)`.
> Resolves the distinct external constant tensors of this method into the
> already-allocated `external_constants_` array and counts them in
> `n_external_constants_`. Assumes all serialized values are non-null (validated
> earlier by `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]`).
> Steps:
>
> 1. Require `external_data_map != nullptr` else return `Error::InvalidState`
>    ("external_data_map is null").
> 2. `flatbuffer_values = serialization_plan_->values()`, `n_value =
>    flatbuffer_values->size()`. Set `n_external_constants_ = 0` (it is only
>    incremented at the very end of each successful iteration, so an early error
>    return leaves it counting only fully-initialized entries for the
>    destructor).
> 3. For each `i` in `[0, n_value)`: read `serialization_value =
>    flatbuffer_values->Get(i)`. If its `val_type() != Tensor`, skip. Cast
>    `val()` to a flatbuffer `Tensor`. Skip (continue) unless it is an external
>    constant: `extra_tensor_info() != nullptr` AND `location() == EXTERNAL` AND
>    `allocation_info() == nullptr` (tensors with allocation_info are
>    memory-planned/mutable and handled by `parse_values`).
> 4. Require `extra_tensor_info()->fully_qualified_name() != nullptr` else
>    return `Error::InvalidExternalData` ("Fully qualified name of external
>    tensor is null at index `<i>`"). Take `key = fully_qualified_name()->c_str()`.
> 5. Deduplicate: if `get_data_by_key(key, Span<NamedData>(external_constants_,
>    n_external_constants_))` already finds an entry with this key, skip
>    (continue) — multiple tensors may share one qualified name / buffer.
> 6. Fetch metadata: `tensor_layout = external_data_map->get_tensor_layout(key)`;
>    if not ok, log and return its Error.
> 7. Validate compatibility: `deserialization::validateTensorLayout(s_tensor,
>    tensor_layout.get())`; if it returns non-Ok, return that Error.
> 8. Store `external_constants_[n_external_constants_].key = key`.
> 9. Fetch the buffer: `buffer = external_data_map->get_data(key)`; require
>    `buffer.ok()` else return `Error::InvalidExternalData` ("Buffer retrieved
>    from get_data is not valid"). Placement-new
>    `external_constants_[n_external_constants_].buffer` as a `FreeableBuffer`
>    move-constructed from `buffer.get()` (the method now owns it; freed in
>    `~Method`).
> 10. `n_external_constants_ += 1`.
> 11. Return `Error::Ok`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.parse-values-fn]
> Error Method::parse_values(const NamedDataMap* external_data_map)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.parse-values-fn]
> `Error Method::parse_values(const NamedDataMap* external_data_map)`.
> Materializes the master `values_` EValue table from the serialized values,
> allocating from `memory_manager_->method_allocator()`. Steps:
>
> 1. `flatbuffer_values = serialization_plan_->values()`; require non-null else
>    `Error::InvalidProgram` ("Missing values"). `n_value =
>    flatbuffer_values->size()`. Allocate `values_ = EValue[n_value]`; null →
>    log and return `Error::MemoryAllocationFailed`.
> 2. `n_input = inputs_size()` per `[spec:et:sem:method.executorch.et-runtime-namespace.method.inputs-size-fn]`.
>    If `n_input > 0`, allocate `input_set_ = bool[n_input]` (null → log and
>    return `Error::MemoryAllocationFailed`) and initialize every element to
>    `false`.
> 3. Count external constants: `max_external_constants =
>    get_num_external_constants()` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-num-external-constants-fn]`
>    (also validates all values non-null); on error return it. If `> 0`,
>    allocate `external_constants_ = NamedData[max_external_constants]` (null →
>    log and return `Error::MemoryAllocationFailed`), then call
>    `parse_external_constants(external_data_map)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.parse-external-constants-fn]`;
>    on error return it.
> 4. Set `n_value_ = 0` (incremented at the bottom of each iteration; on error
>    it counts only fully-initialized entries so `~Method` cleans up exactly
>    those). For each `i` in `[0, n_value)`, read `serialization_value =
>    flatbuffer_values->Get(i)` and `val = serialization_value->val()`, then
>    switch on `val_type()` and placement-new `values_[i]` (placement-new
>    because the slot is uninitialized garbage — copy-assign into a non-trivial
>    EValue would be UB):
>    - `Null`: `new (&values_[i]) EValue()` (None).
>    - `Int`: `EValue(Int::int_val())` (int64).
>    - `Double`: `EValue(Double::double_val())`.
>    - `Bool`: `EValue(Bool::bool_val())`.
>    - `IntList`: require `items() != nullptr` else `Error::InvalidProgram`
>      ("Missing list at index `<i>`"). Allocate a boxed `EValue*[items.size()]`
>      and an unboxed `int64_t[items.size()]`. For each `j`, read `value_index =
>      items->Get(j)`; require `value_index >= 0 && (size_t)value_index <
>      n_value` else `Error::InvalidProgram` ("Invalid value index ..."); set
>      `evalp_list[j] = &values_[value_index]`. Allocate a
>      `BoxedEvalueList<int64_t>(evalp_list, int_list, items.size())` and wrap it
>      in the EValue. (The unboxed `int_list` is filled lazily on first
>      `toIntList()`.)
>    - `BoolList`: require `items() != nullptr` (else same error). Allocate an
>      `ArrayRef<bool>` viewing `items->data()`/`items->size()` directly (the
>      flatbuffer bytes are reinterpreted as `bool`; noted as technically
>      non-portable but accepted) and wrap it.
>    - `DoubleList`: require `items() != nullptr`. Allocate an `ArrayRef<double>`
>      over `items->data()`/`items->size()` and wrap it.
>    - `String`: read `string_val()`; require non-null else `Error::InvalidProgram`
>      ("Missing string at index `<i>`"). Allocate an `ArrayRef<char>` over
>      `c_str()`/`size()` and wrap it.
>    - `Tensor`: call `deserialization::parseTensor(program_, memory_manager_,
>      s_tensor, external_data_map, Span<NamedData>(external_constants_,
>      n_external_constants_))`; on error log and return it; else
>      `EValue(tensor)`.
>    - `TensorList`: require `items() != nullptr`. Call
>      `deserialization::parseTensorList(items, values_, n_value,
>      memory_manager_)` (on error log and return it); allocate a
>      `BoxedEvalueList<Tensor>` from the result and wrap it.
>    - `OptionalTensorList`: require `items() != nullptr`. Call
>      `deserialization::parseListOptionalType<Tensor>(items, values_, n_value,
>      memory_manager_)` (on error log and return it); allocate a
>      `BoxedEvalueList<std::optional<Tensor>>` and wrap it.
>    - default: log "Unknown KernelTypes value `<val_type-1>` at index `<i>`"
>      (the message subtracts one to undo the hidden NONE enum offset) and
>      return `Error::InvalidProgram`.
>    After a successful case, set `n_value_ = i + 1`.
> 5. Return `Error::Ok`.
>
> Any allocateList returning null within a list case is not separately checked
> here beyond the ones noted; the boxed/unboxed list allocations for IntList
> and instance allocations follow the same allocator and would surface as null
> deref only on a misconfigured allocator (the guarded allocations above are the
> validated ones).

> [spec:et:def:method.executorch.et-runtime-namespace.method.reset-execution-fn]
> Error reset_execution()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.reset-execution-fn]
> Public declaration of `Error reset_execution()`. Behavior defined by its
> definition; see `[spec:et:sem:method.method.reset-execution-fn]`. Summary:
> requires execution to have reached end-of-method (`chain_idx == n_chains_`)
> and resets `step_state_` to `{0,0}`, else returns `Error::InvalidState`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.resolve-operator-fn]
> Error Method::resolve_operator( int32_t op_index, OpFunction* kernels, size_t kernel_index, InstructionArgs args, size_t n_args)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.resolve-operator-fn]
> `Error Method::resolve_operator(int32_t op_index, OpFunction* kernels, size_t
> kernel_index, InstructionArgs args, size_t n_args)`. Looks up the concrete
> kernel function pointer for one KernelCall instruction and stores it at
> `kernels[kernel_index]`. Steps:
>
> 1. Resolve name: fixed local buffer `char operator_name[100]`. Read `ops =
>    serialization_plan_->operators()`; require `ops != nullptr && (uoffset_t)
>    op_index < ops->size()` else `Error::InvalidProgram` ("Op index ... out of
>    range"). `op = ops->Get(op_index)`. Call `populate_operator_name(op, 100,
>    operator_name)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.populate-operator-name-fn]`;
>    on error return it. Result is `"<name>"` or `"<name>.<overload>"`.
> 2. Choose an allocator for temporary `TensorMeta`: prefer
>    `memory_manager_->temp_allocator()`; if it is null or its `size() == 0`,
>    fall back to `memory_manager_->method_allocator()`.
> 3. Allocate `meta = TensorMeta[n_args]`. If null: if the chosen allocator is
>    the temp allocator, `reset()` it; return `Error::MemoryAllocationFailed`.
> 4. Build tensor metadata: iterate `i` in `[0, n_args)`, `eval = args[i]`.
>    Only for `eval->isTensor()`: set `meta[count].dtype_ =
>    tensor.scalar_type()`; allocate `dim_order_ptr =
>    DimOrderType[tensor.dim()]` (null → reset temp allocator if used, return
>    `Error::MemoryAllocationFailed`); call `get_dim_order(tensor,
>    dim_order_ptr, size=tensor.dim())`; require it returns `Error::Ok` else
>    `Error::InvalidArgument` ("Error setting dim_order ..."); set
>    `meta[count].dim_order_ = Span(dim_order_ptr, size)`; `count++`.
>    Non-tensor args are skipped, so `count <= n_args` — `meta[0..count)`
>    describes only the tensor arguments, in order.
> 5. Resolve the kernel: first, if `kernel_registry_` (the method-scoped
>    registry) is non-empty, try
>    `get_op_function_from_registry(operator_name, {meta, count},
>    kernel_registry_)`; if that is ok, use it. Otherwise fall back to the
>    global `get_op_function_from_registry(operator_name, {meta, count})`.
> 6. If resolution failed: log "Missing operator: [`<op_index>`] `<name>`",
>    reset the temp allocator if it was the one used, and return the resolver's
>    Error (typically `Error::OperatorMissing`).
> 7. On success set `kernels[kernel_index] = op_function.get()`, reset the temp
>    allocator if it was used, and return `Error::Ok`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.set-input-fn]
> ET_NODISCARD Error

> [spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn]
> `Error Method::set_input(const EValue& input_evalue, size_t input_idx)`. Sets
> one method input from the caller-provided EValue, validating type/value
> agreement with what was traced. Steps:
>
> 1. Require `initialized()` else `Error::InvalidState` ("Input can not be set
>    until method has been initialized.").
> 2. Require not mid-execution: `step_state_.instr_idx == 0 &&
>    step_state_.chain_idx == 0` else `Error::InvalidState` ("Inputs can not be
>    set mid execution.").
> 3. Require `input_idx < inputs_size()` else `Error::InvalidArgument`.
> 4. `e = get_value(get_input_index(input_idx))` — the destination slot in
>    `values_`. Require `e` is None, Tensor, Scalar (Int/Double/Bool), or String
>    else log and return `Error::InvalidArgument` ("... expected to be a Tensor
>    or primitive ...").
> 5. Require `e.tag == input_evalue.tag` (same EValue kind) else log and return
>    `Error::InvalidArgument`.
> 6. Dispatch on `e`'s kind:
>    - None: no-op.
>    - Tensor: let `t_dst = e.toTensor()`, `t_src = input_evalue.toTensor()`.
>      Require `t_dst.scalar_type() == t_src.scalar_type()` else
>      `Error::InvalidArgument` ("... unexpected scalar type ..."). Compute
>      `numel` as the product of `t_src.size(i)` over all dims using
>      `c10::mul_overflows` on `ssize_t`; on any multiply overflow return
>      `Error::InvalidArgument` ("numel overflowed at dimension ..."). Compute
>      `nbytes = numel * elementSize(t_src.scalar_type())` with
>      `c10::mul_overflows` on `size_t`; on overflow return
>      `Error::InvalidArgument` ("nbytes overflowed ..."). Resize the
>      destination to the source shape: `resize_tensor(t_dst, t_src.sizes())`;
>      on error return it (supports dynamic shapes). Query
>      `this->method_meta().input_tensor_meta(input_idx)`: if the input
>      `is_memory_planned()`, deep-copy via `internal::copy_tensor_data(t_dst,
>      t_src)`; otherwise alias via `internal::share_tensor_data(t_dst, t_src)`
>      (Method keeps a pointer into the caller's tensor data). Return the
>      copy/share error on failure.
>    - Int: require `e.toInt() == input_evalue.toInt()` else
>      `Error::InvalidArgument` (traced prims must be re-supplied unchanged).
>    - Bool: require `e.toBool() == input_evalue.toBool()` else
>      `Error::InvalidArgument`.
>    - Double: compare with tolerance. Let `lhs = input_evalue.toDouble()`, `rhs
>      = e.toDouble()`. Treat NaN==NaN as equal; treat two same-signed infinities
>      as equal; otherwise `is_equal` iff `abs(lhs - rhs)` is finite and `<=
>      1e-4 + abs(1e-5 * rhs)` (atol=1e-4, rtol=1e-5). Require `is_equal` else
>      `Error::InvalidArgument`.
>    - String: require `e.toString() == input_evalue.toString()` else
>      `Error::InvalidArgument`.
>    - any other tag (unreachable given step 4): log "Unsupported input type"
>      and return `Error::InvalidArgument`.
> 7. On success set `input_set_[input_idx] = true` and return `Error::Ok`.

> [spec:et:def:method.executorch.et-runtime-namespace.method.set-inputs-fn]
> ET_NODISCARD Error

> [spec:et:sem:method.executorch.et-runtime-namespace.method.set-inputs-fn]
> Public declaration of `Error set_inputs(const ArrayRef<EValue>&
> input_evalues)`. Behavior defined by its definition; see
> `[spec:et:sem:method.executorch.method.set-inputs-fn]`. Summary: requires the
> array length to equal `inputs_size()`, then calls `set_input` for each input
> in order, stopping and returning the first error.

> [spec:et:def:method.executorch.et-runtime-namespace.method.set-output-data-ptr-fn]
> ET_NODISCARD Error

> [spec:et:sem:method.executorch.et-runtime-namespace.method.set-output-data-ptr-fn]
> Public declaration of `Error set_output_data_ptr(void* buffer, size_t size,
> size_t output_idx)`. Behavior defined by its definition; see
> `[spec:et:sem:method.executorch.method.set-output-data-ptr-fn]`. Summary:
> points a non-memory-planned output tensor at a caller-owned buffer after
> validating the method is initialized, the output is a tensor, is not
> memory-planned, and the buffer is large enough.

> [spec:et:def:method.executorch.et-runtime-namespace.method.step-fn]
> Error step()

> [spec:et:sem:method.executorch.et-runtime-namespace.method.step-fn]
> Public declaration of `Error step()`. Behavior defined by its definition; see
> `[spec:et:sem:method.method.step-fn]`. Summary: advances execution by one
> instruction (or skips an empty chain), requiring the method initialized;
> returns `Error::EndOfMethod` once all chains are done.

> [spec:et:def:method.executorch.et-runtime-namespace.method.step-state]
> struct StepState {
>   size_t chain_idx;
>   size_t instr_idx;
> }

> [spec:et:def:method.executorch.et-runtime-namespace.parse-cond-value-fn]
> Result<bool> parse_cond_value(const EValue& cond_value)

> [spec:et:sem:method.executorch.et-runtime-namespace.parse-cond-value-fn]
> `Result<bool> parse_cond_value(const EValue& cond_value)`. Interprets the
> condition attached to a JumpFalseCall as a boolean. Two shapes occur: a bool
> Tensor (at the head of an if/else) or a Bool scalar (at the end of an if
> branch). Steps:
>
> 1. If `cond_value.isTensor()`: let `cond_val = cond_value.toTensor()`. Require
>    `cond_val.scalar_type() == ScalarType::Bool` else `Error::InvalidProgram`
>    ("Expected dtype ..."). Get `cond_data = cond_val.const_data_ptr<bool>()`;
>    require non-null else `Error::InvalidState` ("Tensor data is null"). Iterate
>    all `numel()` elements; if any element is `false`, return `false`
>    immediately. (I.e. the tensor condition is true iff every element is true;
>    an empty tensor yields true.)
> 2. Else if `cond_value.isBool()`: if `!cond_value.toBool()`, return `false`.
> 3. Else: log "Unsupported JF EValue type `<tag>`" and return
>    `Error::InvalidProgram`.
> 4. If none of the above returned `false`/error, return `true`.
>
> A returned `false` means "jump to the destination"; `true` means "fall
> through" (per `[spec:et:sem:method.method.execute-instruction-fn]`).

> [spec:et:def:method.executorch.et-runtime-namespace.populate-operator-name-fn]
> Error populate_operator_name( const executorch_flatbuffer::Operator* const& op, const size_t operator_name_size, char* operator_name)

> [spec:et:sem:method.executorch.et-runtime-namespace.populate-operator-name-fn]
> `Error populate_operator_name(const executorch_flatbuffer::Operator* const&
> op, size_t operator_name_size, char* operator_name)`. Writes the fully
> qualified operator name into the caller-provided fixed buffer. Steps:
>
> 1. `has_overload = op->overload() != nullptr && op->overload()->size() > 0`.
> 2. Require `op->name() != nullptr` else `Error::InvalidProgram` ("Missing
>    operator name").
> 3. `cx = snprintf(operator_name, operator_name_size, "%s%s%s",
>    op->name()->c_str(), has_overload ? "." : "", has_overload ?
>    op->overload()->c_str() : "")`. So the result is `"<name>"` when there is
>    no overload and `"<name>.<overload>"` when there is.
> 4. Require `cx >= 0` else `Error::Internal` ("snprintf failed").
> 5. Require `(size_t)cx < operator_name_size` (the full name fit without
>    truncation, leaving room for the NUL) else `Error::Internal` ("... truncated
>    ... due to internal buffer limit.").
> 6. Return `Error::Ok`.

> [spec:et:def:method.executorch.method.set-inputs-fn]
> ET_NODISCARD Error

> [spec:et:sem:method.executorch.method.set-inputs-fn]
> `Error Method::set_inputs(const ArrayRef<EValue>& input_evalues)`. Sets all
> method inputs at once. Steps:
>
> 1. `n_input = inputs_size()`. Require `input_evalues.size() == n_input` else
>    `Error::InvalidArgument` ("Invalid number of inputs provided ...").
> 2. For each `i` in `[0, n_input)`: call `set_input(input_evalues[i], i)` per
>    `[spec:et:sem:method.executorch.et-runtime-namespace.method.set-input-fn]`;
>    on the first non-Ok result, return that Error immediately (inputs already
>    set before the failure remain set).
> 3. Return `Error::Ok`.

> [spec:et:def:method.executorch.method.set-output-data-ptr-fn]
> ET_NODISCARD Error

> [spec:et:sem:method.executorch.method.set-output-data-ptr-fn]
> `Error Method::set_output_data_ptr(void* buffer, size_t size, size_t
> output_idx)`. Redirects a non-memory-planned output tensor to a caller-owned
> buffer. Steps:
>
> 1. Require `initialized()` else `Error::InvalidState`.
> 2. Require `output_idx < outputs_size()` else `Error::InvalidArgument`
>    ("output_idx ... > num_outputs ...").
> 3. `output = mutable_value(get_output_index(output_idx))`. Require
>    `output.isTensor()` else log and return `Error::InvalidArgument` ("Output
>    type ... is not a tensor.").
> 4. Query `this->method_meta().output_tensor_meta(output_idx)`. If
>    `is_memory_planned()`, log and return `Error::InvalidState` ("Output ... is
>    memory planned, or is a constant. Cannot override the existing data
>    pointer.").
> 5. `t = output.toTensor()`. (A redundant second `isTensor()` re-check exists
>    but always passes here.)
> 6. Require `t.nbytes() <= size` else `Error::InvalidArgument` ("buffer size
>    ... is smaller then expected tensor size ...").
> 7. Return `internal::set_tensor_data(t, buffer, size)` — points the tensor's
>    data at `buffer`; the caller must keep `buffer` alive across execution.

> [spec:et:def:method.method.execute-fn]
> Error Method::execute()

> [spec:et:sem:method.method.execute-fn]
> `Error Method::execute()`. Runs the whole method to completion. Opens an event
> tracer block "Execute" and a profiling scope "Method::execute". Steps:
>
> 1. Require `initialized()` else `Error::NotSupported` ("Cannot execute until
>    method has been initialized.").
> 2. Require `!in_progress()` per
>    `[spec:et:sem:method.method.in-progress-fn]` else `Error::InvalidState`
>    ("Method execution is in progress") — you cannot `execute()` while a
>    `step()`-based run is mid-way.
> 3. For each input `i` in `[0, inputs_size())`, require `input_set_[i]` else
>    `Error::InvalidArgument` ("Input `<i>` has not been set.").
> 4. Reset the temp allocator if present.
> 5. Iterate chains sequentially: for `step_state_.chain_idx` from `0` to
>    `n_chains_ - 1`. For each chain, read `instructions =
>    chain.s_chain_->instructions()`; require non-null else `Error::Internal`.
>    Set `step_state_.instr_idx = 0`, then loop while `instr_idx <
>    instructions->size()`. Inside the loop:
>    - Enforce the infinite-loop guard: maintain a running `instruction_count`
>      across all chains; if `instruction_count >= kMaxInstructions`
>      (compile-time `ET_MAX_INSTRUCTIONS`, default 10,000,000), log, set
>      `step_state_ = {0,0}`, and return `Error::InvalidProgram`. Otherwise
>      `++instruction_count`.
>    - Open per-instruction profiling/tracer scopes, then call
>      `execute_instruction()` per
>      `[spec:et:sem:method.method.execute-instruction-fn]`. On non-Ok, set
>      `step_state_ = {0,0}` (auto-reset so the method can be retried) and return
>      that Error. Note `execute_instruction` advances `instr_idx` itself (and a
>      JumpFalse may move it backward/forward), so this loop follows control flow
>      rather than a simple increment.
> 6. After all chains: `log_outputs()` per
>    `[spec:et:sem:method.method.log-outputs-fn]`, then return
>    `reset_execution()` (which sets `step_state_ = {0,0}` since `chain_idx ==
>    n_chains_`). The overall return is that `reset_execution()` result, i.e.
>    `Error::Ok` on a normal run.

> [spec:et:def:method.method.execute-instruction-fn]
> Error Method::execute_instruction()

> [spec:et:sem:method.method.execute-instruction-fn]
> `Error Method::execute_instruction()`. Executes the single instruction at
> `step_state_` and advances/branches the instruction pointer. Steps:
>
> 1. `chain = chains_[step_state_.chain_idx]`; `instructions =
>    chain.s_chain_->instructions()`. Require `step_state_.instr_idx <
>    instructions->size()` else `Error::Internal` ("Instr index ... >= ...
>    instr count ...").
> 2. `instruction = instructions->Get(step_state_.instr_idx)`. Initialize
>    `next_instr_idx = step_state_.instr_idx + 1` and `err = Error::Ok`.
> 3. Switch on `instruction->instr_args_type()`:
>    - `KernelCall`: open OPERATOR_CALL profiling/tracer scopes. Build a
>      `KernelRuntimeContext(event_tracer_, temp_allocator_)`. Fetch `args =
>      chain.argument_lists_[instr_idx]` and invoke the resolved kernel
>      `chain.kernels_[instr_idx](context, args)`. Set `err =
>      context.failure_state()`. On non-Ok, log the failing op name/overload and
>      each arg's tag (diagnostic only). (The kernel writes its outputs in place
>      through the arg EValue pointers.)
>    - `DelegateCall`: open DELEGATE_CALL scopes. Read `delegate_idx =
>      instr_args_as_DelegateCall()->delegate_index()`; require `(size_t)
>      delegate_idx < n_delegate_` else `Error::Internal` ("DELEGATE_CALL index
>      ... >= num delegates ..."). Build `BackendExecutionContext(event_tracer_,
>      temp_allocator_, method_name)` and call
>      `delegates_[delegate_idx].Execute(ctx, chain.argument_lists_[instr_idx])`
>      per
>      `[spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.execute-fn]`,
>      storing the result in `err`; log on failure. If event tracing is enabled,
>      log every argument EValue.
>    - `JumpFalseCall`: open JF_CALL scopes. `jf_call =
>      instr_args_as_JumpFalseCall()`; `index = jf_call->cond_value_index()`
>      (validated in-bounds at init). Compute `parse_cond_value(values_[index])`
>      per `[spec:et:sem:method.executorch.et-runtime-namespace.parse-cond-value-fn]`.
>      If ok and the condition is `false`, set `next_instr_idx =
>      jf_call->destination_instruction()` (branch/jump). If ok and true, fall
>      through. On error, set `err` to that error.
>    - `MoveCall`: open MOVE_CALL scopes. `move_call =
>      instr_args_as_MoveCall()`; assign `mutable_value(move_call->move_to()) =
>      get_value(move_call->move_from())` (EValue copy-assign; indices validated
>      at init). `err` stays Ok.
>    - `FreeCall`: open FREE_CALL scopes. `free_call =
>      instr_args_as_FreeCall()`; `t =
>      mutable_value(free_call->value_index()).tryToTensor()`. If not a Tensor,
>      log and set `err = t.error()` and break. Otherwise
>      `internal::reset_data_ptr(t.get())` (releases the tensor's data pointer).
>    - default: log "Unknown instruction ..." and set `err =
>      Error::InvalidProgram`.
> 4. After the switch, if `temp_allocator_ != nullptr`, `reset()` it (per
>    instruction).
> 5. If `err == Error::Ok`, set `step_state_.instr_idx = next_instr_idx` (commit
>    the advance/branch). Return `err`.

> [spec:et:def:method.method.experimental-reset-execution-fn]
> Error Method::experimental_reset_execution()

> [spec:et:sem:method.method.experimental-reset-execution-fn]
> `Error Method::experimental_reset_execution()`. Deprecated alias. Simply
> returns `reset_execution()` per
> `[spec:et:sem:method.method.reset-execution-fn]` with identical behavior.

> [spec:et:def:method.method.experimental-step-fn]
> Error Method::experimental_step()

> [spec:et:sem:method.method.experimental-step-fn]
> `Error Method::experimental_step()`. Deprecated alias. Simply returns
> `step()` per `[spec:et:sem:method.method.step-fn]` with identical behavior.

> [spec:et:def:method.method.get-inputs-fn]
> ET_NODISCARD Error Method::get_inputs(EValue* input_evalues, size_t length)

> [spec:et:sem:method.method.get-inputs-fn]
> `Error Method::get_inputs(EValue* input_evalues, size_t length)`. Deprecated.
> Shallow-copies the method's inputs into a caller array. Steps:
>
> 1. Require `initialized()` else `Error::InvalidState`.
> 2. `n_input = inputs_size()`. Require `length >= n_input` else
>    `Error::InvalidArgument` ("The given array is not large enough ...").
> 3. For each `i` in `[0, n_input)`: `input_evalues[i] =
>    values_[get_input_index(i)]` (shallow copy — tensor elements alias internal
>    tensors, so callers must not mutate them) and set `input_set_[i] = true`
>    (this accessor assumes the caller becomes responsible for the input's
>    value).
> 4. For each `i` in `[n_input, length)`: `input_evalues[i] = EValue()` (None).
> 5. Return `Error::Ok`.

> [spec:et:def:method.method.get-outputs-fn]
> ET_NODISCARD Error Method::get_outputs(EValue* output_evalues, size_t length)

> [spec:et:sem:method.method.get-outputs-fn]
> `Error Method::get_outputs(EValue* output_evalues, size_t length)`.
> Shallow-copies the method's outputs into a caller array. Steps:
>
> 1. Require `initialized()` else `Error::InvalidState`.
> 2. `n_output = outputs_size()`. Require `length >= n_output` else
>    `Error::InvalidArgument` ("The given array is not large enough to hold all
>    outputs.").
> 3. For each `i` in `[0, n_output)`: `output_evalues[i] = get_output(i)`
>    (shallow copy; do not mutate returned tensor elements).
> 4. For each `i` in `[n_output, length)`: `output_evalues[i] = EValue()`
>    (None).
> 5. Return `Error::Ok`.

> [spec:et:def:method.method.in-progress-fn]
> bool Method::in_progress() const

> [spec:et:sem:method.method.in-progress-fn]
> `bool Method::in_progress() const`. Returns `(step_state_.chain_idx != 0 ||
> step_state_.instr_idx != 0) && step_state_.chain_idx < n_chains_`. That is,
> true only when execution has advanced past the initial `{0,0}` position AND
> has not yet reached/passed the end (`chain_idx == n_chains_`). Consequently it
> is false at the start, false after a completed run (state reset to `{0,0}`),
> and false after any error (state auto-reset). Pure read of `step_state_`; no
> side effects.

> [spec:et:def:method.method.log-outputs-fn]
> void Method::log_outputs()

> [spec:et:sem:method.method.log-outputs-fn]
> `void Method::log_outputs()`. Only compiled when `ET_EVENT_TRACER_ENABLED` is
> defined; otherwise an empty no-op. When enabled: if `event_tracer_ != nullptr`
> and `event_tracer_->event_tracer_debug_level() >=
> EventTracerDebugLogLevel::kProgramOutputs`, iterate `i` in `[0,
> outputs_size())` and call `internal::event_tracer_log_evalue_output(
> event_tracer_, get_output(i))` for each output. No return value, no state
> change beyond the tracer's own logging.

> [spec:et:def:method.method.method-meta-fn]
> MethodMeta Method::method_meta() const

> [spec:et:sem:method.method.method-meta-fn]
> `MethodMeta Method::method_meta() const`. Returns the metadata for this
> method. Steps: take `name = serialization_plan_->name()->c_str()`; call
> `program_->method_meta(name)`; `ET_CHECK_MSG` that the result is ok (fatal
> panic on failure — see `[spec:et:sem:method.et-check-msg-fn]` — with message
> "Internal error: method_meta(`<name>`) returned `<error>`"); return
> `method_meta.get()` by value. No side effects.

> [spec:et:def:method.method.outputs-size-fn]
> size_t Method::outputs_size() const

> [spec:et:sem:method.method.outputs-size-fn]
> `size_t Method::outputs_size() const`. Reads `serialization_plan_->outputs()`
> (the serialized output index list). If null, returns 0; otherwise returns
> `outputs()->size()`. Pure accessor, no side effects. Defines the valid range
> for output ordinals used by
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn]`.

> [spec:et:def:method.method.reset-execution-fn]
> Error Method::reset_execution()

> [spec:et:sem:method.method.reset-execution-fn]
> `Error Method::reset_execution()`. Resets the step-based execution cursor.
> Require `step_state_.chain_idx == n_chains_` (execution has reached
> end-of-method) else return `Error::InvalidState` ("Cannot reset until
> EndOfMethod has been reached."). On success set `step_state_ = StepState{0,
> 0}` and return `Error::Ok`.

> [spec:et:def:method.method.step-fn]
> Error Method::step()

> [spec:et:sem:method.method.step-fn]
> `Error Method::step()`. Advances execution by one instruction (the
> incremental counterpart to `execute()`). Opens per-instruction
> profiling/tracer scopes and a "Method::step" profiling event. Steps:
>
> 1. Require `initialized()` else `Error::InvalidState` ("Cannot execute until
>    method has been initialized.").
> 2. If `step_state_.chain_idx == n_chains_`, all chains are done: return
>    `Error::EndOfMethod`.
> 3. `num_instructions =
>    chains_[step_state_.chain_idx].s_chain_->instructions()->size()`. If it is
>    0 (empty chain, e.g. a model that just returns an input/constant): advance
>    `step_state_.chain_idx += 1` and return `Error::Ok`.
> 4. Call `execute_instruction()` per
>    `[spec:et:sem:method.method.execute-instruction-fn]`; on non-Ok, return
>    that Error (state is not auto-reset here, unlike `execute()`).
> 5. End the profiling event. If, after the instruction advanced,
>    `step_state_.instr_idx == num_instructions` (reached end of this chain),
>    set `instr_idx = 0`, `chain_idx += 1`, and call `log_outputs()`.
> 6. Return `Error::Ok`.
>
> The caller keeps calling `step()` until it returns `Error::EndOfMethod`, then
> calls `reset_execution()` before reusing the method.

> [spec:et:def:method.executorch.et-runtime-namespace.backend-delegate.operator-fn]
> BackendDelegate& operator=(const BackendDelegate&) = delete

> [spec:et:sem:method.executorch.et-runtime-namespace.backend-delegate.operator-fn]
> `BackendDelegate& operator=(const BackendDelegate&) = delete`. Copy
> assignment is explicitly deleted (along with the copy ctor, move ctor, and
> move assignment): a `BackendDelegate` owns a backend handle and a
> `FreeableBuffer` and must not be copied or moved. No runtime behavior — this
> is a compile-time prohibition. A Rust port models the type as non-`Clone`,
> non-`Copy`, and pinned in place (the delegates array holds them by value and
> only ever constructs them in place via `Init`).

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-input-fn]
> ET_DEPRECATED const EValue& get_input(size_t i) const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-fn]
> `const EValue& Method::get_input(size_t i) const`. Deprecated accessor for the
> i-th input. Side effect: marks `input_set_[i] = true` (the accessor assumes
> the caller takes responsibility for the input's value). Then returns
> `get_value(get_input_index(i))` — i.e. translates the input ordinal to a
> `values_` index via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn]`
> and reads that slot via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn]`
> (which fatally panics if the translated index is out of range). No bounds
> check on `i` itself beyond what `get_input_index` asserts.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-output-fn]
> const EValue& get_output(size_t i) const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-fn]
> `const EValue& Method::get_output(size_t i) const`. Returns the i-th output
> by delegating to `get_value(get_output_index(i))` — translates the output
> ordinal to a `values_` index via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn]`
> (fatal panic if `i >= outputs_size()`) and reads that slot via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn]`. No
> side effects.

> [spec:et:def:method.executorch.et-runtime-namespace.method.get-value-fn]
> const EValue& get_value(size_t i) const

> [spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn]
> `const EValue& Method::get_value(size_t i) const`. Returns a const reference
> to `values_[i]` in the master value table. `ET_CHECK_MSG(i < n_value_, ...)`
> — fatal panic if `i >= n_value_` (see `[spec:et:sem:method.et-check-msg-fn]`),
> with message "`<i>` >= `<n_value_>`". No side effects.

> [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-input-fn]
> ET_DEPRECATED EValue& mutable_input(size_t i)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-input-fn]
> `EValue& Method::mutable_input(size_t i)`. Deprecated mutable accessor for the
> i-th input. Side effect: sets `input_set_[i] = true`. Returns
> `mutable_value(get_input_index(i))` — translates the input ordinal to a
> `values_` index via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-input-index-fn]`
> and returns a mutable reference to that slot via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-value-fn]`
> (fatal panic if the translated index is out of range).

> [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-output-fn]
> ET_DEPRECATED EValue& mutable_output(size_t i)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-output-fn]
> `EValue& Method::mutable_output(size_t i)`. Deprecated mutable accessor for
> the i-th output. Returns `mutable_value(get_output_index(i))` — translates the
> output ordinal to a `values_` index via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-output-index-fn]`
> (fatal panic if `i >= outputs_size()`) and returns a mutable reference to that
> slot via
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-value-fn]`.
> No `input_set_` side effect (outputs, not inputs).

> [spec:et:def:method.executorch.et-runtime-namespace.method.mutable-value-fn]
> EValue& mutable_value(size_t i)

> [spec:et:sem:method.executorch.et-runtime-namespace.method.mutable-value-fn]
> `EValue& Method::mutable_value(size_t i)`. Returns a mutable reference to
> `values_[i]` in the master value table. `ET_CHECK_MSG(i < n_value_, ...)` —
> fatal panic if `i >= n_value_` (see `[spec:et:sem:method.et-check-msg-fn]`),
> with message "`<i>` >= `<n_value_>`". The mutable counterpart to
> `[spec:et:sem:method.executorch.et-runtime-namespace.method.get-value-fn]`;
> used by MoveCall/FreeCall and the output/input mutation accessors.

> [spec:et:def:method.executorch.et-runtime-namespace.method.operator-fn]
> Method& operator=(const Method&) noexcept = delete

> [spec:et:sem:method.executorch.et-runtime-namespace.method.operator-fn]
> `Method& operator=(const Method&) noexcept = delete`. Copy assignment is
> explicitly deleted (as are the copy ctor and move assignment; only the move
> ctor per `[spec:et:sem:method.executorch.et-runtime-namespace.method.method-fn]`
> is allowed). A `Method` uniquely owns its allocated `values_`, `delegates_`,
> external constants, and merged data map, so it must not be copied or
> reassigned. No runtime behavior — a compile-time prohibition. A Rust port
> models `Method` as a non-`Clone`, move-only owner.

