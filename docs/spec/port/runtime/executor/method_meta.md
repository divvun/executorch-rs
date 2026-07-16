# runtime/executor/method_meta.cpp, runtime/executor/method_meta.h

> [spec:et:def:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]
> Result<size_t> calculate_nbytes( Span<const int32_t> sizes, executorch::aten::ScalarType scalar_type)

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]
> File-local helper (anonymous namespace). Computes the total byte size of a
> tensor from its `sizes` span and `scalar_type`. Returns `Result<size_t>`.
>
> Algorithm:
> - Initialize `n = 1` (type `size_t`), the running product of the dimensions.
> - Iterate `i` from `0` to `sizes.size() - 1` inclusive:
>   - Validate `sizes[i] >= 0`; on failure return `Error::InvalidProgram`
>     (message notes the size must not be negative).
>   - Compute `next_n = n * (size_t)sizes[i]` using a checked multiply
>     (`c10::mul_overflows`). If the multiply overflows `size_t`, return
>     `Error::InvalidArgument`.
>   - Set `n = next_n`.
> - So `n` is the product of all sizes; an empty `sizes` span yields `n = 1`
>   (scalar), and any zero dimension yields `n = 0`.
> - Look up `elem_size = elementSize(scalar_type)`, the byte width of one
>   element of `scalar_type`.
> - Compute `total_bytes = n * elem_size` with a checked multiply; on overflow
>   return `Error::InvalidArgument`.
> - Return `total_bytes`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.get-tag-fn]
> Result<Tag> get_tag( flatbuffers::Vector<flatbuffers::Offset<executorch_flatbuffer::EValue>>:: return_type serialization_value, size_t index)

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.get-tag-fn]
> File-local helper (anonymous namespace). Maps a serialized flatbuffer EValue
> to its runtime `Tag`. Returns `Result<Tag>`. `index` is used only for the
> error log message.
>
> Switch on `serialization_value->val_type()` (the flatbuffer `KernelTypes`
> discriminator):
> - `Null` → `Tag::None`
> - `Int` → `Tag::Int`
> - `Double` → `Tag::Double`
> - `Bool` → `Tag::Bool`
> - `String` → `Tag::String`
> - `Tensor` → `Tag::Tensor`
> - any other value → log an Error ("Invalid tag: <val_type> input idx:
>   <index>") and return `Error::Internal`.
>
> Note the mapping is only over this restricted set: list types and other
> EValue kinds are not representable here and fall to the `Internal` error.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta]
> class MethodMeta final {
>   const executorch_flatbuffer::ExecutionPlan* s_plan_;
> }

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn]
> Result<TensorInfo> MethodMeta::attribute_tensor_meta(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn]
> Returns `TensorInfo` for the `index`-th *attribute* tensor (0-based over the
> attribute-only enumeration), or an error. Attribute tensors are the subset of
> serialized values that are named constants (see
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn]`
> for the exact predicate).
>
> Algorithm:
> - Initialize `counter = 0`.
> - Iterate `i` over all serialized values `s_plan_->values()` in order:
>   - Skip values whose `val_type()` is not `Tensor`.
>   - For a `Tensor`, obtain `tensor_value = value->val_as_Tensor()`. Treat it
>     as an attribute only if `tensor_value != nullptr` AND
>     `extra_tensor_info() != nullptr` AND
>     `extra_tensor_info()->fully_qualified_name() != nullptr` AND that name's
>     `c_str() != nullptr`. Skip values that fail this predicate (do not
>     advance `counter`).
>   - When an attribute is found: if `counter == index`, this is the target:
>     - Validate `tensor_value->sizes() != nullptr` AND
>       `tensor_value->dim_order() != nullptr`; on failure return
>       `Error::InvalidProgram` ("Null tensor metadata for attribute <index>").
>     - Build a `TensorInfo` via
>       `[spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]`
>       with: `sizes` from `tensor_value->sizes()` (data ptr + size),
>       `dim_order` from `tensor_value->dim_order()`, `scalar_type` cast from
>       `tensor_value->scalar_type()`, `is_memory_planned = (allocation_info()
>       != nullptr) || (data_buffer_idx() != 0)`, and `name` set to the
>       attribute's fully-qualified name (`c_str()` + `size()`). Return that
>       result (propagating any error from `create`).
>     - Otherwise increment `counter` and continue.
> - If the loop completes without reaching `index`, log an Error ("No attribute
>   tensor found at index <index>") and return `Error::InvalidArgument`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.get-backend-name-fn]
> Result<const char*> MethodMeta::get_backend_name(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.get-backend-name-fn]
> Returns the backend id string at position `index` in this method's delegate
> list, or an error.
>
> - Compute `count = num_backends()` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn]`).
> - Validate `index < count`; on failure return `Error::InvalidArgument`
>   ("Index <index> out of range. num_backends: <count>").
> - Return `s_plan_->delegates()->Get(index)->id()->c_str()`, a NUL-terminated
>   C string that borrows from the underlying Program (valid only while the
>   Program outlives the caller).

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]
> Result<Tag> MethodMeta::input_tag(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]
> Returns the `Tag` of the `index`-th method input, or an error.
>
> - Compute `num_inputs = this->num_inputs()`.
> - Validate `index < num_inputs`; on failure return `Error::InvalidArgument`
>   ("index <index> out of range. num_inputs: <num_inputs>").
> - Look up the internal value index: `input_index =
>   s_plan_->inputs()->Get(index)` (a signed integer into the values table).
> - Let `num_values = s_plan_->values()->size()`. Validate `input_index >= 0 &&
>   (size_t)input_index < num_values`; on failure return `Error::InvalidProgram`
>   ("internal value index ... out of range ...").
> - Fetch `serialization_value = s_plan_->values()->Get(input_index)` and return
>   `get_tag(serialization_value, index)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.get-tag-fn]`),
>   which yields one of `Tensor`, `Int`, `Bool`, `Double`, `String`, `None`, or
>   `Error::Internal` for an unrepresentable value type.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn]
> Result<TensorInfo> MethodMeta::input_tensor_meta(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn]
> Returns `TensorInfo` for the `index`-th input, valid only when that input is
> a tensor. Returns an error otherwise.
>
> - Call `input_tag(index)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]`).
>   If it is an error, propagate that error unchanged (this also covers the
>   out-of-range and invalid-program cases).
> - Validate the tag equals `Tag::Tensor`; otherwise return
>   `Error::InvalidArgument` ("Tag: <tag> input: <index> is not Tensor").
> - Re-resolve `input_index = s_plan_->inputs()->Get(index)` (already validated
>   in range by `input_tag`) and fetch `tensor_value =
>   s_plan_->values()->Get(input_index)->val_as_Tensor()`.
> - Validate `tensor_value != nullptr && tensor_value->sizes() != nullptr &&
>   tensor_value->dim_order() != nullptr`; on failure return
>   `Error::InvalidProgram` ("Null tensor metadata for input <index>").
> - Return `TensorInfo::create(...)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]`)
>   with: `sizes` = span over `tensor_value->sizes()` (data + size), `dim_order`
>   = span over `tensor_value->dim_order()`, `scalar_type` cast from
>   `tensor_value->scalar_type()`, `is_memory_planned = (allocation_info() !=
>   nullptr) || (data_buffer_idx() != 0)` (constant returns count as memory
>   planned), and `name` = the empty string_view `{nullptr, 0}`. Any error from
>   `create` (e.g. nbytes overflow) is propagated.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn]
> Result<etensor::Device> MethodMeta::memory_planned_buffer_device( size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn]
> Returns the `etensor::Device` on which the `index`-th memory-planned buffer
> should be allocated (0-based, same indexing as
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]`).
>
> - Compute `num_buffers = this->num_memory_planned_buffers()`. Validate `index
>   < num_buffers`; on failure return `Error::InvalidArgument` ("index <index>
>   out of range. num_buffers: <num_buffers>").
> - Fetch `buffer_devices = s_plan_->non_const_buffer_device()`. This field is
>   optional: if it is `nullptr` (CPU-only or legacy PTE), return `Device{CPU,
>   0}`.
> - Otherwise the list is sparse and contains entries only for non-CPU buffers.
>   Compute `internal_idx = (int32_t)(index + 1)` (buffer indices are 1-based
>   internally with slot 0 reserved). Iterate the entries in order; if an
>   entry's `buffer_idx()` equals `internal_idx`, return `Device{
>   (DeviceType)entry->device_type(), (DeviceIndex)entry->device_index() }`.
> - If no matching entry is found, the buffer is on CPU: return `Device{CPU,
>   0}`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]
> Result<int64_t> MethodMeta::memory_planned_buffer_size(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]
> Returns the size in bytes of the `index`-th memory-planned buffer, or an
> error.
>
> - Compute `num_buffers = this->num_memory_planned_buffers()`. Validate `index
>   < num_buffers`; on failure return `Error::InvalidArgument` ("index <index>
>   out of range. num_buffers: <num_buffers>").
> - Buffer slot 0 in `non_const_buffer_sizes()` is reserved and hidden from
>   users, so read the raw size at the shifted index:
>   `size = s_plan_->non_const_buffer_sizes()->Get(index + 1)` (as `int64_t`).
> - Validate `size >= 0`; on failure return `Error::InvalidProgram`
>   ("memory_planned_buffer_size(<index>) has invalid negative size: <size>").
> - Return `size`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.method-meta-fn]
> MethodMeta::MethodMeta(const executorch_flatbuffer::ExecutionPlan* s_plan)

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.method-meta-fn]
> Private explicit constructor (only `Program` may call it). Stores the borrowed
> pointer `s_plan` into the member `s_plan_`. Performs no copying, validation, or
> allocation; the referenced `ExecutionPlan` (and its owning Program) must
> outlive the `MethodMeta`. `s_plan_` is the sole source of truth for all
> queries.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.name-fn]
> const char* MethodMeta::name() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.name-fn]
> Returns the method's name as a NUL-terminated C string:
> `s_plan_->name()->c_str()`. The string borrows from the underlying Program and
> is valid only while the Program outlives the caller. No validation; assumes
> `name()` is non-null (guaranteed by a well-formed program).

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.non-const-buffer-size-fn]
> Result<int64_t> non_const_buffer_size(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.non-const-buffer-size-fn]
> Deprecated alias. Inline header method that forwards directly to
> `memory_planned_buffer_size(index)` (see
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-size-fn]`)
> and returns its `Result<int64_t>` unchanged. No additional behavior.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn]
> size_t MethodMeta::num_attributes() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-attributes-fn]
> Returns the count of attribute tensors (named constants) in the method.
>
> - Initialize `counter = 0`.
> - Iterate `i` over all serialized values `s_plan_->values()` in order.
> - For each value whose `val_type()` is `Tensor`, get `tensor_value =
>   val_as_Tensor()` and increment `counter` iff the predicate holds:
>   `tensor_value != nullptr` AND `extra_tensor_info() != nullptr` AND
>   `extra_tensor_info()->fully_qualified_name() != nullptr` AND that name's
>   `c_str() != nullptr` (i.e. the tensor carries a non-null fully-qualified
>   name). Non-tensor values and tensors failing the predicate are not counted.
> - Return `counter`. This is the exact count that indexes
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.attribute-tensor-meta-fn]`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn]
> size_t MethodMeta::num_backends() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-backends-fn]
> Returns the number of delegate/backend entries in this method. Fetch
> `delegates = s_plan_->delegates()`; if it is non-null return
> `delegates->size()`, otherwise (null delegates vector) return `0`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-inputs-fn]
> size_t MethodMeta::num_inputs() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-inputs-fn]
> Returns the number of method inputs: `s_plan_->inputs()->size()`. No
> validation; assumes the `inputs()` vector is non-null (guaranteed by a
> well-formed program).

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-instructions-fn]
> size_t MethodMeta::num_instructions() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-instructions-fn]
> Returns the total instruction count across all chains of the method.
>
> - Fetch `chains = s_plan_->chains()`; if `nullptr` return `0`.
> - Initialize an accumulator `num_instructions = 0`.
> - Iterate `i` over `chains->size()`:
>   - Get `s_chain = chains->Get(i)`; if `nullptr`, skip.
>   - Get `s_instructions = s_chain->instructions()`; if non-null, add
>     `s_instructions->size()` to the accumulator.
> - Return the accumulated sum.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn]
> size_t MethodMeta::num_memory_planned_buffers() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn]
> Returns the number of user-visible memory-planned buffers.
>
> - If `s_plan_->non_const_buffer_sizes() == nullptr`, return `0`.
> - Otherwise let `size = non_const_buffer_sizes()->size()`. Slot 0 is reserved
>   internally and hidden from users, so return `size - 1` when `size > 0`, else
>   `0`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-non-const-buffers-fn]
> ET_DEPRECATED size_t num_non_const_buffers() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-non-const-buffers-fn]
> Deprecated alias. Inline header method that forwards directly to
> `num_memory_planned_buffers()` (see
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-memory-planned-buffers-fn]`)
> and returns its `size_t` result unchanged. No additional behavior.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.num-outputs-fn]
> size_t MethodMeta::num_outputs() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.num-outputs-fn]
> Returns the number of method outputs: `s_plan_->outputs()->size()`. No
> validation; assumes the `outputs()` vector is non-null (guaranteed by a
> well-formed program).

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.output-tag-fn]
> Result<Tag> MethodMeta::output_tag(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tag-fn]
> Returns the `Tag` of the `index`-th method output, or an error. Identical in
> structure to
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tag-fn]`
> but over the outputs vector.
>
> - Compute `num_outputs = this->num_outputs()`. Validate `index < num_outputs`;
>   on failure return `Error::InvalidArgument` ("index <index> out of range.
>   num_outputs: <num_outputs>").
> - Let `output_index = s_plan_->outputs()->Get(index)` and `num_values =
>   s_plan_->values()->size()`. Validate `output_index >= 0 &&
>   (size_t)output_index < num_values`; on failure return
>   `Error::InvalidProgram`.
> - Return `get_tag(s_plan_->values()->Get(output_index), index)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.get-tag-fn]`).

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.output-tensor-meta-fn]
> Result<TensorInfo> MethodMeta::output_tensor_meta(size_t index) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.output-tensor-meta-fn]
> Returns `TensorInfo` for the `index`-th output, valid only when that output is
> a tensor. Structurally identical to
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.input-tensor-meta-fn]`
> but over outputs.
>
> - Call `output_tag(index)`; propagate any error unchanged.
> - Validate the tag equals `Tag::Tensor`; otherwise return
>   `Error::InvalidArgument` ("Tag: <tag> output: <index> is not Tensor").
> - Resolve `output_index = s_plan_->outputs()->Get(index)` and `tensor_value =
>   s_plan_->values()->Get(output_index)->val_as_Tensor()`.
> - Validate `tensor_value != nullptr && tensor_value->sizes() != nullptr &&
>   tensor_value->dim_order() != nullptr`; on failure return
>   `Error::InvalidProgram` ("Null tensor metadata for output <index>").
> - Return `TensorInfo::create(...)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]`)
>   with `sizes`/`dim_order` spans from the tensor, `scalar_type` cast from
>   `tensor_value->scalar_type()`, `is_memory_planned = (allocation_info() !=
>   nullptr) || (data_buffer_idx() != 0)`, and empty `name` `{nullptr, 0}`.
>   Errors from `create` are propagated.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.uses-backend-fn]
> bool MethodMeta::uses_backend(const char* backend_name) const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.uses-backend-fn]
> Returns `true` iff a delegate with the exact id `backend_name` is present in
> this method.
>
> - `backend_name` must be non-null: on null it triggers `ET_CHECK_MSG`
>   ("backend name is null"), which aborts (fatal in debug/enabled builds). A
>   Rust port should treat a null/absent name as a precondition violation.
> - Fetch `delegates = s_plan_->delegates()` and iterate `i` over
>   `delegates->size()`:
>   - Let `delegate = delegates->Get(i)`. Compare by length first:
>     `backend_name_len = strlen(backend_name)` vs `delegate_id_len =
>     delegate->id()->size()`. If lengths are equal AND the first
>     `backend_name_len` bytes match (`strncmp(delegate->id()->c_str(),
>     backend_name, backend_name_len) == 0`), return `true`.
> - If no delegate matches, return `false`. (Comparison is exact, byte-for-byte;
>   the flatbuffer id length is used rather than relying on NUL termination of
>   the stored id.)

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info]
> class TensorInfo final {
>   Span<const int32_t> sizes_;
>   Span<const uint8_t> dim_order_;
>   std::string_view name_;
>   executorch::aten::ScalarType scalar_type_;
>   bool is_memory_planned_;
>   size_t nbytes_;
> }

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]
> Result<TensorInfo> TensorInfo::create( Span<const int32_t> sizes, Span<const uint8_t> dim_order, executorch::aten::ScalarType scalar_type, const bool is_memory_planned, std::string_view name)

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.create-fn]
> Static factory that builds a `TensorInfo`, validating that its byte size is
> computable. Returns `Result<TensorInfo>`.
>
> - Compute `nbytes = calculate_nbytes(sizes, scalar_type)` (see
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]`).
> - Validate `nbytes.ok()`; on failure return `Error::InvalidArgument` ("Failed
>   to calculate nbytes for TensorInfo"). Note: the original error from
>   `calculate_nbytes` (which may itself be `InvalidProgram` for a negative size
>   or `InvalidArgument` for overflow) is discarded and replaced by
>   `InvalidArgument` here.
> - On success construct and return a `TensorInfo` via
>   `[spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn]`
>   with the passed-through `sizes`, `dim_order`, `scalar_type`,
>   `is_memory_planned`, `name`, and the computed `nbytes.get()`.
> - The `sizes`, `dim_order`, and `name` spans/views are stored by reference
>   (not copied); the underlying Program memory must outlive the `TensorInfo`.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.dim-order-fn]
> Span<const uint8_t> TensorInfo::dim_order() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.dim-order-fn]
> Const accessor. Returns the stored `dim_order_` member unchanged: a
> `Span<const uint8_t>` borrowing from the underlying Program (valid only while
> the Program outlives the `TensorInfo`). No copying or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.is-memory-planned-fn]
> bool TensorInfo::is_memory_planned() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.is-memory-planned-fn]
> Const accessor. Returns the stored `is_memory_planned_` boolean member
> unchanged (whether the tensor's memory was planned during export). No copying
> or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.name-fn]
> std::string_view TensorInfo::name() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.name-fn]
> Const accessor. Returns the stored `name_` member unchanged: a
> `std::string_view` holding the fully-qualified tensor name, which may be empty
> (`{nullptr, 0}`) when the tensor is nameless (e.g. inputs/outputs). Borrows
> from the underlying Program; no copying or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.nbytes-fn]
> size_t TensorInfo::nbytes() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.nbytes-fn]
> Const accessor. Returns the stored `nbytes_` member unchanged: the total size
> in bytes of the tensor (product of sizes times element size), as computed at
> construction time by
> `[spec:et:sem:method-meta.executorch.et-runtime-namespace.calculate-nbytes-fn]`.
> No recomputation or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.scalar-type-fn]
> executorch::aten::ScalarType TensorInfo::scalar_type() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.scalar-type-fn]
> Const accessor. Returns the stored `scalar_type_` member unchanged (an
> `executorch::aten::ScalarType`). No copying or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.sizes-fn]
> Span<const int32_t> TensorInfo::sizes() const

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.sizes-fn]
> Const accessor. Returns the stored `sizes_` member unchanged: a
> `Span<const int32_t>` borrowing from the underlying Program (valid only while
> the Program outlives the `TensorInfo`). No copying or validation.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn]
> TensorInfo::TensorInfo( Span<const int32_t> sizes, Span<const uint8_t> dim_order, executorch::aten::ScalarType scalar_type, const bool is_memory_planned, std::string_view name, size_t nbytes) : sizes_(sizes), dim_order_(dim_order), name_...

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.tensor-info-fn]
> Private all-fields constructor (only `MethodMeta`, `TensorInfo::create`, and
> the test friend may call it; there is no public/default constructor —
> `TensorInfo() = delete`). Directly stores each argument into the corresponding
> member via the member-initializer list, in declaration order:
> `sizes_ = sizes`, `dim_order_ = dim_order`, `name_ = name`,
> `scalar_type_ = scalar_type`, `is_memory_planned_ = is_memory_planned`,
> `nbytes_ = nbytes`. Performs no validation, computation, or copying of the
> referenced buffers; the spans/string_view continue to borrow from the Program.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.method-meta.operator-fn]
> MethodMeta& operator=(const MethodMeta&) = default

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.operator-fn]
> Compiler-defaulted copy-assignment operator (`= default`). Member-wise copies
> the single member `s_plan_` (a borrowed raw pointer) from the source into
> `*this` and returns `*this`. This is a shallow pointer copy: both objects then
> reference the same `ExecutionPlan`; no deep copy, allocation, or validation. A
> defaulted move-assignment operator also exists (behaves identically for a raw
> pointer member). `MethodMeta` is trivially copyable.

> [spec:et:def:method-meta.executorch.et-runtime-namespace.tensor-info.operator-fn]
> TensorInfo& operator=(const TensorInfo&) = default

> [spec:et:sem:method-meta.executorch.et-runtime-namespace.tensor-info.operator-fn]
> Compiler-defaulted copy-assignment operator (`= default`). Member-wise copies
> every member from the source into `*this` and returns `*this`: `sizes_`,
> `dim_order_` (span copies — pointer + length, sharing the underlying Program
> buffers), `name_` (string_view copy), `scalar_type_`, `is_memory_planned_`,
> and `nbytes_`. Shallow copy of the borrowed spans/view; no deep copy,
> allocation, or validation. A defaulted move-assignment operator also exists
> and behaves equivalently. `TensorInfo` is trivially copyable.

