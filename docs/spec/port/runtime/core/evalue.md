# runtime/core/evalue.cpp, runtime/core/evalue.h

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list]
> class BoxedEvalueList {
>   executorch::aten::ArrayRef<EValue*> wrapped_vals_;
>   mutable T* unwrapped_vals_;
> }

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]
> executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]
> Full specialization of `BoxedEvalueList<T>::get()` for
> `T = std::optional<executorch::aten::Tensor>`. Materializes the unwrapped
> list of optional tensors from the wrapped EValue pointers and returns it as
> an ArrayRef.
>
> Behavior:
> - Let `n = wrapped_vals_.size()` (the size captured at construction).
> - Iterate `i` from 0 to `n-1` in ascending order. For each element:
>   - If `wrapped_vals_[i]` (the stored EValue pointer) is null, write
>     `executorch::aten::nullopt` (an empty optional) into
>     `unwrapped_vals_[i]`. A null pointer is a legal representation of
>     `std::nullopt` for this optional-tensor specialization (unlike the
>     non-optional `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]`,
>     where null is an error).
>   - Otherwise assign into `unwrapped_vals_[i]` the result of
>     `wrapped_vals_[i]->to<std::optional<executorch::aten::Tensor>>()`. That
>     `to<...>()` dispatches to `toOptional<Tensor>()`
>     (`[spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn]`):
>     a `None`-tagged EValue yields an empty optional, otherwise the EValue
>     must be tensor-tagged and its tensor is returned wrapped in the optional
>     (aborting via ET_CHECK if it is neither None nor Tensor).
> - `unwrapped_vals_` is the caller-supplied scratch buffer (mutable); it must
>   have at least `n` slots. Writes are in-place into that buffer.
> - Return an ArrayRef `{unwrapped_vals_, n}` pointing at the freshly written
>   buffer.
>
> Rematerialization rationale: the wrapped pointers point into the runtime
> values table, which may mutate between calls, so the unwrapped list is
> rebuilt from scratch on every call rather than cached.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]
> Result<executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>>

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]
> Full specialization of `BoxedEvalueList<T>::tryGet()` for
> `T = std::optional<executorch::aten::Tensor>`. Result-returning counterpart
> of the specialized `get()`
> (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]`)
> that never aborts on malformed data.
>
> Behavior:
> - Let `n = wrapped_vals_.size()`.
> - Iterate `i` from 0 to `n-1` in ascending order. For each element:
>   - If `wrapped_vals_[i]` is null, write `std::nullopt` into
>     `unwrapped_vals_[i]` and continue (null is a valid empty optional here).
>   - Otherwise call
>     `wrapped_vals_[i]->tryToOptional<executorch::aten::Tensor>()`
>     (`[spec:et:sem:evalue.executorch.runtime.e-value.try-to-optional-fn]`).
>     If that Result is not ok, return its error immediately (short-circuit;
>     remaining elements are not processed and the buffer is left partially
>     written). Errors surface as `Error::InvalidType` when the referenced
>     EValue is neither None nor Tensor.
>   - On success move the produced optional into `unwrapped_vals_[i]`.
> - On completing all elements, return an ArrayRef `{unwrapped_vals_, n}`.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]
> executorch::aten::ArrayRef<T> BoxedEvalueList<T>::get() const

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]
> Generic (non-optional) template definition of `BoxedEvalueList<T>::get()`,
> used for `T = int64_t` and `T = executorch::aten::Tensor`. Materializes the
> unwrapped list from the wrapped EValue pointers and returns it as an ArrayRef.
>
> Behavior:
> - Let `n = wrapped_vals_.size()`.
> - Iterate `i` from 0 to `n-1` in ascending order. For each element:
>   - ET_CHECK that `wrapped_vals_[i] != nullptr`; if it is null the check
>     aborts the process (this is the abort-on-failure path, contrasted with
>     `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]`).
>     Unlike the optional-tensor specialization, null is never valid here.
>   - Assign `unwrapped_vals_[i] = wrapped_vals_[i]->to<T>()`, which dispatches
>     by the EValue tag to the matching accessor (e.g. `toInt()`/`toTensor()`)
>     and aborts via ET_CHECK if the tag does not match `T`.
> - Writes are in-place into the caller-supplied `unwrapped_vals_` buffer
>   (must hold at least `n` slots).
> - Return an ArrayRef `{unwrapped_vals_, n}`.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]
> Result<executorch::aten::ArrayRef<T>> BoxedEvalueList<T>::tryGet() const

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]
> Generic (non-optional) template definition of `BoxedEvalueList<T>::tryGet()`,
> used for `T = int64_t` and `T = executorch::aten::Tensor`. Result-returning
> counterpart of `get()`
> (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]`) that
> validates each element and never aborts.
>
> Behavior:
> - Let `n = wrapped_vals_.size()`.
> - Iterate `i` from 0 to `n-1` in ascending order. For each element:
>   - If `wrapped_vals_[i]` is null, return `Error::InvalidState` immediately
>     (null pointer is invalid for the non-optional case).
>   - Otherwise call `wrapped_vals_[i]->tryTo<T>()`
>     (`[spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn]`). If that
>     Result is not ok, return its error immediately (short-circuit; typically
>     `Error::InvalidType` on tag mismatch). Remaining elements are not
>     processed; the buffer is left partially written.
>   - On success move the produced value into `unwrapped_vals_[i]`.
> - On completing all elements, return an ArrayRef `{unwrapped_vals_, n}`.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.boxed-evalue-list-fn]
> BoxedEvalueList(EValue** wrapped_vals, T* unwrapped_vals, int size)

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.boxed-evalue-list-fn]
> Constructs a BoxedEvalueList that correlates a table of EValue pointers with
> a same-sized scratch buffer for the unwrapped values.
>
> Behavior:
> - Validate `wrapped_vals` via `checkWrappedVals(wrapped_vals, size)`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn]`):
>   aborts if `wrapped_vals` is null or `size < 0`. Store the returned pointer
>   together with `size` into `wrapped_vals_` as an
>   `ArrayRef<EValue*>{wrapped_vals, size}`. This ArrayRef captures the size
>   as the source of truth for the list length; the pointers are not
>   dereferenced here (dereferencing happens lazily in `get()`/`tryGet()`).
> - Validate `unwrapped_vals` via `checkUnwrappedVals(unwrapped_vals)`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn]`):
>   aborts if `unwrapped_vals` is null. Store the returned pointer into the
>   mutable `unwrapped_vals_` field.
> - `unwrapped_vals` must point at storage of at least `size` elements of `T`;
>   it serves as scratch memory in which `get()`/`tryGet()` construct the
>   unwrapped values on demand.
> - Member initialization order is `wrapped_vals_` then `unwrapped_vals_`.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn]
> static T* checkUnwrappedVals(T* unwrapped_vals)

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-unwrapped-vals-fn]
> Static validation helper for the `unwrapped_vals` constructor argument.
>
> Behavior:
> - ET_CHECK_MSG that `unwrapped_vals != nullptr` with message
>   "unwrapped_vals cannot be null"; on failure aborts the process.
> - On success returns `unwrapped_vals` unchanged.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn]
> static EValue** checkWrappedVals(EValue** wrapped_vals, int size)

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.check-wrapped-vals-fn]
> Static validation helper for the `wrapped_vals` / `size` constructor
> arguments.
>
> Behavior:
> - ET_CHECK_MSG that `wrapped_vals != nullptr` with message "wrapped_vals
>   cannot be null"; on failure aborts the process.
> - ET_CHECK_MSG that `size >= 0` with message "size cannot be negative"; on
>   failure aborts the process.
> - On success returns `wrapped_vals` unchanged. (It does not validate the
>   individual pointers inside the array, only the outer pointer and size.)

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn]
> void destroy_elements()

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn]
> Destroys the previously-constructed unwrapped elements in the scratch buffer
> without touching the wrapped EValue pointers.
>
> Behavior:
> - Let `n = wrapped_vals_.size()`.
> - Iterate `i` from 0 to `n-1` in ascending order and explicitly invoke the
>   destructor `unwrapped_vals_[i].~T()` on each slot.
> - It does NOT dereference `wrapped_vals_[i]`; this is deliberate so it is
>   safe to call during EValue destruction even when the wrapped EValues have
>   been mutated by MoveCall instructions. The buffer memory itself is not
>   freed (ownership stays with whoever provided `unwrapped_vals`).
> - For trivially destructible `T` (e.g. `int64_t`) the loop is a no-op in
>   effect. It matters for `T = executorch::aten::Tensor` /
>   `std::optional<Tensor>`, where element destruction releases held tensor
>   handles.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.get-fn]
> executorch::aten::ArrayRef<T> get() const

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.get-fn]
> In-class declaration of `BoxedEvalueList<T>::get()`; the behavior is defined
> out-of-line. For non-optional `T` see
> `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]`; for
> `T = std::optional<Tensor>` see
> `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]`.
> Constructs and returns, as an `ArrayRef<T>`, the list of `T` values
> corresponding to the wrapped EValue pointers, rematerialized into
> `unwrapped_vals_` on every call.

> [spec:et:def:evalue.executorch.runtime.boxed-evalue-list.try-get-fn]
> Result<executorch::aten::ArrayRef<T>> tryGet() const

> [spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.try-get-fn]
> In-class declaration of `BoxedEvalueList<T>::tryGet()`; the behavior is
> defined out-of-line. Result-returning counterpart of `get()` that validates
> each wrapped EValue's tag before materializing: returns `Error::InvalidType`
> if any element's tag does not match `T`, and `Error::InvalidState` if any
> element pointer is null (except for the optional-tensor specialization where
> null means empty optional). Use when materializing lists from untrusted .pte
> data so a malformed program cannot force an abort inside `to<T>()`/ET_CHECK.
> For non-optional `T` see
> `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]`; for
> `T = std::optional<Tensor>` see
> `[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]`.

> [spec:et:def:evalue.executorch.runtime.e-value]
> struct EValue {
>   union Payload { // When in ATen mode at::Tensor is not trivially copyable, this nested union // lets us handle tensor as a special case while leaving the res...;
>   Payload payload;
>   Tag tag;
> }

> [spec:et:def:evalue.executorch.runtime.e-value.clear-to-none-fn]
> void clearToNone() noexcept

> [spec:et:sem:evalue.executorch.runtime.e-value.clear-to-none-fn]
> Private helper that resets this EValue to the `None` state. Precondition:
> any non-trivial payload value (e.g. a stored tensor) has already had its
> destructor called; this function does not destroy anything.
>
> Behavior:
> - Set `payload.copyable_union.as_int = 0` (zeroing the trivially-copyable
>   payload; writing the int member is sufficient because all copyable members
>   overlap in the union).
> - Set `tag = Tag::None`.
> - noexcept.

> [spec:et:def:evalue.executorch.runtime.e-value.destroy-fn]
> void destroy()

> [spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn]
> Private helper that releases any non-trivial resource held by this EValue's
> payload, based on its tag. Called by the destructor and by move-assignment
> before overwriting.
>
> Behavior (mutually exclusive branches):
> - If `isTensor()` (tag == Tensor): explicitly call
>   `payload.as_tensor.~Tensor()`. In ATen mode this decrements the intrusive
>   refcount on the TensorImpl that was incremented when the tensor was placed
>   in the EValue. In lean (ExecuTorch) mode Tensor destruction is effectively
>   a no-op.
> - Else if `isTensorList()` (tag == ListTensor) AND
>   `payload.copyable_union.as_tensor_list_ptr != nullptr`: call
>   `as_tensor_list_ptr->destroy_elements()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list.destroy-elements-fn]`),
>   destroying the unwrapped tensor elements in the list's scratch buffer
>   without dereferencing the wrapped pointers.
> - Else if `isListOptionalTensor()` (tag == ListOptionalTensor) AND
>   `payload.copyable_union.as_list_optional_tensor_ptr != nullptr`: call
>   `as_list_optional_tensor_ptr->destroy_elements()` similarly.
> - For all other tags (None, Int, Double, Bool, String, ListInt, ListBool,
>   ListDouble) the payload is trivially destructible and nothing is done.
> - It does not free the BoxedEvalueList objects or the ArrayRef targets
>   themselves (those are owned externally, typically in program memory); it
>   only runs element destructors on list scratch buffers.

> [spec:et:def:evalue.executorch.runtime.e-value.e-value-fn]
> EValue(executorch::aten::Scalar s)

> [spec:et:sem:evalue.executorch.runtime.e-value.e-value-fn]
> Implicit constructor building an EValue from an `executorch::aten::Scalar`,
> collapsing the Scalar onto one of the three primitive tags.
>
> Behavior (checked in this order):
> - If `s.isIntegral(false)` (integral, excluding bool): set `tag = Tag::Int`
>   and `payload.copyable_union.as_int = s.to<int64_t>()`.
> - Else if `s.isFloatingPoint()`: set `tag = Tag::Double` and
>   `payload.copyable_union.as_double = s.to<double>()`.
> - Else if `s.isBoolean()`: set `tag = Tag::Bool` and
>   `payload.copyable_union.as_bool = s.to<bool>()`.
> - Else: ET_CHECK_MSG(false, "Scalar passed to EValue is not initialized.")
>   which aborts the process (an uninitialized/untagged Scalar).
> - `isIntegral(false)` treats bool as non-integral, so booleans fall through
>   to the isBoolean branch and are stored as Tag::Bool, not Tag::Int.

> [spec:et:def:evalue.executorch.runtime.e-value.is-bool-fn]
> bool isBool() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::Bool`, i.e. the
> payload's active member is `copyable_union.as_bool`. Every other tag
> (None/Int/Double/String, all list tags, Tensor) yields false. Reads only
> `tag`; does not touch the payload and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-bool-list-fn]
> bool isBoolList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-bool-list-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::ListBool`, i.e. the
> payload's active member is `copyable_union.as_bool_list_ptr` (a pointer to a
> directly-stored `ArrayRef<bool>`). Every other tag yields false. Reads only
> `tag`; does not dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-double-fn]
> bool isDouble() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-double-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::Double`, i.e. the
> payload's active member is `copyable_union.as_double`. Every other tag yields
> false. Reads only `tag`; does not touch the payload and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-double-list-fn]
> bool isDoubleList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-double-list-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::ListDouble`, i.e. the
> payload's active member is `copyable_union.as_double_list_ptr` (a pointer to a
> directly-stored `ArrayRef<double>`). Every other tag yields false. Reads only
> `tag`; does not dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-int-fn]
> bool isInt() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-int-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::Int`, i.e. the
> payload's active member is `copyable_union.as_int`. The Int tag also backs the
> ScalarType/MemoryFormat/Layout/Device accessors, which reinterpret the same
> stored int. Every other tag yields false. Reads only `tag`; does not touch the
> payload and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-int-list-fn]
> bool isIntList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-int-list-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::ListInt`, i.e. the
> payload's active member is `copyable_union.as_int_list_ptr` (a pointer to a
> `BoxedEvalueList<int64_t>`). Every other tag yields false. Reads only `tag`;
> does not dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-list-optional-tensor-fn]
> bool isListOptionalTensor() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-list-optional-tensor-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::ListOptionalTensor`,
> i.e. the payload's active member is
> `copyable_union.as_list_optional_tensor_ptr` (a pointer to a
> `BoxedEvalueList<std::optional<Tensor>>`). Every other tag yields false. Reads
> only `tag`; does not dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-none-fn]
> bool isNone() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-none-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::None`, the
> default-constructed/uninitialized state whose payload is a zeroed
> `copyable_union` (`as_int == 0`). Every other tag yields false. Used by
> `[spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn]` and
> `[spec:et:sem:evalue.executorch.runtime.e-value.try-to-optional-fn]` to map
> None to an empty optional. Reads only `tag`; never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-scalar-fn]
> bool isScalar() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-scalar-fn]
> Returns true iff the tag is one of the three scalar-carrying tags:
> `tag == Tag::Int || tag == Tag::Double || tag == Tag::Bool`. Any other tag
> (including None and all list/tensor/string tags) yields false.

> [spec:et:def:evalue.executorch.runtime.e-value.is-string-fn]
> bool isString() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-string-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::String`, i.e. the
> payload's active member is `copyable_union.as_string_ptr` (a pointer to an
> `ArrayRef<char>`). Every other tag yields false. Reads only `tag`; does not
> dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.is-tensor-fn]
> bool isTensor() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-tensor-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::Tensor`, the one tag
> whose active payload member is `payload.as_tensor` (the non-trivial Tensor in
> the outer union) rather than a member of `copyable_union`. Every other tag
> yields false. Reads only `tag`; does not touch the tensor and never aborts.
> This predicate gates the tensor-specific move/destroy paths in
> `[spec:et:sem:evalue.executorch.runtime.e-value.move-from-fn]` and
> `[spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn]`.

> [spec:et:def:evalue.executorch.runtime.e-value.is-tensor-list-fn]
> bool isTensorList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.is-tensor-list-fn]
> Pure const tag predicate. Returns true iff `tag == Tag::ListTensor`, i.e. the
> payload's active member is `copyable_union.as_tensor_list_ptr` (a pointer to a
> `BoxedEvalueList<Tensor>`). Every other tag yields false. Reads only `tag`;
> does not dereference the pointer and never aborts.

> [spec:et:def:evalue.executorch.runtime.e-value.move-from-fn]
> void moveFrom(EValue&& rhs) noexcept

> [spec:et:sem:evalue.executorch.runtime.e-value.move-from-fn]
> Private helper implementing the shared move logic. Moves the payload and tag
> out of `rhs` into `*this`, then resets `rhs` to None. Precondition: `*this`
> holds no live non-trivial payload (either freshly constructed, or `destroy()`
> was already called), since it constructs into the payload without first
> destroying it.
>
> Behavior:
> - If `rhs.isTensor()` (rhs tag == Tensor):
>   - Placement-new `payload.as_tensor` as
>     `executorch::aten::Tensor(std::move(rhs.payload.as_tensor))`
>     (move-constructing the tensor into this union field; in ATen mode this
>     moves the intrusive_ptr with no net refcount change).
>   - Explicitly destroy the source: `rhs.payload.as_tensor.~Tensor()`.
> - Else (rhs holds a trivially-copyable payload):
>   - Copy the whole `copyable_union`:
>     `payload.copyable_union = rhs.payload.copyable_union`. This copies
>     whichever scalar or pointer member is active (int/double/bool or one of
>     the ArrayRef*/BoxedEvalueList* pointers).
> - Set `tag = rhs.tag`.
> - Call `rhs.clearToNone()`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.clear-to-none-fn]`),
>   leaving `rhs` as a valid `None` EValue (zeroed payload, Tag::None). For
>   pointer-carrying tags this means both source and destination briefly
>   referenced the same list/string pointer, but the source is immediately
>   reset so ownership is not double-tracked; the pointed-to list objects are
>   not owned by the EValue.
> - noexcept.

> [spec:et:def:evalue.executorch.runtime.e-value.payload]
> union Payload {
>   union TriviallyCopyablePayload { TriviallyCopyablePayload() : as_int(0) {} // Scalar supported through these 3 types int64_t as_int; double as_double; bool a...;
>   executorch::aten::Tensor as_tensor;
> }

> [spec:et:def:evalue.executorch.runtime.e-value.payload.payload-fn]
> Payload()

> [spec:et:sem:evalue.executorch.runtime.e-value.payload.payload-fn]
> Default constructor of the `Payload` union. Empty body `{}`: it constructs
> no union member and initializes nothing. Because the union contains a
> non-trivial member (`as_tensor`, a Tensor in ATen mode), the union needs a
> user-provided constructor/destructor; this ctor leaves the union in an
> indeterminate state that the enclosing EValue constructor is responsible for
> initializing (e.g. by writing `copyable_union` or placement-newing
> `as_tensor`). The paired destructor `~Payload()` is likewise empty; actual
> teardown is done by `EValue::destroy()`
> (`[spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn]`).

> [spec:et:def:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload]
> union TriviallyCopyablePayload {
>   int64_t as_int;
>   double as_double;
>   bool as_bool;
>   executorch::aten::ArrayRef<char>* as_string_ptr;
>   executorch::aten::ArrayRef<double>* as_double_list_ptr;
>   executorch::aten::ArrayRef<bool>* as_bool_list_ptr;
>   BoxedEvalueList<int64_t>* as_int_list_ptr;
>   BoxedEvalueList<executorch::aten::Tensor>* as_tensor_list_ptr;
>   BoxedEvalueList<std::optional<executorch::aten::Tensor>>* as_list_optional_tensor_ptr;
> }

> [spec:et:def:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload.trivially-copyable-payload-fn]
> TriviallyCopyablePayload() : as_int(0)

> [spec:et:sem:evalue.executorch.runtime.e-value.payload.trivially-copyable-payload.trivially-copyable-payload-fn]
> Default constructor of the inner `TriviallyCopyablePayload` union. Activates
> the `as_int` member and initializes it to 0 (empty body). Since all members
> of this union overlap in storage, zeroing `as_int` zero-initializes the
> full-width (8-byte) representation, which also yields a null value for any
> of the pointer members. All members here are trivially copyable, so the
> union can be bit-copied via plain assignment.

> [spec:et:def:evalue.executorch.runtime.e-value.to-bool-fn]
> bool toBool() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-fn]
> Returns the stored bool.
>
> Behavior:
> - ET_CHECK_MSG(isBool(), "EValue is not a Bool."): aborts the process if the
>   tag is not Bool.
> - Returns `payload.copyable_union.as_bool`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-bool-list-fn]
> executorch::aten::ArrayRef<bool> toBoolList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-bool-list-fn]
> Returns the stored bool list as an ArrayRef.
>
> Behavior:
> - ET_CHECK_MSG(isBoolList(), "EValue is not a Bool List."): aborts if tag is
>   not ListBool.
> - ET_CHECK_MSG that `payload.copyable_union.as_bool_list_ptr != nullptr`
>   ("EValue bool list pointer is null."): aborts if null.
> - Returns `*(payload.copyable_union.as_bool_list_ptr)`, i.e. a copy of the
>   pointed-to `ArrayRef<bool>` (pointer + size; the underlying bool storage is
>   not copied). Unlike int/tensor lists there is no BoxedEvalueList
>   indirection: the bool list is stored directly as an ArrayRef.

> [spec:et:def:evalue.executorch.runtime.e-value.to-device-fn]
> executorch::aten::Device toDevice() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-device-fn]
> Interprets the stored int as a Device.
>
> Behavior:
> - ET_CHECK_MSG(isInt(), "EValue is not a Device."): aborts if tag is not Int
>   (Device is serialized as an int, so it shares the Int tag).
> - Constructs and returns
>   `executorch::aten::Device(static_cast<DeviceType>(payload.copyable_union.as_int), -1)`
>   — the stored int is cast to the DeviceType enum and the device index is
>   fixed at -1 (default/current device).

> [spec:et:def:evalue.executorch.runtime.e-value.to-double-fn]
> double toDouble() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-double-fn]
> Returns the stored double.
>
> Behavior:
> - ET_CHECK_MSG(isDouble(), "EValue is not a Double."): aborts if tag is not
>   Double.
> - Returns `payload.copyable_union.as_double`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-double-list-fn]
> executorch::aten::ArrayRef<double> toDoubleList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-double-list-fn]
> Returns the stored double list as an ArrayRef.
>
> Behavior:
> - ET_CHECK_MSG(isDoubleList(), "EValue is not a Double List."): aborts if tag
>   is not ListDouble.
> - ET_CHECK_MSG that `payload.copyable_union.as_double_list_ptr != nullptr`
>   ("EValue double list pointer is null."): aborts if null.
> - Returns `*(payload.copyable_union.as_double_list_ptr)`, a copy of the
>   pointed-to `ArrayRef<double>` (stored directly, no BoxedEvalueList
>   indirection).

> [spec:et:def:evalue.executorch.runtime.e-value.to-fn]
> T to() &&

> [spec:et:sem:evalue.executorch.runtime.e-value.to-fn]
> Templated typed accessor `to<T>()`. Declared for three ref-qualifier
> overloads (rvalue `&&`, const-lvalue `const&`, lvalue `&`); the bodies are
> generated by the `EVALUE_DEFINE_TO(T, method_name)` macro, which is
> explicitly instantiated for each supported `T`.
>
> Behavior of the generated specialization for a given `(T, method_name)`:
> - The rvalue overload (`to<T>() &&`) returns
>   `static_cast<T>(std::move(*this).method_name())`. This matters for Tensor:
>   `std::move(*this).toTensor()` moves the tensor out and resets this EValue
>   to None (see
>   `[spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-fn]`).
> - The const-lvalue overload (`to<T>() const&`) returns
>   `static_cast<return_type>(this->method_name())` where `return_type` is
>   `evalue_to_const_ref_overload_return<T>::type` — `const Tensor&` for
>   `T = Tensor`, otherwise plain `T`.
> - The lvalue overload (`to<T>() &`) returns
>   `static_cast<return_type>(this->method_name())` where `return_type` is
>   `evalue_to_ref_overload_return<T>::type` — `Tensor&` for `T = Tensor`,
>   otherwise plain `T`.
> - `method_name` is the corresponding non-templated accessor and carries that
>   accessor's ET_CHECK tag validation (aborting on mismatch). The mapping is:
>   Scalar→toScalar, int64_t→toInt, bool→toBool, double→toDouble,
>   string_view→toString, ScalarType→toScalarType, MemoryFormat→toMemoryFormat,
>   Layout→toLayout, Device→toDevice, optional<Tensor>→toOptional<Tensor>,
>   Tensor→toTensor, ArrayRef<int64_t>→toIntList,
>   optional<ArrayRef<int64_t>>→toOptional<ArrayRef<int64_t>>,
>   ArrayRef<double>→toDoubleList, optional variant→toOptional<...>,
>   ArrayRef<bool>→toBoolList, optional variant→toOptional<...>,
>   ArrayRef<Tensor>→toTensorList, optional variant→toOptional<...>,
>   ArrayRef<optional<Tensor>>→toListOptionalTensor.
> - There is no runtime `T`-parameter switch; the correct accessor is selected
>   at compile time by the explicit specialization.

> [spec:et:def:evalue.executorch.runtime.e-value.to-int-fn]
> int64_t toInt() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-int-fn]
> Returns the stored int.
>
> Behavior:
> - ET_CHECK_MSG(isInt(), "EValue is not an int."): aborts if tag is not Int.
> - Returns `payload.copyable_union.as_int`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-int-list-fn]
> executorch::aten::ArrayRef<int64_t> toIntList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-int-list-fn]
> Returns the stored int list as an ArrayRef, materializing it via the
> BoxedEvalueList.
>
> Behavior:
> - ET_CHECK_MSG(isIntList(), "EValue is not an Int List."): aborts if tag is
>   not ListInt.
> - ET_CHECK_MSG that `payload.copyable_union.as_int_list_ptr != nullptr`
>   ("EValue int list pointer is null."): aborts if null.
> - Returns `as_int_list_ptr->get()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]`),
>   which rematerializes the int values from the wrapped EValue pointers into
>   the list's scratch buffer (aborting if any wrapped element is null or not
>   an int) and returns an ArrayRef over that buffer.

> [spec:et:def:evalue.executorch.runtime.e-value.to-layout-fn]
> executorch::aten::Layout toLayout() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-layout-fn]
> Interprets the stored int as a Layout.
>
> Behavior:
> - ET_CHECK_MSG(isInt(), "EValue is not a Layout."): aborts if tag is not Int.
> - Returns `static_cast<executorch::aten::Layout>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn]
> executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>

> [spec:et:sem:evalue.executorch.runtime.e-value.to-list-optional-tensor-fn]
> Returns the stored list-of-optional-tensors as an ArrayRef, materializing it
> via the BoxedEvalueList.
>
> Behavior:
> - ET_CHECK_MSG(isListOptionalTensor(), "EValue is not a List Optional
>   Tensor."): aborts if tag is not ListOptionalTensor.
> - ET_CHECK_MSG that
>   `payload.copyable_union.as_list_optional_tensor_ptr != nullptr` ("EValue
>   list optional tensor pointer is null."): aborts if null.
> - Returns `as_list_optional_tensor_ptr->get()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.get-fn]`),
>   which rematerializes each optional tensor (null wrapped pointer → empty
>   optional) into the scratch buffer and returns an ArrayRef over it.

> [spec:et:def:evalue.executorch.runtime.e-value.to-memory-format-fn]
> executorch::aten::MemoryFormat toMemoryFormat() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-memory-format-fn]
> Interprets the stored int as a MemoryFormat.
>
> Behavior:
> - ET_CHECK_MSG(isInt(), "EValue is not a MemoryFormat."): aborts if tag is
>   not Int.
> - Returns
>   `static_cast<executorch::aten::MemoryFormat>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-optional-fn]
> inline std::optional<T> toOptional() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn]
> Templated conversion to `std::optional<T>`, representing both a value and the
> uninitialized (None) state.
>
> Behavior:
> - If `isNone()` (tag == None): return `executorch::aten::nullopt` (an empty
>   optional).
> - Otherwise return `this->to<T>()`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.to-fn]`), wrapping the
>   converted value into the optional. The underlying `to<T>()` aborts via
>   ET_CHECK if the tag does not match `T`. Note this is the const-lvalue
>   `to<T>()`; for `T = Tensor` it does not move out of the EValue.

> [spec:et:def:evalue.executorch.runtime.e-value.to-scalar-fn]
> executorch::aten::Scalar toScalar() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-scalar-fn]
> Reconstructs an `executorch::aten::Scalar` from the stored primitive.
>
> Behavior (checked in this order, using implicit Scalar constructors):
> - If `isDouble()`: return `toDouble()` (implicitly constructs a
>   floating-point Scalar).
> - Else if `isInt()`: return `toInt()` (implicitly constructs an integral
>   Scalar).
> - Else if `isBool()`: return `toBool()` (implicitly constructs a boolean
>   Scalar).
> - Else: ET_CHECK_MSG(false, "EValue is not a Scalar.") — aborts the process.
> - Ordering matters only in that each branch also revalidates via the
>   underlying accessor's ET_CHECK, but since the branch condition already
>   matches, those inner checks pass.

> [spec:et:def:evalue.executorch.runtime.e-value.to-scalar-type-fn]
> executorch::aten::ScalarType toScalarType() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-scalar-type-fn]
> Interprets the stored int as a ScalarType (dtype enum).
>
> Behavior:
> - ET_CHECK_MSG(isInt(), "EValue is not a ScalarType."): aborts if tag is not
>   Int.
> - Returns
>   `static_cast<executorch::aten::ScalarType>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.to-string-fn]
> std::string_view toString() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-string-fn]
> Returns the stored string as a `std::string_view`.
>
> Behavior:
> - ET_CHECK_MSG(isString(), "EValue is not a String."): aborts if tag is not
>   String.
> - ET_CHECK_MSG that `payload.copyable_union.as_string_ptr != nullptr`
>   ("EValue string pointer is null."): aborts if null.
> - Returns `std::string_view(as_string_ptr->data(), as_string_ptr->size())`,
>   i.e. a view over the pointed-to `ArrayRef<char>`'s data with its size (no
>   copy, no null-terminator assumption).

> [spec:et:def:evalue.executorch.runtime.e-value.to-tensor-fn]
> executorch::aten::Tensor toTensor() &&

> [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-fn]
> Rvalue-qualified accessor that moves the stored Tensor out of this EValue.
> (There are also `&` and `const&` overloads that return `Tensor&` /
> `const Tensor&` references without moving; this rule covers the `&&` move
> overload.)
>
> Behavior of the `&&` overload:
> - ET_CHECK_MSG(isTensor(), "EValue is not a Tensor."): aborts if tag is not
>   Tensor.
> - Move-construct `res = std::move(payload.as_tensor)`.
> - Call `clearToNone()`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.clear-to-none-fn]`),
>   resetting this EValue to None. Note the moved-from `payload.as_tensor` is
>   left in a valid but unspecified state; clearToNone overwrites the union
>   without calling the tensor destructor (the move already emptied it).
> - Return `res` by value (the caller receives ownership; in ATen mode this is
>   a moved intrusive_ptr with no net refcount change).
> - The `&`/`const&` overloads instead ET_CHECK isTensor() and return
>   `payload.as_tensor` by reference, leaving the EValue unchanged.

> [spec:et:def:evalue.executorch.runtime.e-value.to-tensor-list-fn]
> executorch::aten::ArrayRef<executorch::aten::Tensor> toTensorList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.to-tensor-list-fn]
> Returns the stored tensor list as an ArrayRef, materializing it via the
> BoxedEvalueList.
>
> Behavior:
> - ET_CHECK_MSG(isTensorList(), "EValue is not a Tensor List."): aborts if tag
>   is not ListTensor.
> - ET_CHECK_MSG that `payload.copyable_union.as_tensor_list_ptr != nullptr`
>   ("EValue tensor list pointer is null."): aborts if null.
> - Returns `as_tensor_list_ptr->get()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.get-fn]`),
>   which rematerializes the tensor values from the wrapped EValue pointers
>   into the list's scratch buffer (aborting if any wrapped element is null or
>   not a tensor) and returns an ArrayRef over that buffer.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-bool-fn]
> Result<bool> tryToBool() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-fn]
> Result-returning bool accessor (never aborts).
>
> Behavior:
> - If `!isBool()`: return `Error::InvalidType`.
> - Else return `payload.copyable_union.as_bool`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-bool-list-fn]
> Result<executorch::aten::ArrayRef<bool>> tryToBoolList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-bool-list-fn]
> Result-returning bool-list accessor (never aborts).
>
> Behavior:
> - If `!isBoolList()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_bool_list_ptr == nullptr`: return
>   `Error::InvalidState`.
> - Else return `*(payload.copyable_union.as_bool_list_ptr)` (copy of the
>   pointed-to `ArrayRef<bool>`).

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-device-fn]
> Result<executorch::aten::Device> tryToDevice() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-device-fn]
> Result-returning Device accessor (never aborts).
>
> Behavior:
> - If `!isInt()`: return `Error::InvalidType`.
> - Else return
>   `executorch::aten::Device(static_cast<DeviceType>(payload.copyable_union.as_int), -1)`
>   (device index fixed at -1), mirroring
>   `[spec:et:sem:evalue.executorch.runtime.e-value.to-device-fn]`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-double-fn]
> Result<double> tryToDouble() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-fn]
> Result-returning double accessor (never aborts).
>
> Behavior:
> - If `!isDouble()`: return `Error::InvalidType`.
> - Else return `payload.copyable_union.as_double`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-double-list-fn]
> Result<executorch::aten::ArrayRef<double>> tryToDoubleList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-double-list-fn]
> Result-returning double-list accessor (never aborts).
>
> Behavior:
> - If `!isDoubleList()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_double_list_ptr == nullptr`: return
>   `Error::InvalidState`.
> - Else return `*(payload.copyable_union.as_double_list_ptr)` (copy of the
>   pointed-to `ArrayRef<double>`).

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-fn]
> Result<T> tryTo() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn]
> Templated Result-returning accessor `tryTo<T>()`, the non-aborting
> equivalent of `to<T>()`. Bodies are generated by the
> `EVALUE_DEFINE_TRY_TO(T, method_name)` macro, explicitly instantiated per
> supported `T`.
>
> Behavior of the generated specialization for a given `(T, method_name)`:
> - Returns `this->method_name()`, i.e. delegates to the matching `tryTo*`
>   accessor which returns `Result<T>`. Tag mismatch surfaces as
>   `Error::InvalidType`; a null list/string payload surfaces as
>   `Error::InvalidState`. The `T → method_name` mapping is:
>   Scalar→tryToScalar, int64_t→tryToInt, bool→tryToBool, double→tryToDouble,
>   string_view→tryToString, ScalarType→tryToScalarType,
>   MemoryFormat→tryToMemoryFormat, Layout→tryToLayout, Device→tryToDevice,
>   Tensor→tryToTensor, optional<Tensor>→tryToOptional<Tensor>,
>   ArrayRef<int64_t>→tryToIntList, optional variant→tryToOptional<...>,
>   ArrayRef<double>→tryToDoubleList, optional variant→tryToOptional<...>,
>   ArrayRef<bool>→tryToBoolList, optional variant→tryToOptional<...>,
>   ArrayRef<Tensor>→tryToTensorList, optional variant→tryToOptional<...>,
>   ArrayRef<optional<Tensor>>→tryToListOptionalTensor.
> - Selection is at compile time via explicit specialization; no runtime
>   switch on `T`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-int-fn]
> Result<int64_t> tryToInt() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-fn]
> Result-returning int accessor (never aborts).
>
> Behavior:
> - If `!isInt()`: return `Error::InvalidType`.
> - Else return `payload.copyable_union.as_int`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-int-list-fn]
> Result<executorch::aten::ArrayRef<int64_t>> tryToIntList() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-int-list-fn]
> Result-returning int-list accessor (never aborts).
>
> Behavior:
> - If `!isIntList()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_int_list_ptr == nullptr`: return
>   `Error::InvalidState`.
> - Else return `as_int_list_ptr->tryGet()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]`),
>   which validates and materializes each element, propagating
>   `Error::InvalidState` (null wrapped pointer) or `Error::InvalidType` (wrong
>   element tag) instead of aborting.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-layout-fn]
> Result<executorch::aten::Layout> tryToLayout() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-layout-fn]
> Result-returning Layout accessor (never aborts).
>
> Behavior:
> - If `!isInt()`: return `Error::InvalidType`.
> - Else return
>   `static_cast<executorch::aten::Layout>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-list-optional-tensor-fn]
> Result<executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>>

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-list-optional-tensor-fn]
> Result-returning list-of-optional-tensors accessor (never aborts).
>
> Behavior:
> - If `!isListOptionalTensor()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_list_optional_tensor_ptr == nullptr`:
>   return `Error::InvalidState`.
> - Else return `as_list_optional_tensor_ptr->tryGet()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-std.optional-executorch.aten.tensor.try-get-fn]`),
>   which materializes each optional (null wrapped pointer → empty optional)
>   and propagates any element error instead of aborting.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-memory-format-fn]
> Result<executorch::aten::MemoryFormat> tryToMemoryFormat() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-memory-format-fn]
> Result-returning MemoryFormat accessor (never aborts).
>
> Behavior:
> - If `!isInt()`: return `Error::InvalidType`.
> - Else return
>   `static_cast<executorch::aten::MemoryFormat>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-optional-fn]
> inline Result<std::optional<T>> tryToOptional() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-optional-fn]
> Result-returning equivalent of `toOptional<T>()`
> (`[spec:et:sem:evalue.executorch.runtime.e-value.to-optional-fn]`).
>
> Behavior:
> - If `isNone()`: return `std::optional<T>(std::nullopt)` (an ok Result
>   wrapping an empty optional).
> - Otherwise call `this->tryTo<T>()`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.try-to-fn]`). If that
>   Result is not ok, return its error (e.g. `Error::InvalidType` on tag
>   mismatch). On success, return
>   `std::optional<T>(std::move(r.get()))` — an ok Result wrapping the moved
>   value.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-scalar-fn]
> Result<executorch::aten::Scalar> tryToScalar() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-fn]
> Result-returning Scalar accessor (never aborts).
>
> Behavior (checked in this order):
> - If `isDouble()`: return
>   `executorch::aten::Scalar(payload.copyable_union.as_double)`.
> - Else if `isInt()`: return
>   `executorch::aten::Scalar(payload.copyable_union.as_int)`.
> - Else if `isBool()`: return
>   `executorch::aten::Scalar(payload.copyable_union.as_bool)`.
> - Else (any non-scalar tag): return `Error::InvalidType`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-scalar-type-fn]
> Result<executorch::aten::ScalarType> tryToScalarType() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-scalar-type-fn]
> Result-returning ScalarType accessor (never aborts).
>
> Behavior:
> - If `!isInt()`: return `Error::InvalidType`.
> - Else return
>   `static_cast<executorch::aten::ScalarType>(payload.copyable_union.as_int)`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-string-fn]
> Result<std::string_view> tryToString() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-string-fn]
> Result-returning string accessor (never aborts).
>
> Behavior:
> - If `!isString()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_string_ptr == nullptr`: return
>   `Error::InvalidState`.
> - Else return `std::string_view(as_string_ptr->data(),
>   as_string_ptr->size())`.

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-tensor-fn]
> Result<executorch::aten::Tensor> tryToTensor() const

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-fn]
> Result-returning Tensor accessor (never aborts). Returns a copy of the Tensor
> handle by value (Result<T> cannot wrap a reference); in ATen mode this is one
> intrusive_ptr refcount bump, free in lean mode.
>
> Behavior:
> - If `!isTensor()`: return `Error::InvalidType`.
> - Else return `payload.as_tensor` (copied into the Result; the EValue is left
>   unchanged, unlike the rvalue `toTensor()` which moves out).

> [spec:et:def:evalue.executorch.runtime.e-value.try-to-tensor-list-fn]
> Result<executorch::aten::ArrayRef<executorch::aten::Tensor>> tryToTensorList()

> [spec:et:sem:evalue.executorch.runtime.e-value.try-to-tensor-list-fn]
> Result-returning tensor-list accessor (never aborts).
>
> Behavior:
> - If `!isTensorList()`: return `Error::InvalidType`.
> - Else if `payload.copyable_union.as_tensor_list_ptr == nullptr`: return
>   `Error::InvalidState`.
> - Else return `as_tensor_list_ptr->tryGet()`
>   (`[spec:et:sem:evalue.executorch.runtime.boxed-evalue-list-t.try-get-fn]`),
>   which validates and materializes each element, propagating
>   `Error::InvalidState` (null wrapped pointer) or `Error::InvalidType` (wrong
>   element tag) instead of aborting.

> [spec:et:def:evalue.executorch.runtime.internal.evalue-to-const-ref-overload-return]
> struct evalue_to_const_ref_overload_return

> [spec:et:def:evalue.executorch.runtime.internal.evalue-to-const-ref-overload-return-executorch-aten-tensor]
> struct evalue_to_const_ref_overload_return<executorch::aten::Tensor>

> [spec:et:def:evalue.executorch.runtime.internal.evalue-to-ref-overload-return]
> struct evalue_to_ref_overload_return

> [spec:et:def:evalue.executorch.runtime.internal.evalue-to-ref-overload-return-executorch-aten-tensor]
> struct evalue_to_ref_overload_return<executorch::aten::Tensor>

> [spec:et:def:evalue.executorch.runtime.e-value.operator-fn]
> EValue& operator=(EValue&& rhs) & noexcept

> [spec:et:sem:evalue.executorch.runtime.e-value.operator-fn]
> Move-assignment operator (lvalue-qualified, noexcept). Replaces this EValue's
> contents with those of `rhs`, leaving `rhs` as None.
>
> Behavior:
> - Self-assignment guard: if `&rhs == this`, return `*this` unchanged (no
>   destroy, no move — protects against destroying then reading the same
>   object).
> - Call `destroy()`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.destroy-fn]`) to release
>   any resource currently held (e.g. this EValue's tensor or list elements).
> - Call `moveFrom(std::move(rhs))`
>   (`[spec:et:sem:evalue.executorch.runtime.e-value.move-from-fn]`) to move
>   rhs's payload and tag in and reset rhs to None.
> - Return `*this`.
> - The copy-assignment operator (not this rule) is defined in terms of this
>   one: `*this = EValue(rhs)`.

