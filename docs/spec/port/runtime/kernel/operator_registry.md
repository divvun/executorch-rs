# runtime/kernel/operator_registry.cpp, runtime/kernel/operator_registry.h

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn]
> int copy_char_as_number_to_buf(int num, char* buf, size_t buf_size)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn]
> Writes the non-negative integer `num` as its shortest decimal ASCII
> representation into `buf`, without a NUL terminator, and returns the number
> of bytes written. `buf_size` is the number of bytes available at `buf`.
> Only values in `[0, 99]` are supported (this is used to serialize
> `ScalarType` enum values and `DimOrderType` dim indices, which are small).
>
> Cases, in order:
> - `num < 0`: unsupported; return `-1`, write nothing.
> - `num < 10` (single digit): if `buf_size < 1` return `-1` (write nothing);
>   otherwise write the one character `'0' + num` at `buf[0]` and return `1`.
> - `num < 100` (two digits): if `buf_size < 2` return `-1` (write nothing);
>   otherwise write the tens digit `'0' + (num / 10)` at `buf[0]` and the
>   units digit `'0' + (num % 10)` at `buf[1]`, then return `2`. Note this
>   always writes two characters for values `10..99` (e.g. no leading-zero
>   suppression is needed because the single-digit branch already handled
>   `0..9`).
> - `num >= 100`: unsupported; return `-1`, write nothing.
>
> The function never NUL-terminates; the caller advances its write pointer by
> the returned count and handles terminators/separators itself (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]`).

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]
> Result<OpFunction> get_op_function_from_registry( const char* name, Span<const TensorMeta> meta_list, Span<const Kernel> kernel_list)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]
> Looks up the `OpFunction` registered for operator `name` whose kernel key
> matches the dtype/dim-order signature described by `meta_list`, searching
> `kernel_list`. Returns a `Result<OpFunction>` that is either the found
> function or an `Error`.
>
> There are two overloads:
> - The three-argument overload `(name, meta_list, kernel_list)` does the
>   actual work (described below).
> - The two-argument overload `(name, meta_list)` forwards to the three-arg
>   overload passing `get_registered_kernels()` (see
>   `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]`)
>   as `kernel_list`, i.e. it searches the global registry.
>
> Three-argument algorithm:
> - Allocate a stack buffer of `internal::kKernelKeyBufSize` (659) bytes and
>   build the key string for `meta_list` into it via `make_kernel_key_string`
>   (see
>   `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]`).
>   If that returns any error other than `Error::Ok`, log "Failed to make
>   kernel key string" and return that error.
> - Construct a `KernelKey kernel_key` from the buffer's data pointer. Because
>   `make_kernel_key_string` writes an empty string ("") when `meta_list` is
>   empty, `kernel_key` is then a non-null pointer to "" — it is NOT a
>   fallback key (a fallback key has a null data pointer). See
>   `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]`.
> - Initialize `fallback_idx = -1`. Iterate `idx` from `0` to
>   `kernel_list.size() - 1` in order. For each entry:
>   - Skip unless `strcmp(kernel_list[idx].name_, name) == 0` (exact name
>     match).
>   - If its `kernel_key_` equals `kernel_key` (see
>     `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]`),
>     immediately return that entry's `op_` — an exact specialized match wins
>     and short-circuits the search.
>   - Otherwise, if that entry's `kernel_key_.is_fallback()` is true, record
>     `fallback_idx = idx` (later fallbacks overwrite earlier ones, so the
>     last matching fallback in list order is remembered).
> - After the loop, if no exact match was found but `fallback_idx != -1`,
>   return `kernel_list[fallback_idx].op_`.
> - Otherwise log "kernel '<name>' not found." followed by the tensor meta
>   list, and return `Error::OperatorMissing`.
>
> Note: an exact specialized match is only possible when `meta_list` is
> non-empty and matches some registered specialized key; when `meta_list` is
> empty the built key is "" which will not `equals` a fallback (null) key,
> so only a registered fallback kernel for that name can satisfy the lookup.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]
> Span<const Kernel> get_registered_kernels()

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]
> Returns a read-only `Span<const Kernel>` that views the global kernel
> table: `{registered_kernels, num_registered_kernels}` — the base pointer of
> the process-global registered-kernels array and the current count of
> registered kernels. The span aliases the live registry storage (no copy);
> its length reflects however many kernels have been registered so far via
> `register_kernels` /
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]`.
> Takes no arguments and never fails. In a Rust port the global registry is
> the equivalent of a static, append-only slice of `Kernel` guarded by the
> registration functions.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]
> Error make_kernel_key_string( Span<const TensorMeta> key, char* buf, size_t buf_size)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.internal.make-kernel-key-string-fn]
> Serializes the per-tensor dtype/dim-order metadata in `key` into the kernel
> key string format `"v1/<tensor_meta>|<tensor_meta>|..."`, where each
> `<tensor_meta>` is `"<dtype>;<d0>,<d1>,..."` (dtype is the `ScalarType`
> enum's integer value; the dim-order entries are the `DimOrderType` integer
> values). Writes into `buf` (capacity `buf_size` bytes, NUL-terminated) and
> returns `Error::Ok` on success or `Error::InvalidArgument` if the buffer is
> too small at any step or a value cannot be encoded. All numeric fields are
> encoded via
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.copy-char-as-number-to-buf-fn]`,
> which only supports values in `[0, 99]`.
>
> Steps:
> - Empty `key` (no tensors): the kernel key does not apply. If `buf_size > 0`
>   write `'\0'` at `buf[0]` (empty string); if `buf_size == 0` write
>   nothing. Return `Error::Ok`. (The resulting empty-string key is a
>   non-fallback key when later wrapped in a `KernelKey`.)
> - Reserve one byte for the trailing NUL: if `buf_size < 1` return
>   `Error::InvalidArgument`; otherwise decrement the working `buf_size` by 1
>   (this reserved byte is filled at the very end).
> - Write the 3-byte prefix `"v1/"`: if the remaining `buf_size < 3` return
>   `Error::InvalidArgument`; otherwise `memcpy` "v1/" (3 bytes, no NUL),
>   advance `buf` by 3 and decrement `buf_size` by 3.
> - For each tensor `meta` in `key`, in index order `i = 0..key.size()-1`:
>   - Encode `meta.dtype_` (cast to int) via `copy_char_as_number_to_buf`
>     into `buf` with the current remaining `buf_size`. If it returns `< 0`
>     (unsupported value or insufficient space) return `Error::InvalidArgument`;
>     otherwise advance `buf` and decrement `buf_size` by the returned count.
>   - Write the dtype/dim-order separator `';'`: if `buf_size < 1` return
>     `Error::InvalidArgument`; otherwise write `';'`, advance `buf` by 1 and
>     decrement `buf_size` by 1.
>   - For each dim-order entry `j = 0..meta.dim_order_.size()-1`:
>     - Encode `meta.dim_order_[j]` (cast to int) via
>       `copy_char_as_number_to_buf`; on `< 0` return `Error::InvalidArgument`;
>       otherwise advance `buf` / decrement `buf_size` by the count.
>     - If this is not the last dim entry (`j < size-1`), write the intra-tensor
>       separator `','`: if `buf_size < 1` return `Error::InvalidArgument`;
>       otherwise write `','` and advance/decrement by 1. (No trailing comma
>       after the final dim; a tensor with zero dims produces no dim digits
>       and no commas, yielding just `"<dtype>;"`.)
>   - If this is not the last tensor (`i < key.size()-1`), write the
>     inter-tensor separator `'|'`: if `buf_size < 1` return
>     `Error::InvalidArgument`; otherwise write `'|'` and advance/decrement
>     by 1.
> - Finally write `'\0'` at the current `buf` position (using the byte
>   reserved in the first step) and return `Error::Ok`.
>
> Every length check compares against the buffer space remaining *after*
> reserving the NUL byte, so a successful result always leaves room for the
> terminator. `kKernelKeyBufSize` (659) is sized to hold 16 tensors of 16
> dimensions plus the NUL.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel]
> struct Kernel {
>   const char* name_;
>   KernelKey kernel_key_;
>   OpFunction op_;
> }

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-buffer]
> struct alignas(Kernel) KernelBuffer {
>   uint8_t data[sizeof(Kernel)];
> }

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key]
> struct KernelKey {
>   const char* kernel_key_data_ = nullptr;
> }

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.data-fn]
> const char* data() const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.data-fn]
> Const accessor returning the raw `kernel_key_data_` pointer this `KernelKey`
> was constructed with. Returns `nullptr` for a default-constructed
> (fallback) key, or the borrowed C-string pointer for a specialized key (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]`).
> The pointer is not owned by the `KernelKey`; the string's lifetime must
> outlive the key. No copying, no mutation.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]
> bool equals(const KernelKey& other) const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]
> Value equality between two kernel keys. Steps:
> - If the two keys differ in fallback status
>   (`is_fallback() != other.is_fallback()`, see
>   `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]`),
>   return `false`.
> - If both are fallback keys (this key `is_fallback()` is true, and by the
>   previous check the other is too), return `true` — all fallback keys are
>   considered equal regardless of data pointer.
> - Otherwise both are specialized keys with non-null data pointers: return
>   `strcmp(kernel_key_data_, other.kernel_key_data_) == 0`, i.e. equal iff
>   the two NUL-terminated key strings are byte-for-byte identical. (Note:
>   this dereferences both pointers; it relies on the fallback checks above
>   guaranteeing neither is null.)

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]
> bool is_fallback() const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]
> Const predicate: returns `true` iff this `KernelKey`'s `kernel_key_data_`
> pointer is null (`kernel_key_data_ == nullptr`), i.e. iff the key was
> default-constructed and represents a non-specialized fallback that matches
> any dtype/dim-order signature. Returns `false` for a specialized key built
> from a non-null C-string pointer (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]`).
> No mutation, no dereference of the pointer. Note the empty-string key ""
> produced by `make_kernel_key_string` for an empty `meta_list` has a
> non-null data pointer, so it is NOT a fallback key.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]
> constexpr KernelKey(const char* kernel_key_data)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.kernel-key-fn]
> Implicit `constexpr` constructor from a `const char*`: stores the given
> `kernel_key_data` pointer verbatim into the private `kernel_key_data_`
> member (borrowed, not copied — the pointed-to string's lifetime must
> outlive the key). This produces a *specialized* (non-fallback) key whenever
> `kernel_key_data` is non-null; the key then matches only the specific
> dtype/dim-order signature encoded in that string (format per the
> `KernelKey` struct doc: `"v1/<tensor_meta>|<tensor_meta>..."`).
>
> The separate default constructor `KernelKey()` (not this rule) leaves
> `kernel_key_data_` at its default `nullptr`, producing a fallback key (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.is-fallback-fn]`).
> Passing a null pointer to this constructor would therefore also yield a
> fallback key, but callers are expected to pass a real key string generated
> from kernel YAML rather than construct keys by hand.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel-key.operator-fn]
> bool operator==(const KernelKey& other) const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.operator-fn]
> `operator==` for `KernelKey`: returns `this->equals(other)`, i.e. delegates
> entirely to
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]`
> (two fallback keys compare equal; a fallback and a specialized key are
> unequal; two specialized keys are equal iff their key strings are
> byte-for-byte identical via `strcmp`). The companion `operator!=` (not a
> spec rule) returns the negation, `!this->equals(other)`.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.kernel.kernel-fn]
> constexpr explicit Kernel(const char* name, OpFunction func)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel.kernel-fn]
> Two-argument `constexpr explicit` constructor `Kernel(const char* name,
> OpFunction func)`: builds a fallback `Kernel`. It stores `name` into
> `name_` (borrowed pointer — the string's lifetime must outlive the
> registry) and `func` into `op_`, and leaves `kernel_key_` at its default
> value, i.e. a default-constructed `KernelKey` whose `kernel_key_data_` is
> `nullptr`. That makes `kernel_key_.is_fallback()` true, so a `Kernel` built
> this way is a fallback kernel that matches any dtype/dim-order signature for
> `name`.
>
> Two sibling constructors (not this rule): `Kernel(const char* name,
> KernelKey key, OpFunction func)` additionally stores the supplied `key` into
> `kernel_key_` (used for specialized kernels), and the default `Kernel()`
> leaves `name_` and `op_` null with a fallback `kernel_key_` (used only to
> zero-fill the registry backing store).

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernel-fn]
> ET_NODISCARD inline Error register_kernel(const Kernel& kernel)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernel-fn]
> Convenience `inline` wrapper that registers a single kernel: forms a
> one-element `Span<const Kernel>` `{&kernel, 1}` viewing the caller's
> `kernel` and forwards it to
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]`,
> returning that call's `Error` verbatim. It is `ET_NODISCARD`. As with
> `register_kernels`, the return is `Error::Ok` on success and the function
> panics (via `ET_CHECK_MSG`) rather than returning on a duplicate or
> over-capacity registration. The `kernel` reference need only outlive this
> call (its `name_`/key-string pointers are copied by value into the registry
> slot, but must themselves remain valid for the registry's lifetime).

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]
> Error register_kernels(const Span<const Kernel> kernels)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]
> Public registration entry point. Calls
> `register_kernels_internal(kernels)` (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]`)
> and captures its `Error` result. If that result is either
> `Error::RegistrationAlreadyRegistered` or
> `Error::RegistrationExceedingMaxKernels`, it fires
> `ET_CHECK_MSG(false, ...)` — an unconditional fatal check that aborts the
> process, formatting the numeric error value into the message — so on those
> two failure modes control does not actually return. On any other outcome
> (in practice `Error::Ok`) it returns the internal result unchanged.
>
> The function is `ET_NODISCARD` and returns non-void only so it can be
> invoked at static-initialization time; the documented contract is "returns
> `Error::Ok` always, panics on error." A Rust port should treat the two
> registration errors as a panic/abort at this boundary and otherwise
> propagate `Error::Ok`.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]
> Error register_kernels_internal(const Span<const Kernel> kernels)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-internal-fn]
> Internal (anonymous-namespace) routine that appends the kernels in the
> `kernels` span to the process-global kernel table, returning an `Error`
> rather than panicking. The global table is a fixed-capacity array
> `registered_kernels` of `kMaxRegisteredKernels` slots (backed by static
> zeroed `KernelBuffer` storage) with a running count `num_registered_kernels`
> (initially 0). `kMaxRegisteredKernels` resolves at compile time: a
> user-defined `MAX_KERNEL_NUM` if set, else a selective-build
> `EXECUTORCH_SELECTED_MAX_KERNEL_NUM` if defined, else the default
> `250 * 8 = 2000`.
>
> Steps, in order:
> - Call `::et_pal_init()` first. Kernel registration can run at static-init
>   time before or after PAL init; this call is idempotent and safe to repeat.
> - Capacity check: if `kernels.size() + num_registered_kernels >
>   kMaxRegisteredKernels`, log an error describing the limit, the count
>   already registered, and the count being added; then log every kernel
>   currently in the registry (name + kernel key) and every kernel being
>   registered; and return `Error::RegistrationExceedingMaxKernels` without
>   registering anything.
> - Fetch `lib_name = et_pal_get_shared_library_name(kernels.data())` for
>   diagnostic logging only (unused except in messages).
> - For each `kernel` in `kernels`, in span order:
>   - Duplicate check: linearly scan the existing entries `i` in
>     `[0, num_registered_kernels)`. A duplicate is an existing entry `k`
>     where `strcmp(kernel.name_, k.name_) == 0` AND
>     `kernel.kernel_key_ == k.kernel_key_` (kernel-key value equality per
>     `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.kernel-key.equals-fn]`
>     — note two fallback keys count as equal). On a duplicate: log the
>     re-registration, fire `ET_CHECK_MSG(false, ...)` (fatal abort in this
>     variant), and (if that check were compiled out) return
>     `Error::RegistrationAlreadyRegistered`.
>   - Otherwise append: write `kernel` by value into
>     `registered_kernels[num_registered_kernels]` and post-increment
>     `num_registered_kernels`. The `Kernel` is copied by value (its `name_`
>     and key-string are borrowed pointers that must outlive the registry).
> - After all kernels are appended, log a debug success message and return
>   `Error::Ok`.
>
> Registration is not idempotent across the whole span: kernels are appended
> one at a time, so if the span contains a duplicate the entries before it in
> the span will already have been added before the failure. Iteration order
> is span order, and later `get_op_function_from_registry` fallback selection
> depends on this insertion order (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]`).

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn]
> bool registry_has_op_function( const char* name, Span<const TensorMeta> meta_list)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn]
> Existence check for a registered kernel. Calls
> `get_op_function_from_registry(name, meta_list)` (the two-argument overload
> that searches the global registry, see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]`)
> and returns the boolean `.ok()` of the resulting `Result<OpFunction>`:
> `true` if a matching specialized or fallback kernel was found, `false`
> otherwise (i.e. the lookup produced `Error::OperatorMissing`, or an
> `Error::InvalidArgument` from key-string construction). `meta_list` defaults
> to an empty span; an empty `meta_list` means the op has no specialized
> kernels, so this then reports whether a fallback kernel exists for `name`.
> No output beyond the boolean; any error is swallowed into `false`.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta]
> struct TensorMeta {
>   executorch::aten::ScalarType dtype_;
>   Span<executorch::aten::DimOrderType> dim_order_;
> }

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn]
> bool equals(const TensorMeta& other) const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn]
> Value equality between two `TensorMeta`. Returns `true` iff both the dtype
> and the full dim-order sequence match:
> - If `dtype_ != other.dtype_` (ScalarType enum comparison), return `false`.
> - If `dim_order_.size() != other.dim_order_.size()`, return `false`.
> - Otherwise compare element-by-element for `i = 0..dim_order_.size()-1`; if
>   any `dim_order_[i] != other.dim_order_[i]` (DimOrderType comparison),
>   return `false`.
> - If all checks pass (including the trivial zero-length case where the loop
>   makes no comparisons), return `true`.
>
> This is the underlying comparison used by both
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn]`
> and the sibling `operator!=`.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn]
> bool operator==(const TensorMeta& other) const

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.operator-fn]
> `operator==` for `TensorMeta`: returns `this->equals(other)`, delegating
> entirely to
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.equals-fn]`
> (equal iff same dtype and identical dim-order sequence). The sibling
> `operator!=` (not a spec rule) returns `!this->equals(other)`.

> [spec:et:def:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn]
> TensorMeta( executorch::aten::ScalarType dtype, Span<executorch::aten::DimOrderType> order) : dtype_(dtype), dim_order_(order)

> [spec:et:sem:operator-registry.executorch.et-runtime-namespace.tensor-meta.tensor-meta-fn]
> Two-argument constructor `TensorMeta(ScalarType dtype, Span<DimOrderType>
> order)`: stores `dtype` into `dtype_` and `order` into `dim_order_` (the
> span is copied by value — pointer + length — so the underlying dim-order
> array is borrowed, not owned, and must outlive this `TensorMeta`). No
> validation is performed. The separate defaulted `TensorMeta() = default`
> (not this rule) leaves both members default-initialized.

> [spec:et:def:operator-registry.torch.executor.get-kernels-fn]
> inline ArrayRef<Kernel> get_kernels()

> [spec:et:sem:operator-registry.torch.executor.get-kernels-fn]
> Deprecated `torch::executor` compatibility alias. Calls
> `::executorch::ET_RUNTIME_NAMESPACE::get_registered_kernels()` (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-registered-kernels-fn]`)
> to obtain a `Span<const Kernel>` over the live global registry, then
> re-wraps the same pointer and length into an `ArrayRef<Kernel>`
> (`ArrayRef<Kernel>(kernels.data(), kernels.size())`) and returns it. No
> copy; the returned view aliases the registry storage. This is a thin
> forwarding shim retained only for callers not yet migrated to the
> `::executorch` namespace.

> [spec:et:def:operator-registry.torch.executor.get-ops-fn-fn]
> inline OpFunction getOpsFn( const char* name, ArrayRef<TensorMeta> meta_list = {})

> [spec:et:sem:operator-registry.torch.executor.get-ops-fn-fn]
> Deprecated `torch::executor` compatibility alias that returns the resolved
> `OpFunction` directly (not a `Result`). Forwards `name` and the `meta_list`
> `ArrayRef` (converted to a `Span<const TensorMeta>` via
> `{meta_list.data(), meta_list.size()}`, defaulting to empty) to
> `::executorch::ET_RUNTIME_NAMESPACE::get_op_function_from_registry` (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.get-op-function-from-registry-fn]`).
> It then asserts success with `ET_CHECK(result.ok())` — a fatal abort if the
> operator was not found (the underlying call already logged the details) —
> and returns the dereferenced `*result` (the `OpFunction`). In a Rust port
> this is the panic-on-missing variant of the registry lookup.

> [spec:et:def:operator-registry.torch.executor.has-ops-fn-fn]
> inline bool hasOpsFn(const char* name, ArrayRef<TensorMeta> meta_list =

> [spec:et:sem:operator-registry.torch.executor.has-ops-fn-fn]
> Deprecated `torch::executor` compatibility alias. Converts the `meta_list`
> `ArrayRef` to a `Span<const TensorMeta>` (`{meta_list.data(),
> meta_list.size()}`, defaulting to empty) and forwards `name` plus that span
> to `::executorch::ET_RUNTIME_NAMESPACE::registry_has_op_function` (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.registry-has-op-function-fn]`),
> returning its `bool` verbatim: `true` iff a matching specialized or fallback
> kernel exists for `name` (an empty `meta_list` means "does any fallback
> kernel exist"). Never aborts.

> [spec:et:def:operator-registry.torch.executor.register-kernels-fn]
> inline ::executorch::runtime::Error register_kernels(ArrayRef<Kernel> kernels)

> [spec:et:sem:operator-registry.torch.executor.register-kernels-fn]
> Deprecated `torch::executor` compatibility alias. Converts the `kernels`
> `ArrayRef` to a `Span<const Kernel>` (`{kernels.data(), kernels.size()}`)
> and forwards to
> `::executorch::ET_RUNTIME_NAMESPACE::register_kernels` (see
> `[spec:et:sem:operator-registry.executorch.et-runtime-namespace.register-kernels-fn]`),
> returning its `Error` verbatim. Same contract: returns `Error::Ok` on
> success, aborts via the underlying fatal check on a duplicate or
> over-capacity registration.

