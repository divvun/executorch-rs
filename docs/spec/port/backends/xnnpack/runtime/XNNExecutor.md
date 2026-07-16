# backends/xnnpack/runtime/XNNExecutor.cpp, backends/xnnpack/runtime/XNNExecutor.h

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard]
> class InUseGuard {
>   std::atomic<bool>& flag_;
>   bool dismissed_ = false;
> }

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn]
> void dismiss()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn]
> `InUseGuard` is an RAII scope guard over an `std::atomic<bool>& flag_`
> reference (bound at construction) plus a `bool dismissed_` (default false).
> `dismiss()` sets `dismissed_ = true`. Effect: after `dismiss()`, the guard's
> destructor will NOT reset the flag (see
> `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]`).
> Used to opt out of the automatic release, either on the normal success path
> (once the caller wants the flag left as-is) or when the guard should not
> touch a flag it did not itself set. No return value.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]
> ~InUseGuard()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]
> Destructor of `InUseGuard`. If `dismissed_` is false, stores `false` into
> the referenced atomic `flag_` using `std::memory_order_release` (clearing
> the in-use flag). If `dismissed_` is true (see
> `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.dismiss-fn]`),
> does nothing. This is the automatic-release half of the guard: it makes a
> `prepare_args`/`forward` critical section clear the executor's `in_use_`
> flag exactly once on scope exit unless dismissed. Rust equivalent: a `Drop`
> impl that conditionally does a release-store of `false`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor]
> class XNNExecutor {
>   std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{ nullptr, &xnn_delete_runtime};
>   profiling::XNNProfiler profiler_;
>   std::vector<uint32_t> input_ids_;
>   std::vector<uint32_t> output_ids_;
>   std::vector<xnn_external_value> externals_;
>   std::vector<std::string> packed_data_names_;
>   std::shared_ptr<XNNWorkspace> workspace_;
>   std::shared_ptr<XNNWeightsCache> weights_cache_;
>   std::atomic<bool> in_use_{false};
>   std::atomic<bool> destroyed_{false};
>   ET_NODISCARD executorch::runtime::Error initialize( xnn_runtime_t runtime, std::vector<uint32_t>&& input_ids, std::vector<uint32_t>&& output_ids, std::vector...;
>   ET_NODISCARD executorch::runtime::Error prepare_args( executorch::runtime::Span<executorch::runtime::EValue*> args);
>   ET_NODISCARD executorch::runtime::Error forward( executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext& context);
>   ET_NODISCARD executorch::runtime::Error resize_outputs( executorch::runtime::Span<executorch::runtime::EValue*> args) const;
>   ET_NODISCARD executorch::runtime::Error convert_outputs( executorch::runtime::Span<executorch::runtime::EValue*> args) const;
> }

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn]
> ET_NODISCARD Error XNNExecutor::convert_outputs(Span<EValue*> args) const

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.convert-outputs-fn]
> Post-processes output tensors to fix up dtypes XNNPACK cannot natively
> produce. Specifically, XNNPACK does not support int64; for `arg_max`-style
> ops the graph produces int32 index data, but the ExecuTorch output tensor is
> declared as Long (int64). This function widens that int32 data to int64
> in place. `const`. Returns `Error` (`ET_NODISCARD`); always `Error::Ok`.
>
> Steps — loop `i` over the output range `[input_ids_.size(),
> externals_.size())`:
> 1. `ext_id = externals_[i].id`; `out_tensor = &args[ext_id]->toTensor()`.
> 2. If `out_tensor->scalar_type() == ScalarType::Long` (int64):
>    a. Treat the tensor's storage as holding int32 values packed from index 0
>       (written there by XNNPACK), and widen them to int64 in the same buffer.
>       Let `data_64 = out_tensor->mutable_data_ptr<int64_t>()` and `data_32 =
>       out_tensor->const_data_ptr<int32_t>()` (both alias the same underlying
>       storage).
>    b. Iterate `j` from `numel() - 1` DOWN to `0` (descending), setting
>       `data_64[j] = data_32[j]`. The descending order is essential: because
>       the int64 element at index `j` occupies bytes that overlap the int32
>       elements at higher indices, writing from the high end backward avoids
>       clobbering not-yet-read int32 values (an in-place expanding copy). The
>       int32 value at `data_32[j]` is sign-extended to int64.
>    Tensors whose scalar type is not Long are left untouched.
> 3. Return `Error::Ok`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn]
> ET_NODISCARD Error XNNExecutor::forward(BackendExecutionContext& context)

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.forward-fn]
> Executes the XNNPACK runtime using the externals prepared by `prepare_args`,
> wrapping the invocation in profiler start/end and the in-use guard. Returns
> `Error` (`ET_NODISCARD`).
>
> Steps:
> 1. Construct an `InUseGuard` over `in_use_` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]`).
>    This guard is NOT dismissed here, so on any exit from `forward` (success
>    or error) it clears `in_use_` — releasing the flag that `prepare_args`
>    left set (step 11 of
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]`).
> 2. `ET_CHECK_OR_RETURN_ERROR(runtime_ != nullptr, Internal, ...)`: if the
>    runtime is null, return `Error::Internal`.
> 3. `status = xnn_setup_runtime_v2(runtime_.get(), externals_.size(),
>    externals_.data())` to bind the external values (ids + data pointers) into
>    the runtime. If `status != xnn_status_success`, log an Error (with
>    `xnn_status_to_string(status)`) and return `Error::Internal`.
> 4. `error = profiler_.start(context.event_tracer())`; if not `Error::Ok`,
>    log an Error with the numeric value but continue (profiling is non-fatal).
> 5. `status = xnn_invoke_runtime(runtime_.get())` — runs the graph.
> 6. `error = profiler_.end()`; if not `Error::Ok`, log an Error but continue.
>    (End is always called after invoke, regardless of invoke's status.)
> 7. `ET_CHECK_OR_RETURN_ERROR(status == xnn_status_success, Internal, ...)`:
>    evaluate the invoke status; if it failed, return `Error::Internal` (message
>    includes `xnn_status_to_string(status)`).
> 8. Return `Error::Ok`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-inputs-fn]
> inline size_t getNumInputs()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-inputs-fn]
> Returns the number of graph inputs, i.e. `input_ids_.size()` (the count of
> external input ids stored during `initialize`). Pure accessor, no side
> effects.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-outputs-fn]
> inline size_t getNumOutputs()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-num-outputs-fn]
> Returns the number of graph outputs, i.e. `output_ids_.size()` (the count of
> external output ids stored during `initialize`). Pure accessor, no side
> effects.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-packed-data-names-fn]
> inline std::vector<std::string> get_packed_data_names()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-packed-data-names-fn]
> Returns a copy of `packed_data_names_` (a `std::vector<std::string>` of the
> names of this executor's packed weight-cache entries, set during
> `initialize`). Returns by value (a copy), not a reference. Used by
> `destroy` to tell the shared weights cache which entries to release.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn]
> inline std::shared_ptr<XNNWeightsCache> get_weights_cache() const

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-weights-cache-fn]
> Returns a copy of the `weights_cache_` shared_ptr (`std::shared_ptr<
> XNNWeightsCache>`), which may be empty (null) when no file-backed weights
> cache is in use for this PTE. `const` accessor. Returning a copy of the
> shared_ptr bumps the refcount so callers (`execute`, `destroy`) can keep the
> cache alive while locking its mutex.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn]
> inline std::shared_ptr<XNNWorkspace> get_workspace()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.get-workspace-fn]
> Returns a copy of the `workspace_` shared_ptr (`std::shared_ptr<
> XNNWorkspace>`), the memory arena this executor uses (set at construction).
> Returning a copy bumps the refcount so callers can hold the workspace (and
> its lock) alive across `execute`/`destroy` even while the executor is being
> torn down. Non-const accessor.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn]
> ET_NODISCARD Error XNNExecutor::initialize( xnn_runtime_t runtime, std::vector<uint32_t>&& input_ids, std::vector<uint32_t>&& output_ids, std::vector<std::string>&& packed_data_names)

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.initialize-fn]
> Populates a freshly-constructed `XNNExecutor` with a compiled XNNPACK
> runtime and its external input/output id lists. Takes ownership of the
> runtime and moves-in the id vectors and packed-data-name vector. Returns
> `Error` (`ET_NODISCARD`); always returns `Error::Ok` in practice (profiler
> failure is logged, not propagated).
>
> Steps:
> 1. Take ownership of `runtime` by assigning `runtime_` a
>    `unique_ptr<xnn_runtime, &xnn_delete_runtime>` wrapping it (so the runtime
>    is deleted via `xnn_delete_runtime` when the executor is destroyed).
> 2. Initialize the profiler: `error = profiler_.initialize(runtime)`. If it
>    is not `Error::Ok`, log an Error message with the numeric error value but
>    continue (profiling failure is non-fatal and NOT returned).
> 3. Move `input_ids` into `input_ids_`, then sort `input_ids_` ascending
>    (`std::sort`). The ids are sorted so their position in `externals_`
>    corresponds to ascending external id order.
> 4. Move `output_ids` into `output_ids_`, then sort `output_ids_` ascending.
> 5. Resize `externals_` to `input_ids_.size() + output_ids_.size()` elements
>    (uninitialized `xnn_external_value` entries; filled later by
>    `prepare_args`). The first `input_ids_.size()` slots correspond to inputs,
>    the remainder to outputs.
> 6. Move `packed_data_names` into `packed_data_names_`.
> 7. Return `Error::Ok`.
>
> No argument validation is performed on the id vectors or runtime here; a
> null runtime would be detected later in `prepare_args`/`forward`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]
> ET_NODISCARD Error XNNExecutor::prepare_args(Span<EValue*> args)

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.prepare-args-fn]
> Binds the caller's `EValue*` args into the executor's XNNPACK external
> values, reshapes input tensors into the runtime, propagates shapes through
> the whole runtime, and resizes output tensors to the computed shapes.
> `args` is indexed by external id (each input/output external id is an index
> into `args`). Returns `Error` (`ET_NODISCARD`).
>
> Concurrency / lifecycle guards:
> 1. Debug assertion: `destroyed_.load(acquire)` must be false (must not be
>    called after `destroy`).
> 2. `was_in_use = in_use_.exchange(true, acquire)`. If `was_in_use` is true,
>    log an Error ("called concurrently") and debug-assert failure — a
>    concurrent call is a usage error. The exchange atomically marks the
>    executor in-use.
> 3. Construct an `InUseGuard` over `in_use_` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.in-use-guard-fn]`),
>    which will clear `in_use_` on scope exit. If `was_in_use` was true (this
>    call did NOT acquire the flag), immediately `dismiss()` the guard so it
>    does not clear a flag owned by the other in-flight call.
>
> Preconditions:
> 4. `ET_CHECK_OR_RETURN_ERROR(runtime_ != nullptr, Internal, ...)`: if the
>    runtime is null (compile failed), return `Error::Internal`.
>
> Bind externals and reshape inputs — loop `i` over `[0, externals_.size())`:
> 5. Set `externals_[i].id`: for `i < input_ids_.size()`, it is
>    `input_ids_[i]`; otherwise it is `output_ids_[i - input_ids_.size()]`.
>    Let `ext_id = externals_[i].id`.
> 6. `ET_CHECK_OR_RETURN_ERROR(args[ext_id]->isTensor(), InvalidArgument, ...)`:
>    if the arg at index `ext_id` is not a Tensor, return
>    `Error::InvalidArgument` (message includes `i` and the EValue tag).
> 7. `tensor = &args[ext_id]->toTensor()`; set `externals_[i].data =
>    tensor->mutable_data_ptr<float>()`. Note the data pointer is always taken
>    as `float*` here regardless of the tensor's actual dtype — XNNPACK
>    externals carry a raw data pointer; the byte address is what matters.
> 8. Only for inputs (`i < input_ids_.size()`), reshape the runtime input:
>    a. `num_dims = tensor->dim()`.
>    b. Retrieve the tensor's dim order into a stack buffer `dim_order`
>       (length `kTensorDimensionLimit`) via `get_dim_order(*tensor, dim_order,
>       num_dims)`. If it does not return `Error::Ok`, return `Error::Internal`.
>    c. `ET_CHECK_OR_RETURN_ERROR(num_dims <= XNN_MAX_TENSOR_DIMS,
>       InvalidArgument, ...)`: XNNPACK accepts at most `XNN_MAX_TENSOR_DIMS`
>       dims; else return `Error::InvalidArgument`.
>    d. Build a `size_t dims[XNN_MAX_TENSOR_DIMS]` array in XNNPACK's
>       (logical/physical) order: for `j` in `[0, num_dims)`, `dims[j] =
>       tensor->size(dim_order[j])`. That is, iterate in dim-order and read the
>       size of the dimension named by `dim_order[j]`, producing the extents in
>       the tensor's physical memory layout order.
>    e. Call `status = xnn_reshape_external_value(runtime_.get(), ext_id,
>       num_dims, dims)`. If `status != xnn_status_success`, return
>       `Error::Internal` (message includes `xnn_status_to_string(status)`, per
>       `[spec:et:sem:xnn-status.executorch.backends.xnnpack.delegate.xnn-status-to-string-fn]`).
>    Outputs are not reshaped here (they are sized by `resize_outputs`).
>
> Propagate and finish:
> 9. `status = xnn_reshape_runtime(runtime_.get())` to propagate input shapes
>    through the graph and (re)plan memory. If `status != xnn_status_success`,
>    return `Error::Internal`.
> 10. Call `resize_outputs(args)` (per
>    `[spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn]`).
>    If it returns non-Ok, return that error.
> 11. `in_use_guard.dismiss()` — intentionally leave `in_use_` set true across
>    the boundary to `forward` (the flag is cleared by `forward`'s own guard),
>    so the in-use protection spans the prepare→forward pair.
> 12. Return `Error::Ok`.
>
> Note: on every early-return error path in steps 4–10, the still-active
> `InUseGuard` (unless it was dismissed in step 3 for the concurrent case)
> clears `in_use_` on unwind.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn]
> ET_NODISCARD Error XNNExecutor::resize_outputs(Span<EValue*> args) const

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.resize-outputs-fn]
> Resizes each output tensor in `args` to the shape XNNPACK computed for the
> corresponding external value after `xnn_reshape_runtime`. `const`. Returns
> `Error` (`ET_NODISCARD`).
>
> Steps — loop `i` over the output range `[input_ids_.size(),
> externals_.size())`:
> 1. `ext_id = externals_[i].id`; `out_tensor = &args[ext_id]->toTensor()`
>    (assumes it is a Tensor; validated earlier in `prepare_args`).
> 2. Query the runtime's current shape for this external:
>    `status = xnn_get_external_value_shape(runtime_.get(), ext_id, &num_dim,
>    dims)`, where `dims` is a stack array `size_t dims[XNN_MAX_TENSOR_DIMS]`
>    and `num_dim` receives the rank. If `status != xnn_status_success`, return
>    `Error::Internal`.
> 3. Retrieve the output tensor's dim order into `dim_order` (length
>    `kTensorDimensionLimit`) via `get_dim_order(*out_tensor, dim_order,
>    num_dim)`. If not `Error::Ok`, return `Error::Internal`.
> 4. Convert XNNPACK's shape (`dims`, in physical/dim-order order) back into
>    ExecuTorch logical `SizesType` order: for `j` in `[0, num_dim)`,
>    `expected_output_size[dim_order[j]] = (SizesType)dims[j]`. This is the
>    inverse of the input reshape mapping in `prepare_args`: `dims[j]` is the
>    extent of the `dim_order[j]`-th logical dimension, so it is scattered back
>    to logical index `dim_order[j]`. `expected_output_size` is a stack array
>    of length `kTensorDimensionLimit`.
> 5. Form `output_size = ArrayRef<SizesType>{expected_output_size, num_dim}`.
> 6. Log a Debug message ("Resizing output tensor to a new shape") and call
>    `resize_tensor(*out_tensor, output_size)`. If it returns non-Ok, log an
>    Error and return that error.
> 7. After all outputs are processed, return `Error::Ok`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn]
> inline void set_weights_cache(std::shared_ptr<XNNWeightsCache> cache)

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.set-weights-cache-fn]
> Stores the given `std::shared_ptr<XNNWeightsCache>` into `weights_cache_`
> via `std::move` (takes ownership of the caller's shared_ptr, transferring
> the refcount). Called once by `XNNPACKBackend::init` after `compileModel`
> succeeds. Passing an empty shared_ptr is valid and means "no file-backed
> cache for this PTE" — treated identically to never calling this. No return
> value.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn]
> inline bool uses_weight_cache() const

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.uses-weight-cache-fn]
> Returns `true` iff `packed_data_names_` is non-empty (`!packed_data_names_
> .empty()`), i.e. iff this executor has packed weight entries registered in a
> shared weights cache. `const`, no side effects. Note this is independent of
> whether `weights_cache_` is set: it reflects whether packed data names
> exist. Used in `destroy` (together with a non-empty cache) to decide whether
> to call `delete_packed_data`.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn]
> ~XNNExecutor()

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.xnn-executor.xnn-executor-fn]
> Destructor of `XNNExecutor`.
> 1. Debug assertion (`ET_DCHECK_MSG`, active only in debug builds): asserts
>    `in_use_.load(std::memory_order_acquire) == false`, i.e. the executor must
>    not be destroyed while a `prepare_args`/`forward` critical section is
>    active. A Rust port should debug-assert the same.
> 2. Stores `true` into `destroyed_` with `std::memory_order_release`, marking
>    the executor as torn down (this is checked by a debug assertion at the top
>    of `prepare_args`).
> 3. Member destruction then runs implicitly in reverse declaration order. In
>    particular `runtime_` is a `std::unique_ptr<xnn_runtime, &
>    xnn_delete_runtime>`, so destroying it calls `xnn_delete_runtime` on the
>    owned runtime (freeing the XNNPACK runtime). `profiler_`, the id vectors,
>    `externals_`, `packed_data_names_`, and the `workspace_`/`weights_cache_`
>    shared_ptrs are also released (decrementing refcounts). The
>    `xnn_delete_runtime` call is expected to be serialized by the caller
>    (`destroy` holds the workspace lock) since it is not thread-safe.

> [spec:et:def:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.operator-fn]
> InUseGuard& operator=(const InUseGuard&) = delete

> [spec:et:sem:xnn-executor.executorch.backends.xnnpack.delegate.in-use-guard.operator-fn]
> The copy-assignment operator of `InUseGuard` is explicitly deleted
> (`= delete`), as is the copy constructor. `InUseGuard` is therefore non-
> copyable (it holds a reference member and manages a unique release
> responsibility). There is no runtime behavior; this is a compile-time
> restriction. In Rust this is the default (no `Copy`/`Clone` derived) — the
> guard is a move-only/scope-local value.

