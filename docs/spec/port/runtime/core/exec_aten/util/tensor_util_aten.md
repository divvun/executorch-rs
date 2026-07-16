# runtime/core/exec_aten/util/tensor_util_aten.cpp

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]
> Error get_dim_order( const at::Tensor& tensor, executorch::aten::DimOrderType* out_dim_order, size_t out_dim_order_size)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]
> ATen-mode implementation. Writes the dim order of `tensor` into the
> caller-provided buffer `out_dim_order` (length `out_dim_order_size`).
>
> Validation: `ET_CHECK_OR_RETURN_ERROR(out_dim_order_size == tensor.dim())`.
> If `out_dim_order_size` (a `size_t`) does not equal the tensor rank
> `tensor.dim()` (an `int64_t`), return `Error::InvalidArgument` and write
> nothing to `out_dim_order`.
>
> On success, derive the dim order from the tensor's strides: call
> `stride_to_dim_order(tensor.strides().data(), tensor.dim(), out_dim_order)`
> and return its `Error` result directly. That helper fills `out_dim_order`
> with the permutation of dimension indices sorted by decreasing stride
> (ties broken toward the lower dimension index / the trailing dimension per
> the shared helper's contract), producing e.g. `[0,1,2,3]` for a contiguous
> 4-D tensor and `[0,2,3,1]` for channels-last. The stride-derived dim order
> is the canonical way ATen tensors express layout, since at::Tensor has no
> explicit dim_order field (contrast the portable version which reads
> `tensor.dim_order()` directly, `[spec:et:sem:tensor-util-portable.executorch.runtime.get-dim-order-fn]`).

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.copy-tensor-data-fn]
> Error copy_tensor_data(const at::Tensor& t_dst, const at::Tensor& t_src)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.copy-tensor-data-fn]
> ATen-mode implementation. Copies the raw bytes of `t_src` into the
> pre-allocated storage of `t_dst`.
>
> Step 1: obtain the destination data pointer by reaching through the
> TensorImpl to its StorageImpl and reading `data_ptr().get()` (raw `void*`
> to the destination storage's memory).
>
> Step 2: `ET_CHECK_OR_RETURN_ERROR(dst_data_ptr != nullptr)`. If the
> destination storage pointer is null, return `Error::InvalidArgument`
> ("Destination tensor data pointer must not be null."). Note: even 0-sized
> tensors are expected to receive a non-null data pointer under pre-allocated
> memory planning, so this check is unconditional on the destination side.
>
> Step 3: if `t_src.const_data_ptr() != nullptr` (a source with a size-0
> dimension may legitimately be null, in which case nothing is copied and the
> function succeeds): check `ET_CHECK_OR_RETURN_ERROR(t_dst.nbytes() ==
> t_src.nbytes())`, returning `Error::InvalidArgument` on mismatch, then
> `std::memcpy(dst_data_ptr, t_src.const_data_ptr(), t_src.nbytes())`.
>
> Step 4: return `Error::Ok`. Only the data bytes are copied; sizes, strides,
> dtype, and storage ownership of `t_dst` are unchanged.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.reset-data-ptr-fn]
> void reset_data_ptr(const at::Tensor& tensor)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.reset-data-ptr-fn]
> ATen-mode implementation. Detaches the tensor from its backing memory.
> Obtain the underlying `c10::TensorImpl*` via `tensor.unsafeGetTensorImpl()`,
> then: (1) call `impl->set_sizes_contiguous(0)`, resizing the tensor to a
> single dimension of length 0 (rank becomes 1, numel becomes 0, strides
> contiguous); (2) reach through to the StorageImpl via
> `impl->unsafe_storage().unsafeGetStorageImpl()->reset()`, which clears the
> storage's data pointer and byte count. Returns void; has no failure path.
> Unlike the portable variant (`[spec:et:sem:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn]`),
> this also collapses the sizes, and it resets StorageImpl state rather than
> only nulling the data pointer.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.resize-tensor-impl-fn]
> Error resize_tensor_impl( c10::TensorImpl* impl, c10::ArrayRef<executorch::aten::SizesType> new_sizes)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.resize-tensor-impl-fn]
> ATen-mode implementation. Resizes the tensor behind `impl` to `new_sizes`,
> enforcing the runtime constraint that a tensor's rank is immutable.
>
> Step 1: if `impl->dim() != new_sizes.size()` (old rank differs from the
> requested number of dimensions), log an Error ("Tensor rank is not mutable:
> old dim ... new dim ...") and return `Error::NotSupported`. This guard is
> specific to ATen mode: at::Tensor would otherwise permit a rank change, but
> the higher-level runtime forbids it. (The portable `TensorImpl` performs an
> equivalent check internally.)
>
> Step 2: call `impl->set_sizes_contiguous(new_sizes)` to apply the new sizes
> with contiguous strides. This will panic (abort) on internal failure rather
> than return an error.
>
> Step 3: return `Error::Ok`. Most callers should use the higher-level
> `resize_tensor()` wrapper instead of calling this directly.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.set-tensor-data-fn]
> ET_NODISCARD Error

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.set-tensor-data-fn]
> ATen-mode implementation. Points the tensor's storage at a caller-supplied
> buffer. Return value is `ET_NODISCARD` (callers must inspect it).
>
> Step 1: `ET_CHECK_OR_RETURN_ERROR(buffer_size >= t.nbytes())`. If the
> provided `buffer_size` is smaller than the number of bytes the tensor needs
> (`t.nbytes()`), return `Error::InvalidArgument` and do not modify the
> tensor.
>
> Step 2: set the storage's data pointer to `buffer` by calling
> `t.unsafeGetTensorImpl()->unsafe_storage().set_data_ptr(at::DataPtr(buffer,
> at::DeviceType::CPU))`. The buffer is treated as CPU memory and is NOT owned
> by the tensor (no deleter is installed); the caller retains responsibility
> for its lifetime.
>
> Step 3: return `Error::Ok`.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.internal.share-tensor-data-fn]
> Error share_tensor_data(const at::Tensor& t_dst, const at::Tensor& t_src)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.share-tensor-data-fn]
> ATen-mode implementation. Makes `t_dst` alias the data buffer of `t_src`
> (no copy).
>
> Step 1: obtain the destination StorageImpl via
> `t_dst.unsafeGetTensorImpl()->unsafe_storage().unsafeGetStorageImpl()`.
>
> Step 2: `ET_CHECK_OR_RETURN_ERROR(t_dst.nbytes() == t_src.nbytes())`. On
> byte-count mismatch return `Error::InvalidArgument`.
>
> Step 3: `ET_CHECK_OR_RETURN_ERROR(t_src.mutable_data_ptr() != nullptr)`. If
> the source has a null data pointer, return `Error::InvalidArgument` (unlike
> the portable variant, an empty source is NOT tolerated here —
> `[spec:et:sem:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn]`).
>
> Step 4: point the destination storage at the source's buffer:
> `storage->set_data_ptr(at::DataPtr(t_src.mutable_data_ptr(),
> at::DeviceType::CPU))` (CPU memory, no deleter installed — the source
> retains ownership), then `storage->set_nbytes(t_src.nbytes())`.
>
> Step 5: return `Error::Ok`. After this, `t_dst` and `t_src` share the same
> underlying memory.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensor-has-valid-dim-order-fn]
> bool tensor_has_valid_dim_order(at::Tensor t)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensor-has-valid-dim-order-fn]
> ATen-mode implementation. Returns true iff `t`'s stride-derived dim order is
> a valid permutation.
>
> Step 1: declare a stack buffer `dim_order[kTensorDimensionLimit]`
> (`kTensorDimensionLimit` is the compile-time max supported rank).
>
> Step 2: call `get_dim_order(t, dim_order, t.dim())`
> (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]`).
> If it does not return `Error::Ok`, `ET_CHECK_OR_RETURN_FALSE` logs "Failed
> to retrieve dim order from tensor!" and returns false.
>
> Step 3: call `validate_dim_order(dim_order, t.dim())` (checks that the
> filled entries form a valid permutation of `0..dim-1` with every value in
> range). If it returns false, log an Error header plus one line per
> dimension `"    dim_order(<d>): <value>"` for `d` in `0..dim-1`, then return
> false.
>
> Step 4: otherwise return true.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensor-is-default-or-channels-last-dim-order-fn]
> inline bool tensor_is_default_or_channels_last_dim_order(at::Tensor t)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensor-is-default-or-channels-last-dim-order-fn]
> ATen-mode implementation (marked `inline`). Returns true iff `t`'s
> stride-derived dim order is either contiguous (default/NCHW-style) or
> channels-last (NHWC-style).
>
> Step 1: declare `dim_order[kTensorDimensionLimit]` and fill it via
> `get_dim_order(t, dim_order, t.dim())`
> (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]`);
> if that does not return `Error::Ok`, `ET_CHECK_OR_RETURN_FALSE` logs
> "Failed to retrieve dim order from tensor!" and returns false.
>
> Step 2: compute `ret_val = is_contiguous_dim_order(dim_order, t.dim()) ||
> is_channels_last_dim_order(dim_order, t.dim())`. `is_contiguous_dim_order`
> is true when the dim order is the ascending identity `[0,1,...,dim-1]`;
> `is_channels_last_dim_order` is true for the 4-D (or 5-D) NHWC permutation
> per the shared helper's contract.
>
> Step 3: if `ret_val` is false, log an Error header ("Expected tensor to
> have default or channels last dim order, but got") followed by one
> `"    dim_order(<d>): <value>"` line per dimension.
>
> Step 4: return `ret_val`.

> [spec:et:def:tensor-util-aten.executorch.et-runtime-namespace.tensors-have-same-dim-order-fn]
> bool tensors_have_same_dim_order( const executorch::aten::ArrayRef<executorch::aten::Tensor> tensor_list)

> [spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensors-have-same-dim-order-fn]
> ATen-mode implementation. Given an `ArrayRef<Tensor> tensor_list`, returns
> true iff every tensor shares one common layout — either all contiguous or
> all channels-last.
>
> Step 1: if `tensor_list.size() < 2`, return true (0 or 1 tensor trivially
> "agree").
>
> Step 2: declare two stack buffers `first_dim_order[kTensorDimensionLimit]`
> and `other_dim_order[kTensorDimensionLimit]`.
>
> Step 3: fill `first_dim_order` from `tensor_list[0]` via `get_dim_order`
> (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]`);
> if it does not return `Error::Ok`, `ET_CHECK_OR_RETURN_FALSE` logs "Failed
> to retrieve dim order from 1st input tensor!" and returns false. Initialize
> `all_contiguous = is_contiguous_dim_order(first_dim_order, dim)` and
> `all_channels_last = is_channels_last_dim_order(first_dim_order, dim)`.
>
> Step 4: for `i` in `1..size-1` (ascending): fill `other_dim_order` from
> `tensor_list[i]` via `get_dim_order`; on non-Ok result return false via
> `ET_CHECK_OR_RETURN_FALSE` ("Failed to retrieve dim order from <i>-th input
> tensor!"). Then `all_contiguous &&= is_contiguous_dim_order(other_dim_order,
> tensor_list[i].dim())` and `all_channels_last &&=
> is_channels_last_dim_order(other_dim_order, tensor_list[i].dim())`. Each
> tensor's own rank is used when interpreting its dim order.
>
> Step 5: `ET_CHECK_OR_RETURN_FALSE(all_contiguous || all_channels_last)`. If
> neither flag survived (the tensors disagree), log "<size> input tensors have
> different dim orders" and return false.
>
> Step 6: return `all_contiguous || all_channels_last` (true here).

