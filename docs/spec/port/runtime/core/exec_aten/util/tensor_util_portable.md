# runtime/core/exec_aten/util/tensor_util_portable.cpp

> [spec:et:def:tensor-util-portable.executorch.runtime.get-dim-order-fn]
> Error get_dim_order( const torch::executor::Tensor& tensor, executorch::aten::DimOrderType* out_dim_order, size_t out_dim_order_size)

> [spec:et:sem:tensor-util-portable.executorch.runtime.get-dim-order-fn]
> Portable (lean-mode) implementation. Copies the tensor's stored dim order
> into the caller buffer `out_dim_order` (length `out_dim_order_size`).
>
> Validation: `ET_CHECK_OR_RETURN_ERROR(out_dim_order_size ==
> tensor.dim_order().size())`. If the requested size does not equal the length
> of the tensor's `dim_order()` array, return `Error::InvalidArgument` and
> write nothing.
>
> On success: `std::memcpy(out_dim_order, tensor.dim_order().data(),
> tensor.dim_order().size() * sizeof(DimOrderType))`, then return `Error::Ok`.
> Unlike the ATen variant (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.get-dim-order-fn]`),
> the portable Tensor stores its dim order explicitly, so this is a direct
> byte copy rather than a stride-to-dim-order derivation.

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.copy-tensor-data-fn]
> Error copy_tensor_data( const torch::executor::Tensor& t_dst, const torch::executor::Tensor& t_src)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.copy-tensor-data-fn]
> Portable implementation. Copies the raw bytes of `t_src` into the
> pre-allocated storage of `t_dst`.
>
> Step 1: `ET_CHECK_OR_RETURN_ERROR(t_dst.const_data_ptr() != nullptr ||
> (t_dst.nbytes() == 0 && t_src.nbytes() == 0))`. The destination must have a
> non-null data pointer, unless both tensors are zero-byte (in which case a
> null destination is tolerated). Otherwise return `Error::InvalidArgument`
> ("ExecutionPlan input supposed to preallocated but has nullptr for data").
>
> Step 2: if `t_src.const_data_ptr() != nullptr` (a size-0-dimension source
> may be null, in which case nothing is copied): check
> `ET_CHECK_OR_RETURN_ERROR(t_dst.nbytes() == t_src.nbytes())`, returning
> `Error::InvalidArgument` on mismatch, then `std::memcpy(t_dst.mutable_data_ptr(),
> t_src.const_data_ptr(), t_src.nbytes())`.
>
> Step 3: return `Error::Ok`. Only data bytes are copied; metadata is
> unchanged. Compare the ATen variant's unconditional non-null destination
> check (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.copy-tensor-data-fn]`).

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn]
> void reset_data_ptr(const torch::executor::Tensor& tensor)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.reset-data-ptr-fn]
> Portable (lean-mode) implementation. Sets the tensor's data pointer to null
> by calling `tensor.unsafeGetTensorImpl()->set_data(nullptr)`. Sizes, strides,
> dim order, and dtype are left unchanged. Returns void; no failure path. In
> lean mode this does NOT deallocate any memory (the runtime allocator owns
> the buffer), unlike the ATen variant which also collapses sizes and resets
> the StorageImpl (`[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.reset-data-ptr-fn]`).

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.resize-tensor-impl-fn]
> Error resize_tensor_impl( torch::executor::TensorImpl* impl, torch::executor::ArrayRef<executorch::aten::SizesType> new_sizes)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.resize-tensor-impl-fn]
> Portable implementation. Free-function wrapper that forwards to
> `TensorResizerFriend::resize_tensor_impl(impl, new_sizes)`
> (`[spec:et:sem:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn]`)
> and returns its `Error` result directly. It exists only to reach the
> private/friend resize entry point on `TensorImpl`; the friend class holds
> the actual resize logic. Most callers should use the higher-level
> `resize_tensor()` wrapper instead.

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.set-tensor-data-fn]
> ET_NODISCARD Error set_tensor_data( const torch::executor::Tensor& t, void* buffer, size_t buffer_size)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.set-tensor-data-fn]
> Portable implementation. Points the tensor's data pointer at a
> caller-supplied buffer. Return value is `ET_NODISCARD`.
>
> Step 1: `ET_CHECK_OR_RETURN_ERROR(buffer_size >= t.nbytes())`. If the
> buffer is smaller than the tensor's required byte count, return
> `Error::InvalidArgument` and leave the tensor unmodified.
>
> Step 2: `t.unsafeGetTensorImpl()->set_data(buffer)` — installs `buffer` as
> the tensor's data pointer without transferring ownership (the runtime does
> not free it).
>
> Step 3: return `Error::Ok`.

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn]
> Error share_tensor_data( const torch::executor::Tensor& t_dst, const torch::executor::Tensor& t_src)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.share-tensor-data-fn]
> Portable implementation. Makes `t_dst` alias the data buffer of `t_src`
> (no copy).
>
> Step 1: `ET_CHECK_OR_RETURN_ERROR(t_dst.nbytes() == t_src.nbytes())`. On
> byte-count mismatch return `Error::InvalidArgument`.
>
> Step 2: `ET_CHECK_OR_RETURN_ERROR(t_src.mutable_data_ptr() != nullptr ||
> t_src.nbytes() == 0)`. The source may be null only when it is empty
> (0 bytes); otherwise return `Error::InvalidArgument`. (Contrast the ATen
> variant, which requires a non-null source unconditionally —
> `[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.internal.share-tensor-data-fn]`.)
>
> Step 3: compute the pointer to install: `t_src_data_ptr = (t_src.numel() ==
> 0) ? nullptr : t_src.mutable_data_ptr()` — an empty source explicitly
> installs a null data pointer.
>
> Step 4: `t_dst.unsafeGetTensorImpl()->set_data(t_src_data_ptr)`, then return
> `Error::Ok`. After this `t_dst` shares the source's memory (or is null if
> the source was empty); ownership is not transferred.

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend]
> class TensorResizerFriend final

> [spec:et:def:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn]
> ET_NODISCARD static Error resize_tensor_impl( executorch::aten::TensorImpl* impl, executorch::aten::ArrayRef<executorch::aten::SizesType> new_sizes)

> [spec:et:sem:tensor-util-portable.executorch.runtime.internal.tensor-resizer-friend.resize-tensor-impl-fn]
> Portable implementation. `TensorResizerFriend` is a friend of `TensorImpl`
> whose sole `static` method exposes the impl's private resize entry point.
> The method calls `impl->internal_resize_contiguous(new_sizes)` and returns
> its `Error` result directly. That impl method enforces the rank-immutability
> and capacity constraints (returning e.g. `Error::NotSupported` on a rank
> change) and rewrites sizes with contiguous strides. Return value is
> `ET_NODISCARD`.

> [spec:et:def:tensor-util-portable.executorch.runtime.tensor-has-valid-dim-order-fn]
> bool tensor_has_valid_dim_order(torch::executor::Tensor t)

> [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-has-valid-dim-order-fn]
> Portable implementation. Reads `t.dim_order()` directly (no intermediate
> buffer) and calls `validate_dim_order(t.dim_order().data(),
> t.dim_order().size())`, which checks the stored dim order is a valid
> permutation of `0..dim-1`. If validation fails, log an Error header ("Tensor
> dim order is not valid:") followed by one `"    dim_order(<d>): <value>"`
> line per dimension `d` in `0..dim-1`, then return false. Otherwise return
> true.

> [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-channels-last-dim-order-fn]
> bool tensor_is_channels_last_dim_order(torch::executor::Tensor t)

> [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-channels-last-dim-order-fn]
> Portable implementation. Computes `ret_val =
> is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size())`,
> which is true for the NHWC-style channels-last permutation per the shared
> helper's contract. If false, log an Error header ("Expected tensor to have
> channels last dim order, but got") followed by one `"    dim_order(<d>):
> <value>"` line per dimension. Return `ret_val`.

> [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-default-dim-order-fn]
> bool tensor_is_default_dim_order(torch::executor::Tensor t)

> [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-dim-order-fn]
> Portable implementation. Computes `ret_val =
> is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size())`, which
> is true when the dim order is the ascending identity `[0,1,...,dim-1]`
> (default/contiguous layout). If false, log an Error header ("Expected tensor
> to have default dim order, but got") followed by one `"    dim_order(<d>):
> <value>"` line per dimension. Return `ret_val`.

> [spec:et:def:tensor-util-portable.executorch.runtime.tensor-is-default-or-channels-last-dim-order-fn]
> bool tensor_is_default_or_channels_last_dim_order(torch::executor::Tensor t)

> [spec:et:sem:tensor-util-portable.executorch.runtime.tensor-is-default-or-channels-last-dim-order-fn]
> Portable implementation. Computes `ret_val =
> is_contiguous_dim_order(t.dim_order().data(), t.dim_order().size()) ||
> is_channels_last_dim_order(t.dim_order().data(), t.dim_order().size())` —
> true when the stored dim order is either the ascending identity (default) or
> the NHWC channels-last permutation. If false, log an Error header ("Expected
> tensor to have default or channels last dim order, but got") followed by one
> `"    dim_order(<d>): <value>"` line per dimension. Return `ret_val`.

> [spec:et:def:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn]
> bool tensors_have_same_dim_order( const executorch::aten::ArrayRef<executorch::aten::Tensor> tensor_list)

> [spec:et:sem:tensor-util-portable.executorch.runtime.tensors-have-same-dim-order-fn]
> Portable implementation. Given `ArrayRef<Tensor> tensor_list`, returns true
> iff all tensors share one common layout — all contiguous or all
> channels-last.
>
> Step 1: if `tensor_list.size() < 2`, return true.
>
> Step 2: initialize `all_contiguous = true` and `all_channels_last = true`.
>
> Step 3: for `i` in `0..size-1` (ascending, including index 0, reading each
> tensor's stored `dim_order()` directly): `all_contiguous &&=
> is_contiguous_dim_order(tensor_list[i].dim_order().data(),
> tensor_list[i].dim_order().size())` and `all_channels_last &&=
> is_channels_last_dim_order(tensor_list[i].dim_order().data(),
> tensor_list[i].dim_order().size())`.
>
> Step 4: `ET_CHECK_OR_RETURN_FALSE(all_contiguous || all_channels_last)`. If
> neither flag survived, log "<size> input tensors have different dim orders"
> and return false.
>
> Step 5: return true. (Note: this portable variant unconditionally returns
> the literal `true` after the check passes, rather than `all_contiguous ||
> all_channels_last` as the ATen variant does —
> `[spec:et:sem:tensor-util-aten.executorch.et-runtime-namespace.tensors-have-same-dim-order-fn]`;
> the values are equal at that point since the check already guaranteed at
> least one is true.)

