# kernels/portable/cpu/op_glu.cpp

> [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor]
> struct SplitGLUInputTensor {
>   SizesArray half_sizes;
>   TensorImpl first_half_impl;
>   TensorImpl second_half_impl;
>   Tensor first_half;
>   Tensor second_half;
> }

> [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn]
> static SizesArray get_half_sizes(const Tensor& self, int64_t dim)

> [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn]
> Static helper that computes the shape of one GLU half. Copies `self.sizes()`
> into a `SizesArray` (a `std::array<SizesType, kTensorDimensionLimit>`) and then
> halves the entry at index `dim`: `half_sizes[dim] = self.size(dim) / 2` (integer
> division; `dim` is assumed already non-negative and the split dimension is
> assumed even, per the caller's `check_glu_args`). Returns the resulting
> `SizesArray` (only the first `self.dim()` entries are meaningful).

> [spec:et:def:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn]
> SplitGLUInputTensor::SplitGLUInputTensor(const Tensor& self, int64_t dim)

> [spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn]
> Constructor that splits `self` in half along `dim` into two views without
> copying data. Given `self` and a non-negative `dim`:
> 1. `half_sizes` = `get_half_sizes(self, dim)` per
>    `[spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.get-half-sizes-fn]`
>    (same shape as `self` but with the `dim` axis halved).
> 2. Build `first_half_impl`, a `TensorImpl` aliasing `self`'s data: same
>    scalar_type, same `self.dim()` rank, sizes = `half_sizes`, data pointer =
>    `self.mutable_data_ptr()` (start of the buffer), and the same dim_order,
>    strides, and shape_dynamism as `self`. This view covers the first half of the
>    split dimension.
> 3. Build `second_half_impl`, identical to the first but with its data pointer
>    advanced to the start of the second half:
>    `base + strides[dim] * (size(dim)/2) * element_size()` bytes. This is the
>    offset (in elements, times element size) of index `size(dim)/2` along `dim`.
> 4. Wrap the two impls as `first_half` and `second_half` `Tensor`s.
> Both halves share `self`'s underlying storage (aliasing views); mutating them
> would mutate `self`. The construction assumes `self.dim() <= kTensorDimensionLimit`
> (checked by the caller).

> [spec:et:def:op-glu.torch.executor.native.glu-out-fn]
> Tensor& glu_out( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, Tensor& out)

> [spec:et:sem:op-glu.torch.executor.native.glu-out-fn]
> Public GLU entry point: `out = self_a * sigmoid(self_b)` where `self_a` and
> `self_b` are the two halves of `self` split along `dim`. Returns `out`.
>
> Steps:
> 1. Resize `out`: ET_KERNEL_CHECK `resize_glu_out(self, dim, out) == Error::Ok`
>    per `[spec:et:sem:activation-ops-util.resize-glu-out]` (output shape equals
>    `self`'s shape with the `dim` axis halved). On failure Error::InvalidArgument,
>    return `out`.
> 2. ET_KERNEL_CHECK: `tensors_have_same_dim_order(self, out)`; else
>    Error::InvalidArgument, return `out`.
> 3. ET_KERNEL_CHECK: `check_glu_args(self, dim, out)` per
>    `[spec:et:sem:activation-ops-util.check-glu-args]`, which requires `self` to
>    be a floating dtype, `dim` (after normalization) a valid axis whose size is
>    even, `out` a floating dtype, and `out`'s shape to be `self`'s with `dim`
>    halved. On failure Error::InvalidArgument, return `out`.
> 4. Normalize the dim: `non_negative_dim = dim < 0 ? dim + self.dim() : dim`.
> 5. Double-dispatch: outer switch over `self.scalar_type()` (CTYPE_IN) and inner
>    switch over `out.scalar_type()` (CTYPE_OUT), both restricted to FLOATHBF16 =
>    {Half, Float, Double, BFloat16}; call
>    `glu_out_tensor<CTYPE_IN, CTYPE_OUT>(ctx, self, non_negative_dim, out)` per
>    `[spec:et:sem:op-glu.torch.executor.native.glu-out-tensor-fn]`.
> 6. Return `out`.

> [spec:et:def:op-glu.torch.executor.native.glu-out-tensor-fn]
> Tensor& glu_out_tensor( KernelRuntimeContext& ctx, const Tensor& self, int64_t dim, Tensor& out)

> [spec:et:sem:op-glu.torch.executor.native.glu-out-tensor-fn]
> Templated on `CTYPE_IN`/`CTYPE_OUT`. Computes the GLU on an already-validated
> `self` and non-negative `dim`, writing to `out`. Returns `out`.
>
> Steps:
> 1. ET_KERNEL_CHECK: `self.dim() <= kTensorDimensionLimit`; else
>    Error::InvalidArgument, return `out`.
> 2. Construct `split_input = SplitGLUInputTensor(self, dim)` per
>    `[spec:et:sem:op-glu.torch.executor.native.split-glu-input-tensor.split-glu-input-tensor-fn]`,
>    yielding `first_half` and `second_half` views of `self` (each with the
>    `dim` axis halved).
> 3. Select the compute type: if `self.scalar_type()` is floating, use it;
>    otherwise use Float. (In practice `self` is always floating, guarded by the
>    caller.)
> 4. Dispatch over the compute type in FLOATHBF16 and apply the bitensor
>    elementwise function with broadcasting per
>    `[spec:et:sem:elementwise-util...apply-bitensor-elementwise-fn]`, passing
>    `SupportNoncontiguousInputTensors()` so the two non-contiguous half-views are
>    handled correctly. Both halves are read over the FLOATHBF16 input set and the
>    output over FLOATHBF16. For each aligned pair `(val_a, val_b)` (a from
>    `first_half`, b from `second_half`) compute
>    `val_a * (1 / (1 + exp(-val_b)))` = `val_a * sigmoid(val_b)`, storing the
>    result (as `CTYPE_COMPUTE`) into `out`.
> 5. Return `out`.

