# backends/xnnpack/runtime/XNNCompiler.cpp, backends/xnnpack/runtime/XNNCompiler.h

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator]
> class CompileAllocator {
>   std::vector<std::unique_ptr<uint8_t[]>> temporaries_;
> }

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn]
> void* allocateTemporary(size_t size)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn]
> Allocates a scratch buffer of `size` bytes whose lifetime is bound to the
> `CompileAllocator` instance (i.e. freed automatically when the allocator is
> destroyed at the end of compilation).
>
> Steps:
> 1. Allocate a raw byte buffer of `size` bytes (C++: `new uint8_t[size]`). No
>    alignment guarantee beyond that of the platform's default array-new;
>    contents are uninitialized.
> 2. Take ownership of the buffer by appending it to the allocator's internal
>    `temporaries_` vector of owning smart pointers, so the buffer stays alive
>    until the allocator is dropped and is then freed exactly once.
> 3. Return the raw pointer to the newly allocated buffer.
>
> There is no failure path in the source: allocation failure would throw
> (C++ `std::bad_alloc`); a Rust port may instead allocate on the arena/`Vec`
> that owns these temporaries and return a pointer/handle with the same
> ownership semantics. `size == 0` is permitted (returns a valid, non-owning-
> distinguishable pointer that must not be dereferenced). Used by
> `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]`
> to hold bf16-converted block scales.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn]
> void convertF32TensorToBF16( const float* f32_data, uint16_t* bf16_data_out, size_t numel)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn]
> Converts an array of `numel` IEEE-754 fp32 values (`f32_data`) into bf16
> (stored as `uint16_t` in `bf16_data_out`) using round-to-nearest via a
> multiplicative pre-scale followed by high-half truncation.
>
> For each index i in [0, numel):
> 1. Compute `f32_adjusted = f32_data[i] * 1.00389105f` (a single fp32
>    multiply; the constant scales 1+2^-8 to 1+2^-7 so that truncating the
>    low 16 bits approximates round-to-nearest rather than round-toward-zero).
> 2. Reinterpret the 32 bits of `f32_adjusted` as a `uint32_t` `f32_bits`
>    (bitwise copy, e.g. `memcpy`; no numeric conversion).
> 3. Write `bf16_data_out[i] = (uint16_t)(f32_bits >> 16)` — the top 16 bits
>    (sign + exponent + 7 high mantissa bits), discarding the low 16 bits.
>
> Iteration is in ascending index order; input and output are separate buffers
> (caller must size `bf16_data_out` for `numel` uint16 entries). NaN/inf: the
> multiply and shift preserve the sign/exponent pattern, so infinities map to
> bf16 infinities and NaNs to bf16 NaNs (payload may change / a NaN may
> collapse to inf-like patterns only in the pathological case where the scaled
> mantissa overflows the exponent). No rounding of denormals beyond the
> truncation described. `numel == 0` writes nothing.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-arg-max-pooling2d-node-fn]
> Error defineArgMaxPooling2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-arg-max-pooling2d-node-fn]
> Defines an argmax pooling 2d op (produces both pooled values and their
> indices) into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNArgMaxPooling2d()`.
> 2. REMAP_ID `input_id` -> amp_input, `output_value_id` -> amp_out_val,
>    `output_index_id` -> amp_out_idx.
> 3. Call `xnn_define_argmax_pooling_2d(subgraph, padding_top, padding_right,
>    padding_bottom, padding_left, pooling_height, pooling_width, amp_input,
>    amp_out_val, amp_out_idx, flags)` reading each field from `graph_node`.
>    (No stride/dilation/min-max: argmax pooling uses a non-overlapping window
>    with stride equal to the pool size.)
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    argmaxpool2d node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-avg-pooling2d-node-fn]
> Error defineAvgPooling2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-avg-pooling2d-node-fn]
> Defines an average pooling 2d op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNAvgPooling2d()`.
> 2. `min_max = getOutputMinMax(node)`.
> 3. REMAP_ID `input_id`, `output_id` into ap_input/ap_output.
> 4. Call `xnn_define_average_pooling_2d(subgraph, padding_top, padding_right,
>    padding_bottom, padding_left, pooling_height, pooling_width,
>    stride_height, stride_width, min_max.first, min_max.second, ap_input,
>    ap_output, flags)` reading each field from `graph_node`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    average pooling node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-batch-matrix-multiply-node-fn]
> Error defineBatchMatrixMultiplyNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-batch-matrix-multiply-node-fn]
> Defines a batch matrix multiply op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNBatchMatrixMultiply()`.
> 2. REMAP_ID `input1_id`, `input2_id`, `output_id` into bmm_in1/bmm_in2/
>    bmm_out.
> 3. Call `xnn_define_batch_matrix_multiply(subgraph, bmm_in1, bmm_in2,
>    bmm_out, graph_node->flags())`. (Transpose behavior, if any, is carried in
>    `flags`.)
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    BMM node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate2-node-fn]
> Error defineConcatenate2Node( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate2-node-fn]
> Defines a 2-input concatenate op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConcatenate2()`.
> 2. REMAP_ID `input1_id`, `input2_id`, `output_id` into cat2_in1/cat2_in2/
>    cat2_out.
> 3. Call `xnn_define_concatenate2(subgraph, graph_node->axis(), cat2_in1,
>    cat2_in2, cat2_out, graph_node->flags())`. `axis` is the concatenation
>    dimension.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    cat2 node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate3-node-fn]
> Error defineConcatenate3Node( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate3-node-fn]
> Defines a 3-input concatenate op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConcatenate3()`.
> 2. REMAP_ID `input1_id`, `input2_id`, `input3_id`, `output_id` into
>    cat3_in1..cat3_in3, cat3_out.
> 3. Call `xnn_define_concatenate3(subgraph, graph_node->axis(), cat3_in1,
>    cat3_in2, cat3_in3, cat3_out, graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    cat3 node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate4-node-fn]
> Error defineConcatenate4Node( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate4-node-fn]
> Defines a 4-input concatenate op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConcatenate4()`.
> 2. REMAP_ID `input1_id`..`input4_id`, `output_id` into cat4_in1..cat4_in4,
>    cat4_out.
> 3. Call `xnn_define_concatenate4(subgraph, graph_node->axis(), cat4_in1,
>    cat4_in2, cat4_in3, cat4_in4, cat4_out, graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    cat4 node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate5-node-fn]
> Error defineConcatenate5Node( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate5-node-fn]
> Defines a 5-input concatenate op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConcatenate5()`.
> 2. REMAP_ID `input1_id`..`input5_id`, `output_id` into cat5_in1..cat5_in5,
>    cat5_out.
> 3. Call `xnn_define_concatenate5(subgraph, graph_node->axis(), cat5_in1,
>    cat5_in2, cat5_in3, cat5_in4, cat5_in5, cat5_out, graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    cat5 node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv-transpose2d-node-fn]
> Error defineConvTranspose2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv-transpose2d-node-fn]
> Defines a transposed (deconvolution) 2d op into the subgraph. `noexcept`;
> `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConvTranspose2d()`.
> 2. `min_max = getOutputMinMax(node)`.
> 3. REMAP_ID `input1_id`, `filter_id`, `bias_id`, `output_id` into
>    dconv_input1/dconv_filter/dconv_bias/dconv_output.
> 4. Call `xnn_define_deconvolution_2d(subgraph, padding_top, padding_right,
>    padding_bottom, padding_left, adjustment_height, adjustment_width,
>    kernel_height, kernel_width, subsampling_height, subsampling_width,
>    dilation_height, dilation_width, groups, group_input_channels,
>    group_output_channels, min_max.first, min_max.second, dconv_input1,
>    dconv_filter, dconv_bias, dconv_output, flags)` reading each field from
>    `graph_node`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    deconvolution node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv2d-node-fn]
> Error defineConv2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv2d-node-fn]
> Defines a 2d convolution op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNConv2d()`.
> 2. `min_max = getOutputMinMax(node)`.
> 3. REMAP_ID `input1_id`, `filter_id`, `bias_id`, `output_id` into
>    conv_input1/conv_filter/conv_bias/conv_output.
> 4. Call `xnn_define_convolution_2d(subgraph, padding_top, padding_right,
>    padding_bottom, padding_left, kernel_height, kernel_width,
>    subsampling_height, subsampling_width, dilation_height, dilation_width,
>    groups, group_input_channels, group_output_channels, min_max.first,
>    min_max.second, conv_input1, conv_filter, conv_bias, conv_output, flags)`
>    reading each field from `graph_node`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    convolution node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]
> Error defineConvertNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* flatbuffer_graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]
> Defines an XNNPACK convert (dtype cast / quantize) op into the subgraph.
> `noexcept`; `flatbuffer_graph`/`graph` is unused except under Kleidi.
>
> Steps:
> 1. `graph_node = node->as_XNNConvert()`.
> 2. `flags = graph_node->flags()` (int32).
> 3. If `ENABLE_XNNPACK_KLEIDI` is defined and
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn]`
>    returns true for this node, OR-in the flag `0x00000100`
>    (XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM) and log a debug message. (Without
>    Kleidi this step is absent.)
> 4. Remap input id: look up `graph_node->input_id()` in `remapped_ids`; if
>    missing return Error::Internal ("Remapped id not found"). Same for
>    `graph_node->output_id()`.
> 5. Call `xnn_define_convert(subgraph, cvt_input_id, cvt_output_id, flags)`.
> 6. If the returned status is not `xnn_status_success`, return Error::Internal
>    ("Failed to create convert node", with node->debug_handle() and the
>    status string). Otherwise return Error::Ok.
>
> Note: every "REMAP_ID" reference below denotes this same lookup with the
> same missing-key Error::Internal behavior; `XNN_INVALID_VALUE_ID` is
> pre-seeded to map to itself so optional (invalid) ids remap trivially.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-copy-node-fn]
> Error defineCopyNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-copy-node-fn]
> Defines a copy op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNCopy()`.
> 2. REMAP_ID `input_id`, `output_id` into copy_input/copy_output.
> 3. Call `xnn_define_copy(subgraph, copy_input, copy_output,
>    graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    copy node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-depthwise-conv2d-node-fn]
> Error defineDepthwiseConv2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-depthwise-conv2d-node-fn]
> Defines a depthwise 2d convolution op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNDepthwiseConv2d()`.
> 2. `min_max = getOutputMinMax(node)`.
> 3. REMAP_ID `input1_id`, `filter_id`, `bias_id`, `output_id` into
>    dw_input1/dw_filter/dw_bias/dw_output.
> 4. Call `xnn_define_depthwise_convolution_2d(subgraph, padding_top,
>    padding_right, padding_bottom, padding_left, kernel_height, kernel_width,
>    subsampling_height, subsampling_width, dilation_height, dilation_width,
>    depth_multiplier, input_channels, min_max.first, min_max.second,
>    dw_input1, dw_filter, dw_bias, dw_output, flags)` where
>    `depth_multiplier = group_output_channels / group_input_channels`
>    (integer division) and `input_channels = groups` (for depthwise conv the
>    input channel count equals the group count).
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    depthwise convolution node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-exp-node-fn]
> Error defineExpNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-exp-node-fn]
> Defines an elementwise exp op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNExp()`.
> 2. REMAP_ID `input_id`, `output_id` into exp_input/exp_output.
> 3. Call `xnn_define_exp(subgraph, exp_input, exp_output,
>    graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    exp node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-fully-connected-node-fn]
> Error defineFullyConnectedNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-fully-connected-node-fn]
> Defines a fully-connected (linear) op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNFullyConnected()`.
> 2. Compute `min_max = getOutputMinMax(node)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]`.
> 3. REMAP_ID (see
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]`
>    for the lookup+missing-key semantics) for `input1_id`, `filter_id`,
>    `bias_id`, `output_id` into fc_input1/fc_filter/fc_bias/fc_output.
> 4. Call `xnn_define_fully_connected(subgraph, min_max.first, min_max.second,
>    fc_input1, fc_filter, fc_bias, fc_output, graph_node->flags())`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    linear node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn]
> Error defineGenericBinaryNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const fb_xnnpack::_XNNNode2x1* graph_node, xnn_binary_operator op_type, const struct xnn_binary_params* params, fb_xn...

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn]
> Shared implementation for all two-input/single-output binary ops. Takes the
> serialized `_XNNNode2x1*` `graph_node` (which exposes `input1_id`,
> `input2_id`, `output_id`, `flags`), an `xnn_binary_operator` `op_type`, an
> optional `xnn_binary_params*` `params` (null for parameterless ops), plus
> `node_type`/`debug_handle` for error messages. `noexcept`.
>
> Steps:
> 1. REMAP_ID `graph_node->input1_id()` -> bin_in1, `input2_id()` -> bin_in2,
>    `output_id()` -> bin_out (missing-key -> Error::Internal).
> 2. Call `xnn_define_binary(subgraph, op_type, params, bin_in1, bin_in2,
>    bin_out, graph_node->flags())`. XNNPACK applies NumPy-style broadcasting
>    between the two inputs.
> 3. If status != xnn_status_success, return Error::Internal ("Failed to create
>    <node_type> node"). Else Error::Ok.
>
> The concrete binary node definers are macro-generated and dispatched by
> `getDefineNodeFunc`:
> - With min/max output-clamp params (`params.output_min/output_max =
>   getOutputMinMax(node)`): Add->xnn_binary_add,
>   Subtract->xnn_binary_subtract, Multiply->xnn_binary_multiply,
>   Div->xnn_binary_divide.
> - Without params (`params = null`): Minimum->xnn_binary_minimum,
>   Maximum->xnn_binary_maximum.
> Each generated definer reads `node->as_XNN<Name>()` as the `_XNNNode2x1*`,
> fills `params` (if any), and forwards `node->xnode_union_type()` and
> `node->debug_handle()`.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn]
> Error defineGenericUnaryNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, uint32_t input_id, uint32_t output_id, uint32_t flags, xnn_unary_operator op_type, const union xnn_unary_params* param...

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn]
> Shared implementation for all single-input/single-output unary ops. Takes a
> pre-extracted `input_id`, `output_id`, `flags`, an `xnn_unary_operator`
> `op_type`, an optional `xnn_unary_params*` `params` (may be null for
> parameterless ops), plus `node_type`/`debug_handle` for error messages.
> `noexcept`.
>
> Steps:
> 1. REMAP_ID `input_id` -> remapped_input, `output_id` -> remapped_output
>    (missing-key -> Error::Internal, per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]`).
> 2. Call `xnn_define_unary(subgraph, op_type, params, remapped_input,
>    remapped_output, flags)`.
> 3. If status != xnn_status_success, return Error::Internal ("Failed to create
>    <node_type> node"). Else Error::Ok.
>
> The concrete unary node definers are macro-generated. `getDefineNodeFunc`
> dispatches the following XNode types to unary definers that call this helper:
> - No-params (`params = null`): Sigmoid->xnn_unary_sigmoid,
>   Floor->xnn_unary_floor, SquareRoot->xnn_unary_square_root,
>   ReciprocalSquareRoot->xnn_unary_reciprocal_square_root,
>   Ceiling->xnn_unary_ceiling, Gelu->xnn_unary_gelu,
>   Hardswish->xnn_unary_hardswish, Log->xnn_unary_log,
>   Negate->xnn_unary_negate, Square->xnn_unary_square, Abs->xnn_unary_abs,
>   Sin->xnn_unary_sine, Cos->xnn_unary_cosine.
> - With min/max clamp params (`params.clamp.min/max = getOutputMinMax(node)`):
>   Clamp->xnn_unary_clamp.
> - LeakyReLU: `params.leaky_relu.negative_slope =
>   graph_node->negative_slope()`, op_type xnn_unary_leaky_relu.
> - ELU: `params.elu.alpha = graph_node->alpha()`, op_type xnn_unary_elu.
> Each generated definer reads its op-specific serialized union
> (`node->as_XNN<Name>()`), fills `params` (if any), and passes
> `input_id()`/`output_id()`/`flags()`, `node->xnode_union_type()`, and
> `node->debug_handle()` into this helper.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-global-avg-pooling2d-node-fn]
> Error defineGlobalAvgPooling2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-global-avg-pooling2d-node-fn]
> Defines a global average pooling 2d op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNGlobalAvgPooling2d()`.
> 2. `min_max = getOutputMinMax(node)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]`.
> 3. REMAP_ID `input_id`, `output_id` into gap_input/gap_output.
> 4. Call `xnn_define_global_average_pooling_2d(subgraph, min_max.first,
>    min_max.second, gap_input, gap_output, graph_node->flags())`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    global average pooling node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-max-pooling2d-node-fn]
> Error defineMaxPooling2dNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-max-pooling2d-node-fn]
> Defines a max pooling 2d op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNMaxPooling2d()`.
> 2. `min_max = getOutputMinMax(node)`.
> 3. REMAP_ID `input_id`, `output_id` into mp_input/mp_output.
> 4. Call `xnn_define_max_pooling_2d(subgraph, padding_top, padding_right,
>    padding_bottom, padding_left, pooling_height, pooling_width,
>    stride_height, stride_width, dilation_height, dilation_width,
>    min_max.first, min_max.second, mp_input, mp_output, flags)` reading each
>    field from `graph_node`.
> 5. If status != xnn_status_success, return Error::Internal ("Failed to create
>    maxpool2d node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn]
> Error defineNotImplementedNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn]
> Fallback handler for serialized node types the compiler does not implement.
> `noexcept`. Unconditionally returns Error::NotImplemented with message
> "Unhandled node type: <name>" where <name> is the string name of
> `node->xnode_union_type()`. Never touches the subgraph or remapped_ids. Used
> as the default target of
> `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]`.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-p-re-lu-node-fn]
> Error definePReLUNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-p-re-lu-node-fn]
> Defines a PReLU op (parametric ReLU with a learned negative-slope tensor)
> into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNPReLU()`.
> 2. REMAP_ID `input1_id` -> prelu_input1 (activations), `input2_id` ->
>    prelu_input2 (slope tensor), `output_id` -> prelu_output.
> 3. Call `xnn_define_prelu(subgraph, prelu_input1, prelu_input2,
>    prelu_output, graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    prelu node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-softmax-node-fn]
> Error defineSoftmaxNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-softmax-node-fn]
> Defines a softmax op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNSoftmax()`.
> 2. REMAP_ID `input_id`, `output_id` into sm_input/sm_output (missing-key ->
>    Error::Internal, per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]`).
> 3. Call `xnn_define_softmax(subgraph, sm_input, sm_output,
>    graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    softmax node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-constant-pad-node-fn]
> Error defineStaticConstantPadNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-constant-pad-node-fn]
> Defines a static constant-value pad op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNStaticConstantPad()`.
> 2. Require both `pre_paddings()` and `post_paddings()` non-null; else
>    Error::InvalidProgram ("pre_paddings or post_paddings is null").
> 3. `pre_paddings_dims = flatbufferDimsToVector(pre_paddings)` and
>    `post_paddings_dims = flatbufferDimsToVector(post_paddings)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]`.
> 4. REMAP_ID `input_id`, `output_id` into scp_input/scp_output.
> 5. Call `xnn_define_static_constant_pad(subgraph, pre_paddings_dims.data(),
>    post_paddings_dims.data(), graph_node->padding_value(), scp_input,
>    scp_output, graph_node->flags())`.
> 6. If status != xnn_status_success, return Error::Internal ("Failed to create
>    StaticConstantPad node"). Else Error::Ok.
>
> Note: the pre/post padding array lengths (implicitly the input rank) are
> passed to XNNPACK only via the pointers; the source does not pass an explicit
> count, so XNNPACK infers it from the input tensor's rank.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-reshape-node-fn]
> Error defineStaticReshapeNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-reshape-node-fn]
> Defines a static reshape op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNStaticReshape()`.
> 2. Require `graph_node->new_shape()` non-null; else Error::InvalidProgram
>    ("StaticReshape: new_shape is null").
> 3. `dims_data = flatbufferDimsToVector(new_shape)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]`.
> 4. REMAP_ID `input_id`, `output_id` into sr_input/sr_output.
> 5. Call `xnn_define_static_reshape(subgraph, dims_data.size(),
>    dims_data.data(), sr_input, sr_output, graph_node->flags())`.
> 6. If status != xnn_status_success, return Error::Internal (message text
>    reads "Failed to create squeeze node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-resize-bilinear2-d-node-fn]
> Error defineStaticResizeBilinear2DNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-resize-bilinear2-d-node-fn]
> Defines a static bilinear resize 2d op into the subgraph. `noexcept`; `graph`
> unused.
>
> Steps:
> 1. `graph_node = node->as_XNNStaticResizeBilinear2D()`.
> 2. REMAP_ID `input_id`, `output_id` into rb_input/rb_output.
> 3. Call `xnn_define_static_resize_bilinear_2d(subgraph,
>    graph_node->new_height(), graph_node->new_width(), rb_input, rb_output,
>    graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    StaticResizeBilinear2DNode node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-slice-node-fn]
> Error defineStaticSliceNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-slice-node-fn]
> Defines a static slice op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNStaticSlice()`.
> 2. Require both `offsets()` and `sizes()` non-null; else Error::InvalidProgram
>    ("StaticSlice: offsets or sizes is null").
> 3. `offsets = flatbufferDimsToVector(offsets)` and `sizes =
>    flatbufferDimsToVector(sizes)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]`
>    (both size_t vectors).
> 4. Require `offsets.size() == sizes.size()`; else Error::InvalidProgram
>    (mismatched lengths).
> 5. REMAP_ID `input_id`, `output_id` into ss_input/ss_output.
> 6. Call `xnn_define_static_slice(subgraph, offsets.size(), offsets.data(),
>    sizes.data(), ss_input, ss_output, graph_node->flags())` (rank =
>    offsets.size()).
> 7. If status != xnn_status_success, return Error::Internal ("Failed to create
>    static slice node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-transpose-node-fn]
> Error defineStaticTransposeNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-transpose-node-fn]
> Defines a static transpose (permute) op into the subgraph. `noexcept`;
> `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNStaticTranspose()`.
> 2. Require `graph_node->perm()` non-null; else Error::InvalidProgram
>    ("StaticTranspose: perm is null").
> 3. `dims_data = flatbufferDimsToVector(perm)` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]`
>    (the permutation as size_t).
> 4. REMAP_ID `input_id`, `output_id` into st_input/st_output.
> 5. Call `xnn_define_static_transpose(subgraph, dims_data.size(),
>    dims_data.data(), st_input, st_output, graph_node->flags())`.
> 6. If status != xnn_status_success, return Error::Internal ("Failed to create
>    static transpose node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-tanh-node-fn]
> Error defineTanhNode( xnn_subgraph_t subgraph_ptr, const std::unordered_map<uint32_t, uint32_t>& remapped_ids, const NodePtr node, const fb_xnnpack::XNNGraph* graph) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tanh-node-fn]
> Defines an elementwise tanh op into the subgraph. `noexcept`; `graph` unused.
>
> Steps:
> 1. `graph_node = node->as_XNNTanh()`.
> 2. REMAP_ID `input_id`, `output_id` into tanh_input/tanh_output.
> 3. Call `xnn_define_tanh(subgraph, tanh_input, tanh_output,
>    graph_node->flags())`.
> 4. If status != xnn_status_success, return Error::Internal ("Failed to create
>    tanh node"). Else Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]
> Error defineTensor( xnn_subgraph_t subgraph_ptr, std::unordered_map<uint32_t, uint32_t>& remapped_ids, ValuePtr value, GraphPtr flatbuffer_graph, const uint8_t* constant_data_ptr, uint64_t constant_data_size, std::vector<uint32_t>& input...

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]
> Defines one serialized XValue (tensor or quantized tensor) into the XNNPACK
> subgraph, records the mapping from its serialized id to the newly generated
> XNNPACK value id in `remapped_ids`, and appends external input/output ids to
> `input_ids`/`output_ids`. Returns `Error::Ok` on success, else an `Error`.
>
> Steps:
>
> 1. Discriminate `value->xvalue_union_type()`:
>    - `XNNTensorValue`: `tensor_value = as_XNNTensorValue`; `qtensor_value =
>      null`.
>    - `XNNQuantizedTensorValue`: `qtensor_value = as_XNNQuantizedTensorValue`;
>      `tensor_value = qtensor_value->tensor_value()`.
>    - Any other type: return Error::NotImplemented ("Unhandled value type").
> 2. If `tensor_value` is null: return Error::InvalidProgram ("Deserialized
>    tensor is null").
> 3. Validate flags: only `XNN_VALUE_FLAG_EXTERNAL_INPUT` and
>    `XNN_VALUE_FLAG_EXTERNAL_OUTPUT` may be set; if any other bit is set,
>    return Error::InvalidProgram ("unsupported flag bits").
> 4. Build `dims_data` (`vector<size_t>`): if `tensor_value->dims()` is
>    non-null, convert it per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]`;
>    else leave empty (rank-0 scalar).
> 5. Initialize `id = XNN_INVALID_VALUE_ID`.
> 6. Resolve constant data pointer `buffer_ptr` via
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]`
>    (tensor overload). Propagate its error on failure. `buffer_ptr` is null
>    for non-constant tensors.
> 7. Compute `dq_datatype = getDataType(tensor_value->dq_datatype())` per
>    `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn]`.
>    If `dq_datatype != xnn_datatype_invalid`:
>    - If it is not `xnn_datatype_qint8`: return Error::Internal ("Only int8_t
>      is supported for dq_datatype for now").
>    - Else require the EXTERNAL_INPUT flag be set on the tensor; if not,
>      return Error::Internal ("Dynamic quantization ... only allowed for the
>      external input tensor").
> 8. Branch on quantization:
>
> 8a. `qtensor_value == null` (plain / dq tensor):
>    - If `dq_datatype` is NOT a quantized datatype (per
>      `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn]`;
>      note invalid is not quantized, so this is the normal fp path): define a
>      standard tensor value with `xnn_define_tensor_value` using
>      `getDataType(tensor_value->datatype())`, `dims_data`, `buffer_ptr`,
>      `tensor_value->external_id()`, `tensor_value->flags()`; output the new
>      id into `id`.
>    - Else if `dq_datatype != xnn_datatype_invalid` (legacy dynamic-quant via
>      dq_datatype; only qint8 reaches here): re-assert it is a quantized
>      dtype (else Error::Internal), that `external_id != XNN_INVALID_VALUE_ID`
>      (else Error::Internal), and that `buffer_ptr == null` (else
>      Error::Internal). Then for the `xnn_datatype_qint8` case, synthesize the
>      pattern `fp32_input -> convert -> qdint8_input`:
>      * `xnn_define_dynamically_quantized_tensor_value` with datatype
>        `xnn_datatype_qdint8`, `num_nonbatch_dims = 1` (per-token),
>        `external_id = XNN_INVALID_VALUE_ID`, `flags = 0`, producing `id`.
>      * `xnn_define_tensor_value` for the float external input using
>        `getDataType(tensor_value->datatype())`, `buffer_ptr`, real
>        `external_id`/`flags`, producing `float_id`.
>      * `xnn_define_convert(subgraph, float_id, id, flags=0)`.
>      (Any other dq_datatype in this branch: Error::NotImplemented.)
>    - Else (dq_datatype invalid but claimed quantized — unreachable normally):
>      Error::NotImplemented ("Unhandled fp32 tensor").
>    Note: `status` holds the last `xnn_define_*` call result and is checked in
>    step 9.
>
> 8b. `qtensor_value != null`: dispatch on
>    `qtensor_value->quant_params_type()`:
>    - `PerTensorQuant`: read `qparams` (scale fp32, zero_point). Call
>      `xnn_define_quantized_tensor_value` with `getDataType(datatype())`,
>      `zero_point`, `scale`, `dims_data`, `buffer_ptr`, `external_id`,
>      `flags`; output `id`.
>    - `PerChannelQuant`: compute `dtype = getDataType(datatype())`;
>      `zero_point = 8` if `dtype == xnn_datatype_qcint4` else 0. Default
>      `scale = qparams->scale()->data()` (fp32 array). If
>      `qparams->scale_buffer_idx() != 0`, resolve the scale array via
>      `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]`
>      (index overload), reinterpret as `const float*`, and require non-null
>      (else Error::Internal). Call
>      `xnn_define_channelwise_quantized_tensor_value_v2` with `dtype`,
>      `zero_point`, `scale`, `dims_data.size()`, `qparams->channel_dim()`,
>      `dims_data`, `buffer_ptr`, `external_id`, `flags`; output `id`.
>    - `PerChannelGroupQuant`: require `getDataType(datatype()) ==
>      xnn_datatype_qbint4` (else Error::Internal). Read `group_size =
>      qparams->group_size()`, `output_channels = dims->Get(0)`,
>      `input_channels = dims->Get(1)`. Obtain bf16 block scales:
>      * If `scale_buffer_idx() != 0`: resolve via get-constant-data-ptr,
>        reinterpret as `const uint16_t*` (already bf16), require non-null
>        (else Error::Internal); `scale_numel = qparams->num_scales()`.
>      * Else: read the fp32 `qparams->scale()` array, allocate a temporary
>        `uint16_t` buffer of `scale()->size()` entries via
>        `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn]`,
>        convert fp32->bf16 per
>        `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn]`;
>        `scale_numel = scale()->size()`; `scale_data` points at the temp.
>      Require `scale_numel == output_channels * input_channels / group_size`
>      (else Error::Internal). `zero_point = 8` (qbint4). Call
>      `xnn_define_blockwise_quantized_tensor_value` with datatype qbint4,
>      `zero_point`, `scale_data`, `dims_data.size()`,
>      `qparams->channel_dim()`, `block_size = group_size`, `dims_data`,
>      `buffer_ptr`, `external_id`, `flags`; output `id`.
>    - `PerTokenDynamicQuant`: require `buffer_ptr == null` (else
>      Error::Internal, "Dynamically quantized tensor should not have constant
>      data"). Call `xnn_define_dynamically_quantized_tensor_value` with
>      `getDataType(datatype())`, `dims_data`, `qparams->num_nonbatch_dims()`,
>      `external_id`, `flags`; output `id`.
>    - Any other quant_params_type: Error::NotImplemented ("Unhandled
>      Quantization Parameters").
>
> 9. After the dispatch, check the last-recorded `xnn_status status`: if not
>    `xnn_status_success`, return Error::Internal ("Failed to define tensor").
> 10. Insert `remapped_ids[tensor_value->id_out()] = id` (records serialized
>     id -> new xnnpack id).
> 11. If EXTERNAL_INPUT flag set, push `tensor_value->external_id()` onto
>     `input_ids`. If EXTERNAL_OUTPUT flag set, push it onto `output_ids`.
>     (Both can never be set simultaneously given step 3's mask but each is
>     checked independently.)
> 12. Return Error::Ok.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]
> std::vector<T> flatbufferDimsToVector( const flatbuffers::Vector<uint32_t>* fb_dims)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]
> Converts a flatbuffer vector of `uint32_t` dims into a `std::vector<T>`
> (default `T = size_t`), static-casting each element. Used because XNNPACK
> takes `size_t*` dims/perm/padding arrays while the shapes are serialized as
> uint32.
>
> Steps:
> 1. Create an empty output vector; reserve `fb_dims->size()` entries.
> 2. Iterate the flatbuffer vector in stored order; for each `fb_dim`
>    (uint32), push `static_cast<T>(fb_dim)` onto the output.
> 3. Return the output vector (length equals the input length, preserving
>    order).
>
> Precondition: `fb_dims` must be non-null (the source dereferences it without
> a null check; all callers null-check the field before calling, so a null
> argument is a caller bug). An empty input yields an empty output (rank-0 /
> scalar tensors). The cast from uint32 to size_t is always widening on
> supported targets, so no truncation.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]
> Result<const uint8_t*> getConstantDataPtr( uint32_t buffer_idx, GraphPtr flatbuffer_graph, const uint8_t* constant_data_ptr, uint64_t constant_data_size, const NamedDataMap* named_data_map, std::vector<FreeableBuffer>& freeable_buffers, ...

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]
> Resolves the constant-weight data pointer for a tensor given its serialized
> `buffer_idx`. Returns a `Result<const uint8_t*>`; the success value is
> `nullptr` when the tensor has no associated constant data. There is also a
> thin overload taking `tensor_value` that simply forwards
> `tensor_value->constant_buffer_idx()` (and all other args) to this one.
>
> Steps:
> 1. If `buffer_idx == 0`: the tensor is non-constant. Return success with
>    `nullptr`.
> 2. Otherwise (`buffer_idx != 0`), branch on whether `constant_data_ptr` (the
>    external constant-data region base pointer) is null:
>
> 2a. `constant_data_ptr == nullptr` (deprecated in-flatbuffer path, kept for
>    BC): read `flatbuffer_graph->constant_buffer()`.
>    - If it is null: return Error::InvalidProgram ("constant_buffer is null").
>    - If `buffer_idx >= constant_buffer size`: return Error::InvalidProgram
>      (out-of-bounds).
>    - Fetch entry `(*constant_buffer)[buffer_idx]`; if the entry is null or
>      its `storage()` is null: return Error::InvalidProgram ("Null
>      constant_buffer entry").
>    - Return success with `entry->storage()->data()` (pointer into the
>      flatbuffer blob).
>
> 2b. `constant_data_ptr != nullptr` (current path via offsets): read
>    `flatbuffer_graph->constant_data()`.
>    - If it is null: return Error::InvalidProgram ("constant_data is null").
>    - If `buffer_idx >= constant_data size`: return Error::InvalidProgram
>      (out-of-bounds).
>    - Fetch `ConstantDataOffset` entry at `buffer_idx`; if null: return
>      Error::InvalidProgram.
>    - Read `offset = entry->offset()` (u64) and `entry_size = entry->size()`
>      (u64). Determine whether the optional `named_key` field is present.
>    - If `named_key` is NOT present (data lives inline in the external
>      constant-data region):
>      * Bounds-check: require `offset <= constant_data_size` AND
>        `entry_size <= constant_data_size - offset` (this ordering avoids u64
>        overflow). On failure return Error::InvalidProgram (out of bounds).
>      * Return success with `constant_data_ptr + offset`.
>    - If `named_key` IS present (data lives in a NamedDataMap / weights
>      cache):
>      * If `entry->named_key()` is null: return Error::InvalidProgram ("Named
>        key is null").
>      * Let `data_name = named_key` string.
>      * If `use_weight_cache` is true: call
>        `weights_cache->load_unpacked_data(data_name)`. On error, log
>        "Failed to load weights from cache" and return that error; otherwise
>        return success with the loaded pointer.
>      * If `use_weight_cache` is false: call
>        `named_data_map->get_data(data_name)`. On error, log the key and the
>        numeric error code and return that error. On success, take the
>        `FreeableBuffer`'s `.data()` pointer, move the FreeableBuffer into the
>        caller-provided `freeable_buffers` vector (so it stays alive and is
>        freed later), and return success with that pointer.
>
> Notes: the tensor overload delegates entirely to this by passing
> `constant_buffer_idx()`. This function is also invoked directly by
> `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]`
> to resolve per-channel and per-channel-group scale buffers via their
> `scale_buffer_idx`. All error returns propagate an `Error` enum value; no
> partial state is written on failure.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn]
> xnn_datatype getDataType(const DataType& data_type)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn]
> Maps a serialized flatbuffer datatype enum (`fb_xnnpack::XNNDatatype`) to the
> corresponding runtime `xnn_datatype` enum. Pure lookup, total function.
>
> One-to-one mapping (serialized -> xnnpack):
> - xnn_datatype_fp32 -> xnn_datatype_fp32
> - xnn_datatype_fp16 -> xnn_datatype_fp16
> - xnn_datatype_qint8 -> xnn_datatype_qint8
> - xnn_datatype_quint8 -> xnn_datatype_quint8
> - xnn_datatype_qint32 -> xnn_datatype_qint32
> - xnn_datatype_qcint8 -> xnn_datatype_qcint8
> - xnn_datatype_qcint32 -> xnn_datatype_qcint32
> - xnn_datatype_qcint4 -> xnn_datatype_qcint4
> - xnn_datatype_qdint8 -> xnn_datatype_qdint8
> - xnn_datatype_qbint4 -> xnn_datatype_qbint4
> - xnn_datatype_qpint8 -> xnn_datatype_qpint8
> - xnn_datatype_int32 -> xnn_datatype_int32
> - xnn_datatype_pfp32 -> xnn_datatype_pfp32
> - xnn_datatype_bf16 -> xnn_datatype_bf16
>
> Any serialized value not in this list (including the serialized "invalid"
> sentinel) maps to `xnn_datatype_invalid`. Callers use `xnn_datatype_invalid`
> as a "not a quantized dq target" / "no conversion" sentinel (see
> `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]`).

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]
> DefineNodeFunc getDefineNodeFunc(fb_xnnpack::XNodeUnion nodeType)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]
> Maps a serialized `XNodeUnion` type to the function pointer that defines that
> node into the subgraph (all definers share the `DefineNodeFunc` signature:
> `(subgraph, remapped_ids, node, graph) -> Error`, noexcept). Pure lookup.
>
> Dispatch table (node type -> definer):
> - Binary: Add, Subtract, Multiply, Div, Minimum, Maximum -> their macro-
>   generated binary definers (see
>   `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn]`).
> - Unary: Softmax, SquareRoot, ReciprocalSquareRoot, Ceiling, Gelu,
>   Hardswish, Log, Tanh, Negate, Square, Clamp, LeakyReLU, ELU, Exp, Abs,
>   Floor, PReLU, Sigmoid, Sin, Cos -> their respective definers (softmax,
>   tanh, exp, prelu have dedicated definers; the rest are macro-generated via
>   `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn]`).
> - Others: FullyConnected, StaticTranspose, Conv2d, ConvTranspose2d,
>   StaticResizeBilinear2D, StaticConstantPad, AvgPooling2d, DepthwiseConv2d,
>   MaxPooling2d, Convert, GlobalAvgPooling2d, StaticReshape, ArgMaxPooling2d,
>   Concatenate2, Concatenate3, Concatenate4, Concatenate5, StaticSlice,
>   BatchMatrixMultiply, Copy -> their respective definers.
> - `XNodeUnion::NONE` and any type not listed above (default) ->
>   `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn]`.
>
> Always returns a valid non-null function pointer (the not-implemented
> definer is the catch-all).

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]
> std::pair<float, float> getOutputMinMax(const NodePtr node) noexcept

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]
> Extracts the fused output clamp (activation min/max) for a serialized node,
> defaulting to an unbounded range. `noexcept`.
>
> Steps:
> 1. Initialize `output_min = -infinity` (fp32) and `output_max = +infinity`
>    (fp32).
> 2. Read `node->output_min_max()` (an optional serialized sub-table).
> 3. If it is non-null, overwrite `output_min = output_min_max->output_min()`
>    and `output_max = output_min_max->output_max()` (both fp32).
> 4. Return the pair `{output_min, output_max}`.
>
> The returned pair's `.first` is the min, `.second` is the max. When absent
> the range is (-inf, +inf), i.e. no clamping. Callers pass these straight
> into the corresponding `xnn_define_*` call's output_min/output_max
> arguments.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn]
> bool isQP8(const fb_xnnpack::XNNGraph* graph, const NodePtr node)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn]
> Only compiled when `ENABLE_XNNPACK_KLEIDI` is defined. Decides whether a
> Convert node's qdint8 output feeds a fully-connected (linear) node whose
> filter is a KleidiAI-supported low-bit type, in which case QP8 (packed
> per-token int8) can be used. Returns bool.
>
> Precondition/assert: `node` must be an `XNNConvert` node.
>
> Define a helper `check_dtype(id, dtype)`: scan `graph->xvalues()` in order;
> for each value that is an `XNNQuantizedTensorValue`, take its inner
> `tensor_value`; if `tensor_value->id_out() == id`, return
> `tensor_value->datatype() == dtype`; if no matching quantized value is
> found, return false.
>
> Steps:
> 1. Let `cvt_output_id = node->as_XNNConvert()->output_id()`.
> 2. If `check_dtype(cvt_output_id, xnn_datatype_qdint8)` is false, return
>    false (the convert output is not qdint8).
> 3. Define the supported filter dtype set: { qbint4, qcint4, qcint8 }.
> 4. Scan `graph->xnodes()` in order; for each `XNNFullyConnected` node whose
>    `input1_id() == cvt_output_id`, check whether its `filter_id()` has any
>    dtype in the supported set (via `check_dtype`). If a match is found,
>    return true immediately.
> 5. If no such linear node is found, return false.
>
> The comment notes the assumption that finding one valid consuming linear
> node is sufficient to enable QP8 for all linear nodes consuming this convert
> output.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn]
> bool isQuantizedDataType(const xnn_datatype data_type)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn]
> Returns true iff the given runtime `xnn_datatype` is one of the quantized
> integer types handled by this compiler. Pure predicate.
>
> Returns true for exactly this set:
> - xnn_datatype_qint8
> - xnn_datatype_quint8
> - xnn_datatype_qint32
> - xnn_datatype_qcint8
> - xnn_datatype_qcint32
> - xnn_datatype_qcint4
> - xnn_datatype_qdint8
>
> Returns false for every other value, including `xnn_datatype_invalid`,
> floating types (fp32/fp16/bf16/pfp32), int32, and the quantized types
> qbint4, qpint8 which are deliberately NOT in this set.

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler]
> class XNNCompiler {
>   ET_NODISCARD static executorch::runtime::Error compileModel( const void* buffer_pointer, size_t num_bytes, XNNExecutor* executor, XNNWeightsCache* weights_ca...;
> }

> [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn]
> ET_NODISCARD Error XNNCompiler::compileModel( const void* buffer_pointer, size_t num_bytes, XNNExecutor* executor, XNNWeightsCache* weights_cache, xnn_workspace_t workspace, const NamedDataMap* named_data_map, bool use_weight_cache)

> [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn]
> Static entry point that parses the serialized XNNPACK delegate blob, builds
> an XNNPACK subgraph and runtime, and initializes the given `XNNExecutor`.
> `ET_NODISCARD`; returns `Error::Ok` on success else an `Error`. Any error
> return leaves the executor uninitialized; the locally created subgraph is
> always deleted (RAII) on any exit path.
>
> Steps:
> 1. Create a `CompileAllocator` for temporary bf16 scale conversions.
> 2. Parse the optional XNNHeader from `buffer_pointer`/`num_bytes`
>    (`XNNHeader::Parse`). Header parse can only yield Error::Ok or
>    Error::NotFound.
>    - If Ok: `flatbuffer_data = buffer_pointer + header.flatbuffer_offset`,
>      `flatbuffer_size = header.flatbuffer_size`, `constant_data =
>      buffer_pointer + header.constant_data_offset`, `constant_data_size =
>      header.constant_data_size`.
>    - If NotFound (no header, legacy layout): `flatbuffer_data =
>      buffer_pointer`, `flatbuffer_size = num_bytes`, `constant_data = null`,
>      `constant_data_size = 0`.
>    - Any other error: log "XNNHeader may be corrupt" and return that error.
> 3. Version check: read the 4-byte flatbuffer identifier; require it equals
>    "XN00" or "XN01"; else return Error::DelegateInvalidCompatibility.
> 4. Verify flatbuffer integrity: run a flatbuffers `Verifier` over
>    `flatbuffer_data`/`flatbuffer_size` for `fb_xnnpack::XNNGraph`. On failure
>    return Error::DelegateInvalidCompatibility ("FlatBuffer verification
>    failed").
> 5. Get `flatbuffer_graph = GetXNNGraph(flatbuffer_data)`; require it and its
>    `xvalues()` and `xnodes()` all non-null; else Error::InvalidProgram.
> 6. Call `xnn_initialize(null)`; if not success, Error::Internal.
> 7. Read `num_externs = flatbuffer_graph->num_externs()`; require
>    `num_externs <= 4096`; else Error::InvalidProgram (guards against corrupt
>    graphs).
> 8. Call `xnn_create_subgraph(num_externs, flags=0, &subgraph_ptr)`; if not
>    success, Error::Internal. Wrap in an owning handle whose deleter is
>    `xnn_delete_subgraph`.
> 9. Create `remapped_ids` map and pre-seed `XNN_INVALID_VALUE_ID ->
>    XNN_INVALID_VALUE_ID` (invalid ids remap to themselves). Create empty
>    `unpacked_buffers` (vector<FreeableBuffer>), `input_ids`, `output_ids`.
> 10. For each value in `flatbuffer_graph->xvalues()` (in order), call
>     `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]`
>     passing the subgraph, remapped_ids, value, graph, constant_data,
>     constant_data_size, input_ids, output_ids, the compile allocator,
>     named_data_map, unpacked_buffers, weights_cache, use_weight_cache. On any
>     error, return it immediately.
> 11. For each node in `flatbuffer_graph->xnodes()` (in order), look up the
>     definer via
>     `[spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]`
>     and call it with (subgraph, remapped_ids, node, flatbuffer_graph). On any
>     error, return it immediately.
> 12. Compute `runtime_flags = 0`, OR-in `XNN_FLAG_BASIC_PROFILING` if
>     `ENABLE_XNNPACK_PROFILING` or `ET_EVENT_TRACER_ENABLED` is defined.
> 13. Determine the weights-cache pointer: if `use_weight_cache`, require
>     `unpacked_buffers` is empty (else Error::Internal — the cache should own
>     unpacked buffers); then `weights_cache_ptr = weights_cache->get()` if
>     `weights_cache->get_num_unpacked_data() > 0` else null. If not using the
>     cache, `weights_cache_ptr = null`.
> 14. Call `xnn_create_runtime_v4(subgraph, weights_cache_ptr, workspace,
>     get_pthreadpool(), runtime_flags, &runtime_ptr)`; if not success,
>     Error::Internal.
> 15. Finalize weights: if `use_weight_cache`, call
>     `weights_cache->finalize_for_runtime()`; on failure Error::Internal;
>     otherwise take the returned packed-weights-names vector. If not using the
>     cache, free every buffer in `unpacked_buffers` (each `buffer.Free()`),
>     and packed_weights_names is empty.
> 16. Call `executor->initialize(runtime_ptr, move(input_ids),
>     move(output_ids), move(packed_weights_names))` and return its Error
>     (the executor takes ownership of the runtime).

