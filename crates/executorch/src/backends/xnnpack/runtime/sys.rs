//! Hand-written `extern "C"` declarations for the subset of the XNNPACK C API
//! (`<xnnpack.h>`) that the XNNPACK delegate runtime uses.
//!
//! PORT-NOTE: The XNNPACK third-party submodule
//! (`backends/xnnpack/third-party/XNNPACK`) is not checked out in this tree, so
//! these declarations are transcribed by hand from the public XNNPACK C ABI
//! rather than generated from the header. Opaque handle types are modeled as
//! `#[repr(C)]` newtypes over raw pointers, mirroring how XNNPACK typedefs its
//! `*_t` handles. `xnn_status` is a `#[repr(C)]` newtype over `u32` carrying the
//! documented enum discriminants. Everything here is behind the `xnnpack`
//! feature; the delegate's non-feature fallback mirrors "XNNPACK absent"
//! behavior in the callers.
#![cfg(feature = "xnnpack")]
#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_void};

/// `enum xnn_status` — XNNPACK operation status codes.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_status(pub u32);

impl xnn_status {
    pub const SUCCESS: xnn_status = xnn_status(0);
    pub const UNINITIALIZED: xnn_status = xnn_status(1);
    pub const INVALID_PARAMETER: xnn_status = xnn_status(2);
    pub const INVALID_STATE: xnn_status = xnn_status(3);
    pub const UNSUPPORTED_PARAMETER: xnn_status = xnn_status(4);
    pub const UNSUPPORTED_HARDWARE: xnn_status = xnn_status(5);
    pub const OUT_OF_MEMORY: xnn_status = xnn_status(6);
    pub const REINITIALIZATION_REQUIRED: xnn_status = xnn_status(7);
    pub const DEPRECATED: xnn_status = xnn_status(8);
}

pub const xnn_status_success: xnn_status = xnn_status::SUCCESS;
pub const xnn_status_uninitialized: xnn_status = xnn_status::UNINITIALIZED;
pub const xnn_status_invalid_parameter: xnn_status = xnn_status::INVALID_PARAMETER;
pub const xnn_status_invalid_state: xnn_status = xnn_status::INVALID_STATE;
pub const xnn_status_unsupported_parameter: xnn_status = xnn_status::UNSUPPORTED_PARAMETER;
pub const xnn_status_unsupported_hardware: xnn_status = xnn_status::UNSUPPORTED_HARDWARE;
pub const xnn_status_out_of_memory: xnn_status = xnn_status::OUT_OF_MEMORY;

/// `enum xnn_datatype` — XNNPACK runtime tensor datatypes. Discriminants match
/// the XNNPACK C ABI (`include/xnnpack.h`). `XNNCompiler` maps the serialized
/// flatbuffer datatype enum onto these.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_datatype(pub u32);

impl xnn_datatype {
    pub const INVALID: xnn_datatype = xnn_datatype(0);
    pub const FP32: xnn_datatype = xnn_datatype(1);
    pub const FP16: xnn_datatype = xnn_datatype(2);
    pub const QINT8: xnn_datatype = xnn_datatype(3);
    pub const QUINT8: xnn_datatype = xnn_datatype(4);
    pub const QINT32: xnn_datatype = xnn_datatype(5);
    pub const QCINT8: xnn_datatype = xnn_datatype(6);
    pub const QCINT32: xnn_datatype = xnn_datatype(7);
    pub const QCINT4: xnn_datatype = xnn_datatype(8);
    pub const QDINT8: xnn_datatype = xnn_datatype(9);
    pub const QBINT4: xnn_datatype = xnn_datatype(10);
    pub const QPINT8: xnn_datatype = xnn_datatype(11);
    pub const INT32: xnn_datatype = xnn_datatype(12);
    pub const PFP32: xnn_datatype = xnn_datatype(13);
    pub const BF16: xnn_datatype = xnn_datatype(14);
}

pub const xnn_datatype_invalid: xnn_datatype = xnn_datatype::INVALID;
pub const xnn_datatype_fp32: xnn_datatype = xnn_datatype::FP32;
pub const xnn_datatype_fp16: xnn_datatype = xnn_datatype::FP16;
pub const xnn_datatype_qint8: xnn_datatype = xnn_datatype::QINT8;
pub const xnn_datatype_quint8: xnn_datatype = xnn_datatype::QUINT8;
pub const xnn_datatype_qint32: xnn_datatype = xnn_datatype::QINT32;
pub const xnn_datatype_qcint8: xnn_datatype = xnn_datatype::QCINT8;
pub const xnn_datatype_qcint32: xnn_datatype = xnn_datatype::QCINT32;
pub const xnn_datatype_qcint4: xnn_datatype = xnn_datatype::QCINT4;
pub const xnn_datatype_qdint8: xnn_datatype = xnn_datatype::QDINT8;
pub const xnn_datatype_qbint4: xnn_datatype = xnn_datatype::QBINT4;
pub const xnn_datatype_qpint8: xnn_datatype = xnn_datatype::QPINT8;
pub const xnn_datatype_int32: xnn_datatype = xnn_datatype::INT32;
pub const xnn_datatype_pfp32: xnn_datatype = xnn_datatype::PFP32;
pub const xnn_datatype_bf16: xnn_datatype = xnn_datatype::BF16;

/// `enum xnn_unary_operator` — elementwise unary op selector for
/// `xnn_define_unary`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_unary_operator(pub u32);

// Only the discriminants XNNCompiler references are named. Values MUST match the
// actual `enum xnn_unary_operator` in the linked `include/xnnpack.h`
// (abs=0, approxgelu=1, bankers_rounding=2, ceiling=3, clamp=4, convert=5,
// cosine=6, elu=7, exp=8, floor=9, gelu=10, hardswish=11, leaky_relu=12, log=13,
// negate=14, reciprocal_square_root=15, sigmoid=16, sine=17, square_root=18,
// square=19, tanh=20, ...). An earlier layout had these shifted, which made
// `gelu` collide with `leaky_relu` (which requires params) -> xnn_define_unary
// returned invalid_parameter, and silently mis-mapped other unary ops.
pub const xnn_unary_abs: xnn_unary_operator = xnn_unary_operator(0);
pub const xnn_unary_ceiling: xnn_unary_operator = xnn_unary_operator(3);
pub const xnn_unary_clamp: xnn_unary_operator = xnn_unary_operator(4);
pub const xnn_unary_elu: xnn_unary_operator = xnn_unary_operator(7);
pub const xnn_unary_floor: xnn_unary_operator = xnn_unary_operator(9);
pub const xnn_unary_gelu: xnn_unary_operator = xnn_unary_operator(10);
pub const xnn_unary_hardswish: xnn_unary_operator = xnn_unary_operator(11);
pub const xnn_unary_leaky_relu: xnn_unary_operator = xnn_unary_operator(12);
pub const xnn_unary_log: xnn_unary_operator = xnn_unary_operator(13);
pub const xnn_unary_negate: xnn_unary_operator = xnn_unary_operator(14);
pub const xnn_unary_sigmoid: xnn_unary_operator = xnn_unary_operator(16);
pub const xnn_unary_square: xnn_unary_operator = xnn_unary_operator(19);
pub const xnn_unary_square_root: xnn_unary_operator = xnn_unary_operator(18);
pub const xnn_unary_reciprocal_square_root: xnn_unary_operator = xnn_unary_operator(15);
pub const xnn_unary_sine: xnn_unary_operator = xnn_unary_operator(17);
pub const xnn_unary_cosine: xnn_unary_operator = xnn_unary_operator(6);

/// `enum xnn_binary_operator` — elementwise binary op selector for
/// `xnn_define_binary`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_binary_operator(pub u32);

// MUST match `enum xnn_binary_operator` in the linked header, which starts at 0
// (`invalid = -1, add = 0, subtract = 1, multiply = 2, divide = 3, maximum = 4,
// minimum = 5, ...`). An earlier off-by-one made `multiply` map to XNNPACK
// `divide` etc., silently corrupting delegated graphs (division-by-zero -> NaN).
pub const xnn_binary_add: xnn_binary_operator = xnn_binary_operator(0);
pub const xnn_binary_subtract: xnn_binary_operator = xnn_binary_operator(1);
pub const xnn_binary_multiply: xnn_binary_operator = xnn_binary_operator(2);
pub const xnn_binary_divide: xnn_binary_operator = xnn_binary_operator(3);
pub const xnn_binary_maximum: xnn_binary_operator = xnn_binary_operator(4);
pub const xnn_binary_minimum: xnn_binary_operator = xnn_binary_operator(5);

/// `struct xnn_binary_params` — output clamp bounds for binary ops. NOTE: the C
/// struct uses `double`, unlike the `float`-valued `xnn_unary_params.clamp`, so
/// these MUST be `f64`. An `f32` layout is 8 bytes where XNNPACK reads 16,
/// making it read garbage clamp bounds (past the struct) and corrupt every
/// add/sub/mul/div output — e.g. zeroing FastPitch durations.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_binary_params {
    pub output_min: f64,
    pub output_max: f64,
}

/// `struct xnn_unary_clamp_params` — clamp op bounds.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_clamp_params {
    pub min: f32,
    pub max: f32,
}

/// `struct xnn_unary_elu_params` — ELU alpha.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_elu_params {
    pub alpha: f32,
}

/// `struct xnn_unary_leaky_relu_params` — leaky-ReLU negative slope.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_unary_leaky_relu_params {
    pub negative_slope: f32,
}

/// `union xnn_unary_params` — per-op parameters for `xnn_define_unary`. Modeled
/// as a `#[repr(C)]` union mirroring XNNPACK's C union; the compiler writes one
/// field (clamp / elu / leaky_relu) per op.
#[repr(C)]
#[derive(Clone, Copy)]
pub union xnn_unary_params {
    pub clamp: xnn_unary_clamp_params,
    pub elu: xnn_unary_elu_params,
    pub leaky_relu: xnn_unary_leaky_relu_params,
    // Padding: the real C union has additional per-op members; the compiler only
    // ever writes the three above, so the union is at least this wide.
    _bytes: [u8; 8],
}

/// `#define XNN_INVALID_VALUE_ID UINT32_MAX` — sentinel tensor id.
pub const XNN_INVALID_VALUE_ID: u32 = u32::MAX;

/// `#define XNN_VALUE_FLAG_EXTERNAL_INPUT 0x00000001`.
pub const XNN_VALUE_FLAG_EXTERNAL_INPUT: u32 = 0x00000001;
/// `#define XNN_VALUE_FLAG_EXTERNAL_OUTPUT 0x00000002`.
pub const XNN_VALUE_FLAG_EXTERNAL_OUTPUT: u32 = 0x00000002;
/// `#define XNN_FLAG_BASIC_PROFILING 0x00000008` (runtime creation flag).
pub const XNN_FLAG_BASIC_PROFILING: u32 = 0x00000008;

/// `#define XNN_MAX_TENSOR_DIMS 6` — maximum tensor rank XNNPACK accepts.
pub const XNN_MAX_TENSOR_DIMS: usize = 6;

/// `enum xnn_profile_info` — selector for `xnn_get_runtime_profiling_info`.
/// Modeled as a `#[repr(C)]` newtype over the C enum's underlying `u32`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_profile_info(pub u32);

/// `xnn_profile_info_num_operators` — number of operators in the runtime.
pub const xnn_profile_info_num_operators: xnn_profile_info = xnn_profile_info(0);
/// `xnn_profile_info_operator_name` — concatenated NUL-terminated op names.
pub const xnn_profile_info_operator_name: xnn_profile_info = xnn_profile_info(1);
/// `xnn_profile_info_operator_timing` — per-operator elapsed microseconds.
pub const xnn_profile_info_operator_timing: xnn_profile_info = xnn_profile_info(2);

// Opaque handle types. XNNPACK typedefs these as pointers to incomplete
// structs; model each as a `#[repr(C)]` newtype over a raw pointer.

/// `xnn_workspace_t` — opaque handle to an XNNPACK workspace (memory arena).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_workspace_t(pub *mut c_void);

/// `xnn_subgraph_t` — opaque handle to an XNNPACK subgraph.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_subgraph_t(pub *mut c_void);

/// `xnn_runtime_t` — opaque handle to an instantiated XNNPACK runtime.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_runtime_t(pub *mut c_void);

/// `xnn_weights_cache_t` — opaque handle to a weights cache, as passed to
/// XNNPACK's runtime-creation APIs.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct xnn_weights_cache_t(pub *mut c_void);

/// `pthreadpool_t` — opaque handle to the thread pool used during runtime
/// creation / invocation.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct pthreadpool_t(pub *mut c_void);

/// `struct xnn_external_value` — external tensor binding for
/// `xnn_setup_runtime`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_external_value {
    pub id: u32,
    pub data: *mut c_void,
}

/// `struct xnn_weights_cache_look_up_key` — cache key passed to the weights
/// cache provider callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct xnn_weights_cache_look_up_key {
    pub seed: u32,
    pub kernel: *const c_void,
    pub bias: *const c_void,
}

/// `struct xnn_weights_cache_provider` — vtable of C function pointers XNNPACK
/// invokes to drive a custom weights cache. The `context` field is passed back
/// as the first argument of every callback.
#[repr(C)]
pub struct xnn_weights_cache_provider {
    pub context: *mut c_void,
    pub look_up: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            cache_key: *const xnn_weights_cache_look_up_key,
        ) -> usize,
    >,
    pub reserve_space: Option<unsafe extern "C" fn(context: *mut c_void, n: usize) -> *mut c_void>,
    pub look_up_or_insert: Option<
        unsafe extern "C" fn(
            context: *mut c_void,
            cache_key: *const xnn_weights_cache_look_up_key,
            ptr: *mut c_void,
            size: usize,
        ) -> usize,
    >,
    pub is_finalized: Option<unsafe extern "C" fn(context: *mut c_void) -> bool>,
    pub offset_to_addr:
        Option<unsafe extern "C" fn(context: *mut c_void, offset: usize) -> *mut c_void>,
    pub delete_cache: Option<unsafe extern "C" fn(context: *mut c_void) -> xnn_status>,
}

unsafe extern "C" {
    /// `enum xnn_status xnn_initialize(const struct xnn_allocator* allocator)`.
    /// The delegate always passes a null allocator.
    pub fn xnn_initialize(allocator: *const c_void) -> xnn_status;

    /// `enum xnn_status xnn_create_workspace(xnn_workspace_t* workspace_out)`.
    pub fn xnn_create_workspace(workspace_out: *mut xnn_workspace_t) -> xnn_status;

    /// `enum xnn_status xnn_release_workspace(xnn_workspace_t workspace)`.
    pub fn xnn_release_workspace(workspace: xnn_workspace_t) -> xnn_status;

    /// `enum xnn_status xnn_create_runtime_v4(xnn_subgraph_t subgraph,
    /// xnn_weights_cache_t weights_cache, xnn_workspace_t workspace,
    /// pthreadpool_t threadpool, uint32_t flags, xnn_runtime_t* runtime_out)`.
    pub fn xnn_create_runtime_v4(
        subgraph: xnn_subgraph_t,
        weights_cache: xnn_weights_cache_t,
        workspace: xnn_workspace_t,
        threadpool: pthreadpool_t,
        flags: u32,
        runtime_out: *mut xnn_runtime_t,
    ) -> xnn_status;

    /// `enum xnn_status xnn_reshape_runtime(xnn_runtime_t runtime)`.
    pub fn xnn_reshape_runtime(runtime: xnn_runtime_t) -> xnn_status;

    /// `enum xnn_status xnn_setup_runtime(xnn_runtime_t runtime,
    /// size_t num_external_values, const struct xnn_external_value*
    /// external_values)`.
    pub fn xnn_setup_runtime(
        runtime: xnn_runtime_t,
        num_external_values: usize,
        external_values: *const xnn_external_value,
    ) -> xnn_status;

    /// `enum xnn_status xnn_setup_runtime_v2(xnn_runtime_t runtime,
    /// size_t num_external_values, const struct xnn_external_value*
    /// external_values)`. Like `xnn_setup_runtime` but does not require all
    /// external values to be specified up front. Used by `XNNExecutor::forward`.
    pub fn xnn_setup_runtime_v2(
        runtime: xnn_runtime_t,
        num_external_values: usize,
        external_values: *const xnn_external_value,
    ) -> xnn_status;

    /// `enum xnn_status xnn_reshape_external_value(xnn_runtime_t runtime,
    /// uint32_t external_id, size_t num_dims, const size_t* dims)`. Reshapes a
    /// runtime external input to a new set of extents.
    pub fn xnn_reshape_external_value(
        runtime: xnn_runtime_t,
        external_id: u32,
        num_dims: usize,
        dims: *const usize,
    ) -> xnn_status;

    /// `enum xnn_status xnn_get_external_value_shape(xnn_runtime_t runtime,
    /// uint32_t external_id, size_t* num_dims_out, size_t* dims_out)`. Retrieves
    /// the computed shape of a runtime external value after
    /// `xnn_reshape_runtime`.
    pub fn xnn_get_external_value_shape(
        runtime: xnn_runtime_t,
        external_id: u32,
        num_dims_out: *mut usize,
        dims_out: *mut usize,
    ) -> xnn_status;

    /// `enum xnn_status xnn_get_runtime_profiling_info(xnn_runtime_t runtime,
    /// enum xnn_profile_info param_name, size_t param_value_size,
    /// void* param_value, size_t* param_value_size_ret)`. Used by the profiler
    /// to fetch operator counts / names / timings.
    pub fn xnn_get_runtime_profiling_info(
        runtime: xnn_runtime_t,
        param_name: xnn_profile_info,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> xnn_status;

    /// `enum xnn_status xnn_invoke_runtime(xnn_runtime_t runtime)`.
    pub fn xnn_invoke_runtime(runtime: xnn_runtime_t) -> xnn_status;

    /// `enum xnn_status xnn_delete_runtime(xnn_runtime_t runtime)`.
    pub fn xnn_delete_runtime(runtime: xnn_runtime_t) -> xnn_status;

    /// `enum xnn_status xnn_delete_subgraph(xnn_subgraph_t subgraph)`.
    pub fn xnn_delete_subgraph(subgraph: xnn_subgraph_t) -> xnn_status;

    /// `enum xnn_status xnn_create_weights_cache_with_size(size_t size,
    /// xnn_weights_cache_t* weights_cache_out)`.
    pub fn xnn_create_weights_cache_with_size(
        size: usize,
        weights_cache_out: *mut xnn_weights_cache_t,
    ) -> xnn_status;

    /// `enum xnn_status xnn_delete_weights_cache(xnn_weights_cache_t
    /// weights_cache)`.
    pub fn xnn_delete_weights_cache(weights_cache: xnn_weights_cache_t) -> xnn_status;

    /// `const char* xnn_datatype_to_string(enum xnn_datatype type)` — present
    /// for symmetry with other diagnostics callers use.
    pub fn xnn_datatype_to_string(type_: xnn_datatype) -> *const c_char;

    /// `enum xnn_status xnn_create_subgraph(uint32_t external_value_ids,
    /// uint32_t flags, xnn_subgraph_t* subgraph_out)`.
    pub fn xnn_create_subgraph(
        external_value_ids: u32,
        flags: u32,
        subgraph_out: *mut xnn_subgraph_t,
    ) -> xnn_status;

    // --- Tensor value definition -------------------------------------------

    pub fn xnn_define_tensor_value(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        num_dims: usize,
        dims: *const usize,
        data: *const c_void,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    pub fn xnn_define_quantized_tensor_value(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        zero_point: i32,
        scale: f32,
        num_dims: usize,
        dims: *const usize,
        data: *const c_void,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    pub fn xnn_define_channelwise_quantized_tensor_value_v2(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        zero_point: i32,
        scale: *const f32,
        num_dims: usize,
        channel_dim: usize,
        dims: *const usize,
        data: *const c_void,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    pub fn xnn_define_blockwise_quantized_tensor_value(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        zero_point: i32,
        scale: *const u16,
        num_dims: usize,
        channel_dim: usize,
        block_size: usize,
        dims: *const usize,
        data: *const c_void,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    pub fn xnn_define_dynamically_quantized_tensor_value(
        subgraph: xnn_subgraph_t,
        datatype: xnn_datatype,
        num_dims: usize,
        num_nonbatch_dims: usize,
        dims: *const usize,
        external_id: u32,
        flags: u32,
        id_out: *mut u32,
    ) -> xnn_status;

    // --- Node definition ----------------------------------------------------

    pub fn xnn_define_unary(
        subgraph: xnn_subgraph_t,
        op_type: xnn_unary_operator,
        params: *const xnn_unary_params,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_binary(
        subgraph: xnn_subgraph_t,
        op_type: xnn_binary_operator,
        params: *const xnn_binary_params,
        input1_id: u32,
        input2_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_convert(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_fully_connected(
        subgraph: xnn_subgraph_t,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_softmax(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_global_average_pooling_2d(
        subgraph: xnn_subgraph_t,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_average_pooling_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        pooling_height: u32,
        pooling_width: u32,
        stride_height: u32,
        stride_width: u32,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_convolution_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        kernel_height: u32,
        kernel_width: u32,
        subsampling_height: u32,
        subsampling_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        groups: u32,
        group_input_channels: usize,
        group_output_channels: usize,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_deconvolution_2d(
        subgraph: xnn_subgraph_t,
        padding_top: u32,
        padding_right: u32,
        padding_bottom: u32,
        padding_left: u32,
        adjustment_height: u32,
        adjustment_width: u32,
        kernel_height: u32,
        kernel_width: u32,
        upsampling_height: u32,
        upsampling_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        groups: u32,
        group_input_channels: usize,
        group_output_channels: usize,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_depthwise_convolution_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        kernel_height: u32,
        kernel_width: u32,
        subsampling_height: u32,
        subsampling_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        depth_multiplier: u32,
        input_channels: usize,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        filter_id: u32,
        bias_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_max_pooling_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        pooling_height: u32,
        pooling_width: u32,
        stride_height: u32,
        stride_width: u32,
        dilation_height: u32,
        dilation_width: u32,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_argmax_pooling_2d(
        subgraph: xnn_subgraph_t,
        input_padding_top: u32,
        input_padding_right: u32,
        input_padding_bottom: u32,
        input_padding_left: u32,
        pooling_height: u32,
        pooling_width: u32,
        input_id: u32,
        output_value_id: u32,
        output_index_id: u32,
        flags: u32,
    ) -> xnn_status;

    /// `enum xnn_status xnn_define_clamp(xnn_subgraph_t subgraph,
    /// float output_min, float output_max, uint32_t input_id,
    /// uint32_t output_id, uint32_t flags)`. Used by the XNNExecutor tests to
    /// build a trivial single-node subgraph.
    pub fn xnn_define_clamp(
        subgraph: xnn_subgraph_t,
        output_min: f32,
        output_max: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    /// `enum xnn_status xnn_create_runtime(xnn_subgraph_t subgraph,
    /// xnn_runtime_t* runtime_out)`. The single-argument convenience form used
    /// by the XNNExecutor tests (the delegate itself uses
    /// `xnn_create_runtime_v4`).
    pub fn xnn_create_runtime(
        subgraph: xnn_subgraph_t,
        runtime_out: *mut xnn_runtime_t,
    ) -> xnn_status;

    pub fn xnn_define_static_transpose(
        subgraph: xnn_subgraph_t,
        num_dims: usize,
        perm: *const usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_resize_bilinear_2d(
        subgraph: xnn_subgraph_t,
        new_height: usize,
        new_width: usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_constant_pad(
        subgraph: xnn_subgraph_t,
        pre_paddings: *const usize,
        post_paddings: *const usize,
        padding_value: f32,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_reshape(
        subgraph: xnn_subgraph_t,
        num_dims: usize,
        new_shape: *const usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_static_slice(
        subgraph: xnn_subgraph_t,
        num_dims: usize,
        offsets: *const usize,
        sizes: *const usize,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_batch_matrix_multiply(
        subgraph: xnn_subgraph_t,
        input1_id: u32,
        input2_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_copy(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_prelu(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        slope_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_exp(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_tanh(
        subgraph: xnn_subgraph_t,
        input_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_concatenate2(
        subgraph: xnn_subgraph_t,
        axis: usize,
        input1_id: u32,
        input2_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_concatenate3(
        subgraph: xnn_subgraph_t,
        axis: usize,
        input1_id: u32,
        input2_id: u32,
        input3_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_concatenate4(
        subgraph: xnn_subgraph_t,
        axis: usize,
        input1_id: u32,
        input2_id: u32,
        input3_id: u32,
        input4_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;

    pub fn xnn_define_concatenate5(
        subgraph: xnn_subgraph_t,
        axis: usize,
        input1_id: u32,
        input2_id: u32,
        input3_id: u32,
        input4_id: u32,
        input5_id: u32,
        output_id: u32,
        flags: u32,
    ) -> xnn_status;
}

// The C++ delegate calls `::executorch::extension::threadpool::get_pthreadpool()`
// — a process-wide shared `pthreadpool` sized to the core count — and passes it
// to `xnn_create_runtime_v4`. The extension/threadpool module is out of the port
// scope, but `libpthreadpool` is linked as an XNNPACK dependency, so the Rust
// port provides an equivalent process-wide singleton directly over the
// pthreadpool C API. A null pool is also valid (XNNPACK then runs serially);
// creating a real pool matches the multi-threaded C++ default.
unsafe extern "C" {
    /// `pthreadpool_t pthreadpool_create(size_t threads_count)` — `0` sizes the
    /// pool to the number of logical processors.
    fn pthreadpool_create(threads_count: usize) -> pthreadpool_t;
}

/// Returns the process-wide shared thread pool, creating it on first use.
/// Mirrors the C++ `extension::threadpool::get_pthreadpool()` singleton.
pub fn get_pthreadpool() -> pthreadpool_t {
    use core::sync::atomic::{AtomicPtr, Ordering};
    // Process-lifetime singleton; never destroyed (matches the C++ leaked
    // static). Stored as the raw inner pointer so it fits an AtomicPtr.
    static POOL: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
    let existing = POOL.load(Ordering::Acquire);
    if !existing.is_null() {
        return pthreadpool_t(existing);
    }
    // Read once, before the first XNNPACK runtime is created. Zero retains the
    // upstream default of one worker per logical processor.
    let threads_count = std::env::var("EXECUTORCH_XNNPACK_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    crate::et_log!(
        Info,
        "Creating the process-wide XNNPACK thread pool with {} threads (0 = auto)",
        threads_count
    );

    // SAFETY: pthreadpool_create is the linked libpthreadpool entry point.
    let created = unsafe { pthreadpool_create(threads_count) };
    match POOL.compare_exchange(
        core::ptr::null_mut(),
        created.0,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => created,
        // Lost the race: another thread installed its pool first. Keep theirs;
        // ours leaks (harmless for a process-lifetime singleton).
        Err(winner) => pthreadpool_t(winner),
    }
}
