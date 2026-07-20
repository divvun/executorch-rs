//! Literal port of backends/xnnpack/runtime/XNNCompiler.cpp +
//! backends/xnnpack/runtime/XNNCompiler.h.
#![cfg(feature = "xnnpack")]
#![allow(non_snake_case, clippy::too_many_arguments)]

extern crate alloc;

use alloc::vec::Vec;

use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::Result;

use crate::backends::xnnpack::runtime::XNNHeader::XNNHeader;
use crate::backends::xnnpack::runtime::XNNWeightsCache::XNNWeightsCache;
use crate::backends::xnnpack::runtime::generated::fb_xnnpack;
use crate::backends::xnnpack::runtime::sys::{self, xnn_datatype, xnn_subgraph_t};

extern crate std;
use std::string::String;

#[cfg(any(
    feature = "xnnpack-profiling",
    feature = "profiling-enabled",
    feature = "event-tracer"
))]
fn profile_runtime_enabled() -> bool {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_RUNTIME_INDEX: AtomicUsize = AtomicUsize::new(0);
    let index = NEXT_RUNTIME_INDEX.fetch_add(1, Ordering::Relaxed);
    let start = std::env::var("EXECUTORCH_XNNPACK_PROFILE_START")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let count = std::env::var("EXECUTORCH_XNNPACK_PROFILE_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let enabled = index >= start && index.saturating_sub(start) < count;

    if std::env::var_os("EXECUTORCH_XNNPACK_PROFILE_START").is_some()
        || std::env::var_os("EXECUTORCH_XNNPACK_PROFILE_COUNT").is_some()
    {
        eprintln!("XNN_PROFILE_RUNTIME index={index} enabled={enabled}");
    }
    enabled
}

// PORT-NOTE: `ENABLE_XNNPACK_KLEIDI` gates `isQP8` and the extra convert-node
// flag. Mirror it as a cargo feature `xnnpack_kleidi`; nothing in the tree
// enables it yet, so the gated code compiles only when that feature is on.

// Flatbuffer types
type ValuePtr<'a> = fb_xnnpack::XValue<'a>;
type NodePtr<'a> = fb_xnnpack::XNode<'a>;
type GraphPtr<'a> = fb_xnnpack::XNNGraph<'a>;
type DataType = fb_xnnpack::XNNDatatype;

// Type for define node function. This is the function signature
// for any function that takes in a flatbuffer node and defines it
// into our xnn_subgraph
// PORT-NOTE: the C++ `std::unordered_map<uint32_t, uint32_t>` is mirrored as
// `std::collections::HashMap<u32, u32>`.
type DefineNodeFunc =
    fn(xnn_subgraph_t, &std::collections::HashMap<u32, u32>, NodePtr, GraphPtr) -> Error;

/*
 * Provide compile-time allocation.
 */
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator]
pub struct CompileAllocator {
    // PORT-NOTE: C++ holds `std::vector<std::unique_ptr<uint8_t[]>>`. Modeled as
    // a `Vec<Vec<u8>>`: each temporary owns its buffer and is freed when the
    // allocator is dropped, matching the C++ ownership semantics exactly.
    temporaries_: Vec<Vec<u8>>,
}

impl CompileAllocator {
    pub fn new() -> Self {
        CompileAllocator {
            temporaries_: Vec::new(),
        }
    }

    /*
     * Allocate memory which will be automatically freed at the end
     * of the compilation process.
     */
    // [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn]
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn]
    pub fn allocateTemporary(&mut self, size: usize) -> *mut u8 {
        // PORT-NOTE: C++ `new uint8_t[size]` leaves the buffer uninitialized;
        // Rust zero-initializes here. The only caller immediately overwrites
        // every byte via convertF32TensorToBF16, so the observable behavior is
        // identical.
        let mut mem: Vec<u8> = alloc::vec![0u8; size];
        let ptr = mem.as_mut_ptr();
        self.temporaries_.push(mem);
        ptr
    }
}

impl Default for CompileAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/*
Convert a tensor from fp32 to bf16.
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn]
pub unsafe fn convertF32TensorToBF16(f32_data: *const f32, bf16_data_out: *mut u16, numel: usize) {
    let mut i: usize = 0;
    while i < numel {
        // Adjust the f32 value such that it rounds properly after truncation.
        // Constant factor scales 1+2^-8 to 1+2e-7.
        let f32_adjusted: f32 = unsafe { *f32_data.add(i) } * 1.00389105f32;
        let f32_bits: u32 = f32_adjusted.to_bits();
        unsafe { *bf16_data_out.add(i) = (f32_bits >> 16) as u16 };
        i += 1;
    }
}

/*
Gets the output min and output max for a given node operator
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn]
pub fn getOutputMinMax(node: NodePtr) -> (f32, f32) {
    let mut output_min = f32::NEG_INFINITY;
    let mut output_max = f32::INFINITY;
    let output_min_max = node.output_min_max();
    if let Some(output_min_max) = output_min_max {
        output_min = output_min_max.output_min();
        output_max = output_min_max.output_max();
    }

    (output_min, output_max)
}

/*
Converts flatbuffer xnn data type to xnnpack data type
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn]
pub fn getDataType(data_type: DataType) -> xnn_datatype {
    match data_type {
        DataType::xnn_datatype_fp32 => sys::xnn_datatype_fp32,
        DataType::xnn_datatype_fp16 => sys::xnn_datatype_fp16,
        DataType::xnn_datatype_qint8 => sys::xnn_datatype_qint8,
        DataType::xnn_datatype_quint8 => sys::xnn_datatype_quint8,
        DataType::xnn_datatype_qint32 => sys::xnn_datatype_qint32,
        DataType::xnn_datatype_qcint8 => sys::xnn_datatype_qcint8,
        DataType::xnn_datatype_qcint32 => sys::xnn_datatype_qcint32,
        DataType::xnn_datatype_qcint4 => sys::xnn_datatype_qcint4,
        DataType::xnn_datatype_qdint8 => sys::xnn_datatype_qdint8,
        DataType::xnn_datatype_qbint4 => sys::xnn_datatype_qbint4,
        DataType::xnn_datatype_qpint8 => sys::xnn_datatype_qpint8,
        DataType::xnn_datatype_int32 => sys::xnn_datatype_int32,
        DataType::xnn_datatype_pfp32 => sys::xnn_datatype_pfp32,
        DataType::xnn_datatype_bf16 => sys::xnn_datatype_bf16,
        _ => sys::xnn_datatype_invalid,
    }
}

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn]
pub fn isQuantizedDataType(data_type: xnn_datatype) -> bool {
    match data_type {
        sys::xnn_datatype_qint8
        | sys::xnn_datatype_quint8
        | sys::xnn_datatype_qint32
        | sys::xnn_datatype_qcint8
        | sys::xnn_datatype_qcint32
        | sys::xnn_datatype_qcint4
        | sys::xnn_datatype_qdint8 => true,
        _ => false,
    }
}

/**
Converts dims from uint32 to size_t. Takes in a flatbuffer vector
of uint32_t and returns a std::vector of size_t. XNNPACK takes in
dims of size_t* but tensor shape is serialized in flatbuffer as
int32_t. As a result, we need to static cast the shapes to size_t
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn]
// PORT-NOTE: the C++ template defaults `T = size_t`; every call site in this
// file uses the `size_t` (Rust `usize`) instantiation, so it is monomorphized
// to `usize` here. The pointer to `fb_dims` is non-null by precondition (all
// callers null-check the field before calling).
pub fn flatbufferDimsToVector(fb_dims: ::flatbuffers::Vector<'_, u32>) -> Vec<usize> {
    let mut dims_data: Vec<usize> = Vec::new();
    dims_data.reserve(fb_dims.len());
    for fb_dim in fb_dims.iter() {
        dims_data.push(fb_dim as usize);
    }
    dims_data
}

/**
Gets the constant data pointer associated with the given tensor value.
Obtaining the constant data pointer can either be from within the flatbuffer
payload (deprecated) or via offsets to the constant_data_ptr.

Failures are returned as an Error, and the successful value may be nullptr
when the tensor has no associated constant data.
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn]
pub fn getConstantDataPtr(
    buffer_idx: u32,
    flatbuffer_graph: GraphPtr,
    constant_data_ptr: *const u8,
    constant_data_size: u64,
    named_data_map: *const dyn NamedDataMap,
    freeable_buffers: &mut Vec<FreeableBuffer>,
    weights_cache: *mut XNNWeightsCache,
    use_weight_cache: bool,
) -> Result<*const u8> {
    if buffer_idx != 0 {
        if constant_data_ptr.is_null() {
            // TODO(T172265611): Remove constant_buffer in flatbuffer path after BC
            // window
            let cb = flatbuffer_graph.constant_buffer();
            crate::et_check_or_return_error!(
                cb.is_some(),
                InvalidProgram,
                "constant_buffer is null"
            );
            let cb = cb.unwrap();
            crate::et_check_or_return_error!(
                buffer_idx < cb.len() as u32,
                InvalidProgram,
                "buffer_idx {} out of bounds for constant_buffer of size {}",
                buffer_idx,
                cb.len()
            );
            let buffer_entry = cb.get(buffer_idx as usize);
            crate::et_check_or_return_error!(
                buffer_entry.storage().is_some(),
                InvalidProgram,
                "Null constant_buffer entry at buffer_idx {}",
                buffer_idx
            );
            return Ok(buffer_entry.storage().unwrap().bytes().as_ptr() as *const u8);
        } else {
            let cd = flatbuffer_graph.constant_data();
            crate::et_check_or_return_error!(cd.is_some(), InvalidProgram, "constant_data is null");
            let cd = cd.unwrap();
            crate::et_check_or_return_error!(
                buffer_idx < cd.len() as u32,
                InvalidProgram,
                "buffer_idx {} out of bounds for constant_data of size {}",
                buffer_idx,
                cd.len()
            );
            let constant_data_offset = cd.get(buffer_idx as usize);
            let offset: u64 = constant_data_offset.offset();
            let entry_size: u64 = constant_data_offset.size();
            // PORT-NOTE: C++ uses `flatbuffers::IsFieldPresent(.., VT_NAMED_KEY)`;
            // the generated Rust accessor returns `Option<&str>`, whose
            // `is_some()` is the presence check.
            let has_named_key = constant_data_offset.named_key().is_some();
            // If there is no tensor name
            if !has_named_key {
                crate::et_check_or_return_error!(
                    offset <= constant_data_size && entry_size <= constant_data_size - offset,
                    InvalidProgram,
                    "ConstantDataOffset {{offset={}, size={}}} out of bounds for constant_data region of size {}",
                    offset,
                    entry_size,
                    constant_data_size
                );
                return Ok(unsafe { constant_data_ptr.add(offset as usize) });
            } else {
                crate::et_check_or_return_error!(
                    constant_data_offset.named_key().is_some(),
                    InvalidProgram,
                    "Named key is null"
                );
                let data_name: &str = constant_data_offset.named_key().unwrap();
                if use_weight_cache {
                    let data_ptr = unsafe { (*weights_cache).load_unpacked_data(data_name) };
                    if data_ptr.is_err() {
                        crate::et_log!(Error, "Failed to load weights from cache");
                        return Err(data_ptr.unwrap_err());
                    }
                    return Ok(data_ptr.unwrap());
                } else {
                    let buffer = unsafe { (*named_data_map).get_data(data_name) };
                    let buffer = match buffer {
                        Ok(buffer) => buffer,
                        Err(e) => {
                            crate::et_log!(
                                Error,
                                "Failed to get constant data for key {} from named_data_map. Error code: {}",
                                data_name,
                                e as u32
                            );
                            return Err(e);
                        }
                    };
                    let data_ptr = buffer.data() as *const u8;
                    freeable_buffers.push(buffer);
                    return Ok(data_ptr);
                }
            }
        }
    }

    Ok(core::ptr::null())
}

// PORT-NOTE: the C++ thin overload takes an `XNNTensorValue*` and forwards
// `tensor_value->constant_buffer_idx()` plus all other args. Rust cannot
// overload by name, so it is spelled `getConstantDataPtrForTensor`.
pub fn getConstantDataPtrForTensor(
    tensor_value: fb_xnnpack::XNNTensorValue,
    flatbuffer_graph: GraphPtr,
    constant_data_ptr: *const u8,
    constant_data_size: u64,
    named_data_map: *const dyn NamedDataMap,
    freeable_buffers: &mut Vec<FreeableBuffer>,
    weights_cache: *mut XNNWeightsCache,
    use_weight_cache: bool,
) -> Result<*const u8> {
    getConstantDataPtr(
        tensor_value.constant_buffer_idx(),
        flatbuffer_graph,
        constant_data_ptr,
        constant_data_size,
        named_data_map,
        freeable_buffers,
        weights_cache,
        use_weight_cache,
    )
}

/**
Define serialized tensor value into
the subgraph. While also keeping track of the remapped ids from
the serialized id to the newly generated id.
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn]
// PORT-NOTE: `unused_assignments` is allowed to preserve the C++ dead writes
// bug-for-bug: in the dq_datatype qint8 HACK branch `status` is assigned three
// times but only the last read; `scale_data`/`scale_numel` are initialized then
// unconditionally overwritten in both PerChannelGroupQuant sub-branches.
#[allow(unused_assignments)]
pub fn defineTensor(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &mut std::collections::HashMap<u32, u32>,
    value: ValuePtr,
    flatbuffer_graph: GraphPtr,
    constant_data_ptr: *const u8,
    constant_data_size: u64,
    input_ids: &mut Vec<u32>,
    output_ids: &mut Vec<u32>,
    allocator: &mut CompileAllocator,
    named_data_map: *const dyn NamedDataMap,
    freeable_buffers: &mut Vec<FreeableBuffer>,
    weights_cache: *mut XNNWeightsCache,
    use_weight_cache: bool,
) -> Error {
    let mut tensor_value: Option<fb_xnnpack::XNNTensorValue> = None;
    let mut qtensor_value: Option<fb_xnnpack::XNNQuantizedTensorValue> = None;

    match value.xvalue_union_type() {
        fb_xnnpack::XValueUnion::XNNTensorValue => {
            tensor_value = value.xvalue_union_as_xnntensor_value();
        }
        fb_xnnpack::XValueUnion::XNNQuantizedTensorValue => {
            qtensor_value = value.xvalue_union_as_xnnquantized_tensor_value();
            tensor_value = qtensor_value.unwrap().tensor_value();
        }
        _ => {
            crate::et_check_or_return_error!(
                false,
                NotImplemented,
                "Unhandled value type: {:?}",
                value.xvalue_union_type()
            );
        }
    }

    crate::et_check_or_return_error!(
        tensor_value.is_some(),
        InvalidProgram,
        "Deserialized tensor is null"
    );
    let tensor_value = tensor_value.unwrap();

    // Validate that tensor_value->flags() is a subset of the allowed flags.
    const K_ALLOWED_FLAGS_MASK: u32 =
        sys::XNN_VALUE_FLAG_EXTERNAL_INPUT | sys::XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    crate::et_check_or_return_error!(
        (tensor_value.flags() & !K_ALLOWED_FLAGS_MASK) == 0,
        InvalidProgram,
        "Tensor value has unsupported flag bits 0x{:x}",
        tensor_value.flags()
    );

    // Get tensor dims, here we need to use a vector in order to properly
    // convert the uint32_t* to size_t*. Scalar tensors (rank 0) are permitted
    // to have a null dims vector; in that case dims_data is empty.
    let mut dims_data: Vec<usize> = Vec::new();
    if let Some(dims) = tensor_value.dims() {
        dims_data = flatbufferDimsToVector(dims);
    }

    // XNNPACK Id
    let mut id: u32 = sys::XNN_INVALID_VALUE_ID;

    // Get Pointer to constant data from flatbuffer, if its non-constant
    // it is a nullptr
    let buffer_result = getConstantDataPtrForTensor(
        tensor_value,
        flatbuffer_graph,
        constant_data_ptr,
        constant_data_size,
        named_data_map,
        freeable_buffers,
        weights_cache,
        use_weight_cache,
    );
    if buffer_result.is_err() {
        return buffer_result.unwrap_err();
    }
    let buffer_ptr: *const u8 = buffer_result.unwrap();

    let mut status: sys::xnn_status = sys::xnn_status_success;
    // The type we might have to convert to
    let dq_datatype = getDataType(tensor_value.dq_datatype());

    if dq_datatype != sys::xnn_datatype_invalid {
        if dq_datatype != sys::xnn_datatype_qint8 {
            crate::et_check_or_return_error!(
                false,
                Internal,
                "Only int8_t is supported for dq_datatype for now, got: {:?}",
                dq_datatype
            );
        } else {
            crate::et_check_or_return_error!(
                (tensor_value.flags() & sys::XNN_VALUE_FLAG_EXTERNAL_INPUT) != 0,
                Internal,
                "Dynamic quantization of tensor is only allowed for the external input tensor value for now! got flags: {}",
                tensor_value.flags()
            );
        }
    }

    if qtensor_value.is_none() {
        // FP32 tensor
        if !isQuantizedDataType(dq_datatype) {
            // Define non-quantied tensor
            status = unsafe {
                sys::xnn_define_tensor_value(
                    /*subgraph=*/ subgraph_ptr,
                    /*datatype=*/ getDataType(tensor_value.datatype()),
                    /*num_dims=*/ dims_data.len(),
                    /*dims=*/ dims_data.as_ptr(),
                    /*data=*/ buffer_ptr as *const core::ffi::c_void,
                    /*external_id=*/ tensor_value.external_id(),
                    /*flags=*/ tensor_value.flags(),
                    /*id_out=*/ &mut id,
                )
            };
        } else if dq_datatype != sys::xnn_datatype_invalid {
            crate::et_check_or_return_error!(
                isQuantizedDataType(dq_datatype),
                Internal,
                "Dynamic quantization can only produce supported quantized dtypes"
            );
            crate::et_check_or_return_error!(
                tensor_value.external_id() != sys::XNN_INVALID_VALUE_ID,
                Internal,
                "Dynamic quantization can only work with external inputs for now, got an internal ID"
            );
            crate::et_check_or_return_error!(
                buffer_ptr.is_null(),
                Internal,
                "Dynamic quantization can only work with external inputs for now, got const data"
            );

            match dq_datatype {
                sys::xnn_datatype_qint8 => {
                    // HACK TO Maintain FC/BC for ASR this will be removed after 01/2024

                    // When encountering a dynamically quantized tensor via dq_datatype,
                    // which is the old flow for serializing dynamically quantized linear.
                    // We replace the definition of a single tensor with a new dynamic
                    // Quantization pattern. We change the pattern from:
                    //     serialized_qd_input
                    //           to
                    // (fp32_input --> convert --> qdint8_input)

                    status = unsafe {
                        sys::xnn_define_dynamically_quantized_tensor_value(
                            /*subgraph=*/ subgraph_ptr,
                            /*datatype=*/ sys::xnn_datatype_qdint8,
                            /*num_dims=*/ dims_data.len(),
                            /*num_nonbatch_dims=*/
                            1, // always do per token quantization
                            /*dims=*/ dims_data.as_ptr(),
                            /*external_id=*/
                            sys::XNN_INVALID_VALUE_ID, // always internal value id
                            /*flags=*/ 0, // this is netiher external input or output
                            /*id_out=*/ &mut id,
                        )
                    };

                    // this is the FP16 or FP32 external value that is being dynamically
                    // quantized
                    let mut float_id: u32 = 0;
                    let fp_datatype = getDataType(tensor_value.datatype());
                    status = unsafe {
                        sys::xnn_define_tensor_value(
                            /*subgraph=*/ subgraph_ptr,
                            /*datatype=*/ fp_datatype,
                            /*num_dims=*/ dims_data.len(),
                            /*dims=*/ dims_data.as_ptr(),
                            /*data=*/ buffer_ptr as *const core::ffi::c_void,
                            /*external_id=*/ tensor_value.external_id(),
                            /*flags=*/ tensor_value.flags(),
                            /*id_out=*/ &mut float_id,
                        )
                    };

                    // Define dynamic conversion from float to qdint8
                    status = unsafe {
                        sys::xnn_define_convert(
                            /*subgraph=*/ subgraph_ptr,
                            /*input_id=*/ float_id,
                            /*output_id=*/ id,
                            /*flags=*/ 0,
                        )
                    };
                }
                _ => {
                    crate::et_check_or_return_error!(
                        false,
                        NotImplemented,
                        "Unhandled Dyanmic Quantization dtype: {:?}",
                        dq_datatype
                    );
                }
            }
        } else {
            crate::et_check_or_return_error!(false, NotImplemented, "Unhandled fp32 tensor");
        }
    } else {
        let qtensor_value = qtensor_value.unwrap();
        // define tensor for quantized
        match qtensor_value.quant_params_type() {
            fb_xnnpack::XNNQuantParams::PerTensorQuant => {
                let qparams = qtensor_value.quant_params_as_per_tensor_quant().unwrap();
                crate::et_log!(
                    Debug,
                    "define quant tensor (per tensor): buffer_ptr: {:p}, scale: {}, zp: {}\n",
                    buffer_ptr,
                    qparams.scale(),
                    qparams.zero_point()
                );
                status = unsafe {
                    sys::xnn_define_quantized_tensor_value(
                        /*subgraph=*/ subgraph_ptr,
                        /*datatype=*/ getDataType(tensor_value.datatype()),
                        /*zero_point=*/ qparams.zero_point(),
                        /*scale=*/ qparams.scale(),
                        /*num_dims=*/ dims_data.len(),
                        /*dims=*/ dims_data.as_ptr(),
                        /*data=*/ buffer_ptr as *const core::ffi::c_void,
                        /*external_id=*/ tensor_value.external_id(),
                        /*flags=*/ tensor_value.flags(),
                        /*id_out=*/ &mut id,
                    )
                };
            }
            fb_xnnpack::XNNQuantParams::PerChannelQuant => {
                let qparams = qtensor_value.quant_params_as_per_channel_quant().unwrap();
                let dtype = getDataType(tensor_value.datatype());
                let zero_point: i32 = if dtype == sys::xnn_datatype_qcint4 {
                    8
                } else {
                    0
                };

                crate::et_log!(
                    Debug,
                    "define quant tensor (per channel): buffer_ptr: {:p}, scale.numel(): {}, channel_dim: {}, dtype: {:?}, zero_point: {}\n",
                    buffer_ptr,
                    qparams.scale().unwrap().len(),
                    qparams.channel_dim(),
                    dtype,
                    zero_point
                );

                let mut scale: *const f32 = qparams.scale().unwrap().bytes().as_ptr() as *const f32;

                if qparams.scale_buffer_idx() != 0 {
                    let scale_result = getConstantDataPtr(
                        qparams.scale_buffer_idx(),
                        flatbuffer_graph,
                        constant_data_ptr,
                        constant_data_size,
                        named_data_map,
                        freeable_buffers,
                        weights_cache,
                        use_weight_cache,
                    );
                    if scale_result.is_err() {
                        return scale_result.unwrap_err();
                    }
                    scale = scale_result.unwrap() as *const f32;
                    crate::et_check_or_return_error!(
                        !scale.is_null(),
                        Internal,
                        "Failed to load scale data."
                    );
                }
                status = unsafe {
                    sys::xnn_define_channelwise_quantized_tensor_value_v2(
                        /*subgraph=*/ subgraph_ptr,
                        /*datatype=*/ dtype,
                        /*zero_point=*/ zero_point,
                        /*scale=*/ scale,
                        /*num_dims=*/ dims_data.len(),
                        /*channel_dim*/ qparams.channel_dim() as usize,
                        /*dims=*/ dims_data.as_ptr(),
                        /*data=*/ buffer_ptr as *const core::ffi::c_void,
                        /*external_id=*/ tensor_value.external_id(),
                        /*flags=*/ tensor_value.flags(),
                        /*id_out=*/ &mut id,
                    )
                };
            }
            fb_xnnpack::XNNQuantParams::PerChannelGroupQuant => {
                let datatype = getDataType(tensor_value.datatype());
                crate::et_check_or_return_error!(
                    datatype == sys::xnn_datatype_qbint4,
                    Internal,
                    "Unsupported datatype for per channel group quantization: {:?}",
                    datatype
                );
                let qparams = qtensor_value
                    .quant_params_as_per_channel_group_quant()
                    .unwrap();
                let group_size: usize = qparams.group_size() as usize;
                let output_channels: usize = tensor_value.dims().unwrap().get(0) as usize;
                let input_channels: usize = tensor_value.dims().unwrap().get(1) as usize;

                let mut scale_data: *const u16 = core::ptr::null();
                let mut scale_numel: u32 = 0;

                // Block scales are preferably serialized as bf16 but can also be
                // serialized as fp32 for backwards compatibility.
                if qparams.scale_buffer_idx() != 0 {
                    let scale_data_result = getConstantDataPtr(
                        qparams.scale_buffer_idx(),
                        flatbuffer_graph,
                        constant_data_ptr,
                        constant_data_size,
                        named_data_map,
                        freeable_buffers,
                        weights_cache,
                        use_weight_cache,
                    );
                    if scale_data_result.is_err() {
                        return scale_data_result.unwrap_err();
                    }
                    scale_data = scale_data_result.unwrap() as *const u16;
                    crate::et_check_or_return_error!(
                        !scale_data.is_null(),
                        Internal,
                        "Failed to load scale data."
                    );
                    scale_numel = qparams.num_scales();
                } else {
                    // Read fp32 scales, convert to bf16.
                    let conv_buffer = allocator.allocateTemporary(
                        qparams.scale().unwrap().len() * core::mem::size_of::<u16>(),
                    ) as *mut u16;
                    scale_numel = qparams.scale().unwrap().len() as u32;
                    unsafe {
                        convertF32TensorToBF16(
                            qparams.scale().unwrap().bytes().as_ptr() as *const f32,
                            conv_buffer,
                            scale_numel as usize,
                        )
                    };
                    scale_data = conv_buffer;
                }

                crate::et_check_or_return_error!(
                    scale_numel as usize == output_channels * input_channels / group_size,
                    Internal,
                    "scale size {} != output channels {} * group size {}",
                    scale_numel as usize,
                    output_channels,
                    group_size
                );
                let zero_point: i32 = if datatype == sys::xnn_datatype_qbint4 {
                    8
                } else {
                    0
                };
                crate::et_log!(
                    Debug,
                    "define quant tensor (per channel group): buffer_ptr: {:p}, scale.numel(): {}, channel_dim: {}, grpup_size: {}, output_channels: {}, dtype: {:?}, zero_point: {}, datatype: {:?}\n",
                    buffer_ptr,
                    scale_numel,
                    qparams.channel_dim(),
                    group_size,
                    output_channels,
                    datatype,
                    zero_point,
                    datatype
                );

                status = unsafe {
                    sys::xnn_define_blockwise_quantized_tensor_value(
                        /*subgraph=*/ subgraph_ptr,
                        /*datatype=*/ datatype,
                        /*zero_point=*/ zero_point,
                        /*scale=*/ scale_data,
                        /*num_dims=*/ dims_data.len(),
                        /*channel_dim=*/ qparams.channel_dim() as usize,
                        /*block_size=*/ qparams.group_size() as usize,
                        /*dims=*/ dims_data.as_ptr(),
                        /*data=*/ buffer_ptr as *const core::ffi::c_void,
                        /*external_id=*/ tensor_value.external_id(),
                        /*flags=*/ tensor_value.flags(),
                        /*id_out=*/ &mut id,
                    )
                };
            }
            fb_xnnpack::XNNQuantParams::PerTokenDynamicQuant => {
                let qparams = qtensor_value
                    .quant_params_as_per_token_dynamic_quant()
                    .unwrap();
                crate::et_log!(
                    Debug,
                    "define quant tensor (dynamic): num_dims: {}, num_nonbatch_dims: {}\n",
                    dims_data.len(),
                    qparams.num_nonbatch_dims()
                );
                crate::et_check_or_return_error!(
                    buffer_ptr.is_null(),
                    Internal,
                    "Dynamically quantized tensor should not have constant data but found non-nullptr"
                );
                status = unsafe {
                    sys::xnn_define_dynamically_quantized_tensor_value(
                        /*subgraph=*/ subgraph_ptr,
                        /*datatype=*/ getDataType(tensor_value.datatype()),
                        /*num_dims=*/ dims_data.len(),
                        /*num_nonbatch_dims*/ qparams.num_nonbatch_dims() as usize,
                        /*dims=*/ dims_data.as_ptr(),
                        /*external_id=*/ tensor_value.external_id(),
                        /*flags=*/ tensor_value.flags(),
                        /*id_out=*/ &mut id,
                    )
                };
            }
            _ => {
                crate::et_check_or_return_error!(
                    false,
                    NotImplemented,
                    "Unhandled Quantization Parameters: {:?}",
                    qtensor_value.quant_params_type()
                );
            }
        }
    }

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to define tensor {} with code: {:?}",
        tensor_value.id_out(),
        status
    );

    // map serialized id to newly generated id
    remapped_ids.insert(tensor_value.id_out(), id);

    // Add external ids to either list of input or output ids
    if (tensor_value.flags() & sys::XNN_VALUE_FLAG_EXTERNAL_INPUT) != 0 {
        input_ids.push(tensor_value.external_id());
    }
    if (tensor_value.flags() & sys::XNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0 {
        output_ids.push(tensor_value.external_id());
    }

    Error::Ok
}

#[cfg(feature = "xnnpack_kleidi")]
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn]
pub fn isQP8(graph: GraphPtr, node: NodePtr) -> bool {
    debug_assert!(node.xnode_union_type() == fb_xnnpack::XNodeUnion::XNNConvert);
    let graph_node = node.xnode_union_as_xnnconvert().unwrap();
    let cvt_output_id = graph_node.output_id();

    let check_dtype = |id: u32, dtype: DataType| -> bool {
        for value in graph.xvalues().unwrap().iter() {
            if value.xvalue_union_type() != fb_xnnpack::XValueUnion::XNNQuantizedTensorValue {
                continue;
            }
            let tensor = value
                .xvalue_union_as_xnnquantized_tensor_value()
                .unwrap()
                .tensor_value()
                .unwrap();
            if tensor.id_out() == id {
                return tensor.datatype() == dtype;
            }
        }
        false
    };

    // Check if the output tensor is qint8 else bail early.
    if !check_dtype(cvt_output_id, DataType::xnn_datatype_qdint8) {
        return false;
    }

    // XNNPACK dtypes which have qp8 support.
    let supported_filter_dtypes: [DataType; 3] = [
        DataType::xnn_datatype_qbint4,
        DataType::xnn_datatype_qcint4,
        DataType::xnn_datatype_qcint8,
    ];

    // Find if the convert output is going to the right linear node.
    // Assuming if we can find one valid linear node, then we can use QP8
    // for all the linear nodes consuming this convert output.
    for node in graph.xnodes().unwrap().iter() {
        if node.xnode_union_type() == fb_xnnpack::XNodeUnion::XNNFullyConnected {
            let linear_node = node.xnode_union_as_xnnfully_connected().unwrap();
            if linear_node.input1_id() == cvt_output_id {
                for supported_filter_dtype in supported_filter_dtypes.iter() {
                    if check_dtype(linear_node.filter_id(), *supported_filter_dtype) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

// Safely look up a remapped tensor id. Declares `out_var` initialized to the
// value mapped from `key`, or returns Error::Internal if the key is missing.
// Avoids std::unordered_map::at(), which throws std::out_of_range inside
// noexcept functions and causes std::terminate(). Portable across MSVC,
// Clang, and GCC (no statement-expression extension).
// PORT-NOTE: C++ `REMAP_ID` is a macro that both declares the target variable
// and can `return` from the enclosing function on a missing key. This mirrors
// that: it binds `$out_var` and expands `et_check_or_return_error!`, so it must
// be used in a function returning `Error`.
macro_rules! remap_id {
    ($map:expr, $key:expr, $out_var:ident) => {
        let $out_var: u32;
        {
            let _et_remap_key: u32 = $key;
            let _et_remap_it = ($map).get(&_et_remap_key);
            crate::et_check_or_return_error!(
                _et_remap_it.is_some(),
                Internal,
                "Remapped id not found for key {}",
                _et_remap_key as core::ffi::c_uint
            );
            $out_var = *_et_remap_it.unwrap();
        }
    };
}

/*
Define Convert operator Node into the subgraph
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn]
pub fn defineConvertNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    flatbuffer_graph: GraphPtr,
) -> Error {
    let _ = flatbuffer_graph;
    let graph_node = node.xnode_union_as_xnnconvert().unwrap();

    let mut flags: i32 = graph_node.flags() as i32;
    #[cfg(feature = "xnnpack_kleidi")]
    {
        // This is not currently exposed at include/xnnpack.h yet once it is
        // we can remove this runtime logic and do this ahead-of-time
        const XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM: i32 = 0x00000100;
        if isQP8(flatbuffer_graph, node) {
            flags |= XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM;
            crate::et_log!(
                Debug,
                "Setting XNN_FLAG_MAYBE_PACK_FOR_QB4W_GEMM flag for convert node {}",
                node.debug_handle()
            );
        }
    }

    remap_id!(remapped_ids, graph_node.input_id(), cvt_input_id);
    remap_id!(remapped_ids, graph_node.output_id(), cvt_output_id);

    let status =
        unsafe { sys::xnn_define_convert(subgraph_ptr, cvt_input_id, cvt_output_id, flags as u32) };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create convert node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized linear(fully-connected) node into the subgraph using
the remapped ids to map the serialized ids, to the new ids generated
when defining the tensor values
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-fully-connected-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-fully-connected-node-fn]
pub fn defineFullyConnectedNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnfully_connected().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input1_id(), fc_input1);
    remap_id!(remapped_ids, graph_node.filter_id(), fc_filter);
    remap_id!(remapped_ids, graph_node.bias_id(), fc_bias);
    remap_id!(remapped_ids, graph_node.output_id(), fc_output);

    let status = unsafe {
        sys::xnn_define_fully_connected(
            subgraph_ptr,
            min_max.0,
            min_max.1,
            fc_input1,
            fc_filter,
            fc_bias,
            fc_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create linear node {}, with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized softmax node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining
the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-softmax-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-softmax-node-fn]
pub fn defineSoftmaxNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnsoftmax().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), sm_input);
    remap_id!(remapped_ids, graph_node.output_id(), sm_output);

    let status =
        unsafe { sys::xnn_define_softmax(subgraph_ptr, sm_input, sm_output, graph_node.flags()) };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create softmax node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-global-avg-pooling2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-global-avg-pooling2d-node-fn]
pub fn defineGlobalAvgPooling2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnglobal_avg_pooling_2d().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input_id(), gap_input);
    remap_id!(remapped_ids, graph_node.output_id(), gap_output);

    let status = unsafe {
        sys::xnn_define_global_average_pooling_2d(
            subgraph_ptr,
            min_max.0,
            min_max.1,
            gap_input,
            gap_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create global average pooling node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-avg-pooling2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-avg-pooling2d-node-fn]
pub fn defineAvgPooling2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnavg_pooling_2d().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input_id(), ap_input);
    remap_id!(remapped_ids, graph_node.output_id(), ap_output);

    let status = unsafe {
        sys::xnn_define_average_pooling_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.pooling_height(),
            graph_node.pooling_width(),
            graph_node.stride_height(),
            graph_node.stride_width(),
            min_max.0,
            min_max.1,
            ap_input,
            ap_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create average pooling node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized conv2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv2d-node-fn]
pub fn defineConv2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnconv_2d().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input1_id(), conv_input1);
    remap_id!(remapped_ids, graph_node.filter_id(), conv_filter);
    remap_id!(remapped_ids, graph_node.bias_id(), conv_bias);
    remap_id!(remapped_ids, graph_node.output_id(), conv_output);

    let status = unsafe {
        sys::xnn_define_convolution_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.kernel_height(),
            graph_node.kernel_width(),
            graph_node.subsampling_height(),
            graph_node.subsampling_width(),
            graph_node.dilation_height(),
            graph_node.dilation_width(),
            graph_node.groups(),
            graph_node.group_input_channels() as usize,
            graph_node.group_output_channels() as usize,
            min_max.0,
            min_max.1,
            conv_input1,
            conv_filter,
            conv_bias,
            conv_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create convolution node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized conv_transpose2d node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the tensor
value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv-transpose2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv-transpose2d-node-fn]
pub fn defineConvTranspose2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;
    let graph_node = node.xnode_union_as_xnnconv_transpose_2d().unwrap();

    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input1_id(), dconv_input1);
    remap_id!(remapped_ids, graph_node.filter_id(), dconv_filter);
    remap_id!(remapped_ids, graph_node.bias_id(), dconv_bias);
    remap_id!(remapped_ids, graph_node.output_id(), dconv_output);

    let status = unsafe {
        sys::xnn_define_deconvolution_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.adjustment_height(),
            graph_node.adjustment_width(),
            graph_node.kernel_height(),
            graph_node.kernel_width(),
            graph_node.subsampling_height(),
            graph_node.subsampling_width(),
            graph_node.dilation_height(),
            graph_node.dilation_width(),
            graph_node.groups(),
            graph_node.group_input_channels() as usize,
            graph_node.group_output_channels() as usize,
            min_max.0,
            min_max.1,
            dconv_input1,
            dconv_filter,
            dconv_bias,
            dconv_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create deconvolution node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized maxpool2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-max-pooling2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-max-pooling2d-node-fn]
pub fn defineMaxPooling2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnmax_pooling_2d().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input_id(), mp_input);
    remap_id!(remapped_ids, graph_node.output_id(), mp_output);

    let status = unsafe {
        sys::xnn_define_max_pooling_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.pooling_height(),
            graph_node.pooling_width(),
            graph_node.stride_height(),
            graph_node.stride_width(),
            graph_node.dilation_height(),
            graph_node.dilation_width(),
            min_max.0,
            min_max.1,
            mp_input,
            mp_output,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create maxpool2d node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized static transpose node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-transpose-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-transpose-node-fn]
pub fn defineStaticTransposeNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnstatic_transpose().unwrap();

    // Get tensor dims, we need to convert the uint32_t* to size_t*
    crate::et_check_or_return_error!(
        graph_node.perm().is_some(),
        InvalidProgram,
        "StaticTranspose: perm is null"
    );
    let dims_data: Vec<usize> = flatbufferDimsToVector(graph_node.perm().unwrap());

    remap_id!(remapped_ids, graph_node.input_id(), st_input);
    remap_id!(remapped_ids, graph_node.output_id(), st_output);

    let status = unsafe {
        sys::xnn_define_static_transpose(
            subgraph_ptr,
            dims_data.len(),
            dims_data.as_ptr(),
            st_input,
            st_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create static transpose node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized static resize bilinear 2d node into the subgraph, using the
remapped ids to map the serialized ids, to the new ids generated when defining
the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-resize-bilinear2-d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-resize-bilinear2-d-node-fn]
pub fn defineStaticResizeBilinear2DNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnstatic_resize_bilinear_2_d().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), rb_input);
    remap_id!(remapped_ids, graph_node.output_id(), rb_output);

    let status = unsafe {
        sys::xnn_define_static_resize_bilinear_2d(
            subgraph_ptr,
            graph_node.new_height() as usize,
            graph_node.new_width() as usize,
            rb_input,
            rb_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create StaticResizeBilinear2DNode node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized static constant pad node into the subgraph, using the
remapped ids to map the serialized ids, to the new ids generated when defining
the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-constant-pad-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-constant-pad-node-fn]
pub fn defineStaticConstantPadNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnstatic_constant_pad().unwrap();

    crate::et_check_or_return_error!(
        graph_node.pre_paddings().is_some() && graph_node.post_paddings().is_some(),
        InvalidProgram,
        "StaticConstantPad: pre_paddings or post_paddings is null"
    );
    let pre_paddings_dims: Vec<usize> = flatbufferDimsToVector(graph_node.pre_paddings().unwrap());
    let post_paddings_dims: Vec<usize> =
        flatbufferDimsToVector(graph_node.post_paddings().unwrap());

    remap_id!(remapped_ids, graph_node.input_id(), scp_input);
    remap_id!(remapped_ids, graph_node.output_id(), scp_output);

    let status = unsafe {
        sys::xnn_define_static_constant_pad(
            subgraph_ptr,
            pre_paddings_dims.as_ptr(),
            post_paddings_dims.as_ptr(),
            graph_node.padding_value(),
            scp_input,
            scp_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create StaticConstantPad node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized depthwise conv2d node into the subgraph, using the remapped
ids to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-depthwise-conv2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-depthwise-conv2d-node-fn]
pub fn defineDepthwiseConv2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnndepthwise_conv_2d().unwrap();
    let min_max = getOutputMinMax(node);
    remap_id!(remapped_ids, graph_node.input1_id(), dw_input1);
    remap_id!(remapped_ids, graph_node.filter_id(), dw_filter);
    remap_id!(remapped_ids, graph_node.bias_id(), dw_bias);
    remap_id!(remapped_ids, graph_node.output_id(), dw_output);

    let status = unsafe {
        sys::xnn_define_depthwise_convolution_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.kernel_height(),
            graph_node.kernel_width(),
            graph_node.subsampling_height(),
            graph_node.subsampling_width(),
            graph_node.dilation_height(),
            graph_node.dilation_width(),
            graph_node.group_output_channels() / graph_node.group_input_channels(), // depth_multiplier
            graph_node.groups() as usize, // input_channels = groups for depthwise conv
            min_max.0,
            min_max.1,
            dw_input1,
            dw_filter,
            dw_bias,
            dw_output,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create depthwise convolution node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-reshape-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-reshape-node-fn]
pub fn defineStaticReshapeNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnstatic_reshape().unwrap();

    // Get tensor dims, we need to convert the uint32_t* to size_t*
    crate::et_check_or_return_error!(
        graph_node.new_shape().is_some(),
        InvalidProgram,
        "StaticReshape: new_shape is null"
    );
    let dims_data: Vec<usize> = flatbufferDimsToVector(graph_node.new_shape().unwrap());

    remap_id!(remapped_ids, graph_node.input_id(), sr_input);
    remap_id!(remapped_ids, graph_node.output_id(), sr_output);

    let status = unsafe {
        sys::xnn_define_static_reshape(
            subgraph_ptr,
            dims_data.len(),
            dims_data.as_ptr(),
            sr_input,
            sr_output,
            graph_node.flags(),
        )
    };
    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create squeeze node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized maxpool2d node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-arg-max-pooling2d-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-arg-max-pooling2d-node-fn]
pub fn defineArgMaxPooling2dNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnarg_max_pooling_2d().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), amp_input);
    remap_id!(remapped_ids, graph_node.output_value_id(), amp_out_val);
    remap_id!(remapped_ids, graph_node.output_index_id(), amp_out_idx);

    let status = unsafe {
        sys::xnn_define_argmax_pooling_2d(
            subgraph_ptr,
            graph_node.padding_top(),
            graph_node.padding_right(),
            graph_node.padding_bottom(),
            graph_node.padding_left(),
            graph_node.pooling_height(),
            graph_node.pooling_width(),
            amp_input,
            amp_out_val,
            amp_out_idx,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create argmaxpool2d node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized exp node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-exp-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-exp-node-fn]
pub fn defineExpNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnexp().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), exp_input);
    remap_id!(remapped_ids, graph_node.output_id(), exp_output);

    let status =
        unsafe { sys::xnn_define_exp(subgraph_ptr, exp_input, exp_output, graph_node.flags()) };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create exp node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Define serialized tanh node into the subgraph, using the remapped ids
to map the serialized ids, to the new ids generated when defining the
tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-tanh-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tanh-node-fn]
pub fn defineTanhNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnntanh().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), tanh_input);
    remap_id!(remapped_ids, graph_node.output_id(), tanh_output);

    let status =
        unsafe { sys::xnn_define_tanh(subgraph_ptr, tanh_input, tanh_output, graph_node.flags()) };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create tanh node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized prelu node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-p-re-lu-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-p-re-lu-node-fn]
pub fn definePReLUNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnpre_lu().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), prelu_input1);
    remap_id!(remapped_ids, graph_node.input2_id(), prelu_input2);
    remap_id!(remapped_ids, graph_node.output_id(), prelu_output);

    let status = unsafe {
        sys::xnn_define_prelu(
            subgraph_ptr,
            prelu_input1,
            prelu_input2,
            prelu_output,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create prelu node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized concatenate2 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate2-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate2-node-fn]
pub fn defineConcatenate2Node(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnconcatenate_2().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), cat2_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), cat2_in2);
    remap_id!(remapped_ids, graph_node.output_id(), cat2_out);

    let status = unsafe {
        sys::xnn_define_concatenate2(
            subgraph_ptr,
            graph_node.axis() as usize,
            cat2_in1,
            cat2_in2,
            cat2_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create cat2 node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized concatenate3 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate3-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate3-node-fn]
pub fn defineConcatenate3Node(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnconcatenate_3().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), cat3_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), cat3_in2);
    remap_id!(remapped_ids, graph_node.input3_id(), cat3_in3);
    remap_id!(remapped_ids, graph_node.output_id(), cat3_out);

    let status = unsafe {
        sys::xnn_define_concatenate3(
            subgraph_ptr,
            graph_node.axis() as usize,
            cat3_in1,
            cat3_in2,
            cat3_in3,
            cat3_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create cat3 node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized concatenate4 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate4-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate4-node-fn]
pub fn defineConcatenate4Node(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnconcatenate_4().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), cat4_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), cat4_in2);
    remap_id!(remapped_ids, graph_node.input3_id(), cat4_in3);
    remap_id!(remapped_ids, graph_node.input4_id(), cat4_in4);
    remap_id!(remapped_ids, graph_node.output_id(), cat4_out);

    let status = unsafe {
        sys::xnn_define_concatenate4(
            subgraph_ptr,
            graph_node.axis() as usize,
            cat4_in1,
            cat4_in2,
            cat4_in3,
            cat4_in4,
            cat4_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create cat4 node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized concatenate5 node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate5-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate5-node-fn]
pub fn defineConcatenate5Node(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnconcatenate_5().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), cat5_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), cat5_in2);
    remap_id!(remapped_ids, graph_node.input3_id(), cat5_in3);
    remap_id!(remapped_ids, graph_node.input4_id(), cat5_in4);
    remap_id!(remapped_ids, graph_node.input5_id(), cat5_in5);
    remap_id!(remapped_ids, graph_node.output_id(), cat5_out);

    let status = unsafe {
        sys::xnn_define_concatenate5(
            subgraph_ptr,
            graph_node.axis() as usize,
            cat5_in1,
            cat5_in2,
            cat5_in3,
            cat5_in4,
            cat5_in5,
            cat5_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create cat5 node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines serialized static_slice node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-slice-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-slice-node-fn]
pub fn defineStaticSliceNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnstatic_slice().unwrap();

    crate::et_check_or_return_error!(
        graph_node.offsets().is_some() && graph_node.sizes().is_some(),
        InvalidProgram,
        "StaticSlice: offsets or sizes is null"
    );
    let offsets: Vec<usize> = flatbufferDimsToVector(graph_node.offsets().unwrap());
    let sizes: Vec<usize> = flatbufferDimsToVector(graph_node.sizes().unwrap());

    crate::et_check_or_return_error!(
        offsets.len() == sizes.len(),
        InvalidProgram,
        "StaticSlice: offsets size {} does not match sizes size {}",
        offsets.len(),
        sizes.len()
    );

    remap_id!(remapped_ids, graph_node.input_id(), ss_input);
    remap_id!(remapped_ids, graph_node.output_id(), ss_output);

    let status = unsafe {
        sys::xnn_define_static_slice(
            subgraph_ptr,
            offsets.len(),
            offsets.as_ptr(),
            sizes.as_ptr(),
            ss_input,
            ss_output,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create static slice node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Defines batch matrix multiply node into the subgraph,
using the remapped ids to map the serialized ids,
to the new ids generated when defining the tensor value
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-batch-matrix-multiply-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-batch-matrix-multiply-node-fn]
pub fn defineBatchMatrixMultiplyNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnnbatch_matrix_multiply().unwrap();
    remap_id!(remapped_ids, graph_node.input1_id(), bmm_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), bmm_in2);
    remap_id!(remapped_ids, graph_node.output_id(), bmm_out);

    let status = unsafe {
        sys::xnn_define_batch_matrix_multiply(
            subgraph_ptr,
            bmm_in1,
            bmm_in2,
            bmm_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create BMM node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
 * Defines a copy node in the XNN subgraph.
 */
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-copy-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-copy-node-fn]
pub fn defineCopyNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = graph;

    let graph_node = node.xnode_union_as_xnncopy().unwrap();
    remap_id!(remapped_ids, graph_node.input_id(), copy_input);
    remap_id!(remapped_ids, graph_node.output_id(), copy_output);

    let status =
        unsafe { sys::xnn_define_copy(subgraph_ptr, copy_input, copy_output, graph_node.flags()) };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create copy node {} with code: {:?}",
        node.debug_handle(),
        status
    );

    Error::Ok
}

/*
Returns not Implemented Error code. This function is meant to be
called when the compiler encountes a XNodeType from the flatbuffer
that has not yet been implemented
*/
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn]
pub fn defineNotImplementedNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    node: NodePtr,
    graph: GraphPtr,
) -> Error {
    let _ = subgraph_ptr;
    let _ = remapped_ids;
    let _ = graph;

    crate::et_check_or_return_error!(
        false,
        NotImplemented,
        "Unhandled node type: {:?}",
        node.xnode_union_type()
    );

    Error::Ok
}

// Generic helper function for unary operations
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn]
pub fn defineGenericUnaryNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    input_id: u32,
    output_id: u32,
    flags: u32,
    op_type: sys::xnn_unary_operator,
    params: *const sys::xnn_unary_params,
    node_type: fb_xnnpack::XNodeUnion,
    debug_handle: u32,
) -> Error {
    remap_id!(remapped_ids, input_id, remapped_input);
    remap_id!(remapped_ids, output_id, remapped_output);

    let status = unsafe {
        sys::xnn_define_unary(
            subgraph_ptr,
            op_type,
            params,
            remapped_input,
            remapped_output,
            flags,
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create {:?} node {} with code: {:?}",
        node_type,
        debug_handle,
        status
    );

    Error::Ok
}

// Macro for unary operations with no parameters
// PORT-NOTE: the C++ `_DEFINE_UNARY_NODE_NO_PARAMS(name, op_type)` macro
// token-pastes `define##name##Node` and `xnode_union_as_XNN##name`. Rust
// `macro_rules!` cannot paste identifiers, so the Rust definer name and the
// snake_case accessor are passed explicitly.
macro_rules! define_unary_node_no_params {
    ($fn_name:ident, $accessor:ident, $op_type:expr) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.$accessor().unwrap();
            defineGenericUnaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node.input_id(),
                graph_node.output_id(),
                graph_node.flags(),
                $op_type,
                core::ptr::null(),
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Macro for unary operations with min/max parameters
macro_rules! define_unary_node_with_minmax {
    ($fn_name:ident, $accessor:ident, $op_type:expr) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.$accessor().unwrap();
            let min_max = getOutputMinMax(node);
            let params = sys::xnn_unary_params {
                clamp: sys::xnn_unary_clamp_params {
                    min: min_max.0,
                    max: min_max.1,
                },
            };
            defineGenericUnaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node.input_id(),
                graph_node.output_id(),
                graph_node.flags(),
                $op_type,
                &params,
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Macro for unary operations with leaky_relu parameters
macro_rules! define_unary_node_with_leaky_relu {
    ($fn_name:ident) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.xnode_union_as_xnnleaky_re_lu().unwrap();
            let params = sys::xnn_unary_params {
                leaky_relu: sys::xnn_unary_leaky_relu_params {
                    negative_slope: graph_node.negative_slope(),
                },
            };
            defineGenericUnaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node.input_id(),
                graph_node.output_id(),
                graph_node.flags(),
                sys::xnn_unary_leaky_relu,
                &params,
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Macro for unary operations with elu parameters
macro_rules! define_unary_node_with_elu {
    ($fn_name:ident) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.xnode_union_as_xnnelu().unwrap();
            let params = sys::xnn_unary_params {
                elu: sys::xnn_unary_elu_params {
                    alpha: graph_node.alpha(),
                },
            };
            defineGenericUnaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node.input_id(),
                graph_node.output_id(),
                graph_node.flags(),
                sys::xnn_unary_elu,
                &params,
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Generic helper function for binary operations
// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn]
pub fn defineGenericBinaryNode(
    subgraph_ptr: xnn_subgraph_t,
    remapped_ids: &std::collections::HashMap<u32, u32>,
    graph_node: fb_xnnpack::_XNNNode2x1,
    op_type: sys::xnn_binary_operator,
    params: *const sys::xnn_binary_params,
    node_type: fb_xnnpack::XNodeUnion,
    debug_handle: u32,
) -> Error {
    remap_id!(remapped_ids, graph_node.input1_id(), bin_in1);
    remap_id!(remapped_ids, graph_node.input2_id(), bin_in2);
    remap_id!(remapped_ids, graph_node.output_id(), bin_out);

    let status = unsafe {
        sys::xnn_define_binary(
            subgraph_ptr,
            op_type,
            params,
            bin_in1,
            bin_in2,
            bin_out,
            graph_node.flags(),
        )
    };

    crate::et_check_or_return_error!(
        status == sys::xnn_status_success,
        Internal,
        "Failed to create {:?} node {} with code: {:?}",
        node_type,
        debug_handle,
        status
    );

    Error::Ok
}

// Macro for binary operations with min/max parameters
macro_rules! define_binary_node_with_minmax {
    ($fn_name:ident, $accessor:ident, $op_type:expr) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.$accessor().unwrap();
            let min_max = getOutputMinMax(node);
            let params = sys::xnn_binary_params {
                output_min: min_max.0 as f64,
                output_max: min_max.1 as f64,
            };
            defineGenericBinaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node,
                $op_type,
                &params,
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Macro for binary operations without parameters
macro_rules! define_binary_node_no_params {
    ($fn_name:ident, $accessor:ident, $op_type:expr) => {
        pub fn $fn_name(
            subgraph_ptr: xnn_subgraph_t,
            remapped_ids: &std::collections::HashMap<u32, u32>,
            node: NodePtr,
            graph: GraphPtr,
        ) -> Error {
            let _ = graph;
            let graph_node = node.$accessor().unwrap();
            defineGenericBinaryNode(
                subgraph_ptr,
                remapped_ids,
                graph_node,
                $op_type,
                core::ptr::null(),
                node.xnode_union_type(),
                node.debug_handle(),
            )
        }
    };
}

// Unary Ops with no params
define_unary_node_no_params!(
    defineSigmoidNode,
    xnode_union_as_xnnsigmoid,
    sys::xnn_unary_sigmoid
);
define_unary_node_no_params!(
    defineFloorNode,
    xnode_union_as_xnnfloor,
    sys::xnn_unary_floor
);
define_unary_node_no_params!(
    defineSquareRootNode,
    xnode_union_as_xnnsquare_root,
    sys::xnn_unary_square_root
);
define_unary_node_no_params!(
    defineReciprocalSquareRootNode,
    xnode_union_as_xnnreciprocal_square_root,
    sys::xnn_unary_reciprocal_square_root
);
define_unary_node_no_params!(
    defineCeilingNode,
    xnode_union_as_xnnceiling,
    sys::xnn_unary_ceiling
);
define_unary_node_no_params!(defineGeluNode, xnode_union_as_xnngelu, sys::xnn_unary_gelu);
define_unary_node_no_params!(
    defineHardswishNode,
    xnode_union_as_xnnhardswish,
    sys::xnn_unary_hardswish
);
define_unary_node_no_params!(defineLogNode, xnode_union_as_xnnlog, sys::xnn_unary_log);
define_unary_node_no_params!(
    defineNegateNode,
    xnode_union_as_xnnnegate,
    sys::xnn_unary_negate
);
define_unary_node_no_params!(
    defineSquareNode,
    xnode_union_as_xnnsquare,
    sys::xnn_unary_square
);
define_unary_node_no_params!(defineAbsNode, xnode_union_as_xnnabs, sys::xnn_unary_abs);
define_unary_node_no_params!(defineSinNode, xnode_union_as_xnnsin, sys::xnn_unary_sine);
define_unary_node_no_params!(defineCosNode, xnode_union_as_xnncos, sys::xnn_unary_cosine);

// Unary Ops with min/max params
define_unary_node_with_minmax!(
    defineClampNode,
    xnode_union_as_xnnclamp,
    sys::xnn_unary_clamp
);

// Unary Ops with specific params
define_unary_node_with_leaky_relu!(defineLeakyReLUNode);
define_unary_node_with_elu!(defineELUNode);

// Binary Ops with params
define_binary_node_with_minmax!(defineAddNode, xnode_union_as_xnnadd, sys::xnn_binary_add);
define_binary_node_with_minmax!(
    defineSubtractNode,
    xnode_union_as_xnnsubtract,
    sys::xnn_binary_subtract
);
define_binary_node_with_minmax!(
    defineMultiplyNode,
    xnode_union_as_xnnmultiply,
    sys::xnn_binary_multiply
);
define_binary_node_with_minmax!(defineDivNode, xnode_union_as_xnndiv, sys::xnn_binary_divide);

// Binary Ops without params
define_binary_node_no_params!(
    defineMinimumNode,
    xnode_union_as_xnnminimum,
    sys::xnn_binary_minimum
);
define_binary_node_no_params!(
    defineMaximumNode,
    xnode_union_as_xnnmaximum,
    sys::xnn_binary_maximum
);

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]
// [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn]
pub fn getDefineNodeFunc(node_type: fb_xnnpack::XNodeUnion) -> DefineNodeFunc {
    match node_type {
        // Binary ops
        fb_xnnpack::XNodeUnion::XNNAdd => defineAddNode,
        fb_xnnpack::XNodeUnion::XNNSubtract => defineSubtractNode,
        fb_xnnpack::XNodeUnion::XNNMultiply => defineMultiplyNode,
        fb_xnnpack::XNodeUnion::XNNDiv => defineDivNode,
        fb_xnnpack::XNodeUnion::XNNMinimum => defineMinimumNode,
        fb_xnnpack::XNodeUnion::XNNMaximum => defineMaximumNode,

        // Unary ops
        fb_xnnpack::XNodeUnion::XNNSoftmax => defineSoftmaxNode,
        fb_xnnpack::XNodeUnion::XNNSquareRoot => defineSquareRootNode,
        fb_xnnpack::XNodeUnion::XNNReciprocalSquareRoot => defineReciprocalSquareRootNode,
        fb_xnnpack::XNodeUnion::XNNCeiling => defineCeilingNode,
        fb_xnnpack::XNodeUnion::XNNGelu => defineGeluNode,
        fb_xnnpack::XNodeUnion::XNNHardswish => defineHardswishNode,
        fb_xnnpack::XNodeUnion::XNNLog => defineLogNode,
        fb_xnnpack::XNodeUnion::XNNTanh => defineTanhNode,
        fb_xnnpack::XNodeUnion::XNNNegate => defineNegateNode,
        fb_xnnpack::XNodeUnion::XNNSquare => defineSquareNode,
        fb_xnnpack::XNodeUnion::XNNClamp => defineClampNode,
        fb_xnnpack::XNodeUnion::XNNLeakyReLU => defineLeakyReLUNode,
        fb_xnnpack::XNodeUnion::XNNELU => defineELUNode,
        fb_xnnpack::XNodeUnion::XNNExp => defineExpNode,
        fb_xnnpack::XNodeUnion::XNNAbs => defineAbsNode,
        fb_xnnpack::XNodeUnion::XNNFloor => defineFloorNode,
        fb_xnnpack::XNodeUnion::XNNPReLU => definePReLUNode,
        fb_xnnpack::XNodeUnion::XNNSigmoid => defineSigmoidNode,
        fb_xnnpack::XNodeUnion::XNNSin => defineSinNode,
        fb_xnnpack::XNodeUnion::XNNCos => defineCosNode,

        // Others
        fb_xnnpack::XNodeUnion::XNNFullyConnected => defineFullyConnectedNode,
        fb_xnnpack::XNodeUnion::XNNStaticTranspose => defineStaticTransposeNode,
        fb_xnnpack::XNodeUnion::XNNConv2d => defineConv2dNode,
        fb_xnnpack::XNodeUnion::XNNConvTranspose2d => defineConvTranspose2dNode,
        fb_xnnpack::XNodeUnion::XNNStaticResizeBilinear2D => defineStaticResizeBilinear2DNode,
        fb_xnnpack::XNodeUnion::XNNStaticConstantPad => defineStaticConstantPadNode,
        fb_xnnpack::XNodeUnion::XNNAvgPooling2d => defineAvgPooling2dNode,
        fb_xnnpack::XNodeUnion::XNNDepthwiseConv2d => defineDepthwiseConv2dNode,
        fb_xnnpack::XNodeUnion::XNNMaxPooling2d => defineMaxPooling2dNode,
        fb_xnnpack::XNodeUnion::XNNConvert => defineConvertNode,
        fb_xnnpack::XNodeUnion::XNNGlobalAvgPooling2d => defineGlobalAvgPooling2dNode,
        fb_xnnpack::XNodeUnion::XNNStaticReshape => defineStaticReshapeNode,
        fb_xnnpack::XNodeUnion::XNNArgMaxPooling2d => defineArgMaxPooling2dNode,
        fb_xnnpack::XNodeUnion::XNNConcatenate2 => defineConcatenate2Node,
        fb_xnnpack::XNodeUnion::XNNConcatenate3 => defineConcatenate3Node,
        fb_xnnpack::XNodeUnion::XNNConcatenate4 => defineConcatenate4Node,
        fb_xnnpack::XNodeUnion::XNNConcatenate5 => defineConcatenate5Node,
        fb_xnnpack::XNodeUnion::XNNStaticSlice => defineStaticSliceNode,
        fb_xnnpack::XNodeUnion::XNNBatchMatrixMultiply => defineBatchMatrixMultiplyNode,
        fb_xnnpack::XNodeUnion::XNNCopy => defineCopyNode,

        // fb_xnnpack::XNodeUnion::NONE and any type not listed above (default) -
        // Adding here as a catch all, just in case
        _ => defineNotImplementedNode,
    }
}

// [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler]
pub struct XNNCompiler;

impl XNNCompiler {
    // [spec:et:def:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn]
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn]
    #[allow(clippy::too_many_arguments)]
    pub fn compileModel(
        buffer_pointer: *const core::ffi::c_void,
        num_bytes: usize,
        executor: *mut crate::backends::xnnpack::runtime::XNNExecutor::XNNExecutor,
        weights_cache: *mut XNNWeightsCache,
        workspace: sys::xnn_workspace_t,
        named_data_map: *const dyn NamedDataMap,
        use_weight_cache: bool,
    ) -> Error {
        // PORT-NOTE: C++ carries raw `const void*`/`size_t`. To reach the
        // flatbuffer/header helpers (which take `&[u8]`) we build the backing
        // slice once here; individual constant-data pointers stay raw to match
        // the C++ pointer arithmetic bug-for-bug.
        let buffer_slice: &[u8] =
            unsafe { core::slice::from_raw_parts(buffer_pointer as *const u8, num_bytes) };

        let header: Result<XNNHeader> = XNNHeader::parse(buffer_slice);
        let flatbuffer_data: *const u8;
        let mut constant_data: *const u8 = core::ptr::null();
        let mut constant_data_size: u64 = 0;
        let flatbuffer_size: usize;
        let mut compile_allocator = CompileAllocator::new();

        // Header status can only either be Error::Ok or Error::NotFound
        match header {
            Ok(header) => {
                flatbuffer_data =
                    unsafe { (buffer_pointer as *const u8).add(header.flatbuffer_offset as usize) };
                flatbuffer_size = header.flatbuffer_size as usize;
                constant_data = unsafe {
                    (buffer_pointer as *const u8).add(header.constant_data_offset as usize)
                };
                constant_data_size = header.constant_data_size;
            }
            Err(Error::NotFound) => {
                flatbuffer_data = buffer_pointer as *const u8;
                flatbuffer_size = num_bytes;
            }
            Err(e) => {
                crate::et_log!(Error, "XNNHeader may be corrupt");
                return e;
            }
        }

        // PORT-NOTE: C++ uses raw `flatbuffer_data`/`flatbuffer_size`. The Rust
        // flatbuffers helpers (identifier check, verifier, root accessor) take a
        // `&[u8]`, so reconstruct that view over the flatbuffer region.
        let flatbuffer_slice: &[u8] =
            unsafe { core::slice::from_raw_parts(flatbuffer_data, flatbuffer_size) };

        // Temporarily support identifier XN00 and XN01
        let is_supported_version = fb_xnnpack::xnngraph_buffer_has_identifier(flatbuffer_slice)
            || ::flatbuffers::buffer_has_identifier(flatbuffer_slice, "XN01", false);
        crate::et_check_or_return_error!(
            is_supported_version,
            DelegateInvalidCompatibility,
            "XNNPACK Delegate Serialization Format version identifier '{:.4}' != expected XN00 or XN01'",
            core::str::from_utf8(&flatbuffer_slice[4..8]).unwrap_or("????")
        );

        // Verify the FlatBuffer data integrity before accessing it. Without this,
        // malformed data could cause out-of-bounds reads when traversing the
        // FlatBuffer's internal offset tables.
        // PORT-NOTE: C++ constructs a `flatbuffers::Verifier` then calls
        // `VerifyBuffer<XNNGraph>`. The Rust generated `root_as_xnngraph` runs the
        // equivalent verifier and returns the root on success, folding steps 4 and
        // 5's `GetXNNGraph` into one call.
        let flatbuffer_graph = fb_xnnpack::root_as_xnngraph(flatbuffer_slice);
        crate::et_check_or_return_error!(
            flatbuffer_graph.is_ok(),
            DelegateInvalidCompatibility,
            "FlatBuffer verification failed; data may be truncated or corrupt"
        );
        let flatbuffer_graph = flatbuffer_graph.unwrap();
        crate::et_check_or_return_error!(
            flatbuffer_graph.xvalues().is_some() && flatbuffer_graph.xnodes().is_some(),
            InvalidProgram,
            "Failed to deserialize XNNPACK flatbuffer graph; null graph, xvalues, or xnodes."
        );

        // initialize xnnpack
        let mut status: sys::xnn_status = unsafe { sys::xnn_initialize(core::ptr::null()) };
        crate::et_check_or_return_error!(
            sys::xnn_status_success == status,
            Internal,
            "XNN Initialize failed with code: {:?}",
            status
        );

        // create xnnpack subgraph
        let num_externs: u32 = flatbuffer_graph.num_externs();
        crate::et_check_or_return_error!(
            num_externs <= 4096,
            InvalidProgram,
            "XNNPACK flatbuffer blob has num_externs ({}) which exceeds maximum (4096). This likely indicates a corrupted or invalid serialized graph",
            num_externs
        );

        let mut subgraph_ptr: sys::xnn_subgraph_t = sys::xnn_subgraph_t(core::ptr::null_mut());
        status = unsafe {
            sys::xnn_create_subgraph(
                /*external_value_ids=*/ num_externs,
                /*flags=*/ 0,
                &mut subgraph_ptr,
            )
        };
        crate::et_check_or_return_error!(
            sys::xnn_status_success == status,
            Internal,
            "XNN Subgraph creation failed with code: {:?}",
            status
        );

        // PORT-NOTE: C++ wraps the subgraph in a
        // `unique_ptr<xnn_subgraph, &xnn_delete_subgraph>` so it is deleted on
        // every early return (RAII). Rust mirrors that with a Drop guard whose
        // destructor calls `xnn_delete_subgraph`. `subgraph.ptr` is the live
        // handle passed to the definers and runtime creation.
        struct SubgraphGuard {
            ptr: sys::xnn_subgraph_t,
        }
        impl Drop for SubgraphGuard {
            fn drop(&mut self) {
                unsafe {
                    sys::xnn_delete_subgraph(self.ptr);
                }
            }
        }
        let subgraph = SubgraphGuard { ptr: subgraph_ptr };

        // mapping from old ids to new created value ids
        // The old ids that were serialied were generated AoT, since
        // we are re-defining tensor values, the defined IDs could be
        // different from the ones generated AoT, as a result, we need
        // a new mapping from the old ids to the newly created ones
        let mut remapped_ids: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        // Invalid ids do not need to be remapped
        remapped_ids.insert(sys::XNN_INVALID_VALUE_ID, sys::XNN_INVALID_VALUE_ID);

        // If weight cache is not on we hold onto all the unpacked buffers
        // and we free them at the end
        let mut unpacked_buffers: Vec<FreeableBuffer> = Vec::new();

        // External Ids for inputs and outputs
        let mut input_ids: Vec<u32> = Vec::new();
        let mut output_ids: Vec<u32> = Vec::new();
        let mut err: Error = Error::Ok;
        for value in flatbuffer_graph.xvalues().unwrap().iter() {
            err = defineTensor(
                subgraph.ptr,
                &mut remapped_ids,
                value,
                flatbuffer_graph,
                constant_data,
                constant_data_size,
                &mut input_ids,
                &mut output_ids,
                &mut compile_allocator,
                named_data_map,
                &mut unpacked_buffers,
                weights_cache,
                use_weight_cache,
            );

            if err != Error::Ok {
                return err;
            }
        }

        for node in flatbuffer_graph.xnodes().unwrap().iter() {
            err = getDefineNodeFunc(node.xnode_union_type())(
                subgraph.ptr,
                &remapped_ids,
                node,
                flatbuffer_graph,
            );
            if err != Error::Ok {
                return err;
            }
        }
        let mut runtime_flags: u32 = 0;
        #[cfg(not(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        )))]
        let profile_runtime = false;

        #[cfg(any(
            feature = "xnnpack-profiling",
            feature = "profiling-enabled",
            feature = "event-tracer"
        ))]
        let profile_runtime = profile_runtime_enabled();

        if profile_runtime {
            runtime_flags |= sys::XNN_FLAG_BASIC_PROFILING;
        }

        let mut runtime_ptr: sys::xnn_runtime_t = sys::xnn_runtime_t(core::ptr::null_mut());

        let mut weights_cache_ptr: sys::xnn_weights_cache_t =
            sys::xnn_weights_cache_t(core::ptr::null_mut());
        if use_weight_cache {
            crate::et_check_or_return_error!(
                unpacked_buffers.is_empty(),
                Internal,
                "Weight Cache is enabled, which means unpacked buffers should be owned by the cache"
            );
            weights_cache_ptr = if unsafe { (*weights_cache).get_num_unpacked_data() } > 0 {
                unsafe { (*weights_cache).get() }
            } else {
                sys::xnn_weights_cache_t(core::ptr::null_mut())
            };
        }

        // NOLINTBEGIN(facebook-hte-NullableDereference) - weights cache is allowed
        // to be null
        status = unsafe {
            sys::xnn_create_runtime_v4(
                subgraph.ptr,
                weights_cache_ptr,
                workspace,
                sys::get_pthreadpool(),
                runtime_flags,
                &mut runtime_ptr,
            )
        };
        // NOLINTEND(facebook-hte-NullableDereference)

        crate::et_check_or_return_error!(
            sys::xnn_status_success == status,
            Internal,
            "XNN Runtime creation failed with code: {:?}",
            status
        );

        let mut packed_weights_names: Vec<String> = Vec::new();
        if use_weight_cache {
            let packed_weights_names_result = unsafe { (*weights_cache).finalize_for_runtime() };
            crate::et_check_or_return_error!(
                packed_weights_names_result.is_ok(),
                Internal,
                "Failed to finalize weights cache after creating the xnn runtime"
            );
            packed_weights_names = packed_weights_names_result.unwrap();
        }
        // PORT-NOTE (deviation from C++): the C++ frees unpacked_buffers here,
        // violating xnnpack.h's contract that static tensor data outlive the
        // Runtime (upstream survives only because its named-data constants all
        // feed weight-packing consumers). Constants feeding binary/elementwise
        // ops are read at invoke time, so the buffers are moved into the
        // executor and live until delegate destroy.

        err = unsafe {
            (*executor).initialize(
                // NOLINT: runtime_ptr is non-null
                runtime_ptr,
                input_ids,
                output_ids,
                packed_weights_names,
                unpacked_buffers,
                profile_runtime,
            )
        };

        err
    }
}

// PORT-NOTE: the vendored XNNPACK C library is linked whenever the `xnnpack`
// feature is on (see build.rs), so these tests exercise both the pure helpers
// (getDataType, isQuantizedDataType, flatbufferDimsToVector, getOutputMinMax,
// convertF32TensorToBF16, getConstantDataPtr) and the link-dependent paths:
// defineTensor / the define*Node definers against a real `xnn_subgraph`, and
// compileModel end-to-end over a serialized XNNGraph flatbuffer built with the
// generated builders (the same wire format the AoT serializer emits).
#[cfg(test)]
mod tests {
    use super::*;

    // getDataType maps every serialized flatbuffer XNNDatatype onto the
    // corresponding xnn_datatype, mirroring the C++ switch exactly, and any
    // unlisted value falls through to xnn_datatype_invalid.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-data-type-fn/test]
    #[test]
    fn get_data_type_maps_every_case() {
        use fb_xnnpack::XNNDatatype as D;
        assert_eq!(getDataType(D::xnn_datatype_fp32), sys::xnn_datatype_fp32);
        assert_eq!(getDataType(D::xnn_datatype_fp16), sys::xnn_datatype_fp16);
        assert_eq!(getDataType(D::xnn_datatype_qint8), sys::xnn_datatype_qint8);
        assert_eq!(
            getDataType(D::xnn_datatype_quint8),
            sys::xnn_datatype_quint8
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qint32),
            sys::xnn_datatype_qint32
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qcint8),
            sys::xnn_datatype_qcint8
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qcint32),
            sys::xnn_datatype_qcint32
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qcint4),
            sys::xnn_datatype_qcint4
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qdint8),
            sys::xnn_datatype_qdint8
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qbint4),
            sys::xnn_datatype_qbint4
        );
        assert_eq!(
            getDataType(D::xnn_datatype_qpint8),
            sys::xnn_datatype_qpint8
        );
        assert_eq!(getDataType(D::xnn_datatype_int32), sys::xnn_datatype_int32);
        assert_eq!(getDataType(D::xnn_datatype_pfp32), sys::xnn_datatype_pfp32);
        assert_eq!(getDataType(D::xnn_datatype_bf16), sys::xnn_datatype_bf16);
        // invalid and any unknown discriminant map to invalid.
        assert_eq!(
            getDataType(D::xnn_datatype_invalid),
            sys::xnn_datatype_invalid
        );
        assert_eq!(
            getDataType(fb_xnnpack::XNNDatatype(999)),
            sys::xnn_datatype_invalid
        );
    }

    // isQuantizedDataType returns true for exactly the seven quantized types in
    // the C++ switch (qint8, quint8, qint32, qcint8, qcint32, qcint4, qdint8)
    // and false for everything else — notably NOT qbint4/qpint8, which the C++
    // deliberately omits.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-quantized-data-type-fn/test]
    #[test]
    fn is_quantized_data_type_matches_cpp_set() {
        for &q in &[
            sys::xnn_datatype_qint8,
            sys::xnn_datatype_quint8,
            sys::xnn_datatype_qint32,
            sys::xnn_datatype_qcint8,
            sys::xnn_datatype_qcint32,
            sys::xnn_datatype_qcint4,
            sys::xnn_datatype_qdint8,
        ] {
            assert!(isQuantizedDataType(q), "{:?} should be quantized", q);
        }
        for &nq in &[
            sys::xnn_datatype_invalid,
            sys::xnn_datatype_fp32,
            sys::xnn_datatype_fp16,
            sys::xnn_datatype_int32,
            sys::xnn_datatype_bf16,
            sys::xnn_datatype_pfp32,
            // These two are quantized-family names but the C++ switch omits them.
            sys::xnn_datatype_qbint4,
            sys::xnn_datatype_qpint8,
        ] {
            assert!(!isQuantizedDataType(nq), "{:?} should not be quantized", nq);
        }
    }

    // flatbufferDimsToVector widens each uint32 flatbuffer dim to usize,
    // preserving order and length, including the empty case.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.flatbuffer-dims-to-vector-fn/test]
    #[test]
    fn flatbuffer_dims_to_vector_widens_and_preserves_order() {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let dims = [1u32, 3, 224, 224];
        let dims_off = fbb.create_vector(&dims);
        let tv = fb_xnnpack::XNNTensorValue::create(
            &mut fbb,
            &fb_xnnpack::XNNTensorValueArgs {
                dims: Some(dims_off),
                ..Default::default()
            },
        );
        fbb.finish_minimal(tv);
        let buf = fbb.finished_data();
        let tv = unsafe { ::flatbuffers::root_unchecked::<fb_xnnpack::XNNTensorValue>(buf) };
        let out = flatbufferDimsToVector(tv.dims().unwrap());
        assert_eq!(out, alloc::vec![1usize, 3, 224, 224]);

        // Empty dims vector -> empty result.
        let mut fbb2 = ::flatbuffers::FlatBufferBuilder::new();
        let empty: [u32; 0] = [];
        let dims_off2 = fbb2.create_vector(&empty);
        let tv2 = fb_xnnpack::XNNTensorValue::create(
            &mut fbb2,
            &fb_xnnpack::XNNTensorValueArgs {
                dims: Some(dims_off2),
                ..Default::default()
            },
        );
        fbb2.finish_minimal(tv2);
        let buf2 = fbb2.finished_data();
        let tv2 = unsafe { ::flatbuffers::root_unchecked::<fb_xnnpack::XNNTensorValue>(buf2) };
        assert!(flatbufferDimsToVector(tv2.dims().unwrap()).is_empty());
    }

    // getOutputMinMax returns the node's stored (min, max) when the
    // output_min_max table is present, and (-inf, +inf) when it is absent.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-output-min-max-fn/test]
    #[test]
    fn get_output_min_max_present_and_absent() {
        // Present: node carries an explicit clamp range.
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let omm = fb_xnnpack::OutputMinMax::create(
            &mut fbb,
            &fb_xnnpack::OutputMinMaxArgs {
                output_min: 0.0,
                output_max: 6.0,
            },
        );
        let node = fb_xnnpack::XNode::create(
            &mut fbb,
            &fb_xnnpack::XNodeArgs {
                output_min_max: Some(omm),
                ..Default::default()
            },
        );
        fbb.finish_minimal(node);
        let buf = fbb.finished_data();
        let node = unsafe { ::flatbuffers::root_unchecked::<fb_xnnpack::XNode>(buf) };
        assert_eq!(getOutputMinMax(node), (0.0f32, 6.0f32));

        // Absent: no output_min_max -> unbounded range.
        let mut fbb2 = ::flatbuffers::FlatBufferBuilder::new();
        let node2 = fb_xnnpack::XNode::create(&mut fbb2, &fb_xnnpack::XNodeArgs::default());
        fbb2.finish_minimal(node2);
        let buf2 = fbb2.finished_data();
        let node2 = unsafe { ::flatbuffers::root_unchecked::<fb_xnnpack::XNode>(buf2) };
        let (mn, mx) = getOutputMinMax(node2);
        assert_eq!(mn, f32::NEG_INFINITY);
        assert_eq!(mx, f32::INFINITY);
    }

    // convertF32TensorToBF16 truncates the rounding-adjusted f32 bit pattern to
    // its high 16 bits, per element. For clean magnitudes the adjustment does
    // not perturb the retained bits, so the output equals the bf16 truncation
    // of the input.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.convert-f32-tensor-to-bf16-fn/test]
    #[test]
    fn convert_f32_to_bf16_truncates_high_bits() {
        let input: [f32; 5] = [0.0, 1.0, -1.0, 2.0, 0.5];
        let expected: [u16; 5] = [0x0000, 0x3f80, 0xbf80, 0x4000, 0x3f00];
        let mut out = [0u16; 5];
        unsafe {
            convertF32TensorToBF16(input.as_ptr(), out.as_mut_ptr(), input.len());
        }
        assert_eq!(out, expected);

        // numel == 0 writes nothing.
        let sentinel = [0xAAAAu16; 2];
        let mut out0 = sentinel;
        unsafe {
            convertF32TensorToBF16(input.as_ptr(), out0.as_mut_ptr(), 0);
        }
        assert_eq!(out0, sentinel);
    }

    // CompileAllocator::allocateTemporary hands out a distinct, writable buffer
    // of the requested size on each call and retains ownership of every buffer
    // (the C++ pushes each into `temporaries_`) until the allocator is dropped.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.compile-allocator.allocate-temporary-fn/test]
    #[test]
    fn allocate_temporary_owns_distinct_writable_buffers() {
        let mut allocator = CompileAllocator::new();

        let p0 = allocator.allocateTemporary(8);
        let p1 = allocator.allocateTemporary(16);
        assert!(!p0.is_null());
        assert!(!p1.is_null());
        // Distinct allocations do not alias.
        assert_ne!(p0, p1);

        // Every byte of each buffer is writable and independent.
        unsafe {
            for i in 0..8 {
                *p0.add(i) = (i as u8).wrapping_add(1);
            }
            for i in 0..16 {
                *p1.add(i) = (i as u8).wrapping_add(100);
            }
            for i in 0..8 {
                assert_eq!(*p0.add(i), (i as u8).wrapping_add(1));
            }
            for i in 0..16 {
                assert_eq!(*p1.add(i), (i as u8).wrapping_add(100));
            }
        }

        // The allocator retains each temporary (bug-for-bug with the C++
        // `temporaries_` vector that owns them until compilation ends).
        assert_eq!(allocator.temporaries_.len(), 2);
        assert_eq!(allocator.temporaries_[0].len(), 8);
        assert_eq!(allocator.temporaries_[1].len(), 16);
    }

    // getDefineNodeFunc dispatches each XNodeUnion tag to the matching definer,
    // exactly as the C++ switch does, and falls through to
    // defineNotImplementedNode for NONE / any unmapped tag.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-define-node-func-fn/test]
    #[test]
    fn get_define_node_func_dispatches_like_cpp_switch() {
        use fb_xnnpack::XNodeUnion as U;
        // fn-pointer identity check: cast both to a raw address.
        fn addr(f: DefineNodeFunc) -> usize {
            f as usize
        }

        // Binary ops.
        assert_eq!(addr(getDefineNodeFunc(U::XNNAdd)), addr(defineAddNode));
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNSubtract)),
            addr(defineSubtractNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNMultiply)),
            addr(defineMultiplyNode)
        );
        assert_eq!(addr(getDefineNodeFunc(U::XNNDiv)), addr(defineDivNode));
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNMinimum)),
            addr(defineMinimumNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNMaximum)),
            addr(defineMaximumNode)
        );

        // A representative spread of the unary / other ops.
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNSoftmax)),
            addr(defineSoftmaxNode)
        );
        assert_eq!(addr(getDefineNodeFunc(U::XNNTanh)), addr(defineTanhNode));
        assert_eq!(addr(getDefineNodeFunc(U::XNNExp)), addr(defineExpNode));
        assert_eq!(addr(getDefineNodeFunc(U::XNNClamp)), addr(defineClampNode));
        assert_eq!(addr(getDefineNodeFunc(U::XNNPReLU)), addr(definePReLUNode));
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNFullyConnected)),
            addr(defineFullyConnectedNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNConv2d)),
            addr(defineConv2dNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNConvert)),
            addr(defineConvertNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNStaticSlice)),
            addr(defineStaticSliceNode)
        );
        assert_eq!(addr(getDefineNodeFunc(U::XNNCopy)), addr(defineCopyNode));
        assert_eq!(
            addr(getDefineNodeFunc(U::XNNBatchMatrixMultiply)),
            addr(defineBatchMatrixMultiplyNode)
        );

        // NONE and any unmapped tag -> catch-all defineNotImplementedNode.
        assert_eq!(
            addr(getDefineNodeFunc(U::NONE)),
            addr(defineNotImplementedNode)
        );
        assert_eq!(
            addr(getDefineNodeFunc(fb_xnnpack::XNodeUnion(255))),
            addr(defineNotImplementedNode)
        );
    }

    // Builds a minimal XNNGraph whose only populated field is a constant_buffer
    // vector: entry 0 is empty (reserved sentinel) and entry 1 carries `bytes`.
    fn graph_with_constant_buffer(bytes: &[u8]) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        // Index 0 is a reserved empty buffer (mirrors the AoT serializer).
        let empty = fbb.create_vector::<u8>(&[]);
        let b0 = fb_xnnpack::Buffer::create(
            &mut fbb,
            &fb_xnnpack::BufferArgs {
                storage: Some(empty),
            },
        );
        let stored = fbb.create_vector::<u8>(bytes);
        let b1 = fb_xnnpack::Buffer::create(
            &mut fbb,
            &fb_xnnpack::BufferArgs {
                storage: Some(stored),
            },
        );
        let cb = fbb.create_vector(&[b0, b1]);
        let graph = fb_xnnpack::XNNGraph::create(
            &mut fbb,
            &fb_xnnpack::XNNGraphArgs {
                constant_buffer: Some(cb),
                ..Default::default()
            },
        );
        fb_xnnpack::finish_xnngraph_buffer(&mut fbb, graph);
        fbb.finished_data().to_vec()
    }

    // Builds a minimal XNNGraph carrying a single constant_data offset entry
    // (offset/size, no named_key) at index 1.
    fn graph_with_constant_data_offset(offset: u64, size: u64) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let cdo0 = fb_xnnpack::ConstantDataOffset::create(
            &mut fbb,
            &fb_xnnpack::ConstantDataOffsetArgs::default(),
        );
        let cdo1 = fb_xnnpack::ConstantDataOffset::create(
            &mut fbb,
            &fb_xnnpack::ConstantDataOffsetArgs {
                offset,
                size,
                named_key: None,
            },
        );
        let cd = fbb.create_vector(&[cdo0, cdo1]);
        let graph = fb_xnnpack::XNNGraph::create(
            &mut fbb,
            &fb_xnnpack::XNNGraphArgs {
                constant_data: Some(cd),
                ..Default::default()
            },
        );
        fb_xnnpack::finish_xnngraph_buffer(&mut fbb, graph);
        fbb.finished_data().to_vec()
    }

    // getConstantDataPtr covers the three pointer-only branches that do not
    // touch the XNNPACK C library: buffer_idx==0 (no constant data -> null), the
    // deprecated in-flatbuffer constant_buffer path (returns a pointer into the
    // buffer's storage bytes), and the constant_data-offset path (returns
    // constant_data_ptr + offset, with bounds checking). The named-key /
    // weights-cache branch requires a real NamedDataMap and is exercised
    // end-to-end via defineTensor + a delegated .pte fixture (gap).
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.get-constant-data-ptr-fn/test]
    #[test]
    fn get_constant_data_ptr_pointer_only_branches() {
        // A typed null `*const dyn NamedDataMap`. None of the branches under
        // test dereference the map, so a null fat pointer is never followed.
        let null_map: *const dyn NamedDataMap =
            core::ptr::null::<NullMap>() as *const dyn NamedDataMap;
        let null_cache: *mut XNNWeightsCache = core::ptr::null_mut();

        // buffer_idx == 0 -> Ok(null), never dereferencing any pointer.
        {
            let buf = graph_with_constant_buffer(&[1, 2, 3, 4]);
            let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
            let mut freeable: Vec<FreeableBuffer> = Vec::new();
            let r = getConstantDataPtr(
                0,
                graph,
                core::ptr::null(),
                0,
                null_map,
                &mut freeable,
                null_cache,
                false,
            );
            assert!(r.is_ok());
            assert!(r.unwrap().is_null());
            assert!(freeable.is_empty());
        }

        // Deprecated in-flatbuffer constant_buffer path: constant_data_ptr is
        // null, so entry `buffer_idx` is read from the flatbuffer storage.
        {
            let payload: [u8; 4] = [10, 20, 30, 40];
            let buf = graph_with_constant_buffer(&payload);
            let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
            let mut freeable: Vec<FreeableBuffer> = Vec::new();
            let r = getConstantDataPtr(
                1,
                graph,
                core::ptr::null(),
                0,
                null_map,
                &mut freeable,
                null_cache,
                false,
            );
            assert!(r.is_ok());
            let p = r.unwrap();
            assert!(!p.is_null());
            let got = unsafe { core::slice::from_raw_parts(p, payload.len()) };
            assert_eq!(got, &payload);

            // buffer_idx out of bounds -> InvalidProgram.
            let mut freeable2: Vec<FreeableBuffer> = Vec::new();
            let r_oob = getConstantDataPtr(
                99,
                graph,
                core::ptr::null(),
                0,
                null_map,
                &mut freeable2,
                null_cache,
                false,
            );
            assert_eq!(r_oob.unwrap_err(), Error::InvalidProgram);
        }

        // constant_data-offset path: constant_data_ptr non-null and no named
        // key, so it returns constant_data_ptr + offset.
        {
            let region: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let buf = graph_with_constant_data_offset(/*offset=*/ 2, /*size=*/ 3);
            let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
            let mut freeable: Vec<FreeableBuffer> = Vec::new();
            let r = getConstantDataPtr(
                1,
                graph,
                region.as_ptr(),
                region.len() as u64,
                null_map,
                &mut freeable,
                null_cache,
                false,
            );
            assert!(r.is_ok());
            assert_eq!(r.unwrap(), unsafe { region.as_ptr().add(2) });

            // Offset+size out of bounds for the region -> InvalidProgram.
            let buf_bad = graph_with_constant_data_offset(/*offset=*/ 6, /*size=*/ 4);
            let graph_bad = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf_bad) };
            let mut freeable_bad: Vec<FreeableBuffer> = Vec::new();
            let r_bad = getConstantDataPtr(
                1,
                graph_bad,
                region.as_ptr(),
                region.len() as u64,
                null_map,
                &mut freeable_bad,
                null_cache,
                false,
            );
            assert_eq!(r_bad.unwrap_err(), Error::InvalidProgram);
        }
    }

    // ---------------------------------------------------------------------
    // Link-dependent tests: these drive the real XNNPACK C library (linked by
    // build.rs whenever `--features xnnpack` is on). Serialized graphs are
    // built with the generated flatbuffer builders — the same wire format the
    // AoT serializer emits — then defined into live `xnn_subgraph` handles.
    // ---------------------------------------------------------------------

    use fb_xnnpack::XNNDatatype as Dt;
    use fb_xnnpack::XNodeUnion as Nu;

    // Never-instantiated map used only to spell a null `*const dyn
    // NamedDataMap`; the paths under test never dereference it (no named keys
    // are serialized in any test graph).
    struct NullMap;
    impl NamedDataMap for NullMap {
        fn get_tensor_layout(
            &self,
            _key: &str,
        ) -> crate::runtime::core::result::Result<crate::runtime::core::tensor_layout::TensorLayout>
        {
            unreachable!()
        }
        fn get_data(
            &self,
            _key: &str,
        ) -> crate::runtime::core::result::Result<
            crate::runtime::core::freeable_buffer::FreeableBuffer,
        > {
            unreachable!()
        }
        fn load_data_into(
            &self,
            _key: &str,
            _buffer: *mut core::ffi::c_void,
            _size: usize,
        ) -> Error {
            unreachable!()
        }
        fn get_num_keys(&self) -> crate::runtime::core::result::Result<u32> {
            unreachable!()
        }
        fn get_key(
            &self,
            _index: u32,
        ) -> crate::runtime::core::result::Result<*const core::ffi::c_char> {
            unreachable!()
        }
    }

    fn null_named_data_map() -> *const dyn NamedDataMap {
        core::ptr::null::<NullMap>() as *const dyn NamedDataMap
    }

    const EXT_IN: u32 = sys::XNN_VALUE_FLAG_EXTERNAL_INPUT;
    const EXT_OUT: u32 = sys::XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
    // external_id for internal / static tensors (XNN_INVALID_VALUE_ID).
    const NO_EXT: u32 = sys::XNN_INVALID_VALUE_ID;

    fn xnn_test_init() {
        let status = unsafe { sys::xnn_initialize(core::ptr::null()) };
        assert_eq!(status, sys::xnn_status_success);
    }

    // RAII holder for a live xnn_subgraph, mirroring compileModel's guard.
    struct TestSubgraph(sys::xnn_subgraph_t);
    impl TestSubgraph {
        fn new(num_externs: u32) -> Self {
            let mut sg = sys::xnn_subgraph_t(core::ptr::null_mut());
            let status = unsafe { sys::xnn_create_subgraph(num_externs, 0, &mut sg) };
            assert_eq!(status, sys::xnn_status_success);
            assert!(!sg.0.is_null());
            TestSubgraph(sg)
        }
    }
    impl Drop for TestSubgraph {
        fn drop(&mut self) {
            unsafe {
                sys::xnn_delete_subgraph(self.0);
            }
        }
    }

    fn zeros_f32(numel: usize) -> Vec<u8> {
        alloc::vec![0u8; numel * core::mem::size_of::<f32>()]
    }

    // Serializes an fp32-style XNNTensorValue wrapped in an XValue.
    // `constant_buffer_idx` of 0 means non-constant; otherwise it is a 1-based
    // index into the `constants` slice handed to `finish_graph`.
    fn tensor_xvalue<'a>(
        fbb: &mut ::flatbuffers::FlatBufferBuilder<'a>,
        dims: &[u32],
        datatype: Dt,
        id_out: u32,
        external_id: u32,
        flags: u32,
        constant_buffer_idx: u32,
    ) -> ::flatbuffers::WIPOffset<fb_xnnpack::XValue<'a>> {
        let dims_off = fbb.create_vector(dims);
        let tv = fb_xnnpack::XNNTensorValue::create(
            fbb,
            &fb_xnnpack::XNNTensorValueArgs {
                datatype,
                num_dims: dims.len() as u32,
                dims: Some(dims_off),
                constant_buffer_idx,
                external_id,
                flags,
                id_out,
                dq_datatype: Dt::xnn_datatype_invalid,
            },
        );
        fb_xnnpack::XValue::create(
            fbb,
            &fb_xnnpack::XValueArgs {
                xvalue_union_type: fb_xnnpack::XValueUnion::XNNTensorValue,
                xvalue_union: Some(tv.as_union_value()),
            },
        )
    }

    // Finishes a single-node XNNGraph buffer (identifier XN00). `constants`
    // become constant_buffer entries 1.. (entry 0 is the reserved empty
    // buffer, mirroring the AoT serializer).
    fn finish_graph<'a>(
        fbb: &mut ::flatbuffers::FlatBufferBuilder<'a>,
        xvalues: &[::flatbuffers::WIPOffset<fb_xnnpack::XValue<'a>>],
        node_type: Nu,
        node_union: ::flatbuffers::WIPOffset<::flatbuffers::UnionWIPOffset>,
        constants: &[&[u8]],
        num_externs: u32,
    ) -> Vec<u8> {
        let node = fb_xnnpack::XNode::create(
            fbb,
            &fb_xnnpack::XNodeArgs {
                xnode_union_type: node_type,
                xnode_union: Some(node_union),
                debug_handle: 0,
                output_min_max: None,
            },
        );
        let xnodes = fbb.create_vector(&[node]);
        let xvalues = fbb.create_vector(xvalues);
        let constant_buffer = if constants.is_empty() {
            None
        } else {
            let mut bufs = Vec::new();
            let empty = fbb.create_vector::<u8>(&[]);
            bufs.push(fb_xnnpack::Buffer::create(
                fbb,
                &fb_xnnpack::BufferArgs {
                    storage: Some(empty),
                },
            ));
            for c in constants {
                let stored = fbb.create_vector::<u8>(c);
                bufs.push(fb_xnnpack::Buffer::create(
                    fbb,
                    &fb_xnnpack::BufferArgs {
                        storage: Some(stored),
                    },
                ));
            }
            Some(fbb.create_vector(&bufs))
        };
        let graph = fb_xnnpack::XNNGraph::create(
            fbb,
            &fb_xnnpack::XNNGraphArgs {
                version: None,
                xnodes: Some(xnodes),
                xvalues: Some(xvalues),
                num_externs,
                input_ids: None,
                output_ids: None,
                constant_buffer,
                mem_buffer_sizes: None,
                constant_data: None,
            },
        );
        fb_xnnpack::finish_xnngraph_buffer(fbb, graph);
        fbb.finished_data().to_vec()
    }

    // Defines every serialized tensor into the subgraph via defineTensor,
    // mirroring compileModel's tensor loop, and returns the remap table plus
    // the collected external input/output id lists.
    fn define_all_tensors(
        sg: &TestSubgraph,
        graph: GraphPtr,
    ) -> (std::collections::HashMap<u32, u32>, Vec<u32>, Vec<u32>) {
        let mut remapped: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        remapped.insert(sys::XNN_INVALID_VALUE_ID, sys::XNN_INVALID_VALUE_ID);
        let mut input_ids: Vec<u32> = Vec::new();
        let mut output_ids: Vec<u32> = Vec::new();
        let mut allocator = CompileAllocator::new();
        let mut freeable: Vec<FreeableBuffer> = Vec::new();
        for value in graph.xvalues().unwrap().iter() {
            let err = defineTensor(
                sg.0,
                &mut remapped,
                value,
                graph,
                core::ptr::null(),
                0,
                &mut input_ids,
                &mut output_ids,
                &mut allocator,
                null_named_data_map(),
                &mut freeable,
                core::ptr::null_mut(),
                false,
            );
            assert_eq!(err, Error::Ok);
        }
        (remapped, input_ids, output_ids)
    }

    // Common happy-path harness: parse the buffer, define all tensors into a
    // fresh subgraph, then run the node definer and expect Error::Ok.
    fn run_node_test(buf: &[u8], num_externs: u32, definer: DefineNodeFunc) {
        xnn_test_init();
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(buf) };
        let sg = TestSubgraph::new(num_externs);
        let (remapped, _inputs, _outputs) = define_all_tensors(&sg, graph);
        let node = graph.xnodes().unwrap().get(0);
        assert_eq!(definer(sg.0, &remapped, node, graph), Error::Ok);
    }

    // Two-tensor graph joined by a _XNNNode1x1-shaped node (convert, softmax,
    // exp, tanh, copy, global avg pooling, ...).
    fn node1x1_graph(
        in_dims: &[u32],
        in_dt: Dt,
        out_dims: &[u32],
        out_dt: Dt,
        union_type: Nu,
    ) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let t0 = tensor_xvalue(&mut fbb, in_dims, in_dt, 0, 0, EXT_IN, 0);
        let t1 = tensor_xvalue(&mut fbb, out_dims, out_dt, 1, 1, EXT_OUT, 0);
        let node = fb_xnnpack::_XNNNode1x1::create(
            &mut fbb,
            &fb_xnnpack::_XNNNode1x1Args {
                input_id: 0,
                output_id: 1,
                flags: 0,
            },
        );
        finish_graph(
            &mut fbb,
            &[t0, t1],
            union_type,
            node.as_union_value(),
            &[],
            2,
        )
    }

    // defineTensor defines external inputs/outputs and static (constant
    // buffer) tensors into a real subgraph, records serialized-id -> new-id
    // remappings, and appends external ids to the input/output lists exactly
    // as the C++ does. A tensor carrying flag bits outside
    // EXTERNAL_INPUT|EXTERNAL_OUTPUT is rejected with InvalidProgram.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tensor-fn/test]
    #[test]
    fn define_tensor_defines_external_and_static_tensors() {
        xnn_test_init();
        let weight_bytes = zeros_f32(4);
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            // Static fp32 tensor backed by constant_buffer entry 1.
            let t2 = tensor_xvalue(&mut fbb, &[4], Dt::xnn_datatype_fp32, 2, NO_EXT, 0, 1);
            let node = fb_xnnpack::_XNNNode1x1::create(
                &mut fbb,
                &fb_xnnpack::_XNNNode1x1Args {
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2],
                Nu::XNNCopy,
                node.as_union_value(),
                &[&weight_bytes],
                2,
            )
        };
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let sg = TestSubgraph::new(2);
        let (remapped, input_ids, output_ids) = define_all_tensors(&sg, graph);

        // Every serialized id got a (valid) new id.
        for id in 0..3u32 {
            let new_id = remapped.get(&id).copied();
            assert!(new_id.is_some(), "serialized id {} not remapped", id);
            assert_ne!(new_id.unwrap(), sys::XNN_INVALID_VALUE_ID);
        }
        // External ids were appended to the right lists.
        assert_eq!(input_ids, alloc::vec![0u32]);
        assert_eq!(output_ids, alloc::vec![1u32]);

        // Unsupported flag bits -> InvalidProgram before any XNNPACK call.
        let bad_buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 0, 0, 0x4, 0);
            let node =
                fb_xnnpack::_XNNNode1x1::create(&mut fbb, &fb_xnnpack::_XNNNode1x1Args::default());
            finish_graph(&mut fbb, &[t0], Nu::XNNCopy, node.as_union_value(), &[], 2)
        };
        let bad_graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&bad_buf) };
        let value = bad_graph.xvalues().unwrap().get(0);
        let mut remapped2 = std::collections::HashMap::new();
        let mut in2: Vec<u32> = Vec::new();
        let mut out2: Vec<u32> = Vec::new();
        let mut alloc2 = CompileAllocator::new();
        let mut free2: Vec<FreeableBuffer> = Vec::new();
        let err = defineTensor(
            sg.0,
            &mut remapped2,
            value,
            bad_graph,
            core::ptr::null(),
            0,
            &mut in2,
            &mut out2,
            &mut alloc2,
            null_named_data_map(),
            &mut free2,
            core::ptr::null_mut(),
            false,
        );
        assert_eq!(err, Error::InvalidProgram);
    }

    // defineConvertNode defines an xnn convert (fp32 -> fp16) between two
    // remapped tensors; a missing remap entry fails with Internal before any
    // XNNPACK call.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-convert-node-fn/test]
    #[test]
    fn define_convert_node_real_subgraph() {
        let buf = node1x1_graph(
            &[1, 4],
            Dt::xnn_datatype_fp32,
            &[1, 4],
            Dt::xnn_datatype_fp16,
            Nu::XNNConvert,
        );
        run_node_test(&buf, 2, defineConvertNode);

        // Missing remapped id -> Internal (REMAP_ID guard).
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let node = graph.xnodes().unwrap().get(0);
        let empty: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let err = defineConvertNode(
            sys::xnn_subgraph_t(core::ptr::null_mut()),
            &empty,
            node,
            graph,
        );
        assert_eq!(err, Error::Internal);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-softmax-node-fn/test]
    #[test]
    fn define_softmax_node_real_subgraph() {
        let buf = node1x1_graph(
            &[1, 4],
            Dt::xnn_datatype_fp32,
            &[1, 4],
            Dt::xnn_datatype_fp32,
            Nu::XNNSoftmax,
        );
        run_node_test(&buf, 2, defineSoftmaxNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-exp-node-fn/test]
    #[test]
    fn define_exp_node_real_subgraph() {
        let buf = node1x1_graph(
            &[1, 4],
            Dt::xnn_datatype_fp32,
            &[1, 4],
            Dt::xnn_datatype_fp32,
            Nu::XNNExp,
        );
        run_node_test(&buf, 2, defineExpNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-tanh-node-fn/test]
    #[test]
    fn define_tanh_node_real_subgraph() {
        let buf = node1x1_graph(
            &[1, 4],
            Dt::xnn_datatype_fp32,
            &[1, 4],
            Dt::xnn_datatype_fp32,
            Nu::XNNTanh,
        );
        run_node_test(&buf, 2, defineTanhNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-copy-node-fn/test]
    #[test]
    fn define_copy_node_real_subgraph() {
        let buf = node1x1_graph(
            &[2, 2],
            Dt::xnn_datatype_fp32,
            &[2, 2],
            Dt::xnn_datatype_fp32,
            Nu::XNNCopy,
        );
        run_node_test(&buf, 2, defineCopyNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-global-avg-pooling2d-node-fn/test]
    #[test]
    fn define_global_avg_pooling2d_node_real_subgraph() {
        let buf = node1x1_graph(
            &[1, 4, 4, 3],
            Dt::xnn_datatype_fp32,
            &[1, 1, 1, 3],
            Dt::xnn_datatype_fp32,
            Nu::XNNGlobalAvgPooling2d,
        );
        run_node_test(&buf, 2, defineGlobalAvgPooling2dNode);
    }

    // _XNNPooling2D-shaped graph (avg / max pooling): NHWC input [1,4,4,3],
    // 2x2 pooling with 2x2 stride and dilation 1 -> output [1,2,2,3].
    fn pooling2d_graph(union_type: Nu) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let t0 = tensor_xvalue(
            &mut fbb,
            &[1, 4, 4, 3],
            Dt::xnn_datatype_fp32,
            0,
            0,
            EXT_IN,
            0,
        );
        let t1 = tensor_xvalue(
            &mut fbb,
            &[1, 2, 2, 3],
            Dt::xnn_datatype_fp32,
            1,
            1,
            EXT_OUT,
            0,
        );
        let node = fb_xnnpack::_XNNPooling2D::create(
            &mut fbb,
            &fb_xnnpack::_XNNPooling2DArgs {
                padding_top: 0,
                padding_right: 0,
                padding_bottom: 0,
                padding_left: 0,
                pooling_height: 2,
                pooling_width: 2,
                stride_height: 2,
                stride_width: 2,
                dilation_height: 1,
                dilation_width: 1,
                input_id: 0,
                output_id: 1,
                flags: 0,
            },
        );
        finish_graph(
            &mut fbb,
            &[t0, t1],
            union_type,
            node.as_union_value(),
            &[],
            2,
        )
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-avg-pooling2d-node-fn/test]
    #[test]
    fn define_avg_pooling2d_node_real_subgraph() {
        let buf = pooling2d_graph(Nu::XNNAvgPooling2d);
        run_node_test(&buf, 2, defineAvgPooling2dNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-max-pooling2d-node-fn/test]
    #[test]
    fn define_max_pooling2d_node_real_subgraph() {
        let buf = pooling2d_graph(Nu::XNNMaxPooling2d);
        run_node_test(&buf, 2, defineMaxPooling2dNode);
    }

    // _XNNNodeConv-shaped graph for conv / deconv: NHWC input [1,4,4,3],
    // static OHWI filter [4,2,2,3] + bias [4], 2x2 kernel, stride/dilation 1.
    fn conv_graph(union_type: Nu, out_dims: &[u32]) -> Vec<u8> {
        let filter = zeros_f32(4 * 2 * 2 * 3);
        let bias = zeros_f32(4);
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let t0 = tensor_xvalue(
            &mut fbb,
            &[1, 4, 4, 3],
            Dt::xnn_datatype_fp32,
            0,
            0,
            EXT_IN,
            0,
        );
        let t1 = tensor_xvalue(
            &mut fbb,
            &[4, 2, 2, 3],
            Dt::xnn_datatype_fp32,
            1,
            NO_EXT,
            0,
            1,
        );
        let t2 = tensor_xvalue(&mut fbb, &[4], Dt::xnn_datatype_fp32, 2, NO_EXT, 0, 2);
        let t3 = tensor_xvalue(&mut fbb, out_dims, Dt::xnn_datatype_fp32, 3, 1, EXT_OUT, 0);
        let node = fb_xnnpack::_XNNNodeConv::create(
            &mut fbb,
            &fb_xnnpack::_XNNNodeConvArgs {
                padding_top: 0,
                padding_right: 0,
                padding_bottom: 0,
                padding_left: 0,
                kernel_height: 2,
                kernel_width: 2,
                subsampling_height: 1,
                subsampling_width: 1,
                dilation_height: 1,
                dilation_width: 1,
                group_input_channels: 3,
                group_output_channels: 4,
                groups: 1,
                adjustment_height: 0,
                adjustment_width: 0,
                input1_id: 0,
                filter_id: 1,
                bias_id: 2,
                output_id: 3,
                flags: 0,
            },
        );
        finish_graph(
            &mut fbb,
            &[t0, t1, t2, t3],
            union_type,
            node.as_union_value(),
            &[&filter, &bias],
            2,
        )
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv2d-node-fn/test]
    #[test]
    fn define_conv2d_node_real_subgraph() {
        let buf = conv_graph(Nu::XNNConv2d, &[1, 3, 3, 4]);
        run_node_test(&buf, 2, defineConv2dNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-conv-transpose2d-node-fn/test]
    #[test]
    fn define_conv_transpose2d_node_real_subgraph() {
        let buf = conv_graph(Nu::XNNConvTranspose2d, &[1, 5, 5, 4]);
        run_node_test(&buf, 2, defineConvTranspose2dNode);
    }

    // Depthwise conv: groups == input channels (4), group_input_channels ==
    // group_output_channels == 1 (depth multiplier 1), static filter
    // [1,2,2,4] + bias [4].
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-depthwise-conv2d-node-fn/test]
    #[test]
    fn define_depthwise_conv2d_node_real_subgraph() {
        let filter = zeros_f32(2 * 2 * 4);
        let bias = zeros_f32(4);
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(
                &mut fbb,
                &[1, 4, 4, 4],
                Dt::xnn_datatype_fp32,
                0,
                0,
                EXT_IN,
                0,
            );
            let t1 = tensor_xvalue(
                &mut fbb,
                &[1, 2, 2, 4],
                Dt::xnn_datatype_fp32,
                1,
                NO_EXT,
                0,
                1,
            );
            let t2 = tensor_xvalue(&mut fbb, &[4], Dt::xnn_datatype_fp32, 2, NO_EXT, 0, 2);
            let t3 = tensor_xvalue(
                &mut fbb,
                &[1, 3, 3, 4],
                Dt::xnn_datatype_fp32,
                3,
                1,
                EXT_OUT,
                0,
            );
            let node = fb_xnnpack::_XNNNodeConv::create(
                &mut fbb,
                &fb_xnnpack::_XNNNodeConvArgs {
                    padding_top: 0,
                    padding_right: 0,
                    padding_bottom: 0,
                    padding_left: 0,
                    kernel_height: 2,
                    kernel_width: 2,
                    subsampling_height: 1,
                    subsampling_width: 1,
                    dilation_height: 1,
                    dilation_width: 1,
                    group_input_channels: 1,
                    group_output_channels: 1,
                    groups: 4,
                    adjustment_height: 0,
                    adjustment_width: 0,
                    input1_id: 0,
                    filter_id: 1,
                    bias_id: 2,
                    output_id: 3,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2, t3],
                Nu::XNNDepthwiseConv2d,
                node.as_union_value(),
                &[&filter, &bias],
                2,
            )
        };
        run_node_test(&buf, 2, defineDepthwiseConv2dNode);
    }

    // Fully connected: input [1,4], static filter [2,4] + bias [2],
    // output [1,2].
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-fully-connected-node-fn/test]
    #[test]
    fn define_fully_connected_node_real_subgraph() {
        let filter = zeros_f32(2 * 4);
        let bias = zeros_f32(2);
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[2, 4], Dt::xnn_datatype_fp32, 1, NO_EXT, 0, 1);
            let t2 = tensor_xvalue(&mut fbb, &[2], Dt::xnn_datatype_fp32, 2, NO_EXT, 0, 2);
            let t3 = tensor_xvalue(&mut fbb, &[1, 2], Dt::xnn_datatype_fp32, 3, 1, EXT_OUT, 0);
            let node = fb_xnnpack::XNNFullyConnected::create(
                &mut fbb,
                &fb_xnnpack::XNNFullyConnectedArgs {
                    input1_id: 0,
                    filter_id: 1,
                    bias_id: 2,
                    output_id: 3,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2, t3],
                Nu::XNNFullyConnected,
                node.as_union_value(),
                &[&filter, &bias],
                2,
            )
        };
        run_node_test(&buf, 2, defineFullyConnectedNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-transpose-node-fn/test]
    #[test]
    fn define_static_transpose_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[2, 3], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[3, 2], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            let perm = fbb.create_vector(&[1u32, 0]);
            let node = fb_xnnpack::XNNStaticTranspose::create(
                &mut fbb,
                &fb_xnnpack::XNNStaticTransposeArgs {
                    num_dims: 2,
                    perm: Some(perm),
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNStaticTranspose,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineStaticTransposeNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-resize-bilinear2-d-node-fn/test]
    #[test]
    fn define_static_resize_bilinear2d_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(
                &mut fbb,
                &[1, 4, 4, 3],
                Dt::xnn_datatype_fp32,
                0,
                0,
                EXT_IN,
                0,
            );
            let t1 = tensor_xvalue(
                &mut fbb,
                &[1, 8, 8, 3],
                Dt::xnn_datatype_fp32,
                1,
                1,
                EXT_OUT,
                0,
            );
            let node = fb_xnnpack::XNNStaticResizeBilinear2D::create(
                &mut fbb,
                &fb_xnnpack::XNNStaticResizeBilinear2DArgs {
                    new_height: 8,
                    new_width: 8,
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNStaticResizeBilinear2D,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineStaticResizeBilinear2DNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-constant-pad-node-fn/test]
    #[test]
    fn define_static_constant_pad_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[2, 2], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[4, 4], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            let pre = fbb.create_vector(&[1u32, 1]);
            let post = fbb.create_vector(&[1u32, 1]);
            let node = fb_xnnpack::XNNStaticConstantPad::create(
                &mut fbb,
                &fb_xnnpack::XNNStaticConstantPadArgs {
                    pre_paddings: Some(pre),
                    post_paddings: Some(post),
                    padding_value: 0.0,
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNStaticConstantPad,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineStaticConstantPadNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-reshape-node-fn/test]
    #[test]
    fn define_static_reshape_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[2, 3], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[3, 2], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            let new_shape = fbb.create_vector(&[3u32, 2]);
            let node = fb_xnnpack::XNNStaticReshape::create(
                &mut fbb,
                &fb_xnnpack::XNNStaticReshapeArgs {
                    num_dims: 2,
                    new_shape: Some(new_shape),
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNStaticReshape,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineStaticReshapeNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-static-slice-node-fn/test]
    #[test]
    fn define_static_slice_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[4, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[2, 2], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            let offsets = fbb.create_vector(&[1u32, 1]);
            let sizes = fbb.create_vector(&[2u32, 2]);
            let node = fb_xnnpack::XNNStaticSlice::create(
                &mut fbb,
                &fb_xnnpack::XNNStaticSliceArgs {
                    num_dims: 2,
                    offsets: Some(offsets),
                    sizes: Some(sizes),
                    input_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNStaticSlice,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineStaticSliceNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-arg-max-pooling2d-node-fn/test]
    #[test]
    fn define_arg_max_pooling2d_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(
                &mut fbb,
                &[1, 4, 4, 3],
                Dt::xnn_datatype_fp32,
                0,
                0,
                EXT_IN,
                0,
            );
            let t1 = tensor_xvalue(
                &mut fbb,
                &[1, 2, 2, 3],
                Dt::xnn_datatype_fp32,
                1,
                1,
                EXT_OUT,
                0,
            );
            // Index output is an internal int32 tensor.
            let t2 = tensor_xvalue(
                &mut fbb,
                &[1, 2, 2, 3],
                Dt::xnn_datatype_int32,
                2,
                NO_EXT,
                0,
                0,
            );
            let node = fb_xnnpack::XNNArgMaxPooling2d::create(
                &mut fbb,
                &fb_xnnpack::XNNArgMaxPooling2dArgs {
                    padding_top: 0,
                    padding_right: 0,
                    padding_bottom: 0,
                    padding_left: 0,
                    pooling_height: 2,
                    pooling_width: 2,
                    input_id: 0,
                    output_value_id: 1,
                    output_index_id: 2,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2],
                Nu::XNNArgMaxPooling2d,
                node.as_union_value(),
                &[],
                2,
            )
        };
        run_node_test(&buf, 2, defineArgMaxPooling2dNode);
    }

    // PReLU maps onto a binary prelu op: input [1,2,2,3], static slope [3].
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-p-re-lu-node-fn/test]
    #[test]
    fn define_prelu_node_real_subgraph() {
        let slope = zeros_f32(3);
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(
                &mut fbb,
                &[1, 2, 2, 3],
                Dt::xnn_datatype_fp32,
                0,
                0,
                EXT_IN,
                0,
            );
            let t1 = tensor_xvalue(&mut fbb, &[3], Dt::xnn_datatype_fp32, 1, NO_EXT, 0, 1);
            let t2 = tensor_xvalue(
                &mut fbb,
                &[1, 2, 2, 3],
                Dt::xnn_datatype_fp32,
                2,
                1,
                EXT_OUT,
                0,
            );
            let node = fb_xnnpack::_XNNNode2x1::create(
                &mut fbb,
                &fb_xnnpack::_XNNNode2x1Args {
                    input1_id: 0,
                    input2_id: 1,
                    output_id: 2,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2],
                Nu::XNNPReLU,
                node.as_union_value(),
                &[&slope],
                2,
            )
        };
        run_node_test(&buf, 2, definePReLUNode);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-batch-matrix-multiply-node-fn/test]
    #[test]
    fn define_batch_matrix_multiply_node_real_subgraph() {
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[2, 3, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[2, 4, 5], Dt::xnn_datatype_fp32, 1, 1, EXT_IN, 0);
            let t2 = tensor_xvalue(
                &mut fbb,
                &[2, 3, 5],
                Dt::xnn_datatype_fp32,
                2,
                2,
                EXT_OUT,
                0,
            );
            let node = fb_xnnpack::_XNNNode2x1::create(
                &mut fbb,
                &fb_xnnpack::_XNNNode2x1Args {
                    input1_id: 0,
                    input2_id: 1,
                    output_id: 2,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2],
                Nu::XNNBatchMatrixMultiply,
                node.as_union_value(),
                &[],
                3,
            )
        };
        run_node_test(&buf, 3, defineBatchMatrixMultiplyNode);
    }

    // Builds an n-way concatenation graph along axis 0: n fp32 inputs [2,3]
    // (external ids 0..n-1) and one output [2n,3] (external id n).
    fn cat_graph(n: u32, union_type: Nu) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let mut xvalues = Vec::new();
        for i in 0..n {
            xvalues.push(tensor_xvalue(
                &mut fbb,
                &[2, 3],
                Dt::xnn_datatype_fp32,
                i,
                i,
                EXT_IN,
                0,
            ));
        }
        xvalues.push(tensor_xvalue(
            &mut fbb,
            &[2 * n, 3],
            Dt::xnn_datatype_fp32,
            n,
            n,
            EXT_OUT,
            0,
        ));
        let node = fb_xnnpack::_XNNCat::create(
            &mut fbb,
            &fb_xnnpack::_XNNCatArgs {
                axis: 0,
                input1_id: 0,
                input2_id: 1,
                input3_id: if n >= 3 { 2 } else { 0 },
                input4_id: if n >= 4 { 3 } else { 0 },
                input5_id: if n >= 5 { 4 } else { 0 },
                output_id: n,
                flags: 0,
            },
        );
        finish_graph(
            &mut fbb,
            &xvalues,
            union_type,
            node.as_union_value(),
            &[],
            n + 1,
        )
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate2-node-fn/test]
    #[test]
    fn define_concatenate2_node_real_subgraph() {
        let buf = cat_graph(2, Nu::XNNConcatenate2);
        run_node_test(&buf, 3, defineConcatenate2Node);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate3-node-fn/test]
    #[test]
    fn define_concatenate3_node_real_subgraph() {
        let buf = cat_graph(3, Nu::XNNConcatenate3);
        run_node_test(&buf, 4, defineConcatenate3Node);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate4-node-fn/test]
    #[test]
    fn define_concatenate4_node_real_subgraph() {
        let buf = cat_graph(4, Nu::XNNConcatenate4);
        run_node_test(&buf, 5, defineConcatenate4Node);
    }

    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-concatenate5-node-fn/test]
    #[test]
    fn define_concatenate5_node_real_subgraph() {
        let buf = cat_graph(5, Nu::XNNConcatenate5);
        run_node_test(&buf, 6, defineConcatenate5Node);
    }

    // defineNotImplementedNode unconditionally reports NotImplemented for the
    // encountered node type, without touching the subgraph.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-not-implemented-node-fn/test]
    #[test]
    fn define_not_implemented_node_returns_not_implemented() {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
        let node = fb_xnnpack::XNode::create(&mut fbb, &fb_xnnpack::XNodeArgs::default());
        let xnodes = fbb.create_vector(&[node]);
        let graph = fb_xnnpack::XNNGraph::create(
            &mut fbb,
            &fb_xnnpack::XNNGraphArgs {
                xnodes: Some(xnodes),
                ..Default::default()
            },
        );
        fb_xnnpack::finish_xnngraph_buffer(&mut fbb, graph);
        let buf = fbb.finished_data().to_vec();
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let node = graph.xnodes().unwrap().get(0);
        let empty: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let err = defineNotImplementedNode(
            sys::xnn_subgraph_t(core::ptr::null_mut()),
            &empty,
            node,
            graph,
        );
        assert_eq!(err, Error::NotImplemented);
    }

    // defineGenericUnaryNode remaps both ids and defines the requested unary
    // op into a real subgraph; a missing remap entry fails with Internal.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-unary-node-fn/test]
    #[test]
    fn define_generic_unary_node_real_subgraph() {
        xnn_test_init();
        let buf = node1x1_graph(
            &[1, 4],
            Dt::xnn_datatype_fp32,
            &[1, 4],
            Dt::xnn_datatype_fp32,
            Nu::XNNAbs,
        );
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let sg = TestSubgraph::new(2);
        let (remapped, _inputs, _outputs) = define_all_tensors(&sg, graph);
        let err = defineGenericUnaryNode(
            sg.0,
            &remapped,
            /*input_id=*/ 0,
            /*output_id=*/ 1,
            /*flags=*/ 0,
            sys::xnn_unary_abs,
            core::ptr::null(),
            Nu::XNNAbs,
            /*debug_handle=*/ 42,
        );
        assert_eq!(err, Error::Ok);

        // Missing remapped id -> Internal (REMAP_ID guard).
        let empty: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let err = defineGenericUnaryNode(
            sg.0,
            &empty,
            0,
            1,
            0,
            sys::xnn_unary_abs,
            core::ptr::null(),
            Nu::XNNAbs,
            42,
        );
        assert_eq!(err, Error::Internal);
    }

    // defineGenericBinaryNode remaps all three ids and defines the requested
    // binary op into a real subgraph.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.define-generic-binary-node-fn/test]
    #[test]
    fn define_generic_binary_node_real_subgraph() {
        xnn_test_init();
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 1, 1, EXT_IN, 0);
            let t2 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 2, 2, EXT_OUT, 0);
            let node = fb_xnnpack::_XNNNode2x1::create(
                &mut fbb,
                &fb_xnnpack::_XNNNode2x1Args {
                    input1_id: 0,
                    input2_id: 1,
                    output_id: 2,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1, t2],
                Nu::XNNAdd,
                node.as_union_value(),
                &[],
                3,
            )
        };
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let sg = TestSubgraph::new(3);
        let (remapped, _inputs, _outputs) = define_all_tensors(&sg, graph);
        let node = graph.xnodes().unwrap().get(0);
        let graph_node = node.xnode_union_as_xnnadd().unwrap();
        let err = defineGenericBinaryNode(
            sg.0,
            &remapped,
            graph_node,
            sys::xnn_binary_add,
            core::ptr::null(),
            Nu::XNNAdd,
            /*debug_handle=*/ 7,
        );
        assert_eq!(err, Error::Ok);
    }

    // compileModel drives the whole pipeline over a headerless serialized
    // XNNGraph (identifier XN00): tensor definition, node definition, runtime
    // creation against a real workspace, and executor initialization with the
    // external input/output id lists.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn/test]
    #[test]
    fn compile_model_end_to_end_add_graph() {
        use crate::backends::xnnpack::runtime::XNNExecutor::XNNExecutor;
        use crate::backends::xnnpack::runtime::XNNWorkspace::XNNWorkspace;

        xnn_test_init();
        // add(x, x): one external input [1,4], one external output [1,4].
        let buf = {
            let mut fbb = ::flatbuffers::FlatBufferBuilder::new();
            let t0 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 0, 0, EXT_IN, 0);
            let t1 = tensor_xvalue(&mut fbb, &[1, 4], Dt::xnn_datatype_fp32, 1, 1, EXT_OUT, 0);
            let node = fb_xnnpack::_XNNNode2x1::create(
                &mut fbb,
                &fb_xnnpack::_XNNNode2x1Args {
                    input1_id: 0,
                    input2_id: 0,
                    output_id: 1,
                    flags: 0,
                },
            );
            finish_graph(
                &mut fbb,
                &[t0, t1],
                Nu::XNNAdd,
                node.as_union_value(),
                &[],
                2,
            )
        };

        let workspace = XNNWorkspace::create().unwrap();
        let mut executor = XNNExecutor::new(workspace.clone());
        let err = XNNCompiler::compileModel(
            buf.as_ptr() as *const core::ffi::c_void,
            buf.len(),
            &mut executor,
            /*weights_cache=*/ core::ptr::null_mut(),
            workspace.unsafe_get_workspace(),
            null_named_data_map(),
            /*use_weight_cache=*/ false,
        );
        assert_eq!(err, Error::Ok);
        assert_eq!(executor.getNumInputs(), 1);
        assert_eq!(executor.getNumOutputs(), 1);
        assert!(executor.get_packed_data_names().is_empty());
    }

    // A buffer without the XN00/XN01 identifier (and no XNNHeader magic) is
    // rejected with DelegateInvalidCompatibility before touching XNNPACK.
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.xnn-compiler.compile-model-fn/test]
    #[test]
    fn compile_model_rejects_bad_identifier() {
        use crate::backends::xnnpack::runtime::XNNExecutor::XNNExecutor;
        use crate::backends::xnnpack::runtime::XNNWorkspace::XNNWorkspace;

        xnn_test_init();
        let garbage = [0u8; 64];
        let workspace = XNNWorkspace::create().unwrap();
        let mut executor = XNNExecutor::new(workspace.clone());
        let err = XNNCompiler::compileModel(
            garbage.as_ptr() as *const core::ffi::c_void,
            garbage.len(),
            &mut executor,
            core::ptr::null_mut(),
            workspace.unsafe_get_workspace(),
            null_named_data_map(),
            false,
        );
        assert_eq!(err, Error::DelegateInvalidCompatibility);
    }

    // isQP8 detects the qdint8-convert-into-qb4w-linear pattern: the convert
    // output must be a qdint8 quantized tensor consumed as input1 of a fully
    // connected node whose filter is qbint4/qcint4/qcint8. Only compiled with
    // the `xnnpack_kleidi` feature, matching the C++ ENABLE_XNNPACK_KLEIDI
    // gate.
    #[cfg(feature = "xnnpack_kleidi")]
    fn qtensor_xvalue<'a>(
        fbb: &mut ::flatbuffers::FlatBufferBuilder<'a>,
        dims: &[u32],
        datatype: Dt,
        id_out: u32,
    ) -> ::flatbuffers::WIPOffset<fb_xnnpack::XValue<'a>> {
        let dims_off = fbb.create_vector(dims);
        let tv = fb_xnnpack::XNNTensorValue::create(
            fbb,
            &fb_xnnpack::XNNTensorValueArgs {
                datatype,
                num_dims: dims.len() as u32,
                dims: Some(dims_off),
                constant_buffer_idx: 0,
                external_id: NO_EXT,
                flags: 0,
                id_out,
                dq_datatype: Dt::xnn_datatype_invalid,
            },
        );
        let qtv = fb_xnnpack::XNNQuantizedTensorValue::create(
            fbb,
            &fb_xnnpack::XNNQuantizedTensorValueArgs {
                tensor_value: Some(tv),
                quant_params_type: fb_xnnpack::XNNQuantParams::NONE,
                quant_params: None,
            },
        );
        fb_xnnpack::XValue::create(
            fbb,
            &fb_xnnpack::XValueArgs {
                xvalue_union_type: fb_xnnpack::XValueUnion::XNNQuantizedTensorValue,
                xvalue_union: Some(qtv.as_union_value()),
            },
        )
    }

    #[cfg(feature = "xnnpack_kleidi")]
    fn qp8_graph(cvt_out_dtype: Dt, filter_dtype: Dt) -> Vec<u8> {
        let mut fbb = ::flatbuffers::FlatBufferBuilder::new();

        // id 1: the convert output; id 2: the linear filter.
        let v1 = qtensor_xvalue(&mut fbb, &[1, 4], cvt_out_dtype, 1);
        let v2 = qtensor_xvalue(&mut fbb, &[2, 4], filter_dtype, 2);
        let xvalues = fbb.create_vector(&[v1, v2]);

        let cvt = fb_xnnpack::_XNNNode1x1::create(
            &mut fbb,
            &fb_xnnpack::_XNNNode1x1Args {
                input_id: 0,
                output_id: 1,
                flags: 0,
            },
        );
        let cvt_node = fb_xnnpack::XNode::create(
            &mut fbb,
            &fb_xnnpack::XNodeArgs {
                xnode_union_type: Nu::XNNConvert,
                xnode_union: Some(cvt.as_union_value()),
                debug_handle: 0,
                output_min_max: None,
            },
        );
        let fc = fb_xnnpack::XNNFullyConnected::create(
            &mut fbb,
            &fb_xnnpack::XNNFullyConnectedArgs {
                input1_id: 1,
                filter_id: 2,
                bias_id: 3,
                output_id: 4,
                flags: 0,
            },
        );
        let fc_node = fb_xnnpack::XNode::create(
            &mut fbb,
            &fb_xnnpack::XNodeArgs {
                xnode_union_type: Nu::XNNFullyConnected,
                xnode_union: Some(fc.as_union_value()),
                debug_handle: 0,
                output_min_max: None,
            },
        );
        let xnodes = fbb.create_vector(&[cvt_node, fc_node]);
        let graph = fb_xnnpack::XNNGraph::create(
            &mut fbb,
            &fb_xnnpack::XNNGraphArgs {
                xnodes: Some(xnodes),
                xvalues: Some(xvalues),
                ..Default::default()
            },
        );
        fb_xnnpack::finish_xnngraph_buffer(&mut fbb, graph);
        fbb.finished_data().to_vec()
    }

    #[cfg(feature = "xnnpack_kleidi")]
    // [spec:et:sem:xnn-compiler.executorch.backends.xnnpack.delegate.is-qp8-fn/test]
    #[test]
    fn is_qp8_detects_dynamic_convert_into_qb4w_linear() {
        // qdint8 convert output feeding a qbint4-filter linear -> QP8.
        let buf = qp8_graph(Dt::xnn_datatype_qdint8, Dt::xnn_datatype_qbint4);
        let graph = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf) };
        let cvt_node = graph.xnodes().unwrap().get(0);
        assert!(isQP8(graph, cvt_node));

        // Convert output that is not qdint8 -> not QP8.
        let buf2 = qp8_graph(Dt::xnn_datatype_qint8, Dt::xnn_datatype_qbint4);
        let graph2 = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf2) };
        let cvt_node2 = graph2.xnodes().unwrap().get(0);
        assert!(!isQP8(graph2, cvt_node2));

        // Unsupported filter dtype (qcint32) -> not QP8.
        let buf3 = qp8_graph(Dt::xnn_datatype_qdint8, Dt::xnn_datatype_qcint32);
        let graph3 = unsafe { fb_xnnpack::root_as_xnngraph_unchecked(&buf3) };
        let cvt_node3 = graph3.xnodes().unwrap().get(0);
        assert!(!isQP8(graph3, cvt_node3));
    }
}
