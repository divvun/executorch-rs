//! Literal port of runtime/executor/tensor_parser_portable.cpp.
//!
//! Portable/ETensor-mode `parseTensor`. Compiled when the `aten` feature is
//! off (the ATen-mode body lives in `tensor_parser_aten.rs`).
//!
//! PORT-NOTE (cross-module, unresolved): `MemoryManager` and `Program`
//! (`runtime/executor/{memory_manager,program}.rs`) are still stubs; this
//! module calls their expected C++-mirrored API
//! (`method_allocator()`, `planned_memory()`).

// PORT-NOTE: the C++ build compiles this file only in non-ATen mode; the
// Rust port keeps it available under `aten` too, standing in for the
// unbuilt ATen parser (see tensor_parser.rs).

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::exec_aten::util::dim_order_util::dim_order_to_stride;
use crate::runtime::core::exec_aten::util::scalar_type_util::is_valid;
use crate::runtime::core::memory_allocator::MemoryAllocatorExt;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::portable_type::device::{DeviceIndex, DeviceType};
use crate::runtime::core::portable_type::scalar_type::ScalarType;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{
    DimOrderType, SizesType, StridesType, TensorImpl, ssize_t,
};
use crate::runtime::core::result::ResultExt;
use crate::runtime::core::span::Span;
use crate::runtime::core::tensor_shape_dynamism::TensorShapeDynamism;
use crate::runtime::executor::memory_manager::MemoryManager;
use crate::runtime::executor::program::Program;
use crate::runtime::executor::tensor_parser::{NamedData, get_tensor_data_ptr};
use crate::schema::generated::executorch_flatbuffer;

// PORT-NOTE: `kTensorDimensionLimit` lives in
// runtime/core/exec_aten/util/tensor_dimension_limit.h, which has no ported
// module yet. Inlined here with its committed value (16), matching the same
// PORT-NOTE in `dim_order_util.rs`. Unresolved cross-module reference.
const K_TENSOR_DIMENSION_LIMIT: usize = 16;

// [spec:et:def:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn]
// [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn]
#[must_use]
#[allow(deprecated)] // Tensor::set_data is #[deprecated] but mirrors the C++ call.
pub fn parse_tensor<'a>(
    program: &Program,
    memory_manager: &mut MemoryManager,
    s_tensor: &executorch_flatbuffer::Tensor,
    named_data_map: Option<&dyn NamedDataMap>,
    external_constants: Span<NamedData>,
) -> Result<Tensor<'a>> {
    // EXECUTORCH_SCOPE_PROF("TensorParser::parseTensor");
    // PORT-NOTE: RAII profiling scope; gated behind `profiling-enabled` like
    // the C++ `EXECUTORCH_SCOPE_PROF` macro. See tensor_parser_exec_aten.rs.
    #[cfg(feature = "profiling-enabled")]
    let _prof = crate::runtime::platform::profiler::ExecutorchProfiler::new(
        c"TensorParser::parseTensor".as_ptr(),
    );
    // PORT-NOTE: `MemoryManager::method_allocator()` returns a raw
    // `*mut MemoryAllocator`; each `method_allocator` use below derefs it to
    // `&mut`, mirroring `memory_manager->method_allocator()->...`. planned_memory()
    // is fetched at the getTensorDataPtr call site.

    crate::et_check_or_return_error!(
        s_tensor.storage_offset() == 0,
        NotSupported,
        "Non-zero storage offset {} not supported",
        s_tensor.storage_offset()
    );

    let scalar_type: ScalarType = static_cast_scalar_type(s_tensor.scalar_type());
    crate::et_check_or_return_error!(
        is_valid(scalar_type),
        InvalidProgram,
        "Invalid or unsupported ScalarType {}",
        scalar_type as i8
    );

    let dynamism: TensorShapeDynamism = static_cast_dynamism(s_tensor.shape_dynamism());
    // TODO(T175194371): Remove this check once fully dynamic shapes are
    // supported.
    crate::et_check_or_return_error!(
        dynamism != TensorShapeDynamism::DYNAMIC_UNBOUND,
        NotSupported,
        "Fully dynamic tensor shapes not yet supported: T175194371"
    );

    crate::et_check_or_return_error!(
        s_tensor.sizes().is_some(),
        InvalidProgram,
        "Missing sizes field"
    );
    let serialized_sizes: *const SizesType =
        s_tensor.sizes().unwrap().bytes().as_ptr() as *const SizesType;
    let dim: u32 = s_tensor.sizes().unwrap().len() as u32;

    crate::et_check_or_return_error!(
        (dim as usize) <= K_TENSOR_DIMENSION_LIMIT,
        InvalidProgram,
        "Tensor rank too large {} > {}",
        dim,
        K_TENSOR_DIMENSION_LIMIT
    );

    crate::et_check_or_return_error!(
        s_tensor.dim_order().is_some(),
        InvalidProgram,
        "Missing dim_order field"
    );
    crate::et_check_or_return_error!(
        s_tensor.dim_order().unwrap().len() as u32 == dim,
        InvalidProgram,
        "dim_order size {} != dim {}",
        s_tensor.dim_order().unwrap().len(),
        dim
    );
    let serialized_dim_order: *const DimOrderType =
        s_tensor.dim_order().unwrap().bytes().as_ptr() as *const DimOrderType;

    let sizes: *mut SizesType;
    let dim_order: *mut DimOrderType;
    // For dynamic shape tensors, allocate local buffers to allow mutable sizes
    // and strides
    if dynamism != TensorShapeDynamism::STATIC {
        // copy sizes and dim order out of flatbuffer
        // kimishpate: I think dim order can remain immutable and point to fb
        // memory, unless we plan to implement in-place permute
        let sizes_buf: *mut SizesType = unsafe { &mut *memory_manager.method_allocator() }
            .allocate_list::<SizesType>(dim as usize, core::mem::align_of::<SizesType>());
        if sizes_buf.is_null() {
            return Err(Error::MemoryAllocationFailed);
        }

        let dim_order_buf: *mut DimOrderType =
            unsafe { &mut *memory_manager.method_allocator() }
                .allocate_list::<DimOrderType>(dim as usize, core::mem::align_of::<DimOrderType>());
        if dim_order_buf.is_null() {
            return Err(Error::MemoryAllocationFailed);
        }

        unsafe {
            core::ptr::copy_nonoverlapping(serialized_sizes, sizes_buf, dim as usize);
            core::ptr::copy_nonoverlapping(serialized_dim_order, dim_order_buf, dim as usize);
        }

        sizes = sizes_buf;
        dim_order = dim_order_buf;
    } else {
        // Const cast safe here as these tensors can't be resized, so these fields
        // will not be modified.
        sizes = serialized_sizes as *mut SizesType;
        dim_order = serialized_dim_order as *mut DimOrderType;
    }
    // Validate sizes before using them in case the PTE data is bad. We can't
    // detect bad positive values, but we can reject negative values, which would
    // otherwise panic in the TensorImpl ctor. dim_order_to_stride() will validate
    // dim_order.
    let mut i: u32 = 0;
    while i < dim {
        crate::et_check_or_return_error!(
            unsafe { *sizes.add(i as usize) } >= 0,
            InvalidProgram,
            "Negative size[{}] {}",
            i as usize,
            unsafe { *sizes.add(i as usize) }
        );
        i += 1;
    }

    // We will remove strides from schema.
    // Allocating strides buffer here and populating it.
    // In subsequent diffs we can remove strides accessor, however this
    // will introduce incompatible APIs between ATen Tensor and ETensor.
    let strides: *mut StridesType = unsafe { &mut *memory_manager.method_allocator() }
        .allocate_list::<StridesType>(dim as usize, core::mem::align_of::<StridesType>());
    if strides.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }

    let status = unsafe { dim_order_to_stride(sizes, dim_order, dim as usize, strides) };
    crate::et_check_or_return_error!(
        status == Error::Ok,
        Internal,
        "dim_order_to_stride returned invalid status"
    );

    // Extract device info from serialized tensor metadata.
    // Defaults to CPU/0 for backward compatibility when extra_tensor_info is
    // absent (e.g., older PTE files without device annotations).
    let mut device_type: DeviceType = DeviceType::CPU;
    let mut device_index: DeviceIndex = 0;
    if s_tensor.extra_tensor_info().is_some() {
        device_type = static_cast_device_type(s_tensor.extra_tensor_info().unwrap().device_type());
        device_index = s_tensor.extra_tensor_info().unwrap().device_index() as DeviceIndex;
    }

    let tensor_impl: *mut TensorImpl = unsafe { &mut *memory_manager.method_allocator() }
        .allocate_instance::<TensorImpl>(core::mem::align_of::<TensorImpl>());
    if tensor_impl.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }

    // Placement new on the allocated memory space. Note that we create this first
    // with null data so we can find its expected size before getting its memory.
    unsafe {
        core::ptr::write(
            tensor_impl,
            TensorImpl::new(
                scalar_type,
                dim as ssize_t,
                sizes,
                /*data=*/ core::ptr::null_mut(),
                dim_order,
                strides,
                dynamism,
                device_type,
                device_index,
            ),
        );
    }

    // Now that we know how big the tensor is, find and assign its memory.
    let data_ptr: Result<*mut core::ffi::c_void> = get_tensor_data_ptr(
        s_tensor,
        program,
        unsafe { (*tensor_impl).nbytes() },
        Some(unsafe { &mut *memory_manager.planned_memory() }),
        named_data_map,
        external_constants,
    );
    if !ResultExt::ok(&data_ptr) {
        crate::et_log!(
            Error,
            "getTensorDataPtr() failed: 0x{:x}",
            data_ptr.error() as u32
        );
        return Err(data_ptr.error());
    }
    let tensor = Tensor::new(tensor_impl);
    tensor.set_data(r_into_ok(data_ptr));

    Ok(tensor)
}

// PORT-NOTE: `static_cast<ScalarType>(s_tensor->scalar_type())` — see the same
// note in tensor_parser_exec_aten.rs.
fn static_cast_scalar_type(st: executorch_flatbuffer::ScalarType) -> ScalarType {
    unsafe { core::mem::transmute::<i8, ScalarType>(st.0) }
}

// PORT-NOTE: `static_cast<TensorShapeDynamism>(s_tensor->shape_dynamism())` —
// the serialized `executorch_flatbuffer::TensorShapeDynamism` (a
// `#[repr(transparent)]` `i8` newtype) and the runtime `TensorShapeDynamism`
// (a `#[repr(u8)]` enum) share discriminants; transmute the byte to reproduce
// the static_cast.
fn static_cast_dynamism(d: executorch_flatbuffer::TensorShapeDynamism) -> TensorShapeDynamism {
    unsafe { core::mem::transmute::<u8, TensorShapeDynamism>(d.0 as u8) }
}

// PORT-NOTE: `static_cast<DeviceType>(extra_tensor_info()->device_type())` —
// the serialized `executorch_flatbuffer::DeviceType` (`i8` newtype) and the
// runtime `DeviceType` (`#[repr(i8)]`) share discriminants.
fn static_cast_device_type(d: executorch_flatbuffer::DeviceType) -> DeviceType {
    unsafe { core::mem::transmute::<i8, DeviceType>(d.0) }
}

// PORT-NOTE: local copy of evalue.rs's private `r_into_ok`.
fn r_into_ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(_) => unreachable!(),
    }
}
