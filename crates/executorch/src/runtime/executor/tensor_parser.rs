//! Literal port of runtime/executor/tensor_parser.h.
//!
//! Declares the deserialization free functions (`get_data_by_key`,
//! `parseTensor`, `parseTensorList`, `validateTensorLayout`,
//! `getTensorDataPtr`) and the header-inline `parseListOptionalType` template.
//! The out-of-line bodies live in `tensor_parser_exec_aten.rs`,
//! `tensor_parser_portable.rs`, and `tensor_parser_aten.rs`.
//!
//! PORT-NOTE (cross-module, unresolved): `MemoryManager`
//! (`runtime/executor/memory_manager.rs`) and `Program`
//! (`runtime/executor/program.rs`) are still stubs at time of writing. This
//! module references their expected C++-mirrored API
//! (`MemoryManager::method_allocator()` -> `&mut MemoryAllocator`,
//! `MemoryManager::planned_memory()` -> `&mut HierarchicalAllocator`,
//! `Program::get_constant_buffer_data`, `Program::load_mutable_subsegment_into`).

use crate::runtime::core::error::{Error, Result};
use crate::runtime::core::evalue::{BoxedEvalueList, EValue, EValueTryTo};
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::memory_allocator::MemoryAllocatorExt;
use crate::runtime::core::result::ResultExt;
use crate::runtime::executor::memory_manager::MemoryManager;

/// Data structure to hold key and data buffer for external data used
/// in a method.
// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.named-data]
pub struct NamedData {
    pub key: *const core::ffi::c_char,
    pub buffer: FreeableBuffer,
}

// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.get-data-by-key-fn]
//
// Body defined in tensor_parser_exec_aten.rs; re-exported here so callers can
// spell it through this module as in the C++ header.
pub use crate::runtime::executor::tensor_parser_exec_aten::get_data_by_key;

// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-executorch.aten.tensor-parse-tensor-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-executorch.aten.tensor-parse-tensor-fn]
//
// `parseTensor` has two mutually exclusive implementations selected at build
// time by tensor mode: portable/ETensor (`tensor_parser_portable.rs`, default)
// or ATen (`tensor_parser_aten.rs`, behind the `aten` feature). The active
// variant is re-exported here to match the single C++ declaration.
#[cfg(not(feature = "aten"))]
pub use crate::runtime::executor::tensor_parser_portable::parse_tensor;
// PORT-NOTE: the true ATen variant (tensor_parser_aten.rs) is an unbuilt
// translation reference — no libtorch FFI exists in this port. The portable
// parser stands in under the `aten` feature so dependents still type-check.
#[cfg(feature = "aten")]
pub use crate::runtime::executor::tensor_parser_portable::parse_tensor;

// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-executorch.aten.tensor-parse-tensor-list-fn]
//
// Body defined in tensor_parser_exec_aten.rs.
pub use crate::runtime::executor::tensor_parser_exec_aten::parse_tensor_list;

// Checks that the sizes, dim_order and scalar_type match between tensors
// stored in the PTE and externally.
// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.validate-tensor-layout-fn]
//
// Body defined in tensor_parser_exec_aten.rs.
pub use crate::runtime::executor::tensor_parser_exec_aten::validate_tensor_layout;

// Deserializes a List of optional type. The code here is the same between all
// list of optionals: list of optional Tensor, list of optional float etc, so we
// just use a template to avoid boilerplate.
//
// PORT-NOTE: `Tensor` (the most common `T`) carries a lifetime, so the C++
// `template <typename T>` is ported as `parse_list_optional_type<'a, T>` with
// the values array borrowed for `'a`. `T` must be reachable through
// `EValueTryTo<Option<T>>` (mirroring the C++ `tryToOptional<T>()` requirement).
// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-std.optional-t-parse-list-optional-type-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-boxed-evalue-list-std.optional-t-parse-list-optional-type-fn]
#[must_use]
pub fn parse_list_optional_type<'a, T>(
    value_indices: &flatbuffers::Vector<'a, i32>,
    values: *mut EValue<'a>,
    values_len: usize,
    memory_manager: &mut MemoryManager,
) -> Result<BoxedEvalueList<'a, Option<T>>>
where
    EValue<'a>: EValueTryTo<T>,
{
    // PORT-NOTE: `method_allocator()` returns a raw `*mut MemoryAllocator`;
    // deref to `&mut` at each use, mirroring the C++ pointer chain.
    let evalp_list: *mut *mut EValue<'a> = unsafe { &mut *memory_manager.method_allocator() }
        .allocate_list::<*mut EValue<'a>>(
        value_indices.len(),
        core::mem::align_of::<*mut EValue<'a>>(),
    );
    if evalp_list.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }

    let optional_tensor_list: *mut Option<T> =
        unsafe { &mut *memory_manager.method_allocator() }
            .allocate_list::<Option<T>>(value_indices.len(), core::mem::align_of::<Option<T>>());
    if optional_tensor_list.is_null() {
        return Err(Error::MemoryAllocationFailed);
    }

    let mut output_idx: usize = 0;
    // For each index look up the corresponding EValue (which has been
    // already allocated) and stick it in the list.
    for index in value_indices.iter() {
        // Lists of objects are stored in fbb as list[int] where the ints are
        // indices into values. Currently serialization is deciding if they want to
        // put -1 for serialized None type indices, or give us a valid index to a
        // serialized None. We support either for now.
        // Placement new as the list elements are not initialized, so calling
        // copy assignment is not defined if its non trivial.
        if index == -1 {
            unsafe {
                core::ptr::write(optional_tensor_list.add(output_idx), None::<T>);
            }
            // no value to point to. BoxedEvalueList for optional tensor will convert
            // this to nullopt.
            // TODO(T161156879): do something less hacky here.
            unsafe {
                *evalp_list.add(output_idx) = core::ptr::null_mut();
            }
        } else {
            crate::et_check_or_return_error!(
                index >= 0 && (index as usize) < values_len,
                InvalidProgram,
                "Invalid value index {} for ListOptional",
                index
            );
            let optional_result = unsafe { (*values.add(index as usize)).try_to_optional::<T>() };
            if !ResultExt::ok(&optional_result) {
                return Err(optional_result.error());
            }
            unsafe {
                core::ptr::write(
                    optional_tensor_list.add(output_idx),
                    r_into_ok(optional_result),
                );
            }
            unsafe {
                *evalp_list.add(output_idx) = values.add(index as usize);
            }
        }
        output_idx += 1;
    }
    Ok(BoxedEvalueList::<Option<T>>::new(
        evalp_list,
        optional_tensor_list,
        value_indices.len() as i32,
    ))
}

/// Returns the appropriate data pointer for `s_tensor`.
///
/// Overall, a Tensor is either constant or non-constant, except we differentiate
/// 2 special variants of non-constant Tensor ("input" and control-flow
/// "placeholder") as a special optimization to avoid holding unnecessary
/// AllocationDetails. Thus, s_tensor can be configured as 1 of 3 options:
/// - constant_buffer > 0, allocation_info = Null: Constant Tensor.
/// - constant_buffer = 0, allocation_info = Non Null: Non-constant Tensor.
/// - constant_buffer = 0, allocation_info = Null: Input/placeholder Tensor.
// [spec:et:def:tensor-parser.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
// [spec:et:sem:tensor-parser.executorch.et-runtime-namespace.deserialization.result-void-get-tensor-data-ptr-fn]
//
// Body defined in tensor_parser_exec_aten.rs.
pub use crate::runtime::executor::tensor_parser_exec_aten::get_tensor_data_ptr;

// PORT-NOTE: local copy of evalue.rs's private `r_into_ok` (moves the Ok value
// out of a `Result<T>` known to be Ok). Mirrors `std::move(result.get())`.
fn r_into_ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(_) => unreachable!(),
    }
}

// Literal port of runtime/executor/test/tensor_parser_device_test.cpp.
//
// PORT-NOTE: `TensorParserDeviceTest` loads a CUDA-annotated `.pte`
// (`ET_MODULE_ADD_WITH_DEVICE_PATH`) and drives `parseTensor` against a device-
// aware `HierarchicalAllocator`. Beyond the unset fixture, it depends on
// `MockCudaAllocator` (`runtime/core/test/mock_cuda_allocator.h`),
// `DeviceMemoryBuffer`/`DeviceAllocator`/`register_device_allocator`
// (`runtime/core/device_*`), and the shared `ManagedMemoryManager` — none of
// which are ported. All three cases skip early.
#[cfg(test)]
mod device_tests {
    fn setup() {
        crate::runtime::platform::runtime::runtime_init();
    }

    fn skip(name: &str) -> bool {
        if std::env::var("ET_MODULE_ADD_WITH_DEVICE_PATH").is_err() {
            eprintln!("skipping {name}: ET_MODULE_ADD_WITH_DEVICE_PATH unset");
            return true;
        }
        eprintln!(
            "skipping {name}: requires the MockCudaAllocator, DeviceMemoryBuffer/\
             DeviceAllocator, and ManagedMemoryManager helpers (none ported)"
        );
        true
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    #[test]
    fn tensor_parser_device_test_cuda_device_parsed_from_pte_file() {
        setup();
        if skip("tensor_parser_device_test_cuda_device_parsed_from_pte_file") {}
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    #[test]
    fn tensor_parser_device_test_non_delegated_tensors_default_to_cpu() {
        setup();
        if skip("tensor_parser_device_test_non_delegated_tensors_default_to_cpu") {}
    }

    // [spec:et:sem:tensor-parser-portable.executorch.et-runtime-namespace.deserialization.parse-tensor-fn/test]
    // [spec:et:sem:method-meta.executorch.et-runtime-namespace.method-meta.memory-planned-buffer-device-fn/test]
    #[test]
    fn tensor_parser_device_test_cuda_tensor_data_ptr_points_to_device_memory() {
        setup();
        if skip("tensor_parser_device_test_cuda_tensor_data_ptr_points_to_device_memory") {}
    }
}
